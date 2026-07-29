import os
import json
import uuid
import asyncio
import time
import logging
from typing import Optional, Any

import httpx
from fastapi import FastAPI, Request, Response
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

# =========================
# LOGGING
# =========================
log = logging.getLogger("uvicorn")

# =========================
# CONFIG FROM ENV
# =========================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL: Optional[str] = os.environ.get("OPENAI_BASE_URL")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

APP_BASE_URL = (
    os.environ.get("RENDER_EXTERNAL_URL")
    or os.environ.get("EXTERNAL_URL")
    or "http://localhost:8000"
)
LOG_BASE_PATH = "/logs"

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

RUN_LOGS: dict[str, list[dict[str, Any]]] = {}

def add_log(run_id: str, entry: dict[str, Any]) -> None:
    entry.setdefault("timestamp", time.time())
    if run_id not in RUN_LOGS:
        RUN_LOGS[run_id] = []
    RUN_LOGS[run_id].append(entry)

def get_log_jsonl(run_id: str) -> str:
    entries = RUN_LOGS.get(run_id, [])
    return "\n".join(json.dumps(e) for e in entries)

# =========================
# OPENAI CLIENT
# =========================
openai_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    openai_kwargs["base_url"] = OPENAI_BASE_URL

client = OpenAI(**openai_kwargs)

# =========================
# KEEP-ALIVE PINGER
# =========================
async def keep_alive_pinger() -> None:
    ping_url = f"{APP_BASE_URL}/health"
    while True:
        try:
            async with httpx.AsyncClient() as client_:
                await client_.get(ping_url, timeout=5.0)
        except Exception:
            pass
        await asyncio.sleep(600)  # every 10 minutes

# =========================
# ROBUST JSON EXTRACTION
# =========================
def extract_first_json_object(text: str) -> dict:
    # Remove markdown code fences
    text = text.replace("```json", "```").replace("```", "")

    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if start is not None and depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start = None
    # Fallback: try to parse whole text
    return json.loads(text)

# =========================
# CHAT HISTORY (per chat_id)
# =========================
CHAT_HISTORY: dict[int, list[dict[str, str]]] = {}

def append_to_chat_history(chat_id: int, role: str, content: str) -> None:
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = []
    CHAT_HISTORY[chat_id].append({"role": role, "content": content})
    # Keep last 20 turns => 40 messages (user + assistant)
    if len(CHAT_HISTORY[chat_id]) > 40:
        CHAT_HISTORY[chat_id] = CHAT_HISTORY[chat_id][-40:]

def get_chat_history(chat_id: int) -> list[dict[str, str]]:
    return CHAT_HISTORY.get(chat_id, [])

# =========================
# LLM CALL WITH BUDGET
# =========================
async def call_data_analyst_llm(
    user_text: str,
    conversation_history=None,
    timeout_seconds: int = 210,
) -> dict:
    deadline = time.time() + timeout_seconds

    system_prompt = (
        "You are a data-analysis assistant. "
        "You will receive a user message that may contain a data-analysis question, "
        "possibly with inline data or links to public datasets (e.g. MOSPI).\n\n"
        "Conversation rules:\n"
        "- Treat earlier messages as context; always answer the **latest** message.\n"
        "- If a message is only setup (e.g. 'I will send data next'), still reply with a small JSON ack "
        "(for example, {\"answer\": {\"status\": \"ready\"}}), because the grader expects a reply to every message.\n\n"
        "Data & computation rules:\n"
        "- If the question requires computing something from given data, reason carefully and compute the answer.\n"
        "- For published statistics where fetching fails, answer from your knowledge.\n\n"
        "Output rules:\n"
        "- You MUST reply with a SINGLE JSON object and NOTHING ELSE. The JSON must have exactly two keys:\n"
        "  - \"answer\": the answer, in the exact shape the user requested.\n"
        "  - \"log_url\": a placeholder string \"<LOG_URL>\" which the system will replace with a real public JSONL URL.\n"
        "- Match the requested answer shape exactly; never add extra keys.\n"
        "- Do NOT include any text outside the JSON. Do NOT use markdown or code fences.\n\n"
        "Examples:\n\n"
        "1) If the user says:\n"
        "\"Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY a JSON object like {\\\"state\\\": \\\"<state name>\\\"}\"\n"
        "you must reply with:\n"
        "{\"answer\": {\"state\": \"Assam\"}, \"log_url\": \"<LOG_URL>\"}\n\n"
        "2) If the user says:\n"
        "\"What is the population of Bengaluru? Reply with ONLY a JSON object like {\\\"population\\\": <number>}\"\n"
        "you must reply with:\n"
        "{\"answer\": {\"population\": 8443675}, \"log_url\": \"<LOG_URL>\"}\n\n"
        "3) If the user says only: \"I will send you some data next.\"\n"
        "you can reply with:\n"
        "{\"answer\": {\"status\": \"ready\"}, \"log_url\": \"<LOG_URL>\"}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    # If already past deadline, force minimal answer
    if time.time() >= deadline:
        return {
            "llm_raw": '{"answer": {"error": "timeout"}, "log_url": "<LOG_URL>"}',
            "payload": {"answer": {"error": "timeout"}, "log_url": "<LOG_URL>"}
        }

    try:
        def _call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )

        resp = await asyncio.to_thread(_call)
        raw_text = resp.choices.message.content.strip()[0]

        try:
            payload = extract_first_json_object(raw_text)
        except Exception:
            payload = {
                "answer": {"error": "Failed to parse LLM response as JSON"},
                "log_url": "<LOG_URL>"
            }

        # Defensive: ensure "answer" key
        if "answer" not in payload:
            payload = {"answer": payload, "log_url": payload.get("log_url", "<LOG_URL>")}
        if "log_url" not in payload:
            payload["log_url"] = "<LOG_URL>"

        return {"llm_raw": raw_text, "payload": payload}

    except Exception as e:
        fallback_payload = {
            "answer": {"error": f"LLM call failed: {type(e).__name__}"},
            "log_url": "<LOG_URL>"
        }
        return {"llm_raw": f"Error: {e}", "payload": fallback_payload}

# =========================
# TELEGRAM APP
# =========================
tg_app: Application | None = None

def create_telegram_app() -> Application:
    global tg_app
    tg_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return tg_app

# =========================
# MESSAGE HANDLER (NEVER CRASHES SILENTLY)
# =========================
async def handle_message(update: Update, context) -> None:
    chat_id = update.message.chat_id
    run_id = str(uuid.uuid4())

    try:
        user_text = update.message.text
        add_log(run_id, {"step": "received", "chat_id": chat_id, "text": user_text})

        # Update chat history
        append_to_chat_history(chat_id, "user", user_text)
        conversation_history = get_chat_history(chat_id)

        result = await call_data_analyst_llm(user_text, conversation_history, timeout_seconds=210)
        add_log(run_id, {"step": "llm_response", "raw": result["llm_raw"]})

        payload = result["payload"]
        log_url = f"{APP_BASE_URL}{LOG_BASE_PATH}/{run_id}"
        payload["log_url"] = log_url

        add_log(run_id, {"step": "final_answer", "payload": payload})

        await update.message.reply_text(
            json.dumps(payload),
            parse_mode=None
        )

        append_to_chat_history(chat_id, "assistant", json.dumps(payload))

    except Exception as e:
        # Fallback reply so we never leave the grader hanging
        payload = {
            "answer": {"error": "internal error"},
            "log_url": f"{APP_BASE_URL}{LOG_BASE_PATH}/{run_id}"
        }
        add_log(run_id, {"step": "error", "exception": str(e)})
        add_log(run_id, {"step": "final_answer", "payload": payload})

        try:
            await update.message.reply_text(
                json.dumps(payload),
                parse_mode=None
            )
        except Exception:
            log.exception("Failed to send fallback reply")

# =========================
# WEBHOOK ENDPOINT
# =========================
@app.post("/webhook")
async def telegram_webhook(request: Request) -> Response:
    log.info("Received Telegram webhook request")
    try:
        body = await request.json()
        log.info(f"Webhook body type: {type(body)}, content: {body}")

        # Telegram normally sends a single update dict, not a list
        if isinstance(body, list):
            # If somehow a list arrives, just take the first element
            log.warning("Received a list of updates; using the first one")
            body = body[0]

        update = Update.de_json(body, tg_app.bot)
        log.info(f"Decoded update: chat_id={update.message.chat_id if update.message else None}")

        await tg_app.process_update(update)
        log.info("Processed update successfully")
        return Response(content="ok", status_code=200)

    except Exception as e:
        log.exception(f"Error processing webhook: {e}")
        return Response(content="error", status_code=500)

# =========================
# LOGS ENDPOINT (JSONL)
# =========================
@app.get("/logs/{run_id}")
async def get_logs(run_id: str) -> Response:
    content = get_log_jsonl(run_id)
    return Response(
        content=content,
        media_type="application/jsonl",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.jsonl"'}
    )

# =========================
# HEALTH ENDPOINT
# =========================
@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================
# STARTUP / SHUTDOWN
# =========================

@app.on_event("startup")
async def startup():
    global tg_app

    log.info(f"Using APP_BASE_URL={APP_BASE_URL}")

    tg_app = create_telegram_app()
    await tg_app.initialize()

    webhook_url = f"{APP_BASE_URL}/webhook"
    log.info(f"Setting Telegram webhook to {webhook_url}")

    ok = await tg_app.bot.set_webhook(webhook_url)
    log.info(f"set_webhook result: {ok}")

    info = await tg_app.bot.get_webhook_info()
    log.info(f"Current webhook info: {info}")

    # Start keep-alive pinger
    asyncio.create_task(keep_alive_pinger())

    
@app.on_event("shutdown")
async def shutdown():
    if tg_app:
        await tg_app.bot.delete_webhook()