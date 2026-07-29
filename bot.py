import os
import json
import uuid
import asyncio
import logging
from typing import Optional, Any

import httpx
from fastapi import FastAPI, Request, Response
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

# =========================
# CONFIG FROM ENV
# =========================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL: Optional[str] = os.environ.get("OPENAI_BASE_URL")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

APP_BASE_URL = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("EXTERNAL_URL", "http://localhost:8000")
LOG_BASE_PATH = "/logs"

log = logging.getLogger("uvicorn")

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

RUN_LOGS: dict[str, list[dict[str, Any]]] = {}

def add_log(run_id: str, entry: dict[str, Any]) -> None:
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
# LLM CALL
# =========================
async def call_data_analyst_llm(user_text: str, conversation_history=None) -> dict:
    system_prompt = (
        "You are a data-analysis assistant. "
        "You will receive a user message that may contain a data-analysis question, "
        "possibly with inline data or links to public datasets (e.g. MOSPI).\n\n"
        "Your task:\n"
        "1. Understand the question (if multiple turns, focus on the last message).\n"
        "2. Reason about what data is needed and how to answer.\n"
        "3. Produce a final answer in the exact JSON shape requested by the user.\n\n"
        "You MUST reply with a SINGLE JSON object and NOTHING ELSE. The JSON must have exactly two keys:\n"
        "- \"answer\": the answer, in the exact shape the user requested.\n"
        "- \"log_url\": a placeholder string \"<LOG_URL>\" which the system will replace with a real public JSONL URL.\n\n"
        "Examples:\n\n"
        "1) If the user says:\n"
        "\"Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY a JSON object like {\\\"state\\\": \\\"<state name>\\\"}\"\n"
        "you must reply with:\n"
        "{\"answer\": {\"state\": \"Assam\"}, \"log_url\": \"<LOG_URL>\"}\n\n"
        "2) If the user says:\n"
        "\"What is the population of Bengaluru? Reply with ONLY a JSON object like {\\\"population\\\": <number>}\"\n"
        "you must reply with:\n"
        "{\"answer\": {\"population\": 8443675}, \"log_url\": \"<LOG_URL>\"}\n\n"
        "Do NOT include any text outside the JSON. Do NOT use markdown or code fences."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_text})

    try:
        def _call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )

        resp = await asyncio.to_thread(_call)
        raw_text = resp.choices[0].message.content.strip()

        try:
            payload = json.loads(raw_text)
        except Exception:
            payload = {
                "answer": {"error": "Failed to parse LLM response as JSON"},
                "log_url": "<LOG_URL>"
            }

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

async def handle_message(update: Update, context) -> None:
    user_text = update.message.text
    chat_id = update.message.chat_id

    run_id = str(uuid.uuid4())
    add_log(run_id, {"step": "received", "chat_id": chat_id, "text": user_text})

    conversation_history = None

    result = await call_data_analyst_llm(user_text, conversation_history)
    add_log(run_id, {"step": "llm_response", "raw": result["llm_raw"]})

    payload = result["payload"]
    log_url = f"{APP_BASE_URL}{LOG_BASE_PATH}/{run_id}"
    payload["log_url"] = log_url

    add_log(run_id, {"step": "final_answer", "payload": payload})

    await update.message.reply_text(
        json.dumps(payload),
        parse_mode=None
    )

# =========================
# WEBHOOK ENDPOINT
# =========================
@app.post("/webhook")
async def telegram_webhook(request: Request) -> Response:
    body = await request.json()
    update = Update.de_json(body, tg_app.bot)
    await tg_app.process_update(update)
    return Response(content="ok", status_code=200)

# =========================
# LOGS ENDPOINT
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
    global tg_app, APP_BASE_URL

    if not APP_BASE_URL:
        APP_BASE_URL = os.environ.get("EXTERNAL_URL", "http://localhost:8000")

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