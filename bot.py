import os
import json
import uuid
import asyncio
from typing import Optional, Any
from openai import OpenAI

from fastapi import FastAPI, Request, Response

from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, filters

# =========================
# CONFIG FROM ENV
# =========================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL: Optional[str] = os.environ.get("OPENAI_BASE_URL")  # e.g. https://your-proxy.com/v1
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

APP_BASE_URL = os.environ["RENDER_EXTERNAL_URL"]  # Render sets this; fallback in local test
LOG_BASE_PATH = "/logs"

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

# In-memory logs: run_id -> list of entries
RUN_LOGS: dict[str, list[dict[str, Any]]] = {}

def add_log(run_id: str, entry: dict[str, Any]) -> None:
    if run_id not in RUN_LOGS:
        RUN_LOGS[run_id] = []
    RUN_LOGS[run_id].append(entry)

def get_log_jsonl(run_id: str) -> str:
    entries = RUN_LOGS.get(run_id, [])
    return "\n".join(json.dumps(e) for e in entries)

from openai import OpenAI

# =========================
# OPENAI CLIENT
# =========================
openai_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    openai_kwargs["base_url"] = OPENAI_BASE_URL

client = OpenAI(**openai_kwargs)


# =========================
# LLM “DATA ANALYST” LOGIC
# =========================
import asyncio
async def call_data_analyst_llm(user_text: str, conversation_history=None) -> dict:
    """
    Returns a dict with:
      - answer: whatever shape the question asks for
      - log_url: will be filled by caller
    Also returns internal reasoning steps for logging.
    """

    # Build a simple conversation context if you want multi-turn support later.
    # For now, we just use the last user message.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data-analysis assistant. "
                "You will receive a user message that may contain a data-analysis question.\n\n"
                "You MUST reply with a SINGLE JSON object and NOTHING ELSE. The JSON must have exactly two keys:\n"
                "- \"answer\": the answer, in the exact shape the user requested.\n"
                "- \"log_url\": a placeholder string \"<LOG_URL>\" which the system will replace with a real public JSONL URL.\n\n"
                "Do NOT include any text outside the JSON. Do NOT use markdown or code fences."
            )
        }
    ]

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
        # Catch everything for now (including 429) so the bot still replies
        fallback_payload = {
            "answer": {"error": f"LLM call failed: {type(e).__name__}"},
            "log_url": "<LOG_URL>"
        }
        return {"llm_raw": f"Error: {e}", "payload": fallback_payload}


# =========================
# TELEGRAM APP
# =========================
tg_app: Application = None

async def handle_message(update: Update, context) -> None:
    user_text = update.message.text
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    # Create a run_id for this interaction
    run_id = str(uuid.uuid4())

    # Log start
    add_log(run_id, {"step": "received", "chat_id": chat_id, "message_id": message_id, "text": user_text})

    # Optionally fetch conversation history from context here (for multi-turn)
    conversation_history = None  # placeholder

    # Call LLM
    result = await call_data_analyst_llm(user_text, conversation_history)
    add_log(run_id, {"step": "llm_response", "raw": result["llm_raw"]})

    payload = result["payload"]

    # Construct real log_url
    # Render external URL example: https://your-app.onrender.com
    log_url = f"{APP_BASE_URL}{LOG_BASE_PATH}/{run_id}"
    payload["log_url"] = log_url

    add_log(run_id, {"step": "final_answer", "payload": payload})

    # Send back ONLY JSON text, no markdown
    await update.message.reply_text(
        json.dumps(payload),
        parse_mode=None
    )

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
# WEBHOOK ENDPOINT
# =========================
@app.post("/webhook")
async def telegram_webhook(request: Request) -> Response:
    """
    Telegram will POST updates to this endpoint.
    """
    body = await request.json()
    update = Update.de_json(body, tg_app.bot)
    await tg_app.process_update(update)
    return Response(content="ok", status_code=200)

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
# STARTUP / SHUTDOWN
# =========================
@app.on_event("startup")
async def startup():
    global tg_app, APP_BASE_URL

    # Ensure RENDER_EXTERNAL_URL is available
    if not APP_BASE_URL:
        APP_BASE_URL = os.environ.get("EXTERNAL_URL", "http://localhost:8000")

    tg_app = create_telegram_app()

    # Initialize the telegram Application
    await tg_app.initialize()

    # Set webhook
    webhook_url = f"{APP_BASE_URL}/webhook"
    await tg_app.bot.set_webhook(webhook_url)

@app.on_event("shutdown")
async def shutdown():
    if tg_app:
        await tg_app.bot.delete_webhook()


@app.post("/debug-echo")
async def debug_echo(request: Request) -> dict:
    body = await request.json()
    return {"received": body}
# =========================
# RUN COMMAND
# =========================
# For Render: `uvicorn bot:app --host 0.0.0.0 --port $PORT`