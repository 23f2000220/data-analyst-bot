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
# async def call_data_analyst_llm(
#     user_text: str,
#     conversation_history=None,
#     timeout_seconds: int = 210,
# ) -> dict:
#     deadline = time.time() + timeout_seconds

#     system_prompt = (
#         "You are a data-analysis assistant. "
#         "You will receive a user message that may contain a data-analysis question, "
#         "possibly with inline data or links to public datasets (e.g. MOSPI).\n\n"
#         "Conversation rules:\n"
#         "- Treat earlier messages as context; always answer the **latest** message.\n"
#         "- If a message is only setup (e.g. 'I will send data next'), still reply with a small JSON ack "
#         "(for example, {\"answer\": {\"status\": \"ready\"}}), because the grader expects a reply to every message.\n\n"
#         "Data & computation rules:\n"
#         "- If the question requires computing something from given data, reason carefully and compute the answer.\n"
#         "- For published statistics where fetching fails, answer from your knowledge.\n\n"
#         "Output rules:\n"
#         "- You MUST reply with a SINGLE JSON object and NOTHING ELSE. The JSON must have exactly two keys:\n"
#         "  - \"answer\": the answer, in the exact shape the user requested.\n"
#         "  - \"log_url\": a placeholder string \"<LOG_URL>\" which the system will replace with a real public JSONL URL.\n"
#         "- Match the requested answer shape exactly; never add extra keys.\n"
#         "- Do NOT include any text outside the JSON. Do NOT use markdown or code fences.\n\n"
#         "Examples:\n\n"
#         "1) If the user says:\n"
#         "\"Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY a JSON object like {\\\"state\\\": \\\"<state name>\\\"}\"\n"
#         "you must reply with:\n"
#         "{\"answer\": {\"state\": \"Assam\"}, \"log_url\": \"<LOG_URL>\"}\n\n"
#         "2) If the user says:\n"
#         "\"What is the population of Bengaluru? Reply with ONLY a JSON object like {\\\"population\\\": <number>}\"\n"
#         "you must reply with:\n"
#         "{\"answer\": {\"population\": 8443675}, \"log_url\": \"<LOG_URL>\"}\n\n"
#         "3) If the user says only: \"I will send you some data next.\"\n"
#         "you can reply with:\n"
#         "{\"answer\": {\"status\": \"ready\"}, \"log_url\": \"<LOG_URL>\"}"
#     )

#     messages = [{"role": "system", "content": system_prompt}]
#     if conversation_history:
#         messages.extend(conversation_history)
#     messages.append({"role": "user", "content": user_text})

#     # If already past deadline, force minimal answer
#     if time.time() >= deadline:
#         return {
#             "llm_raw": '{"answer": {"error": "timeout"}, "log_url": "<LOG_URL>"}',
#             "payload": {"answer": {"error": "timeout"}, "log_url": "<LOG_URL>"}
#         }

#     try:
#         def _call():
#             return client.chat.completions.create(
#                 model=OPENAI_MODEL,
#                 messages=messages,
#                 temperature=0,
#                 response_format={"type": "json_object"},
#             )

#         resp = await asyncio.to_thread(_call)
#         raw_text = resp.choices[0].message.content.strip()

#         try:
#             payload = extract_first_json_object(raw_text)
#         except Exception:
#             payload = {
#                 "answer": {"error": "Failed to parse LLM response as JSON"},
#                 "log_url": "<LOG_URL>"
#             }

#         # Defensive: ensure "answer" key
#         if "answer" not in payload:
#             payload = {"answer": payload, "log_url": payload.get("log_url", "<LOG_URL>")}
#         if "log_url" not in payload:
#             payload["log_url"] = "<LOG_URL>"

#         return {"llm_raw": raw_text, "payload": payload}

#     except Exception as e:
#         fallback_payload = {
#             "answer": {"error": f"LLM call failed: {type(e).__name__}"},
#             "log_url": "<LOG_URL>"
#         }
#         return {"llm_raw": f"Error: {e}", "payload": fallback_payload}

import io
import contextlib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---- Piece 1: tool definition (menu GPT sees) ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code to fetch and/or analyze data. "
                "pd, np, requests, and BeautifulSoup are already imported. "
                "Always print() the specific value you need — only stdout is returned to you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute."
                    }
                },
                "required": ["code"],
            },
        },
    }
]

# ---- Piece 2: executor ----
def run_python_tool(code: str) -> str:
    namespace = {
        "pd": pd,
        "np": np,
        "requests": requests,
        "BeautifulSoup": BeautifulSoup,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, namespace)
        output = buf.getvalue()
        if not output.strip():
            output = "(code ran with no output — remember to print() the value you need)"
    except Exception as e:
        output = f"ERROR: {type(e).__name__}: {e}"

    return output[:8000]



import os

MODEL_CANDIDATES = [
    m.strip() for m in os.environ.get(
        "MODEL_CANDIDATES",
        OPENAI_MODEL  # falls back to your existing single model if unset
    ).split(",")
    if m.strip()
]

MAX_TURNS = 6

SYSTEM_PROMPT = (
    "You are a data-analysis assistant. "
    "You will receive a user message that may contain a data-analysis question, "
    "possibly with inline data or links/references to public datasets or APIs.\n\n"
    "Conversation rules:\n"
    "- Treat earlier messages as context; always answer the **latest** message.\n"
    "- If a message is only setup (e.g. 'I will send data next'), still reply with a small JSON ack "
    "(for example, {\"answer\": {\"status\": \"ready\"}}), because the grader expects a reply to every message.\n\n"
    "Data & computation rules:\n"
    "- For any question referencing a named dataset, indicator, or public API, you must attempt at "
    "least one run_python fetch before answering — even if you're not fully certain of the exact "
    "endpoint or query syntax. Write your best-guess request first.\n"
    "- If a fetch fails or returns unexpected data, use the error/response to try a different URL "
    "pattern, query parameter, or approach — treat this like iterative debugging, not a one-shot attempt.\n"
    "- Only conclude data is genuinely unavailable after multiple distinct attempts have failed.\n"
    "- Never answer 'unavailable', 'unknown', or a placeholder value without tool-call evidence in "
    "this conversation that you actually tried.\n\n"
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
    "2) If the user says only: \"I will send you some data next.\"\n"
    "you can reply with:\n"
    "{\"answer\": {\"status\": \"ready\"}, \"log_url\": \"<LOG_URL>\"}"
)


async def _run_with_model(
    model: str,
    base_messages: list,
    run_id: Optional[str],
    deadline: float,
) -> tuple[dict, int]:
    """
    Runs the tool-calling loop with a single model.
    Returns (result_dict, tool_call_count).
    result_dict has the same {"llm_raw":..., "payload":...} shape as before.
    """
    messages = list(base_messages)  # copy — don't mutate the caller's list
    tool_call_count = 0

    for turn in range(MAX_TURNS):
        if time.time() >= deadline:
            break

        try:
            def _call():
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=0,
                )
            resp = await asyncio.to_thread(_call)
        except Exception as e:
            fallback_payload = {
                "answer": {"error": f"LLM call failed: {type(e).__name__}"},
                "log_url": "<LOG_URL>"
            }
            return {"llm_raw": f"Error: {e}", "payload": fallback_payload}, tool_call_count

        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())

            for tool_call in msg.tool_calls:
                tool_call_count += 1
                try:
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                except Exception:
                    code = ""

                result = run_python_tool(code)

                if run_id:
                    add_log(run_id, {
                        "step": "tool_call",
                        "model": model,
                        "code": code,
                        "output": result,
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue

        else:
            raw_text = (msg.content or "").strip()
            try:
                payload = extract_first_json_object(raw_text)
            except Exception:
                payload = {
                    "answer": {"error": "Failed to parse LLM response as JSON"},
                    "log_url": "<LOG_URL>"
                }

            if "answer" not in payload:
                payload = {"answer": payload, "log_url": payload.get("log_url", "<LOG_URL>")}
            if "log_url" not in payload:
                payload["log_url"] = "<LOG_URL>"

            return {"llm_raw": raw_text, "payload": payload}, tool_call_count

    fallback_payload = {
        "answer": {"error": "ran out of turns/time before reaching a final answer"},
        "log_url": "<LOG_URL>"
    }
    return {"llm_raw": "(no final response — loop exhausted)", "payload": fallback_payload}, tool_call_count


async def call_data_analyst_llm(
    user_text: str,
    conversation_history=None,
    run_id: str = None,
    timeout_seconds: int = 210,
) -> dict:
    deadline = time.time() + timeout_seconds

    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        base_messages.extend(conversation_history)
    base_messages.append({"role": "user", "content": user_text})

    # Heuristic: a "real" data question is more than a short setup/ack message.
    # Short messages (e.g. "I will send data next") are allowed to skip tool use.
    looks_like_real_question = len(user_text.split()) > 8

    last_result = None
    for model in MODEL_CANDIDATES:
        if time.time() >= deadline:
            break

        result, tool_call_count = await _run_with_model(model, base_messages, run_id, deadline)
        last_result = result

        answer_str = json.dumps(result["payload"].get("answer", {})).lower()
        looks_like_giveup = any(
            phrase in answer_str for phrase in ["unavailable", "unknown", "n/a", "error"]
        )

        if run_id:
            add_log(run_id, {
                "step": "model_attempt",
                "model": model,
                "tool_call_count": tool_call_count,
                "gave_up_without_trying": looks_like_real_question and tool_call_count == 0 and looks_like_giveup,
            })

        # Success condition: either it used tools, or the question didn't need them,
        # or it didn't give a give-up-style answer.
        if tool_call_count > 0 or not looks_like_real_question or not looks_like_giveup:
            return result

        # Otherwise: this model skipped tool use on a real question and gave up — try next model

    # All models exhausted (or none configured) — return whatever we last got
    if last_result:
        return last_result

    fallback_payload = {
        "answer": {"error": "no models available"},
        "log_url": "<LOG_URL>"
    }
    return {"llm_raw": "(no models attempted)", "payload": fallback_payload}


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