import json
import os
import re

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.groq_retry import post_with_retry
from app.services.tools import TOOL_SCHEMAS, run_tool

MAX_STEPS = 5
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You answer questions about the user's uploaded documents.
ALWAYS call a tool before answering. Never answer from memory and never ask the user for more context: look it up with the tools.
Never guess a document ID: call list_documents first when you need one.
Answer only from tool results and cite chunk IDs like [Chunk ID: 12] only for passages returned by search_document, never when listing documents.
If the tools do not return enough information, say:
"I don't have enough information in the provided documents."
"""


async def _chat(client: httpx.AsyncClient, messages: list) -> dict:
    if PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        response = await post_with_retry(
            client,
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    response = await client.post(
        url,
        json={
            "model": settings.llm_model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
            "think": False,
        },
    )
    response.raise_for_status()
    return response.json().get("message", {})


def _assistant_message(message: dict) -> dict:
    if PROVIDER != "groq":
        return message

    clean = {"role": "assistant", "content": message.get("content")}

    if message.get("tool_calls"):
        clean["tool_calls"] = message["tool_calls"]
    else:
        clean["content"] = message.get("content") or ""

    return clean


def _tool_message(call: dict, name: str, result) -> dict:
    content = json.dumps(result)

    if PROVIDER == "groq":
        return {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": name,
            "content": content,
        }

    return {"role": "tool", "tool_name": name, "content": content}


def clean_answer(answer: str, tool_log: list[dict]) -> str:
    if any(call["tool"] == "search_document" for call in tool_log):
        return answer

    return re.sub(r"\s*\[Chunk ID:\s*\d+\]", "", answer)


async def run_agent(question: str, db: AsyncSession) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_log = []
    nudged = False

    async with httpx.AsyncClient(timeout=180.0) as client:
        for _ in range(MAX_STEPS):
            message = await _chat(client, messages)
            messages.append(_assistant_message(message))
            tool_calls = message.get("tool_calls")

            if not tool_calls and not tool_log and not nudged:
                nudged = True
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call a tool before answering. "
                        "Call list_documents or search_document now.",
                    }
                )
                continue

            if not tool_calls:
                answer = (message.get("content") or "").strip()
                return {
                    "question": question,
                    "provider": PROVIDER,
                    "answer": clean_answer(answer, tool_log),
                    "tool_calls": tool_log,
                }

            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments") or {}

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if not isinstance(arguments, dict):
                    arguments = {}

                result = await run_tool(db, name, arguments)
                tool_log.append(
                    {"tool": name, "arguments": arguments, "result": result}
                )
                messages.append(_tool_message(call, name, result))

    return {
        "question": question,
        "provider": PROVIDER,
        "answer": "I could not finish within the allowed number of steps.",
        "tool_calls": tool_log,
    }
