import re
import json

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.tools import TOOL_SCHEMAS, run_tool

MAX_STEPS = 5

SYSTEM_PROMPT = """You answer questions about the user's uploaded documents.
Use the tools to look things up. Never guess a document ID: call list_documents first when you need one.
Answer only from tool results and cite chunk IDs like [Chunk ID: 12] only for passages returned by search_document, never when listing documents.
If the tools do not return enough information, say:
"I don't have enough information in the provided documents."
"""


async def run_agent(question: str, db: AsyncSession) -> dict:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_log = []

    async with httpx.AsyncClient(timeout=180.0) as client:
        for _ in range(MAX_STEPS):
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

            message = response.json().get("message", {})
            messages.append(message)

            tool_calls = message.get("tool_calls")

            if not tool_calls:
                return {
                    "question": question,
                    "answer": clean_answer((message.get("content") or "").strip(), tool_log),
                    "tool_calls": tool_log,
                }

            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments") or {}

                if not isinstance(arguments, dict):
                    arguments = {}

                result = await run_tool(db, name, arguments)
                tool_log.append({"tool": name, "arguments": arguments, "result": result})

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(result),
                    }
                )

    return {
        "question": question,
        "answer": "I could not finish within the allowed number of steps.",
        "tool_calls": tool_log,
    }


def clean_answer(answer: str, tool_log: list[dict]) -> str:
    if any(call["tool"] == "search_document" for call in tool_log):
        return answer

    return re.sub(r"\s*\[Chunk ID:\s*\d+\]", "", answer)
