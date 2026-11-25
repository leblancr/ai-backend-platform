import os

import httpx

from app.core.config import settings
from app.services.groq_retry import post_with_retry

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


async def generate_answer(
    question: str,
    context: str,
) -> str:
    prompt = f"""Answer the question using only the supplied document passages.

If the passages do not contain enough information to answer, say:
"I don't have enough information in the provided documents."

Do not invent facts or use outside knowledge.
When making claims, cite the relevant chunk using its [Chunk ID].

DOCUMENT PASSAGES:
{context}

QUESTION:
{question}
"""

    async with httpx.AsyncClient(timeout=180.0) as client:
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
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"].get("content")
        else:
            url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
            response = await client.post(
                url,
                json={
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                },
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The model returned an empty answer")

    return answer.strip()
