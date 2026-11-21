import httpx

from app.core.config import settings


async def generate_answer(
    question: str,
    context: str,
) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

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
        response = await client.post(
            url,
            json={
                "model": settings.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "think": False,
            },
        )

        response.raise_for_status()
        data = response.json()

    answer = data.get("message", {}).get("content")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an empty answer")

    return answer.strip()
