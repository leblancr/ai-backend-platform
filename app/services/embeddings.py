import httpx

from app.core.config import settings

EMBEDDING_DIMENSIONS = 768


async def _generate_embedding(
    text: str,
    client: httpx.AsyncClient,
) -> list[float]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embeddings"

    response = await client.post(
        url,
        json={
            "model": settings.embedding_model,
            "prompt": text,
        },
    )
    response.raise_for_status()

    embedding = response.json().get("embedding")

    if not isinstance(embedding, list):
        raise ValueError("Ollama did not return an embedding")

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(embedding)}"
        )

    return embedding


async def generate_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        return await _generate_embedding(text, client)


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        return [
            await _generate_embedding(text, client)
            for text in texts
        ]
