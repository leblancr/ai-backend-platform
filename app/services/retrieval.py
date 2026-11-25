import re

from sqlalchemy import and_, func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document_chunk import DocumentChunk
from app.services.embeddings import generate_embedding


async def retrieve(
    db: AsyncSession,
    document_id: int,
    question: str,
    vector_limit: int = 4,
    keyword_limit: int = 3,
):
    """Find relevant chunks using vector similarity and optional keyword search."""
    embedding = await generate_embedding(question)
    distance = DocumentChunk.embedding.cosine_distance(embedding)

    base = [
        DocumentChunk.document_id == document_id,
        DocumentChunk.embedding.is_not(None),
        DocumentChunk.embedding_model == settings.embedding_model,
    ]

    vector_rows = (
        await db.execute(
            select(DocumentChunk, distance.label("distance"))
            .where(*base)
            .order_by(distance)
            .limit(vector_limit)
        )
    ).all()

    keyword_rows = []
    if keyword_limit > 0:
        english = literal_column("'english'")
        simple = literal_column("'simple'")

        match = func.to_tsvector(english, DocumentChunk.content).op("@@")(
            func.plainto_tsquery(english, question)
        )

        numbers = re.findall(r"\b\d{2,}\b", question)
        if numbers:
            number_match = and_(
                *[
                    func.to_tsvector(simple, DocumentChunk.content).op("@@")(
                        func.plainto_tsquery(simple, number)
                    )
                    for number in numbers
                ]
            )
            match = or_(match, number_match)

        keyword_rows = (
            await db.execute(
                select(DocumentChunk, distance.label("distance"))
                .where(*base, match)
                .order_by(distance)
                .limit(keyword_limit)
            )
        ).all()

    merged = []
    seen = set()
    for chunk, dist in [*keyword_rows, *vector_rows]:
        if chunk.id not in seen:
            seen.add(chunk.id)
            merged.append((chunk, dist))

    return merged
