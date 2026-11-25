from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document_chunk import DocumentChunk
from app.services.document_db_operations import (
    get_document_chunks,
    save_chunk_embeddings,
)
from app.services.embeddings import generate_embeddings


async def embed_document(db: AsyncSession, document_id: int):
    """Generate embeddings for a document's chunks and persist the vectors."""
    chunks = await get_document_chunks(db, document_id)
    if not chunks:
        return []

    embeddings = await generate_embeddings([chunk.content for chunk in chunks])
    await save_chunk_embeddings(
        db, chunks, embeddings, settings.embedding_model
    )
    return chunks
