from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.chunking import chunk_text
from app.services.embeddings import generate_embedding, generate_embeddings


async def list_documents(db: AsyncSession):
    """Return documents with their total and embedded chunk counts."""
    result = await db.execute(
        select(
            Document,
            func.count(DocumentChunk.id).label("total_chunks"),
            func.count(DocumentChunk.embedding).label("embedded_chunks"),
        )
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id)
        .order_by(Document.id)
    )

    return [
        {
            "id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "created_at": document.created_at,
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "embedded": (
                total_chunks > 0 and total_chunks == embedded_chunks
            ),
        }
        for document, total_chunks, embedded_chunks in result.all()
    ]


async def search_document(
    db: AsyncSession, document_id: int, question: str, limit: int
):
    """Find the closest embedded chunks in one document."""
    query_embedding = await generate_embedding(question)
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    result = await db.execute(
        select(DocumentChunk, distance.label("distance"))
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.is_not(None),
            DocumentChunk.embedding_model == settings.embedding_model,
        )
        .order_by(distance)
        .limit(limit)
    )
    return result.all()


async def embed_document(db: AsyncSession, document_id: int):
    """Generate and save an embedding for every chunk in a document."""
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = result.scalars().all()

    if not chunks:
        return []

    embeddings = await generate_embeddings(
        [chunk.content for chunk in chunks]
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.embedding_model = settings.embedding_model

    await db.commit()
    return chunks


async def reprocess_document(db: AsyncSession, document: Document):
    """Replace a document's existing chunks with newly generated chunks."""
    chunks = chunk_text(document.content)

    await db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    )

    db.add_all([
        DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk,
        )
        for index, chunk in enumerate(chunks)
    ])

    await db.commit()
    return chunks


async def save_document(
    db: AsyncSession, filename: str, content_type: str, content: str
):
    """Save the original document text and its initial chunks."""
    document = Document(
        filename=filename,
        content_type=content_type,
        content=content,
    )
    db.add(document)
    await db.flush()

    chunks = chunk_text(content)
    db.add_all([
        DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk,
        )
        for index, chunk in enumerate(chunks)
    ])

    await db.commit()
    await db.refresh(document)
    return document


async def get_document_chunks(db: AsyncSession, document_id: int):
    """Return a document's chunks in their original order."""
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    return result.scalars().all()


async def delete_document(db: AsyncSession, document: Document):
    """Delete a document's chunks and then the document itself."""
    await db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    )
    await db.delete(document)
    await db.commit()
