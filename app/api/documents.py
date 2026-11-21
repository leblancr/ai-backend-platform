from io import BytesIO
from pathlib import Path

from docx import Document as WordDocument
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pypdf import PdfReader
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentDetail, DocumentResponse
from app.models.document_chunk import DocumentChunk
from app.services.chunking import chunk_text
from app.core.config import settings
from app.services.embeddings import generate_embeddings

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@router.post("/{document_id}/embed")
async def embed_document(
        document_id: int,
        db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)

    if document is None:
        raise HTTPException(404, "Document not found")

    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )

    chunks = result.scalars().all()

    if not chunks:
        raise HTTPException(400, "Document has no chunks to embed")

    embeddings = await generate_embeddings(
        [chunk.content for chunk in chunks]
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.embedding_model = settings.embedding_model

    await db.commit()

    return {
        "document_id": document_id,
        "chunks_embedded": len(chunks),
        "embedding_model": settings.embedding_model,
        "dimensions": 768,
        "status": "completed",
    }


@router.post("/{document_id}/reprocess")
async def reprocess_document(
        document_id: int,
        db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)

    if document is None:
        raise HTTPException(404, "Document not found")

    chunks = chunk_text(document.content)

    await db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
    )

    db.add_all(
        [
            DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                content=chunk,
            )
            for index, chunk in enumerate(chunks)
        ]
    )

    await db.commit()

    return {
        "document_id": document_id,
        "chunks_created": len(chunks),
        "status": "reprocessed",
    }


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
):
    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            "Only PDF, TXT, Markdown, and DOCX files are supported",
        )

    raw = await file.read(MAX_FILE_SIZE + 1)

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(413, "File exceeds the 10 MB limit")

    try:
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            reader = PdfReader(BytesIO(raw))

            if reader.is_encrypted:
                raise HTTPException(400, "Encrypted PDFs are not supported")

            content = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        elif suffix == ".docx":
            word_document = WordDocument(BytesIO(raw))
            content = "\n".join(
                paragraph.text
                for paragraph in word_document.paragraphs
            )

        else:
            content = raw.decode("utf-8")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Could not extract document text") from exc

    if not content.strip():
        raise HTTPException(400, "No extractable text found in document")

    document = Document(
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        content=content,
    )

    db.add(document)
    await db.flush()

    chunks = chunk_text(content)

    db.add_all(
        [
            DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
            )
            for index, chunk in enumerate(chunks)
        ]
    )

    await db.commit()
    await db.refresh(document)

    return document


@router.get("/{document_id}/chunks")
async def get_document_chunks(
        document_id: int,
        db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)

    if document is None:
        raise HTTPException(404, "Document not found")

    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )

    chunks = result.scalars().all()

    return [
        {
            "id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk in chunks
    ]