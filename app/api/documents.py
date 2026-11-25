from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.core.config import settings
from app.services.document_extraction import extract_document_text
from app.services.document_db_operations import (
    delete_document as delete_document_record,
    embed_document as embed_document_chunks,
    get_document_chunks,
    list_documents as fetch_documents,
    reprocess_document as rebuild_document_chunks,
    save_document,
    search_document as search_document_chunks,
)
from app.services.llm import generate_answer
from app.services.retrieval import retrieve
from app.services.agent import run_agent

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@router.get("/{document_id}/ask")
async def ask_document(
    document_id: int,
    q: str = Query(min_length=1),
    db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    matches = await retrieve(db, document_id, q)
    if not matches:
        raise HTTPException(400, "Document has no embeddings. Embed it first.")

    context = "\n\n".join(
        f"[Chunk ID: {chunk.id}]\n{chunk.content}"
        for chunk, _ in matches
    )
    answer = await generate_answer(q, context)

    return {
        "document_id": document_id,
        "question": q,
        "answer": answer,
        "sources": [
            {
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "similarity": round(1 - float(distance), 4),
            }
            for chunk, distance in matches
        ],
    }


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_db)):
    return await fetch_documents(db)


@router.get("/{document_id}/search")
async def search_document(
    document_id: int,
    q: str = Query(min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    matches = await search_document_chunks(db, document_id, q, limit)
    if not matches:
        raise HTTPException(400, "Document has no embeddings. Embed it first.")

    return {
        "document_id": document_id,
        "question": q,
        "matches": [
            {
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "similarity": round(1 - float(distance), 4),
                "content": chunk.content,
            }
            for chunk, distance in matches
        ],
    }


@router.post("/{document_id}/embed")
async def embed_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    chunks = await embed_document_chunks(db, document_id)
    if not chunks:
        raise HTTPException(400, "Document has no chunks to embed")

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

    chunks = await rebuild_document_chunks(db, document)
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
        content = extract_document_text(filename, raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not content.strip():
        raise HTTPException(400, "No extractable text found in document")

    return await save_document(
        db,
        filename,
        file.content_type or "application/octet-stream",
        content,
    )


@router.get("/{document_id}/chunks")
async def get_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    chunks = await get_document_chunks(db, document_id)
    return [
        {
            "id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk in chunks
    ]


@router.get("/assistant")
async def document_assistant(
    q: str = Query(min_length=1),
    db: AsyncSession = Depends(get_db),
):
    return await run_agent(q, db)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")

    await delete_document_record(db, document)
