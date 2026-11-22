from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.embeddings import generate_embedding
from app.services.retrieval import retrieve

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List all uploaded documents with their id, "
                "filename and number of chunks."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search one document for passages relevant to a query. "
                "Returns the best matching chunks with chunk IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "integer",
                        "description": "ID of the document to search",
                    },
                    "query": {
                        "type": "string",
                        "description": "What to look for",
                    },
                },
                "required": ["document_id", "query"],
            },
        },
    },
]


async def list_documents(db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(Document.id, Document.filename, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .group_by(Document.id, Document.filename)
        .order_by(Document.id)
    )
    return [
        {"id": doc_id, "filename": filename, "chunks": chunks}
        for doc_id, filename, chunks in result.all()
    ]


async def search_document(
    db: AsyncSession,
    document_id: int,
    query: str,
    limit: int = 4,
) -> list[dict] | dict:
    document = await db.get(Document, document_id)

    if document is None:
        return {"error": f"Document {document_id} not found"}

    matches = await retrieve(db, document_id, query)

    if not matches:
        return {"error": "Document has no embeddings. Embed it first."}

    return [
        {"chunk_id": chunk.id, "content": chunk.content}
        for chunk, _ in matches
    ]


async def run_tool(db: AsyncSession, name: str, arguments: dict):
    try:
        if name == "list_documents":
            return await list_documents(db)

        if name == "search_document":
            return await search_document(
                db,
                int(arguments["document_id"]),
                str(arguments["query"]),
            )

        return {"error": f"Unknown tool: {name}"}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
