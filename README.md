# AI Backend Platform

A Python backend for document processing, semantic search, and AI-powered question answering.

## Features

- Upload PDF, DOCX, TXT, and Markdown documents.
- Extract text and split it into overlapping chunks.
- Generate text embeddings using Ollama.
- Store documents, chunks, and embeddings in PostgreSQL with pgvector.
- Search document content using keyword and vector similarity.
- Ask questions about uploaded documents.
- Reprocess documents and regenerate embeddings.
- Run automated tests for backend functionality.

## Technology

- Python 3.14+
- FastAPI
- PostgreSQL
- SQLAlchemy (async)
- Alembic
- pgvector
- Ollama
- Pydantic Settings
- Pytest

## Requirements

- Python 3.14 or later
- PostgreSQL with the