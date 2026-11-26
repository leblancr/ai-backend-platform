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
- Automated backend tests.

## Technology

- Python 3.14+
- FastAPI and Uvicorn
- PostgreSQL and pgvector
- SQLAlchemy (async), asyncpg, and Alembic
- Ollama
- Pydantic Settings
- Pytest

## Requirements

- Python 3.14 or later
- PostgreSQL with the pgvector extension
- Ollama with the configured embedding model
- uv for dependency management

## Local setup

Clone the repository:

    git clone https://github.com/leblancr/ai-backend-platform.git
    cd ai-backend-platform

Install dependencies:

    uv sync --locked

Configure the required environment variables in a local `.env` file. Do not commit secrets.

Apply database migrations:

    uv run alembic upgrade head

Start the development server:

    uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000

API: http://127.0.0.1:8000
Interactive documentation: http://127.0.0.1:8000/docs

## Database migrations

    uv run alembic upgrade head
    uv run alembic current
    uv run alembic history

## Tests

    uv run pytest

## Production deployment

The production deployment uses Ubuntu, PostgreSQL with pgvector, uv, systemd, and Nginx.

Deployment files:

- `deploy/ai-backend-platform.service`
- `deploy/nginx-ai-backend-platform.conf`

The systemd service runs Uvicorn on `127.0.0.1:8000`. Nginx exposes the API under:

    https://skyebeau.com/ai-backend-platform/

The production `.env` file belongs in the application directory and must not be committed. Configure production database and AI-provider settings in that file.

After deploying application changes:

    uv sync --locked
    uv run alembic upgrade head
    sudo systemctl restart ai-backend-platform

Check service status and logs:

    sudo systemctl status ai-backend-platform
    sudo journalctl -u ai-backend-platform -n 100 --no-pager

## Health checks

- `/health`
- `/health/database`

## Project structure

    app/
      api/
      core/
      db/
      models/
      schemas/
      services/
    deploy/
    migrations/
    scripts/
    tests/
    main.py
    pyproject.toml
    uv.lock
