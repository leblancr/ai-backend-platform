# AI Backend Platform

A Python backend for document processing and AI-powered question answering.

Upload documents, extract their text, split the text into smaller sections, generate vector embeddings, and search for relevant content. Ask questions about individual documents or use an AI agent that can search across uploaded documents.

## Features

- **Document upload:** PDF, DOCX, TXT, and Markdown files.
- **Text extraction:** Extracts readable text from supported document formats.
- **Text chunking:** Splits documents into smaller sections with overlap.
- **Embeddings:** Generates 768-dimensional vector embeddings using Ollama.
- **Vector search:** Uses PostgreSQL and pgvector to find semantically relevant sections.
- **Keyword search:** Combines vector similarity with PostgreSQL full-text search.
- **Question answering:** Uses retrieved document sections as context for LLM responses.
- **AI agent:** Uses LLM tool calling to list documents and search their contents.
- **Automated tests:** Tests document extraction, chunking, and upload validation.

## Technology

- Python 3.14+
- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy (async)
- Alembic
- Ollama
- Groq (optional LLM provider)
- uv
- pytest

## Requirements

- Python 3.14 or newer
- PostgreSQL with the pgvector extension
- Ollama running locally or at a reachable URL
- uv for Python dependency management

For the default Ollama configuration, the following models are used:

- Embeddings: `nomic-embed-text`
- Language model: `qwen3:4b-instruct`

Install the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/leblancr/ai-backend-platform.git
cd ai-backend-platform
uv sync
```

### Environment configuration

Create a `.env` file in the project root:

```dotenv
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/ai_backend
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=qwen3:4b-instruct
```

Replace the database credentials and name with your PostgreSQL settings.

### Optional Groq provider

Groq can be used for language-model responses instead of Ollama. Add these settings to `.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Ollama is still used for embeddings when Groq is selected.

## Database migrations

```bash
uv run alembic upgrade head
```

## Run the application

```bash
uv run uvicorn app.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Application health check |
| GET | `/health/database` | Database connection check |
| POST | `/documents/upload` | Upload a document |
| GET | `/documents/` | List documents |
| GET | `/documents/{document_id}/chunks` | List document chunks |
| POST | `/documents/{document_id}/embed` | Generate embeddings |
| POST | `/documents/{document_id}/reprocess` | Recreate document chunks |
| GET | `/documents/{document_id}/search?q=...` | Search a document |
| GET | `/documents/{document_id}/ask?q=...` | Ask about a document |
| GET | `/documents/assistant?q=...` | Ask the AI assistant |
| DELETE | `/documents/{document_id}` | Delete a document |

## Tests

Run the automated tests:

```bash
uv run pytest -v
```

## Project structure

```text
app/
├── api/
├── core/
├── db/
├── models/
├── schemas/
└── services/
migrations/
tests/
```
