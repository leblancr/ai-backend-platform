# AI Document Platform

**Goal:** Build a portfolio project demonstrating Python backend engineering, LLM integration, RAG, and AI tool calling
for remote backend/AI jobs.

## Repositories

- **Backend:** `ai-backend-platform` — Python, FastAPI, PostgreSQL, Alembic, Poetry.
- **Frontend:** `ai-backend-platform-react` — Vite, React, TypeScript.
- Both projects can be attached in PyCharm.

## Backend features

- Document upload and text extraction (PDF, TXT, Markdown, DOCX).
- Document chunking and PostgreSQL storage.
- Ollama embeddings using `nomic-embed-text` (768 dimensions).
- pgvector similarity search.
- Hybrid retrieval: vector similarity + PostgreSQL full-text keyword search.
- RAG question answering using `qwen3:4b-instruct`.
- Source attribution with chunk IDs.
- AI tool calling through Ollama.
- RAG evaluation script: `scripts/evaluate_rag.py`.

## Frontend features

- Document upload.
- Document listing and embedding.
- Ask questions about selected documents.
- Display generated answers and source chunks.
- Dark interface.

## Current status

- RAG pipeline implemented and working.
- Hybrid retrieval implemented and examined.
- Tool calling implemented.
- RAG evaluation completed: 4/4 tests passed (100%); average response time 1.19 seconds.
- React production build passed.
- PostgreSQL and pgvector confirmed working.
- Local Ollama integration uses `httpx`.

## GitHub

- Backend repository has been pushed to GitHub previously.
- Latest backend changes: push status needs verification.
- React repository: push status needs verification.

## Development decisions

- Learn plain Python and understand the code before adopting additional frameworks.
- Skip LangGraph for now.
- Prioritize backend and AI engineering over frontend polish.
- Begin applying for jobs now rather than waiting for the project to be perfect.

## Remaining work — priority order

1. Verify Git status and push outstanding changes in both repositories.
2. Add automated backend tests.
3. Finish README with architecture explanation, setup instructions, screenshots, and demo.
4. Deploy the application.
5. Add deployment details and demo to GitHub portfolio.
6. Continue job applications while completing the project.