# AI Document Platform

Goal: Python backend job with LLM/RAG skills. Building a document Q&A app.

Repos:
- Backend: Python, Alembic, poetry. Document ingestion, embeddings, RAG Q&A with source attribution, RAG evaluation script (scripts/evaluate_rag.py)
- Frontend: Vite/React/TypeScript. DocumentUpload, EmbedDocument, AskDocument components

Stage: RAG, evaluation and tool calling (Ollama) built.
Next: tests, README with demo, deploy, push to GitHub.
Then: tests, README with demo, deploy, push to GitHub.

Known: LLM and embeddings run on local Ollama via httpx. pgvector confirmed (768 dims).
Unknown: whether pushed to GitHub.
Decisions: learn plain Python first, skip LangGraph until later. Apply for jobs now.
