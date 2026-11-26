from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.documents import get_db
from app.main import app


@pytest.fixture
def client():
    async def override_get_db():
        yield object()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_rejects_unsupported_file_type(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("script.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Only PDF, TXT, Markdown, and DOCX" in response.json()["detail"]


def test_rejects_file_larger_than_10_mb(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("large.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File exceeds the 10 MB limit"


def test_rejects_document_with_no_extractable_text(client, monkeypatch):
    import app.api.documents as documents_api

    monkeypatch.setattr(
        documents_api, "extract_document_text", lambda filename, raw: " \n "
    )

    response = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b" ", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No extractable text found in document"


def test_returns_400_when_extraction_fails(client, monkeypatch):
    import app.api.documents as documents_api

    def fail_extraction(filename, raw):
        raise ValueError("Could not extract document text")

    monkeypatch.setattr(
        documents_api, "extract_document_text", fail_extraction
    )

    response = client.post(
        "/documents/upload",
        files={"file": ("broken.pdf", b"invalid", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not extract document text"


def test_successfully_uploads_text_document(client, monkeypatch):
    import app.api.documents as documents_api

    saved_document = SimpleNamespace(
        id=42,
        filename="notes.txt",
        content_type="text/plain",
        created_at=datetime.now(timezone.utc),
    )
    save_mock = AsyncMock(return_value=saved_document)

    monkeypatch.setattr(documents_api, "save_document", save_mock)
    embed_mock = AsyncMock(return_value=[object()])
    monkeypatch.setattr(documents_api, "embed_document_chunks", embed_mock)

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"Hello from the test.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["id"] == 42
    assert response.json()["filename"] == "notes.txt"
    assert response.json()["content_type"] == "text/plain"

    save_mock.assert_awaited_once()
    args = save_mock.await_args.args
    assert args[1:] == (
        "notes.txt",
        "text/plain",
        "Hello from the test.",
    )
    embed_mock.assert_awaited_once()
    assert embed_mock.await_args.args[1] == 42
