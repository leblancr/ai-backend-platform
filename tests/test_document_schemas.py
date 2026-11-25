from datetime import datetime, timezone
from types import SimpleNamespace

from app.schemas.document import DocumentDetail, DocumentResponse


def test_document_response_accepts_orm_style_object():
    created_at = datetime.now(timezone.utc)
    document = SimpleNamespace(
        id=7,
        filename="notes.txt",
        content_type="text/plain",
        created_at=created_at,
    )

    result = DocumentResponse.model_validate(document)

    assert result.id == 7
    assert result.filename == "notes.txt"
    assert result.content_type == "text/plain"
    assert result.created_at == created_at


def test_document_detail_includes_content():
    created_at = datetime.now(timezone.utc)
    document = SimpleNamespace(
        id=8,
        filename="guide.md",
        content_type="text/markdown",
        created_at=created_at,
        content="# Guide",
    )

    result = DocumentDetail.model_validate(document)

    assert result.id == 8
    assert result.filename == "guide.md"
    assert result.content == "# Guide"
    assert result.created_at == created_at
