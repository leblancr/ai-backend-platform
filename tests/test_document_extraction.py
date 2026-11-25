from io import BytesIO

import pytest
from docx import Document as WordDocument
from pypdf import PdfWriter

from app.services.document_extraction import (
    DocumentExtractionError,
    extract_document_text,
)


def test_extracts_txt_as_utf8():
    raw = "Hello, world!\nSecond line.".encode("utf-8")

    assert extract_document_text("notes.txt", raw) == (
        "Hello, world!\nSecond line."
    )


def test_extracts_markdown():
    raw = "# Heading\n\nSome **Markdown** text.".encode("utf-8")

    assert extract_document_text("notes.md", raw) == (
        "# Heading\n\nSome **Markdown** text."
    )


def test_extension_matching_is_case_insensitive():
    assert extract_document_text("NOTES.TXT", b"Hello") == "Hello"


def test_extracts_docx():
    document = WordDocument()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")

    buffer = BytesIO()
    document.save(buffer)

    result = extract_document_text("notes.docx", buffer.getvalue())

    assert result == "First paragraph.\nSecond paragraph."


def test_extracts_text_from_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    buffer = BytesIO()
    writer.write(buffer)

    # A valid PDF with a blank page contains no extractable text.
    assert extract_document_text("blank.pdf", buffer.getvalue()) == ""


def test_invalid_pdf_raises_extraction_error():
    with pytest.raises(DocumentExtractionError):
        extract_document_text("broken.pdf", b"not a PDF")


def test_invalid_docx_raises_extraction_error():
    with pytest.raises(DocumentExtractionError):
        extract_document_text("broken.docx", b"not a DOCX file")


def test_invalid_utf8_raises_extraction_error():
    with pytest.raises(DocumentExtractionError):
        extract_document_text("notes.txt", b"\xff\xfe")
