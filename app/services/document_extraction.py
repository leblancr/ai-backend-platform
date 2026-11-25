from io import BytesIO
from pathlib import Path

from docx import Document as WordDocument
from pypdf import PdfReader


class DocumentExtractionError(ValueError):
    """Raised when an uploaded document cannot be converted to text."""


def extract_document_text(filename: str, raw: bytes) -> str:
    """Extract readable text from a supported uploaded document."""
    suffix = Path(filename).suffix.lower()

    try:
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(raw))
            if reader.is_encrypted:
                raise DocumentExtractionError("Encrypted PDFs are not supported")

            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        if suffix == ".docx":
            document = WordDocument(BytesIO(raw))
            return "\n".join(
                paragraph.text for paragraph in document.paragraphs
            )

        # TXT and Markdown files are UTF-8 text.
        return raw.decode("utf-8")

    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError("Could not extract document text") from exc
