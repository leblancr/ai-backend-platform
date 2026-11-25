import pytest

from app.services.chunking import chunk_text


def test_short_text_produces_one_chunk():
    assert chunk_text("Hello world") == ["Hello world"]


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []


def test_long_text_is_split_into_multiple_chunks():
    words = [
        "alpha", "beta", "gamma", "delta",
        "epsilon", "zeta", "eta", "theta",
    ]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=20, overlap=6)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)

    # Every original word must still appear in the resulting chunks.
    combined = " ".join(chunks)
    for word in words:
        assert word in combined


def test_chunks_prefer_paragraph_boundaries():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text(text, chunk_size=30, overlap=5)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_rejects_non_positive_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("Some text", chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 10])
def test_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("Some text", chunk_size=10, overlap=overlap)
