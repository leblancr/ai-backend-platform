from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.document_db_operations as db_ops
import app.services.document_embedding as document_embedding
import app.services.embeddings as embeddings
import app.services.retrieval as retrieval


@pytest.mark.anyio
async def test_save_chunk_embeddings_assigns_vectors_and_commits():
    db = SimpleNamespace(commit=AsyncMock())
    chunks = [SimpleNamespace(), SimpleNamespace()]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    await db_ops.save_chunk_embeddings(db, chunks, vectors, "test-model")

    assert chunks[0].embedding == vectors[0]
    assert chunks[1].embedding == vectors[1]
    assert all(chunk.embedding_model == "test-model" for chunk in chunks)
    db.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_save_chunk_embeddings_rejects_mismatched_counts():
    db = SimpleNamespace(commit=AsyncMock())

    with pytest.raises(ValueError, match="must match"):
        await db_ops.save_chunk_embeddings(
            db,
            [SimpleNamespace()],
            [],
            "test-model",
        )

    db.commit.assert_not_awaited()


@pytest.mark.anyio
async def test_reprocess_document_replaces_chunks_and_commits():
    db = SimpleNamespace(
        execute=AsyncMock(),
        add_all=MagicMock(),
        commit=AsyncMock(),
    )
    document = SimpleNamespace(id=12, content="alpha " * 500)

    chunks = await db_ops.reprocess_document(db, document)

    db.execute.assert_awaited_once()
    db.add_all.assert_called_once()
    db.commit.assert_awaited_once()

    saved_chunks = db.add_all.call_args.args[0]
    assert len(saved_chunks) == len(chunks)
    assert [chunk.chunk_index for chunk in saved_chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == 12 for chunk in saved_chunks)
    assert [chunk.content for chunk in saved_chunks] == chunks


@pytest.mark.anyio
async def test_list_documents_reports_chunk_counts_and_embedding_status():
    documents = [
        SimpleNamespace(
            id=1,
            filename="complete.txt",
            content_type="text/plain",
            created_at=None,
        ),
        SimpleNamespace(
            id=2,
            filename="partial.txt",
            content_type="text/plain",
            created_at=None,
        ),
        SimpleNamespace(
            id=3,
            filename="empty.txt",
            content_type="text/plain",
            created_at=None,
        ),
    ]
    rows = [
        (documents[0], 2, 2),
        (documents[1], 3, 1),
        (documents[2], 0, 0),
    ]
    result = SimpleNamespace(all=lambda: rows)
    db = SimpleNamespace(execute=AsyncMock(return_value=result))

    listed = await db_ops.list_documents(db)

    assert [item["id"] for item in listed] == [1, 2, 3]
    assert [item["embedded"] for item in listed] == [True, False, False]
    assert [item["total_chunks"] for item in listed] == [2, 3, 0]
    assert [item["embedded_chunks"] for item in listed] == [2, 1, 0]


@pytest.mark.anyio
async def test_get_document_chunks_returns_ordered_results():
    chunks = [SimpleNamespace(chunk_index=0), SimpleNamespace(chunk_index=1)]
    result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: chunks))
    db = SimpleNamespace(execute=AsyncMock(return_value=result))

    returned = await db_ops.get_document_chunks(db, 7)

    assert returned == chunks
    db.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_delete_document_deletes_chunks_and_document():
    document = SimpleNamespace(id=7)
    db = SimpleNamespace(
        execute=AsyncMock(),
        delete=AsyncMock(),
        commit=AsyncMock(),
    )

    await db_ops.delete_document(db, document)

    db.execute.assert_awaited_once()
    db.delete.assert_awaited_once_with(document)
    db.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_embed_document_returns_empty_when_no_chunks(monkeypatch):
    get_chunks = AsyncMock(return_value=[])
    generate = AsyncMock()
    save = AsyncMock()

    monkeypatch.setattr(document_embedding, "get_document_chunks", get_chunks)
    monkeypatch.setattr(document_embedding, "generate_embeddings", generate)
    monkeypatch.setattr(document_embedding, "save_chunk_embeddings", save)

    result = await document_embedding.embed_document(object(), 9)

    assert result == []
    get_chunks.assert_awaited_once()
    generate.assert_not_awaited()
    save.assert_not_awaited()


@pytest.mark.anyio
async def test_embed_document_generates_and_saves_vectors(monkeypatch):
    chunks = [
        SimpleNamespace(content="first"),
        SimpleNamespace(content="second"),
    ]
    vectors = [[0.1], [0.2]]
    get_chunks = AsyncMock(return_value=chunks)
    generate = AsyncMock(return_value=vectors)
    save = AsyncMock()

    monkeypatch.setattr(document_embedding, "get_document_chunks", get_chunks)
    monkeypatch.setattr(document_embedding, "generate_embeddings", generate)
    monkeypatch.setattr(document_embedding, "save_chunk_embeddings", save)
    monkeypatch.setattr(
        document_embedding,
        "settings",
        SimpleNamespace(embedding_model="test-model"),
    )

    result = await document_embedding.embed_document(object(), 9)

    assert result == chunks
    generate.assert_awaited_once_with(["first", "second"])
    save.assert_awaited_once_with(
        get_chunks.call_args.args[0],
        chunks,
        vectors,
        "test-model",
    )


@pytest.mark.anyio
async def test_generate_embedding_sends_ollama_request(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://localhost:11434/",
            embedding_model="test-model",
        ),
    )
    vector = [0.1] * embeddings.EMBEDDING_DIMENSIONS
    response = SimpleNamespace(
        json=lambda: {"embedding": vector},
        raise_for_status=MagicMock(),
    )
    client = SimpleNamespace(post=AsyncMock(return_value=response))

    result = await embeddings._generate_embedding("hello", client)

    assert result == vector
    client.post.assert_awaited_once_with(
        "http://localhost:11434/api/embeddings",
        json={"model": "test-model", "prompt": "hello"},
    )
    response.raise_for_status.assert_called_once()


@pytest.mark.anyio
async def test_generate_embedding_rejects_missing_vector(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            embedding_model="test-model",
        ),
    )
    response = SimpleNamespace(
        json=lambda: {},
        raise_for_status=MagicMock(),
    )
    client = SimpleNamespace(post=AsyncMock(return_value=response))

    with pytest.raises(ValueError, match="did not return an embedding"):
        await embeddings._generate_embedding("hello", client)


@pytest.mark.anyio
async def test_generate_embedding_rejects_wrong_dimensions(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            embedding_model="test-model",
        ),
    )
    response = SimpleNamespace(
        json=lambda: {"embedding": [0.1, 0.2]},
        raise_for_status=MagicMock(),
    )
    client = SimpleNamespace(post=AsyncMock(return_value=response))

    with pytest.raises(ValueError, match="Expected 768 dimensions"):
        await embeddings._generate_embedding("hello", client)


@pytest.mark.anyio
async def test_generate_embeddings_processes_each_text(monkeypatch):
    calls = []

    async def fake_generate(text, client):
        calls.append(text)
        return [float(len(text))]

    monkeypatch.setattr(embeddings, "_generate_embedding", fake_generate)

    result = await embeddings.generate_embeddings(["one", "two", "three"])

    assert calls == ["one", "two", "three"]
    assert result == [[3.0], [3.0], [5.0]]


@pytest.mark.anyio
async def test_retrieve_returns_vector_matches(monkeypatch):
    chunk = SimpleNamespace(id=4, content="matching content")
    result = SimpleNamespace(all=lambda: [(chunk, 0.15)])
    db = SimpleNamespace(execute=AsyncMock(return_value=result))
    generate = AsyncMock(return_value=[0.1] * 768)

    monkeypatch.setattr(retrieval, "generate_embedding", generate)
    monkeypatch.setattr(
        retrieval,
        "settings",
        SimpleNamespace(embedding_model="test-model"),
    )

    matches = await retrieval.retrieve(
        db,
        document_id=12,
        question="What is this?",
        keyword_limit=0,
    )

    assert matches == [(chunk, 0.15)]
    generate.assert_awaited_once_with("What is this?")
    db.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_retrieve_merges_keyword_and_vector_matches(monkeypatch):
    keyword_chunk = SimpleNamespace(id=1, content="keyword match")
    shared_chunk = SimpleNamespace(id=2, content="both matches")
    vector_chunk = SimpleNamespace(id=3, content="vector match")

    results = [
        SimpleNamespace(
            all=lambda: [
                (keyword_chunk, 0.1),
                (shared_chunk, 0.2),
            ]
        ),
        SimpleNamespace(
            all=lambda: [
                (shared_chunk, 0.2),
                (vector_chunk, 0.3),
            ]
        ),
    ]
    db = SimpleNamespace(execute=AsyncMock(side_effect=results))

    async def fake_embedding(question):
        return [0.1] * 768

    monkeypatch.setattr(retrieval, "generate_embedding", fake_embedding)
    monkeypatch.setattr(
        retrieval,
        "settings",
        SimpleNamespace(embedding_model="test-model"),
    )

    matches = await retrieval.retrieve(
        db,
        document_id=12,
        question="What happened in 2024?",
    )

    assert [chunk.id for chunk, distance in matches] == [2, 3, 1]
    assert len(matches) == 3
    assert db.execute.await_count == 2
