"""Unit tests for Retriever — ChromaDB and Embedder are fully mocked."""
from unittest.mock import MagicMock, patch, call
import pytest

from adaptiverag.components.retriever import Retriever


# ── fixtures ───────────────────────────────────────────────────────────────

def _fake_embedding(dim: int = 8) -> list[float]:
    return [0.1] * dim


def make_retriever() -> tuple[Retriever, MagicMock, MagicMock]:
    """Return (retriever, mock_embedder, mock_collection)."""
    embedder   = MagicMock()
    embedder.batch_size = 50
    embedder.embed.return_value = [_fake_embedding()]

    collection = MagicMock()
    collection.name = "test_collection"

    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    with patch("adaptiverag.components.retriever.chromadb.PersistentClient",
               return_value=client):
        retriever = Retriever(embedder, persist_dir="/tmp/test_chroma")

    return retriever, embedder, collection


# ── count ──────────────────────────────────────────────────────────────────

class TestCount:
    def test_count_delegates_to_collection(self):
        retriever, _, collection = make_retriever()
        collection.count.return_value = 42
        assert retriever.count == 42

    def test_count_zero_on_empty(self):
        retriever, _, collection = make_retriever()
        collection.count.return_value = 0
        assert retriever.count == 0


# ── clear ──────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_deletes_and_recreates_collection(self):
        retriever, _, collection = make_retriever()
        retriever.clear()
        retriever._client.delete_collection.assert_called_once_with("test_collection")
        retriever._client.get_or_create_collection.assert_called_with("test_collection")

    def test_clear_resets_collection_reference(self):
        retriever, _, collection = make_retriever()
        new_collection = MagicMock()
        retriever._client.get_or_create_collection.return_value = new_collection
        retriever.clear()
        assert retriever.collection is new_collection


# ── index ──────────────────────────────────────────────────────────────────

class TestIndex:
    def test_index_calls_collection_add(self):
        retriever, embedder, collection = make_retriever()
        chunks = ["chunk one", "chunk two"]
        metas  = [{"source": "a.txt"}, {"source": "a.txt"}]
        embedder.embed.return_value = [_fake_embedding(), _fake_embedding()]

        retriever.index(chunks, metas)

        collection.add.assert_called_once()
        kwargs = collection.add.call_args.kwargs
        assert kwargs["documents"] == chunks
        assert kwargs["metadatas"] == metas
        assert len(kwargs["ids"]) == 2

    def test_index_generates_unique_ids(self):
        retriever, embedder, collection = make_retriever()
        chunks = ["a", "b", "c"]
        metas  = [{"source": "f"}] * 3
        embedder.embed.return_value = [_fake_embedding()] * 3

        retriever.index(chunks, metas)

        ids = collection.add.call_args.kwargs["ids"]
        assert len(ids) == len(set(ids)), "IDs must be unique"

    def test_index_batches_embedding_calls(self):
        retriever, embedder, collection = make_retriever()
        retriever.embedder.batch_size = 2
        chunks = ["a", "b", "c", "d", "e"]
        metas  = [{"source": "f"}] * 5
        embedder.embed.return_value = [_fake_embedding(), _fake_embedding()]

        retriever.index(chunks, metas)

        # 5 chunks / batch_size 2 → ceil(5/2) = 3 embed calls
        assert embedder.embed.call_count == 3

    def test_index_empty_chunks_does_not_call_add(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = []
        retriever.index([], [])
        collection.add.assert_called_once()
        kwargs = collection.add.call_args.kwargs
        assert kwargs["documents"] == []


# ── retrieve ───────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_retrieve_returns_documents(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = [_fake_embedding()]
        collection.query.return_value = {"documents": [["doc one", "doc two"]]}

        results = retriever.retrieve("test query", top_k=2)
        assert results == ["doc one", "doc two"]

    def test_retrieve_empty_collection_returns_empty_list(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = [_fake_embedding()]
        collection.query.return_value = {"documents": []}

        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_passes_top_k(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = [_fake_embedding()]
        collection.query.return_value = {"documents": [[]]}

        retriever.retrieve("q", top_k=7)

        kwargs = collection.query.call_args.kwargs if collection.query.call_args.kwargs \
                 else collection.query.call_args[1]
        # n_results is passed as positional or keyword
        args, kwargs = collection.query.call_args
        assert kwargs.get("n_results") == 7

    def test_retrieve_with_source_filter(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = [_fake_embedding()]
        collection.query.return_value = {"documents": [["filtered doc"]]}

        results = retriever.retrieve("q", top_k=3, where={"source": "notes.txt"})

        args, kwargs = collection.query.call_args
        assert kwargs.get("where") == {"source": "notes.txt"}
        assert results == ["filtered doc"]

    def test_retrieve_without_filter_omits_where(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = [_fake_embedding()]
        collection.query.return_value = {"documents": [[]]}

        retriever.retrieve("q")

        args, kwargs = collection.query.call_args
        assert "where" not in kwargs

    def test_retrieve_embeds_query_before_searching(self):
        retriever, embedder, collection = make_retriever()
        embedder.embed.return_value = [_fake_embedding()]
        collection.query.return_value = {"documents": [[]]}

        retriever.retrieve("my query", top_k=3)

        embedder.embed.assert_called_once_with(["my query"])
