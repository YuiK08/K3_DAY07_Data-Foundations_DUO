from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": dict(doc.metadata),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                embedding = self._embedding_fn(doc.content)
                ids.append(doc.id)
                documents.append(doc.content)
                embeddings.append(embedding)
                metadatas.append(doc.metadata if doc.metadata else {})
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
            output = []
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]
            for i, doc_id in enumerate(ids):
                score = -distances[i] if distances else 0.0
                output.append({
                    "id": doc_id,
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "score": score,
                })
            return output
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            # Build ChromaDB where filter
            where = {k: v for k, v in metadata_filter.items()}
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
            output = []
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]
            for i, doc_id in enumerate(ids):
                score = -distances[i] if distances else 0.0
                output.append({
                    "id": doc_id,
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "score": score,
                })
            return output
        else:
            # In-memory: filter records by metadata, then run similarity search
            filtered = [
                record for record in self._store
                if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            # Check if exists before deleting
            try:
                existing = self._collection.get(ids=[doc_id])
                if not existing["ids"]:
                    return False
                self._collection.delete(ids=[doc_id])
                return True
            except Exception:
                return False
        else:
            before = len(self._store)
            self._store = [
                record for record in self._store
                if record["metadata"].get("doc_id", record["id"]) != doc_id
                and record["id"] != doc_id
            ]
            return len(self._store) < before
