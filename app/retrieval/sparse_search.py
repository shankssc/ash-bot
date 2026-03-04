"""
Sparse keyword search using BM25Okapi for hybrid retrieval.
"""

from __future__ import annotations

import asyncio
import time

from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.vector_store import QdrantVectorStore, VectorPoint

logger = get_logger(__name__)
settings = get_settings()


class BM25Corpus:
    """
    In-memory BM25 corpus built from Qdrant chunks.

    Design decisions (per spec):
    - Phase 2: In-memory corpus rebuilt on startup (simple, fast for <10K docs)
    - Phase 3: Redis-backed persistence (future enhancement)
    - Corpus includes chunk text + metadata fields for richer keyword matching
    - Tokenization: Whitespace + punctuation splitting (anime-specific tuning)
    """

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.bm25: BM25Okapi | None = None
        self.corpus: list[str] = []
        self.chunk_ids: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self._last_rebuild: float = 0.0
        self._rebuild_lock = asyncio.Lock()

    async def ensure_initialized(self, force: bool = False) -> None:
        """
        Ensure BM25 corpus is initialized (lazy loading).

        Args:
            force: Force rebuild even if already initialized
        """
        if self.bm25 is not None and not force:
            return

        async with self._rebuild_lock:
            # Double-check after acquiring lock
            if self.bm25 is not None and not force:
                return

            logger.info("Building BM25 corpus from Qdrant chunks...")
            start_time = time.time()

            try:
                # Fetch all chunks from Qdrant (with payload)
                points = await self._fetch_all_chunks()

                if not points:
                    logger.warning("No chunks found in Qdrant for BM25 corpus")
                    self.bm25 = None
                    self.corpus = []
                    self.chunk_ids = []
                    self.payloads = []
                    return

                # Build corpus with text + metadata for richer matching
                self.corpus = []
                self.chunk_ids = []
                self.payloads = []

                for point in points:
                    chunk_text = point.payload.get("text", "")
                    anime_title = point.payload.get("anime_title", "")
                    source_type = point.payload.get("source_type", "")

                    # Enrich corpus with metadata for better keyword matching
                    enriched_text = f"{anime_title} {source_type} {chunk_text}"
                    self.corpus.append(enriched_text)
                    self.chunk_ids.append(point.id)
                    self.payloads.append(point.payload)

                # Tokenize and build BM25 index
                tokenized_corpus = [self._tokenize(text) for text in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)

                duration = time.time() - start_time
                logger.info(
                    f"BM25 corpus built successfully: {len(self.corpus)} chunks in {duration:.2f}s"
                )
                self._last_rebuild = time.time()

            except Exception as e:
                logger.error(f"Failed to build BM25 corpus: {e}", exc_info=True)
                raise

    async def _fetch_all_chunks(self) -> list[VectorPoint]:
        """
        Fetch all chunks from Qdrant collection.

        Returns:
            List of VectorPoint objects with payloads
        """
        client = self.vector_store._client
        if client is None:
            raise RuntimeError(
                "Qdrant client not initialized. Call vector_store.initialize() first."
            )

        # Use scroll API for efficient large collection retrieval
        points: list[VectorPoint] = []
        offset = None

        while True:
            # Qdrant scroll returns points + next offset
            scroll_result = await asyncio.to_thread(
                client.scroll,
                collection_name=self.vector_store.collection_name,
                limit=100,  # Batch size for free tier efficiency
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            batch_points, next_offset = scroll_result

            # Convert to VectorPoint objects
            for record in batch_points:
                points.append(
                    VectorPoint(
                        id=str(record.id), vector=[], payload=record.payload or {}, score=None
                    )
                )

            offset = next_offset
            if offset is None:
                break

        return points

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 (anime-specific tuning).

        Handles:
        - Japanese romanization (e.g., "Shingeki no Kyojin")
        - Anime-specific terms ("isekai", "shonen", "mecha")
        - Punctuation removal while preserving meaningful hyphens

        Args:
            text: Raw text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and normalize whitespace
        text = text.lower().strip()

        # Replace common punctuation with spaces (preserve hyphens in terms like "space-western")
        text = text.replace("'", " ").replace('"', " ").replace(",", " ").replace(".", " ")
        text = text.replace("!", " ").replace("?", " ").replace(":", " ").replace(";", " ")
        text = text.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
        text = text.replace("{", " ").replace("}", " ").replace("/", " ")

        # Split on whitespace
        tokens = text.split()

        # Filter out very short tokens (likely noise)
        tokens = [t for t in tokens if len(t) >= 2]

        return tokens

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of results with BM25 scores
        """
        if self.bm25 is None:
            logger.warning("BM25 corpus not initialized. Returning empty results.")
            return []

        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)

            if not tokenized_query:
                logger.debug(f"Query tokenized to empty: '{query}'")
                return []

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k results with scores
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] <= 0:  # Skip zero/negative scores
                    continue

                results.append(
                    {
                        "id": self.chunk_ids[idx],
                        "score": float(scores[idx]),  # BM25 score (not normalized)
                        "payload": self.payloads[idx],
                        "text_snippet": self._get_snippet(self.corpus[idx], query),
                    }
                )

            logger.debug(f"BM25 search returned {len(results)} results for query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}", exc_info=True)
            return []

    def _get_snippet(self, text: str, query: str) -> str:
        """
        Extract relevant snippet around query terms.

        Args:
            text: Full chunk text
            query: Original query

        Returns:
            Short excerpt containing query terms
        """
        query_lower = query.lower()
        text_lower = text.lower()

        # Find first occurrence of any query word
        for word in query_lower.split():
            if word in text_lower:
                start = max(0, text_lower.index(word) - 50)
                end = min(len(text), text_lower.index(word) + 100)
                return text[start:end] + "..." if end < len(text) else text[start:end]

        # Fallback: first 100 chars
        return text[:100] + "..." if len(text) > 100 else text

    def get_stats(self) -> dict[str, Any]:
        """Get corpus statistics."""
        return {
            "corpus_size": len(self.corpus),
            "last_rebuild": self._last_rebuild,
            "is_initialized": self.bm25 is not None,
        }


class SparseSearchEngine:
    """
    Production-ready sparse search engine with BM25.

    Usage:
        sparse_engine = SparseSearchEngine(vector_store)
        await sparse_engine.ensure_initialized()
        results = sparse_engine.search("space western anime")
    """

    _instance: SparseSearchEngine | None = None

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.bm25_corpus = BM25Corpus(vector_store)

    @classmethod
    def get_instance(cls, vector_store: QdrantVectorStore) -> SparseSearchEngine:
        """
        Get singleton instance (for dependency injection).

        Args:
            vector_store: Qdrant vector store instance

        Returns:
            SparseSearchEngine singleton
        """
        if cls._instance is None:
            cls._instance = cls(vector_store)
        return cls._instance

    async def ensure_initialized(self, force: bool = False) -> None:
        """Ensure BM25 corpus is initialized."""
        await self.bm25_corpus.ensure_initialized(force=force)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Perform BM25 search."""
        return self.bm25_corpus.search(query, top_k)

    def get_stats(self) -> dict[str, Any]:
        """Get search engine statistics."""
        return self.bm25_corpus.get_stats()
