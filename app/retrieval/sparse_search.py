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
        Ensure BM25 corpus is initialized (lazy loading) with retry for Qdrant indexing delays.

        Qdrant indexing is asynchronous - chunks may not be immediately searchable after upsert.
        This method retries up to 3 times with exponential backoff to handle indexing delays.
        """
        if self.bm25 is not None and not force:
            return

        async with self._rebuild_lock:
            # Double-check after acquiring lock
            if self.bm25 is not None and not force:
                return

            logger.info("Building BM25 corpus from Qdrant chunks...")
            start_time = time.time()

            # Retry parameters for Qdrant indexing delays
            max_retries = 3
            base_delay = 0.5  # seconds

            for attempt in range(max_retries):
                try:
                    # Fetch all chunks from Qdrant (with payload)
                    points = await self._fetch_all_chunks()

                    # Handle empty results (Qdrant indexing delay)
                    if not points:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)  # Exponential backoff: 0.5s, 1s, 2s
                            logger.warning(
                                f"No chunks found in collection '{self.vector_store.collection_name}' "
                                f"(attempt {attempt + 1}/{max_retries}). "
                                f"Retrying in {delay:.1f}s (Qdrant indexing delay)..."
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(
                                "No chunks found in Qdrant after retries. "
                                "BM25 corpus will be empty (searches will return 0 results)."
                            )
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
                        f"✓ BM25 corpus built successfully: "
                        f"{len(self.corpus)} chunks in {duration:.2f}s "
                        f"(after {attempt + 1} attempt(s))"
                    )
                    self._last_rebuild = time.time()
                    return  # Success - exit function

                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to build BM25 corpus after {max_retries} attempts: {e}",
                            exc_info=True,
                        )
                        raise
                    else:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Corpus build attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)

    async def _fetch_all_chunks(self) -> list[VectorPoint]:
        """
        Fetch all chunks from Qdrant collection (bulletproof for all qdrant-client versions).

        Handles:
        - Legacy (<1.8): scroll() returns tuple (points, next_offset)
        - Modern (≥1.8): scroll() returns ScrollResponse with .points/.next_page_offset
        - Empty first scroll (Qdrant indexing delay) - retries internally

        Returns:
            List of VectorPoint objects with payloads
        """
        client = self.vector_store._client
        if client is None:
            raise RuntimeError(
                "Qdrant client not initialized. Call vector_store._initialize() first."
            )

        points: list[VectorPoint] = []
        offset = None
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            try:
                # Execute scroll
                scroll_result = await asyncio.to_thread(
                    client.scroll,
                    collection_name=self.vector_store.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                # ✅ BULLETPROOF TYPE DETECTION
                if isinstance(scroll_result, tuple):
                    # Legacy API (<1.8)
                    batch_points, next_offset = scroll_result
                    logger.debug(
                        f"Detected legacy qdrant-client API: fetched {len(batch_points)} points"
                    )
                else:
                    # Modern API (≥1.8)
                    batch_points = getattr(scroll_result, "points", [])
                    next_offset = getattr(scroll_result, "next_page_offset", None)
                    logger.debug(
                        f"Detected modern qdrant-client API: fetched {len(batch_points)} points"
                    )

                # ✅ CRITICAL FIX: Handle empty first scroll (Qdrant indexing delay)
                if not batch_points and offset is None and attempt < max_attempts - 1:
                    attempt += 1
                    delay = 0.5 * (2**attempt)  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Empty scroll result on attempt {attempt} (Qdrant indexing delay). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                # Process points
                for record in batch_points:
                    points.append(
                        VectorPoint(
                            id=str(record.id),
                            vector=[],
                            payload=record.payload or {},
                            score=None,
                        )
                    )

                # Check for more pages
                if next_offset is None:
                    break
                offset = next_offset
                attempt = 0  # Reset attempt counter on successful page fetch

            except Exception as e:
                logger.error(f"Scroll failed on attempt {attempt + 1}: {e}", exc_info=True)
                attempt += 1
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))

        logger.info(
            f"✓ Fetched {len(points)} chunks from '{self.vector_store.collection_name}' "
            f"after {attempt + 1} attempt(s)"
        )
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
                if scores[idx] < 0:  # Skip only negative scores
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
