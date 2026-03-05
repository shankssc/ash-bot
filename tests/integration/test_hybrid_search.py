"""
Integration tests for hybrid search engine.
Requires Qdrant collection with test data (run ingestion pipeline first).
"""

import asyncio

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.secrets import secrets
from app.ingestion.embedder import EmbeddingGenerator
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.sparse_search import SparseSearchEngine
from app.retrieval.vector_store import QdrantVectorStore

# IMPORTANT: pytest-asyncio requires this marker for async tests
pytestmark = pytest.mark.asyncio

logger = get_logger(__name__)


@pytest_asyncio.fixture(scope="function")
async def vector_store() -> AsyncGenerator[QdrantVectorStore, None]:
    """Fixture: Qdrant vector store instance (function-scoped for safety)."""
    settings = get_settings()
    store = QdrantVectorStore(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    # Initialize connection
    await store._initialize()
    yield store
    # No teardown needed for read-only tests


@pytest_asyncio.fixture(scope="function")
async def sparse_engine(
    vector_store: QdrantVectorStore,
) -> AsyncGenerator[SparseSearchEngine, None]:
    """Fixture: Sparse search engine instance."""
    engine = SparseSearchEngine.get_instance(vector_store)
    # Build corpus (may take 10-30s for first run)
    await engine.ensure_initialized()
    yield engine


@pytest_asyncio.fixture(scope="function")
async def hybrid_engine(
    vector_store: QdrantVectorStore, sparse_engine: SparseSearchEngine
) -> AsyncGenerator[HybridSearchEngine, None]:
    """Fixture: Hybrid search engine instance."""
    engine = HybridSearchEngine(
        vector_store=vector_store,
        sparse_engine=sparse_engine,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
    )
    yield engine


@pytest_asyncio.fixture(scope="function")
async def embedder() -> AsyncGenerator[EmbeddingGenerator, None]:
    """Fixture: Embedding generator instance."""
    generator = EmbeddingGenerator()
    yield generator


async def test_bm25_corpus_initialization(sparse_engine: SparseSearchEngine):
    """Test BM25 corpus builds successfully from Qdrant."""
    stats = sparse_engine.get_stats()
    assert stats["is_initialized"] is True
    assert stats["corpus_size"] > 0, "BM25 corpus should contain chunks"
    logger.info(f"\nBM25 corpus size: {stats['corpus_size']} chunks")


async def test_sparse_search_basic(sparse_engine: SparseSearchEngine):
    """Test basic BM25 keyword search."""
    results = sparse_engine.search("space western", top_k=5)
    assert len(results) > 0, "Should return results for 'space western'"
    # Verify at least one result has reasonable score
    assert any(r["score"] > 0.1 for r in results), "Should have non-zero BM25 scores"


'''
async def test_dense_search_basic(vector_store: QdrantVectorStore, embedder: EmbeddingGenerator):
    """Test basic dense vector search."""
    query = "space western anime"
    embedding, _ = await asyncio.to_thread(embedder.generate_single, query)

    results = await vector_store.search(
        query_vector=embedding.tolist(),
        top_k=5,
        score_threshold=0.5
    )

    assert len(results) > 0, "Should return results for space western query"
    assert all(r.score >= 0.5 for r in results), "All results should meet score threshold"
'''


async def test_hybrid_search_fusion(
    hybrid_engine: HybridSearchEngine, embedder: EmbeddingGenerator
):
    """Test hybrid search fuses dense + sparse results."""
    query = "space western bounty hunter"
    embedding, _ = await asyncio.to_thread(embedder.generate_single, query)

    results = await hybrid_engine.search(
        query=query, query_vector=embedding.tolist(), top_k=5, use_sparse=True
    )

    assert len(results) > 0, "Hybrid search should return results"
    assert all("fusion_score" in r for r in results), "All results should have fusion_score"
    assert all(r["fusion_method"] == "hybrid_rrf" for r in results), "Should use RRF fusion"

    # Verify fusion incorporates both dense and sparse ranks
    assert any(r.get("dense_rank") is not None for r in results), "Should have dense ranks"
    assert any(r.get("sparse_rank") is not None for r in results), "Should have sparse ranks"

    logger.info(f"\nHybrid search results for '{query}':")
    for r in results[:3]:
        title = r["payload"].get("anime_title", "Unknown")
        logger.info(
            f"  • {title} (fusion_score={r['fusion_score']:.4f}, dense_rank={r.get('dense_rank')}, sparse_rank={r.get('sparse_rank')})"
        )


async def test_hybrid_search_dense_fallback(
    hybrid_engine: HybridSearchEngine, embedder: EmbeddingGenerator
):
    """Test hybrid search falls back to dense-only when sparse unavailable."""
    query = "anime recommendation"
    embedding, _ = await asyncio.to_thread(embedder.generate_single, query)

    # Temporarily disable sparse engine
    original_sparse = hybrid_engine.sparse_engine
    hybrid_engine.sparse_engine = None

    try:
        results = await hybrid_engine.search(
            query=query,
            query_vector=embedding.tolist(),
            top_k=5,
            use_sparse=True,  # Should fallback to dense-only
        )

        assert len(results) > 0
        assert all(r["fusion_method"] == "dense_only" for r in results)
        logger.info(f"\nDense-only fallback successful: {len(results)} results")

    finally:
        # Restore sparse engine
        hybrid_engine.sparse_engine = original_sparse


async def test_hybrid_search_filters(
    hybrid_engine: HybridSearchEngine, embedder: EmbeddingGenerator
):
    """Test hybrid search respects metadata filters."""
    query = "shonen battle anime"
    embedding, _ = await asyncio.to_thread(embedder.generate_single, query)

    # Filter for a specific anime (should return only chunks from that anime)
    filters = {"anime_id": 5114}  # Fullmetal Alchemist: Brotherhood

    results = await hybrid_engine.search(
        query=query, query_vector=embedding.tolist(), top_k=5, filters=filters, use_sparse=True
    )

    if results:
        # All results should be from the filtered anime
        assert all(r["payload"].get("anime_id") == 5114 for r in results)
        logger.info(f"\nFilter test passed: {len(results)} chunks from FMA:B")


async def test_hybrid_search_performance(
    hybrid_engine: HybridSearchEngine, embedder: EmbeddingGenerator
):
    """Test hybrid search meets latency requirements (<500ms P95)."""
    query = "best isekai anime recommendations"
    embedding, _ = await asyncio.to_thread(embedder.generate_single, query)

    # Run search 5 times to measure performance
    durations = []
    for _ in range(5):
        start = asyncio.get_event_loop().time()
        await hybrid_engine.search(
            query=query, query_vector=embedding.tolist(), top_k=10, use_sparse=True
        )
        duration = (asyncio.get_event_loop().time() - start) * 1000  # ms
        durations.append(duration)

    p95 = sorted(durations)[int(len(durations) * 0.95)]
    avg = sum(durations) / len(durations)

    logger.info("\nHybrid search performance (5 runs):")
    logger.info(f"  Avg: {avg:.2f}ms, P95: {p95:.2f}ms")

    # P95 should be < 500ms for free tier constraints
    assert p95 < 500, f"P95 latency {p95:.2f}ms exceeds 500ms target"


if __name__ == "__main__":
    # Manual test runner (for quick validation without pytest)
    async def run_tests():
        from app.core.config import settings

        # Initialize vector store
        store = QdrantVectorStore(
            url=settings.QDRANT_URL,
            api_key=secrets.QDRANT_API_KEY.get_secret_value(),
            collection_name=settings.QDRANT_COLLECTION_NAME,
        )
        await store._initialize()

        # Initialize sparse engine
        sparse = SparseSearchEngine.get_instance(store)
        await sparse.ensure_initialized()

        # Initialize hybrid engine
        hybrid = HybridSearchEngine(store, sparse)

        # Initialize embedder
        embedder = EmbeddingGenerator()

        # Run search
        query = "space western anime"
        embedding, _ = await asyncio.to_thread(embedder.generate_single, query)

        results = await hybrid.search(query=query, query_vector=embedding.tolist(), top_k=5)

        logger.info(f"\n✅ Hybrid search returned {len(results)} results:")
        for r in results:
            title = r["payload"].get("anime_title", "Unknown")
            logger.info(f"  • {title} (score: {r['fusion_score']:.4f})")

    asyncio.run(run_tests())
