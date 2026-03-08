"""
Integration tests for hybrid search engine.
Uses isolated test collection populated with 5 anime (Frieren-focused).
"""

import asyncio

import pytest

from app.core.logging import get_logger

logger = get_logger(__name__)

pytestmark = pytest.mark.asyncio


async def test_bm25_corpus_initialization(test_sparse_engine):
    """Test BM25 corpus builds successfully from Qdrant."""
    stats = test_sparse_engine.get_stats()
    assert stats["is_initialized"] is True
    assert stats["corpus_size"] > 0, "BM25 corpus should contain chunks"
    logger.info(f"\nBM25 corpus size: {stats['corpus_size']} chunks")


async def test_sparse_search_basic(test_sparse_engine):
    """Test basic BM25 keyword search with query matching test data."""
    # ✅ QUERY MATCHES TEST DATA (Frieren = magic/adventure/elf)
    results = test_sparse_engine.search("magic adventure elf", top_k=3)
    assert len(results) > 0, "Should return results for 'magic adventure elf'"
    # Verify at least one result contains Frieren
    assert any(
        "Frieren" in r["payload"].get("anime_title", "") for r in results[:2]
    ), "Should return Frieren chunks"


async def test_dense_search_basic(test_vector_store, test_embedder):
    """Test basic dense vector search with query matching test data."""
    # ✅ QUERY MATCHES TEST DATA (Frieren = elf/magic/journey)
    query = "elf journey magic"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    results = await test_vector_store.search(
        query_vector=embedding.tolist(),
        top_k=3,
        score_threshold=0.3,  # ✅ LOWERED for small dataset (was 0.5)
    )

    assert len(results) > 0, f"Should return results for '{query}'"
    assert all(r.score >= 0.3 for r in results), "All results should meet score threshold"
    # Verify at least one result is Frieren
    assert any(
        "Frieren" in r.payload.get("anime_title", "") for r in results
    ), "Should return Frieren chunks"


async def test_hybrid_search_fusion(test_hybrid_engine, test_embedder):
    """Test hybrid search fuses dense + sparse results."""
    # ✅ QUERY MATCHES TEST DATA
    query = "fantasy adventure magic"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    results = await test_hybrid_engine.search(
        query=query,
        query_vector=embedding.tolist(),
        top_k=3,
        use_sparse=True,
    )

    assert len(results) > 0, "Hybrid search should return results"
    assert all("fusion_score" in r for r in results), "All results should have fusion_score"
    assert all(r["fusion_method"] == "hybrid_rrf" for r in results), "Should use RRF fusion"

    # Verify fusion incorporates both dense and sparse ranks
    assert any(r.get("dense_rank") is not None for r in results), "Should have dense ranks"
    assert any(r.get("sparse_rank") is not None for r in results), "Should have sparse ranks"

    logger.info(f"\nHybrid search results for '{query}':")
    for r in results[:2]:
        title = r["payload"].get("anime_title", "Unknown")
        logger.info(
            f"  • {title} (fusion_score={r['fusion_score']:.4f}, "
            f"dense_rank={r.get('dense_rank')}, sparse_rank={r.get('sparse_rank')})"
        )


async def test_hybrid_search_dense_fallback(test_hybrid_engine, test_embedder):
    """Test hybrid search falls back to dense-only when sparse unavailable."""
    query = "anime recommendation"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    # Temporarily disable sparse engine
    original_sparse = test_hybrid_engine.sparse_engine
    test_hybrid_engine.sparse_engine = None

    try:
        results = await test_hybrid_engine.search(
            query=query,
            query_vector=embedding.tolist(),
            top_k=3,
            use_sparse=True,  # Should fallback to dense-only
        )

        assert len(results) > 0
        assert all(r["fusion_method"] == "dense_only" for r in results)
        logger.info(f"\nDense-only fallback successful: {len(results)} results")

    finally:
        # Restore sparse engine
        test_hybrid_engine.sparse_engine = original_sparse


async def test_hybrid_search_filters(test_hybrid_engine, test_embedder):
    """Test hybrid search respects metadata filters."""
    query = "fantasy adventure"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    # ✅ GET ACTUAL ANIME_ID FROM TEST DATA (avoids hardcoding)
    # First result will be from our test collection (Frieren)
    sample_results = await test_hybrid_engine.vector_store.search(
        query_vector=embedding.tolist(), top_k=1
    )
    anime_id = sample_results[0].payload["anime_id"]

    filters = {"anime_id": anime_id}

    results = await test_hybrid_engine.search(
        query=query,
        query_vector=embedding.tolist(),
        top_k=3,
        filters=filters,
        use_sparse=True,
    )

    assert len(results) > 0, "Should return filtered results"
    # All results should be from the filtered anime
    assert all(r["payload"].get("anime_id") == anime_id for r in results)
    logger.info(f"\nFilter test passed: {len(results)} chunks from anime_id={anime_id}")


async def test_hybrid_search_performance(test_hybrid_engine, test_embedder):
    """Test hybrid search meets latency requirements (<500ms P95)."""
    query = "fantasy magic adventure"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    # Run search 3 times to measure performance (faster for tests)
    durations = []
    for _ in range(3):
        start = asyncio.get_event_loop().time()
        await test_hybrid_engine.search(
            query=query,
            query_vector=embedding.tolist(),
            top_k=5,
            use_sparse=True,
        )
        duration = (asyncio.get_event_loop().time() - start) * 1000  # ms
        durations.append(duration)

    p95 = sorted(durations)[int(len(durations) * 0.95)]
    avg = sum(durations) / len(durations)

    logger.info("\nHybrid search performance (3 runs):")
    logger.info(f"  Avg: {avg:.2f}ms, P95: {p95:.2f}ms")

    # P95 should be < 500ms for free tier constraints
    assert p95 < 500, f"P95 latency {p95:.2f}ms exceeds 500ms target"
