"""
Integration tests for hybrid search engine.
Uses isolated test collection populated with 5 anime (top-ranked from Jikan).
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


async def test_sparse_search_basic(test_sparse_engine, ingested_titles):
    """Test basic BM25 keyword search returns results from the ingested collection."""
    results = test_sparse_engine.search("magic adventure elf", top_k=3)
    assert len(results) > 0, "Should return results for 'magic adventure elf'"

    result_titles = {r["payload"].get("anime_title", "") for r in results}
    logger.info(f"\nSparse search returned titles: {result_titles}")
    logger.info(f"Ingested titles: {ingested_titles}")

    assert any(
        title in ingested_titles for title in result_titles
    ), f"Should return chunks from ingested anime. Got: {result_titles}, expected one of: {ingested_titles}"


async def test_dense_search_basic(test_vector_store, test_embedder):
    """Test basic dense vector search with query matching test data."""
    query = "elf journey magic"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    results = await test_vector_store.search(
        query_vector=embedding.tolist(),
        top_k=3,
        score_threshold=0.3,
    )

    assert len(results) > 0, f"Should return results for '{query}'"
    assert all(r.score >= 0.3 for r in results), "All results should meet score threshold"


async def test_hybrid_search_fusion(test_hybrid_engine, test_embedder):
    """
    Test hybrid search correctly fuses dense + sparse results using RRF.

    Uses top_k=10 (not 3) to get a wide enough result window to observe
    both dense-only and sparse-contributing hits. With only 21 chunks across
    5 anime, the top-3 results are often all from dense (Frieren dominates
    semantically), so we need to look at the broader fused list to confirm
    that sparse_rank is being set on at least some results.

    What this test actually verifies:
    - fusion_method is "hybrid_rrf" (not dense_only fallback)
    - fusion_score is present on all results
    - dense_rank is present on at least one result
    - sparse_rank is present on at least one result (proves BM25 IDs match Qdrant IDs)
    """
    query = "fantasy adventure magic"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    # FIX: Use top_k=10 to give RRF enough results to show both dense and sparse ranks.
    # With top_k=3 on a 21-chunk corpus, top results are all Frieren (dense-dominant)
    # and sparse hits (FMAB, Steins;Gate) only appear further down the fused list.
    results = await test_hybrid_engine.search(
        query=query,
        query_vector=embedding.tolist(),
        top_k=10,
        use_sparse=True,
    )

    assert len(results) > 0, "Hybrid search should return results"
    assert all("fusion_score" in r for r in results), "All results should have fusion_score"
    assert all(r["fusion_method"] == "hybrid_rrf" for r in results), "Should use RRF fusion"
    assert any(r.get("dense_rank") is not None for r in results), "Should have dense ranks"

    logger.info(f"\nHybrid search rank breakdown (top_k=10, {len(results)} results):")
    for r in results:
        title = r["payload"].get("anime_title", "Unknown")
        logger.info(
            f"  • {title} (fusion_score={r['fusion_score']:.4f}, "
            f"dense_rank={r.get('dense_rank')}, sparse_rank={r.get('sparse_rank')})"
        )

    # Verify BM25 is contributing ranks — at least one result should have a sparse_rank.
    # If this fails, BM25 corpus IDs don't match Qdrant IDs (initialization ordering bug).
    assert any(r.get("sparse_rank") is not None for r in results), (
        "Should have at least one result with sparse_rank across the full fused list. "
        f"Got dense_ranks: {[r.get('dense_rank') for r in results]}, "
        f"sparse_ranks: {[r.get('sparse_rank') for r in results]}. "
        "This means BM25 and Qdrant IDs are mismatched — check fixture ordering."
    )


async def test_hybrid_search_dense_fallback(test_hybrid_engine, test_embedder):
    """Test hybrid search falls back to dense-only when sparse unavailable."""
    query = "anime recommendation"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    original_sparse = test_hybrid_engine.sparse_engine
    test_hybrid_engine.sparse_engine = None

    try:
        results = await test_hybrid_engine.search(
            query=query,
            query_vector=embedding.tolist(),
            top_k=3,
            use_sparse=True,
        )

        assert len(results) > 0
        assert all(r["fusion_method"] == "dense_only" for r in results)
        logger.info(f"\nDense-only fallback successful: {len(results)} results")

    finally:
        test_hybrid_engine.sparse_engine = original_sparse


async def test_hybrid_search_filters(test_hybrid_engine, test_embedder):
    """Test hybrid search respects metadata filters."""
    query = "fantasy adventure"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    sample_results = await test_hybrid_engine.vector_store.search(
        query_vector=embedding.tolist(), top_k=1
    )
    anime_id = sample_results[0].payload["anime_id"]

    results = await test_hybrid_engine.search(
        query=query,
        query_vector=embedding.tolist(),
        top_k=3,
        filters={"anime_id": anime_id},
        use_sparse=True,
    )

    assert len(results) > 0, "Should return filtered results"
    assert all(r["payload"].get("anime_id") == anime_id for r in results)
    logger.info(f"\nFilter test passed: {len(results)} chunks from anime_id={anime_id}")


async def test_hybrid_search_performance(test_hybrid_engine, test_embedder):
    """Test hybrid search meets latency requirements (<500ms P95)."""
    query = "fantasy magic adventure"
    embedding, _ = await asyncio.to_thread(test_embedder.generate_single, query)

    durations = []
    for _ in range(3):
        start = asyncio.get_event_loop().time()
        await test_hybrid_engine.search(
            query=query,
            query_vector=embedding.tolist(),
            top_k=5,
            use_sparse=True,
        )
        duration = (asyncio.get_event_loop().time() - start) * 1000
        durations.append(duration)

    p95 = sorted(durations)[int(len(durations) * 0.95)]
    avg = sum(durations) / len(durations)

    logger.info("\nHybrid search performance (3 runs):")
    logger.info(f"  Avg: {avg:.2f}ms, P95: {p95:.2f}ms")

    assert p95 < 500, f"P95 latency {p95:.2f}ms exceeds 500ms target"
