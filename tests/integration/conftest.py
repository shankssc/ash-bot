"""
Integration test configuration with isolated Qdrant test collection.
Ensures test data never pollutes production collection.
"""

import asyncio

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest_asyncio

from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.secrets import secrets
from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.pipeline import run_minimal_pipeline
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.sparse_search import SparseSearchEngine
from app.retrieval.vector_store import QdrantVectorStore

# Unique test collection name (prevents conflicts between test runs)
TEST_COLLECTION_NAME = f"anime_knowledge_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

logger = get_logger(__name__)


@pytest_asyncio.fixture(scope="session")
async def test_vector_store():
    """
    Session-scoped Qdrant vector store for integration tests.

    Lifecycle:
    1. Deletes ALL stale anime_knowledge_test_* collections from previous runs
    2. Creates a fresh isolated test collection
    3. Yields vector store for all tests in the session
    4. Deletes the test collection on teardown
    """
    settings = get_settings()

    # Clean up ALL stale test collections (not just the current name)
    temp_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        timeout=30,
        prefer_grpc=False,
    )
    try:
        collections = await asyncio.to_thread(temp_client.get_collections)
        stale = [
            c.name for c in collections.collections if c.name.startswith("anime_knowledge_test_")
        ]
        for name in stale:
            logger.info(f"\n🗑️  Deleting stale test collection: {name}")
            await asyncio.to_thread(temp_client.delete_collection, collection_name=name)
        if stale:
            logger.info(f"✅ Cleaned up {len(stale)} stale test collection(s)")
    except Exception as e:
        logger.info(f"\n⚠️  Cleanup warning (non-fatal): {e}")
    finally:
        await asyncio.to_thread(temp_client.close)

    # Create fresh test vector store
    store = QdrantVectorStore(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        collection_name=TEST_COLLECTION_NAME,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    await store._initialize()

    logger.info(f"\n✅ Created isolated test collection: {TEST_COLLECTION_NAME}")
    yield store

    # Teardown: delete test collection
    try:
        await asyncio.to_thread(
            store._client.delete_collection, collection_name=TEST_COLLECTION_NAME
        )
        logger.info(f"\n✅ Deleted test collection: {TEST_COLLECTION_NAME}")
    except Exception as e:
        logger.info(f"\n⚠️  Cleanup warning (non-fatal): {e}")


@pytest_asyncio.fixture(scope="session")
async def test_sparse_engine(
    test_vector_store: QdrantVectorStore,
) -> AsyncGenerator[SparseSearchEngine, None]:
    """
    Session-scoped sparse search engine (corpus NOT initialized here).

    The BM25 corpus is intentionally left uninitialized at this point.
    populated_test_collection calls ensure_initialized(force=True) AFTER
    data is in Qdrant, which is the only safe moment to build the corpus.
    This eliminates the ID mismatch bug where BM25 and Qdrant returned
    results from different collections.
    """
    import app.retrieval.sparse_search as ss_module

    # Clear any stale singleton from a prior session in the same process
    ss_module.SparseSearchEngine._instance = None

    # Direct instantiation — never use get_instance() in tests
    engine = SparseSearchEngine(test_vector_store)

    yield engine

    # Teardown: clear singleton to avoid cross-session pollution
    ss_module.SparseSearchEngine._instance = None


@pytest_asyncio.fixture(scope="session", autouse=True)
async def populated_test_collection(test_vector_store, test_sparse_engine):
    """
    Populate the test collection AND rebuild the BM25 corpus atomically.

    This fixture is the single source of truth for test data setup. It:
      1. Runs the ingestion pipeline to upsert 5 anime into Qdrant
      2. Waits for Qdrant indexing to settle
      3. Calls ensure_initialized(force=True) on test_sparse_engine

    Because test_sparse_engine is a parameter, pytest guarantees it exists
    before this fixture runs. Because this is autouse=True with scope=session,
    it runs before any test in the session.

    The strict ordering (upsert → sleep → corpus build) ensures BM25 chunk IDs
    always match the live Qdrant collection IDs, so RRF fusion produces valid
    sparse_rank values on overlapping results.

    Returns:
        set[str] of anime titles actually ingested (consumed by ingested_titles fixture)
    """
    logger.info("\n🚀 Populating test collection with 5 anime...")

    result = await run_minimal_pipeline(
        max_anime=5,
        vector_store=test_vector_store,
        save_to_disk=False,
    )

    logger.info(f"✅ Pipeline complete: {result['uploaded_to_qdrant']} chunks uploaded")

    # Wait for Qdrant indexing before building BM25 corpus
    await asyncio.sleep(2)

    # Build BM25 corpus NOW — data is confirmed in Qdrant
    await test_sparse_engine.ensure_initialized(force=True)

    stats = test_sparse_engine.get_stats()
    logger.info(f"✅ BM25 corpus ready: {stats['corpus_size']} chunks")

    # Return ingested titles for downstream fixtures
    # "anime_titles" key added to pipeline.py summary (see that fix)
    ingested_titles: set[str] = set(result.get("anime_titles", []))
    logger.info(f"✅ Ingested titles: {ingested_titles}")

    return ingested_titles


@pytest_asyncio.fixture(scope="session")
async def ingested_titles(populated_test_collection) -> set[str]:
    """
    The set of anime titles actually ingested during this test session.

    Tests should assert against this instead of hardcoding anime names,
    since run_minimal_pipeline fetches whatever Jikan returns as top-5.

    Usage:
        async def test_foo(test_sparse_engine, ingested_titles):
            results = test_sparse_engine.search("magic", top_k=3)
            result_titles = {r["payload"].get("anime_title", "") for r in results}
            assert result_titles & ingested_titles
    """
    return populated_test_collection


@pytest_asyncio.fixture(scope="session")
async def test_hybrid_engine(test_vector_store, test_sparse_engine):
    """Hybrid search engine wired to the test collection."""
    engine = HybridSearchEngine(
        vector_store=test_vector_store,
        sparse_engine=test_sparse_engine,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
    )
    yield engine


@pytest_asyncio.fixture(scope="session")
async def test_embedder():
    """Embedding generator for encoding test queries."""
    generator = EmbeddingGenerator()
    yield generator
