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
    1. Creates isolated test collection at session start
    2. Yields vector store for all tests in session
    3. Deletes test collection after session (cleanup)
    """
    settings = get_settings()

    # Clean up: Delete collection if exists from previous run
    temp_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        timeout=30,
        prefer_grpc=False,
    )
    try:
        collections = await asyncio.to_thread(temp_client.get_collections)
        existing_names = [c.name for c in collections.collections]
        if TEST_COLLECTION_NAME in existing_names:
            logger.info(f"\n🗑️  Deleting stale test collection: {TEST_COLLECTION_NAME}")
            await asyncio.to_thread(
                temp_client.delete_collection, collection_name=TEST_COLLECTION_NAME
            )
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

    # Cleanup: Delete test collection after session
    try:
        await asyncio.to_thread(
            store._client.delete_collection, collection_name=TEST_COLLECTION_NAME
        )
        logger.info(f"\n✅ Deleted test collection: {TEST_COLLECTION_NAME}")
    except Exception as e:
        logger.info(f"\n⚠️  Cleanup warning (non-fatal): {e}")


@pytest_asyncio.fixture(scope="session", autouse=True)
async def populate_test_collection(test_vector_store):
    """
    Populate test collection with controlled dataset before tests.

    Why 5 anime?
    - Fast enough for CI/CD (<15 seconds)
    - Large enough to validate search logic
    - Small enough to debug failures
    """
    logger.info("\n🚀 Populating test collection with 5 anime...")

    result = await run_minimal_pipeline(
        max_anime=5,
        vector_store=test_vector_store,
        save_to_disk=False,  # Skip disk output for tests
    )

    logger.info(f"✅ Populated test collection: {result['uploaded_to_qdrant']} chunks")

    # Wait for Qdrant indexing to complete (critical for search tests)
    await asyncio.sleep(2)


@pytest_asyncio.fixture(scope="session")
async def test_sparse_engine(
    test_vector_store: QdrantVectorStore,
) -> AsyncGenerator[SparseSearchEngine, None]:
    """
    Sparse search engine for test collection.

    CRITICAL: Direct instantiation (NOT get_instance()) to avoid singleton pollution
    from production collection.
    """
    # DO NOT USE get_instance() - creates singleton tied to first vector store
    engine = SparseSearchEngine(test_vector_store)  # Direct instantiation

    # Force rebuild corpus from test collection
    await engine.ensure_initialized(force=True)

    yield engine

    # Optional: Clear singleton cache to prevent cross-test pollution
    import app.retrieval.sparse_search as ss_module

    ss_module.SparseSearchEngine._instance = None


@pytest_asyncio.fixture(scope="session")
async def test_hybrid_engine(test_vector_store, test_sparse_engine):
    """Hybrid search engine for test collection."""
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
    """Embedding generator for test queries."""
    generator = EmbeddingGenerator()
    yield generator
