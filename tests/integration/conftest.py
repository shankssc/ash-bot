"""
Integration test configuration with isolated Qdrant test collection.
Ensures test data never pollutes production collection.
"""

import hashlib
import os

from typing import Any

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import asyncio

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest
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

TEST_COLLECTION_NAME = f"anime_knowledge_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

logger = get_logger(__name__)


@pytest.fixture(scope="session", autouse=True)
def preload_embedding_model():
    """
    Load SentenceTransformer synchronously before any async fixture.

    MUST be @pytest.fixture — running this inside a coroutine deadlocks on
    Windows because SentenceTransformer.__init__ makes blocking HTTP/filesystem
    calls that hang inside the ProactorEventLoop.
    """
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=settings.EMBEDDING_DEVICE)


@pytest.fixture(scope="session")
def test_embedder(preload_embedding_model):
    """
    Single EmbeddingGenerator instance reused across all tests.

    MUST be @pytest.fixture (not pytest_asyncio) — EmbeddingGenerator.__init__
    is synchronous and must not run inside an event loop coroutine on Windows.
    """
    return EmbeddingGenerator()


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

    temp_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        timeout=10,
        prefer_grpc=False,
    )
    try:
        collections = await asyncio.wait_for(
            asyncio.to_thread(temp_client.get_collections), timeout=15.0
        )
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

    store = QdrantVectorStore(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        collection_name=TEST_COLLECTION_NAME,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    await store._initialize()

    logger.info(f"\n✅ Created isolated test collection: {TEST_COLLECTION_NAME}")
    yield store

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
    """Session-scoped sparse search engine (corpus NOT initialized here)."""
    import app.retrieval.sparse_search as ss_module

    ss_module.SparseSearchEngine._instance = None
    engine = SparseSearchEngine(test_vector_store)
    yield engine
    ss_module.SparseSearchEngine._instance = None


@pytest_asyncio.fixture(scope="session", autouse=True)
async def populated_test_collection(
    request, test_vector_store, test_sparse_engine, preload_embedding_model
):
    """
    Populate the test collection AND rebuild the BM25 corpus atomically.

    Depends on preload_embedding_model so the model is cached before the
    pipeline creates its own EmbeddingGenerator internally.
    """
    if request.node.get_closest_marker("cache_only"):
        yield set()
        return

    logger.info("\n🚀 Populating test collection with 5 anime...")

    result = await run_minimal_pipeline(
        max_anime=5,
        vector_store=test_vector_store,
        save_to_disk=False,
    )

    logger.info(f"✅ Pipeline complete: {result['uploaded_to_qdrant']} chunks uploaded")

    await asyncio.sleep(2)
    await test_sparse_engine.ensure_initialized(force=True)

    stats = test_sparse_engine.get_stats()
    logger.info(f"✅ BM25 corpus ready: {stats['corpus_size']} chunks")

    ingested_titles: set[str] = set(result.get("anime_titles", []))
    logger.info(f"✅ Ingested titles: {ingested_titles}")

    yield ingested_titles


@pytest_asyncio.fixture(scope="session")
async def ingested_titles(populated_test_collection) -> set[str]:
    """The set of anime titles actually ingested during this test session."""
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


@pytest.fixture
def mock_embedder():
    """
    Mock embedder matching EmbeddingGenerator interface exactly.

    Returns:
        - generate_single: tuple[np.ndarray, dict[str, Any]]
        - generate: tuple[np.ndarray, list[dict[str, Any]]]

    Embeddings are deterministic (SHA256-based) and pre-normalized,
    matching the real model's normalize_embeddings=True behavior.
    """

    class MockEmbedder:
        def __init__(self, dim: int = 384):
            self.dim = dim
            self.model_name = "mock/sentence-transformer"

        def generate_single(self, text: str) -> tuple[np.ndarray, dict[str, Any]]:
            if not text.strip():
                raise ValueError("Cannot generate embedding for empty text")

            # Deterministic embedding from SHA256 hash
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            embedding = np.array(
                [(hash_bytes[i % 32] - 128) / 128.0 for i in range(self.dim)],
                dtype=np.float32,
            )
            # Pre-normalize to match real model's normalize_embeddings=True
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm

            metadata = {
                "text_length": len(text),
                "model": self.model_name,
                "dimension": self.dim,
            }
            return embedding, metadata

        def generate(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
            if not texts:
                return np.array([]), []

            embeddings = []
            metadata_list = []
            for text in texts:
                emb, meta = self.generate_single(text)
                embeddings.append(emb)
                metadata_list.append(meta)

            return np.stack(embeddings), metadata_list

    return MockEmbedder()
