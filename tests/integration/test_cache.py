"""
Integration tests for semantic cache manager.

Uses fakeredis (in-memory Redis) instead of aioredis to avoid the
Windows ProactorEventLoop / pytest-asyncio 0.23.8 incompatibility where
aioredis connections established in session fixtures cannot be reused
across tests because each test runs on a different event loop.

fakeredis has no event loop dependency — it's pure Python in-memory,
so it works correctly regardless of which loop is running.
"""

import asyncio
import sys
import time

import fakeredis
import pytest

from app.services.cache_manager import SemanticCacheManager

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.cache_only,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="fakeredis.aioredis SCAN command deadlocks on Windows — runs in CI (Linux)",
    ),
]


def make_manager(
    fake_redis,
    embedder,
    ttl: int = 60,
    namespace: str = "cache_test",
) -> SemanticCacheManager:
    """
    Create a SemanticCacheManager backed by fakeredis.

    Bypasses _initialize() entirely by injecting the fake Redis client
    and pre-loaded embedder directly. This means:
    - No network calls to Upstash
    - No event loop binding issues
    - No EmbeddingGenerator.__init__ deadlock (model already loaded)
    - Tests run in milliseconds
    """
    manager = SemanticCacheManager(
        redis_url="redis://localhost:6379",  # ignored — we inject fake_redis
        redis_token="fake-token",  # noqa: S106
        similarity_threshold=0.95,
        ttl_seconds=ttl,
        namespace=namespace,
    )
    # Inject fake Redis — bypasses _initialize() entirely
    manager._redis = fake_redis
    manager._redis_available = True
    manager._is_initialized = True
    # Inject pre-loaded embedder — bypasses EmbeddingGenerator() init
    manager._embedder = embedder
    return manager


class AsyncFakeRedisWrapper:
    """
    Wrap sync fakeredis with async methods for cache_manager compatibility.
    Avoids fakeredis.aioredis SCAN deadlock on CI.
    """

    def __init__(self, sync_redis):
        self._redis = sync_redis

    async def scan(self, cursor=0, match=None, count=100):
        # Sync fakeredis returns (cursor_int, keys_list)
        result = self._redis.scan(cursor=cursor, match=match, count=count)
        return result  # Already compatible

    async def get(self, key):
        return self._redis.get(key)

    async def set(self, key, value, ex=None):
        return self._redis.set(key, value, ex=ex)

    async def delete(self, key):
        return self._redis.delete(key)

    async def ping(self):
        return self._redis.ping()

    async def close(self):
        pass  # Sync redis has no async close


@pytest.fixture
def fake_redis():
    """Fresh sync in-memory Redis per test — wrapped for async compatibility."""
    sync_redis = fakeredis.FakeRedis(decode_responses=True)
    return AsyncFakeRedisWrapper(sync_redis)


async def test_cache_set_and_get(fake_redis, mock_embedder):
    """Test basic cache set/get with semantic matching."""
    manager = make_manager(fake_redis, mock_embedder)

    query = "Who directed Cowboy Bebop?"
    answer = "Shinichirō Watanabe directed Cowboy Bebop."
    sources = [{"anime_id": 1, "title": "Cowboy Bebop", "chunk_id": "test_chunk_1"}]

    result = await manager.set(query, answer, sources)
    assert result is True, "Cache set should return True"

    cached = await manager.get(query)
    assert cached is not None, "Exact query should hit cache"
    assert cached["answer"] == answer
    assert cached["similarity"] >= 0.99


async def test_semantic_cache_hit(fake_redis, mock_embedder):
    """Test cache hit on semantically similar queries (not exact match)."""
    manager = make_manager(fake_redis, mock_embedder)

    original_query = "Who directed the anime Cowboy Bebop?"
    answer = "Shinichirō Watanabe"
    sources = [{"anime_id": 1, "title": "Cowboy Bebop"}]

    await manager.set(original_query, answer, sources)

    similar_query = "Who was the director of Cowboy Bebop?"
    cached = await manager.get(similar_query)

    assert cached is not None, "Semantically similar query should hit cache"
    assert cached["answer"] == answer
    assert cached["similarity"] >= 0.90


async def test_cache_miss_on_dissimilar_query(fake_redis, mock_embedder):
    """Test cache miss on semantically dissimilar queries."""
    manager = make_manager(fake_redis, mock_embedder)

    await manager.set(
        "Who directed Cowboy Bebop?",
        "Shinichirō Watanabe",
        [{"anime_id": 1, "title": "Cowboy Bebop"}],
    )

    cached = await manager.get("What is the capital of France?")
    assert cached is None, "Dissimilar query should miss cache"


async def test_cache_ttl_expiration(mock_embedder):
    """Test cache entry expires after TTL."""
    # Fresh fakeredis per test — no shared state issues
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    manager = make_manager(fake_redis, mock_embedder, ttl=1, namespace="cache_test_ttl")

    query = "Test TTL query for expiration"
    await manager.set(query, "Test answer", [{"anime_id": 1, "title": "Test"}])

    cached = await manager.get(query)
    assert cached is not None, "Should hit cache immediately after set"

    await asyncio.sleep(1.5)

    cached = await manager.get(query)
    assert cached is None, "Should miss cache after TTL expiry"


async def test_cache_graceful_degradation():
    """Test cache manager degrades gracefully when Redis is unavailable."""
    bad_manager = SemanticCacheManager(
        redis_url="redis://invalid-url:6379",
        redis_token="invalid-token",  # noqa: S106
        similarity_threshold=0.95,
        ttl_seconds=10,
        namespace="cache_test_bad",
    )

    result = await bad_manager.get("test query")
    assert result is None

    result = await bad_manager.set("test query", "test answer", [])
    assert result is False

    health = await bad_manager.health_check()
    assert health["status"] in ["degraded", "error"]


async def test_cache_circuit_breaker():
    """Test circuit breaker prevents repeated Redis connection attempts."""
    bad_manager = SemanticCacheManager(
        redis_url="redis://invalid-url:6379",
        redis_token="invalid-token",  # noqa: S106
        similarity_threshold=0.95,
        ttl_seconds=10,
        namespace="cache_test_circuit",
    )

    await bad_manager.get("test")
    assert bad_manager._redis_available is False

    start_time = time.time()
    for _ in range(5):
        await bad_manager.get("test")
    duration = time.time() - start_time

    assert duration < 0.1, "Circuit breaker should prevent slow Redis timeouts"
