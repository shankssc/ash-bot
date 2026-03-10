"""
Integration tests for semantic cache manager.
Requires Upstash Redis configuration in .env (REDIS_URL, REDIS_TOKEN).
"""

import asyncio
import time

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from app.core.config import get_settings
from app.core.secrets import secrets
from app.services.cache_manager import SemanticCacheManager

# Module-level marker for ALL async tests
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(scope="session")
async def cache_manager() -> AsyncGenerator[SemanticCacheManager, None]:
    """Fixture: Semantic cache manager instance with isolated namespace."""
    settings = get_settings()

    # Use test namespace to avoid polluting production cache
    manager = SemanticCacheManager(
        redis_url=settings.REDIS_URL,
        redis_token=secrets.REDIS_TOKEN.get_secret_value(),
        similarity_threshold=0.95,
        ttl_seconds=10,  # Short TTL for tests
        namespace="cache_test",
    )
    await manager._initialize()

    yield manager

    # Cleanup: Delete all test cache keys
    if manager._redis:
        pattern = f"{manager.namespace}:*"
        cursor = "0"
        deleted = 0
        while cursor != 0:
            cursor, keys = await manager._redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await manager._redis.delete(*keys)
                deleted += len(keys)
        # print(f"\n✅ Cleaned up {deleted} test cache keys")


async def test_cache_set_and_get(cache_manager: SemanticCacheManager):
    """Test basic cache set/get with semantic matching."""
    query = "Who directed Cowboy Bebop?"
    answer = "Shinichirō Watanabe directed Cowboy Bebop."
    sources = [{"anime_id": 1, "title": "Cowboy Bebop", "chunk_id": "test_chunk_1"}]

    # Set cache
    result = await cache_manager.set(query, answer, sources)
    assert result is True

    # Get exact match (should hit cache)
    cached = await cache_manager.get(query)
    assert cached is not None
    assert cached["answer"] == answer
    assert cached["similarity"] >= 0.99  # Near-perfect match for identical query


async def test_semantic_cache_hit(cache_manager: SemanticCacheManager):
    """Test cache hit on semantically similar queries (not exact match)."""
    # Original query
    original_query = "Who directed the anime Cowboy Bebop?"
    answer = "Shinichirō Watanabe"
    sources = [{"anime_id": 1, "title": "Cowboy Bebop"}]

    # Cache original query
    await cache_manager.set(original_query, answer, sources)

    # Semantically similar query (should hit cache)
    similar_query = "Who was the director of Cowboy Bebop?"
    cached = await cache_manager.get(similar_query)

    assert cached is not None
    assert cached["answer"] == answer
    assert cached["similarity"] >= 0.90  # High similarity for paraphrased query


async def test_cache_miss_on_dissimilar_query(cache_manager: SemanticCacheManager):
    """Test cache miss on semantically dissimilar queries."""
    # Cache anime query
    await cache_manager.set(
        "Who directed Cowboy Bebop?",
        "Shinichirō Watanabe",
        [{"anime_id": 1, "title": "Cowboy Bebop"}],
    )

    # Dissimilar query (should miss cache)
    dissimilar_query = "What is the capital of France?"
    cached = await cache_manager.get(dissimilar_query)

    assert cached is None


async def test_cache_ttl_expiration(cache_manager: SemanticCacheManager):
    """Test cache entry expires after TTL."""
    query = "Test TTL query"
    answer = "Test answer"
    sources = [{"anime_id": 1, "title": "Test"}]

    # Create manager with short TTL
    manager = SemanticCacheManager(
        redis_url=cache_manager.redis_url,
        redis_token=cache_manager.redis_token,
        similarity_threshold=0.95,
        ttl_seconds=1,  # 1 second TTL
        namespace="cache_test_ttl",
    )
    await manager._initialize()

    await manager.set(query, answer, sources)

    # Should hit cache immediately
    cached = await manager.get(query)
    assert cached is not None

    # Wait for TTL to expire
    await asyncio.sleep(1.5)

    # Should miss cache after TTL
    cached = await manager.get(query)
    assert cached is None


async def test_cache_graceful_degradation(cache_manager: SemanticCacheManager):
    """Test cache manager degrades gracefully when Redis is unavailable."""
    # Simulate Redis failure by using invalid URL
    bad_manager = SemanticCacheManager(
        redis_url="redis://invalid-url:6379",
        redis_token="invalid-token",  # noqa: S106 - fake token for testing error handling
        similarity_threshold=0.95,
        ttl_seconds=10,
        namespace="cache_test_bad",
    )

    # Should not raise exception - cache should be disabled gracefully
    result = await bad_manager.get("test query")
    assert result is None  # Cache miss due to Redis failure

    result = await bad_manager.set("test query", "test answer", [])
    assert result is False  # Set failed due to Redis failure

    # Health check should report degraded status
    health = await bad_manager.health_check()
    assert health["status"] in ["degraded", "error"]


async def test_cache_circuit_breaker(cache_manager: SemanticCacheManager):
    """Test circuit breaker prevents repeated Redis connection attempts."""
    # Force Redis failure
    bad_manager = SemanticCacheManager(
        redis_url="redis://invalid-url:6379",
        redis_token="invalid-token",  # noqa: S106 - fake token for testing error handling
        similarity_threshold=0.95,
        ttl_seconds=10,
        namespace="cache_test_circuit",
    )

    # First failure should mark Redis unavailable
    await bad_manager.get("test")
    assert bad_manager._redis_available is False

    # Subsequent calls should skip Redis entirely (no connection attempts)
    start_time = time.time()
    for _ in range(5):
        await bad_manager.get("test")
    duration = time.time() - start_time

    # Should be very fast (no network timeouts)
    assert duration < 0.1, "Circuit breaker should prevent slow Redis timeouts"
