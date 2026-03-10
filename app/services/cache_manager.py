"""
Semantic cache manager using Upstash Redis for low-latency response caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from collections.abc import Awaitable
from typing import Any, cast

import numpy as np

from redis import asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.secrets import secrets
from app.ingestion.embedder import EmbeddingGenerator

logger = get_logger(__name__)
settings = get_settings()


class SemanticCacheManager:
    """
    Production-grade semantic cache manager with Redis backend.

    Design decisions (per spec):
    - Semantic matching: Cosine similarity ≥0.95 for cache hits (not exact string match)
    - Embedding-based keys: Query embeddings hashed to Redis keys
    - TTL management: 7-day default (configurable via CACHE_TTL_SECONDS)
    - Circuit breaker: Redis failures don't block query processing
    - Async-first: Compatible with FastAPI async event loop
    - Namespace isolation: `cache:` prefix prevents key collisions

    Cache structure in Redis:
    {
        "query": "original query text",
        "embedding": [0.1, 0.2, ...],  # 384-dim
        "answer": "generated answer text",
        "sources": [{"anime_id": 5114, "chunk_id": "...", ...}],
        "metadata": {"intent": "factual", "latency_ms": 342, ...},
        "timestamp": 1709654400.123
    }
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_token: str | None = None,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 604800,  # 7 days
        namespace: str = "cache",
    ):
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_token = redis_token or secrets.REDIS_TOKEN.get_secret_value()
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        self._redis: aioredis.Redis | None = None
        self._embedder: EmbeddingGenerator | None = None
        self._is_initialized: bool = False
        self._initialization_lock = asyncio.Lock()

        # Circuit breaker state
        self._redis_available: bool = True
        self._last_redis_failure: float = 0.0

        logger.info(
            f"Initialized SemanticCacheManager: "
            f"threshold={similarity_threshold}, ttl={ttl_seconds}s, namespace={namespace}"
        )

    async def _initialize(self) -> None:
        """Lazy initialization of Redis client and embedder."""
        if self._is_initialized:
            return

        async with self._initialization_lock:
            if self._is_initialized:
                return

            # Initialize Redis client
            try:
                logger.debug(f"Connecting to Redis at {self.redis_url}")
                self._redis = aioredis.from_url(
                    self.redis_url,
                    password=self.redis_token,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                )
                # Test connection
                await cast("Awaitable[bool]", self._redis.ping())
                logger.info("✓ Redis connection established")
                self._redis_available = True
            except Exception as e:
                logger.warning(f"Redis initialization failed (cache disabled): {e}")
                self._redis_available = False
                self._redis = None
                # Don't raise - cache is optional for query processing

            # Initialize embedder (required for semantic matching)
            try:
                logger.debug("Initializing embedding generator for cache...")
                self._embedder = EmbeddingGenerator()
                logger.info("✓ Embedding generator initialized for semantic cache")
            except Exception as e:
                logger.error(f"Embedder initialization failed: {e}", exc_info=True)
                raise

            self._is_initialized = True

    async def _get_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query (cached internally by EmbeddingGenerator)."""
        if not self._embedder:
            raise RuntimeError("Embedder not initialized. Call _initialize() first.")

        embedding, _ = await asyncio.to_thread(self._embedder.generate_single, query)
        return embedding

    def _embedding_to_key(self, embedding: np.ndarray) -> str:
        """
        Convert embedding to Redis key using SHA256 hash.

        Why hash instead of raw embedding?
        - Redis keys must be strings (not binary blobs)
        - Hashing provides fixed-length keys for efficient lookup
        - Collision probability negligible for 384-dim vectors (SHA256)
        """
        # Normalize embedding first (cosine similarity requires unit vectors)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Convert to bytes and hash
        embedding_bytes = embedding.astype(np.float32).tobytes()
        key_hash = hashlib.sha256(embedding_bytes).hexdigest()
        return f"{self.namespace}:{key_hash}"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    async def get(self, query: str) -> dict[str, Any] | None:
        """
        Retrieve cached result for semantically similar query.

        Args:
            query: User query string

        Returns:
            Cached result dict if similarity ≥ threshold, else None

        Cache hit criteria:
        - Cosine similarity ≥ similarity_threshold (default 0.95)
        - TTL not expired (default 7 days)
        """
        if not self._redis_available:
            logger.debug("Redis unavailable - skipping cache lookup")
            return None

        await self._initialize()

        if self._redis is None:
            raise RuntimeError("Redis client not initialized despite _redis_available=True")

        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)

            # Scan Redis for semantically similar keys (naive approach for <10K entries)
            # Production enhancement (Phase 5): Use RedisVL or dedicated vector index
            similar_results = []

            # Get all cache keys (efficient for <10K entries; free tier limit)
            pattern = f"{self.namespace}:*"
            cursor = 0
            while cursor != 0:
                cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                if not keys:
                    continue

                # Fetch values for batch of keys
                pipe = self._redis.pipeline()
                for key in keys:
                    pipe.get(key)
                values = await pipe.execute()

                # Check similarity for each cached result
                for key, value_json in zip(keys, values, strict=True):
                    if not value_json:
                        continue

                    try:
                        cached = json.loads(value_json)
                        cached_embedding = np.array(cached["embedding"])
                        similarity = self._cosine_similarity(query_embedding, cached_embedding)

                        if similarity >= self.similarity_threshold:
                            cached["similarity"] = similarity
                            similar_results.append(cached)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Invalid cache entry {key}: {e}")
                        # Clean up invalid entry
                        await self._redis.delete(key)

            # Return best match (highest similarity)
            if similar_results:
                best: dict[str, Any] = max(similar_results, key=lambda x: x["similarity"])
                logger.debug(f"Cache hit (similarity={best['similarity']:.4f}): '{query[:50]}...'")
                return best
            else:
                logger.debug(f"Cache miss for query: '{query[:50]}...'")
                return None

        except Exception as e:
            # Circuit breaker: mark Redis unavailable on failure
            self._redis_available = False
            self._last_redis_failure = time.time()
            logger.warning(f"Cache lookup failed (disabling cache temporarily): {e}")
            return None

    async def set(
        self,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Cache query result with semantic key.

        Args:
            query: Original user query
            answer: Generated answer text
            sources: List of source citations
            metadata: Optional metadata (intent, latency, etc.)

        Returns:
            True if cached successfully, False otherwise
        """
        if not self._redis_available:
            logger.debug("Redis unavailable - skipping cache store")
            return False

        await self._initialize()

        if self._redis is None:
            raise RuntimeError("Redis client not initialized despite _redis_available=True")

        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)

            # Build cache entry
            cache_entry = {
                "query": query,
                "embedding": query_embedding.tolist(),
                "answer": answer,
                "sources": sources,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }

            # Generate Redis key from embedding
            key = self._embedding_to_key(query_embedding)

            # Store in Redis with TTL
            await self._redis.set(
                key,
                json.dumps(cache_entry, ensure_ascii=False),
                ex=self.ttl_seconds,
            )

            logger.debug(
                f"Cached result (key={key[-8:]}): '{query[:50]}...' (TTL={self.ttl_seconds}s)"
            )
            return True

        except Exception as e:
            # Circuit breaker: mark Redis unavailable on failure
            self._redis_available = False
            self._last_redis_failure = time.time()
            logger.warning(f"Cache store failed (disabling cache temporarily): {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on Redis connection.

        Returns:
            Health status dictionary for /api/v1/health endpoint
        """
        try:
            if not self._is_initialized:
                await self._initialize()

            if not self._redis_available:
                return {
                    "status": "degraded",
                    "message": "Redis unavailable (circuit breaker open)",
                    "last_failure": self._last_redis_failure,
                }

            if not self._redis:
                return {"status": "error", "message": "Redis client not initialized"}

            # Lightweight ping check
            await cast("Awaitable[bool]", self._redis.ping())

            # Get cache stats
            pattern = f"{self.namespace}:*"
            cursor = 0
            key_count = 0
            while cursor != 0:
                cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                key_count += len(keys)

            return {
                "status": "ok",
                "url": self.redis_url,
                "namespace": self.namespace,
                "key_count": key_count,
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
                "circuit_breaker": {
                    "available": self._redis_available,
                    "last_failure": self._last_redis_failure,
                },
            }

        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return {
                "status": "degraded",
                "message": str(e),
                "url": self.redis_url,
            }


# Singleton instance management for dependency injection
_cache_manager_instance: SemanticCacheManager | None = None


def get_cache_manager() -> SemanticCacheManager:
    """
    Get singleton SemanticCacheManager instance for dependency injection.

    Usage in FastAPI endpoints:
        @router.post("/query")
        async def query_endpoint(
            request: QueryRequest,
            cache_manager: SemanticCacheManager = Depends(get_cache_manager)
        ):
            cached = await cache_manager.get(request.query)
            if cached:
                return cached["answer"]
            # ... proceed to retrieval/generation ...
    """
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = SemanticCacheManager()
    return _cache_manager_instance
