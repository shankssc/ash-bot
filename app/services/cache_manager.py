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

        # FIX: Do not create asyncio.Lock() in __init__.
        # asyncio.Lock() captures the running event loop at creation time.
        # __init__ runs synchronously (outside any coroutine), so there may
        # be no running loop, or the loop captured here differs from the one
        # that later runs _initialize(). The lazy _lock property creates the
        # lock on first access, which always happens inside a coroutine on
        # the correct running loop.
        self._initialization_lock: asyncio.Lock | None = None

        # Circuit breaker state
        self._redis_available: bool = True
        self._last_redis_failure: float = 0.0

        logger.info(
            f"Initialized SemanticCacheManager: "
            f"threshold={similarity_threshold}, ttl={ttl_seconds}s, namespace={namespace}"
        )

    @property
    def _lock(self) -> asyncio.Lock:
        """
        Lazy lock — always created on the currently running event loop.

        Only ever accessed from inside async methods, so the lock is always
        bound to the correct loop. Avoids the 'Future attached to a different
        loop' error that occurs when Lock() is created in __init__.
        """
        if self._initialization_lock is None:
            self._initialization_lock = asyncio.Lock()
        return self._initialization_lock

    async def _initialize(self) -> None:
        """Lazy initialization of Redis client."""
        if self._is_initialized:
            return

        # FIX: use self._lock (lazy property) not self._initialization_lock (raw attr)
        async with self._lock:
            if self._is_initialized:
                return

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
                await cast("Awaitable[bool]", self._redis.ping())
                logger.info("✓ Redis connection established")
                self._redis_available = True
            except Exception as e:
                logger.warning(f"Redis initialization failed (cache disabled): {e}")
                self._redis_available = False
                self._redis = None

            self._is_initialized = True

    def _get_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for query synchronously.

        FIX: Plain def, not async. SentenceTransformer.encode() is CPU-bound
        with no I/O — there is nothing to await. Calling it synchronously
        avoids submitting it to an executor, which caused deadlocks on Windows
        when PyTorch spawned its own threads inside the executor thread.
        """
        if self._embedder is None:
            logger.debug("Lazy initializing embedding generator for cache...")
            self._embedder = EmbeddingGenerator()
            logger.info("✓ Embedding generator initialized for semantic cache")

        embedding, _ = self._embedder.generate_single(query)
        return embedding

    def _embedding_to_key(self, embedding: np.ndarray) -> str:
        """Convert embedding to Redis key — explicit loops avoid coverage deadlock."""
        emb_list = embedding.tolist()

        # Compute L2 norm with explicit loop
        norm_sq = 0.0
        for x in emb_list:
            norm_sq += x * x
        norm = norm_sq**0.5

        if norm > 1e-8:
            emb_list = [x / norm for x in emb_list]

        embedding_bytes = np.array(emb_list, dtype=np.float32).tobytes()
        key_hash = hashlib.sha256(embedding_bytes).hexdigest()
        return f"{self.namespace}:{key_hash}"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity using explicit loops — avoids coverage.py deadlock on Windows."""
        a_list: list[float] = a.tolist()
        b_list: list[float] = b.tolist()

        # FIX: Use explicit loops instead of generator expressions
        dot = 0.0
        a_sq = 0.0
        b_sq = 0.0
        for i in range(len(a_list)):
            dot += a_list[i] * b_list[i]
            a_sq += a_list[i] * a_list[i]
            b_sq += b_list[i] * b_list[i]

        a_norm = a_sq**0.5
        b_norm = b_sq**0.5

        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(dot / (a_norm * b_norm))

    async def get(self, query: str) -> dict[str, Any] | None:
        """
        Retrieve cached result for semantically similar query.

        Args:
            query: User query string

        Returns:
            Cached result dict if similarity ≥ threshold, else None
        """
        if not self._redis_available:
            logger.debug("Redis unavailable - skipping cache lookup")
            return None

        await self._initialize()

        if not self._redis_available or self._redis is None:
            logger.debug("Redis unavailable after initialization - skipping cache lookup")
            return None

        try:
            # FIX: no await — _get_embedding is a plain def
            query_embedding = self._get_embedding(query)

            similar_results = []
            pattern = f"{self.namespace}:*"

            # FIX: cursor starts as "0", loop exits when Redis returns "0" again.
            # Original code used `cursor = 0; while cursor != 0` which never
            # entered the loop body since the initial value matched the exit condition.
            cursor = "0"
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cast(int, cursor), match=pattern, count=100
                )

                if keys:
                    for key in keys:
                        value_json = await self._redis.get(key)
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
                            await self._redis.delete(key)

                if cursor == "0":
                    break

            if similar_results:
                best: dict[str, Any] = max(similar_results, key=lambda x: x["similarity"])
                logger.debug(f"Cache hit (similarity={best['similarity']:.4f}): '{query[:50]}...'")
                return best

            logger.debug(f"Cache miss for query: '{query[:50]}...'")
            return None

        except Exception as e:
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

        if not self._redis_available or self._redis is None:
            logger.debug("Redis unavailable after initialization - skipping cache store")
            return False

        try:
            # FIX: no await — _get_embedding is a plain def
            query_embedding = self._get_embedding(query)

            cache_entry = {
                "query": query,
                "embedding": query_embedding.tolist(),
                "answer": answer,
                "sources": sources,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }

            key = self._embedding_to_key(query_embedding)
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

            await cast("Awaitable[bool]", self._redis.ping())

            # FIX: same scan loop correction as get()
            pattern = f"{self.namespace}:*"
            cursor = "0"
            key_count = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cast(int, cursor), match=pattern, count=100
                )
                key_count += len(keys)
                if cursor == "0":
                    break

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
    """
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = SemanticCacheManager()
    return _cache_manager_instance
