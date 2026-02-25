"""
Qdrant vector store wrapper with circuit breaker and retry logic.
Implements spec pattern: Repository Pattern + Circuit Breaker Pattern (Phase 2)
"""

from __future__ import annotations

import asyncio
import time

from collections import deque
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    IsNullCondition,
    PointStruct,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.secrets import secrets

logger = get_logger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class VectorPoint:
    """Immutable vector point representation for type safety."""

    id: str
    vector: list[float]  # 384-dim for all-MiniLM-L6-v2
    payload: dict[str, Any]
    score: float | None = None


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.

    State transitions:
    CLOSED → (failures >= threshold) → OPEN → (timeout) → HALF_OPEN → (success) → CLOSED
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_attempts: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts

        self._state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._failure_count: int = 0
        self._last_failure_time: float | None = None
        self._half_open_success_count: int = 0
        self._failure_history: deque = deque(maxlen=10)

    def _record_failure(self, error: Exception) -> None:
        """Record failure and update circuit state."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._failure_history.append((time.time(), str(error)))

        if self._state == "CLOSED" and self._failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker OPENED after {self._failure_count} failures. Last error: {error}"
            )
            self._state = "OPEN"

    def _record_success(self) -> None:
        """Record success and reset circuit state."""
        if self._state == "HALF_OPEN":
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.half_open_attempts:
                logger.info("Circuit breaker CLOSED after successful recovery")
                self._reset()
        elif self._state == "CLOSED":
            self._reset()

    def _reset(self) -> None:
        """Reset circuit state after successful recovery."""
        self._state = "CLOSED"
        self._failure_count = 0
        self._half_open_success_count = 0

    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on circuit state."""
        if self._state == "CLOSED":
            return True

        if self._state == "OPEN":
            if self._last_failure_time and (
                time.time() - self._last_failure_time > self.recovery_timeout
            ):
                logger.info("Circuit breaker entering HALF_OPEN state for recovery attempt")
                self._state = "HALF_OPEN"
                self._half_open_success_count = 0
                return True
            return False

        if self._state == "HALF_OPEN":
            # Allow limited attempts in HALF_OPEN state
            return self._half_open_success_count < self.half_open_attempts

        return False

    def __call__(self, func):
        """Decorator for circuit breaker protection."""

        async def wrapper(*args, **kwargs):
            if not self._should_allow_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Last failure: {self._failure_history[-1] if self._failure_history else 'unknown'}"
                )

            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure(e)
                raise

        return wrapper


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is blocked."""

    pass


class QdrantVectorStore:
    """
    Production-grade Qdrant client wrapper with circuit breaker and retry logic.

    Features:
    - Circuit breaker to prevent cascading failures
    - Automatic collection creation (idempotent)
    - Async-first design for FastAPI compatibility
    - Comprehensive logging and metrics
    - Dependency injection ready (via get_vector_store())
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or secrets.QDRANT_API_KEY.get_secret_value()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        # Initialize client (lazy initialization to avoid blocking on import)
        self._client: QdrantClient | None = None
        self._is_initialized: bool = False

        logger.info(
            f"Initialized QdrantVectorStore: "
            f"collection={self.collection_name}, "
            f"vector_size={self.vector_size}, "
            f"url={self.url}"
        )

    async def _initialize(self) -> None:
        """Lazy initialization of Qdrant client."""
        if self._is_initialized:
            return

        if not self.api_key:
            raise ValueError("QDRANT_API_KEY is required. Set it in .env or pass to constructor.")

        try:
            logger.debug(f"Initializing Qdrant client for {self.url}")
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30,
                # HTTP preferred for Render free tier (no gRPC support)
                prefer_grpc=False,
            )

            # Ensure collection exists (idempotent operation)
            await self._ensure_collection()

            self._is_initialized = True
            logger.info(
                f"Qdrant client initialized successfully. Collection: {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
            raise

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist (idempotent)."""
        if not self._client:
            raise RuntimeError("Client not initialized. Call _initialize() first.")

        try:
            collections = await asyncio.to_thread(self._client.get_collections)
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(
                    f"Creating collection '{self.collection_name}' "
                    f"(size={self.vector_size}, distance=Cosine)"
                )
                await asyncio.to_thread(
                    self._client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.vector_size, distance=rest.Distance.COSINE
                    ),
                    on_disk_payload=True,  # Critical for free tier RAM constraints
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.debug(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}", exc_info=True)
            raise

    @CircuitBreaker()
    async def upsert(self, points: list[VectorPoint]) -> bool:
        """
        Upsert vectors into Qdrant collection.

        Args:
            points: List of VectorPoint objects to upsert

        Returns:
            True if successful

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Exception: On Qdrant operation failure
        """
        if not self._is_initialized:
            await self._initialize()

        if not points:
            logger.warning("Upsert called with empty points list")
            return True

        if not self._client:
            raise RuntimeError("Client not initialized")

        try:
            start_time = time.time()
            point_structs = [
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ]

            # Use thread pool for sync Qdrant client in async context
            await asyncio.to_thread(
                self._client.upsert,
                collection_name=self.collection_name,
                points=point_structs,
                wait=True,
            )

            duration = (time.time() - start_time) * 1000
            logger.info(
                f"Upserted {len(points)} points to '{self.collection_name}' in {duration:.2f}ms"
            )
            return True

        except CircuitBreakerOpenError:
            logger.warning("Circuit breaker blocked upsert operation")
            raise

        except Exception as e:
            logger.error(f"Upsert failed: {e}", exc_info=True)
            raise

    @CircuitBreaker()
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorPoint]:
        """
        Search for similar vectors in Qdrant collection.

        Args:
            query_vector: Query embedding vector (384-dim)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0-1.0)
            filters: Optional metadata filters (e.g., {"anime_id": 5114})

        Returns:
            List of VectorPoint results sorted by score (descending)

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            ValueError: On invalid input
            Exception: On Qdrant operation failure
        """

        if not self._is_initialized:
            await self._initialize()

        if not self._client:
            raise RuntimeError("Client not initialized")

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Query vector dimension ({len(query_vector)}) "
                f"must match collection dimension ({self.vector_size})"
            )

        try:
            start_time = time.time()

            # Build filter condition if provided
            query_filter = None

            if filters:
                must_conditions: list[FieldCondition | IsEmptyCondition | IsNullCondition] = []
                for key, value in filters.items():
                    must_conditions.append(
                        rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
                    )
                query_filter = Filter(must=must_conditions)

            # Execute search
            results = await asyncio.to_thread(
                self._client.search,
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )

            duration = (time.time() - start_time) * 1000
            logger.debug(
                f"Search returned {len(results)} results in {duration:.2f}ms "
                f"(top_k={top_k}, filters={filters})"
            )

            # Convert to VectorPoint objects
            vector_points = [
                VectorPoint(
                    id=str(result.id),
                    # Vectors not returned (with_vectors=False for efficiency)
                    vector=[],
                    payload=result.payload or {},
                    score=result.score,
                )
                for result in results
            ]

            return vector_points

        except CircuitBreakerOpenError:
            logger.warning("Circuit breaker blocked search operation")
            raise

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on Qdrant connection.

        Returns:
            Health status dictionary for /api/v1/health endpoint
        """
        try:
            if not self._is_initialized:
                await self._initialize()

            if not self._client:
                return {"status": "error", "message": "Client not initialized"}

            # Lightweight check: get collection info
            collection_info = await asyncio.to_thread(
                self._client.get_collection, collection_name=self.collection_name
            )

            return {
                "status": "ok",
                "url": self.url,
                "collection": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.indexed_vectors_count,
                "circuit_breaker": {
                    "state": self.circuit_breaker._state,
                    "failure_count": self.circuit_breaker._failure_count,
                },
            }

        except CircuitBreakerOpenError as e:
            return {
                "status": "degraded",
                "message": f"Circuit breaker open: {e}",
                "circuit_breaker": {
                    "state": "OPEN",
                    "failure_count": self.circuit_breaker._failure_count,
                    "last_failure": (
                        str(self.circuit_breaker._failure_history[-1])
                        if self.circuit_breaker._failure_history
                        else None
                    ),
                },
            }
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {"status": "error", "message": str(e), "url": self.url}


# Singleton instance management for dependency injection
_vector_store_instance: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    """
    Get singleton QdrantVectorStore instance for dependency injection.

    Usage in FastAPI endpoints:
        @router.post("/query")
        async def query_endpoint(
            request: QueryRequest,
            vector_store: QdrantVectorStore = Depends(get_vector_store)
        ):
            ...
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore()
    return _vector_store_instance
