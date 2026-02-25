"""
Dependency injection utilities for FastAPI endpoints.
Follows spec pattern: Dependency Injection (Phase 3)
"""

from collections.abc import Generator

from app.core.config import get_settings
from app.core.secrets import secrets
from app.retrieval.vector_store import QdrantVectorStore, get_vector_store


def get_settings_dependency():
    """Get settings instance for dependency injection."""
    return get_settings()


def get_vector_store_dependency() -> Generator[QdrantVectorStore, None, None]:
    """
    Get QdrantVectorStore instance for dependency injection.

    Usage in FastAPI endpoints:
        @router.post("/query")
        async def query_endpoint(
            request: QueryRequest,
            vector_store: QdrantVectorStore = Depends(get_vector_store_dependency)
        ):
            ...
    """
    # Use singleton instance from vector_store module
    vector_store = get_vector_store()
    yield vector_store


def get_production_vector_store() -> QdrantVectorStore:
    """
    Production-ready vector store with actual credentials.
    Used for ingestion scripts and background tasks.
    """
    return QdrantVectorStore(
        url=get_settings().QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        collection_name=get_settings().QDRANT_COLLECTION_NAME,
        vector_size=get_settings().QDRANT_VECTOR_SIZE,
    )
