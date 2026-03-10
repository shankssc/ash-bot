from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Public application configuration (NON-SECRET values only)."""

    # ===== APP CONFIGURATION =====
    APP_NAME: str = "ash-bot"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = Field(
        default="development"
    )

    # ===== SERVER CONFIGURATION =====
    HOST: str = Field(
        default="0.0.0.0",  # nosec B104: Intentional binding for containerized deployments
        description="Server bind address. Use 127.0.0.1 for local-only access.",
    )
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ===== CORS CONFIGURATION =====
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8001"

    # ===== AI/LLM CONFIGURATION (Legacy - Ollama) =====
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str | None = "phi3:mini"

    # ===== LOGS CONFIGURATION =====
    LOGS_PATH: Path = Path("logs")

    # ===== FEATURE FLAGS =====
    PREWARM_ON_STARTUP: bool = True
    SERVE_DEMO_FRONTEND: bool = True
    WORKERS: int = 1

    # ===== VECTOR DATABASE (Qdrant - AniRAG) =====
    QDRANT_URL: str = "https://your-uuid.us-east-1-0.aws.cloud.qdrant.io:6333"
    QDRANT_COLLECTION_NAME: str = "anime_knowledge"
    QDRANT_TEST_COLLECTION_NAME: str = "anime_knowledge_test"
    QDRANT_VECTOR_SIZE: int = 384

    # ===== CACHE =====
    REDIS_URL: str = "https://rapid-cardinal-50147.upstash.io"
    CACHE_TTL_SECONDS: int = 604800  # 7 days
    SEMANTIC_CACHE_THRESHOLD: float = 0.95

    # ===== EMBEDDING MODEL =====
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # ===== LLM API (Groq - AniRAG) =====
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.7

    # ===== RATE LIMITING =====
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD_SECONDS: int = 3600  # 1 hour

    # ===== DATA SOURCES =====
    JIKAN_API_URL: str = "https://api.jikan.moe/v4"
    JIKAN_RATE_LIMIT_PER_SECOND: float = 3.0
    ANILIST_API_URL: str = "https://graphql.anilist.co"

    # ===== DEPRECATED (ChromaDB) =====
    # Kept for migration path but NOT used in AniRAG pipeline
    VECTOR_DB_PATH: Path | None = None

    # ===== FIELD VALIDATORS =====
    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def _normalize_environment(cls, v: Any) -> Any:
        """Normalize environment to lowercase BEFORE Literal validation."""
        return v.lower() if isinstance(v, str) else v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: Any) -> Any:
        """Normalize log level to uppercase BEFORE Literal validation."""
        return v.upper() if isinstance(v, str) else v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _validate_and_initialize(self) -> Settings:
        # Skip strict validation for test environment (unit tests)
        if self.ENVIRONMENT == "test":
            self.LOGS_PATH.mkdir(exist_ok=True)
            return self

        # Production validation
        if self.ENVIRONMENT == "production":
            if "your-uuid" in self.QDRANT_URL:
                raise ValueError("QDRANT_URL must be configured for production")
            if self.WORKERS == 1:
                self.WORKERS = 2

        # Create critical directories
        self.LOGS_PATH.mkdir(exist_ok=True)
        if self.VECTOR_DB_PATH:
            self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"env={self.ENVIRONMENT}, "
            f"host={self.HOST}:{self.PORT}, "
            f"model={self.GROQ_MODEL}, "
            f"qdrant_collection={self.QDRANT_COLLECTION_NAME}"
            f")"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for dependency injection."""
    return Settings()


settings = get_settings()  # Global instance for non-DI usage
