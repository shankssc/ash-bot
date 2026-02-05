from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Public application configuration (NON-SECRET values only)."""

    # App config
    APP_NAME: str = "ash-bot"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")

    # Server config
    HOST: str = Field(
        default="0.0.0.0",  # nosec B104: Intentional binding for containerized deployments
        description="Server bind address. Use 127.0.0.1 for local-only access.",
    )
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS config
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8001"

    # AI/LLM config (public endpoints)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "phi3:mini"

    # Vector DB config
    VECTOR_DB_PATH: Path = Path("./data/chroma_db")

    # Logs config (NEW - configurable path)
    LOGS_PATH: Path = Path("logs")

    # Feature flags
    PREWARM_ON_STARTUP: bool = True
    SERVE_DEMO_FRONTEND: bool = True
    WORKERS: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _post_init(self) -> Settings:
        if self.ENVIRONMENT.lower() == "production" and self.WORKERS == 1:
            self.WORKERS = 2

        # Create critical directories using CONFIGURABLE paths
        self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        # ✅ Uses LOGS_PATH instead of hardcoded "logs"
        self.LOGS_PATH.mkdir(exist_ok=True)

        # Normalize environment
        self.ENVIRONMENT = self.ENVIRONMENT.lower()
        self.LOG_LEVEL = self.LOG_LEVEL.upper()
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
            f"model={self.LLM_MODEL}, "
            f"db={self.VECTOR_DB_PATH.name}"
            f")"
        )


settings = Settings()
