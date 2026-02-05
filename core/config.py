"""
Configuration module for ash-bot
=================================
"""
import os
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application configuration settings using os.getenv() pattern."""

    def __init__(self) -> None:
        """Initialize settings with environment variables and defaults."""

        # ============ APP CONFIG ============
        self.APP_NAME: str = os.getenv("APP_NAME", "ash-bot")
        self.APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()

        # ============ SERVER CONFIG ============
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", 8000))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # ============ CORS CONFIG ============
        cors_origins_str = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:8001")
        self.CORS_ORIGINS: List[str] = [
            origin.strip() for origin in cors_origins_str.split(",")
        ]

        # ============ AI/LLM CONFIG ============
        self.OLLAMA_BASE_URL: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434")
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "phi3:mini")

        # ============ VECTOR DB CONFIG ============
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
        self.VECTOR_DB_PATH: Path = Path(vector_db_path)

        # ============ FEATURE FLAGS ============
        prewarm_str = os.getenv("PREWARM_ON_STARTUP", "true").lower()
        self.PREWARM_ON_STARTUP: bool = prewarm_str in (
            "true", "1", "yes", "y")

        self.SERVE_DEMO_FRONTEND: bool = os.getenv(
            "SERVE_DEMO_FRONTEND", "true").lower() in ("true", "1", "yes", "y")
        
        # ============ WORKER CONFIG (for production) ============
        self.WORKERS: int = int(
            os.getenv("WORKERS", 2 if self.ENVIRONMENT == "production" else 1))

        # ============ VALIDATE CRITICAL PATHS ============
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Ensure critical directories exist."""
        # Create vector DB directory if it doesn't exist
        if not self.VECTOR_DB_PATH.exists():
            self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created vector DB directory: {self.VECTOR_DB_PATH}")

        # Create logs directory
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Settings("
            f"env={self.ENVIRONMENT}, "
            f"host={self.HOST}:{self.PORT}, "
            f"model={self.LLM_MODEL}, "
            f"db={self.VECTOR_DB_PATH.name}"
            f")"
        )
settings = Settings()
