from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secret values - NEVER exposed in logs/reprs."""

    # ===== VECTOR DATABASE =====
    QDRANT_API_KEY: SecretStr = SecretStr("")

    # ===== CACHE =====
    REDIS_URL: SecretStr = SecretStr("")

    # ===== LLM APIs =====
    GROQ_API_KEY: SecretStr = SecretStr("")
    TOGETHER_API_KEY: SecretStr = SecretStr("")

    # ===== MONITORING =====
    BETTERSTACK_SOURCE_TOKEN: SecretStr = SecretStr("")

    # ===== LEGACY (keep for now) =====
    AWS_ACCESS_KEY_ID: SecretStr = SecretStr("")
    AWS_SECRET_ACCESS_KEY: SecretStr = SecretStr("")
    COGNITO_CLIENT_SECRET: SecretStr = SecretStr("")
    STRIPE_SECRET_KEY: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __repr__(self) -> str:
        # 🔒 NEVER show actual secret values
        return "Secrets(<redacted>)"

    def __str__(self) -> str:
        return self.__repr__()

    def get_secret_value(self, field: str) -> str:
        """Safely retrieve secret value (for internal use only)."""
        secret = getattr(self, field, None)
        if isinstance(secret, SecretStr):
            return secret.get_secret_value()
        return ""


secrets = Secrets()
