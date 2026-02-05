from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secret values - NEVER exposed in logs/reprs."""

    # AWS credentials (example - use IAM roles in prod!)
    AWS_ACCESS_KEY_ID: SecretStr = SecretStr("")
    AWS_SECRET_ACCESS_KEY: SecretStr = SecretStr("")

    # Cognito
    COGNITO_CLIENT_SECRET: SecretStr = SecretStr("")

    # Stripe
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
