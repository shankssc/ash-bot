"""Test health check endpoints and dependency validators."""

from unittest.mock import patch

import pytest

from app.api.v1.health import (
    _check_groq,
    _check_qdrant,
    _check_redis,
    health_check,
)
from app.core.config import Settings

# ===== HELPER FUNCTION TESTS =====


def test_check_qdrant_degraded_url():
    settings = Settings(QDRANT_URL="https://your-uuid.example.com")
    result = _check_qdrant(settings)
    assert result["status"] == "degraded"
    assert "QDRANT_URL not configured" in result["message"]


def test_check_qdrant_degraded_key():
    settings = Settings(QDRANT_URL="https://valid.example.com")
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.QDRANT_API_KEY.get_secret_value.return_value = ""
        result = _check_qdrant(settings)
        assert result["status"] == "degraded"
        assert "QDRANT_API_KEY missing" in result["message"]


def test_check_qdrant_ok():
    settings = Settings(QDRANT_URL="https://prod.example.com")
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.QDRANT_API_KEY.get_secret_value.return_value = "valid_key"
        result = _check_qdrant(settings)
        assert result["status"] == "ok"
        assert result["url"] == settings.QDRANT_URL


def test_check_qdrant_exception():
    settings = Settings(QDRANT_URL="https://valid.example.com")
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.QDRANT_API_KEY.get_secret_value.side_effect = Exception("Network error")
        result = _check_qdrant(settings)
        assert result["status"] == "error"
        assert "Network error" in result["message"]


def test_check_groq_degraded():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.GROQ_API_KEY.get_secret_value.return_value = ""
        result = _check_groq(mock_secrets)
        assert result["status"] == "degraded"


def test_check_groq_ok():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.GROQ_API_KEY.get_secret_value.return_value = "valid_key"
        result = _check_groq(mock_secrets)
        assert result["status"] == "ok"
        assert result["model"] == "llama-3.1-70b-versatile"


def test_check_groq_exception():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.GROQ_API_KEY.get_secret_value.side_effect = ValueError("Invalid key")
        result = _check_groq(mock_secrets)
        assert result["status"] == "error"


def test_check_redis_degraded_missing():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.REDIS_URL.get_secret_value.return_value = ""
        result = _check_redis(mock_secrets)
        assert result["status"] == "degraded"


def test_check_redis_degraded_invalid_url():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.REDIS_URL.get_secret_value.return_value = "redis://localhost"
        result = _check_redis(mock_secrets)
        assert result["status"] == "degraded"
        assert "REDIS_URL not configured" in result["message"]


def test_check_redis_ok():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.REDIS_URL.get_secret_value.return_value = "https://upstash.example.com"
        result = _check_redis(mock_secrets)
        assert result["status"] == "ok"
        assert result["url"] == "[REDACTED]"


def test_check_redis_exception():
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.REDIS_URL.get_secret_value.side_effect = ConnectionError("Timeout")
        result = _check_redis(mock_secrets)
        assert result["status"] == "error"


# ===== ENDPOINT TESTS =====


@pytest.mark.asyncio
async def test_health_check_all_ok():
    settings = Settings(
        APP_VERSION="1.0.0-test",
        ENVIRONMENT="test",
        QDRANT_URL="https://valid.example.com",
    )
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.QDRANT_API_KEY.get_secret_value.return_value = "q_key"
        mock_secrets.GROQ_API_KEY.get_secret_value.return_value = "g_key"
        mock_secrets.REDIS_URL.get_secret_value.return_value = "https://upstash.example.com"

        result = await health_check(settings)

        assert result["status"] == "ok"
        assert result["checks"]["app"]["status"] == "ok"
        assert result["checks"]["qdrant"]["status"] == "ok"
        assert result["checks"]["groq"]["status"] == "ok"
        assert result["checks"]["redis"]["status"] == "ok"
        # Verifies line using APP_VERSION
        assert result["timestamp"] == "1.0.0-test"


@pytest.mark.asyncio
async def test_health_check_degraded():
    settings = Settings(
        APP_VERSION="test",
        ENVIRONMENT="test",
        QDRANT_URL="https://your-uuid.example.com",  # Triggers degraded
    )
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.QDRANT_API_KEY.get_secret_value.return_value = "key"
        mock_secrets.GROQ_API_KEY.get_secret_value.return_value = "key"
        mock_secrets.REDIS_URL.get_secret_value.return_value = "https://upstash.example.com"

        result = await health_check(settings)

        assert result["status"] == "degraded"
        assert result["checks"]["qdrant"]["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_check_error():
    settings = Settings(
        APP_VERSION="test",
        ENVIRONMENT="test",
        QDRANT_URL="https://valid.example.com",
    )
    with patch("app.api.v1.health.secrets") as mock_secrets:
        # Force exception in Groq check
        mock_secrets.GROQ_API_KEY.get_secret_value.side_effect = Exception("API down")
        mock_secrets.QDRANT_API_KEY.get_secret_value.return_value = "key"
        mock_secrets.REDIS_URL.get_secret_value.return_value = "https://upstash.example.com"

        result = await health_check(settings)

        assert result["status"] == "error"
        assert result["checks"]["groq"]["status"] == "error"


@pytest.mark.asyncio
async def test_health_check_logs_warning_on_degraded(caplog):
    """Verifies the warning log line (critical for coverage)"""
    settings = Settings(
        APP_VERSION="test",
        ENVIRONMENT="test",
        QDRANT_URL="https://your-uuid.example.com",
    )
    with patch("app.api.v1.health.secrets") as mock_secrets:
        mock_secrets.QDRANT_API_KEY.get_secret_value.return_value = "key"
        mock_secrets.GROQ_API_KEY.get_secret_value.return_value = "key"
        mock_secrets.REDIS_URL.get_secret_value.return_value = "valid"

        await health_check(settings)

    assert "Health check degraded" in caplog.text
    assert "degraded" in caplog.text.lower()
