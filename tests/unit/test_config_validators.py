"""Additional tests for config.py validators and edge cases."""

import os

import pytest

from app.core.config import Settings


class TestConfigValidators:
    """Test config field validators and model validators."""

    def test_environment_normalization_lowercase(self):
        """Environment value normalized to lowercase before validation."""
        os.environ["ENVIRONMENT"] = "PRODUCTION"
        settings = Settings()
        assert settings.ENVIRONMENT == "production"
        del os.environ["ENVIRONMENT"]

    def test_log_level_normalization_uppercase(self):
        """Log level normalized to uppercase before validation."""
        os.environ["LOG_LEVEL"] = "debug"
        settings = Settings()
        assert settings.LOG_LEVEL == "DEBUG"
        del os.environ["LOG_LEVEL"]

    def test_cors_origins_list_property(self):
        """cors_origins_list splits comma-separated string."""
        settings = Settings(CORS_ORIGINS="http://a.com, http://b.com,http://c.com")
        assert settings.cors_origins_list == ["http://a.com", "http://b.com", "http://c.com"]

    def test_is_production_helper(self):
        """is_production() returns correct boolean."""
        assert Settings(ENVIRONMENT="production").is_production() is True
        assert Settings(ENVIRONMENT="development").is_production() is False

    def test_repr_includes_key_fields(self):
        """__repr__ includes environment, host, model, collection."""
        settings = Settings()
        repr_str = repr(settings)
        assert "env=" in repr_str
        assert "host=" in repr_str
        assert "model=" in repr_str
        assert "qdrant_collection=" in repr_str

    def test_production_requires_configured_qdrant(self):
        """Production environment validates QDRANT_URL is configured."""
        with pytest.raises(ValueError, match="QDRANT_URL must be configured"):
            Settings(
                ENVIRONMENT="production",
                QDRANT_URL="https://your-uuid.us-east-1-0.aws.cloud.qdrant.io:6333",
            )

    def test_production_auto_sets_workers(self):
        """Production environment auto-sets WORKERS to 2 if 1."""
        settings = Settings(ENVIRONMENT="production", WORKERS=1, QDRANT_URL="http://configured")
        assert settings.WORKERS == 2
