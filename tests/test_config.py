"""Test configuration loading."""

from app.core.config import Settings


def test_config_defaults():
    """Verify default settings load correctly."""
    settings = Settings()

    assert settings.APP_NAME == "ash-bot"
    assert settings.APP_VERSION == "1.0.0"
    assert settings.ENVIRONMENT == "test"  # From conftest.py
    assert settings.PORT == 8000
    assert "localhost:3000" in settings.CORS_ORIGINS


def test_config_bool_parsing():
    """Verify boolean environment variables parse correctly."""
    import os

    # Test truthy values
    for val in ["true", "True", "1", "yes", "y"]:
        os.environ["PREWARM_ON_STARTUP"] = val
        settings = Settings()
        assert settings.PREWARM_ON_STARTUP is True

    # Test falsy values
    for val in ["false", "False", "0", "no", "n"]:
        os.environ["PREWARM_ON_STARTUP"] = val
        settings = Settings()
        assert settings.PREWARM_ON_STARTUP is False

    # Cleanup
    del os.environ["PREWARM_ON_STARTUP"]


def test_config_path_creation():
    """Verify vector DB path is created."""

    settings = Settings()
    assert settings.VECTOR_DB_PATH.exists()
    assert settings.VECTOR_DB_PATH.is_dir()
