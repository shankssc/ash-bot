"""Test configuration loading with Pydantic V2 field validators and model validation."""

import pytest

from pydantic_settings import SettingsConfigDict

from app.core.config import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove config-related env vars before each test to ensure isolation."""
    for var in [
        "ENVIRONMENT",
        "LOG_LEVEL",
        "WORKERS",
        "LOGS_PATH",
        "VECTOR_DB_PATH",
        "OLLAMA_BASE_URL",
        "LLM_MODEL",
        "QDRANT_URL",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield


def test_env_file_overrides_defaults(tmp_path):
    """Verify .env file values override defaults with proper case normalization (non-production)."""
    # Use STAGING to avoid production QDRANT_URL validation requirement
    env_content = """OLLAMA_BASE_URL=http://custom-llm.internal:11434
LLM_MODEL=custom-model
ENVIRONMENT=Staging   # Mixed case to test normalization
LOG_LEVEL=warning     # Lowercase to test normalization
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file,
            env_file_encoding="utf-8",
        )

    test_settings = TestSettings()

    # Field validators normalize case BEFORE validation
    assert test_settings.ENVIRONMENT == "staging"  # Normalized to lowercase
    assert test_settings.LOG_LEVEL == "WARNING"  # Normalized to uppercase
    assert test_settings.OLLAMA_BASE_URL == "http://custom-llm.internal:11434"
    assert test_settings.LLM_MODEL == "custom-model"


def test_production_workers_auto_override(tmp_path):
    """Verify WORKERS auto-set to 2 in production WITH valid QDRANT_URL."""
    # Must provide valid QDRANT_URL to pass production validation
    env_content = """ENVIRONMENT=production
WORKERS=1
QDRANT_URL=https://my-valid-uuid.us-east-1-0.aws.cloud.qdrant.io:6333
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    test_settings = TestSettings()

    # Production validation in model_validator overrides WORKERS=1 → 2
    assert test_settings.WORKERS == 2
    assert test_settings.ENVIRONMENT == "production"


def test_development_workers_preserved(tmp_path):
    """Verify WORKERS value preserved in non-production environments."""
    env_content = """ENVIRONMENT=development
WORKERS=3
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    test_settings = TestSettings()

    # No override in development
    assert test_settings.WORKERS == 3
    assert test_settings.ENVIRONMENT == "development"


def test_paths_created_on_init(tmp_path):
    """Verify critical directories are created during Settings initialization."""
    vector_db_path = tmp_path / "test_chroma"
    logs_path = tmp_path / "test_logs"

    # Verify paths don't exist initially
    assert not vector_db_path.exists()
    assert not logs_path.exists()

    env_content = f"""VECTOR_DB_PATH={vector_db_path}
LOGS_PATH={logs_path}
ENVIRONMENT=development  # Avoid production validation
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    # Initialization triggers directory creation via model_validator
    _ = TestSettings()

    # Directories should now exist
    assert vector_db_path.exists()
    assert logs_path.exists()
    assert logs_path.is_dir()


def test_production_validation_blocks_default_qdrant(tmp_path):
    """Verify production environment REJECTS placeholder QDRANT_URL ('your-uuid')."""
    env_content = """ENVIRONMENT=production
QDRANT_URL=https://your-uuid.us-east-1-0.aws.cloud.qdrant.io:6333
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    # Should raise validation error due to placeholder UUID in production
    with pytest.raises(ValueError, match="QDRANT_URL must be configured for production"):
        TestSettings()


def test_production_validation_accepts_custom_qdrant(tmp_path):
    """Verify production ACCEPTS valid QDRANT_URL (without placeholder)."""
    env_content = """ENVIRONMENT=production
QDRANT_URL=https://my-actual-uuid.us-east-1-0.aws.cloud.qdrant.io:6333
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    settings = TestSettings()
    assert settings.ENVIRONMENT == "production"
    assert "your-uuid" not in settings.QDRANT_URL


def test_log_level_normalization_cases(tmp_path):
    """Verify LOG_LEVEL normalization handles all case variations correctly."""
    test_cases = [
        ("debug", "DEBUG"),
        ("INFO", "INFO"),  # Already correct case
        ("Warning", "WARNING"),
        ("error", "ERROR"),
    ]

    for input_val, expected in test_cases:
        env_content = f"""LOG_LEVEL={input_val}
ENVIRONMENT=development  # Avoid production validation
"""
        env_file = tmp_path / f".env_log_{input_val}"
        env_file.write_text(env_content)

        class TestSettings(Settings):
            model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

        settings = TestSettings()
        assert (
            settings.LOG_LEVEL == expected
        ), f"LOG_LEVEL normalization failed: '{input_val}' → '{settings.LOG_LEVEL}' (expected '{expected}')"


def test_environment_normalization_cases(tmp_path):
    """Verify ENVIRONMENT normalization handles all case variations correctly."""
    # Test non-production environments to avoid QDRANT_URL validation requirement
    test_cases = [
        ("DEVELOPMENT", "development"),
        ("Staging", "staging"),
        ("test", "test"),  # From your updated Literal
    ]

    for input_val, expected in test_cases:
        env_content = f"ENVIRONMENT={input_val}"
        env_file = tmp_path / f".env_env_{input_val}"
        env_file.write_text(env_content)

        class TestSettings(Settings):
            model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

        settings = TestSettings()
        assert (
            settings.ENVIRONMENT == expected
        ), f"ENVIRONMENT normalization failed: '{input_val}' → '{settings.ENVIRONMENT}' (expected '{expected}')"


def test_production_environment_requires_valid_qdrant(tmp_path):
    """
    Comprehensive test: production environment MUST have valid QDRANT_URL.
    This is a critical safety guardrail to prevent accidental production deploys
    with placeholder configuration.
    """
    # Case 1: Missing QDRANT_URL entirely → should fail
    env_content_1 = "ENVIRONMENT=production"
    env_file_1 = tmp_path / ".env_prod_1"
    env_file_1.write_text(env_content_1)

    class TestSettings1(Settings):
        model_config = SettingsConfigDict(env_file=env_file_1, env_file_encoding="utf-8")

    with pytest.raises(ValueError, match="QDRANT_URL must be configured for production"):
        TestSettings1()

    # Case 2: Placeholder URL → should fail
    env_content_2 = """ENVIRONMENT=production
QDRANT_URL=https://your-uuid.us-east-1-0.aws.cloud.qdrant.io:6333
"""
    env_file_2 = tmp_path / ".env_prod_2"
    env_file_2.write_text(env_content_2)

    class TestSettings2(Settings):
        model_config = SettingsConfigDict(env_file=env_file_2, env_file_encoding="utf-8")

    with pytest.raises(ValueError, match="QDRANT_URL must be configured for production"):
        TestSettings2()

    # Case 3: Valid URL → should succeed
    env_content_3 = """ENVIRONMENT=production
QDRANT_URL=https://my-prod-uuid.us-east-1-0.aws.cloud.qdrant.io:6333
"""
    env_file_3 = tmp_path / ".env_prod_3"
    env_file_3.write_text(env_content_3)

    class TestSettings3(Settings):
        model_config = SettingsConfigDict(env_file=env_file_3, env_file_encoding="utf-8")

    settings = TestSettings3()
    assert settings.ENVIRONMENT == "production"
    assert "your-uuid" not in settings.QDRANT_URL
