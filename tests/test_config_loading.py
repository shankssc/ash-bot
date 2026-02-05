"""Test configuration loading."""

from pydantic_settings import SettingsConfigDict

from app.core.config import Settings


def test_env_file_overrides_defaults(monkeypatch, tmp_path):
    """Verify .env file values override defaults (and env vars don't interfere)."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    env_content = """OLLAMA_BASE_URL=http://custom-llm.internal:11434
LLM_MODEL=custom-model
ENVIRONMENT=production
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=env_file,
            env_file_encoding="utf-8",
            env_ignore_empty=True,
        )

    test_settings = TestSettings()

    assert test_settings.ENVIRONMENT == "production"
    assert test_settings.OLLAMA_BASE_URL == "http://custom-llm.internal:11434"
    assert test_settings.LLM_MODEL == "custom-model"


def test_production_workers_override(monkeypatch, tmp_path):
    """Verify WORKERS auto-set to 2 in production."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)

    env_content = """ENVIRONMENT=production
WORKERS=1
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    test_settings = TestSettings()

    assert test_settings.WORKERS == 2


def test_development_workers_not_overridden(monkeypatch, tmp_path):
    """Verify WORKERS not auto-set in development."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)

    env_content = """ENVIRONMENT=development
WORKERS=3
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    test_settings = TestSettings()

    assert test_settings.WORKERS == 3


def test_paths_created(monkeypatch, tmp_path):
    """Verify critical paths are created on init."""
    monkeypatch.delenv("VECTOR_DB_PATH", raising=False)
    monkeypatch.delenv("LOGS_PATH", raising=False)

    vector_db_path = tmp_path / "test_chroma"
    logs_path = tmp_path / "test_logs"

    # Ensure paths don't exist initially
    assert not vector_db_path.exists()
    assert not logs_path.exists()

    env_content = f"""VECTOR_DB_PATH={vector_db_path}
LOGS_PATH={logs_path}
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    _ = TestSettings()

    # Paths should be created at CONFIGURED locations
    assert vector_db_path.exists()
    assert logs_path.exists()
    assert logs_path.is_dir()
