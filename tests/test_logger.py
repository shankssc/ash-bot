"""Test logger initialization and behavior."""

import logging
import os
import shutil
import sys
import tempfile

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.logging import _early_logger, get_logger, init_logging


@pytest.fixture(autouse=True)
def reset_logger_state():
    """Reset ONLY test-created loggers and singleton state. PRESERVE module-level _early_logger."""
    # BEFORE TEST: Clean ONLY loggers created during testing
    _cleanup_test_loggers()

    # Reset singleton WITHOUT importing the variable (fixes F811 redefinition)
    import app.core.logging

    app.core.logging._logger_config = None

    yield

    # AFTER TEST: Aggressive cleanup of test artifacts to release Windows file locks
    _cleanup_test_loggers()
    app.core.logging._logger_config = None


def _cleanup_test_loggers():
    """Clean ONLY loggers created during tests (NOT module-level _early_logger)."""
    # Clean test loggers and main app logger
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        if logger_name.startswith("test.") or logger_name == "ash-bot":
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                except Exception:  # noqa: S110 (best-effort cleanup in tests)
                    pass
            logger.handlers.clear()
            logger.propagate = True

    # Explicitly clean the root "ash-bot" logger if it exists
    main_logger = logging.getLogger("ash-bot")
    for handler in main_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:  # noqa: S110 (best-effort cleanup in tests)
            pass
    main_logger.handlers.clear()
    main_logger.propagate = True

    # CRITICAL: DO NOT TOUCH _early_logger - it's configured once at module import


@pytest.fixture
def temp_log_dir():
    """Provide isolated temp directory that survives handler cleanup."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    # Cleanup happens in reset_logger_state AFTER handlers are closed
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:  # noqa: S110 (best-effort cleanup in tests)
        pass  # Windows may still hold locks; ignore for test stability


def test_early_logger_exists_and_configured():
    """Verify early logger is pre-configured before settings load (module-level constant)."""
    # IMPORTANT: _early_logger MUST retain its handler since module import already happened
    assert _early_logger.name == "ash-bot.early"
    assert _early_logger.level == logging.INFO
    assert len(_early_logger.handlers) == 1, (
        "Early logger lost its handler! "
        "This means cleanup touched module-level state. "
        "Ensure _cleanup_test_loggers() NEVER clears _early_logger.handlers"
    )
    assert isinstance(_early_logger.handlers[0], logging.StreamHandler)
    assert _early_logger.handlers[0].stream == sys.stderr


def test_get_logger_before_init_uses_early_logger(caplog):
    """Verify fallback to early logger when not initialized."""
    logger = get_logger("test.before_init")

    # Should be the early logger instance
    assert logger is _early_logger

    # Should emit warning about uninitialized state
    logger.warning("test message")
    assert "before initialization" in caplog.text
    assert "test.before_init" in caplog.text


def test_init_logging_creates_log_directory(temp_log_dir):
    """Verify log directory is created during initialization."""
    settings = Settings(LOGS_PATH=temp_log_dir, LOG_LEVEL="DEBUG")
    init_logging(settings)

    assert temp_log_dir.exists()
    assert (temp_log_dir / "ash-bot.log").exists()


def test_logger_after_init_has_correct_handlers_and_level(temp_log_dir):
    """Verify properly configured logger after initialization."""
    settings = Settings(LOGS_PATH=temp_log_dir, LOG_LEVEL="WARNING")
    init_logging(settings)

    logger = get_logger("test.after_init")

    # Correct log level
    assert logger.level == logging.WARNING

    # No propagation
    assert logger.propagate is False

    # Two handlers: console (stdout) + file
    assert len(logger.handlers) == 2
    handlers = {type(h): h for h in logger.handlers}

    assert logging.StreamHandler in handlers
    assert logging.FileHandler in handlers

    # Verify console uses stdout (not stderr)
    assert handlers[logging.StreamHandler].stream == sys.stdout

    # Verify file handler points to correct location
    assert str(temp_log_dir / "ash-bot.log") in handlers[logging.FileHandler].baseFilename


@pytest.mark.parametrize(
    "level_str,level_val",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
    ],
)
def test_logger_respects_log_level_from_settings(temp_log_dir, level_str, level_val):
    """Verify log level is correctly applied from settings."""
    settings = Settings(LOGS_PATH=temp_log_dir, LOG_LEVEL=level_str)
    init_logging(settings)

    logger = get_logger(f"test.level.{level_str}")
    assert logger.level == level_val


def test_log_message_appears_in_file(temp_log_dir):
    """Verify log messages are actually written to file."""
    settings = Settings(LOGS_PATH=temp_log_dir, LOG_LEVEL="INFO")
    init_logging(settings)

    logger = get_logger("test.file_output")
    test_msg = "UNIQUE_TEST_MESSAGE_12345"
    logger.info(test_msg)

    # Force flush BEFORE cleanup
    for handler in logger.handlers:
        handler.flush()

    # Read log file while handlers are still open (safe on Windows)
    log_file = temp_log_dir / "ash-bot.log"
    with open(log_file) as f:
        content = f.read()

    assert test_msg in content
    assert "test.file_output" in content
    assert "INFO" in content


def test_logger_config_accepts_none_settings():
    """
    Verify LoggerConfig can instantiate without explicit settings.
    Uses isolated temp dir to avoid polluting cwd.
    """
    # Create isolated temp dir for default logs
    temp_dir = tempfile.mkdtemp()
    default_logs_path = Path(temp_dir) / "logs"

    try:
        # Temporarily override the default LOGS_PATH via environment variable
        original_env = os.environ.get("LOGS_PATH")
        os.environ["LOGS_PATH"] = str(default_logs_path)

        # Force reload of settings module to pick up new env var
        from importlib import reload

        import app.core.config

        reload(app.core.config)

        # Now create LoggerConfig without explicit settings
        from app.core.logging import LoggerConfig

        config = LoggerConfig()

        assert config.settings is not None
        assert config.logs_path == default_logs_path
        assert config.logs_path.exists()

    finally:
        # Cleanup
        if original_env is not None:
            os.environ["LOGS_PATH"] = original_env
        else:
            os.environ.pop("LOGS_PATH", None)

        # Reload settings again to restore original state
        reload(app.core.config)

        # Cleanup temp dir
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:  # noqa: S110 (best-effort cleanup in tests)
            pass


def test_get_logger_different_names_create_distinct_loggers(temp_log_dir):
    """Verify different logger names produce separate Logger instances."""
    settings = Settings(LOGS_PATH=temp_log_dir, LOG_LEVEL="INFO")
    init_logging(settings)

    logger1 = get_logger("service.a")
    logger2 = get_logger("service.b")

    assert logger1 is not logger2
    assert logger1.name == "service.a"
    assert logger2.name == "service.b"
