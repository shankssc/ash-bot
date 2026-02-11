"""
Centralized logging configuration with safe early initialization.

Design: Logger can be used BEFORE settings are fully loaded (for config validation errors),
then reconfigured with proper settings after initialization.
"""

from __future__ import annotations

import logging
import sys

from app.core.config import Settings

# Early logger for config validation errors (before settings load)
_early_logger = logging.getLogger("ash-bot.early")
_early_logger.setLevel(logging.INFO)
if not _early_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _early_logger.addHandler(handler)


class LoggerConfig:
    """Centralized logger configuration using app settings."""

    # Forward ref handled via __future__ annotations
    def __init__(self, settings: Settings | None = None):
        from app.core.config import Settings  # Late import to break circular dependency

        self.settings = settings or Settings()
        self.logs_path = self.settings.LOGS_PATH
        self.logs_path.mkdir(exist_ok=True)

    def get_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.settings.LOG_LEVEL, logging.INFO))
        logger.propagate = False

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(self.logs_path / "ash-bot.log")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

        return logger


# Singleton instance
_logger_config: LoggerConfig | None = None


def get_logger(name: str) -> logging.Logger:
    """Get configured logger. Falls back to early logger if not initialized."""
    global _logger_config
    if _logger_config is None:
        _early_logger.warning(
            f"Logger requested before initialization: {name}. Using early logger."
        )
        return _early_logger
    return _logger_config.get_logger(name)


def init_logging(settings: Settings) -> None:
    """Initialize logging system with app settings."""
    global _logger_config
    _logger_config = LoggerConfig(settings)
    get_logger("ash-bot").info(f"Logging initialized. Level: {settings.LOG_LEVEL}")
