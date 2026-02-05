"""Test logger initialization."""

import logging
import os

from app.core.logger import setup_logger


def test_logger_creation():
    """Verify logger initializes correctly."""
    logger = setup_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO

    # Verify log file created
    assert os.path.exists("logs/ash-bot.log")


def test_logger_does_not_propagate():
    """Verify logger doesn't propagate to root logger."""
    logger = setup_logger("isolated_logger")
    assert logger.propagate is False
