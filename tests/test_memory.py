"""Memory usage tests."""

from tests.utils.memory import get_memory_usage, track_memory


def test_config_memory_footprint():
    """Verify config loading has minimal memory footprint."""
    # Warm-up
    from app.core.config import Settings

    _ = Settings()

    # Measure
    with track_memory():
        settings = Settings()
        assert settings.APP_NAME == "ash-bot"

    # Verify under 50MB
    _, peak = get_memory_usage()
    assert peak < 61440, f"Memory peak {peak}KB exceeds 60MB safety threshold"
