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

    # Verify under 10MB
    _, peak = get_memory_usage()
    assert peak < 10240, f"Memory peak {peak}KB exceeds 10MB limit"
