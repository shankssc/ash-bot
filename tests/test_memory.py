"""Memory usage tests using precise incremental measurement."""

from tests.utils.memory import get_memory_usage, measure_memory_delta


def test_config_incremental_memory():
    """Verify config initialization adds <5MB of memory from OUR code."""
    from app.core.config import Settings

    # Warm-up to avoid cold-start skew
    _ = Settings()

    # Measure ONLY memory allocated by Settings() initialization
    kb_used, settings = measure_memory_delta(lambda: Settings())

    assert settings.APP_NAME == "ash-bot"
    # Realistic threshold: Config should NOT load datasets/models at init
    assert kb_used < 5120, (
        f"Config initialization allocated {kb_used}KB from our code "
        f"(expected <5120KB). Did you accidentally load data in __init__?"
    )


def test_config_process_memory_ceiling():
    """Verify total process memory stays under 60MB (catches dependency bloat)."""
    from app.core.config import Settings

    # Warm-up
    _ = Settings()

    # Measure total process memory (includes Python/pydantic baseline)
    _, peak_kb = get_memory_usage()

    # Realistic ceiling: 60MB accounts for Pydantic v2 + dotenv + CI noise
    assert peak_kb < 61440, (
        f"Total process memory {peak_kb}KB exceeds 60MB ceiling. "
        f"This likely indicates accidental large dependency or data loading."
    )
