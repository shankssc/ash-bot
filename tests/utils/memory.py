"""
Cross-platform memory profiling utilities.

Two distinct tools for different purposes:
1. track_memory() → Interactive debugging (shows top allocations)
2. measure_memory_delta() → Automated tests (precise KB measurement)
"""

import gc
import sys
import tracemalloc

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any


# ======================
# TOOL 1: DEBUGGING (Human-readable)
# ======================
@contextmanager
def track_memory() -> Generator[None, None, None]:
    """
    Cross-platform memory tracking for debugging.

    Shows top 10 memory allocations during context execution.
    Usage:
        with track_memory():
            expensive_operation()
    """
    # Linux-only: try memray first (if installed)
    if sys.platform == "linux":
        try:
            import os
            import tempfile

            import memray

            fd, tmp_path = tempfile.mkstemp(prefix="memray-", suffix=".bin")
            os.close(fd)

            try:
                with memray.Tracker(tmp_path):
                    yield
                # Optional: Generate flame graph report here
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return
        except ImportError:
            pass  # Fall back to tracemalloc

    # Cross-platform fallback
    tracemalloc.start()
    start = tracemalloc.take_snapshot()
    try:
        yield
    finally:
        current = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = current.compare_to(start, "lineno")
        print("\n[Memory usage top 10]:")
        for stat in stats[:10]:
            print(stat)


# ======================
# TOOL 2: TESTING (Precise measurement)
# ======================
def measure_memory_delta(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[int, Any]:
    """
    Measure KB allocated *only by your code* during function execution.

    Filters out noise from stdlib/frameworks by tracing only 'app/' and 'tests/' paths.

    Returns:
        (kb_used: int, result: Any) → Memory delta in KB + function result

    Usage:
        kb_used, settings = measure_memory_delta(lambda: Settings())
        assert kb_used < 5120  # <5MB for config initialization
    """
    gc.collect()
    tracemalloc.start()
    start = tracemalloc.take_snapshot()

    try:
        result = func(*args, **kwargs)
    finally:
        current = tracemalloc.take_snapshot()
        tracemalloc.stop()

    # Filter to ONLY our codebase allocations
    stats = current.compare_to(start, "lineno")
    total_kb = sum(
        stat.size_diff // 1024
        for stat in stats
        if any(path in str(stat.traceback) for path in ["app/core", "app/api", "tests/"])
    )

    return total_kb, result


# ======================
# HELPER: Process-wide measurement (for legacy tests)
# ======================
def get_memory_usage() -> tuple[int, int]:
    """Get current/peak process memory in KB (cross-platform)."""
    try:
        if sys.platform in ("linux", "darwin"):
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                usage //= 1024  # macOS reports bytes
            return (usage, usage)
    except (ImportError, AttributeError, OSError):
        pass

    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        rss_kb = process.memory_info().rss // 1024
        return (rss_kb, rss_kb)
    except ImportError:
        gc.collect()
        return (0, 0)  # Can't measure accurately
