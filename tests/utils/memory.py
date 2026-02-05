"""Cross-platform memory profiling helper."""

import sys
import tracemalloc

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def track_memory() -> Generator[None, None, None]:
    """
    Cross-platform memory tracking context manager.

    Usage:
        with track_memory():
            # ... code to profile ...
    """
    # Linux-only: try memray first (if installed)
    if sys.platform == "linux":
        try:
            import os
            import tempfile

            import memray

            # Create secure temp file
            fd, tmp_path = tempfile.mkstemp(prefix="memray-", suffix=".bin")
            os.close(fd)  # memray will open it

            try:
                with memray.Tracker(tmp_path):
                    yield
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return
        except ImportError:
            pass

    # Cross-platform fallback: stdlib tracemalloc
    tracemalloc.start()
    start = tracemalloc.take_snapshot()
    try:
        yield
    finally:
        current = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Show top 10 memory diffs
        stats = current.compare_to(start, "lineno")
        print("\n[Memory usage top 10]:")
        for stat in stats[:10]:
            print(stat)


def get_memory_usage() -> tuple[int, int]:
    """Get current and peak memory usage in KB."""
    try:
        if sys.platform == "linux" or sys.platform == "darwin":
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                usage //= 1024
            return (usage, usage)
    except (ImportError, AttributeError, OSError):
        # Expected failures on some platforms - safe to ignore
        pass

    # Fallback: psutil (optional dependency)
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        rss_kb = process.memory_info().rss // 1024
        return (rss_kb, rss_kb)
    except ImportError:
        # Last resort: rough estimate
        import gc

        gc.collect()
        return (0, 0)  # Can't measure accurately without psutil
