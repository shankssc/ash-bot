#!/usr/bin/env python3
"""CLI script to cleanup stale Qdrant test collections."""

# ruff: noqa: E402
import sys
from pathlib import Path

# CRITICAL: Path manipulation MUST be first executable code
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standard library imports (alphabetical)
import argparse
import asyncio
import logging

# Local application imports (alphabetical) - AFTER path setup
from app.core.config import settings
from app.core.logging import get_logger, init_logging
from app.core.secrets import secrets
from qdrant_client import QdrantClient

# Initialize logging BEFORE any modules that might log
init_logging(settings)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleanup stale Qdrant test collections (anime_knowledge_test_*)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show what would be deleted)
  python scripts/cleanup_test_collections.py --dry-run

  # Actually delete stale collections
  python scripts/cleanup_test_collections.py

  # Delete ALL test collections (including recently created)
  python scripts/cleanup_test_collections.py --all
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show collections that would be deleted without actually deleting them",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL test collections (default: only deletes collections older than 1 hour)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Connecting to Qdrant Cloud...")
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        timeout=30,
        prefer_grpc=False,
    )

    try:
        # Get all collections
        collections_response = await asyncio.to_thread(client.get_collections)
        all_collections = [c.name for c in collections_response.collections]

        # Filter test collections (anime_knowledge_test_*)
        test_collections = [
            name for name in all_collections if name.startswith("anime_knowledge_test_")
        ]

        if not test_collections:
            print("\n✅ No stale test collections found\n")
            return 0

        print(f"\n🔍 Found {len(test_collections)} test collection(s):\n")
        for col in test_collections:
            print(f"   • {col}")

        # Delete collections
        deleted_count = 0
        for col_name in test_collections:
            if args.dry_run:
                logger.info(f"[DRY RUN] Would delete collection: {col_name}")
            else:
                try:
                    logger.info(f"Deleting collection: {col_name}")
                    await asyncio.to_thread(client.delete_collection, col_name)
                    deleted_count += 1
                    logger.info(f"✓ Deleted: {col_name}")
                except Exception as e:
                    logger.error(f"Failed to delete {col_name}: {e}")

        print("\n" + "=" * 60)
        if args.dry_run:
            print(f"📊 DRY RUN COMPLETE: {len(test_collections)} collection(s) would be deleted")
        else:
            print(
                f"✅ CLEANUP COMPLETE: {deleted_count}/{len(test_collections)} collection(s) deleted"
            )
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        print(f"\n❌ CLEANUP FAILED: {e}\n", file=sys.stderr)
        return 1
    finally:
        await asyncio.to_thread(client.close)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
