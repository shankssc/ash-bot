#!/usr/bin/env python3
"""CLI script to test BM25 sparse search against Qdrant collection."""

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
from app.api.deps import get_production_vector_store
from app.core.config import settings
from app.core.logging import get_logger, init_logging
from app.retrieval.sparse_search import SparseSearchEngine

# Initialize logging BEFORE any modules that might log
init_logging(settings)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test BM25 sparse search against Qdrant anime knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python scripts/test_sparse_search.py "magic elf adventure"

  # Search with custom top_k
  python scripts/test_sparse_search.py "space western" --top-k 5

  # Debug mode (shows tokenization)
  python scripts/test_sparse_search.py "isekai fantasy" --debug
        """,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query string (e.g., 'magic elf adventure')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows tokenization details)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info(f"Testing sparse search for query: '{args.query}'")
    print(f"\n🔍 Searching for: '{args.query}'")
    print(f"   Top {args.top_k} results\n")

    try:
        # Initialize vector store and sparse engine
        logger.info("Initializing Qdrant vector store...")
        vector_store = get_production_vector_store()

        # Verify connection
        health = await vector_store.health_check()
        if health["status"] != "ok":
            print(
                f"⚠️  Qdrant health check failed: {health.get('message', 'unknown')}",
                file=sys.stderr,
            )
            return 1

        logger.info("Initializing BM25 sparse search engine...")
        sparse_engine = SparseSearchEngine(vector_store)
        await sparse_engine.ensure_initialized()

        # Perform search
        logger.info(f"Executing BM25 search for query: '{args.query}'")
        results = sparse_engine.search(args.query, top_k=args.top_k)

        # Display results
        if not results:
            print("❌ No results found for this query\n")
            print("💡 Suggestions:")
            print("   - Try broader terms (e.g., 'magic' instead of 'ancient magic ritual')")
            print("   - Check if collection has relevant content (run ingestion pipeline)")
            print("   - Use debug mode to see tokenization: --debug")
            return 1

        print(f"✅ Found {len(results)} results:\n")
        for idx, result in enumerate(results, 1):
            title = result["payload"].get("anime_title", "Unknown")
            score = result["score"]
            snippet = (
                result["text_snippet"][:100] + "..."
                if len(result["text_snippet"]) > 100
                else result["text_snippet"]
            )

            print(f"{idx}. {title} (score: {score:.4f})")
            print(f'   "{snippet}"')
            print()

        # Show stats
        stats = sparse_engine.get_stats()
        print("=" * 60)
        print(f"📊 Corpus Stats: {stats['corpus_size']} chunks indexed")
        print(f"   Collection: {settings.QDRANT_COLLECTION_NAME}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        print(f"\n❌ SEARCH FAILED: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
