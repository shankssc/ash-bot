"""CLI script to run minimal anime ingestion pipeline."""

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
from datetime import datetime

# Local application imports (alphabetical) - AFTER path setup
from app.api.deps import get_production_vector_store
from app.core.config import settings
from app.core.logging import get_logger, init_logging
from app.ingestion.pipeline import run_minimal_pipeline

# Initialize logging BEFORE any modules that might log
init_logging(settings)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run minimal anime ingestion pipeline (Jikan API → Chunking → Embeddings → Qdrant/Disk)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Disk-only mode (default)
  python scripts/run_minimal_pipeline.py --max-anime 10

  # Upload to Qdrant Cloud (requires .env configured)
  python scripts/run_minimal_pipeline.py --max-anime 10 --to-qdrant

  # Qdrant only (no disk output)
  python scripts/run_minimal_pipeline.py --max-anime 10 --to-qdrant --no-disk
        """,
    )
    parser.add_argument(
        "--max-anime",
        type=int,
        default=10,
        help="Maximum number of anime to process (default: 10, max: 100)",
    )
    parser.add_argument(
        "--to-qdrant",
        action="store_true",
        help="Upload embeddings to Qdrant Cloud (requires QDRANT_URL/QDRANT_API_KEY in .env)",
    )
    parser.add_argument(
        "--no-disk",
        action="store_true",
        help="Skip saving results to disk (Qdrant-only mode)",
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

    if args.max_anime > 100:
        print(
            "❌ Error: --max-anime cannot exceed 100 (Jikan API rate limits)",
            file=sys.stderr,
        )
        return 1

    # Initialize vector store if requested
    vector_store = None
    if args.to_qdrant:
        try:
            logger.info("Initializing Qdrant vector store...")
            vector_store = get_production_vector_store()
            # Test connection with lightweight health check
            health = await vector_store.health_check()
            if health["status"] != "ok":
                print(
                    f"⚠️  Qdrant health check: {health['status']} - {health.get('message', 'unknown')}",
                    file=sys.stderr,
                )
                if health["status"] == "error":
                    print(
                        "\n❌ Cannot proceed with Qdrant upload. Fix configuration and retry.",
                        file=sys.stderr,
                    )
                    return 1
            logger.info(
                f"✓ Qdrant connection verified (collection: {settings.QDRANT_COLLECTION_NAME})"
            )
        except Exception as e:
            print(f"\n❌ Failed to initialize Qdrant: {e}", file=sys.stderr)
            print("\n💡 Troubleshooting:", file=sys.stderr)
            print("   1. Verify QDRANT_URL and QDRANT_API_KEY in .env", file=sys.stderr)
            print("   2. Check network connectivity to Qdrant Cloud", file=sys.stderr)
            print("   3. Run: python scripts/setup_qdrant.py to create collection", file=sys.stderr)
            return 1

    logger.info(f"Starting minimal pipeline with max_anime={args.max_anime}")
    mode = (
        "Qdrant + Disk"
        if args.to_qdrant and not args.no_disk
        else ("Qdrant only" if args.to_qdrant else "Disk only")
    )
    print(f"\n🎬 Fetching top {args.max_anime} anime from MyAnimeList...")
    print(f"   Mode: {mode}\n")

    try:
        start_time = datetime.now()
        result = await run_minimal_pipeline(
            max_anime=args.max_anime, vector_store=vector_store, save_to_disk=not args.no_disk
        )
        duration = (datetime.now() - start_time).total_seconds()
        result["duration_seconds"] = round(duration, 2)

        # Print human-friendly summary
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"   Anime processed:  {result['anime_processed']}")
        print(f"   Chunks created:   {result['chunks_created']}")
        print(f"   Embeddings:       {result['embeddings_generated']}")
        if args.to_qdrant:
            print(f"   Uploaded to Qdrant: {result['uploaded_to_qdrant']}")
        if not args.no_disk:
            print(f"   Saved to disk:    {result['saved_to_disk']}")
            print(f"   Output directory: {result['output_dir']}")
        print(f"   Duration:         {result['duration_seconds']} seconds")
        print("=" * 60 + "\n")

        # Verify Qdrant upload if requested
        if args.to_qdrant and result["uploaded_to_qdrant"] > 0:
            print("☁️  Qdrant Cloud Verification:")
            print(f"   - Collection: {settings.QDRANT_COLLECTION_NAME}")
            print(f"   - Points uploaded: {result['uploaded_to_qdrant']}")
            print(f"   - Vector dimension: {settings.QDRANT_VECTOR_SIZE}")
            print()

        # Verify disk output if saved
        if not args.no_disk and result["saved_to_disk"]:
            output_dir = Path(result["output_dir"])
            required_files = [
                "anime_metadata.json",
                "chunks.json",
                "embeddings.npy",
                "MANIFEST.json",
            ]
            all_exist = all((output_dir / f).exists() for f in required_files)

            if all_exist:
                print("📁 All output files verified successfully!")
                print("\n💡 Next steps:")
                if sys.platform == "win32":
                    print(f"   - Inspect results: type {output_dir}\\anime_metadata.json")
                else:
                    print(f"   - Inspect results: cat {output_dir}/anime_metadata.json | head -30")
                print(f"   - View MANIFEST: cat {output_dir}/MANIFEST.json")
            else:
                missing = [f for f in required_files if not (output_dir / f).exists()]
                logger.warning(f"Missing files: {missing}")
                print(f"⚠️  Missing files: {missing}")

        print("\n🎉 Pipeline execution complete!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ PIPELINE FAILED: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
