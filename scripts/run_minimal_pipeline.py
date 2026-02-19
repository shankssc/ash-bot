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
from app.core.config import settings
from app.core.logging import get_logger, init_logging
from app.ingestion.pipeline import run_minimal_pipeline

# Initialize logging BEFORE any modules that might log
init_logging(settings)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run minimal anime ingestion pipeline (Jikan API → Chunking → Embeddings → Disk)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_minimal_pipeline.py --max-anime 10
  python scripts/run_minimal_pipeline.py --max-anime 5 --debug
        """,
    )
    parser.add_argument(
        "--max-anime",
        type=int,
        default=10,
        help="Maximum number of anime to process (default: 10, max: 100)",
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

    logger.info(f"Starting minimal pipeline with max_anime={args.max_anime}")
    print(f"\n🎬 Fetching top {args.max_anime} anime from MyAnimeList...\n")

    try:
        start_time = datetime.now()
        result = await run_minimal_pipeline(max_anime=args.max_anime)
        duration = (datetime.now() - start_time).total_seconds()

        # Update result with actual duration
        result["duration_seconds"] = round(duration, 2)

        # Print human-friendly summary
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"   Anime processed:  {result['anime_processed']}")
        print(f"   Chunks created:   {result['chunks_created']}")
        print(f"   Embeddings:       {result['embeddings_generated']}")
        print(f"   Duration:         {result['duration_seconds']} seconds")
        print(f"   Output directory: {result['output_dir']}")
        print("=" * 60 + "\n")

        # Verify files exist
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
            print(
                f"   - Load embeddings shape: python -c \"import numpy as np; print(np.load('{output_dir / 'embeddings.npy'}').shape)\""
            )
        else:
            missing = [f for f in required_files if not (output_dir / f).exists()]
            logger.warning(f"Missing files: {missing}")
            print(f"⚠️  Missing files: {missing}")

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
