"""
Minimal ingestion pipeline: Fetch → Chunk → Embed → Save to disk.
"""

from __future__ import annotations

import json

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.processors.chunker import SemanticChunker, TextChunk
from app.ingestion.sources.jikan_client import JikanClient

settings = get_settings()


class MinimalIngestionPipeline:
    """Minimal pipeline for ingesting anime data to disk."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("data/processed/minimal_pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = SemanticChunker()
        self.embedder = EmbeddingGenerator()

    async def run(self, max_anime: int = 10) -> dict[str, Any]:
        """
        Run minimal ingestion pipeline.

        Args:
            max_anime: Maximum number of anime to process

        Returns:
            Summary statistics
        """
        from app.core.logging import get_logger  # Late import

        logger = get_logger(__name__)

        logger.info(f"Starting minimal ingestion pipeline (max_anime={max_anime})")
        logger.info(f"Starting minimal ingestion pipeline (max_anime={max_anime})")

        start_time = datetime.now()

        anime_data = []
        chunks: list[TextChunk] = []
        chunk_texts = []

        duration = (datetime.now() - start_time).total_seconds()

        # Step 1: Fetch anime data
        logger.info("Step 1: Fetching anime data from Jikan API...")
        async with JikanClient() as client:
            async for anime in client.iterate_top_anime(max_entries=max_anime):
                anime_id = anime["mal_id"]
                title = anime["title"]
                synopsis = anime.get("synopsis", "")

                logger.info(f"Processing: {title} (ID: {anime_id})")

                # Store minimal anime metadata
                anime_data.append(
                    {
                        "anime_id": anime_id,
                        "title": title,
                        "url": anime.get("url", ""),
                        "image_url": anime.get("images", {})
                        .get("jpg", {})
                        .get("large_image_url", ""),
                        "score": anime.get("score", 0),
                        "type": anime.get("type", ""),
                        "episodes": anime.get("episodes", 0),
                        "aired": anime.get("aired", {}).get("string", ""),
                        "genres": [g["name"] for g in anime.get("genres", [])],
                    }
                )

                # Step 2: Chunk synopsis
                if synopsis and synopsis.lower() != "no synopsis available":
                    metadata = {
                        "anime_id": anime_id,
                        "anime_title": title,
                        "source": "jikan_api",
                        "field": "synopsis",
                        "fetched_at": datetime.utcnow().isoformat(),
                    }
                    synopsis_chunks = self.chunker.chunk_text(
                        text=synopsis,
                        source_id=f"anime_{anime_id}",
                        source_type="synopsis",
                        meta=metadata,
                    )
                    chunks.extend(synopsis_chunks)
                    chunk_texts.extend([c.text for c in synopsis_chunks])

        # Step 3: Generate embeddings
        logger.info(f"Step 2: Generating embeddings for {len(chunk_texts)} chunks...")
        if chunk_texts:
            embeddings, embedding_metadata = self.embedder.generate(chunk_texts)
        else:
            logger.warning("No chunks to embed!")
            embeddings = np.array([])
            embedding_metadata = []

        duration = (datetime.now() - start_time).total_seconds()

        # Step 4: Save to disk
        logger.info("Step 3: Saving results to disk...")
        self._save_results(anime_data, chunks, embeddings, embedding_metadata, duration)

        # Return summary
        summary = {
            "anime_processed": len(anime_data),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
            "output_dir": str(self.output_dir),
            "duration_seconds": round(duration, 2),
            "status": "success",
        }

        logger.info(f"Pipeline completed successfully: {summary}")
        return summary

    def _save_results(
        self,
        anime_data: list[dict[str, Any]],
        chunks: list[TextChunk],
        embeddings: np.ndarray,
        embedding_metadata: list[dict[str, Any]],
        duration_seconds: float,
    ):
        """Save pipeline results to disk."""
        # Save anime metadata
        with open(self.output_dir / "anime_metadata.json", "w") as f:
            json.dump(anime_data, f, indent=2)

        # Save chunks
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_id": c.source_id,
                "source_type": c.source_type,
                "metadata": c.metadata,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]
        with open(self.output_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # Save embeddings (as .npy for efficiency)
        np.save(self.output_dir / "embeddings.npy", embeddings)

        # Save embedding metadata
        with open(self.output_dir / "embedding_metadata.json", "w") as f:
            json.dump(embedding_metadata, f, indent=2)

        # CREATE MANIFEST.JSON
        manifest = {
            "pipeline_version": settings.APP_VERSION,
            "generated_at": datetime.utcnow().isoformat(),
            "duration_seconds": duration_seconds,
            "anime_count": len(anime_data),
            "chunk_count": len(chunks),
            "embedding_shape": embeddings.shape if len(embeddings) > 0 else [0, 0],
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "vector_size": settings.QDRANT_VECTOR_SIZE,
        }

        # Save MANIFEST.json
        with open(self.output_dir / "MANIFEST.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Save README with pipeline info
        with open(self.output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(
                f"""# Minimal Ingestion Pipeline Output

Generated on: {settings.APP_VERSION}
Duration: {manifest["duration_seconds"]:.2f} seconds
Anime processed: {len(anime_data)}
Chunks created: {len(chunks)}
Embedding dimension: {embeddings.shape[1] if len(embeddings) > 0 else 0}

## Files
- `anime_metadata.json`: Basic anime information
- `chunks.json`: Text chunks with metadata
- `embeddings.npy`: Embedding vectors (numpy array)
- `embedding_metadata.json`: Per-embedding metadata
- `MANIFEST.json`: Pipeline execution metadata

## Usage
```python
import numpy as np
embeddings = np.load('embeddings.npy')
"""
            )


async def run_minimal_pipeline(max_anime: int = 10):
    """Convenience function to run pipeline."""
    pipeline = MinimalIngestionPipeline()
    return await pipeline.run(max_anime=max_anime)
