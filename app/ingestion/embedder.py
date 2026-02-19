"""
Generate embeddings using sentence-transformers.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentence_transformers import SentenceTransformer  # type: ignore

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers with safety checks."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=False,  # Security best practice
            )

            dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Embedding model loaded. Dimension: {dim}")

            # Validate dimension matches Qdrant config
            if dim != settings.QDRANT_VECTOR_SIZE:
                logger.warning(
                    f"Embedding dimension ({dim}) != QDRANT_VECTOR_SIZE ({settings.QDRANT_VECTOR_SIZE}). "
                    "Update .env to match model dimension."
                )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.array([]), []

        logger.debug(f"Generating embeddings for {len(texts)} texts...")

        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            metadata = [
                {
                    "text_length": len(text),
                    "model": self.model_name,
                    "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                }
                for text in texts
            ]

            logger.debug(f"✓ Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
            return embeddings, metadata

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise

    def generate_single(self, text: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        embeddings, metadata = self.generate([text])
        return embeddings[0], metadata[0]
