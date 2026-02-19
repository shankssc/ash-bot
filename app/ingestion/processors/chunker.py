"""
Semantic text chunking for anime knowledge.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextChunk:
    """Immutable chunk of text with metadata (frozen for safety)."""

    text: str
    chunk_id: str
    source_id: str
    source_type: str  # "synopsis", "review", "character_bio", etc.
    metadata: dict[str, Any]
    start_char: int
    end_char: int


class SemanticChunker:
    """Chunks text using semantic boundaries (sentence-aware)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100):
        from app.core.logging import get_logger

        self.logger = get_logger(__name__)

        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex with anime-specific handling."""
        # Handle common edge cases in anime synopses
        # Replace multiple dots with ellipsis
        text = re.sub(r"\.{2,}", "…", text)
        text = re.sub(r"(\w+)\s*\.\s*(\w+)", r"\1. \2", text)  # Fix spacing around periods

        # Split on sentence boundaries while preserving quotes and parentheses
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def chunk_text(
        self, text: str, source_id: str, source_type: str, meta: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        """
        Chunk text with semantic boundaries.

        Args:
            text: Raw text to chunk
            source_id: Unique identifier for source (e.g., "anime_5114")
            source_type: Type of content ("synopsis", "review", etc.)
            meta Additional metadata to attach to each chunk

        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            self.logger.debug(
                f"Skipping chunking for {source_id}: text too short ({len(text)} chars)"
            )
            return []

        metadata = meta or {}
        sentences = self._split_into_sentences(text)
        if not sentences:
            self.logger.warning(f"No valid sentences found for {source_id}")
            return []

        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_length = 0
        chunk_counter = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size and we have content, finalize chunk
            if current_sentences and (current_length + sentence_length) > self.chunk_size:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_id=f"{source_id}_chunk_{chunk_counter}",
                        source_id=source_id,
                        source_type=source_type,
                        metadata=metadata.copy(),
                        start_char=0,  # Will be calculated properly in full implementation
                        end_char=len(chunk_text),
                    )
                )

                # Create overlap with previous chunk (last 2 sentences)
                overlap_sentences = current_sentences[-2:]
                current_sentences = [*overlap_sentences, sentence]
                # -1 for no trailing space
                current_length = sum(len(s) + 1 for s in current_sentences) - 1
                chunk_counter += 1
            else:
                current_sentences.append(sentence)
                current_length += sentence_length + (1 if current_sentences else 0)  # +1 for space

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_id=f"{source_id}_chunk_{chunk_counter}",
                        source_id=source_id,
                        source_type=source_type,
                        metadata=metadata.copy(),
                        start_char=0,
                        end_char=len(chunk_text),
                    )
                )

        self.logger.debug(f"Chunked {source_id} into {len(chunks)} chunks")
        return chunks
