"""Unit tests for ingestion pipeline components."""

import numpy as np
import pytest

from app.ingestion.embedder import EmbeddingGenerator

# from unittest.mock import Mock, patch, MagicMock
from app.ingestion.processors.chunker import SemanticChunker, TextChunk


def test_semantic_chunker_basic():
    """Test semantic chunking with realistic anime synopsis."""
    chunker = SemanticChunker(chunk_size=200, min_chunk_size=50)
    text = (
        "Cowboy Bebop is a 1998 Japanese neo-noir space Western anime series. "
        "The series follows the adventures of bounty hunters Spike Spiegel and Jet Black. "
        "They are joined by Faye Valentine, Edward Wong, and Ein the corgi. "
        "The show explores themes of existentialism, loneliness, and redemption."
    )

    chunks = chunker.chunk_text(
        text=text,
        source_id="anime_1",
        source_type="synopsis",
        meta={"anime_id": 1, "title": "Cowboy Bebop"},
    )

    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.chunk_id.startswith("anime_1_chunk_") for c in chunks)
    assert all(len(c.text) >= 50 for c in chunks)  # Min chunk size respected
    assert sum(len(c.text) for c in chunks) >= len(text)  # No data loss


def test_chunker_empty_text():
    """Test chunker handles empty text gracefully."""
    chunker = SemanticChunker()
    chunks = chunker.chunk_text(text="", source_id="test", source_type="test")
    assert chunks == []


def test_chunker_short_text():
    """Test chunker skips text below min size."""
    chunker = SemanticChunker(min_chunk_size=100)
    chunks = chunker.chunk_text(text="Short text", source_id="test", source_type="test")
    assert chunks == []


@pytest.mark.skip(
    reason="Requires downloading model - run manually with: pytest tests/unit/test_ingestion.py::test_embedding_generator -v"
)
def test_embedding_generator():
    """Manual test: Verify embedding generation works with real model."""
    embedder = EmbeddingGenerator()
    texts = ["Space cowboy bounty hunters", "Magical girl transformation sequence"]

    embeddings, metadata = embedder.generate(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
    assert len(metadata) == 2
    assert metadata[0]["dimension"] == 384
    assert metadata[0]["text_length"] > 0


def test_chunker_special_characters():
    """Test chunker handles anime-specific punctuation."""
    chunker = SemanticChunker()
    text = "Attack on Titan (Shingeki no Kyojin) is a Japanese manga series. It's set in a world where humanity lives inside cities surrounded by enormous walls! The story follows Eren Yeager..."

    chunks = chunker.chunk_text(text=text, source_id="anime_16498", source_type="synopsis", meta={})

    assert len(chunks) > 0
    # Verify no truncation of special characters
    combined = " ".join([c.text for c in chunks])
    assert "Shingeki no Kyojin" in combined
    assert "Eren Yeager" in combined


def test_chunker_validation():
    """Test chunker rejects invalid parameters."""
    with pytest.raises(ValueError, match="chunk_size must be greater than chunk_overlap"):
        SemanticChunker(chunk_size=100, chunk_overlap=150)

    with pytest.raises(ValueError, match="min_chunk_size cannot exceed chunk_size"):
        SemanticChunker(chunk_size=100, min_chunk_size=150)
