"""Unit tests for SemanticChunker."""

from app.ingestion.processors.chunker import SemanticChunker


class TestSemanticChunker:
    """Test text chunking with semantic boundaries."""

    def test_empty_text_returns_empty_list(self):
        """Empty or short text returns no chunks."""
        chunker = SemanticChunker(min_chunk_size=100)
        result = chunker.chunk_text("", "src", "wiki", {})
        assert result == []

    def test_text_below_min_size_returns_empty(self):
        """Text shorter than min_chunk_size returns empty."""
        chunker = SemanticChunker(min_chunk_size=100)
        result = chunker.chunk_text("Short.", "src", "wiki", {})
        assert result == []

    def test_single_sentence_chunk(self):
        """Single long sentence creates one chunk."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        text = "This is a reasonably long sentence that should fit in one chunk. " * 3

        result = chunker.chunk_text(text, "src-1", "jikan", {"anime_id": 1})

        assert len(result) == 1
        assert result[0].source_id == "src-1"
        assert result[0].source_type == "jikan"
        assert result[0].metadata == {"anime_id": 1}
        assert "chunk_0" in result[0].chunk_id

    def test_multiple_chunks_with_overlap(self):
        """Long text creates multiple chunks with overlap."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=30, min_chunk_size=50)
        text = ". ".join([f"Sentence {i} with some content" for i in range(10)]) + "."

        result = chunker.chunk_text(text, "src-2", "wiki", {})

        assert len(result) >= 2
        # Verify overlap: last sentences of chunk N appear in chunk N+1
        if len(result) >= 2:
            assert result[0].text[-20:] in result[1].text or result[1].text[:20] in result[0].text

    def test_chunk_metadata_preserved(self):
        """All chunks preserve source metadata."""
        chunker = SemanticChunker()
        metadata = {"studio": "Sunrise", "year": 1998, "episodes": 26}

        text = "Plot summary. " * 20
        result = chunker.chunk_text(text, "bebop", "jikan", metadata)

        for chunk in result:
            assert chunk.metadata == metadata
            assert chunk.source_id == "bebop"
            assert chunk.source_type == "jikan"
