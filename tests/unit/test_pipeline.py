"""Unit tests for MinimalIngestionPipeline orchestration."""

import json

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.ingestion.pipeline import MinimalIngestionPipeline
from app.ingestion.processors.chunker import TextChunk


class TestMinimalIngestionPipeline:
    """Test minimal ingestion pipeline workflow."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory for tests."""
        return tmp_path / "pipeline_output"

    @pytest.fixture
    def pipeline(self, temp_output_dir):
        """Create pipeline instance with test config."""
        # Mock embedder to avoid loading real model
        with patch("app.ingestion.pipeline.EmbeddingGenerator") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.generate.return_value = (np.array([]), [])
            mock_embedder_class.return_value = mock_embedder

            return MinimalIngestionPipeline(
                output_dir=temp_output_dir,
                vector_store=None,
                save_to_disk=True,
            )

    def _create_mock_anime(self, mal_id=1, title="Test", synopsis="Test synopsis"):
        """Helper to create mock anime data."""
        return {
            "mal_id": mal_id,
            "title": title,
            "synopsis": synopsis,
            "url": f"https://myanimelist.net/anime/{mal_id}",
            "images": {"jpg": {"large_image_url": "https://example.com/image.jpg"}},
            "score": 9.0,
            "type": "TV",
            "episodes": 26,
            "aired": {"string": "1998-2099"},
            "genres": [{"name": "Action"}, {"name": "Sci-Fi"}],
        }

    def _create_mock_chunk(self, anime_id=1, title="Test", chunk_id="chunk_0"):
        """Helper to create mock TextChunk."""
        return TextChunk(
            text="Test chunk text",
            chunk_id=f"anime_{anime_id}_{chunk_id}",
            source_id=f"anime_{anime_id}",
            source_type="synopsis",
            metadata={"anime_id": anime_id, "anime_title": title},
            start_char=0,
            end_char=15,
        )

    @pytest.mark.asyncio
    async def test_run_with_zero_anime(self, pipeline):
        """run() with max_anime=0 returns empty summary."""
        with patch("app.ingestion.pipeline.JikanClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # Properly mock async generator that yields nothing
            async def mock_iterate_empty(*args, **kwargs):
                return  # Early return, then yield to make it async generator
                yield  # This line never executes but makes it a generator

            mock_client.iterate_top_anime = mock_iterate_empty
            mock_client_class.return_value = mock_client

            result = await pipeline.run(max_anime=0)

            assert result["anime_processed"] == 0
            assert result["chunks_created"] == 0
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_run_processes_anime_with_synopsis(self, pipeline, temp_output_dir):
        """run() successfully processes anime with synopsis text."""
        mock_anime = self._create_mock_anime()
        mock_chunk = self._create_mock_chunk()

        with patch("app.ingestion.pipeline.JikanClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # Mock async iterator properly
            async def mock_iterate(*args, **kwargs):
                yield mock_anime

            mock_client.iterate_top_anime = mock_iterate
            mock_client_class.return_value = mock_client

            # Mock chunker and embedder
            with patch.object(pipeline.chunker, "chunk_text", return_value=[mock_chunk]):
                with patch.object(pipeline.embedder, "generate") as mock_embed:
                    mock_embed.return_value = (np.array([[0.1] * 384]), [{}])

                    result = await pipeline.run(max_anime=1)

                    assert result["anime_processed"] == 1
                    assert result["chunks_created"] == 1
                    assert result["embeddings_generated"] == 1
                    assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_run_skips_empty_synopsis(self, pipeline):
        """run() skips chunking when synopsis is empty or unavailable."""
        mock_anime = self._create_mock_anime(synopsis="")

        with patch("app.ingestion.pipeline.JikanClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            async def mock_iterate(*args, **kwargs):
                yield mock_anime

            mock_client.iterate_top_anime = mock_iterate
            mock_client_class.return_value = mock_client

            # Chunker should NOT be called for empty synopsis
            with patch.object(pipeline.chunker, "chunk_text") as mock_chunk:
                result = await pipeline.run(max_anime=1)

                assert result["anime_processed"] == 1
                assert result["chunks_created"] == 0
                mock_chunk.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_skips_disk_save_when_disabled(self, temp_output_dir):
        """run() skips file I/O when save_to_disk=False."""
        # Mock embedder at class level to avoid model loading
        with patch("app.ingestion.pipeline.EmbeddingGenerator") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.generate.return_value = (np.array([]), [])
            mock_embedder_class.return_value = mock_embedder

            pipeline_no_disk = MinimalIngestionPipeline(
                output_dir=temp_output_dir,
                save_to_disk=False,
            )

            with patch("app.ingestion.pipeline.JikanClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)

                # Same proper async generator mock
                async def mock_iterate_empty(*args, **kwargs):
                    return
                    yield

                mock_client.iterate_top_anime = mock_iterate_empty
                mock_client_class.return_value = mock_client

                result = await pipeline_no_disk.run(max_anime=1)

                assert result["saved_to_disk"] is False
                assert result["output_dir"] is None
                # Verify no files were created
                assert not (temp_output_dir / "anime_metadata.json").exists()

    @pytest.mark.asyncio
    async def test_upload_to_qdrant_success(self, pipeline, temp_output_dir):
        """_upload_to_qdrant successfully uploads VectorPoints."""
        # First set up a mock vector store
        mock_store = MagicMock()
        mock_store.upsert = AsyncMock(return_value=True)
        pipeline.vector_store = mock_store

        mock_chunk = self._create_mock_chunk()
        mock_embedding = np.array([0.1] * 384)

        result = await pipeline._upload_to_qdrant([mock_chunk], np.array([mock_embedding]))

        assert result == 1
        mock_store.upsert.assert_called_once()
        # Verify VectorPoint was constructed
        call_args = mock_store.upsert.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].payload["anime_id"] == 1

    @pytest.mark.asyncio
    async def test_upload_to_qdrant_no_vector_store_first(self, pipeline):
        """_upload_to_qdrant raises RuntimeError if vector_store not set (checked first)."""
        pipeline.vector_store = None
        mock_chunk = self._create_mock_chunk()

        # RuntimeError for missing vector_store is checked BEFORE count mismatch
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            await pipeline._upload_to_qdrant([mock_chunk], np.array([[0.1] * 384]))

    @pytest.mark.asyncio
    async def test_upload_to_qdrant_mismatched_counts(self, pipeline):
        """_upload_to_qdrant raises ValueError on chunk/embedding count mismatch."""
        # First set vector_store so we pass the first check
        mock_store = MagicMock()
        pipeline.vector_store = mock_store

        mock_chunk = self._create_mock_chunk()

        # Now test the count mismatch (checked after vector_store)
        with pytest.raises(ValueError, match="Chunk count"):
            await pipeline._upload_to_qdrant([mock_chunk], np.array([]))

    def test_save_results_creates_expected_files(self, pipeline, temp_output_dir):
        """_save_results creates all expected output files."""
        anime_data = [{"anime_id": 1, "title": "Test"}]
        chunks = [self._create_mock_chunk()]
        embeddings = np.array([[0.1] * 384])
        embedding_metadata = [{}]

        pipeline._save_results(
            anime_data=anime_data,
            chunks=chunks,
            embeddings=embeddings,
            embedding_metadata=embedding_metadata,
            duration_seconds=1.5,
            uploaded_to_qdrant=1,
        )

        # Verify all expected files exist
        assert (temp_output_dir / "anime_metadata.json").exists()
        assert (temp_output_dir / "chunks.json").exists()
        assert (temp_output_dir / "embeddings.npy").exists()
        assert (temp_output_dir / "embedding_metadata.json").exists()
        assert (temp_output_dir / "MANIFEST.json").exists()
        assert (temp_output_dir / "README.md").exists()

        # Verify MANIFEST content
        with open(temp_output_dir / "MANIFEST.json") as f:
            manifest = json.load(f)
        assert manifest["anime_count"] == 1
        assert manifest["chunk_count"] == 1
