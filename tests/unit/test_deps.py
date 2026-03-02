"""Unit tests for API dependency injection."""

from unittest.mock import MagicMock, patch


class TestDependencies:
    """Test FastAPI dependency functions."""

    def test_get_settings_returns_cached(self):
        """get_settings returns cached Settings instance (lru_cache)."""
        from app.core.config import get_settings as get_cfg

        s1 = get_cfg()
        s2 = get_cfg()
        assert s1 is s2  # lru_cache returns same instance

    @patch("app.api.deps.get_vector_store")  # ← Patch where deps.py imports it
    def test_get_vector_store_dep(self, mock_get_vs):
        """get_vector_store dependency returns store instance."""
        # Arrange: Configure the mock return value
        mock_store = MagicMock()
        mock_get_vs.return_value = mock_store

        # Import AFTER patching to ensure mock is in place
        from app.api.deps import get_vector_store

        # Act
        result = get_vector_store()

        # Assert
        assert result is mock_store
        mock_get_vs.assert_called_once()
