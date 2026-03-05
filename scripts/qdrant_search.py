"""Test Qdrant search API compatibility."""

# ruff: noqa: E402, I
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.secrets import secrets
from app.retrieval.vector_store import QdrantVectorStore


async def test_search():
    store = QdrantVectorStore(
        url=settings.QDRANT_URL,
        api_key=secrets.QDRANT_API_KEY.get_secret_value(),
        collection_name=settings.QDRANT_COLLECTION_NAME,
    )
    await store._initialize()

    # Test with dummy vector (384-dim)
    dummy_vector = [0.1] * 384

    try:
        results = await store.search(dummy_vector, top_k=3)
        print(f"✅ Search successful! Found {len(results)} results")
        for r in results[:2]:
            print(f"   • {r.payload.get('anime_title', 'Unknown')} (score: {r.score:.4f})")
        return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_search())
    sys.exit(0 if success else 1)
