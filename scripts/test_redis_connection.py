"""Test Redis connectivity before running cache integration tests."""

# ruff: noqa: E402
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.secrets import secrets
import asyncio


async def test_redis():
    try:
        from redis import asyncio as aioredis

        client = aioredis.from_url(
            settings.REDIS_URL,
            password=secrets.REDIS_TOKEN.get_secret_value(),
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        # Test connection
        await client.ping()
        print("✅ Redis connection successful")

        # Test set/get
        await client.set("test_key", "test_value", ex=10)
        value = await client.get("test_key")
        if value != "test_value":
            print(f"❌ Redis set/get failed: Expected 'test_value', got '{value}'", file=sys.stderr)
            await client.close()
            return 1
        print("✅ Redis set/get working")

        # Cleanup
        await client.delete("test_key")
        await client.close()
        print("✅ Redis cleanup complete")
        return 0

    except Exception as e:
        print(f"\n❌ Redis connection failed: {e}", file=sys.stderr)
        print("\n💡 Troubleshooting:", file=sys.stderr)
        print("   1. Verify REDIS_URL format: https://<your-db-id>.upstash.io", file=sys.stderr)
        print("   2. Verify REDIS_TOKEN is correct (not placeholder)", file=sys.stderr)
        print("   3. Check network connectivity to Upstash", file=sys.stderr)
        print("   4. Run: python scripts/test_redis_connection.py --debug", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(test_redis()))
