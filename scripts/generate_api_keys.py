"""Generate cryptographically secure API keys for AniRAG."""

import string
import secrets
import argparse
import sys
from pathlib import Path

# CRITICAL: Path manipulation MUST be first executable code
# (Required for consistency across all scripts, even if not strictly needed here)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standard library imports (alphabetical)


def generate_api_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure API key.

    Args:
        length: Number of bytes to generate (default: 32 bytes = 43 URL-safe chars)

    Returns:
        URL-safe base64-encoded API key string
    """
    return secrets.token_urlsafe(length)


def generate_human_readable_key(length: int = 32) -> str:
    """
    Generate a human-readable API key with groups (easier to transcribe).

    Format: XXXX-XXXX-XXXX-XXXX-XXXX-XXXX

    Args:
        length: Total character length (must be divisible by 4 for clean grouping)

    Returns:
        Human-readable API key with hyphen separators
    """
    # Use alphanumeric characters only (no confusing chars like 0/O, 1/l/I)
    chars = string.ascii_uppercase + string.digits
    chars = (
        chars.replace("0", "").replace("O", "").replace("1", "").replace("I", "").replace("L", "")
    )

    key = "".join(secrets.choice(chars) for _ in range(length))
    # Group into 4-character chunks
    return "-".join([key[i : i + 4] for i in range(0, len(key), 4)])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cryptographically secure API keys for AniRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_api_keys.py
  python scripts/generate_api_keys.py --length 48
  python scripts/generate_api_keys.py --human-readable
  python scripts/generate_api_keys.py --count 3
        """,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=32,
        help="Key length in bytes (default: 32 bytes = 43 URL-safe characters)",
    )
    parser.add_argument(
        "--human-readable",
        action="store_true",
        help="Generate human-readable key with hyphen separators (e.g., ABCD-EFGH-IJKL)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of keys to generate (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n🔐 AniRAG API Key Generator")
    print("=" * 60)

    if args.human_readable:
        print(f"Generating {args.count} human-readable key(s) ({args.length} chars)...\n")
        for i in range(args.count):
            key = generate_human_readable_key(args.length)
            print(f"Key {i + 1}: {key}")
    else:
        print(f"Generating {args.count} URL-safe key(s) ({args.length} bytes)...\n")
        for i in range(args.count):
            key = generate_api_key(args.length)
            print(f"Key {i + 1}: {key}")

    print("\n" + "=" * 60)
    print("⚠️  SECURITY WARNING")
    print("=" * 60)
    print("• NEVER commit API keys to Git")
    print("• Store keys ONLY in .env (add to .gitignore) or GitHub Secrets")
    print("• Rotate keys immediately if accidentally exposed")
    print("• Use different keys for development/staging/production")
    print("=" * 60 + "\n")

    # Exit with success status
    return 0


if __name__ == "__main__":
    sys.exit(main())
