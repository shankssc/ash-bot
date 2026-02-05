"""Pytest configuration."""

import os

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["VECTOR_DB_PATH"] = "./data/test_chroma_db"
