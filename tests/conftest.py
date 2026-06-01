"""Shared pytest configuration — make api/ and project root importable from tests/."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
API_DIR = os.path.join(ROOT, "api")

# Add project root (for `src.*`) and api/ (for `engines.*`) to sys.path.
for p in (ROOT, API_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
