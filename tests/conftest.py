from __future__ import annotations

import os

# Force test environment so .env overrides (e.g. nomic-embed-text) don't leak
# into the test suite. Env vars beat .env in the Settings priority chain.
os.environ.setdefault("MRTA_ENV", "test")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
