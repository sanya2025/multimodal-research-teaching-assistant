from __future__ import annotations

import os

# Force test environment so .env overrides (e.g. nomic-embed-text) don't leak
# into the test suite. Env vars beat .env in the Settings priority chain.
os.environ.setdefault("MRTA_ENV", "test")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def pytest_collection_modifyitems(items) -> None:
    """Apply the directory-derived markers so CI can select by tier.

    Done here rather than with per-file decorators because the unit/integration/
    eval split *is* the directory layout — repeating it in 37 files would be
    three chances to drift. `heavy` stays explicit: whether a test loads real
    weights is a property of the test, and two heavy tests live in a directory
    that is otherwise entirely fast.
    """
    by_directory = {"unit": "unit", "integration": "integration", "evaluation": "eval"}
    for item in items:
        parts = item.path.parts if hasattr(item, "path") else ()
        for directory, marker in by_directory.items():
            if directory in parts:
                item.add_marker(marker)
                break
