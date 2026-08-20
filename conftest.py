import os

import pytest

API_KEYWORDS = (
    "claude",
    "anthropic",
    "openrouter",
    "qwen",
    "inference",
    "verifier",
    "prime",
    "primeintellect",
    "rl",
)

OFFLINE_SAFE_FILES = {"test_offline_eval.py"}


def _matches_api_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in API_KEYWORDS)


def pytest_ignore_collect(collection_path, path=None, config=None):  # type: ignore[override]
    if os.getenv("OFFLINE", "0") != "1":
        return False
    filename = os.path.basename(str(collection_path))
    if filename.startswith("test_") and filename not in OFFLINE_SAFE_FILES:
        return True
    return _matches_api_keyword(str(collection_path))


def pytest_collection_modifyitems(config, items):
    """Skip API/online tests in OFFLINE mode without modifying existing files."""
    if os.getenv("OFFLINE", "0") != "1":
        return
    skip = pytest.mark.skip(reason="OFFLINE=1: skipping network/API-dependent tests")
    for item in items:
        if _matches_api_keyword(item.nodeid):
            item.add_marker(skip)
