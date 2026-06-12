"""Pytest configuration for rat-api tests.

Sets the minimum required environment variables so Settings() validates
without a real .env file present, enabling tests to run from any directory.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Inject stub env vars so Settings() instantiates without errors."""
    defaults = {
        "DATABASE_URL": "postgresql://test:test@localhost/test",
        "DIRECT_DATABASE_URL": "postgresql://test:test@localhost/test",
        "GROQ_API_KEY": "test-groq-key",
    }
    for key, value in defaults.items():
        if not os.environ.get(key):
            monkeypatch.setenv(key, value)
    # Clear lru_cache so each test starts with a fresh Settings instance
    from rat_api.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
