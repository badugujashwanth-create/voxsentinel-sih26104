"""Shared fixtures for backend tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.calls import get_session_store
from app.config import get_settings
from app.main import create_app


def _reset_process_state() -> None:
    """Clears cached singletons so each test starts from a clean process."""
    get_settings.cache_clear()
    get_session_store.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yields a test client backed by a fresh application instance."""
    _reset_process_state()
    with TestClient(create_app()) as test_client:
        yield test_client
    _reset_process_state()
