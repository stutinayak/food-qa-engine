import pytest
from fastapi.testclient import TestClient
from src.api.app import app

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app) 