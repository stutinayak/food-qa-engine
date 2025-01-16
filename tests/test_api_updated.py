from fastapi.testclient import TestClient
import pytest

def test_health_check(test_client: TestClient):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_question(test_client: TestClient):
    """Test asking a simple question."""
    response = test_client.post(
        "/ask",
        json={"text": "What foods are high in protein?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "matches" in data

def test_empty_question(test_client: TestClient):
    """Test asking an empty question."""
    response = test_client.post(
        "/ask",
        json={"text": ""}
    )
    assert response.status_code == 422

def test_malformed_request(test_client: TestClient):
    """Test sending malformed request."""
    response = test_client.post(
        "/ask",
        json={"invalid_field": "test"}
    )
    assert response.status_code == 422 