from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

# Since we removed the root and health endpoints, we should remove those tests.
# Instead, we can test that they are indeed gone (404) or just test the new endpoints with mocks.

def test_root_not_found():
    response = client.get("/")
    assert response.status_code == 404

def test_health_not_found():
    response = client.get("/api/v1/health")
    assert response.status_code == 404

# Mocking services to avoid actual API calls during basic testing
@patch('app.services.comparison_service.ComparisonService.search_vectors')
def test_compare_search(mock_search):
    # Mock return value
    mock_search.return_value = {"matches": [], "namespace": ""}
    
    payload = {
        "text": "test query",
        "embed_type": "openai"
    }
    response = client.post("/api/v1/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["embed_type"] == "openai"
    assert "results" in data

if __name__ == "__main__":
    test_root_not_found()
    test_health_not_found()
    test_compare_search()
    print("Basic tests passed!")
