def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "version" in data

def test_api_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"

def test_endpoint_inventory_count(client):
    """Verify that all 46 API routes (mapping to 55 HTTP operations) are registered in OpenAPI."""
    from src.backend.main import app
    openapi = app.openapi()
    paths = openapi.get("paths", {})
    assert len(paths) == 46, f"Expected 46 unique API routes, got {len(paths)}"
    total_ops = sum(len(methods) for methods in paths.values())
    assert total_ops == 55, f"Expected 55 HTTP operations across routes, got {total_ops}"

