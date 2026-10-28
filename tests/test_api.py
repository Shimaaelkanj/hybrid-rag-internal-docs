from fastapi.testclient import TestClient

from app.api import app


def test_health():
    with TestClient(app) as c:
        r = c.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_documents_shape():
    with TestClient(app) as c:
        body = c.get("/v1/documents").json()
        assert set(body) == {"total_documents", "total_chunks", "documents"}


def test_ingest_rejects_unsupported_file_type():
    with TestClient(app) as c:
        r = c.post("/v1/ingest", files={"files": ("virus.exe", b"abc")})
        assert r.status_code == 415


def test_ingest_rejects_unknown_strategy():
    with TestClient(app) as c:
        r = c.post("/v1/ingest", files={"files": ("a.md", b"# Title\nSome text here.")},
                   data={"strategy": "nonsense"})
        assert r.status_code == 400


def test_search_and_ask_validate_input():
    with TestClient(app) as c:
        assert c.post("/v1/search", json={"query": ""}).status_code == 422
        assert c.post("/v1/ask", json={}).status_code == 422
        assert c.post("/v1/ask", json={"question": "x", "retrieval_mode": "bogus"}).status_code == 422


def test_openapi_lists_the_v1_endpoints():
    with TestClient(app) as c:
        paths = c.get("/openapi.json").json()["paths"]
        for p in ("/v1/ask", "/v1/documents", "/v1/ingest", "/v1/search", "/v1/health"):
            assert p in paths
