"""API tests via httpx ASGITransport (no server process needed)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

from kgraph.server import app

# ensure docs dir has the sample PDF
SAMPLE = Path(__file__).parent.parent / "docs" / "sample_company.pdf"


@pytest.fixture
def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_graph_endpoint(client):
    r = await client.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data and "edges" in data and "stats" in data
    assert data["stats"]["nodes"] > 0


@pytest.mark.asyncio
async def test_stats_endpoint(client):
    r = await client.get("/api/stats")
    assert r.status_code == 200
    assert "nodes" in r.json()


@pytest.mark.asyncio
async def test_search_endpoint(client):
    r = await client.get("/api/search", params={"q": "Musk"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert any("musk" in res["node"] for res in results)


@pytest.mark.asyncio
async def test_documents_endpoint(client):
    r = await client.get("/api/documents")
    assert r.status_code == 200
    assert any("sample_company" in d["filename"] for d in r.json()["documents"])


@pytest.mark.asyncio
async def test_rebuild_endpoint(client):
    r = await client.post("/api/rebuild")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["stats"]["nodes"] > 0


@pytest.mark.asyncio
async def test_export_json(client):
    r = await client.get("/api/export", params={"format": "json"})
    assert r.status_code == 200
    assert "nodes" in r.json() or r.headers["content-type"].startswith("application/json")


@pytest.mark.asyncio
async def test_index_served(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "Knowledge Graph" in r.text
