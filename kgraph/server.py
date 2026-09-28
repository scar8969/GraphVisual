"""FastAPI server: graph API + upload/rebuild + static frontend."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .graph_builder import KnowledgeGraphBuilder, build_graph_from_pdfs

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
FRONTEND_DIR = ROOT / "frontend"
EXPORT_DIR = ROOT / "exports"

app = FastAPI(title="Knowledge Graph Visualization", version="1.0.0")

_builder: KnowledgeGraphBuilder | None = None
_build_lock = threading.Lock()


def _get_builder() -> KnowledgeGraphBuilder:
    global _builder
    if _builder is None:
        with _build_lock:
            if _builder is None:
                _builder = build_graph_from_pdfs(DOCS_DIR)
    return _builder


def _rebuild() -> KnowledgeGraphBuilder:
    global _builder
    with _build_lock:
        _builder = build_graph_from_pdfs(DOCS_DIR)
    return _builder


@app.get("/api/graph")
def api_graph():
    b = _get_builder()
    return b.to_json()


@app.get("/api/stats")
def api_stats():
    return _get_builder().stats()


@app.get("/api/search")
def api_search(q: str = "", depth: int = 2):
    return {"results": _get_builder().search(q, max_depth=depth)}


@app.get("/api/documents")
def api_documents():
    docs = []
    for p in sorted(DOCS_DIR.glob("*.pdf")):
        docs.append({"filename": p.name, "size": p.stat().st_size})
    return {"documents": docs}


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    DOCS_DIR.mkdir(exist_ok=True)
    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = DOCS_DIR / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved.append(f.filename)
    if saved:
        _rebuild()
    return {"saved": saved, "stats": _get_builder().stats()}


@app.post("/api/rebuild")
def api_rebuild():
    b = _rebuild()
    return {"ok": True, "stats": b.stats()}


@app.get("/api/export")
def api_export(format: str = "json"):
    b = _get_builder()
    EXPORT_DIR.mkdir(exist_ok=True)
    if format == "graphml":
        path = EXPORT_DIR / "graph.graphml"
        b.export_graphml(path)
        return FileResponse(path, filename="graph.graphml")
    if format == "png":
        path = EXPORT_DIR / "graph.png"
        b.export_png(path)
        return FileResponse(path, filename="graph.png")
    path = EXPORT_DIR / "graph.json"
    path.write_text(json.dumps(b.to_json(), indent=2), encoding="utf-8")
    return FileResponse(path, filename="graph.json")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
