"""FastAPI server: graph API + upload/rebuild + static frontend."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .graph_builder import KnowledgeGraphBuilder

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
FRONTEND_DIR = ROOT / "frontend"
EXPORT_DIR = ROOT / "exports"

app = FastAPI(title="Knowledge Graph Visualization", version="1.0.0")

_builder: KnowledgeGraphBuilder | None = None
_build_lock = threading.Lock()
_CACHE_FILE = ROOT / "cache" / "build_cache.json"


def _file_hash(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_cache(cache: dict):
    _CACHE_FILE.parent.mkdir(exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def _get_builder() -> KnowledgeGraphBuilder:
    if _builder is None:
        _rebuild()
    assert _builder is not None
    return _builder


def _rebuild() -> KnowledgeGraphBuilder:
    """Incremental rebuild: only reprocess PDFs whose content hash changed.

    Cache stores per-PDF extracted (mentions, relations, pages) keyed by file
    hash. Unchanged PDFs replay from cache; changed/new PDFs reprocess.
    """
    global _builder
    with _build_lock:
        cache = _load_cache()
        from .graph_builder import KnowledgeGraphBuilder
        from .pdf_reader import PDFReader

        reader = PDFReader(DOCS_DIR)
        pdfs = reader.list_pdfs()
        # drop cache entries for deleted PDFs
        cache = {k: v for k, v in cache.items()
                 if k in {p.name for p in pdfs}}

        builder = KnowledgeGraphBuilder()
        changed_any = False
        for p in pdfs:
            h = _file_hash(p)
            entry = cache.get(p.name)
            if entry and entry.get("hash") == h and "mentions" in entry:
                # replay cached extraction
                for m, t, ctx in entry["mentions"]:
                    builder.resolver.add_mention(m, t, p.name, 1, context=ctx)
                for src, tgt, rel, verb, conf, page, sent in entry["relations"]:
                    from .relations import Relation
                    builder._add_relation(Relation(
                        source=src, target=tgt, relation=rel, verb=verb,
                        confidence=conf, filename=p.name, page=page, sentence=sent))
            else:
                doc = reader.read_pdf(p)
                if doc:
                    changed_any = True
                    # snapshot resolver state before this doc
                    before = set(builder.resolver._entities.keys())
                    for page in doc.pages:
                        builder._process_page(doc, page.page, page.text)
                    # cache only this doc's new mentions
                    mentions = []
                    for cid in builder.resolver._entities:
                        if cid not in before:
                            ent = builder.resolver._entities[cid]
                            for alias in ent.aliases:
                                mentions.append((alias, ent.type, ""))
                    relations = []
                    for u, v, d in builder.graph.edges(data=True):
                        if d.get("filename") == p.name:
                            relations.append((u, v, d.get("relation"), d.get("verb", ""),
                                              d.get("confidence", 0.5), d.get("page", 1),
                                              d.get("sentence", "")))
                    cache[p.name] = {"hash": h, "pages": len(doc.pages),
                                     "mentions": mentions, "relations": relations}
        builder._finalize()
        if changed_any:
            _save_cache(cache)
        _builder = builder
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


@app.get("/api/analytics")
def api_analytics():
    return _get_builder().analytics()


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
        fname = f.filename or ""
        if not fname.lower().endswith(".pdf"):
            continue
        dest = DOCS_DIR / fname
        content = await f.read()
        dest.write_bytes(content)
        saved.append(fname)
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
