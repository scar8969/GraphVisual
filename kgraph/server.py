"""FastAPI server: graph API + upload/rebuild + static frontend."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .graph_builder import KnowledgeGraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("kgraph")

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
FRONTEND_DIR = ROOT / "frontend"
EXPORT_DIR = ROOT / "exports"

app = FastAPI(title="Knowledge Graph Visualization", version="1.1.0")


@app.middleware("http")
async def log_requests(request, call_next):
    """Structured request logging: request ID + method + path + timing."""
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    path = str(request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] %s %s failed", rid, request.method, path)
        raise
    dt_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("[%s] %s %s -> %d (%sms)", rid, request.method, path,
                response.status_code, dt_ms)
    response.headers["X-Request-ID"] = rid
    return response

_builder: KnowledgeGraphBuilder | None = None
_build_lock = threading.Lock()
_CACHE_FILE = ROOT / "cache" / "build_cache.json"
_GRAPH_FILE = ROOT / "cache" / "graph.json"


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
    global _builder
    if _builder is None:
        _builder = _load_persisted()
        if _builder is None:
            _rebuild()
    assert _builder is not None
    return _builder


def _load_persisted() -> KnowledgeGraphBuilder | None:
    """Load the last built graph from disk (fast boot)."""
    if not _GRAPH_FILE.exists():
        return None
    try:
        from .graph_builder import KnowledgeGraphBuilder
        data = json.loads(_GRAPH_FILE.read_text(encoding="utf-8"))
        b = KnowledgeGraphBuilder()
        for n in data.get("nodes", []):
            b.graph.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
        for e in data.get("edges", []):
            b.graph.add_edge(e["source"], e["target"],
                             **{k: v for k, v in e.items() if k not in ("source", "target")})
        # rebuild resolver from node aliases
        for n in data.get("nodes", []):
            for alias in n.get("aliases", [n.get("label", n["id"])]):
                b.resolver.add_mention(alias, n.get("type", "OTHER"), "persisted", 1)
        return b
    except Exception:  # noqa: BLE001
        return None


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
        # persist the built graph for fast boot
        try:
            _GRAPH_FILE.parent.mkdir(exist_ok=True)
            _GRAPH_FILE.write_text(json.dumps(builder.to_json()), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
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
    errors = []
    MAX_MB = 50
    before = _get_builder().stats() if _builder is not None else {"nodes": 0, "edges": 0}
    for f in files:
        fname = f.filename or ""
        if not fname.lower().endswith(".pdf"):
            errors.append({"file": fname, "error": "not a .pdf file"})
            continue
        content = await f.read()
        if len(content) > MAX_MB * 1024 * 1024:
            errors.append({"file": fname, "error": f"larger than {MAX_MB}MB"})
            continue
        # magic-byte check: PDFs start with %PDF
        if not content.startswith(b"%PDF"):
            errors.append({"file": fname, "error": "not a valid PDF (missing %PDF header)"})
            continue
        dest = DOCS_DIR / fname
        dest.write_bytes(content)
        saved.append(fname)
    diff = {"added_nodes": 0, "added_edges": 0}
    if saved:
        _rebuild()
        after = _get_builder().stats()
        diff = {"added_nodes": max(0, after["nodes"] - before["nodes"]),
                "added_edges": max(0, after["edges"] - before["edges"])}
    return {"saved": saved, "errors": errors, "diff": diff, "stats": _get_builder().stats()}


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
    if format == "evidence":
        # citations report: every edge + source sentence + page
        lines = ["# Knowledge Graph — Evidence Report", ""]
        for u, v, d in sorted(b.graph.edges(data=True), key=lambda x: -x[2].get("confidence", 0)):
            src = b.graph.nodes[u].get("label", u)
            tgt = b.graph.nodes[v].get("label", v)
            lines.append(f"## {src} → {tgt} ({d.get('relation')}, conf {d.get('confidence')})")
            lines.append(f"- Source: {d.get('filename')}, page {d.get('page')}")
            if d.get("sentence"):
                lines.append(f"- Evidence: \"{d['sentence']}\"")
            lines.append("")
        path = EXPORT_DIR / "evidence.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return FileResponse(path, filename="evidence.md")
    path = EXPORT_DIR / "graph.json"
    path.write_text(json.dumps(b.to_json(), indent=2), encoding="utf-8")
    return FileResponse(path, filename="graph.json")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
