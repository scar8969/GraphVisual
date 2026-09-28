# KnowledgeGraphVisualization — Build & Improvement Plan

**Status:** Engine core done (15/17 tests). Server, UI, CLI, polish pending.

---

## PHASE 1 — ENGINE HARDENING (fix known issues)

| # | Issue | Fix |
|---|-------|-----|
| 1.1 | `test_exact_merge` fails: `mentions` counts initial+increments (2 calls → 3) | Fix test expectation OR make `add_mention` not double-count. Cleaner: `mentions` starts at 0, `+= 1` per call → 2 calls = 2. |
| 1.2 | `test_acronym_resolution` fails: `canonical_id("NASA")` doesn't resolve acronyms | Add acronym lookup to `canonical_id()` (mirror `_resolve` logic) |
| 1.3 | GPE missing: "based in Hawthorne, California" → Hawthorne not extracted | Add city/state pattern: `X, [A-Z]{2}` → GPE; "in <City>" after based/located/headquartered → GPE |
| 1.4 | Co-founder: "co-founded Microsoft with Paul Allen" → Allen not linked | Add `co-founded X with Y` → Y co-founded X (or Y founded X with conf 0.7) |
| 1.5 | "headquartered" → `related_to` (not in verb map) | Add "headquarter"/"headquartered" → based_in |
| 1.6 | Duplicate edges (same source/relation/target from multiple sentences) | Dedup in `_add_relation` (already merges weight, but ensure same-verb dedup) |

## PHASE 2 — SERVER (FastAPI)

| # | Endpoint | What it does |
|---|----------|--------------|
| 2.1 | `GET /` | Serve frontend (static) |
| 2.2 | `GET /api/graph` | Full graph JSON (nodes+edges+stats) |
| 2.3 | `GET /api/stats` | Stats only |
| 2.4 | `GET /api/search?q=...` | Search nodes + neighbors |
| 2.5 | `POST /api/upload` | Upload PDF(s) → save to docs/ → rebuild graph |
| 2.6 | `POST /api/rebuild` | Rebuild from docs/ |
| 2.7 | `GET /api/documents` | List processed PDFs |
| 2.8 | `GET /api/export?format=json|graphml|png` | Export |

- Cache graph in memory, rebuild on upload/rebuild
- Thread-safe (single rebuild lock)

## PHASE 3 — FRONTEND (interactive explorer)

| # | Feature |
|---|---------|
| 3.1 | Canvas force-directed graph (vanilla JS, no deps) — pan/zoom/drag |
| 3.2 | Node color by type (PERSON amber / ORG blue / GPE green) |
| 3.3 | Click node → highlight neighbors + side panel (aliases, sources, pages, confidence) |
| 3.4 | Edge label = relation type (on hover) |
| 3.5 | Search box → highlight matches |
| 3.6 | Filter by node type / relation type |
| 3.7 | Stats bar (nodes, edges, density, top entities) |
| 3.8 | Upload PDFs (drag-drop) + Rebuild button |
| 3.9 | Export buttons (JSON/GraphML/PNG) |
| 3.10 | Amber-on-black OpenTerminal palette (user's standing preference) |

## PHASE 4 — CLI

| # | Command |
|---|---------|
| 4.1 | `python -m kgraph.cli build --docs docs/ --out graph.json` |
| 4.2 | `python -m kgraph.cli stats` |
| 4.3 | `python -m kgraph.cli search "Musk"` |
| 4.4 | `python -m kgraph.cli export --format png` |

## PHASE 5 — TESTS (lock everything)

| # | Test |
|---|------|
| 5.1 | Fix 1.1/1.2 test expectations |
| 5.2 | GPE extraction test ("based in Hawthorne, California") |
| 5.3 | Co-founder test |
| 5.4 | API tests (graph/stats/search/upload/rebuild) via httpx ASGITransport |
| 5.5 | Full pipeline e2e: sample PDF → graph → JSON shape |

## PHASE 6 — POLISH & SHIP

| # | Task |
|---|------|
| 6.1 | README: hook, architecture, quick start, screenshots, sample output |
| 6.2 | Screenshots: explorer page (graph view, search, side panel) |
| 6.3 | pyproject.toml + requirements.txt |
| 6.4 | .gitignore (docs/, __pycache__, graph.json) |
| 6.5 | git init → commit → push to github.com/scar8969/KnowledgeGraphVisualization |
| 6.6 | LinkedIn post (nonchalant, bored tone) |

## VERIFICATION GATES
- After Phase 1: `python -m pytest tests/ -q` → 17/17 pass
- After Phase 5: all tests pass (target 22+)
- After Phase 6: server runs, explorer loads, screenshots show live graph
