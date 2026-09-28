"""CLI for the knowledge graph builder.

Usage:
    python -m kgraph.cli build [--docs DIR] [--out graph.json]
    python -m kgraph.cli stats [--docs DIR]
    python -m kgraph.cli search "Musk" [--docs DIR] [--depth 2]
    python -m kgraph.cli export --format png [--docs DIR]
"""
from __future__ import annotations

import argparse
import json
import sys

from .graph_builder import build_graph_from_pdfs


def main(argv=None):
    p = argparse.ArgumentParser(prog="kgraph", description="PDF → knowledge graph")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build graph and save JSON")
    b.add_argument("--docs", default="docs")
    b.add_argument("--out", default="graph.json")

    s = sub.add_parser("stats", help="print graph stats")
    s.add_argument("--docs", default="docs")

    q = sub.add_parser("search", help="search the graph")
    q.add_argument("query")
    q.add_argument("--docs", default="docs")
    q.add_argument("--depth", type=int, default=2)

    e = sub.add_parser("export", help="export graph")
    e.add_argument("--format", choices=["json", "graphml", "png"], default="json")
    e.add_argument("--docs", default="docs")
    e.add_argument("--out", default=None)

    args = p.parse_args(argv)

    if args.cmd == "build":
        b = build_graph_from_pdfs(args.docs)
        out = args.out
        b.save_json(out)
        print(f"graph saved to {out}: {b.stats()}")
    elif args.cmd == "stats":
        b = build_graph_from_pdfs(args.docs)
        print(json.dumps(b.stats(), indent=2))
    elif args.cmd == "search":
        b = build_graph_from_pdfs(args.docs)
        results = b.search(args.query, max_depth=args.depth)
        if not results:
            print(f"no matches for {args.query!r}")
            return
        for r in results:
            print(f"\n{r['label']} ({r['type']}) — {len(r['connections'])} connections")
            for c in r["connections"][:10]:
                print(f"  {c['source']} -{c['relation']}-> {c['target']} (conf {c['confidence']})")
    elif args.cmd == "export":
        b = build_graph_from_pdfs(args.docs)
        out = args.out or f"graph.{args.format}"
        if args.format == "json":
            b.save_json(out)
        elif args.format == "graphml":
            b.export_graphml(out)
        else:
            b.export_png(out)
        print(f"exported to {out}")


if __name__ == "__main__":
    sys.exit(main())
