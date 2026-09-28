"""Graph builder: PDFs → resolved entities + typed relations → networkx DiGraph."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import networkx as nx

from .entities import EntityExtractor, EntityResolver
from .pdf_reader import Document, PDFReader
from .relations import Relation, RelationExtractor

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Build a knowledge graph from PDF documents."""

    def __init__(self, use_spacy: bool = True):
        self.extractor = EntityExtractor(use_spacy=use_spacy)
        self.resolver = EntityResolver()
        self.relations = RelationExtractor(self.resolver)
        self.graph: nx.DiGraph = nx.DiGraph()

    @property
    def ner_backend(self) -> str:
        return self.extractor.backend

    def build_from_documents(self, documents: List[Document], cache: dict | None = None) -> nx.DiGraph:
        self.graph.clear()
        self.resolver = EntityResolver()
        self.relations = RelationExtractor(self.resolver)

        for doc in documents:
            logger.info("processing %s (%d pages)", doc.filename, doc.page_count)
            for page in doc.pages:
                self._process_page(doc, page.page, page.text)

        self._finalize()
        logger.info("graph: %d nodes, %d edges (NER=%s)", self.graph.number_of_nodes(),
                    self.graph.number_of_edges(), self.ner_backend)
        return self.graph

    def _process_page(self, doc: Document, page_num: int, text: str):
        # entities (with sentence context for disambiguation)
        for sentence in text.split("\n"):
            for mention, etype in self.extractor.extract(sentence):
                self.resolver.add_mention(mention, etype, doc.filename, page_num, context=sentence)

        # relations (resolved)
        for rel in self.relations.extract(text, filename=doc.filename, page=page_num):
            self._add_relation(rel)

    def _add_relation(self, rel: Relation):
        if rel.source == rel.target:
            return
        if not self.graph.has_node(rel.source):
            ent = self.resolver._entities.get(rel.source)
            self.graph.add_node(rel.source, type=ent.type if ent else "ENTITY",
                                label=ent.text if ent else rel.source)
        if not self.graph.has_node(rel.target):
            ent = self.resolver._entities.get(rel.target)
            self.graph.add_node(rel.target, type=ent.type if ent else "ENTITY",
                                label=ent.text if ent else rel.target)
        if self.graph.has_edge(rel.source, rel.target):
            edge = self.graph[rel.source][rel.target]
            edge["weight"] = edge.get("weight", 1) + 1
            rels = edge.get("relations", [])
            if rel.relation not in rels:
                rels.append(rel.relation)
            edge["relations"] = rels
            edge["confidence"] = max(edge.get("confidence", 0), rel.confidence)
            if rel.filename not in edge.get("sources", []):
                edge["sources"] = edge.get("sources", []) + [rel.filename]
        else:
            self.graph.add_edge(rel.source, rel.target, relation=rel.relation,
                                verb=rel.verb, weight=1, confidence=rel.confidence,
                                filename=rel.filename, page=rel.page, sentence=rel.sentence)

    def _finalize(self):
        # add resolved entities that had no relations
        for ent in self.resolver.entities():
            if not self.graph.has_node(ent.id):
                self.graph.add_node(ent.id, type=ent.type, label=ent.text)
            else:
                self.graph.nodes[ent.id]["type"] = ent.type
                self.graph.nodes[ent.id]["label"] = ent.text
            self.graph.nodes[ent.id]["mentions"] = ent.mentions
            self.graph.nodes[ent.id]["confidence"] = ent.confidence
            self.graph.nodes[ent.id]["sources"] = ent.sources
            self.graph.nodes[ent.id]["pages"] = ent.pages
            self.graph.nodes[ent.id]["aliases"] = ent.aliases
        self._retype_from_relations()

    def _retype_from_relations(self):
        """Use relation context to fix mis-typed entities.

        "founded SpaceX" → SpaceX is an ORG. "based in Hawthorne" → Hawthorne is a GPE.
        "CEO of Microsoft" → Microsoft is an ORG. "joined Microsoft" → ORG.
        """
        org_rels = {"founded", "created", "invented", "developed", "built", "designed",
                    "launched", "acquired", "merged_with", "partnered_with", "invested_in",
                    "funded", "led", "joined", "works_at", "employs", "appointed", "hired",
                    "named", "elected", "owns", "controls", "produces", "makes",
                    "manufactures", "sells", "provides", "offers", "resigned_from",
                    "left", "succeeded", "replaced"}
        gpe_rels = {"located_in", "based_in", "moved_to"}
        for u, v, d in self.graph.edges(data=True):
            rel = d.get("relation", "")
            if rel in org_rels:
                # object of these verbs is usually an ORG — but keep PERSON
                # when the object is a known person (e.g. "founded X with Paul Allen"
                # mis-parses Allen as the object)
                v_type = self.graph.nodes[v].get("type")
                v_label = self.graph.nodes[v].get("label", "").lower()
                if v_type == "PERSON" and not self._looks_like_person(v_label):
                    self.graph.nodes[v]["type"] = "ORG"
            if rel in gpe_rels:
                if self.graph.nodes[v].get("type") == "PERSON":
                    self.graph.nodes[v]["type"] = "GPE"

    @staticmethod
    def _looks_like_person(label: str) -> bool:
        """Heuristic: a two-word title-case name is usually a person, not an org."""
        words = label.split()
        if len(words) >= 2 and all(w[:1].isupper() for w in words):
            # "Tesla Motors" / "Blue Origin" are orgs — check last word
            from .entities import _ORG_KEYWORDS
            if words[-1].lower() in _ORG_KEYWORDS:
                return False
            return True
        return False

    # ── stats / search / export ───────────────────────────────────────
    def stats(self) -> Dict:
        node_types: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "OTHER")
            node_types[t] = node_types.get(t, 0) + 1
        rel_types: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation", "related_to")
            rel_types[r] = rel_types.get(r, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
            "node_types": node_types,
            "relation_types": rel_types,
            "ner_backend": self.ner_backend,
            "avg_degree": round(sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()), 2),
        }

    def analytics(self) -> Dict:
        """Centrality + community detection for the explorer."""
        g = self.graph
        if g.number_of_nodes() == 0:
            return {"centrality": {}, "communities": [], "top": []}
        try:
            deg = nx.degree_centrality(g)
            bet = nx.betweenness_centrality(g)
            close = nx.closeness_centrality(g)
        except Exception:  # noqa: BLE001
            deg = bet = close = {}
        # community detection (greedy modularity on undirected version)
        try:
            communities = list(nx.community.greedy_modularity_communities(g.to_undirected()))
        except Exception:  # noqa: BLE001
            communities = []
        top = sorted(
            [{"id": n, "label": g.nodes[n].get("label", n), "type": g.nodes[n].get("type", "OTHER"),
              "degree": round(deg.get(n, 0), 3), "betweenness": round(bet.get(n, 0), 4),
              "closeness": round(close.get(n, 0), 3)}
             for n in g.nodes],
            key=lambda x: -x["degree"],
        )[:10]
        return {
            "centrality": {n: {"degree": round(deg.get(n, 0), 3),
                               "betweenness": round(bet.get(n, 0), 4),
                               "closeness": round(close.get(n, 0), 3)} for n in g.nodes},
            "communities": [list(c) for c in communities],
            "top": top,
        }

    def search(self, query: str, max_depth: int = 2) -> List[Dict]:
        q = query.lower()
        hits = [n for n in self.graph.nodes if q in n.lower() or
                q in self.graph.nodes[n].get("label", "").lower()]
        results = []
        for node in hits:
            neighbors = {node}
            frontier = {node}
            for _ in range(max_depth):
                nxt = set()
                for n in frontier:
                    nxt.update(self.graph.successors(n))
                    nxt.update(self.graph.predecessors(n))
                neighbors.update(nxt)
                frontier = nxt
            sub = self.graph.subgraph(neighbors)
            results.append({
                "node": node,
                "label": self.graph.nodes[node].get("label", node),
                "type": self.graph.nodes[node].get("type", "OTHER"),
                "connections": [
                    {"source": u, "target": v, "relation": d.get("relation"),
                     "confidence": d.get("confidence"), "filename": d.get("filename"),
                     "page": d.get("page"), "sentence": d.get("sentence", "")}
                    for u, v, d in sub.edges(data=True)
                ],
            })
        return results

    def to_json(self) -> Dict:
        nodes = [
            {"id": n, "label": d.get("label", n), "type": d.get("type", "OTHER"),
             "mentions": d.get("mentions", 1), "confidence": d.get("confidence", 0.5),
             "sources": d.get("sources", []), "pages": d.get("pages", []),
             "aliases": d.get("aliases", [])}
            for n, d in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "relation": d.get("relation", "related_to"),
             "verb": d.get("verb", ""), "weight": d.get("weight", 1),
             "confidence": d.get("confidence", 0.5), "filename": d.get("filename", ""),
             "page": d.get("page", 1), "sentence": d.get("sentence", "")}
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"stats": self.stats(), "nodes": nodes, "edges": edges}

    def save_json(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def export_graphml(self, path: str | Path):
        nx.write_graphml(self.graph, str(path))

    def export_png(self, path: str | Path, figsize=(16, 12)):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.graph, seed=42, k=0.8)
        plt.figure(figsize=figsize)
        colors = {"PERSON": "#ff9900", "ORG": "#4fc3f7", "GPE": "#81c784",
                  "PRODUCT": "#f48fb1", "OTHER": "#b0bec5"}
        node_colors = [colors.get(self.graph.nodes[n].get("type", "OTHER"), "#b0bec5")
                       for n in self.graph.nodes]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=300, alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, arrows=True, alpha=0.4, edge_color="#888")
        nx.draw_networkx_labels(self.graph, pos, font_size=7,
                                labels={n: self.graph.nodes[n].get("label", n) for n in self.graph.nodes})
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(path), dpi=120)
        plt.close()


def build_graph_from_pdfs(docs_dir: str | Path, use_spacy: bool = True) -> KnowledgeGraphBuilder:
    reader = PDFReader(docs_dir)
    docs = reader.read_all()
    builder = KnowledgeGraphBuilder(use_spacy=use_spacy)
    builder.build_from_documents(docs)
    return builder
