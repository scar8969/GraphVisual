"""PDF → knowledge graph builder with entity resolution, typed relations, and citations.

Pipeline:
    PDF → text (per page) → entities (NER) → resolve aliases → relations (verb→type)
    → networkx DiGraph → JSON/GraphML/PNG export → interactive web explorer
"""
__version__ = "1.0.0"

from .graph_builder import KnowledgeGraphBuilder, build_graph_from_pdfs
from .entities import EntityExtractor, EntityResolver
from .relations import RelationExtractor
from .pdf_reader import PDFReader

__all__ = [
    "KnowledgeGraphBuilder",
    "build_graph_from_pdfs",
    "EntityExtractor",
    "EntityResolver",
    "RelationExtractor",
    "PDFReader",
]
