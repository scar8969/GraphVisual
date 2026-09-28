"""Unit tests for the knowledge graph engine — extraction, resolution, relations."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kgraph.entities import EntityExtractor, EntityResolver
from kgraph.relations import RelationExtractor
from kgraph.pdf_reader import PDFReader, Document, PageText


# ── entity extraction ────────────────────────────────────────────────
class TestEntityExtractor:
    def test_known_orgs(self):
        ex = EntityExtractor(use_spacy=False)
        out = ex.extract("Elon Musk founded SpaceX and Tesla in California.")
        mentions = {m: t for m, t in out}
        assert mentions.get("SpaceX") == "ORG"
        assert mentions.get("Tesla") == "ORG"
        assert mentions.get("Elon Musk") == "PERSON"

    def test_org_keyword_suffix(self):
        ex = EntityExtractor(use_spacy=False)
        out = ex.extract("Tesla Motors produces electric vehicles.")
        mentions = {m: t for m, t in out}
        assert mentions.get("Tesla Motors") == "ORG"

    def test_camel_case_org(self):
        ex = EntityExtractor(use_spacy=False)
        out = ex.extract("SpaceX launched a rocket.")
        assert ("SpaceX", "ORG") in out

    def test_acronym_org(self):
        ex = EntityExtractor(use_spacy=False)
        out = ex.extract("NASA and IBM partnered.")
        mentions = {m: t for m, t in out}
        assert mentions.get("NASA") == "ORG"
        assert mentions.get("IBM") == "ORG"

    def test_no_stopword_false_positives(self):
        ex = EntityExtractor(use_spacy=False)
        out = ex.extract("The company was founded in 2002. However, it grew fast.")
        for m, _ in out:
            assert m.lower() not in {"the", "however"}


# ── entity resolution ────────────────────────────────────────────────
class TestEntityResolver:
    def test_exact_merge(self):
        r = EntityResolver()
        r.add_mention("Elon Musk", "PERSON", "a.pdf", 1)
        r.add_mention("Elon Musk", "PERSON", "b.pdf", 2)
        assert len(r.entities()) == 1
        assert r.entities()[0].mentions == 2
        assert r.entities()[0].sources == ["a.pdf", "b.pdf"]

    def test_surname_resolution(self):
        r = EntityResolver()
        r.add_mention("Elon Musk", "PERSON", "a.pdf", 1)
        assert r.canonical_id("Musk") == "elonmusk"

    def test_acronym_resolution(self):
        r = EntityResolver()
        r.add_mention("National Aeronautics and Space Administration", "ORG", "a.pdf", 1)
        assert r.canonical_id("NASA") == "nationalaeronauticsandspaceadministration"

    def test_case_insensitive(self):
        r = EntityResolver()
        r.add_mention("Elon Musk", "PERSON", "a.pdf", 1)
        assert r.canonical_id("ELON MUSK") == "elonmusk"


# ── relation extraction ──────────────────────────────────────────────
class TestRelationExtractor:
    def _setup(self, mentions):
        r = EntityResolver()
        for m, t in mentions:
            r.add_mention(m, t, "t.pdf", 1)
        return r, RelationExtractor(r)

    def test_founded(self):
        r, re_ = self._setup([("Elon Musk", "PERSON"), ("SpaceX", "ORG")])
        rels = re_.extract("Elon Musk founded SpaceX in 2002.", "t.pdf", 1)
        assert any(x.source == "elonmusk" and x.relation == "founded"
                   and x.target == "spacex" for x in rels)

    def test_acquired(self):
        r, re_ = self._setup([("Elon Musk", "PERSON"), ("Twitter", "ORG")])
        rels = re_.extract("Elon Musk acquired Twitter in 2022.", "t.pdf", 1)
        assert any(x.relation == "acquired" and x.target == "twitter" for x in rels)

    def test_surname_subject(self):
        r, re_ = self._setup([("Elon Musk", "PERSON"), ("Tesla Motors", "ORG")])
        rels = re_.extract("Musk also founded Tesla Motors.", "t.pdf", 1)
        assert any(x.source == "elonmusk" and x.target == "teslamotors" for x in rels)

    def test_compound_verbs_share_subject(self):
        r, re_ = self._setup([("Elon Musk", "PERSON"), ("Tesla Motors", "ORG"),
                              ("Twitter", "ORG")])
        rels = re_.extract("Musk founded Tesla Motors and acquired Twitter.", "t.pdf", 1)
        assert any(x.relation == "founded" and x.target == "teslamotors" for x in rels)
        assert any(x.relation == "acquired" and x.target == "twitter" for x in rels)

    def test_role_pattern(self):
        r, re_ = self._setup([("Satya Nadella", "PERSON"), ("Microsoft", "ORG")])
        rels = re_.extract("Satya Nadella is the CEO of Microsoft.", "t.pdf", 1)
        assert any(x.relation == "works_at" and x.target == "microsoft" for x in rels)

    def test_skips_with_object(self):
        r, re_ = self._setup([("Bill Gates", "PERSON"), ("Microsoft", "ORG"),
                              ("Paul Allen", "PERSON")])
        rels = re_.extract("Bill Gates co-founded Microsoft with Paul Allen.", "t.pdf", 1)
        # Allen should NOT be the object of founded
        assert not any(x.relation == "founded" and x.target == "paullallen" for x in rels)

    def test_confidence_range(self):
        r, re_ = self._setup([("Elon Musk", "PERSON"), ("SpaceX", "ORG")])
        rels = re_.extract("Elon Musk founded SpaceX.", "t.pdf", 1)
        assert all(0.3 <= x.confidence <= 0.95 for x in rels)


# ── PDF reader ───────────────────────────────────────────────────────
class TestPDFReader:
    def test_page_tracking(self):
        doc = Document(filename="t.pdf", path="t.pdf",
                       pages=[PageText(1, "page one"), PageText(2, "page two")])
        assert doc.page_of(2) == 1
        assert doc.page_of(20) == 2
        assert doc.page_count == 2
        assert "page one" in doc.content
