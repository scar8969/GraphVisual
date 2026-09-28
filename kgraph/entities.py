"""Entity extraction (spaCy optional, heuristic fallback) + alias resolution.

The original repo used spaCy NER and never resolved aliases, so "Elon Musk",
"Musk" and "Elon" became three disconnected nodes. This module:
  1. extracts entities (spaCy if installed, regex+wordlist fallback otherwise)
  2. resolves aliases → canonical entity (exact, case-folded, acronym, surname)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── heuristic NER (zero-dependency fallback) ──────────────────────────
_PERSON_TITLES = {"mr", "mrs", "ms", "dr", "prof", "ceo", "cto", "coo", "founder",
                  "president", "chairman", "minister", "senator", "governor"}
_ORG_KEYWORDS = {"inc", "corp", "corporation", "ltd", "llc", "gmbh", "co", "company",
                 "university", "college", "institute", "laboratory", "lab", "group",
                 "association", "foundation", "agency", "department", "ministry",
                 "bank", "fund", "hospital", "school", "studio", "startup",
                 "motors", "origin", "foods", "systems", "technologies", "technology",
                 "labs", "industries", "airlines", "energy", "robotics", "dynamics",
                 "works", "networks", "software", "hardware", "solutions", "services",
                 "holdings", "aerospace", "automotive", "pharmaceuticals", "biotech",
                 "telecom", "communications", "entertainment", "media", "retail",
                 "manufacturing", "engineering", "consulting", "analytics", "research"}
_GPE_SUFFIXES = {"city", "county", "province", "state", "kingdom", "republic",
                 "empire", "valley", "island", "coast", "bay", "desert"}

# known single-word orgs (not title-case, not acronyms)
_KNOWN_ORGS = {"spacex", "tesla", "amazon", "microsoft", "twitter", "google", "apple",
               "meta", "facebook", "netflix", "nvidia", "intel", "oracle", "salesforce",
               "uber", "lyft", "airbnb", "spotify", "snap", "tiktok", "tencent", "alibaba",
               "samsung", "sony", "nokia", "ericsson", "qualcomm", "amd", "ibm", "hp",
               "dell", "cisco", "vmware", "redhat", "slack", "zoom", "stripe", "square",
               "paypal", "shopify", "dropbox", "box", "hubspot", "workday", "servicenow",
               "palantir", "databricks", "snowflake", "cloudflare", "okta", "twilio",
               "sendgrid", "mailchimp", "wework", "robinhood", "coinbase", "binance",
               "ethereum", "bitcoin", "blockchain", "openai", "anthropic", "deepmind",
               "waymo", "rivian", "lucid", "nio", "bytedance", "huawei", "xiaomi",
               "oppo", "vivo", "oneplus", "lenovo", "asus", "acer", "msi", "gigabyte"}

# title-case sequences (people, orgs, places)
_TITLE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
# single capitalized word that could be an org/place (e.g. "SpaceX", "Tesla")
_SINGLE_CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
# camelCase orgs like SpaceX, PayPal, OpenSea, DeepMind, Canva, Reddit
_CAMEL_ORG_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


@dataclass
class Entity:
    """A resolved entity node."""
    id: str                      # canonical id (resolved)
    text: str                    # display text (canonical form)
    type: str                    # PERSON | ORG | GPE | PRODUCT | OTHER
    aliases: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)   # filenames
    pages: List[int] = field(default_factory=list)     # page numbers
    mentions: int = 0
    confidence: float = 0.5


class EntityExtractor:
    """Extract raw entity mentions from text."""

    def __init__(self, use_spacy: bool = True):
        self._nlp = None
        if use_spacy:
            try:
                import spacy  # noqa: F401
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except Exception:  # noqa: BLE001
                self._nlp = None

    @property
    def backend(self) -> str:
        return "spacy" if self._nlp is not None else "heuristic"

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Return [(mention, type)] raw mentions."""
        if self._nlp is not None:
            return self._extract_spacy(text)
        return self._extract_heuristic(text)

    # ── spaCy path ────────────────────────────────────────────────────
    def _extract_spacy(self, text: str) -> List[Tuple[str, str]]:
        assert self._nlp is not None
        doc = self._nlp(text[:500_000])
        out = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "LOC", "NORP"}:
                out.append((ent.text.strip(), ent.label_))
        return out

    # ── heuristic path (no downloads) ─────────────────────────────────
    def _extract_heuristic(self, text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        seen = set()

        def add(mention: str, etype: str):
            m = mention.strip().strip(".,;:()\"'")
            if len(m) < 2 or len(m) > 60:
                return
            if m.lower() in {"the", "this", "that", "these", "those", "it", "he", "she", "they", "we", "you", "i", "and", "or", "but", "of", "in", "on", "at", "by", "for", "with", "from", "to", "as", "is", "was", "are", "were", "be", "been"}:
                return
            key = (m, etype)
            if key not in seen:
                seen.add(key)
                out.append((m, etype))

        # title-case sequences (people, orgs, places)
        for m in _TITLE_RE.findall(text):
            words = m.split()
            # skip sentence-start false positives: "The", "This", "However"...
            if words[0].lower() in {"the", "this", "that", "these", "those", "however", "therefore", "although", "because", "while", "when", "where", "after", "before", "during", "since", "although", "moreover", "furthermore", "thus", "hence", "also", "then", "there", "here", "it", "its", "his", "her", "their", "our", "your", "my", "we", "they", "he", "she", "i", "you", "and", "or", "but", "if", "as", "at", "by", "for", "from", "in", "of", "on", "to", "with"}:
                continue
            # last word determines type
            last = words[-1].lower()
            if last in _ORG_KEYWORDS or any(w.lower() in _ORG_KEYWORDS for w in words):
                add(m, "ORG")
            elif last in _GPE_SUFFIXES or m in {"United States", "United Kingdom", "New York", "San Francisco", "Los Angeles", "Washington", "London", "Paris", "Berlin", "Tokyo", "Beijing", "Delhi", "Mumbai", "India", "China", "Japan", "Germany", "France", "UK", "USA"}:
                add(m, "GPE")
            else:
                add(m, "PERSON")

        # acronyms (orgs)
        for m in _ACRONYM_RE.findall(text):
            if m in {"I", "II", "III", "IV", "VI", "VII", "VIII", "IX", "X", "AM", "PM", "OK", "US", "UK", "EU", "UN", "CEO", "CTO", "CFO", "COO", "AI", "ML", "API", "PDF", "HTML", "CSS", "JS", "HTTP", "HTTPS", "URL", "SQL", "DB", "PC", "TV", "DNA", "RNA", "COVID", "GPS", "LED", "USB", "VPN", "WIFI", "WWW"}:
                continue
            if len(m) >= 2:
                add(m, "ORG")

        # single capitalized words: known orgs always; unknown only if they
        # appear near org-context verbs ("founded X", "acquired X", "CEO of X")
        for m in _SINGLE_CAP_RE.findall(text):
            ml = m.lower()
            if ml in _KNOWN_ORGS:
                add(m, "ORG")
            elif ml in {"the", "this", "that", "these", "those", "however", "therefore",
                        "although", "because", "while", "when", "where", "after", "before",
                        "during", "since", "moreover", "furthermore", "thus", "hence",
                        "also", "then", "there", "here", "it", "its", "his", "her", "their",
                        "our", "your", "my", "we", "they", "he", "she", "i", "you", "and",
                        "or", "but", "if", "as", "at", "by", "for", "from", "in", "of", "on",
                        "to", "with", "not", "no", "yes", "new", "old", "first", "last",
                        "next", "one", "two", "three", "four", "five", "ten", "twenty",
                        "thirty", "forty", "fifty", "hundred", "thousand", "million",
                        "billion", "january", "february", "march", "april", "may", "june",
                        "july", "august", "september", "october", "november", "december",
                        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
                        "sunday", "north", "south", "east", "west", "central", "american",
                        "european", "asian", "african", "chinese", "japanese", "indian",
                        "german", "french", "british", "canadian", "australian"}:
                continue
            else:
                # context: capitalized word right after an org-context verb
                ctx = re.search(r"(?:founded|acquired|launched|joined|led|runs|heads|owns|bought|sold|invested in|CEO of|CTO of|CFO of|COO of|president of|chairman of|head of)\s+([A-Z][a-z]{2,})\b", text)
                if ctx and ctx.group(1) == m:
                    add(m, "ORG")

        # camelCase orgs (SpaceX, PayPal, DeepMind, OpenSea...)
        for m in _CAMEL_ORG_RE.findall(text):
            ml = m.lower()
            if ml in _KNOWN_ORGS or ml.endswith(("x", "ai", "io", "ly", "fy", "hub", "lab", "labs")):
                add(m, "ORG")

        # GPE: "based in Hawthorne, California" / "headquartered in Austin, Texas"
        for m in re.findall(r"\b([A-Z][a-z]+),\s*([A-Z]{2})\b", text):
            add(f"{m[0]}, {m[1]}", "GPE")
            add(m[0], "GPE")
        # GPE: "in <City>" after based/located/headquartered
        for m in re.findall(r"(?:based|located|headquartered)\s+in\s+([A-Z][a-z]+)\b", text):
            add(m, "GPE")

        return out


class EntityResolver:
    """Merge mentions into canonical entities via alias rules."""

    def __init__(self, disambiguate: bool = True):
        self._canonical: Dict[str, str] = {}   # alias → canonical id
        self._entities: Dict[str, Entity] = {}  # canonical id → Entity
        self._disambiguate = disambiguate
        # context signature per canonical id: set of nearby content words
        self._contexts: Dict[str, set] = {}
        self._context_window = 6  # words around the mention

    # ── normalization ─────────────────────────────────────────────────
    @staticmethod
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9]", "", s.lower())

    @staticmethod
    def _surname(s: str) -> str:
        parts = s.split()
        return parts[-1] if parts else s

    def _acronym(self, s: str) -> str:
        parts = s.split()
        if len(parts) < 2:
            return ""
        skip = {"and", "of", "the", "for", "in", "on", "at", "by", "with", "a", "an"}
        return "".join(p[0] for p in parts if p and p[0].isalpha() and p.lower() not in skip).upper()

    # ── resolution ────────────────────────────────────────────────────
    def add_mention(self, mention: str, etype: str, source: str, page: int, confidence: float = 0.5,
                    context: str = ""):
        """Register a raw mention, resolving it to a canonical entity.

        `context` is the surrounding sentence — used to disambiguate same-name
        entities (two "Apple"s with different contexts become different nodes).
        """
        mention = mention.strip()
        if not mention:
            return None

        key = self._norm(mention)
        if not key:
            return None

        canonical = self._resolve(mention, key, etype, context)
        ent = self._entities.setdefault(
            canonical,
            Entity(id=canonical, text=mention, type=etype),
        )
        # keep the most common display form
        if mention not in ent.aliases:
            ent.aliases.append(mention)
        if source not in ent.sources:
            ent.sources.append(source)
        if page not in ent.pages:
            ent.pages.append(page)
        ent.mentions += 1
        ent.confidence = max(ent.confidence, confidence)
        # merge context signature
        if self._disambiguate and context:
            sig = self._context_words(context, mention)
            self._contexts.setdefault(canonical, set()).update(sig)
        return ent

    def _context_words(self, context: str, mention: str) -> set:
        """Extract content words around the mention (excluding the mention itself)."""
        words = _WORD_RE.findall(context.lower())
        try:
            idx = words.index(mention.lower().split()[0])
        except ValueError:
            return set()
        lo = max(0, idx - self._context_window)
        hi = min(len(words), idx + 1 + self._context_window)
        stop = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "by",
                "for", "with", "from", "to", "as", "is", "was", "were", "be", "been",
                "he", "she", "it", "they", "we", "you", "i", "his", "her", "their",
                "our", "my", "this", "that", "these", "those", "also", "then", "there",
                "here", "not", "no", "yes", "had", "has", "have", "are", "will", "would",
                "could", "should", "may", "might", "must", "said", "says", "say"}
        return {w for w in words[lo:hi] if w not in stop and len(w) > 2}

    def _resolve(self, mention: str, key: str, etype: str, context: str = "") -> str:
        """Find the canonical id for a mention, creating one if needed."""
        if key in self._canonical:
            return self._canonical[key]

        # exact alias already known?
        for alias, canon in self._canonical.items():
            if key == alias:
                return canon

        # fuzzy match: near-duplicate aliases ("Elon Reeve Musk" ≈ "Elon Musk")
        if len(key) >= 6:
            for alias, canon in self._canonical.items():
                ent = self._entities.get(canon)
                if ent and ent.type == etype and self._similar(key, alias) >= 0.85:
                    self._canonical[key] = canon
                    return canon

        # surname match: "Musk" → "Elon Musk" (same type)
        if etype == "PERSON":
            surname = self._surname(mention)
            if len(surname) >= 4:
                for alias, canon in self._canonical.items():
                    if self._surname(alias) == surname and self._entities[canon].type == "PERSON":
                        # disambiguation: if contexts are very different, don't merge
                        if self._disambiguate and self._contexts.get(canon) and context:
                            overlap = len(self._contexts[canon] & self._context_words(context, mention))
                            if overlap == 0 and self._contexts[canon]:
                                continue
                        self._canonical[key] = canon
                        return canon

        # acronym match: "NASA" → "National Aeronautics and Space Administration"
        acro = self._acronym(mention)
        if len(acro) >= 2:
            for alias, canon in self._canonical.items():
                if self._acronym(alias) == acro and self._entities[canon].type == etype:
                    self._canonical[key] = canon
                    return canon

        # new canonical
        self._canonical[key] = key
        return key

    @staticmethod
    def _similar(a: str, b: str) -> float:
        """Levenshtein similarity in [0, 1]."""
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        # prefix/suffix containment counts as similar ("Elon Reeve Musk" ⊃ "Elon Musk")
        if a in b or b in a:
            return 0.9
        # simple Levenshtein
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev = dp[0]
            dp[0] = i
            for j, cb in enumerate(b, 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
                prev = cur
        dist = dp[-1]
        return 1.0 - dist / max(len(a), len(b))

    def entities(self) -> List[Entity]:
        return list(self._entities.values())

    def canonical_id(self, mention: str) -> str:
        key = self._norm(mention)
        if key in self._canonical:
            return self._canonical[key]
        # surname resolution: "Musk" → "Elon Musk" if a person with that surname exists
        if " " not in mention:
            m_lower = mention.lower()
            for alias, canon in self._canonical.items():
                ent = self._entities.get(canon)
                if ent and ent.type == "PERSON" and self._surname(ent.text).lower() == m_lower:
                    self._canonical[key] = canon
                    return canon
            # acronym resolution: "NASA" → full name if acronym matches
            if len(mention) >= 2 and mention.isupper():
                for alias, canon in self._canonical.items():
                    ent = self._entities.get(canon)
                    if ent and self._acronym(ent.text) == mention:
                        self._canonical[key] = canon
                        return canon
        return key
