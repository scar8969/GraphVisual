"""Relation extraction: verb→type mapping with confidence + source citations.

The original repo grabbed raw token text (subject/verb/object) with no typing,
no confidence, and no page citations. This module:
  1. finds subject-verb-object triples via dependency-ish heuristics
  2. maps the verb to a relation type (founded, acquired, works_at, ...)
  3. scores confidence (verb strength, entity types, distance)
  4. attaches source (filename, page, sentence)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# verb → relation type
_VERB_TYPES: Dict[str, str] = {
    "found": "founded", "co-found": "founded", "co-founded": "founded", "establish": "founded",
    "create": "created", "invent": "invented", "develop": "developed",
    "build": "built", "design": "designed", "launch": "launched",
    "acquire": "acquired", "buy": "acquired", "purchase": "acquired",
    "merge": "merged_with", "partner": "partnered_with",
    "invest": "invested_in", "fund": "funded", "back": "funded",
    "lead": "led", "run": "led", "head": "led", "manage": "led",
    "join": "joined", "work": "works_at", "employ": "employs",
    "appoint": "appointed", "hire": "hired", "fire": "fired",
    "name": "named", "elect": "elected",
    "locate": "located_in", "base": "based_in", "move": "moved_to",
    "headquarter": "based_in", "headquartered": "based_in",
    "co-found": "founded", "co-founded": "founded",
    "own": "owns", "control": "controls",
    "produce": "produces", "make": "makes", "manufacture": "manufactures",
    "sell": "sells", "provide": "provides", "offer": "offers",
    "win": "won", "award": "awarded", "receive": "received",
    "publish": "published", "write": "wrote", "author": "authored",
    "study": "studied", "research": "researched",
    "say": "said", "state": "stated", "announce": "announced",
    "become": "became", "turn": "became",
    "replace": "replaced", "succeed": "succeeded",
    "resign": "resigned_from", "leave": "left", "quit": "left",
    "defeat": "defeated", "beat": "defeated", "win_against": "defeated",
}

# strong verbs → higher confidence
_STRONG_VERBS = {"found", "co-found", "establish", "create", "invent", "acquire",
                 "merge", "invest", "fund", "appoint", "hire", "elect", "locate",
                 "base", "own", "produce", "manufacture", "publish", "author",
                 "defeat", "resign", "succeed", "replace"}

_PRONOUNS = {"he", "she", "it", "they", "we", "you", "i", "his", "her", "their",
             "our", "my", "this", "that", "these", "those", "there", "here",
             "who", "which", "what", "one", "someone", "something", "everyone"}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


@dataclass
class Relation:
    """A typed, cited relation between two resolved entities."""
    source: str          # canonical id
    target: str          # canonical id
    relation: str        # typed relation (founded, acquired, ...)
    verb: str            # raw verb
    confidence: float
    filename: str
    page: int
    sentence: str = ""


class RelationExtractor:
    def __init__(self, resolver, use_spacy: bool = True):  # noqa: ANN001
        self.resolver = resolver
        self._nlp = None
        if use_spacy:
            try:
                import spacy  # noqa: F401
                # full pipeline (parser enabled) for dependency relations
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:  # noqa: BLE001
                self._nlp = None

    @staticmethod
    def _type_for(verb: str) -> str:
        v = verb.lower().strip()
        # handle "was founded", "is based", "has acquired"
        for prefix in ("was ", "is ", "has ", "have ", "had ", "were ", "are "):
            if v.startswith(prefix):
                v = v[len(prefix):]
        if v in _VERB_TYPES:
            return _VERB_TYPES[v]
        # strip common suffixes, trying several stems
        for stem in (v[:-3], v[:-2], v[:-1], v + "e"):
            if stem in _VERB_TYPES:
                return _VERB_TYPES[stem]
        return "related_to"

    @staticmethod
    def _confidence(verb: str, subj_type: str, obj_type: str) -> float:
        base = 0.55
        if verb.lower().strip() in _STRONG_VERBS:
            base += 0.2
        if subj_type == obj_type:
            base -= 0.1
        if subj_type == "PERSON" and obj_type == "ORG":
            base += 0.1
        return round(min(0.95, max(0.3, base)), 2)

    def extract(self, text: str, filename: str = "", page: int = 1) -> List[Relation]:
        """Extract typed relations from text. Returns resolved relations."""
        if self._nlp is not None:
            rels = self._extract_spacy(text, filename, page)
            # fall back to heuristics for sentences the parser missed
            rels += self._extract_heuristic(text, filename, page)
            return rels
        return self._extract_heuristic(text, filename, page)

    def _extract_heuristic(self, text: str, filename: str, page: int) -> List[Relation]:
        out: List[Relation] = []
        sentences = _SENT_SPLIT.split(text)
        for sent in sentences[:200]:
            rels = self._extract_sentence(sent, filename, page)
            out.extend(rels)
        return out

    def _extract_spacy(self, text: str, filename: str, page: int) -> List[Relation]:
        """Dependency-parse relations: nsubj → verb → dobj/attr/prep."""
        assert self._nlp is not None
        doc = self._nlp(text[:500_000])
        out: List[Relation] = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ not in ("nsubj", "nsubjpass"):
                    continue
                verb = token.head
                # find the object: dobj, attr, or prep→pobj
                obj = None
                for child in verb.children:
                    if child.dep_ in ("dobj", "attr"):
                        obj = child
                        break
                if obj is None:
                    for child in verb.children:
                        if child.dep_ == "prep":
                            for gc in child.children:
                                if gc.dep_ == "pobj":
                                    obj = gc
                                    break
                            if obj:
                                break
                if obj is None:
                    continue
                subj_text = self._span_text(token)
                obj_text = self._span_text(obj)
                if not subj_text or not obj_text or len(subj_text) < 2 or len(obj_text) < 2:
                    continue
                subj_id = self.resolver.canonical_id(subj_text)
                obj_id = self.resolver.canonical_id(obj_text)
                if subj_id == obj_id:
                    continue
                if subj_id not in self.resolver._entities or obj_id not in self.resolver._entities:
                    continue
                rtype = self._type_for(verb.text)
                if rtype == "related_to":
                    continue
                conf = self._confidence(verb.text, self.resolver._entities[subj_id].type,
                                        self.resolver._entities[obj_id].type)
                out.append(Relation(source=subj_id, target=obj_id, relation=rtype,
                                    verb=verb.text.lower(), confidence=conf,
                                    filename=filename, page=page,
                                    sentence=sent.text.strip()[:300]))
        return out

    def _span_text(self, token) -> str:
        """Full span text for a token (handles multi-word entities)."""
        parts = [token.text]
        # include compound nouns ("Tesla Motors", "Bill Gates")
        for child in token.children:
            if child.dep_ == "compound" and child.i < token.i:
                parts.insert(0, child.text)
        return " ".join(parts)

    def _extract_sentence(self, sent: str, filename: str, page: int) -> List[Relation]:
        # find entity mentions in the sentence (raw)
        mentions = self.resolver_mentions(sent)
        if len(mentions) < 2:
            return []

        rels: List[Relation] = []
        # co-founder pattern: "X co-founded Y with Z" → Z founded Y too
        co = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+co-founded\s+([A-Z][a-zA-Z]+)\s+with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", sent)
        if co:
            subj = self.resolver.canonical_id(co.group(1))
            org = self.resolver.canonical_id(co.group(2))
            partner = self.resolver.canonical_id(co.group(3))
            for s, o in ((subj, org), (partner, org)):
                if s in self.resolver._entities and o in self.resolver._entities and s != o:
                    rels.append(Relation(source=s, target=o, relation="founded",
                                         verb="co-founded", confidence=0.7,
                                         filename=filename, page=page,
                                         sentence=sent.strip()[:300]))
        # role pattern: "X is the CEO of Y" / "X, CEO of Y" / "X is president of Y"
        role = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is\s+)?(?:the\s+)?(CEO|CTO|CFO|COO|president|chairman|founder|head|director|manager|lead|chief)\s+of\s+([A-Z][a-zA-Z]+)\b", sent)
        if role:
            subj = self.resolver.canonical_id(role.group(1))
            obj = self.resolver.canonical_id(role.group(3))
            if subj in self.resolver._entities and obj in self.resolver._entities and subj != obj:
                rtype = "founded" if role.group(2).lower() == "founder" else "works_at"
                rels.append(Relation(source=subj, target=obj, relation=rtype,
                                     verb=role.group(2).lower(), confidence=0.8,
                                     filename=filename, page=page, sentence=sent.strip()[:300]))

        words = _WORD_RE.findall(sent)
        # sentence subject: first PERSON mention before the first verb
        subject = None
        for i, w in enumerate(words):
            if w.lower() in _VERB_TYPES or self._type_for(w.lower()) != "related_to":
                break
            for cid, etype in mentions:
                if etype == "PERSON" and self._mention_text(sent, cid) in " ".join(words[:i+1]):
                    subject = (cid, etype)
        for i in range(len(words) - 1):
            verb = words[i].lower()
            if verb in _VERB_TYPES or self._type_for(verb) != "related_to":
                # find subject before verb, object after
                subj = subject if subject else self._nearest_mention(sent, words, i, mentions, before=True)
                obj = self._nearest_mention(sent, words, i, mentions, before=False)
                # for compound verbs ("founded X and acquired Y"), reuse the subject
                if subj is None:
                    subj = subject
                if subj and obj and subj != obj:
                    # skip "founded X with Paul Allen" → Allen is not the object
                    if self._after_with(sent, words, i, obj):
                        continue
                    rtype = self._type_for(verb)
                    conf = self._confidence(verb, subj[1], obj[1])
                    rels.append(Relation(
                        source=subj[0], target=obj[0], relation=rtype,
                        verb=verb, confidence=conf, filename=filename,
                        page=page, sentence=sent.strip()[:300],
                    ))
        return rels

    def _after_with(self, sent: str, words: List[str], verb_idx: int,
                    obj: Tuple[str, str]) -> bool:
        """True if the object mention appears after a 'with'/'and' following the verb."""
        m_text = self._mention_text(sent, obj[0])
        try:
            m_word_idx = words.index(m_text.split()[0]) if m_text.split() else -1
        except ValueError:
            m_word_idx = -1
        if m_word_idx < 0:
            return False
        for i in range(verb_idx + 1, m_word_idx):
            if words[i].lower() in {"with", "and"}:
                return True
        return False

    def resolver_mentions(self, sent: str) -> List[Tuple[str, str]]:
        """Find (canonical_id, type) mentions in a sentence via the resolver."""
        from .entities import (_TITLE_RE, _ACRONYM_RE, _ORG_KEYWORDS,
                               _SINGLE_CAP_RE, _KNOWN_ORGS, _CAMEL_ORG_RE)

        out: List[Tuple[str, str]] = []
        seen = set()

        def add(cid: str, etype: str):
            if cid not in seen:
                seen.add(cid)
                out.append((cid, etype))

        for m in _TITLE_RE.findall(sent):
            words = m.split()
            if words[0].lower() in {"the", "this", "that", "however", "therefore", "although", "because", "while", "when", "where", "after", "before", "during", "since", "moreover", "furthermore", "thus", "hence", "also", "then", "there", "here", "it", "its", "his", "her", "their", "our", "your", "my", "we", "they", "he", "she", "i", "you", "and", "or", "but", "if", "as", "at", "by", "for", "from", "in", "of", "on", "to", "with"}:
                continue
            last = words[-1].lower()
            etype = "ORG" if (last in _ORG_KEYWORDS or any(w.lower() in _ORG_KEYWORDS for w in words)) else "PERSON"
            add(self.resolver.canonical_id(m), etype)
        for m in _ACRONYM_RE.findall(sent):
            if m in {"I", "II", "III", "IV", "VI", "VII", "VIII", "IX", "X", "AM", "PM", "OK", "US", "UK", "EU", "UN", "CEO", "CTO", "CFO", "COO", "AI", "ML", "API", "PDF", "HTML", "CSS", "JS", "HTTP", "HTTPS", "URL", "SQL", "DB", "PC", "TV", "DNA", "RNA", "COVID", "GPS", "LED", "USB", "VPN", "WIFI", "WWW"}:
                continue
            add(self.resolver.canonical_id(m), "ORG")
        for m in _SINGLE_CAP_RE.findall(sent):
            ml = m.lower()
            if ml in _KNOWN_ORGS:
                add(self.resolver.canonical_id(m), "ORG")
            elif ml not in {"the", "this", "that", "however", "therefore", "although",
                            "because", "while", "when", "where", "after", "before",
                            "during", "since", "moreover", "furthermore", "thus", "hence",
                            "also", "then", "there", "here", "it", "its", "his", "her",
                            "their", "our", "your", "my", "we", "they", "he", "she", "i",
                            "you", "and", "or", "but", "if", "as", "at", "by", "for", "from",
                            "in", "of", "on", "to", "with", "not", "no", "yes", "new", "old",
                            "first", "last", "next", "one", "two", "three", "four", "five",
                            "ten", "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
                            "million", "billion", "january", "february", "march", "april",
                            "may", "june", "july", "august", "september", "october", "november",
                            "december", "monday", "tuesday", "wednesday", "thursday", "friday",
                            "saturday", "sunday", "north", "south", "east", "west", "central",
                            "american", "european", "asian", "african", "chinese", "japanese",
                            "indian", "german", "french", "british", "canadian", "australian"}:
                # surname mention ("Musk") → resolve to known person if possible
                cid = self.resolver.canonical_id(m)
                if cid == m.lower() and self.resolver._entities.get(cid):
                    # canonical_id already resolves via surname rule if registered
                    pass
                if cid in self.resolver._entities and self.resolver._entities[cid].type == "PERSON":
                    add(cid, "PERSON")
        for m in _CAMEL_ORG_RE.findall(sent):
            ml = m.lower()
            if ml in _KNOWN_ORGS or ml.endswith(("x", "ai", "io", "ly", "fy", "hub", "lab", "labs")):
                add(self.resolver.canonical_id(m), "ORG")
        return out

    def _nearest_mention(self, sent: str, words: List[str], verb_idx: int,
                         mentions: List[Tuple[str, str]], before: bool) -> Optional[Tuple[str, str]]:
        """Find the mention closest to the verb on the given side (by word index)."""
        best = None
        best_dist = 1e9
        for cid, etype in mentions:
            m_text = self._mention_text(sent, cid)
            # find the word index of this mention (first word of the alias)
            try:
                m_word_idx = words.index(m_text.split()[0]) if m_text.split() else -1
            except ValueError:
                m_word_idx = -1
            if m_word_idx < 0:
                continue
            dist = verb_idx - m_word_idx if before else m_word_idx - verb_idx
            if dist > 0 and dist < best_dist:
                best_dist = dist
                best = (cid, etype)
        return best

    def _mention_text(self, sent: str, cid: str) -> str:
        # find the alias text in the sentence that resolves to cid
        ent = self.resolver._entities.get(cid)
        if ent is not None:
            for alias in ent.aliases:
                if alias in sent:
                    return alias
            # surname fallback: "Musk" in sentence → "Elon Musk" entity
            if ent.type == "PERSON" and " " in ent.text:
                surname = ent.text.split()[-1]
                if surname in sent:
                    return surname
        return cid
