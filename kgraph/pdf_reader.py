"""PDF text extraction with per-page tracking (for source citations)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """One page of extracted text with its page number (1-based)."""
    page: int
    text: str


@dataclass
class Document:
    """A parsed PDF: metadata + per-page text."""
    filename: str
    path: str
    pages: List[PageText] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page_of(self, char_offset: int) -> int:
        """Map a char offset in `content` back to the 1-based page number."""
        offset = 0
        for p in self.pages:
            offset += len(p.text) + 2  # +2 for the "\n\n" separator
            if char_offset < offset:
                return p.page
        return self.page_count or 1


class PDFReader:
    """Extract text from PDFs using pdfplumber (with a PyPDF2 fallback)."""

    def __init__(self, docs_dir: str | Path = "docs"):
        self.docs_dir = Path(docs_dir)

    def list_pdfs(self) -> List[Path]:
        if not self.docs_dir.exists():
            logger.warning("docs dir not found: %s", self.docs_dir)
            return []
        return sorted(p for p in self.docs_dir.iterdir() if p.suffix.lower() == ".pdf")

    def read_pdf(self, pdf_path: str | Path) -> Optional[Document]:
        pdf_path = Path(pdf_path)
        try:
            import pdfplumber
        except ImportError:
            pdfplumber = None

        pages: List[PageText] = []
        try:
            if pdfplumber is not None:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append(PageText(page=i, text=text.strip()))
            else:
                pages = self._fallback_pypdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to read %s: %s", pdf_path.name, exc)
            return None

        if not pages:
            logger.warning("no extractable text in %s (scanned PDF?)", pdf_path.name)
            return None

        return Document(filename=pdf_path.name, path=str(pdf_path), pages=pages)

    def _fallback_pypdf(self, pdf_path: Path) -> List[PageText]:
        from pypdf import PdfReader as PyPdfReader

        reader = PyPdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(page=i, text=text))
        return pages

    def read_all(self) -> List[Document]:
        docs = []
        for p in self.list_pdfs():
            doc = self.read_pdf(p)
            if doc is not None:
                docs.append(doc)
        return docs
