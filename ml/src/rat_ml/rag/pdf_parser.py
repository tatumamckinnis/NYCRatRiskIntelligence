"""Shared PDF parsing utilities for RAG corpus ingest (T-37).

Public API
----------
parse_pdf(path)                 -> list[str]       (pages of text)
parse_legal_hierarchy(text)     -> list[LegalSection]
chunk_section(section, ...)     -> list[Chunk]
build_contextual_prefix(...)    -> str
extract_defined_terms(text)     -> dict[str, str]
extract_cross_refs(text)        -> list[str]
version_hash(text)              -> str             (SHA-256 hex)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LegalSection:
    citation: str           # e.g. "§151.02(a)(1)"
    title: str              # heading text (may be empty)
    content: str            # full text of this section
    depth: int              # 0 = top, 1 = subsection, 2 = paragraph, …
    children: list["LegalSection"] = field(default_factory=list)


@dataclass
class Chunk:
    citation: str
    content: str
    content_with_prefix: str
    token_count: int
    version_hash: str
    parent_citation: str | None = None
    defined_terms: dict[str, str] = field(default_factory=dict)
    cross_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # noqa: PLC0415
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        # Fallback: rough word-based estimate (1 word ≈ 1.3 tokens)
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------

def parse_pdf(path: str | Path) -> list[str]:
    """Return a list of page-text strings from *path*.

    Tries pdfplumber first; falls back to pypdf.  Returns empty list if both fail.
    """
    path = Path(path)
    if not path.exists():
        log.warning("PDF not found: %s", path)
        return []

    # --- pdfplumber (preferred) ---
    try:
        import pdfplumber  # noqa: PLC0415
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        if any(pages):
            return pages
    except Exception as exc:  # noqa: BLE001
        log.debug("pdfplumber failed for %s: %s", path, exc)

    # --- pypdf fallback ---
    try:
        from pypdf import PdfReader  # noqa: PLC0415
        reader = PdfReader(path)
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        log.warning("pypdf fallback also failed for %s: %s", path, exc)
        return []


# ---------------------------------------------------------------------------
# Legal hierarchy parsing
# ---------------------------------------------------------------------------

# Matches primary section markers like §151.02, §27-2017, §81.23
_SECTION_RE = re.compile(
    r"^(?P<marker>§\s*\d+[\.\-]\d+[\w\.]*)"
    r"(?:\s+(?P<title>[^\n]{0,120}))?",
    re.MULTILINE,
)
# Sub-section markers: (a), (1), (i)
_SUBSEC_RE = re.compile(r"^\s*\(([a-z]|\d+|i{1,3}v?|vi{0,3})\)\s+", re.MULTILINE)


def parse_legal_hierarchy(text: str) -> list[LegalSection]:
    """Parse *text* into a flat list of LegalSection objects.

    Top-level sections are identified by the ``§N.N`` pattern; immediate
    children are inferred from ``(a)``, ``(1)`` paragraph markers within each
    section body.
    """
    sections: list[LegalSection] = []
    matches = list(_SECTION_RE.finditer(text))

    for i, m in enumerate(matches):
        citation = m.group("marker").replace(" ", "")
        title = (m.group("title") or "").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        top = LegalSection(citation=citation, title=title, content=body, depth=0)

        # Extract sub-sections
        sub_matches = list(_SUBSEC_RE.finditer(body))
        for j, sm in enumerate(sub_matches):
            sub_label = sm.group(1)
            sub_citation = f"{citation}({sub_label})"
            sub_start = sm.end()
            sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(body)
            sub_body = body[sub_start:sub_end].strip()
            child = LegalSection(
                citation=sub_citation,
                title="",
                content=sub_body,
                depth=1,
            )
            top.children.append(child)

        sections.append(top)

    if not sections:
        # No section markers found — treat the whole text as one chunk
        sections.append(
            LegalSection(citation="§(full)", title="", content=text.strip(), depth=0)
        )

    return sections


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_section(
    section: LegalSection,
    *,
    max_tokens: int = 600,
    overlap_pct: float = 0.12,
) -> list[tuple[str, str]]:
    """Return ``(citation, text)`` pairs from *section*.

    Strategy:
    - If the section has children (sub-sections), yield each child whose token
      count ≤ max_tokens; split longer children with sliding-window.
    - If the section has no children, split the body with sliding-window.
    - Parent sections (> max_tokens) are yielded at a larger 1500-token budget
      for parent-chunk expansion context.

    Returns ``(citation, text)`` pairs.
    """
    results: list[tuple[str, str]] = []
    overlap = max(1, int(max_tokens * overlap_pct))

    def _sliding(citation: str, text: str, budget: int) -> list[tuple[str, str]]:
        """Split *text* into overlapping chunks of *budget* tokens."""
        words = text.split()
        if not words:
            return []
        # Approximate: 1 token ≈ 0.75 words (rough inverse)
        words_per_chunk = max(1, int(budget * 0.75))
        words_overlap = max(0, int(overlap * 0.75))
        chunks = []
        start = 0
        part = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = " ".join(words[start:end])
            suffix = f"(part {part + 1})" if part > 0 else ""
            chunks.append((f"{citation}{suffix}", chunk_text))
            if end == len(words):
                break
            start = end - words_overlap
            part += 1
        return chunks

    if section.children:
        for child in section.children:
            if _count_tokens(child.content) <= max_tokens:
                results.append((child.citation, child.content))
            else:
                results.extend(_sliding(child.citation, child.content, max_tokens))
        # Also emit the parent at a larger budget for parent expansion
        parent_text = f"{section.citation} {section.title}\n{section.content}".strip()
        if _count_tokens(parent_text) > max_tokens:
            results.append((f"{section.citation}(parent)", parent_text[:4000]))
    else:
        body = section.content
        if _count_tokens(body) <= max_tokens:
            results.append((section.citation, body))
        else:
            results.extend(_sliding(section.citation, body, max_tokens))

    return [r for r in results if r[1].strip()]


# ---------------------------------------------------------------------------
# Contextual prefix
# ---------------------------------------------------------------------------

def build_contextual_prefix(
    authority: str,
    document: str,
    citation: str,
    content: str,
) -> str:
    """Return ``"From <authority> <document> <citation>: <content>"``."""
    return f"From {authority} {document} {citation}: {content}"


# ---------------------------------------------------------------------------
# Defined terms & cross-references
# ---------------------------------------------------------------------------

# Matches: "term" means ... (up to 200 chars)
_DEFINED_TERM_RE = re.compile(
    r'"(?P<term>[^"]{2,60})"\s+(?:means?|shall\s+mean)\s+(?P<def>[^\.]{5,200})',
    re.IGNORECASE,
)

# Matches cross-references like §151.02, §27-2017, Article 3, Title 24
_CROSS_REF_RE = re.compile(
    r"(?:§\s*\d+[\.\-]\w+(?:\([a-z\d]+\))*|Article\s+\d+|Title\s+\d+)",
    re.IGNORECASE,
)


def extract_defined_terms(text: str) -> dict[str, str]:
    """Return ``{term: definition}`` pairs found in *text*."""
    return {
        m.group("term").lower(): m.group("def").strip()
        for m in _DEFINED_TERM_RE.finditer(text)
    }


def extract_cross_refs(text: str) -> list[str]:
    """Return a deduplicated list of cross-reference strings found in *text*."""
    return list(dict.fromkeys(m.group(0) for m in _CROSS_REF_RE.finditer(text)))


# ---------------------------------------------------------------------------
# Version hash
# ---------------------------------------------------------------------------

def version_hash(text: str) -> str:
    """Return a 64-character SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# High-level helper: build chunks from pages of text
# ---------------------------------------------------------------------------

def pages_to_chunks(
    pages: list[str],
    *,
    authority: str,
    document: str,
    max_tokens: int = 600,
    overlap_pct: float = 0.12,
) -> list[Chunk]:
    """Parse pages of text into Chunk objects ready for the vector store."""
    full_text = "\n\n".join(p for p in pages if p.strip())
    sections = parse_legal_hierarchy(full_text)
    chunks: list[Chunk] = []
    for section in sections:
        for citation, content in chunk_section(section, max_tokens=max_tokens, overlap_pct=overlap_pct):
            prefix = build_contextual_prefix(authority, document, citation, content)
            chunks.append(
                Chunk(
                    citation=citation,
                    content=content,
                    content_with_prefix=prefix,
                    token_count=_count_tokens(content),
                    version_hash=version_hash(prefix),
                    defined_terms=extract_defined_terms(content),
                    cross_refs=extract_cross_refs(content),
                )
            )
    return chunks
