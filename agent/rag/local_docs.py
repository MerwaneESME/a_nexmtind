"""Local (repo) documentation search with a strict cascade strategy.

Cascade (max 3 documents total):
1) Domain specialized doc (e.g., isolation.md)
2) 1-2 related docs (e.g., gros_oeuvre.md, corps_de_metier.md)
3) Optional web research (handled elsewhere) if nothing found
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DOCS_DIR = Path(__file__).parent / "documents"


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


_TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ÖØ-öø-ÿ]+", re.UNICODE)

_STOPWORDS_FR = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "et",
    "eux",
    "il",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "ma",
    "mais",
    "me",
    "meme",
    "mêmes",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "y",
    "à",
    "ça",
    "cest",
    "cette",
    "cet",
    "comme",
    "est",
    "sont",
    "être",
    "etre",
    "fait",
    "faire",
    "comment",
    "pourquoi",
    "quoi",
    "quand",
    "combien",
    "où",
    "ou",
}


def _normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    # keep accents (french), just normalize apostrophes/punct already removed by token regex
    return t


def _query_tokens(query: str) -> list[str]:
    tokens = [_normalize_token(t) for t in _TOKEN_RE.findall(query or "")]
    out: list[str] = []
    for t in tokens:
        if len(t) < 3:
            continue
        if t in _STOPWORDS_FR:
            continue
        out.append(t)
    # de-dup but keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
    return ordered


@dataclass(frozen=True)
class LocalSnippet:
    source: str  # filename (e.g. "isolation.md")
    level: int  # 1 or 2 (local cascade), 3 reserved for web
    heading: str | None
    content: str
    score: float


def _split_markdown(text: str) -> list[tuple[str | None, str]]:
    """Return [(heading, chunk_text), ...] as paragraph-like chunks."""
    lines = (text or "").splitlines()
    chunks: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buff: list[str] = []

    def flush() -> None:
        nonlocal buff
        chunk = "\n".join(buff).strip()
        buff = []
        if not chunk:
            return
        # keep chunks reasonably small (for prompts)
        if len(chunk) <= 900:
            chunks.append((current_heading, chunk))
            return
        # naive split by blank lines first
        parts = re.split(r"\n\s*\n+", chunk)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            chunks.append((current_heading, p[:900] + ("…" if len(p) > 900 else "")))

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("#"):
            flush()
            h = line.lstrip("#").strip()
            current_heading = h or current_heading
            continue
        if not line.strip():
            flush()
            continue
        buff.append(line)

    flush()
    return chunks


def _chunk_score(query_tokens: list[str], chunk_text: str, *, heading: str | None) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = {_normalize_token(t) for t in _TOKEN_RE.findall(chunk_text or "")}
    overlap = sum(1 for t in query_tokens if t in chunk_tokens)
    base = overlap / max(1, len(query_tokens))
    if heading:
        heading_tokens = {_normalize_token(t) for t in _TOKEN_RE.findall(heading)}
        overlap_h = sum(1 for t in query_tokens if t in heading_tokens)
        base += 0.10 * min(1.0, overlap_h / max(1, len(query_tokens)))
    # slight length penalty for very long chunks (already truncated, but still)
    base *= 1.0 - min(0.2, max(0, (len(chunk_text) - 700)) / 2000)
    return float(max(0.0, min(base, 1.0)))


def _best_snippets_in_doc(query: str, path: Path, *, level: int, max_snippets: int = 3) -> list[LocalSnippet]:
    text = _read_text(path)
    q_tokens = _query_tokens(query)
    if not q_tokens or not text.strip():
        return []

    scored: list[LocalSnippet] = []
    for heading, chunk in _split_markdown(text):
        score = _chunk_score(q_tokens, chunk, heading=heading)
        if score <= 0:
            continue
        scored.append(
            LocalSnippet(
                source=path.name,
                level=level,
                heading=heading,
                content=chunk,
                score=score,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:max_snippets]


# --- Domain mapping (Level 1) ---

_DOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("isolation", re.compile(r"\b(isolation|isolant|laine\s+de\s+verre|laine\s+de\s+roche|ouate|pare[- ]vapeur|ITI|ITE|ponts?\s+thermiques?)\b", re.IGNORECASE)),
    ("menuiserie", re.compile(r"\b(menuiserie|fen[eê]tre|portes?|baie\s+vitr[eé]e|volet|dormant|ouvrant|double\s+vitrage|triple\s+vitrage)\b", re.IGNORECASE)),
    ("carrelage", re.compile(r"\b(carrelage|carreaux|fa[iï]ence|joints?\b|ragr[eé]age|chape|plinthes?\s+carrelage)\b", re.IGNORECASE)),
    ("gros_oeuvre", re.compile(r"\b(gros\s*œuvre|gros\s*oeuvre|ma[cç]onnerie|b[eé]ton|dalle|fondations?|poutre|linteau|mur\s+porteur|fissures?)\b", re.IGNORECASE)),
    ("electricite", re.compile(r"\b([eé]lectricit[eé]|[eé]lectricien|tableau\s+[eé]lectrique|disjonct|diff[eé]rentiel|prise|interrupteur|gaine|goulotte|consuel)\b", re.IGNORECASE)),
    ("plomberie", re.compile(r"\b(plomberie|plombier|fuite|robinet|sanitaire|evacuation|[eé]vacuation|siphon|wc|toilettes|douche|baignoire|chauffe[- ]eau|ballon)\b", re.IGNORECASE)),
]


def detect_domain(query: str) -> str | None:
    q = (query or "").strip()
    if not q:
        return None
    for domain, pattern in _DOMAIN_PATTERNS:
        if pattern.search(q):
            return domain
    return None


def _doc_path_for_domain(domain: str) -> Path | None:
    if not domain:
        return None
    p = DOCS_DIR / f"{domain}.md"
    return p if p.exists() else None


_RELATED_DOCS: dict[str, list[str]] = {
    # Keep total consulted docs <= 3 in cascade logic (1 primary + up to 2 related)
    "isolation": ["gros_oeuvre", "menuiserie", "corps_de_metier"],
    "menuiserie": ["isolation", "gros_oeuvre", "corps_de_metier"],
    "carrelage": ["gros_oeuvre", "isolation", "corps_de_metier"],
    "gros_oeuvre": ["isolation", "menuiserie", "corps_de_metier"],
    "electricite": ["gros_oeuvre", "corps_de_metier", "isolation"],
    "plomberie": ["gros_oeuvre", "carrelage", "corps_de_metier"],
}


def cascade_search(
    query: str,
    *,
    domain: str | None,
    max_docs: int = 3,
) -> tuple[list[LocalSnippet], list[str]]:
    """Strict cascade local search. Returns (snippets, consulted_sources)."""
    consulted: list[str] = []
    snippets: list[LocalSnippet] = []

    # Level 1: specialized doc
    primary_path = _doc_path_for_domain(domain or "")
    if primary_path and len(consulted) < max_docs:
        consulted.append(primary_path.name)
        s1 = _best_snippets_in_doc(query, primary_path, level=1, max_snippets=3)
        # "found" heuristic: at least one strong snippet
        if s1 and (s1[0].score >= 0.35 or (len(s1) >= 2 and s1[1].score >= 0.30)):
            return (s1, consulted)
        snippets.extend(s1)

    # Level 2: related docs (up to 2 more, total max_docs)
    related = _RELATED_DOCS.get(domain or "", ["corps_de_metier"])
    for rel in related:
        if len(consulted) >= max_docs:
            break
        p = DOCS_DIR / f"{rel}.md"
        if not p.exists():
            continue
        if p.name in consulted:
            continue
        consulted.append(p.name)
        s2 = _best_snippets_in_doc(query, p, level=2, max_snippets=2)
        if s2 and s2[0].score >= 0.35:
            return (s2, consulted)
        snippets.extend(s2)

    # Return whatever weak hints we found (may be empty)
    snippets.sort(key=lambda s: s.score, reverse=True)
    return (snippets[:4], consulted)


def ensure_domain_doc(domain: str) -> Path:
    """Create the domain doc if missing and return its path (used by web-updater)."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"{domain}.md"
    if path.exists():
        return path
    title = domain.replace("_", " ").title()
    path.write_text(f"# {title}\n\n", encoding="utf-8")
    return path
