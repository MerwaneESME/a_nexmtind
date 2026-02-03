"""RAG classifier (True/False) to keep retrieval strictly optional."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_fast_llm
from .logging_config import logger
from .prompts import RAG_CLASSIFIER_PROMPT


_RAG_HINT_RE = re.compile(
    r"\b("
    r"selon|d'apr[eè]s|dans (le|les) (document|pdf|contrat|cgv|conditions)|"
    r"retrouve|recherche|rappelle|historique|base de connaissances|kb|"
    r"qu'est[- ]ce que dit|qu'indique"
    r")\b",
    re.IGNORECASE,
)

_BTP_TECH_HINT_RE = re.compile(
    r"\b("
    r"fissure|fissures|fuite|infiltration|humidit[eé]|moisissure|"
    r"panne|d[eé]faut|casse|ne marche pas|"
    r"refaire|r[eé]nover|r[eé]novation|r[eé]parer|remplacer|poser|installer|"
    r"pr[eé]parer|checklist|v[eé]rifier|contr[oô]ler|diagnostiquer|"
    r"mat[eé]riau|mat[eé]riaux|quantit[eé]|ratio|cadence|taux horaire|"
    r"prix|co[uû]t|budget|tarif|devis|d[eé]lai|dur[eé]e|"
    r"toiture|mur|murs|plomberie|carrelage|peinture|placo|"
    r"[eé]tanch[eé]it[eé]"
    r")\b",
    re.IGNORECASE,
)


def _heuristic(message: str, metadata: dict[str, Any] | None) -> bool | None:
    msg = (message or "").strip()
    if not msg:
        return False

    # If user is validating/calculating, RAG is not needed.
    if isinstance(metadata, dict):
        mode = str(metadata.get("mode") or "").lower()
        if mode in {"validate", "validation"} or metadata.get("validate_section"):
            return False

    if _RAG_HINT_RE.search(msg):
        return True

    # Technical BTP questions benefit from retrieval (ratios, étapes, signaux d'alerte).
    if _BTP_TECH_HINT_RE.search(msg):
        return True

    # Short questions without doc-related hints: likely no RAG.
    if len(msg) < 80 and not (metadata or {}):
        return False

    return None


async def should_use_rag(message: str, *, metadata: dict[str, Any] | None = None) -> bool:
    """Return True only when retrieval is likely required to answer accurately."""
    heuristic = _heuristic(message, metadata)
    if heuristic is not None:
        return heuristic

    llm = get_fast_llm(temperature=0.0).bind(max_tokens=8)
    try:
        raw = await llm.ainvoke(
            [
                SystemMessage(content=RAG_CLASSIFIER_PROMPT),
                HumanMessage(content=(message or "")),
            ]
        )
    except Exception as exc:
        logger.warning("rag classifier failed: %s", exc)
        return False

    content = (getattr(raw, "content", None) or str(raw)).strip().lower()
    if "true" in content:
        return True
    if "false" in content:
        return False
    return False
