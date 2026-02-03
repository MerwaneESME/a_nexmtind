"""SupabaseVectorStore retrievers (general + filtered) and lightweight routing helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agent.config import get_embeddings
from agent.logging_config import logger
from agent.supabase_client import get_client


_STORE: SupabaseVectorStore | None = None
_GENERAL_RETRIEVER: BaseRetriever | None = None
_CORPS_METIER_RETRIEVER: BaseRetriever | None = None


def _get_table_name() -> str:
    # Requirement: use table `documents` (keep env override for existing installs).
    return (os.getenv("SUPABASE_VECTOR_TABLE") or "documents").strip() or "documents"


def _get_query_name() -> str:
    return (os.getenv("SUPABASE_VECTOR_QUERY_NAME") or "match_documents").strip() or "match_documents"


def get_vector_store() -> SupabaseVectorStore | None:
    """Singleton SupabaseVectorStore instance (sync client)."""
    global _STORE
    if _STORE is not None:
        return _STORE

    client = get_client()
    if not client:
        return None

    try:
        _STORE = SupabaseVectorStore(
            client=client,
            embedding=get_embeddings(),
            table_name=_get_table_name(),
            query_name=_get_query_name(),
        )
    except Exception as exc:
        logger.warning("SupabaseVectorStore init failed: %s", exc)
        _STORE = None
    return _STORE


def get_retriever(
    *,
    filter: Dict[str, Any] | None = None,
    k: int = 4,
    score_threshold: float | None = None,
) -> BaseRetriever | None:
    """General retriever with optional metadata filter."""
    store = get_vector_store()
    if not store:
        return None

    search_kwargs: dict[str, Any] = {"k": k}
    if filter:
        search_kwargs["filter"] = filter

    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
        return store.as_retriever(search_type="similarity_score_threshold", search_kwargs=search_kwargs)

    return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def get_corps_metier_retriever(*, k: int = 4, score_threshold: float | None = 0.75) -> BaseRetriever | None:
    """Retriever filtered on metadata: {"type": "corps_metier"}."""
    global _CORPS_METIER_RETRIEVER
    if _CORPS_METIER_RETRIEVER is not None:
        return _CORPS_METIER_RETRIEVER

    _CORPS_METIER_RETRIEVER = get_retriever(filter={"type": "corps_metier"}, k=k, score_threshold=score_threshold)
    return _CORPS_METIER_RETRIEVER


def get_general_retriever(*, k: int = 4, score_threshold: float | None = 0.75) -> BaseRetriever | None:
    """General retriever (no metadata filter)."""
    global _GENERAL_RETRIEVER
    if _GENERAL_RETRIEVER is not None:
        return _GENERAL_RETRIEVER

    _GENERAL_RETRIEVER = get_retriever(filter=None, k=k, score_threshold=score_threshold)
    return _GENERAL_RETRIEVER


_CORPS_METIER_RE = re.compile(
    r"\b("
    r"corps\s+de\s+m[eé]tier|m[eé]tier(s)?\b|artisan\b|quel\s+pro\b|quel\s+professionnel\b|"
    r"qui\s+appeler|qui\s+fait|r[oô]le\b|missions?\b|tarif\b|taux\s+horaire\b|"
    r"plombier|plomberie|[eé]lectricien|[eé]lectricit[eé]|ma[cç]on|ma[cç]onnerie|"
    r"peintre|peinture|carreleur|carrelage|fa[iï]ence|plaquiste|placo|isolation|"
    r"chauffagiste|chauffage|clim(atisation)?|ventilation|vmc|pac|chaudi[eè]re"
    r")\b",
    re.IGNORECASE,
)


def is_corps_metier_question(message: str) -> bool:
    """Heuristic router for 'corps de métier' questions (BTP trades)."""
    msg = (message or "").strip()
    if not msg:
        return False
    return bool(_CORPS_METIER_RE.search(msg))


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float | None = None


class SupabaseRAG:
    """Backwards-compatible wrapper used by legacy pipeline."""

    def __init__(
        self,
        *,
        k: int = 5,
        threshold: float = 0.75,
    ) -> None:
        self.k = k
        self.threshold = threshold

    def is_ready(self) -> bool:
        return get_vector_store() is not None

    def retrieve(self, query: str, *, filter: Dict[str, Any] | None = None) -> List[dict]:
        """Retrieve docs with similarity scores (Supabase RPC returns `similarity`)."""
        store = get_vector_store()
        if not store or not query:
            return []

        try:
            docs_scores = store.similarity_search_with_relevance_scores(
                query,
                k=self.k,
                filter=filter,
                score_threshold=self.threshold,
            )
        except Exception:
            return []

        results: list[dict] = []
        for doc, score in docs_scores:
            if not isinstance(doc, Document):
                continue
            results.append({"content": doc.page_content, "metadata": doc.metadata, "score": float(score)})
        return results

