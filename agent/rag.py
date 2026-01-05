import os
from typing import Any, List, Optional

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

from .config import get_embeddings
from .supabase_client import get_client


class SupabaseRAG:
    """Wrapper simple autour de SupabaseVectorStore pour le RAG."""

    def __init__(self, table_name: Optional[str] = None, query_name: Optional[str] = None, k: int = 5, threshold: float = 0.75):
        self.table_name = table_name or os.getenv("SUPABASE_VECTOR_TABLE", "documents")
        self.query_name = query_name or os.getenv("SUPABASE_VECTOR_QUERY_NAME", "match_documents")
        self.k = k
        self.threshold = threshold
        self._store: Optional[SupabaseVectorStore] = None
        self._init_store()

    def _init_store(self):
        client = get_client()
        if not client:
            return
        try:
            self._store = SupabaseVectorStore(
                client=client,
                embedding=get_embeddings(),
                table_name=self.table_name,
                query_name=self.query_name,
            )
        except Exception:
            self._store = None

    def is_ready(self) -> bool:
        return self._store is not None

    def retrieve(self, query: str) -> List[dict]:
        """
        Récupère les documents les plus similaires depuis SupabaseVectorStore.
        Le score est une similarité cosinus (0-1), où 1 = identique, 0 = différent.
        Seuil par défaut: 0.75 (documents très similaires uniquement).
        """
        if not query or not self._store:
            return []
        try:
            docs_scores = self._store.similarity_search_with_score(query, k=self.k)
        except Exception:
            return []

        results: List[dict] = []
        for doc, score in docs_scores:
            # Score de similarité cosinus: plus élevé = plus similaire
            # On garde seulement les documents avec score >= threshold (très similaires)
            keep = score is None or score >= self.threshold
            if not keep:
                continue
            if isinstance(doc, Document):
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score) if score is not None else None,
                    }
                )
        return results


def match_chunks(chunks: List[dict], max_chars: int = 1200) -> str:
    """Assemble des chunks en respectant une longueur maximale."""
    acc: List[str] = []
    total_len = 0
    for chunk in chunks:
        text = chunk.get("content") or ""
        if not text:
            continue
        if total_len + len(text) > max_chars:
            break
        acc.append(text)
        total_len += len(text)
    return "\n\n".join(acc)
