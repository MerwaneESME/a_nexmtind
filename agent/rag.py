"""RAG avec SupabaseVectorStore."""
import os
from typing import List, Optional

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

from .config import get_embeddings
from .supabase_client import get_client


class SupabaseRAG:
    """Wrapper autour de SupabaseVectorStore pour le RAG."""

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
        """Récupère les documents les plus similaires depuis SupabaseVectorStore."""
        if not query or not self._store:
            return []
        try:
            docs_scores = self._store.similarity_search_with_score(query, k=self.k)
        except Exception:
            return []

        results: List[dict] = []
        for doc, score in docs_scores:
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
