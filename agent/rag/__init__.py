"""RAG package (SupabaseVectorStore).

This package provides:
- ingestion scripts for Markdown documents
- retrievers (general + filtered by metadata)

It also preserves backwards-compatibility with the previous `agent.rag` module by
exporting `SupabaseRAG`.
"""

from .retriever import SupabaseRAG, get_corps_metier_retriever, get_retriever, is_corps_metier_question

__all__ = [
    "SupabaseRAG",
    "get_retriever",
    "get_corps_metier_retriever",
    "is_corps_metier_question",
]

