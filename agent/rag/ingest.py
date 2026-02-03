"""Ingestion for Markdown documents into SupabaseVectorStore.

This file is meant to be executed as a script:
    .venv/Scripts/python.exe -m agent.rag.ingest
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.logging_config import logger
from agent.rag.retriever import get_vector_store


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_markdown(doc: Document) -> List[Document]:
    # Requirement: 300–500 tokens per chunk. Default 420 tokens (+ overlap).
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "420"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "60"))
    chunk_size = max(300, min(chunk_size, 500))
    chunk_overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=os.getenv("RAG_TOKENIZER_MODEL", "gpt-4o-mini"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents([doc])


def _stable_ids(source: str, count: int) -> list[str]:
    # Use UUID5 to keep ids deterministic and safe for `uuid` columns.
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{source}:{i}")) for i in range(count)]


def ingest_corps_metier() -> int:
    """Load `corps_de_metier.md`, chunk it, embed and upsert into `documents` table."""
    path = DOCUMENTS_DIR / "corps_de_metier.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    text = _read_text(path).strip()
    if not text:
        logger.warning("File is empty: %s", path)
        return 0

    base_meta = {"type": "corps_metier", "source": "corps_de_metier.md"}
    base_doc = Document(page_content=text, metadata=base_meta)
    chunks = _chunk_markdown(base_doc)

    # Add deterministic chunk index
    for i, chunk in enumerate(chunks):
        chunk.metadata = {**base_meta, "chunk": i}

    store = get_vector_store()
    if not store:
        raise RuntimeError("SupabaseVectorStore not configured (SUPABASE_URL / keys missing).")

    ids = _stable_ids(base_meta["source"], len(chunks))
    store.add_texts(
        texts=[d.page_content for d in chunks],
        metadatas=[d.metadata for d in chunks],
        ids=ids,
    )

    logger.info("✅ Ingested corps_metier: %d chunks", len(chunks))
    return len(chunks)


def main() -> None:
    ingest_corps_metier()


if __name__ == "__main__":
    main()
