from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    file_name: str
    doc_hash: str
    chunks_added: int
    total_chunks: int


def _document_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _read_pdf(file_bytes: bytes) -> list[Document]:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        return loader.load()
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)


def _build_chunk_ids(doc_hash: str, chunks: list[Document]) -> list[str]:
    ids: list[str] = []
    for i, chunk in enumerate(chunks):
        page = int(chunk.metadata.get("page", -1))
        ids.append(f"{doc_hash}:{page}:{i}")
    return ids


def ingest_pdf_bytes(
    file_name: str,
    file_bytes: bytes,
    vector_store,
    settings: Settings,
) -> IngestionResult:
    """Ingest a single PDF into Chroma with deterministic dedup-friendly IDs."""

    doc_hash = _document_hash(file_bytes)
    pages = _read_pdf(file_bytes)

    if not pages:
        raise ValueError("Uploaded PDF has no readable pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.separators,
    )

    base_docs: list[Document] = []
    for page_doc in pages:
        page_doc.metadata["file_name"] = file_name
        page_doc.metadata["doc_hash"] = doc_hash
        page_doc.metadata["source"] = file_name
        base_docs.append(page_doc)

    chunks = splitter.split_documents(base_docs)
    chunk_ids = _build_chunk_ids(doc_hash, chunks)

    existing = vector_store.get(ids=chunk_ids, include=[])
    existing_ids = set(existing.get("ids", []))

    docs_to_add: list[Document] = []
    ids_to_add: list[str] = []
    for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
        chunk.metadata["chunk_id"] = chunk_id
        if chunk_id not in existing_ids:
            docs_to_add.append(chunk)
            ids_to_add.append(chunk_id)

    if docs_to_add:
        vector_store.add_documents(docs_to_add, ids=ids_to_add)

    logger.info(
        "ingestion_complete file=%s doc_hash=%s chunks_added=%s total_chunks=%s",
        file_name,
        doc_hash,
        len(docs_to_add),
        len(chunks),
    )

    return IngestionResult(
        file_name=file_name,
        doc_hash=doc_hash,
        chunks_added=len(docs_to_add),
        total_chunks=len(chunks),
    )
