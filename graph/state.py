from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """Shared state for the query-time RAG graph."""

    query: str
    doc_names: list[str] | None
    k: int | None
    fetch_k: int | None
    expanded_queries: list[str]
    retrieved: list[Document]
    reranked: list[tuple[Document, float]]
    top_docs: list[Document]
    sources: list[str]
    answer: str
    retry_count: int
    context_ok: bool
    skip_generate: bool
