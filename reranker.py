from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    """Load cross-encoder once per process."""

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query: str, docs: list[Document], top_n: int = 4) -> list[tuple[Document, float]]:
    """Re-rank retrieved chunks with a cross-encoder and return top-n."""

    if not docs:
        return []

    model = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    ranked = sorted(zip(docs, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_n]]
