from __future__ import annotations

import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from config import Settings

logger = logging.getLogger(__name__)


def _normalize_tokens(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if tok]


def expand_query(
    question: str,
    llm,
    n: int = 3,
    doc_names: list[str] | None = None,
) -> list[str]:
    """Create semantic query variations to increase recall, including likely typo fixes."""

    titles = ", ".join(doc_names) if doc_names else "(none provided)"
    prompt = ChatPromptTemplate.from_template(
        """
You rewrite user questions for document retrieval.
Return only valid JSON in this format:
{{"queries": ["q1", "q2", "q3"]}}
Rules:
- Keep intent identical.
- Keep each rewrite short.
- Do not add facts.
- If the question looks misspelled, include a corrected version that matches the document titles when obvious (example: paring -> parsing).
- Produce exactly {n} rewrites.
Document titles: {titles}
Question: {question}
""".strip()
    )

    raw = llm.invoke(prompt.format_messages(question=question, n=n, titles=titles)).content
    queries = [question]

    try:
        payload = json.loads(raw)
        generated = payload.get("queries", [])
        if isinstance(generated, list):
            queries.extend([q.strip() for q in generated if isinstance(q, str) and q.strip()])
    except json.JSONDecodeError:
        logger.warning("query_expansion_parse_failed raw=%s", raw)

    unique = list(OrderedDict.fromkeys(queries))
    return unique[: n + 1]


def _vector_mmr_retrieve(
    vector_store,
    query: str,
    settings: Settings,
    doc_names: list[str] | None,
    k: int | None = None,
    fetch_k: int | None = None,
) -> list[Document]:
    filter_payload = None
    if doc_names:
        filter_payload = {"file_name": {"$in": doc_names}}

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k or settings.retrieval_k,
            "fetch_k": fetch_k or settings.retrieval_fetch_k,
            "filter": filter_payload,
        },
    )
    return retriever.invoke(query)


def _load_keyword_corpus(vector_store, doc_names: list[str] | None) -> tuple[list[str], list[dict]]:
    where = None
    if doc_names:
        where = {"file_name": {"$in": doc_names}}

    raw = vector_store.get(where=where, include=["documents", "metadatas"])
    documents = raw.get("documents", []) or []
    metadatas = raw.get("metadatas", []) or []
    return documents, metadatas


def _bm25_top(
    documents: list[str],
    metadatas: list[dict],
    query: str,
    settings: Settings,
    bm25: BM25Okapi,
) -> list[Document]:
    if not documents:
        return []

    scores = bm25.get_scores(_normalize_tokens(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    top = ranked[: settings.bm25_k]

    output: list[Document] = []
    for idx, _ in top:
        output.append(Document(page_content=documents[idx], metadata=metadatas[idx] or {}))
    return output


def _keyword_retrieve(
    vector_store,
    query: str,
    settings: Settings,
    doc_names: list[str] | None,
) -> list[Document]:
    documents, metadatas = _load_keyword_corpus(vector_store, doc_names)
    if not documents:
        return []
    corpus = [_normalize_tokens(text) for text in documents]
    bm25 = BM25Okapi(corpus)
    return _bm25_top(documents, metadatas, query, settings, bm25)


def merge_and_deduplicate(doc_lists: list[list[Document]]) -> list[Document]:
    dedup: OrderedDict[str, Document] = OrderedDict()
    for docs in doc_lists:
        for doc in docs:
            key = doc.metadata.get("chunk_id") or hash(doc.page_content)
            dedup[str(key)] = doc
    return list(dedup.values())


def hybrid_retrieve(
    queries: list[str],
    vector_store,
    settings: Settings,
    doc_names: list[str] | None = None,
    k: int | None = None,
    fetch_k: int | None = None,
) -> list[Document]:
    """Run MMR vector search and BM25 in parallel, then merge unique chunks."""

    if not queries:
        return []

    documents, metadatas = _load_keyword_corpus(vector_store, doc_names)
    bm25 = BM25Okapi([_normalize_tokens(text) for text in documents]) if documents else None

    all_results: list[list[Document]] = []
    max_workers = min(8, max(2, len(queries)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        vector_futures = [
            pool.submit(
                _vector_mmr_retrieve,
                vector_store,
                q,
                settings,
                doc_names,
                k,
                fetch_k,
            )
            for q in queries
        ]
        if bm25 is not None:
            for q in queries:
                all_results.append(_bm25_top(documents, metadatas, q, settings, bm25))
        for fut in vector_futures:
            all_results.append(fut.result())

    return merge_and_deduplicate(all_results)


def retrieve_documents(
    query: str,
    vector_store,
    llm,
    settings: Settings,
    doc_names: list[str] | None = None,
    k: int | None = None,
    fetch_k: int | None = None,
) -> tuple[list[str], list[Document]]:
    """Hybrid retrieval: query expansion + MMR vector retrieval + BM25 retrieval."""

    expanded_queries = expand_query(
        query,
        llm=llm,
        n=settings.query_expansions,
        doc_names=doc_names,
    )
    merged = hybrid_retrieve(
        queries=expanded_queries,
        vector_store=vector_store,
        settings=settings,
        doc_names=doc_names,
        k=k,
        fetch_k=fetch_k,
    )
    logger.info("retrieval_complete query=%s expanded=%s docs=%s", query, expanded_queries, len(merged))
    return expanded_queries, merged
