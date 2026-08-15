from __future__ import annotations

import json
import logging
from collections import OrderedDict

from langchain_core.prompts import ChatPromptTemplate

from config import Settings
from graph.state import RAGState
from llm import format_sources, generate_answer, get_chat_model
from reranker import rerank_documents
from retriever import expand_query, hybrid_retrieve

logger = logging.getLogger(__name__)

REFUSE_ANSWER = "I don't know."


def make_nodes(vector_store, settings: Settings):
    """Build node callables closed over the vector store and settings."""

    def expand_node(state: RAGState) -> dict:
        llm = get_chat_model(settings, streaming=False)
        queries = expand_query(
            state["query"],
            llm=llm,
            n=settings.query_expansions,
            doc_names=state.get("doc_names"),
        )
        logger.info("graph_expand query=%s queries=%s", state["query"], queries)
        return {"expanded_queries": queries}

    def retrieve_node(state: RAGState) -> dict:
        queries = state.get("expanded_queries") or [state["query"]]
        docs = hybrid_retrieve(
            queries=queries,
            vector_store=vector_store,
            settings=settings,
            doc_names=state.get("doc_names"),
            k=state.get("k"),
            fetch_k=state.get("fetch_k"),
        )
        logger.info("graph_retrieve docs=%s", len(docs))
        return {"retrieved": docs}

    def rerank_node(state: RAGState) -> dict:
        top_n = state.get("k") or settings.rerank_top_n
        queries = state.get("expanded_queries") or [state["query"]]
        rerank_query = queries[1] if len(queries) > 1 else queries[0]
        ranked = rerank_documents(rerank_query, state.get("retrieved") or [], top_n=top_n)
        top_docs = [doc for doc, _ in ranked]
        sources = format_sources(top_docs)
        logger.info("graph_rerank kept=%s", len(ranked))
        return {"reranked": ranked, "top_docs": top_docs, "sources": sources}

    def grade_node(state: RAGState) -> dict:
        ranked = state.get("reranked") or []
        if not ranked:
            context_ok = False
        else:
            top_score = max(float(score) for _, score in ranked)
            context_ok = top_score >= settings.rerank_min_score
        logger.info(
            "graph_grade ok=%s retry=%s docs=%s",
            context_ok,
            state.get("retry_count", 0),
            len(ranked),
        )
        return {"context_ok": context_ok}

    def rewrite_node(state: RAGState) -> dict:
        llm = get_chat_model(settings, streaming=False)
        prompt = ChatPromptTemplate.from_template(
            """
You rewrite a user question so document search can find better chunks.
The first search returned weak or empty results.
Return only valid JSON: {{"queries": ["q1", "q2"]}}
Rules:
- Keep the same intent.
- Use concrete terms that might appear in PDFs.
- Fix likely typos using the document titles when obvious.
- Do not add facts.
- Produce exactly 2 rewrites.
Document titles: {titles}
Question: {question}
""".strip()
        )
        titles = ", ".join(state.get("doc_names") or []) or "(none)"
        raw = llm.invoke(prompt.format_messages(question=state["query"], titles=titles)).content
        extra: list[str] = []
        try:
            payload = json.loads(raw)
            generated = payload.get("queries", [])
            if isinstance(generated, list):
                extra = [q.strip() for q in generated if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError:
            logger.warning("graph_rewrite_parse_failed raw=%s", raw)

        merged = list(OrderedDict.fromkeys([*(state.get("expanded_queries") or []), *extra]))
        retry_count = int(state.get("retry_count") or 0) + 1
        logger.info("graph_rewrite retry=%s queries=%s", retry_count, merged)
        return {"expanded_queries": merged, "retry_count": retry_count}

    def generate_node(state: RAGState) -> dict:
        if state.get("skip_generate"):
            return {}
        docs = state.get("top_docs") or []
        if not docs:
            return {"answer": REFUSE_ANSWER}
        queries = state.get("expanded_queries") or [state["query"]]
        gen_query = queries[1] if len(queries) > 1 else queries[0]
        answer = generate_answer(gen_query, docs, settings)
        return {"answer": answer, "sources": format_sources(docs)}

    def refuse_node(state: RAGState) -> dict:
        logger.info("graph_refuse query=%s", state["query"])
        return {"answer": REFUSE_ANSWER}

    return {
        "expand_query": expand_node,
        "hybrid_retrieve": retrieve_node,
        "rerank": rerank_node,
        "grade_context": grade_node,
        "rewrite_query": rewrite_node,
        "generate": generate_node,
        "refuse_no_context": refuse_node,
    }
