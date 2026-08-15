from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from config import Settings
from graph.nodes import make_nodes
from graph.state import RAGState


def build_rag_graph(vector_store, settings: Settings):
    """Compile the query-time RAG graph: retrieve, grade, one retry, then generate or refuse."""

    nodes = make_nodes(vector_store, settings)

    def route_after_grade(state: RAGState) -> str:
        if state.get("context_ok"):
            return "generate"
        retry_count = int(state.get("retry_count") or 0)
        if retry_count < settings.rag_max_retries:
            return "rewrite_query"
        return "refuse"

    builder = StateGraph(RAGState)
    builder.add_node("expand_query", nodes["expand_query"])
    builder.add_node("hybrid_retrieve", nodes["hybrid_retrieve"])
    builder.add_node("rerank", nodes["rerank"])
    builder.add_node("grade_context", nodes["grade_context"])
    builder.add_node("rewrite_query", nodes["rewrite_query"])
    builder.add_node("generate", nodes["generate"])
    builder.add_node("refuse_no_context", nodes["refuse_no_context"])

    builder.add_edge(START, "expand_query")
    builder.add_edge("expand_query", "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "rerank")
    builder.add_edge("rerank", "grade_context")
    builder.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "refuse": "refuse_no_context",
        },
    )
    builder.add_edge("rewrite_query", "hybrid_retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("refuse_no_context", END)

    return builder.compile()
