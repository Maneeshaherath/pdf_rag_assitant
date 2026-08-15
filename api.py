from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import _looks_like_langsmith_key, get_settings
from graph.nodes import REFUSE_ANSWER
from graph.rag_graph import build_rag_graph
from graph.state import RAGState
from ingestion import ingest_pdf_bytes
from llm import stream_answer
from schemas import QueryRequest, QueryResponse, RetrievedChunk, UploadResponse, UploadResponseItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

settings = get_settings()

if not settings.openai_api_key:
    logger.warning("OPENAI_API_KEY is missing. API calls to OpenAI will fail until it is set.")

if settings.langsmith_tracing:
    if _looks_like_langsmith_key(settings.langsmith_api_key):
        logger.info("langsmith_tracing_enabled project=%s", os.getenv("LANGSMITH_PROJECT"))
        key = settings.langsmith_api_key.strip().strip('"')
        if key.startswith("lsv2_pt_"):
            logger.warning(
                "LANGSMITH_API_KEY looks like a personal token (lsv2_pt_). "
                "Trace ingest often returns 403. Create a service API key (lsv2_sk_) "
                "in LangSmith Settings > API Keys, and set LANGSMITH_WORKSPACE_ID if asked."
            )
    else:
        logger.warning(
            "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is missing or still a placeholder. "
            "Replace <your-api-key> in .env with a real key from https://smith.langchain.com"
        )

app = FastAPI(title="Production RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_openai_key() -> None:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENAI_API_KEY is not set. In PowerShell use "
                '$env:OPENAI_API_KEY="your-key" then restart uvicorn, '
                "or put OPENAI_API_KEY in a .env file or .streamlit/secrets.toml."
            ),
        )


def _get_vector_store() -> Chroma:
    _require_openai_key()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        chunk_size=settings.embedding_batch_size,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


@app.get("/")
async def root() -> dict[str, Any]:
    return {"ok": True, "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "collection": settings.chroma_collection,
        "openai_key_set": bool(settings.openai_api_key),
        "langsmith_tracing": bool(settings.langsmith_tracing),
        "langsmith_key_set": _looks_like_langsmith_key(settings.langsmith_api_key),
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", settings.langsmith_project),
    }


@app.get("/documents")
async def documents() -> dict[str, list[str]]:
    """Return unique document names available in the vector store."""

    vector_store = _get_vector_store()
    raw = vector_store.get(include=["metadatas"])
    metadatas = raw.get("metadatas", [])

    names: set[str] = set()
    for metadata in metadatas:
        if not metadata:
            continue
        file_name = metadata.get("file_name") or metadata.get("source")
        if file_name:
            names.add(str(file_name))

    return {"documents": sorted(names)}


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    vector_store = _get_vector_store()
    results: list[UploadResponseItem] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        file_bytes = await file.read()
        if len(file_bytes) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"{file.filename} exceeds {settings.max_upload_mb} MB")

        try:
            result = await asyncio.to_thread(
                ingest_pdf_bytes,
                file.filename,
                file_bytes,
                vector_store,
                settings,
            )
        except Exception as exc:
            logger.exception("upload_failed file=%s", file.filename)
            raise HTTPException(status_code=500, detail=f"Failed to ingest {file.filename}: {exc}") from exc

        results.append(
            UploadResponseItem(
                file_name=result.file_name,
                doc_hash=result.doc_hash,
                chunks_added=result.chunks_added,
                total_chunks=result.total_chunks,
            )
        )

    return UploadResponse(ingested=results)


def _validate_query(request: QueryRequest) -> None:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    if len(request.query) > settings.max_query_chars:
        raise HTTPException(status_code=400, detail=f"Query exceeds {settings.max_query_chars} characters")


def _initial_state(request: QueryRequest, skip_generate: bool = False) -> RAGState:
    return {
        "query": request.query,
        "doc_names": request.doc_names,
        "k": request.k,
        "fetch_k": request.fetch_k,
        "expanded_queries": [],
        "retrieved": [],
        "reranked": [],
        "top_docs": [],
        "sources": [],
        "answer": "",
        "retry_count": 0,
        "context_ok": False,
        "skip_generate": skip_generate,
    }


def _debug_chunks(state: RAGState, debug: bool) -> list[RetrievedChunk] | None:
    if not debug:
        return None
    return [
        RetrievedChunk(
            chunk_id=doc.metadata.get("chunk_id", "unknown"),
            source=doc.metadata.get("file_name", doc.metadata.get("source", "uploaded.pdf")),
            page=doc.metadata.get("page"),
            score=score,
            content=doc.page_content,
        )
        for doc, score in state.get("reranked") or []
    ]


def _invoke_graph(graph, state: RAGState, query: str) -> RAGState:
    result = graph.invoke(
        state,
        config={
            "run_name": "pdf-rag-query",
            "tags": ["pdf-rag"],
            "metadata": {"query": query},
        },
    )
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:
        logger.debug("langsmith_flush_skipped", exc_info=True)
    return result


async def _run_rag(request: QueryRequest, skip_generate: bool = False) -> RAGState:
    _validate_query(request)
    vector_store = _get_vector_store()
    graph = build_rag_graph(vector_store, settings)
    state = await asyncio.to_thread(
        _invoke_graph,
        graph,
        _initial_state(request, skip_generate=skip_generate),
        request.query,
    )
    logger.info(
        "query_graph_complete query=%s expanded=%s retrieved=%s reranked=%s retry=%s refused=%s",
        request.query,
        state.get("expanded_queries"),
        len(state.get("retrieved") or []),
        len(state.get("top_docs") or []),
        state.get("retry_count", 0),
        state.get("answer") == REFUSE_ANSWER,
    )
    return state


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    state = await _run_rag(request, skip_generate=False)
    return QueryResponse(
        answer=state.get("answer") or REFUSE_ANSWER,
        sources=state.get("sources") or [],
        expanded_queries=state.get("expanded_queries") or [],
        retrieved_count=len(state.get("retrieved") or []),
        chunks=_debug_chunks(state, request.debug),
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    state = await _run_rag(request, skip_generate=True)
    top_docs = state.get("top_docs") or []
    sources = state.get("sources") or []
    expanded_queries = state.get("expanded_queries") or []
    retrieved = state.get("retrieved") or []
    async def event_generator():
        try:
            if not state.get("context_ok") or not top_docs:
                yield json.dumps({"type": "token", "data": {"token": REFUSE_ANSWER}}) + "\n"
            else:
                gen_query = expanded_queries[1] if len(expanded_queries) > 1 else request.query
                for token in stream_answer(gen_query, top_docs, settings):
                    yield json.dumps({"type": "token", "data": {"token": token}}) + "\n"

            end_payload = {
                "type": "end",
                "data": {
                    "sources": sources,
                    "expanded_queries": expanded_queries,
                    "retrieved_count": len(retrieved),
                    "chunks": [
                        {
                            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                            "source": doc.metadata.get("file_name", doc.metadata.get("source", "uploaded.pdf")),
                            "page": doc.metadata.get("page"),
                            "score": score,
                            "content": doc.page_content,
                        }
                        for doc, score in state.get("reranked") or []
                    ]
                    if request.debug
                    else None,
                },
            }
            yield json.dumps(end_payload) + "\n"
        except Exception as exc:
            logger.exception("query_stream_failed")
            yield json.dumps({"type": "error", "data": {"message": str(exc)}}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=settings.api_host, port=settings.api_port, reload=True)
