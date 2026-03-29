from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import get_settings
from ingestion import ingest_pdf_bytes
from llm import format_sources, generate_answer, get_chat_model, stream_answer
from reranker import rerank_documents
from retriever import retrieve_documents
from schemas import QueryRequest, QueryResponse, RetrievedChunk, UploadResponse, UploadResponseItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

settings = get_settings()

if not settings.openai_api_key:
    logger.warning("OPENAI_API_KEY is missing. API calls to OpenAI will fail until it is set.")

app = FastAPI(title="Production RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, chunk_size=settings.embedding_batch_size)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "collection": settings.chroma_collection}


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


async def _prepare_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    if len(request.query) > settings.max_query_chars:
        raise HTTPException(status_code=400, detail=f"Query exceeds {settings.max_query_chars} characters")

    vector_store = _get_vector_store()
    expander_llm = get_chat_model(settings, streaming=False)

    expanded_queries, retrieved = await asyncio.to_thread(
        retrieve_documents,
        request.query,
        vector_store,
        expander_llm,
        settings,
        request.doc_names,
        request.k,
        request.fetch_k,
    )

    reranked = await asyncio.to_thread(
        rerank_documents,
        request.query,
        retrieved,
        request.k or settings.rerank_top_n,
    )

    top_docs = [doc for doc, _ in reranked]
    sources = format_sources(top_docs)

    logger.info(
        "query_prepared query=%s expanded=%s retrieved=%s reranked=%s",
        request.query,
        expanded_queries,
        len(retrieved),
        len(top_docs),
    )

    return expanded_queries, retrieved, reranked, top_docs, sources


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    expanded_queries, retrieved, reranked, top_docs, sources = await _prepare_query(request)

    answer = await asyncio.to_thread(generate_answer, request.query, top_docs, settings)

    chunks: list[RetrievedChunk] | None = None
    if request.debug:
        chunks = [
            RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                source=doc.metadata.get("file_name", doc.metadata.get("source", "uploaded.pdf")),
                page=doc.metadata.get("page"),
                score=score,
                content=doc.page_content,
            )
            for doc, score in reranked
        ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        expanded_queries=expanded_queries,
        retrieved_count=len(retrieved),
        chunks=chunks,
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    expanded_queries, retrieved, reranked, top_docs, sources = await _prepare_query(request)

    async def event_generator():
        try:
            for token in stream_answer(request.query, top_docs, settings):
                payload = {"type": "token", "data": {"token": token}}
                yield json.dumps(payload) + "\n"

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
                        for doc, score in reranked
                    ]
                    if request.debug
                    else None,
                },
            }
            yield json.dumps(end_payload) + "\n"
        except Exception as exc:
            logger.exception("query_stream_failed")
            error_payload = {"type": "error", "data": {"message": str(exc)}}
            yield json.dumps(error_payload) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=settings.api_host, port=settings.api_port, reload=True)
