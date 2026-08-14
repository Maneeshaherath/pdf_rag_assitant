from __future__ import annotations

import json
import logging
from typing import Any

import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings

from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-ui")

settings = get_settings()


@st.cache_resource
def get_cached_embeddings() -> OpenAIEmbeddings:
    """Cache embeddings client across reruns."""

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        chunk_size=settings.embedding_batch_size,
        api_key=settings.openai_api_key or None,
    )


@st.cache_resource
def get_cached_vector_store() -> Chroma:
    """Cache vector store client across reruns."""

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_cached_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []


def upload_documents(api_base_url: str, files) -> None:
    payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    response = requests.post(f"{api_base_url}/upload", files=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    st.session_state.uploaded_docs = [item["file_name"] for item in data.get("ingested", [])]
    st.success("Files indexed successfully")


def fetch_documents(api_base_url: str) -> list[str]:
    response = requests.get(f"{api_base_url}/documents", timeout=30)
    response.raise_for_status()
    data = response.json()
    docs = data.get("documents", [])
    if not isinstance(docs, list):
        return []
    return [str(name) for name in docs]


def stream_query(api_base_url: str, query: str, doc_names: list[str], debug: bool):
    body = {
        "query": query,
        "doc_names": doc_names or None,
        "debug": debug,
        "k": settings.rerank_top_n,
        "fetch_k": settings.retrieval_fetch_k,
    }

    with requests.post(f"{api_base_url}/query/stream", json=body, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)


def render_sidebar() -> tuple[str, bool, list[str]]:
    st.sidebar.header("RAG Controls")
    api_base_url = st.sidebar.text_input("API URL", value=settings.api_base_url)
    debug = st.sidebar.checkbox("Debug retrieved chunks", value=settings.debug_chunks)

    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    available_docs = st.session_state.uploaded_docs
    try:
        available_docs = fetch_documents(api_base_url)
        st.session_state.uploaded_docs = available_docs
    except Exception:
        logger.exception("fetch_documents_failed")

    selected_docs = st.sidebar.multiselect(
        "Filter search to documents",
        options=available_docs,
        default=available_docs,
    )

    if st.sidebar.button("Index Documents", type="primary"):
        if not uploaded_files:
            st.sidebar.warning("Select at least one PDF")
        else:
            try:
                upload_documents(api_base_url, uploaded_files)
            except Exception as exc:
                logger.exception("upload_failed")
                st.sidebar.error(f"Upload failed: {exc}")

    try:
        vector_store = get_cached_vector_store()
        st.sidebar.caption(f"Local chunk count: {vector_store._collection.count()}")
    except Exception:
        st.sidebar.caption("Local chunk count unavailable")

    return api_base_url, debug, selected_docs


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + "; ".join(msg["sources"]))
            if msg.get("debug_chunks"):
                with st.expander("Retrieved Chunks"):
                    for chunk in msg["debug_chunks"]:
                        source = chunk.get("source", "uploaded.pdf")
                        page = chunk.get("page")
                        score = chunk.get("score")
                        st.markdown(f"Source: {source}, page: {page}, score: {score}")
                        st.write(chunk.get("content", ""))
                        st.divider()


def main() -> None:
    st.set_page_config(page_title="Production RAG Chat", page_icon="📄", layout="wide")
    st.title("Production PDF RAG Chat")
    st.caption("Upload PDFs in the sidebar, then ask questions in the chat input below.")

    init_state()
    api_base_url, debug, selected_docs = render_sidebar()
    render_chat()

    query = st.chat_input("Ask a question about your PDFs")
    if not query:
        return

    clean_query = query.strip()
    if not clean_query:
        st.warning("Please enter a non-empty query")
        return
    if len(clean_query) > settings.max_query_chars:
        st.warning(f"Query must be at most {settings.max_query_chars} characters")
        return

    st.session_state.messages.append({"role": "user", "content": clean_query})
    with st.chat_message("user"):
        st.markdown(clean_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        final_sources: list[str] = []
        final_chunks: list[dict[str, Any]] | None = None

        try:
            for event in stream_query(api_base_url, clean_query, selected_docs, debug):
                event_type = event.get("type")
                data = event.get("data", {})

                if event_type == "token":
                    token = data.get("token", "")
                    full_answer += token
                    placeholder.markdown(full_answer)
                elif event_type == "end":
                    final_sources = data.get("sources", [])
                    final_chunks = data.get("chunks")
                elif event_type == "error":
                    raise RuntimeError(data.get("message", "Unknown stream error"))

            if final_sources:
                st.caption("Sources: " + "; ".join(final_sources))

            if debug and final_chunks:
                with st.expander("Retrieved Chunks"):
                    for chunk in final_chunks:
                        st.markdown(
                            f"Source: {chunk.get('source')}, page: {chunk.get('page')}, score: {chunk.get('score')}"
                        )
                        st.write(chunk.get("content", ""))
                        st.divider()

            st.session_state.memory.save_context({"input": clean_query}, {"output": full_answer})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "sources": final_sources,
                    "debug_chunks": final_chunks,
                }
            )
        except Exception as exc:
            logger.exception("query_failed")
            st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
