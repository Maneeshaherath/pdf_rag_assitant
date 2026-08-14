from __future__ import annotations

from collections.abc import Generator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Settings


def get_chat_model(settings: Settings, streaming: bool = False) -> ChatOpenAI:
    kwargs: dict = {
        "model": settings.chat_model,
        "temperature": 0,
        "max_retries": 2,
        "timeout": 60,
        "streaming": streaming,
    }
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return ChatOpenAI(**kwargs)


def _build_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("file_name", doc.metadata.get("source", "uploaded.pdf"))
        page = doc.metadata.get("page")
        page_label = f" page {int(page) + 1}" if page is not None else ""
        blocks.append(f"[{i}] {source}{page_label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _messages(query: str, docs: list[Document]) -> list:
    strict_system = (
        "You are a strict retrieval QA assistant. "
        "Answer ONLY using the provided context. "
        "If the context is insufficient, respond exactly with: I don't know. "
        "Do not invent facts. "
        "Cite sources at the end as [n] markers that match context blocks."
    )
    user_prompt = (
        "Context:\n"
        f"{_build_context(docs)}\n\n"
        f"Question: {query}\n"
        "Return a concise answer followed by a Sources line."
    )
    return [SystemMessage(content=strict_system), HumanMessage(content=user_prompt)]


def generate_answer(query: str, docs: list[Document], settings: Settings) -> str:
    llm = get_chat_model(settings, streaming=False)
    response = llm.invoke(_messages(query, docs))
    return str(response.content)


def stream_answer(query: str, docs: list[Document], settings: Settings) -> Generator[str, None, None]:
    llm = get_chat_model(settings, streaming=True)
    for chunk in llm.stream(_messages(query, docs)):
        text = getattr(chunk, "content", "")
        if text:
            yield str(text)


def format_sources(docs: list[Document]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        source = doc.metadata.get("file_name", doc.metadata.get("source", "uploaded.pdf"))
        page = doc.metadata.get("page")
        if page is not None:
            label = f"{source} (page {int(page) + 1})"
        else:
            label = str(source)
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return unique
