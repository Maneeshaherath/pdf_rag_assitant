from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class UploadResponseItem(BaseModel):
    file_name: str
    doc_hash: str
    chunks_added: int
    total_chunks: int


class UploadResponse(BaseModel):
    ingested: list[UploadResponseItem]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    doc_names: list[str] | None = None
    k: int | None = None
    fetch_k: int | None = None
    debug: bool = False

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return value.strip()


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    score: float | None = None
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    expanded_queries: list[str]
    retrieved_count: int
    chunks: list[RetrievedChunk] | None = None


class StreamEvent(BaseModel):
    type: str
    data: dict[str, Any]
