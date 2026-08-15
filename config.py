from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Central runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="pdf_chunks", alias="CHROMA_COLLECTION")

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    retrieval_k: int = Field(default=8, alias="RETRIEVAL_K")
    retrieval_fetch_k: int = Field(default=25, alias="RETRIEVAL_FETCH_K")
    rerank_top_n: int = Field(default=4, alias="RERANK_TOP_N")
    rerank_min_score: float = Field(default=0.0, alias="RERANK_MIN_SCORE")
    rag_max_retries: int = Field(default=1, alias="RAG_MAX_RETRIES")
    bm25_k: int = Field(default=8, alias="BM25_K")
    query_expansions: int = Field(default=3, alias="QUERY_EXPANSIONS")

    max_query_chars: int = Field(default=500, alias="MAX_QUERY_CHARS")
    max_upload_mb: int = Field(default=15, alias="MAX_UPLOAD_MB")

    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")

    debug_chunks: bool = Field(default=False, alias="DEBUG_CHUNKS")

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="pdf-rag", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT")
    langsmith_workspace_id: str = Field(default="", alias="LANGSMITH_WORKSPACE_ID")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def separators(self) -> list[str]:
        return ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


def _streamlit_secret_key() -> str:
    """Read OPENAI_API_KEY from Streamlit secrets if present (local UI setup)."""

    path = Path(".streamlit/secrets.toml")
    if not path.exists():
        return ""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    value = data.get("OPENAI_API_KEY") or data.get("openai_api_key") or ""
    return str(value).strip()


def _looks_like_langsmith_key(value: str) -> bool:
    key = value.strip().strip('"').strip("'")
    if not key or key in {"<your-api-key>", "your-api-key", "lsv2_..."}:
        return False
    return key.startswith("lsv") or key.startswith("ls-")


def _export_langsmith(settings: Settings) -> None:
    """LangSmith/LangChain tracers read os.environ, not pydantic Settings."""

    project = settings.langsmith_project.strip().strip('"').strip("'")
    endpoint = settings.langsmith_endpoint.strip().strip('"').strip("'")
    key = settings.langsmith_api_key.strip().strip('"').strip("'")
    workspace = settings.langsmith_workspace_id.strip().strip('"').strip("'")

    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
    if workspace:
        os.environ["LANGSMITH_WORKSPACE_ID"] = workspace
        os.environ["LANGCHAIN_WORKSPACE_ID"] = workspace
    if _looks_like_langsmith_key(key):
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGCHAIN_API_KEY"] = key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton settings instance and export keys for OpenAI and LangSmith."""

    if not os.getenv("OPENAI_API_KEY"):
        secret = _streamlit_secret_key()
        if secret:
            os.environ["OPENAI_API_KEY"] = secret

    settings = Settings()
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    _export_langsmith(settings)
    return settings
