from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vector_store_backend: Literal["local", "pinecone"] = "local"
    chroma_persist_directory: Path = Path(".chroma")
    chroma_collection: str = "agentic-ai-ebook"
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str = "agentic-ai-ebook"
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    pdf_url: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    pdf_path: Path = Path("data/Ebook-Agentic-AI.pdf")
    chunk_size: int = Field(default=800, ge=200, le=2000)
    chunk_overlap: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    min_relevance_score: float = Field(default=0.35, ge=0, le=1)

    def validate_runtime(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.vector_store_backend == "pinecone":
            if not self.pinecone_api_key or not self.pinecone_index_name:
                raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME are required for Pinecone")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for Pinecone mode")


@lru_cache
 def get_settings() -> Settings:
    return Settings()
