"""
RAG System Configuration
Centralizes all settings with environment variable support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Multilingual RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./rag_system.db"
    DATABASE_ECHO: bool = False

    # ── Vector Store ─────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    FAISS_INDEX_DIM: int = 768          # paraphrase-multilingual-mpnet-base-v2
    FAISS_TOP_K: int = 5
    FAISS_NLIST: int = 100              # IVF clusters (used when corpus > 10K)

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_MAX_LENGTH: int = 512

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_LENGTH: int = 50

    # ── LLM ──────────────────────────────────────────────────────────────────
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: int = 30

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    RERANK_TOP_K: int = 3

    # ── Cache ────────────────────────────────────────────────────────────────
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_SIZE: int = 1000
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.95
    SEMANTIC_CACHE_PATH: str = "./data/semantic_cache"

    # ── Dataset ──────────────────────────────────────────────────────────────
    DATASET_PATH: str = "./data/natural_questions.jsonl"
    PROCESSED_DATASET_PATH: str = "./data/processed"
    MAX_DATASET_SAMPLES: int = 5000     # set None for full dataset

    # ── Evaluation ───────────────────────────────────────────────────────────
    EVAL_BATCH_SIZE: int = 32
    EVAL_TOP_K_VALUES: list[int] = [1, 3, 5]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton
settings = Settings()
