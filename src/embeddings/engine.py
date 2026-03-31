"""
Phase 1 – Multilingual Embeddings
==================================
Wraps Sentence Transformers with batching, caching, and normalisation.
Supports any multilingual model from the HuggingFace hub.
"""

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Thread-safe multilingual embedding engine.

    Features
    --------
    * Explicit or lazy model loading (on-demand or via `.load()`)
    * L2-normalised embeddings (unit vectors) for cosine similarity via FAISS
    * Batch processing with progress bar
    * In-memory LRU cache for repeated queries
    """

    _instance: Optional["EmbeddingEngine"] = None  # singleton

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
        max_length: int = settings.EMBEDDING_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    # ── Singleton access ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Model management ─────────────────────────────────────────────────────

    def load(self) -> SentenceTransformer:
        """
        Explicitly load the model into memory.
        If already loaded, returns the existing model instance.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_length
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded in {time.time()-t0:.1f}s | "
                f"dimension={self._dimension}"
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.load()
        return self._dimension  # type: ignore[return-value]

    # ── Embedding ────────────────────────────────────────────────────────────

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of texts.

        Returns
        -------
        np.ndarray of shape (N, D), float32, L2-normalised.
        """
        model = self.load()

        all_embeddings: list[np.ndarray] = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches")

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2 normalise → cosine via dot
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings.astype(np.float32))

        return np.vstack(all_embeddings)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text with LRU caching."""
        return self._embed_cached(text)

    @lru_cache(maxsize=settings.CACHE_MAX_SIZE)
    def _embed_cached(self, text: str) -> np.ndarray:
        model = self.load()
        emb = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb[0].astype(np.float32)

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def text_fingerprint(text: str) -> str:
        """Stable fingerprint for cache keying."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (works because embeddings are L2-normalised)."""
        return float(np.dot(a, b))
