"""
Phase 2 – Semantic Caching
==========================
Caches full Q&A responses using vector similarity.
Allows "near-miss" hits for paraphrased questions.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional, Any

import faiss
import numpy as np

from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


class SemanticCache:
    """
    FAISS-based semantic cache for storing and retrieving semantically similar queries.
    """

    def __init__(
        self,
        index_path: str = settings.SEMANTIC_CACHE_PATH,
        dimension: int = settings.FAISS_INDEX_DIM,
        threshold: float = settings.SEMANTIC_CACHE_THRESHOLD,
    ):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.threshold = threshold

        self._index: Optional[faiss.Index] = None
        self._cache_store: dict[int, dict[str, Any]] = {}  # faiss_id -> response_data
        self._next_id: int = 0
        
        # Load existing index if available
        self.load()

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            # We use IndexFlatIP for exact cosine similarity (since vectors are L2 normalised)
            # Wrapped in IndexIDMap to use our own incremental IDs
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        return self._index

    def get(self, query_vector: np.ndarray) -> Optional[dict[str, Any]]:
        """
        Search for a semantically similar query in the cache.
        Returns the cached response if similarity > threshold.
        """
        if self.index.ntotal == 0:
            return None

        # Ensure query vector is (1, D) and float32
        vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # Search for the single nearest neighbor
        scores, ids = self.index.search(vector, 1)
        
        score = float(scores[0][0])
        faiss_id = int(ids[0][0])

        if faiss_id != -1 and score >= self.threshold:
            logger.info(f"Semantic cache hit! Similarity: {score:.4f}")
            return self._cache_store.get(faiss_id)

        return None

    def set(self, query_vector: np.ndarray, response_data: dict[str, Any]) -> None:
        """
        Add a new query-response pair to the semantic cache.
        """
        vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss_id = self._next_id
        
        self.index.add_with_ids(vector, np.array([faiss_id], dtype=np.int64))
        self._cache_store[faiss_id] = response_data
        self._next_id += 1
        
        # Periodically save? For now, we'll save on each set to be safe, 
        # or rely on the app shutdown
        # self.save()

    def save(self) -> None:
        """Persist the semantic cache to disk."""
        if self._index is None:
            return

        idx_file = self.index_path / "semantic.index"
        meta_file = self.index_path / "metadata.pkl"

        faiss.write_index(self.index, str(idx_file))
        with open(meta_file, "wb") as fh:
            pickle.dump({
                "cache_store": self._cache_store,
                "next_id": self._next_id,
                "dimension": self.dimension,
            }, fh)
        logger.info(f"Saved semantic cache ({self.index.ntotal} entries) to {self.index_path}")

    def load(self) -> bool:
        """Load the semantic cache from disk."""
        idx_file = self.index_path / "semantic.index"
        meta_file = self.index_path / "metadata.pkl"

        if not idx_file.exists() or not meta_file.exists():
            return False

        try:
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file, "rb") as fh:
                meta = pickle.load(fh)
            self._cache_store = meta["cache_store"]
            self._next_id = meta["next_id"]
            self.dimension = meta["dimension"]
            logger.info(f"Loaded semantic cache with {self._index.ntotal} entries.")
            return True
        except Exception as exc:
            logger.error(f"Failed to load semantic cache: {exc}")
            return False

    def clear(self) -> None:
        """Clear all entries in the semantic cache."""
        self._index = None
        self._cache_store = {}
        self._next_id = 0
        logger.info("Semantic cache cleared.")
