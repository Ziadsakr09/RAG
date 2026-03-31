"""
Phase 1 – FAISS Vector Store
=============================
Manages the FAISS index and document metadata store.
Supports flat (exact) and IVF (approximate) indices, with
persistence to disk and incremental updates.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.data.processor import Document
from src.embeddings.engine import EmbeddingEngine
from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    document: Document
    score: float          # cosine similarity (higher = more similar)
    rank: int


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS-backed vector store with metadata preservation.

    Architecture
    ------------
    * FAISS index  → numerical vectors, fast ANN search
    * doc_store    → dict[int, Document], keyed by FAISS internal ID
    * id_map       → dict[str, int], doc_id → FAISS ID (for updates/deletes)

    Index strategy
    --------------
    * <10 K docs  → IndexFlatIP  (exact, no training required)
    * ≥10 K docs  → IndexIVFFlat (approximate, trained, faster at scale)
    Both use inner-product (= cosine if vectors are L2-normalised).
    """

    def __init__(
        self,
        index_path: str = settings.FAISS_INDEX_PATH,
        dimension: Optional[int] = None,
    ):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._dimension = dimension or settings.FAISS_INDEX_DIM
        self._index: Optional[faiss.Index] = None
        self._doc_store: dict[int, Document] = {}   # faiss_id → Document
        self._id_map: dict[str, int] = {}            # doc_id   → faiss_id
        self._next_id: int = 0

    # ── Index lifecycle ──────────────────────────────────────────────────────

    def _build_flat_index(self) -> faiss.Index:
        index = faiss.IndexFlatIP(self._dimension)
        # Wrap in IDMap so we can use our own integer IDs
        return faiss.IndexIDMap(index)

    def _build_ivf_index(self, n_docs: int) -> faiss.Index:
        nlist = min(settings.FAISS_NLIST, n_docs // 10)
        quantiser = faiss.IndexFlatIP(self._dimension)
        index = faiss.IndexIVFFlat(quantiser, self._dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIDMap(index)

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            self._index = self._build_flat_index()
        return self._index

    def _ensure_trained(self, vectors: np.ndarray) -> None:
        """Train IVF index if needed (no-op for flat index)."""
        if hasattr(self._index, "index") and hasattr(self._index.index, "is_trained"):
            inner = self._index.index
            if not inner.is_trained:
                logger.info(f"Training IVF index on {len(vectors)} vectors…")
                inner.train(vectors)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        batch_size: int = 1024,
    ) -> None:
        """
        Add documents and their pre-computed embeddings to the store.
        Call build_index() after adding all documents if you need IVF.
        """
        assert len(documents) == len(embeddings), "docs/embeddings length mismatch"

        n = len(documents)
        logger.info(f"Adding {n} documents to vector store…")

        # Switch to IVF if corpus exceeds threshold
        if n >= 10_000 and isinstance(self._index, faiss.IndexIDMap):
            logger.info("Corpus ≥10 K → switching to IVF index.")
            self._index = self._build_ivf_index(n)
            self._ensure_trained(embeddings.astype(np.float32))

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_vecs = embeddings[start:end].astype(np.float32)
            batch_docs = documents[start:end]
            ids = np.arange(self._next_id, self._next_id + len(batch_docs), dtype=np.int64)

            self.index.add_with_ids(batch_vecs, ids)

            for local_i, doc in enumerate(batch_docs):
                faiss_id = int(ids[local_i])
                self._doc_store[faiss_id] = doc
                self._id_map[doc.doc_id] = faiss_id

            self._next_id += len(batch_docs)

        logger.info(f"Index size: {self.index.ntotal} vectors")

    def build_from_documents(
        self,
        documents: list[Document],
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        """
        Full pipeline: embed + index.
        Use this when you don't have pre-computed embeddings.
        """
        engine = embedding_engine or EmbeddingEngine.get_instance()
        texts = [d.text for d in documents]
        logger.info(f"Embedding {len(texts)} documents…")
        t0 = time.time()
        embeddings = engine.embed(texts, show_progress=True)
        logger.info(f"Embedding done in {time.time()-t0:.1f}s")
        self.add_documents(documents, embeddings)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = settings.FAISS_TOP_K,
        threshold: float = settings.SIMILARITY_THRESHOLD,
    ) -> list[SearchResult]:
        """
        Return top-K documents sorted by descending cosine similarity.
        Documents below `threshold` are filtered out.
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty – no results returned.")
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(query, min(top_k * 2, self.index.ntotal))

        results: list[SearchResult] = []
        for rank, (score, faiss_id) in enumerate(zip(scores[0], ids[0])):
            if faiss_id == -1:
                continue
            if float(score) < threshold:
                break
            doc = self._doc_store.get(int(faiss_id))
            if doc is None:
                continue
            results.append(SearchResult(document=doc, score=float(score), rank=rank))
            if len(results) >= top_k:
                break

        return results

    def search_by_text(
        self,
        query: str,
        top_k: int = settings.FAISS_TOP_K,
        threshold: float = settings.SIMILARITY_THRESHOLD,
        engine: Optional[EmbeddingEngine] = None,
    ) -> list[SearchResult]:
        """Convenience: embed query then search."""
        emb_engine = engine or EmbeddingEngine.get_instance()
        vector = emb_engine.embed_one(query)
        return self.search(vector, top_k=top_k, threshold=threshold)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        idx_file = self.index_path / "faiss.index"
        meta_file = self.index_path / "metadata.pkl"

        faiss.write_index(self.index, str(idx_file))
        with open(meta_file, "wb") as fh:
            pickle.dump(
                {
                    "doc_store": self._doc_store,
                    "id_map": self._id_map,
                    "next_id": self._next_id,
                    "dimension": self._dimension,
                },
                fh,
            )
        logger.info(f"Saved index ({self.index.ntotal} vectors) to {self.index_path}")

    def load(self) -> bool:
        idx_file = self.index_path / "faiss.index"
        meta_file = self.index_path / "metadata.pkl"

        if not idx_file.exists() or not meta_file.exists():
            logger.info("No saved index found; starting fresh.")
            return False

        self._index = faiss.read_index(str(idx_file))
        with open(meta_file, "rb") as fh:
            meta = pickle.load(fh)
        self._doc_store = meta["doc_store"]
        self._id_map = meta["id_map"]
        self._next_id = meta["next_id"]
        self._dimension = meta["dimension"]
        logger.info(f"Loaded index with {self._index.ntotal} vectors from {self.index_path}")
        return True

    # ── Utility ──────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self.index.ntotal

    def get_document(self, doc_id: str) -> Optional[Document]:
        faiss_id = self._id_map.get(doc_id)
        if faiss_id is None:
            return None
        return self._doc_store.get(faiss_id)

    def stats(self) -> dict:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self._dimension,
            "index_type": type(self.index).__name__,
        }
