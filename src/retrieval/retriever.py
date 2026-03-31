"""
Phase 2 – Advanced Retriever
==============================
Combines vector search with BM25-style keyword scoring,
re-ranking, and MMR (Maximal Marginal Relevance) for diversity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.processor import Document
from src.embeddings.engine import EmbeddingEngine
from src.retrieval.vector_store import SearchResult, VectorStore
from src.retrieval.query_processor import ProcessedQuery
from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    document: Document
    vector_score: float
    keyword_score: float
    final_score: float
    rank: int


# ─────────────────────────────────────────────────────────────────────────────
# BM25-style keyword scorer (lightweight, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordScorer:
    """
    TF-IDF inspired keyword scorer for re-ranking.
    Pre-computes IDF from the corpus if given; otherwise uses uniform IDF.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 100.0

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, corpus: list[str]) -> None:
        """Compute IDF weights from a corpus of documents."""
        N = len(corpus)
        df: Counter = Counter()
        total_len = 0
        for text in corpus:
            tokens = set(self._tokenise(text))
            df.update(tokens)
            total_len += len(self._tokenise(text))

        self._avg_dl = total_len / max(N, 1)
        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, document: str) -> float:
        query_tokens = self._tokenise(query)
        doc_tokens = self._tokenise(document)
        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)

        bm25 = 0.0
        for term in query_tokens:
            idf = self._idf.get(term, math.log(2))  # fallback IDF
            freq = tf.get(term, 0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
            bm25 += idf * numerator / max(denominator, 1e-6)

        return bm25


# ─────────────────────────────────────────────────────────────────────────────
# MMR diversification
# ─────────────────────────────────────────────────────────────────────────────

def maximal_marginal_relevance(
    query_vec: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list,
    top_k: int,
    lambda_mult: float = 0.5,
) -> list:
    """
    Select top_k items by MMR to balance relevance and diversity.

    lambda_mult=1.0  →  pure relevance (no diversity)
    lambda_mult=0.0  →  pure diversity (maximum spread)
    """
    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    # Precompute query similarities
    query_sims = candidate_embeddings @ query_vec  # (N,)

    while len(selected_indices) < top_k and remaining:
        if not selected_indices:
            # First: pick highest relevance
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            # MMR: balance relevance vs redundancy
            sel_embs = candidate_embeddings[selected_indices]  # (S, D)
            best = -1
            best_score = -np.inf
            for i in remaining:
                rel = query_sims[i]
                red = float(np.max(candidate_embeddings[i] @ sel_embs.T))
                score = lambda_mult * rel - (1 - lambda_mult) * red
                if score > best_score:
                    best_score = score
                    best = i

        selected_indices.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Main Retriever
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Two-stage retriever:
    Stage 1 – ANN search via FAISS (recall-optimised, large candidate pool)
    Stage 2 – Hybrid rerank: vector_score * α + keyword_score * (1-α)
    Optional: MMR diversification of final results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: Optional[EmbeddingEngine] = None,
        alpha: float = 0.7,        # weight for vector score in hybrid
        use_mmr: bool = True,
        mmr_lambda: float = 0.6,
    ):
        self.vector_store = vector_store
        self.engine = embedding_engine or EmbeddingEngine.get_instance()
        self.alpha = alpha
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.kw_scorer = KeywordScorer()
        self._corpus_fitted = False

    def fit_keyword_scorer(self, documents: list[Document]) -> None:
        """Pre-fit the IDF weights on the indexed corpus."""
        texts = [d.text for d in documents]
        self.kw_scorer.fit(texts)
        self._corpus_fitted = True
        logger.info(f"Keyword scorer fitted on {len(texts)} documents.")

    def retrieve(
        self,
        processed_query: ProcessedQuery,
        top_k: int = settings.RETRIEVAL_TOP_K,
        recall_multiplier: int = 4,     # over-fetch factor for stage-2
        query_vec: Optional[np.ndarray] = None,
    ) -> list[RetrievalResult]:
        """
        Full retrieval pipeline.

        Parameters
        ----------
        processed_query   : output of QueryProcessor.process()
        top_k             : final number of results to return
        recall_multiplier : how many more candidates to fetch for reranking
        query_vec         : optional pre-computed embedding vector
        """
        # ── Stage 1: ANN search ──────────────────────────────────────────────
        if query_vec is None:
            query_vec = self.engine.embed_one(processed_query.expanded)
        
        candidates: list[SearchResult] = self.vector_store.search(
            query_vec,
            top_k=top_k * recall_multiplier,
            threshold=settings.SIMILARITY_THRESHOLD,
        )

        if not candidates:
            return []

        # ── Stage 2: Hybrid rerank ───────────────────────────────────────────
        query_text = processed_query.normalised
        results: list[RetrievalResult] = []

        # Normalise vector scores to [0,1]
        vec_scores = np.array([c.score for c in candidates])
        vec_min, vec_max = vec_scores.min(), vec_scores.max()
        if vec_max > vec_min:
            vec_scores = (vec_scores - vec_min) / (vec_max - vec_min)

        # Compute and normalise keyword scores
        kw_scores = np.array([
            self.kw_scorer.score(query_text, c.document.text) for c in candidates
        ])
        kw_min, kw_max = kw_scores.min(), kw_scores.max()
        if kw_max > kw_min:
            kw_scores = (kw_scores - kw_min) / (kw_max - kw_min)

        final_scores = self.alpha * vec_scores + (1 - self.alpha) * kw_scores

        for i, candidate in enumerate(candidates):
            results.append(RetrievalResult(
                document=candidate.document,
                vector_score=float(vec_scores[i]),
                keyword_score=float(kw_scores[i]),
                final_score=float(final_scores[i]),
                rank=i,
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)

        # ── Optional MMR diversification ─────────────────────────────────────
        if self.use_mmr and len(results) > top_k:
            # Embed top candidates for MMR
            candidate_texts = [r.document.text for r in results]
            cand_embs = self.engine.embed(candidate_texts)
            diversified = maximal_marginal_relevance(
                query_vec=query_vec,
                candidate_embeddings=cand_embs,
                candidates=results,
                top_k=top_k,
                lambda_mult=self.mmr_lambda,
            )
            # Re-assign ranks
            for rank, r in enumerate(diversified):
                r.rank = rank
            return diversified

        # Re-rank and slice
        for rank, r in enumerate(results[:top_k]):
            r.rank = rank
        return results[:top_k]

    def retrieve_with_vector(
        self,
        query_vec: np.ndarray,
        processed_query: ProcessedQuery,
        top_k: int = settings.RETRIEVAL_TOP_K,
    ) -> list[RetrievalResult]:
        """Convenience method for when the query vector is already available."""
        return self.retrieve(processed_query, top_k=top_k, query_vec=query_vec)
