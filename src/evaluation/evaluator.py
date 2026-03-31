"""
Phase 3 – Evaluation Framework
================================
Computes retrieval and generation quality metrics:

Retrieval
---------
* Precision@K  – fraction of retrieved docs that are relevant
* Recall@K     – fraction of relevant docs that were retrieved
* MRR          – Mean Reciprocal Rank
* NDCG@K       – Normalised Discounted Cumulative Gain

Generation
----------
* ROUGE-1/2/L  – n-gram overlap with reference answer
* BLEU         – modified n-gram precision (sacrebleu)
* Exact match  – strict string equality (normalised)
* Latency & throughput benchmarks
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)

# Optional imports (degrade gracefully)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(items):
        return sum(
            (1 if item in relevant else 0) / math.log2(i + 2)
            for i, item in enumerate(items[:k])
        )
    ideal_items = list(relevant)[:k]
    idcg = dcg(ideal_items + [""] * max(0, k - len(ideal_items)))
    return dcg(retrieved) / max(idcg, 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Generation metrics
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, reference: str) -> float:
    return float(_normalise(prediction) == _normalise(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_scores(prediction: str, reference: str) -> dict:
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def bleu_score(prediction: str, reference: str) -> float:
    if not BLEU_AVAILABLE:
        return 0.0
    try:
        result = sacrebleu.sentence_bleu(prediction, [reference])
        return round(result.score / 100, 4)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleEvalResult:
    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_doc_ids: list[str]
    # retrieval
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    # generation
    exact_match: float
    token_f1: float
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    # latency
    total_latency_ms: float


class RAGEvaluator:
    """
    Batch evaluation of the full RAG pipeline.

    Strategy for "relevant documents"
    -----------------------------------
    In the absence of document-level relevance labels (standard NQ format),
    we use a soft heuristic: a retrieved document is "relevant" if its
    source_answer has token_f1 > 0.3 against the ground-truth answer.
    This approximates relevance without requiring manual annotations.
    """

    def __init__(self, k_values: list[int] = None):
        self.k_values = k_values or [1, 3, 5]

    # ── Relevance heuristic ──────────────────────────────────────────────────

    @staticmethod
    def _is_relevant(doc_text: str, ground_truth: str) -> bool:
        return token_f1(doc_text, ground_truth) > 0.3

    # ── Single evaluation ────────────────────────────────────────────────────

    async def evaluate_one(
        self,
        question: str,
        ground_truth: str,
        retriever,
        generator,
        query_processor,
        top_k: int = 5,
    ) -> SingleEvalResult:
        t0 = time.time()

        processed = query_processor.process(question)
        results = retriever.retrieve(processed, top_k=top_k)
        response = await generator.generate(processed, results)

        latency_ms = (time.time() - t0) * 1000

        # Relevance labels
        relevant_ids = {
            r.document.doc_id
            for r in results
            if self._is_relevant(r.document.source_answer, ground_truth)
        }
        retrieved_ids = [r.document.doc_id for r in results]

        # Retrieval metrics (use max K from k_values)
        max_k = max(self.k_values)
        p_at_k = precision_at_k(retrieved_ids, relevant_ids, max_k)
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, max_k)
        mrr = reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, max_k)

        # Generation metrics
        rouge = rouge_scores(response.answer, ground_truth)
        bleu = bleu_score(response.answer, ground_truth)

        return SingleEvalResult(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=response.answer,
            retrieved_doc_ids=retrieved_ids,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg,
            exact_match=exact_match(response.answer, ground_truth),
            token_f1=token_f1(response.answer, ground_truth),
            rouge1=rouge["rouge1"],
            rouge2=rouge["rouge2"],
            rougeL=rouge["rougeL"],
            bleu=bleu,
            total_latency_ms=latency_ms,
        )

    # ── Batch evaluation ─────────────────────────────────────────────────────

    async def evaluate_batch(
        self,
        questions: list[str],
        ground_truths: list[str],
        retriever,
        generator,
        query_processor,
        top_k: int = 5,
    ) -> dict:
        logger.info(f"Evaluating {len(questions)} Q&A pairs…")
        t0 = time.time()

        tasks = [
            self.evaluate_one(q, gt, retriever, generator, query_processor, top_k)
            for q, gt in zip(questions, ground_truths)
        ]
        eval_results: list[SingleEvalResult] = await asyncio.gather(*tasks)

        total_time = time.time() - t0

        # Aggregate
        def _mean(values):
            return round(float(np.mean(values)), 4)

        def _p(attr):
            return [getattr(r, attr) for r in eval_results]

        latencies = _p("total_latency_ms")

        report = {
            "num_evaluated": len(eval_results),
            "retrieval_metrics": {
                f"precision@{max(self.k_values)}": _mean(_p("precision_at_k")),
                f"recall@{max(self.k_values)}": _mean(_p("recall_at_k")),
                "mrr": _mean(_p("mrr")),
                f"ndcg@{max(self.k_values)}": _mean(_p("ndcg_at_k")),
            },
            "generation_metrics": {
                "exact_match": _mean(_p("exact_match")),
                "token_f1": _mean(_p("token_f1")),
                "rouge1": _mean(_p("rouge1")),
                "rouge2": _mean(_p("rouge2")),
                "rougeL": _mean(_p("rougeL")),
                "bleu": _mean(_p("bleu")),
            },
            "latency_stats": {
                "mean_ms": _mean(latencies),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p90_ms": float(np.percentile(latencies, 90)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "throughput_qps": round(len(questions) / max(total_time, 0.001), 2),
            },
        }

        logger.info(f"Evaluation complete in {total_time:.1f}s: {report}")
        return report

    # ── Visualisation ────────────────────────────────────────────────────────

    @staticmethod
    def plot_metrics(report: dict, output_path: str = "./data/eval_report.png") -> str:
        """Generate a bar-chart summary of evaluation metrics."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("RAG Evaluation Report", fontsize=14, fontweight="bold")

            # Retrieval
            ret = report["retrieval_metrics"]
            sns.barplot(x=list(ret.keys()), y=list(ret.values()), ax=axes[0], palette="Blues_d")
            axes[0].set_title("Retrieval Metrics")
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis="x", rotation=15)

            # Generation
            gen = report["generation_metrics"]
            sns.barplot(x=list(gen.keys()), y=list(gen.values()), ax=axes[1], palette="Greens_d")
            axes[1].set_title("Generation Metrics")
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Evaluation report saved to {output_path}")
            return output_path
        except Exception as exc:
            logger.warning(f"Could not generate plot: {exc}")
            return ""
