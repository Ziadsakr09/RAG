"""
Phase 2 – Performance Monitoring
==================================
Collects latency, throughput, and retrieval quality metrics.
Exposes Prometheus counters/histograms and an in-process summary.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

    REGISTRY = CollectorRegistry()

    REQUEST_COUNT = Counter(
        "rag_requests_total", "Total RAG requests", ["endpoint", "status"],
        registry=REGISTRY,
    )
    REQUEST_LATENCY = Histogram(
        "rag_request_latency_seconds", "Request latency in seconds",
        ["endpoint"], registry=REGISTRY,
        buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    )
    RETRIEVAL_SCORE = Histogram(
        "rag_retrieval_score", "Top-1 retrieval score",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        registry=REGISTRY,
    )
    CACHE_HIT = Counter(
        "rag_cache_hits_total", "Cache hit counter", ["tier"],
        registry=REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class RequestMetrics:
    endpoint: str
    latency_ms: float
    status: str          # "success" | "error" | "fallback"
    retrieval_score: float = 0.0
    num_results: int = 0
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    Keeps a rolling window for live dashboards.
    """

    def __init__(self, window_size: int = 1000):
        self._window: deque[RequestMetrics] = deque(maxlen=window_size)
        self._lock = Lock()

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._window.append(metrics)

        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(
                endpoint=metrics.endpoint, status=metrics.status
            ).inc()
            REQUEST_LATENCY.labels(endpoint=metrics.endpoint).observe(
                metrics.latency_ms / 1000
            )
            if metrics.retrieval_score > 0:
                RETRIEVAL_SCORE.observe(metrics.retrieval_score)

    @contextmanager
    def measure(self, endpoint: str):
        """Context manager that auto-records latency."""
        t0 = time.time()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            latency = (time.time() - t0) * 1000
            self.record(RequestMetrics(
                endpoint=endpoint,
                latency_ms=latency,
                status=status,
            ))

    def summary(self) -> dict:
        with self._lock:
            window = list(self._window)

        if not window:
            return {"message": "No data collected yet."}

        latencies = [m.latency_ms for m in window]
        latencies.sort()

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        success = [m for m in window if m.status == "success"]
        errors = [m for m in window if m.status == "error"]
        fallbacks = [m for m in window if m.status == "fallback"]

        scores = [m.retrieval_score for m in window if m.retrieval_score > 0]

        return {
            "total_requests": len(window),
            "success_rate": round(len(success) / max(len(window), 1), 3),
            "error_rate": round(len(errors) / max(len(window), 1), 3),
            "fallback_rate": round(len(fallbacks) / max(len(window), 1), 3),
            "latency_ms": {
                "p50": round(percentile(latencies, 50), 1),
                "p90": round(percentile(latencies, 90), 1),
                "p99": round(percentile(latencies, 99), 1),
                "mean": round(sum(latencies) / max(len(latencies), 1), 1),
            },
            "retrieval_score": {
                "mean": round(sum(scores) / max(len(scores), 1), 3),
                "min": round(min(scores, default=0), 3),
                "max": round(max(scores, default=0), 3),
            },
        }


# Global singleton
metrics = MetricsCollector()
