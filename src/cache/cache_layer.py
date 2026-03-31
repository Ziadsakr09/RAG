"""
Phase 2 – Caching Layer
========================
Two-tier cache:
  L1 – in-memory LRU (fast, process-local)
  L2 – Redis (optional, shared across workers)

Falls back gracefully to L1-only when Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional
from cachetools import TTLCache
from threading import Lock

from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


def _cache_key(*args: Any, **kwargs: Any) -> str:
    """Deterministic hash key from arbitrary arguments."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class CacheLayer:
    """
    Thread-safe two-tier cache.

    Usage
    -----
    cache = CacheLayer()
    value = cache.get("my_key")
    cache.set("my_key", value, ttl=600)
    cache.delete("my_key")
    """

    def __init__(
        self,
        max_size: int = settings.CACHE_MAX_SIZE,
        ttl: int = settings.CACHE_TTL_SECONDS,
        redis_url: Optional[str] = settings.REDIS_URL,
    ):
        self._l1: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
        self._lock = Lock()
        self._default_ttl = ttl
        self._redis = None
        self._stats = {"hits_l1": 0, "hits_l2": 0, "misses": 0, "sets": 0}

        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Redis cache connected.")
            except Exception as exc:
                logger.warning(f"Redis unavailable ({exc}); using in-memory cache only.")
                self._redis = None

    # ── Core operations ──────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        # L1
        with self._lock:
            value = self._l1.get(key)
        if value is not None:
            self._stats["hits_l1"] += 1
            return value

        # L2 (Redis)
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    value = json.loads(raw)
                    with self._lock:
                        self._l1[key] = value    # populate L1
                    self._stats["hits_l2"] += 1
                    return value
            except Exception as exc:
                logger.debug(f"Redis get error: {exc}")

        self._stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._default_ttl
        with self._lock:
            self._l1[key] = value
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
            except Exception as exc:
                logger.debug(f"Redis set error: {exc}")
        self._stats["sets"] += 1

    def delete(self, key: str) -> None:
        with self._lock:
            self._l1.pop(key, None)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self._l1.clear()
        if self._redis:
            try:
                self._redis.flushdb()
            except Exception:
                pass

    # ── Decorator ────────────────────────────────────────────────────────────

    def cached(self, ttl: Optional[int] = None):
        """
        Decorator for sync functions.

        @cache.cached(ttl=300)
        def expensive_fn(arg):
            ...
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                key = f"{fn.__qualname__}:{_cache_key(*args, **kwargs)}"
                result = self.get(key)
                if result is None:
                    result = fn(*args, **kwargs)
                    self.set(key, result, ttl=ttl)
                return result
            return wrapper
        return decorator

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = sum(self._stats.values()) - self._stats["sets"]
        hit_rate = (
            (self._stats["hits_l1"] + self._stats["hits_l2"]) / max(total, 1)
        )
        return {**self._stats, "hit_rate": round(hit_rate, 3)}


# Global singleton
cache = CacheLayer()
