"""
Phase 2 – Response Generation
==============================
Integrates with Groq LLM API (or any OpenAI-compatible endpoint).
Handles:
* Context-aware prompt engineering
* Response quality validation
* Fallback for low-confidence / empty retrievals
* Retry logic with exponential back-off
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.retrieval.retriever import RetrievalResult
from src.retrieval.query_processor import ProcessedQuery
from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedResponse:
    answer: str
    confidence: float           # 0.0 – 1.0
    sources: list[str]          # doc_ids of supporting documents
    retrieval_scores: list[float]
    latency_ms: float
    is_fallback: bool = False
    model_used: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions accurately and concisely.
Use ONLY the provided context to formulate your answer.
If the context does not contain enough information, say so clearly rather than guessing.
Always be factual, precise, and helpful."""

_QA_TEMPLATE = """\
### Context
{context}

### Conversation History
{history}

### Question
{question}

### Instructions
Answer the question using only the context above. Be concise and accurate.
If the context is insufficient, respond: "I don't have enough information to answer that confidently."

### Answer
"""

_FALLBACK_TEMPLATE = """\
### Question
{question}

### Note
No relevant documents were retrieved for this query.
Answer based on your general knowledge, but clearly indicate this is not from retrieved documents.

### Answer
"""


def _build_context(results: list[RetrievalResult], max_chars: int = 2500) -> str:
    """Format retrieved documents into a numbered context block."""
    parts: list[str] = []
    total = 0
    for i, r in enumerate(results, 1):
        snippet = r.document.text[:600]
        part = f"[{i}] (score={r.final_score:.3f})\n{snippet}"
        total += len(part)
        if total > max_chars:
            break
        parts.append(part)
    return "\n\n".join(parts) if parts else "No context available."


def _estimate_confidence(results: list[RetrievalResult]) -> float:
    """
    Derive confidence from retrieval scores.
    High-quality retrieval → higher confidence.
    """
    if not results:
        return 0.0
    top_score = results[0].final_score
    avg_score = sum(r.final_score for r in results) / len(results)
    # Weighted: top score matters more
    return min(1.0, 0.7 * top_score + 0.3 * avg_score)


def _validate_response(text: str, min_length: int = 10) -> bool:
    """Basic quality filter: non-empty, not a refusal boilerplate."""
    if not text or len(text.strip()) < min_length:
        return False
    refusal_phrases = ["i cannot", "i'm unable", "as an ai", "i don't have access"]
    lower = text.lower()
    return not any(p in lower for p in refusal_phrases)


# ─────────────────────────────────────────────────────────────────────────────
# Groq Client (thin async wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class GroqClient:
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = settings.LLM_MODEL,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
        timeout: int = settings.LLM_TIMEOUT,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.BASE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    # Sync convenience (for scripts / notebooks)
    def complete_sync(self, system: str, user: str) -> str:
        import asyncio
        return asyncio.run(self.complete(system, user))


# ─────────────────────────────────────────────────────────────────────────────
# Response Generator
# ─────────────────────────────────────────────────────────────────────────────

class ResponseGenerator:
    """
    Orchestrates retrieval results → LLM prompt → validated answer.

    Fallback chain
    --------------
    1. Normal generation with retrieved context
    2. If confidence < threshold → append "low confidence" disclaimer
    3. If no results → fallback prompt (no context injection)
    4. If LLM unavailable → template-based answer from top document
    """

    CONFIDENCE_THRESHOLD = 0.35
    FALLBACK_CONFIDENCE_THRESHOLD = 0.15

    def __init__(self, llm_client: Optional[GroqClient] = None):
        self.llm = llm_client or GroqClient()

    async def generate(
        self,
        processed_query: ProcessedQuery,
        results: list[RetrievalResult],
    ) -> GeneratedResponse:
        t0 = time.time()
        confidence = _estimate_confidence(results)

        try:
            if results:
                answer = await self._generate_with_context(processed_query, results)
            else:
                answer = await self._generate_fallback(processed_query)
                confidence = max(confidence, 0.1)

            # Quality gate
            if not _validate_response(answer):
                answer = self._template_fallback(processed_query, results)
                confidence = max(0.0, confidence - 0.2)

            # Low-confidence disclaimer
            if confidence < self.CONFIDENCE_THRESHOLD and results:
                answer = (
                    f"[Low confidence – please verify]\n{answer}"
                )

        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}. Using template fallback.")
            answer = self._template_fallback(processed_query, results)
            confidence = max(0.0, confidence - 0.3)

        latency_ms = (time.time() - t0) * 1000
        return GeneratedResponse(
            answer=answer,
            confidence=confidence,
            sources=[r.document.doc_id for r in results],
            retrieval_scores=[r.final_score for r in results],
            latency_ms=latency_ms,
            is_fallback=not bool(results),
            model_used=self.llm.model,
        )

    async def _generate_with_context(
        self,
        query: ProcessedQuery,
        results: list[RetrievalResult],
    ) -> str:
        context = _build_context(results)
        prompt = _QA_TEMPLATE.format(
            context=context,
            history=query.context_snippet or "None",
            question=query.original,
        )
        logger.info(f"Generated prompt: {prompt}")
        return await self.llm.complete(system=_SYSTEM_PROMPT, user=prompt)

    async def _generate_fallback(self, query: ProcessedQuery) -> str:
        prompt = _FALLBACK_TEMPLATE.format(question=query.original)
        return await self.llm.complete(system=_SYSTEM_PROMPT, user=prompt)

    @staticmethod
    def _template_fallback(
        query: ProcessedQuery, results: list[RetrievalResult]
    ) -> str:
        if results:
            best = results[0].document
            return (
                f"Based on the available information:\n\n"
                f"{best.source_answer}\n\n"
                f"(Source: {best.source_question[:80]})"
            )
        return (
            "I was unable to find a relevant answer in the knowledge base. "
            "Please try rephrasing your question."
        )
