"""
Phase 3 – FastAPI Application
===============================
REST API exposing the RAG system with async support,
proper error handling, and OpenAPI documentation.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    init_db, get_session,
    Conversation, QueryRecord, ResponseRecord, AnalyticsRecord,
)
from src.data.processor import DatasetProcessor
from src.embeddings.engine import EmbeddingEngine
from src.retrieval.vector_store import VectorStore
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.retriever import Retriever
from src.generation.generator import ResponseGenerator, GroqClient
from src.cache.cache_layer import cache
from src.cache.semantic_cache import SemanticCache
from src.monitoring.metrics import metrics, RequestMetrics
from src.evaluation.evaluator import RAGEvaluator
from src.monitoring.logging_config import get_logger, setup_logging
from configs.settings import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application state (shared across requests)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    vector_store: VectorStore
    embedding_engine: EmbeddingEngine
    query_processor: QueryProcessor
    retriever: Retriever
    generator: ResponseGenerator
    evaluator: RAGEvaluator
    semantic_cache: SemanticCache
    ready: bool = False


app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting RAG system…")

    # DB
    await init_db()

    # Core components
    app_state.embedding_engine = EmbeddingEngine.get_instance()
    app_state.embedding_engine.load()  # Pre-load model on start
    app_state.vector_store = VectorStore()

    # Try loading existing index; build from scratch if needed
    if not app_state.vector_store.load():
        logger.info("No index found – processing dataset and building index…")
        processor = DatasetProcessor()
        documents = processor.run()
        processor.save(documents)
        app_state.vector_store.build_from_documents(documents, app_state.embedding_engine)
        app_state.vector_store.save()

    app_state.query_processor = QueryProcessor()
    app_state.retriever = Retriever(
        vector_store=app_state.vector_store,
        embedding_engine=app_state.embedding_engine,
    )
    app_state.generator = ResponseGenerator(llm_client=GroqClient())
    app_state.evaluator = RAGEvaluator()
    app_state.semantic_cache = SemanticCache()
    app_state.ready = True

    logger.info(
        f"RAG system ready. Index size: {app_state.vector_store.size} documents."
    )
    yield

    # Teardown
    logger.info("Shutting down RAG system.")
    app_state.vector_store.save()
    app_state.semantic_cache.save()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multilingual RAG System API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversations")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    language: Optional[str] = Field(None, description="Override detected language")


class SourceDocument(BaseModel):
    doc_id: str
    score: float
    snippet: str
    domain: str
    difficulty: str


class AskResponse(BaseModel):
    question_id: str
    session_id: str
    answer: str
    confidence: float
    sources: list[SourceDocument]
    is_fallback: bool
    is_semantic_cache: bool = False
    latency_ms: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
    cache_stats: dict
    metrics_summary: dict


class EvaluateRequest(BaseModel):
    questions: list[str]
    ground_truth_answers: list[str]
    top_k: int = Field(default=5, ge=1, le=20)


class EvaluateResponse(BaseModel):
    num_evaluated: int
    retrieval_metrics: dict
    generation_metrics: dict
    latency_stats: dict
    report_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ready():
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="System not yet ready. Please wait.")


async def _persist_query(
    db: AsyncSession,
    question_id: str,
    session_id: str,
    processed_query,
    response,
    analytics_data: dict,
) -> None:
    """Background task: persist Q&A to database."""
    try:
        # Ensure conversation exists
        conv = await db.get(Conversation, session_id)
        if conv is None:
            conv = Conversation(id=session_id)
            db.add(conv)

        qr = QueryRecord(
            id=question_id,
            conversation_id=session_id,
            original_query=processed_query.original,
            normalised_query=processed_query.normalised,
            expanded_query=processed_query.expanded,
            language=processed_query.language,
            is_followup=processed_query.is_followup,
        )
        db.add(qr)

        rr = ResponseRecord(
            query_id=question_id,
            answer=response.answer,
            confidence=response.confidence,
            is_fallback=response.is_fallback,
            model_used=response.model_used,
            sources=response.sources,
            retrieval_scores=response.retrieval_scores,
            latency_ms=response.latency_ms,
        )
        db.add(rr)

        ar = AnalyticsRecord(
            query_id=question_id,
            **analytics_data,
        )
        db.add(ar)

        await db.commit()
    except Exception as exc:
        logger.error(f"Failed to persist Q&A: {exc}")
        await db.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check – returns index size, cache, and request metrics."""
    _check_ready()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        index_size=app_state.vector_store.size,
        cache_stats=cache.stats(),
        metrics_summary=metrics.summary(),
    )


@app.post("/ask-question", response_model=AskResponse, tags=["Q&A"])
async def ask_question(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
):
    """
    Main Q&A endpoint.

    Flow
    ----
    1. Check cache for identical query
    2. Process & expand query
    3. Retrieve top-K documents via hybrid search
    4. Generate answer via LLM with retrieved context
    5. Persist to DB asynchronously
    6. Cache result
    """
    _check_ready()
    t_total = time.time()

    question_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())

    # ── Cache lookup ─────────────────────────────────────────────────────────
    cache_key = f"ask:{request.question}:{request.top_k}"
    cached = cache.get(cache_key)
    if cached:
        logger.debug("Cache hit for query.")
        cached["question_id"] = question_id
        cached["session_id"] = session_id
        return AskResponse(**cached)

    # ── Query processing (now including embedding for semantic cache) ────────
    t0 = time.time()
    processed = app_state.query_processor.process(request.question, session_id)
    
    # Pre-embed for semantic cache and retrieval
    query_vector = app_state.embedding_engine.embed_one(processed.expanded)
    embedding_latency = (time.time() - t0) * 1000

    # ── Semantic Cache lookup ────────────────────────────────────────────────
    if settings.SEMANTIC_CACHE_ENABLED:
        semantic_hit = app_state.semantic_cache.get(query_vector)
        if semantic_hit:
            logger.info("Semantic cache hit for query.")
            semantic_hit["question_id"] = question_id
            semantic_hit["session_id"] = session_id
            semantic_hit["latency_ms"] = round((time.time() - t_total) * 1000, 1)
            semantic_hit["is_semantic_cache"] = True
            return AskResponse(**semantic_hit)

    # ── Retrieval ────────────────────────────────────────────────────────────
    t0 = time.time()
    # Pass pre-computed query_vector to avoid re-embedding in retriever.retrieve
    results = app_state.retriever.retrieve_with_vector(query_vector, processed, top_k=request.top_k)
    retrieval_latency = (time.time() - t0) * 1000

    # ── Generation ───────────────────────────────────────────────────────────
    t0 = time.time()
    response = await app_state.generator.generate(processed, results)
    generation_latency = (time.time() - t0) * 1000

    total_latency = (time.time() - t_total) * 1000

    # ── Build API response ───────────────────────────────────────────────────
    sources = [
        SourceDocument(
            doc_id=r.document.doc_id,
            score=round(r.final_score, 4),
            snippet=r.document.text[:200],
            domain=r.document.domain,
            difficulty=r.document.difficulty,
        )
        for r in results
    ]

    api_response = AskResponse(
        question_id=question_id,
        session_id=session_id,
        answer=response.answer,
        confidence=round(response.confidence, 4),
        sources=sources,
        is_fallback=response.is_fallback,
        is_semantic_cache=False,
        latency_ms=round(total_latency, 1),
        model_used=response.model_used,
    )

    # ── Update conversation context ──────────────────────────────────────────
    app_state.query_processor.add_turn(session_id, "user", request.question)
    app_state.query_processor.add_turn(session_id, "assistant", response.answer)

    # ── Persist in background ────────────────────────────────────────────────
    analytics_data = {
        "embedding_latency_ms": embedding_latency,
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
        "total_latency_ms": total_latency,
        "num_results_retrieved": len(results),
        "top_retrieval_score": results[0].final_score if results else 0.0,
        "cache_hit": False,
    }
    background_tasks.add_task(
        _persist_query, db, question_id, session_id, processed, response, analytics_data
    )

    # ── Record metrics ───────────────────────────────────────────────────────
    metrics.record(RequestMetrics(
        endpoint="/ask-question",
        latency_ms=total_latency,
        status="fallback" if response.is_fallback else "success",
        retrieval_score=results[0].final_score if results else 0.0,
        num_results=len(results),
    ))

    # ── Cache result (Exact + Semantic) ──────────────────────────────────────
    cache.set(cache_key, api_response.model_dump())
    if settings.SEMANTIC_CACHE_ENABLED:
        app_state.semantic_cache.set(query_vector, api_response.model_dump())

    return api_response


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(request: EvaluateRequest):
    """
    Evaluate the RAG pipeline on a set of Q&A pairs.
    Returns retrieval metrics (precision@K, recall@K) and
    generation metrics (ROUGE, BLEU).
    """
    _check_ready()

    if len(request.questions) != len(request.ground_truth_answers):
        raise HTTPException(
            status_code=400,
            detail="questions and ground_truth_answers must have the same length.",
        )

    results = await app_state.evaluator.evaluate_batch(
        questions=request.questions,
        ground_truths=request.ground_truth_answers,
        retriever=app_state.retriever,
        generator=app_state.generator,
        query_processor=app_state.query_processor,
        top_k=request.top_k,
    )

    return EvaluateResponse(**results)


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Prometheus-style metrics endpoint."""
    _check_ready()
    return metrics.summary()
