# Multilingual RAG System
[![GitHub Repo](https://img.shields.io/badge/GitHub-Ziadsakr09/RAG-blue?logo=github)](https://github.com/Ziadsakr09/RAG)

A production-ready **Retrieval-Augmented Generation** system built on top of the Natural Questions dataset. Supports multilingual queries, hybrid semantic search, LLM-powered answer generation, and comprehensive evaluation.

---

## Architecture Overview

```
rag_system/
├── configs/
│   └── settings.py            # Centralised config with env-var support
├── src/
│   ├── data/
│   │   └── processor.py       # Phase 1 – Dataset ingestion & chunking
│   ├── embeddings/
│   │   └── engine.py          # Phase 1 – Multilingual Sentence Transformers
│   ├── retrieval/
│   │   ├── vector_store.py    # Phase 1 – FAISS index management
│   │   ├── query_processor.py # Phase 2 – Query normalisation & expansion
│   │   └── retriever.py       # Phase 2 – Hybrid BM25 + vector retrieval + MMR
│   ├── generation/
│   │   └── generator.py       # Phase 2 – Groq LLM integration & prompting
│   ├── cache/
│   │   └── cache_layer.py     # Phase 2 – Two-tier cache (LRU + Redis)
│   ├── monitoring/
│   │   └── metrics.py         # Phase 2 – Prometheus metrics & request tracking
│   ├── db/
│   │   └── models.py          # Phase 3 – SQLAlchemy async ORM models
│   ├── api/
│   │   └── app.py             # Phase 3 – FastAPI endpoints
│   └── evaluation/
│       └── evaluator.py       # Phase 3 – BLEU, ROUGE, Precision@K, NDCG
├── scripts/
│   └── build_index.py         # One-shot index builder CLI
├── tests/
│   └── test_rag_system.py     # Full test suite (pytest)
├── notebooks/
│   └── rag_demo.ipynb         # Data exploration & demo notebook
├── main.py                    # Server entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

> [!WARNING]
> **Model Loading Latency**: Initial startup can take **up to 5 minutes** to pull and load the multilingual embedding models into memory. Please be patient while the logs show progress.

---

## Quick Start

### Option 1: Local Setup (Without Docker)

#### 1. Pre-requisites
- Python 3.11+
- Virtual environment tool (`venv` or `conda`)

#### 2. Install dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/MacOS:
source .venv/bin/activate

pip install -r requirements.txt
```

#### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — at minimum set your GROQ_API_KEY
```

#### 4. Build the index
```bash
# Processes dataset + builds FAISS index (~2 min)
python scripts/build_index.py --max-samples 1000
```

#### 5. Start the API server
```bash
python main.py
```

### Option 2: Docker Deployment

#### 1. Build and start
```bash
# Build and start all services (API + Redis)
docker-compose up --build -d
```

#### 2. Build index inside container
```bash
docker-compose exec rag-api python scripts/build_index.py --max-samples 1000
```

---

## API Reference

### `POST /ask-question`

Main Q&A endpoint.

**Request:**
```json
{
  "question": "Who invented the telephone?",
  "session_id": "optional-uuid-for-multi-turn",
  "top_k": 5
}
```

**Response:**
```json
{
  "question_id": "uuid",
  "session_id": "uuid",
  "answer": "Alexander Graham Bell invented the telephone in 1876.",
  "confidence": 0.82,
  "sources": [
    {
      "doc_id": "doc_000042_chunk_000",
      "score": 0.891,
      "snippet": "Question: Who invented...",
      "domain": "factual",
      "difficulty": "easy"
    }
  ],
  "is_fallback": false,
  "latency_ms": 312.4,
  "model_used": "llama3-8b-8192"
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "index_size": 12450,
  "cache_stats": {"hits_l1": 42, "hit_rate": 0.67},
  "metrics_summary": {"total_requests": 100, "p90_ms": 450}
}
```

### `POST /evaluate`

```json
{
  "questions": ["What is DNA?", "Who wrote Hamlet?"],
  "ground_truth_answers": ["Deoxyribonucleic acid", "William Shakespeare"],
  "top_k": 5
}
```

**Response:**
```json
{
  "num_evaluated": 2,
  "retrieval_metrics": {
    "precision@5": 0.62,
    "recall@5": 0.74,
    "mrr": 0.81,
    "ndcg@5": 0.78
  },
  "generation_metrics": {
    "exact_match": 0.0,
    "token_f1": 0.54,
    "rouge1": 0.61,
    "rougeL": 0.58,
    "bleu": 0.23
  },
  "latency_stats": {
    "mean_ms": 380,
    "p90_ms": 520,
    "throughput_qps": 2.6
  }
}
```

### `GET /metrics`

Returns aggregated performance statistics across all requests.

---

## Phase Details

### Phase 1 — Dataset Processing & RAG Foundation

| Component | Implementation |
|-----------|---------------|
| Dataset source | Google Natural Questions (HuggingFace hub or local JSONL) |
| Text chunking | Sentence-aware sliding window (configurable size + overlap) |
| Metadata enrichment | Answer type, domain, difficulty, language, token count |
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` (768-dim, 50+ languages) |
| Vector database | FAISS — flat index (<10K docs) or IVF (≥10K docs) |
| Normalisation | L2-normalised embeddings → cosine similarity via inner product |

### Phase 2 — Advanced RAG Features

| Component | Implementation |
|-----------|---------------|
| Query normalisation | Unicode, lowercase, punctuation stripping |
| Query expansion | Keyword → concept synonym mapping (who→person, how→method…) |
| Follow-up detection | Pronoun heuristic + context window |
| Retrieval | Two-stage: ANN (FAISS) → Hybrid rerank (vector + BM25) |
| Diversification | MMR (Maximal Marginal Relevance) with configurable λ |
| LLM integration | Groq API with retry (tenacity), context-aware prompts |
| Fallback chain | Low-confidence disclaimer → no-context prompt → template |
| Caching | Two-tier: in-memory LRU (L1) + Redis (L2, optional) |
| Monitoring | Prometheus counters/histograms + rolling window summary |

### Phase 3 — API & Evaluation

| Component | Implementation |
|-----------|---------------|
| Framework | FastAPI with async/await throughout |
| Database | SQLAlchemy 2.x async ORM (SQLite default, MySQL-ready) |
| Models | Conversation → QueryRecord → ResponseRecord + AnalyticsRecord |
| Retrieval metrics | Precision@K, Recall@K, MRR, NDCG@K |
| Generation metrics | ROUGE-1/2/L, BLEU (sacrebleu), Exact Match, Token F1 |
| Latency benchmarks | p50/p90/p99, throughput QPS |
| Visualisation | Matplotlib/Seaborn bar charts exported as PNG |

---

## Configuration Reference

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required for LLM generation) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | HuggingFace model name |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Index persistence directory |
| `FAISS_TOP_K` | `5` | Default retrieval results |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `LLM_MODEL` | `llama3-8b-8192` | Groq model name |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity for retrieval |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry time-to-live |
| `REDIS_URL` | — | Redis URL (falls back to in-memory) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./rag_system.db` | SQLAlchemy URL |
| `MAX_DATASET_SAMPLES` | `5000` | Cap on dataset samples (None = full) |
| `DEBUG` | `false` | Enable debug mode + hot reload |

---

## License
MIT
