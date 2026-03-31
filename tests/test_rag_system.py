"""
tests/test_rag_system.py
-------------------------
Comprehensive tests for all RAG system components.
Run with:  pytest tests/ -v
"""

import asyncio
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Data Processing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataProcessor:
    def test_chunk_text_basic(self):
        from src.data.processor import chunk_text
        text = "This is a sentence. " * 50
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(c) >= 50 for c in chunks)

    def test_chunk_text_short(self):
        from src.data.processor import chunk_text
        text = "Short text."
        chunks = chunk_text(text, chunk_size=512, overlap=64, min_length=5)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_classify_domain(self):
        from src.data.processor import _classify_domain, QuestionDomain
        assert _classify_domain("How do vaccines work?") == QuestionDomain.PROCEDURAL.value
        assert _classify_domain("Why is the sky blue?") == QuestionDomain.CAUSAL.value
        assert _classify_domain("Who invented the telephone?") == QuestionDomain.FACTUAL.value

    def test_classify_difficulty(self):
        from src.data.processor import _classify_difficulty, DifficultyLevel
        easy = _classify_difficulty("Who?", "Paris")
        hard = _classify_difficulty("Explain in detail the mechanisms by which XYZ operates?", "A" * 300)
        assert easy == DifficultyLevel.EASY.value
        assert hard == DifficultyLevel.HARD.value

    def test_synthetic_dataset_generation(self):
        from src.data.processor import _make_synthetic_dataset
        records = _make_synthetic_dataset(50)
        assert len(records) == 50

    def test_full_processing_pipeline(self):
        from src.data.processor import DatasetProcessor
        processor = DatasetProcessor(max_samples=20)
        documents = processor.run()
        assert len(documents) > 0
        assert all(hasattr(d, "doc_id") for d in documents)
        assert all(len(d.text) > 0 for d in documents)

    def test_to_dataframe(self):
        from src.data.processor import DatasetProcessor
        processor = DatasetProcessor(max_samples=10)
        documents = processor.run()
        df = DatasetProcessor.to_dataframe(documents)
        assert len(df) == len(documents)
        assert "doc_id" in df.columns
        assert "answer_type" in df.columns

    def test_document_serialisation(self):
        from src.data.processor import Document
        doc = Document(
            doc_id="test_001",
            text="Test text",
            source_question="Test question?",
            source_answer="Test answer",
        )
        d = doc.to_dict()
        assert d["doc_id"] == "test_001"
        assert "text" in d


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Vector Store Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def _make_docs_and_embeddings(self, n=20, dim=768):
        from src.data.processor import Document
        docs = [
            Document(
                doc_id=f"doc_{i}",
                text=f"Sample document number {i} about topic {i % 5}",
                source_question=f"Question {i}?",
                source_answer=f"Answer {i}",
            )
            for i in range(n)
        ]
        embeddings = np.random.rand(n, dim).astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return docs, embeddings

    def test_add_and_search(self, tmp_path):
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore(index_path=str(tmp_path / "index"), dimension=768)
        docs, embs = self._make_docs_and_embeddings(20)
        vs.add_documents(docs, embs)
        assert vs.size == 20

        query = embs[0]  # search with known vector
        results = vs.search(query, top_k=5, threshold=0.0)
        assert len(results) > 0
        # Top result should be exact match (score ≈ 1.0)
        assert results[0].score > 0.99

    def test_threshold_filtering(self, tmp_path):
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore(index_path=str(tmp_path / "index"), dimension=768)
        docs, embs = self._make_docs_and_embeddings(10)
        vs.add_documents(docs, embs)

        # Random query unlikely to match
        rng_query = np.random.rand(768).astype(np.float32)
        rng_query /= np.linalg.norm(rng_query)
        results = vs.search(rng_query, top_k=5, threshold=0.99)
        assert len(results) == 0  # nothing should be this similar

    def test_persistence(self, tmp_path):
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore(index_path=str(tmp_path / "index"), dimension=768)
        docs, embs = self._make_docs_and_embeddings(10)
        vs.add_documents(docs, embs)
        vs.save()

        vs2 = VectorStore(index_path=str(tmp_path / "index"), dimension=768)
        loaded = vs2.load()
        assert loaded
        assert vs2.size == 10

    def test_get_document(self, tmp_path):
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore(index_path=str(tmp_path / "index"), dimension=768)
        docs, embs = self._make_docs_and_embeddings(5)
        vs.add_documents(docs, embs)
        doc = vs.get_document("doc_2")
        assert doc is not None
        assert doc.doc_id == "doc_2"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Query Processing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryProcessor:
    def test_normalise(self):
        from src.retrieval.query_processor import QueryProcessor
        qp = QueryProcessor()
        assert qp.normalise("  What IS the capital?  ") == "what is the capital"
        assert qp.normalise("Hello, World!") == "hello world"

    def test_expand(self):
        from src.retrieval.query_processor import QueryProcessor
        qp = QueryProcessor()
        expanded = qp.expand("who invented the telephone")
        assert "person" in expanded or "individual" in expanded

    def test_process_basic(self):
        from src.retrieval.query_processor import QueryProcessor
        qp = QueryProcessor()
        result = qp.process("What is machine learning?", "session_1")
        assert result.original == "What is machine learning?"
        assert result.language  # should detect some language
        assert not result.is_followup  # first query, no context

    def test_followup_detection(self):
        from src.retrieval.query_processor import QueryProcessor
        qp = QueryProcessor()
        session = "test_session"
        qp.add_turn(session, "user", "What is Python?")
        qp.add_turn(session, "assistant", "Python is a programming language.")
        result = qp.process("Tell me more about it", session)
        assert result.is_followup

    def test_conversation_context(self):
        from src.retrieval.query_processor import ConversationContext
        ctx = ConversationContext(max_turns=4)
        ctx.add("user", "Hello")
        ctx.add("assistant", "Hi there!")
        assert not ctx.is_empty()
        snippet = ctx.condense()
        assert "User" in snippet
        assert "Assistant" in snippet


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Retriever Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetriever:
    def test_hybrid_retrieval(self, tmp_path):
        from src.data.processor import DatasetProcessor
        from src.embeddings.engine import EmbeddingEngine
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.retriever import Retriever
        from src.retrieval.query_processor import QueryProcessor

        # Build small index
        processor = DatasetProcessor(max_samples=30)
        docs = processor.run()

        engine = EmbeddingEngine()
        vs = VectorStore(index_path=str(tmp_path / "index"))
        vs.build_from_documents(docs, engine)

        retriever = Retriever(vs, engine)
        retriever.fit_keyword_scorer(docs)

        qp = QueryProcessor()
        processed = qp.process("What is the capital of France?")
        results = retriever.retrieve(processed, top_k=3)

        assert isinstance(results, list)
        # Results should be sorted by score descending
        if len(results) > 1:
            assert results[0].final_score >= results[1].final_score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Cache Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheLayer:
    def test_set_get(self):
        from src.cache.cache_layer import CacheLayer
        c = CacheLayer(max_size=100, ttl=60)
        c.set("key1", {"value": 42})
        result = c.get("key1")
        assert result == {"value": 42}

    def test_miss(self):
        from src.cache.cache_layer import CacheLayer
        c = CacheLayer(max_size=100, ttl=60)
        assert c.get("nonexistent") is None

    def test_delete(self):
        from src.cache.cache_layer import CacheLayer
        c = CacheLayer(max_size=100, ttl=60)
        c.set("key2", "hello")
        c.delete("key2")
        assert c.get("key2") is None

    def test_stats(self):
        from src.cache.cache_layer import CacheLayer
        c = CacheLayer(max_size=100, ttl=60)
        c.set("k", "v")
        c.get("k")  # hit
        c.get("missing")  # miss
        stats = c.stats()
        assert "hits_l1" in stats
        assert stats["sets"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 – Evaluation Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_precision_at_k(self):
        from src.evaluation.evaluator import precision_at_k
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        assert precision_at_k(retrieved, relevant, k=2) == 0.5  # 1/2 top-2 are relevant
        assert precision_at_k(retrieved, relevant, k=5) == 0.4  # 2/5

    def test_recall_at_k(self):
        from src.evaluation.evaluator import recall_at_k
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "e"}
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2/3, abs=0.01)

    def test_mrr(self):
        from src.evaluation.evaluator import reciprocal_rank
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1/3, abs=0.01)
        assert reciprocal_rank(["a"], {"a"}) == 1.0
        assert reciprocal_rank(["x"], {"a"}) == 0.0

    def test_exact_match(self):
        from src.evaluation.evaluator import exact_match
        assert exact_match("Paris", "paris") == 1.0
        assert exact_match("Paris", "London") == 0.0

    def test_token_f1(self):
        from src.evaluation.evaluator import token_f1
        assert token_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0
        assert token_f1("cat", "dog") == 0.0
        score = token_f1("the cat sat", "the cat on the mat")
        assert 0 < score < 1

    def test_ndcg(self):
        from src.evaluation.evaluator import ndcg_at_k
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        assert ndcg_at_k(retrieved, relevant, k=3) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Keyword Scorer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordScorer:
    def test_bm25_scoring(self):
        from src.retrieval.retriever import KeywordScorer
        corpus = [
            "machine learning algorithms and neural networks",
            "the capital of France is Paris",
            "photosynthesis converts light to energy in plants",
        ]
        scorer = KeywordScorer()
        scorer.fit(corpus)

        # Should score Paris doc higher for "capital France" query
        score_paris = scorer.score("capital France", corpus[1])
        score_ml = scorer.score("capital France", corpus[0])
        assert score_paris > score_ml

    def test_zero_score_for_unrelated(self):
        from src.retrieval.retriever import KeywordScorer
        scorer = KeywordScorer()
        scorer.fit(["hello world"])
        score = scorer.score("xyz abc def", "hello world")
        assert score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Async helper
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
