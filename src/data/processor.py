"""
Phase 1 – Dataset Acquisition & Processing
==========================================
Downloads (or synthesises) the Natural Questions dataset, cleans it, and
produces a ready-to-index list of Document objects with rich metadata.

Supported sources
-----------------
1. HuggingFace `datasets` library  → google/natural_questions (preferred)
2. Local JSONL file                → set DATASET_PATH in settings
3. Synthetic mini-dataset          → used for unit-tests / demos
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class AnswerType(str, Enum):
    SHORT = "short"
    LONG = "long"
    BOOLEAN = "boolean"
    UNANSWERABLE = "unanswerable"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionDomain(str, Enum):
    FACTUAL = "factual"
    DESCRIPTIVE = "descriptive"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """
    Atomic unit stored in the vector index.

    Each Document corresponds to one *chunk* derived from a Q-A pair.
    Multiple Documents can share the same source_id (different chunks of
    the same long answer).
    """
    doc_id: str
    text: str                       # chunk text fed to the embedding model
    source_question: str
    source_answer: str
    answer_type: str = AnswerType.SHORT.value
    domain: str = QuestionDomain.FACTUAL.value
    difficulty: str = DifficultyLevel.MEDIUM.value
    language: str = "en"
    token_count: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Unicode normalise, collapse whitespace, strip HTML-like tags."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token."""
    return int(len(text.split()) / 0.75)


def _classify_domain(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["how", "step", "process", "procedure", "method"]):
        return QuestionDomain.PROCEDURAL.value
    if any(w in q for w in ["why", "cause", "reason", "because", "result"]):
        return QuestionDomain.CAUSAL.value
    if any(w in q for w in ["compare", "difference", "vs", "versus", "better"]):
        return QuestionDomain.COMPARATIVE.value
    if any(w in q for w in ["describe", "explain", "what is", "tell me about"]):
        return QuestionDomain.DESCRIPTIVE.value
    if any(w in q for w in ["who", "when", "where", "which", "name"]):
        return QuestionDomain.FACTUAL.value
    return QuestionDomain.UNKNOWN.value


def _classify_difficulty(question: str, answer: str) -> str:
    ans_len = len(answer.split())
    q_len = len(question.split())
    if ans_len < 10 and q_len < 8:
        return DifficultyLevel.EASY.value
    if ans_len > 100 or q_len > 15:
        return DifficultyLevel.HARD.value
    return DifficultyLevel.MEDIUM.value


def _classify_answer_type(short_answer: str, long_answer: str, yes_no: Optional[str]) -> str:
    if yes_no in ("yes", "no"):
        return AnswerType.BOOLEAN.value
    if short_answer:
        return AnswerType.SHORT.value
    if long_answer:
        return AnswerType.LONG.value
    return AnswerType.UNANSWERABLE.value


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
    min_length: int = settings.MIN_CHUNK_LENGTH,
) -> list[str]:
    """
    Sentence-aware sliding-window chunker.

    Strategy
    --------
    1. Split on sentence boundaries (`. `, `? `, `! `).
    2. Greedily fill a window of `chunk_size` characters.
    3. Slide by (chunk_size - overlap) characters.
    """
    # Sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunk = " ".join(current)
            if len(chunk) >= min_length:
                chunks.append(chunk)
            # carry-over for overlap
            overlap_text = " ".join(current)[-overlap:]
            current = [overlap_text, sentence] if overlap_text else [sentence]
            current_len = len(" ".join(current))
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunk = " ".join(current)
        if len(chunk) >= min_length:
            chunks.append(chunk)

    return chunks or [text[:chunk_size]]  # fallback: first chunk_size chars


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_huggingface(max_samples: Optional[int]) -> list[dict]:
    """Load Natural Questions via the HuggingFace datasets hub."""
    from datasets import load_dataset  # lazy import

    logger.info("Loading Natural Questions from HuggingFace…")
    ds = load_dataset(
        "google/natural_questions",
        split="train",
        trust_remote_code=True,
        streaming=True,     # stream to avoid OOM on the full 300K dataset
    )
    raw: list[dict] = []
    for i, example in enumerate(tqdm(ds, desc="Streaming NQ", total=max_samples)):
        if max_samples and i >= max_samples:
            break
        raw.append(example)
    return raw


def _load_from_jsonl(path: str, max_samples: Optional[int]) -> list[dict]:
    """Load from a local JSONL file (Kaggle download or custom)."""
    logger.info(f"Loading dataset from {path}…")
    raw = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="Reading JSONL")):
            if max_samples and i >= max_samples:
                break
            raw.append(json.loads(line))
    return raw


def _make_synthetic_dataset(n: int = 200) -> list[dict]:
    """Generate a lightweight synthetic dataset for demos & tests."""
    logger.warning("Using synthetic dataset – real data not found.")
    samples = [
        ("What is the capital of France?", "Paris", "Paris is the capital and largest city of France.", "yes_no", None),
        ("Who wrote Romeo and Juliet?", "William Shakespeare", "Romeo and Juliet is a tragedy written by William Shakespeare in the late 16th century.", None, None),
        ("How does photosynthesis work?", "Plants use sunlight to convert CO2 and water into glucose and oxygen.", "Photosynthesis is a process used by plants and other organisms to convert light energy, usually from the sun, into chemical energy that can be later released to fuel the organism's activities.", None, None),
        ("When did World War II end?", "1945", "World War II ended in 1945 with the surrender of Germany in May and Japan in September.", None, None),
        ("What is machine learning?", "A subset of AI that enables systems to learn from data.", "Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.", None, None),
        ("Why is the sky blue?", "Due to Rayleigh scattering of sunlight.", "The sky appears blue because of the scattering of sunlight by the atmosphere. When sunlight enters Earth's atmosphere, it collides with gas molecules, scattering the light.", None, None),
        ("What is the speed of light?", "299,792,458 meters per second", "The speed of light in a vacuum is exactly 299,792,458 metres per second (approximately 3×10⁸ m/s).", None, None),
        ("Who invented the telephone?", "Alexander Graham Bell", "Alexander Graham Bell is credited with inventing the first practical telephone in 1876.", None, None),
        ("What is DNA?", "Deoxyribonucleic acid, carrier of genetic information.", "DNA, or deoxyribonucleic acid, is the hereditary material in humans and almost all other organisms. Nearly every cell in a person's body has the same DNA.", None, None),
        ("How do vaccines work?", "They train the immune system to recognise pathogens.", "Vaccines work by training the immune system to recognize and combat pathogens, either viruses or bacteria. To do this, certain molecules from the pathogen must be introduced into the body.", None, None),
    ]
    # repeat and vary to reach n samples
    result = []
    for i in range(n):
        q, short, long, yn, _ = samples[i % len(samples)]
        result.append({
            "question": {"text": q},
            "annotations": [{
                "short_answers": [{"text": short}] if short else [],
                "long_answer": {"candidate_index": 0 if long else -1},
                "yes_no_answer": yn,
            }],
            "document": {"tokens": [{"token": w} for w in long.split()]} if long else {"tokens": []},
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Parser – converts raw NQ record → (question, short_answer, long_answer, yn)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_nq_record(record: dict) -> Optional[tuple[str, str, str, Optional[str]]]:
    """
    Parse one Natural Questions record.
    Returns (question, short_answer, long_answer, yes_no) or None if invalid.
    """
    try:
        question = _clean_text(record["question"]["text"])

        annotations = record.get("annotations", [{}])
        ann = annotations[0] if annotations else {}

        # Short answer
        short_answers = ann.get("short_answers", [])
        short_answer = ""
        if short_answers:
            sa = short_answers[0]
            if isinstance(sa, dict) and "text" in sa:
                short_answer = _clean_text(sa["text"])
            elif isinstance(sa, str):
                short_answer = _clean_text(sa)

        # Long answer (reconstruct from document tokens)
        long_answer = ""
        candidate_index = ann.get("long_answer", {}).get("candidate_index", -1)
        if candidate_index >= 0:
            tokens = record.get("document", {}).get("tokens", [])
            words = [t["token"] for t in tokens if isinstance(t, dict) and not t.get("html_token")]
            long_answer = _clean_text(" ".join(words[:400]))  # cap at 400 words

        yes_no = ann.get("yes_no_answer", None)
        if yes_no and yes_no.lower() in ("none", ""):
            yes_no = None

        if not question or (not short_answer and not long_answer):
            return None

        return question, short_answer, long_answer, yes_no

    except (KeyError, IndexError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DatasetProcessor:
    """
    Full dataset processing pipeline.

    Usage
    -----
    processor = DatasetProcessor()
    documents = processor.run()        # returns list[Document]
    processor.save(documents)          # persists to disk
    df = processor.to_dataframe(docs)  # EDA-friendly format
    """

    def __init__(
        self,
        max_samples: Optional[int] = settings.MAX_DATASET_SAMPLES,
        output_dir: str = settings.PROCESSED_DATASET_PATH,
    ):
        self.max_samples = max_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_raw(self) -> list[dict]:
        local_path = settings.DATASET_PATH
        if os.path.exists(local_path):
            return _load_from_jsonl(local_path, self.max_samples)
        try:
            return _load_from_huggingface(self.max_samples)
        except Exception as exc:
            logger.warning(f"HuggingFace load failed ({exc}); falling back to synthetic data.")
            return _make_synthetic_dataset(self.max_samples or 200)

    # ── Processing ───────────────────────────────────────────────────────────

    def process_record(self, record: dict, record_idx: int) -> list[Document]:
        """Convert one raw NQ record into one or more Documents."""
        parsed = _parse_nq_record(record)
        if parsed is None:
            return []

        question, short_answer, long_answer, yes_no = parsed
        answer_type = _classify_answer_type(short_answer, long_answer, yes_no)
        domain = _classify_domain(question)

        # Choose primary answer text for chunking
        primary_text = long_answer if answer_type == AnswerType.LONG.value else short_answer
        if not primary_text:
            primary_text = short_answer or long_answer

        difficulty = _classify_difficulty(question, primary_text)

        # Build text fed to the embedding model:
        # "Question: … Answer: …" format improves semantic alignment
        full_text = f"Question: {question}\nAnswer: {primary_text}"
        chunks = chunk_text(full_text)

        documents: list[Document] = []
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"doc_{record_idx:06d}_chunk_{chunk_idx:03d}"
            documents.append(Document(
                doc_id=doc_id,
                text=chunk,
                source_question=question,
                source_answer=primary_text,
                answer_type=answer_type,
                domain=domain,
                difficulty=difficulty,
                language="en",
                token_count=_estimate_tokens(chunk),
                chunk_index=chunk_idx,
                total_chunks=len(chunks),
                metadata={
                    "record_index": record_idx,
                    "has_short_answer": bool(short_answer),
                    "has_long_answer": bool(long_answer),
                    "yes_no_answer": yes_no,
                    "short_answer_preview": short_answer[:100] if short_answer else "",
                },
            ))
        return documents

    def run(self) -> list[Document]:
        logger.info("Starting dataset processing pipeline…")
        raw = self.load_raw()
        logger.info(f"Loaded {len(raw)} raw records.")

        documents: list[Document] = []
        skipped = 0
        for idx, record in enumerate(tqdm(raw, desc="Processing records")):
            docs = self.process_record(record, idx)
            if docs:
                documents.extend(docs)
            else:
                skipped += 1

        logger.info(
            f"Processing complete: {len(documents)} chunks from "
            f"{len(raw) - skipped} valid records ({skipped} skipped)."
        )
        return documents

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, documents: list[Document]) -> None:
        """Persist processed documents as JSONL and Parquet."""
        jsonl_path = self.output_dir / "documents.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

        df = self.to_dataframe(documents)
        df.to_parquet(self.output_dir / "documents.parquet", index=False)
        logger.info(f"Saved {len(documents)} documents to {self.output_dir}")

    @staticmethod
    def load_documents(path: str) -> list[Document]:
        """Load previously saved documents."""
        documents = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                d.pop("metadata", None)   # simplify; metadata stored separately
                documents.append(Document(**d))
        return documents

    # ── Analytics ────────────────────────────────────────────────────────────

    @staticmethod
    def to_dataframe(documents: list[Document]) -> pd.DataFrame:
        rows = [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "source_question": d.source_question,
                "source_answer": d.source_answer,
                "answer_type": d.answer_type,
                "domain": d.domain,
                "difficulty": d.difficulty,
                "language": d.language,
                "token_count": d.token_count,
                "chunk_index": d.chunk_index,
                "total_chunks": d.total_chunks,
            }
            for d in documents
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def print_stats(documents: list[Document]) -> None:
        df = DatasetProcessor.to_dataframe(documents)
        print("\n=== Dataset Statistics ===")
        print(f"Total chunks : {len(df)}")
        print(f"Unique questions : {df['source_question'].nunique()}")
        print(f"\nAnswer types:\n{df['answer_type'].value_counts()}")
        print(f"\nDomains:\n{df['domain'].value_counts()}")
        print(f"\nDifficulty:\n{df['difficulty'].value_counts()}")
        print(f"\nAvg tokens per chunk: {df['token_count'].mean():.1f}")
        print(f"Max tokens per chunk: {df['token_count'].max()}")
