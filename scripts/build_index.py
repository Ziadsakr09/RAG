#!/usr/bin/env python
"""
scripts/build_index.py
-----------------------
One-shot script to process the dataset and build the FAISS index.
Run once before starting the API server.

Usage
-----
    python scripts/build_index.py [--max-samples N] [--force-rebuild]
"""

import argparse
import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.processor import DatasetProcessor
from src.embeddings.engine import EmbeddingEngine
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.monitoring.logging_config import setup_logging, get_logger
from configs.settings import settings

setup_logging()
logger = get_logger("build_index")


def main():
    parser = argparse.ArgumentParser(description="Build RAG FAISS index from Natural Questions dataset.")
    parser.add_argument("--max-samples", type=int, default=settings.MAX_DATASET_SAMPLES,
                        help="Maximum number of dataset samples to process")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild index even if one already exists")
    args = parser.parse_args()

    # ── Check existing index ─────────────────────────────────────────────────
    vector_store = VectorStore()
    if not args.force_rebuild:
        if vector_store.load():
            logger.info(f"Existing index loaded ({vector_store.size} vectors). Use --force-rebuild to recreate.")
            DatasetProcessor.print_stats([])  # just show stats
            print(f"\n✓ Index ready with {vector_store.size} vectors.")
            return

    # ── Process dataset ──────────────────────────────────────────────────────
    logger.info(f"Processing dataset (max_samples={args.max_samples})…")
    processor = DatasetProcessor(max_samples=args.max_samples)
    documents = processor.run()
    processor.save(documents)
    DatasetProcessor.print_stats(documents)

    # ── Build embeddings & index ─────────────────────────────────────────────
    engine = EmbeddingEngine.get_instance()
    vector_store.build_from_documents(documents, engine)

    # ── Fit keyword scorer ───────────────────────────────────────────────────
    retriever = Retriever(vector_store=vector_store, embedding_engine=engine)
    retriever.fit_keyword_scorer(documents)

    # ── Save ─────────────────────────────────────────────────────────────────
    vector_store.save()
    print(f"\n✓ Index built successfully: {vector_store.size} vectors at {settings.FAISS_INDEX_PATH}")
    print(f"✓ Processed data saved to {settings.PROCESSED_DATASET_PATH}")


if __name__ == "__main__":
    main()
