#!/usr/bin/env python
"""
main.py
--------
Entry point for the Multilingual RAG System.
Starts the FastAPI server with uvicorn.

Usage
-----
    # Start API server
    python main.py

    # Build index first (if not already done)
    python scripts/build_index.py --max-samples 1000

    # Run tests
    pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
from configs.settings import settings


if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1,  # single worker for shared in-memory state (FAISS index)
    )
