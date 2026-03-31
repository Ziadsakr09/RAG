"""
Phase 2 – Query Processing & Enhancement
==========================================
Handles query normalisation, expansion, language detection,
and multi-turn conversation context management.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from src.monitoring.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    role: str       # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessedQuery:
    original: str
    normalised: str
    expanded: str               # normalised + expansion terms
    language: str
    is_followup: bool
    context_snippet: str        # condensed prior turns injected into prompt
    conversation_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Context Manager
# ─────────────────────────────────────────────────────────────────────────────

class ConversationContext:
    """
    Maintains a sliding window of conversation turns per session.

    Storage is in-memory; for production persistence use the DB layer.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        self._turns.append(ConversationTurn(role=role, content=content))

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def condense(self, max_chars: int = 800) -> str:
        """
        Return a compact string representation for prompt injection.
        Most recent turns are kept; older ones are summarised.
        """
        lines = []
        for t in self._turns:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content[:200]}")
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    def clear(self) -> None:
        self._turns.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Query Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

# Stopwords that can be removed for expansion purposes
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "it", "its",
}

# Question synonym expansions for better recall
_EXPANSION_MAP: dict[str, list[str]] = {
    "who": ["person", "individual", "author", "creator"],
    "what": ["definition", "description", "meaning"],
    "when": ["date", "year", "time", "period"],
    "where": ["location", "place", "country", "city"],
    "why": ["reason", "cause", "purpose", "explanation"],
    "how": ["method", "process", "procedure", "way"],
    "define": ["definition", "meaning", "explanation"],
    "explain": ["description", "explanation", "overview"],
    "difference": ["comparison", "contrast", "distinction"],
    "compare": ["comparison", "versus", "vs"],
    "example": ["instance", "case", "sample", "illustration"],
}


def _detect_language(text: str) -> str:
    """
    Lightweight language detection.
    Falls back to langdetect if available, else assumes 'en'.
    """
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _is_followup(query: str, context: ConversationContext) -> bool:
    """
    Heuristic: query is a follow-up if it contains pronouns or is very short
    and there is prior context.
    """
    if context.is_empty():
        return False
    followup_signals = {"it", "this", "that", "they", "he", "she", "its", "their",
                        "him", "her", "them", "those", "these"}
    tokens = set(query.lower().split())
    return bool(tokens & followup_signals) or len(query.split()) <= 4


class QueryProcessor:
    """
    Full query processing pipeline.

    Steps
    -----
    1. Normalise (lowercase, strip punct, fix spacing)
    2. Detect language
    3. Detect follow-up intent
    4. Expand with synonym/concept terms
    5. Inject conversation context snippet
    """

    def __init__(self, max_expansion_terms: int = 5):
        self.max_expansion_terms = max_expansion_terms
        # session_id → ConversationContext
        self._sessions: dict[str, ConversationContext] = {}

    # ── Session management ───────────────────────────────────────────────────

    def get_context(self, session_id: str) -> ConversationContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationContext()
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self.get_context(session_id).add(role, content)

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    # ── Normalisation ────────────────────────────────────────────────────────

    @staticmethod
    def normalise(query: str) -> str:
        """
        Clean and normalise a raw query string.
        """
        # Lowercase
        text = query.lower().strip()
        # Remove excessive punctuation but keep ? ! for intent signals
        text = re.sub(r"[^\w\s?!'-]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove trailing punctuation
        text = text.rstrip("?.!")
        return text

    # ── Expansion ────────────────────────────────────────────────────────────

    def expand(self, normalised: str) -> str:
        """
        Add semantically related terms to boost recall.
        Strategy: keyword → expansion terms from _EXPANSION_MAP.
        """
        tokens = normalised.split()
        expansions: list[str] = []
        seen: set[str] = set(tokens)

        for token in tokens:
            if token in _STOPWORDS:
                continue
            for term in _EXPANSION_MAP.get(token, []):
                if term not in seen:
                    expansions.append(term)
                    seen.add(term)
                if len(expansions) >= self.max_expansion_terms:
                    break

        if expansions:
            return normalised + " " + " ".join(expansions)
        return normalised

    # ── Main process ─────────────────────────────────────────────────────────

    def process(
        self,
        raw_query: str,
        session_id: Optional[str] = None,
    ) -> ProcessedQuery:
        normalised = self.normalise(raw_query)
        language = _detect_language(raw_query)
        context = self.get_context(session_id or "__default__")
        is_followup = _is_followup(normalised, context)

        # For follow-up queries, prepend last user question for grounding
        query_for_expansion = normalised
        if is_followup and not context.is_empty():
            last_user = next(
                (t.content for t in reversed(context.turns) if t.role == "user"),
                "",
            )
            query_for_expansion = f"{last_user} {normalised}"

        expanded = self.expand(query_for_expansion)
        context_snippet = context.condense() if not context.is_empty() else ""

        return ProcessedQuery(
            original=raw_query,
            normalised=normalised,
            expanded=expanded,
            language=language,
            is_followup=is_followup,
            context_snippet=context_snippet,
            conversation_id=session_id,
        )
