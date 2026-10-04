"""Query classification, sentiment and handoff detection.

Two classifiers share one interface:
* LLMClassifier  - few-shot prompt, asks for one word, parses it safely.
* KeywordClassifier - deterministic rules; used offline, in CI, and as the
  fallback whenever the LLM call fails, so the agent never breaks.
"""

from __future__ import annotations

import difflib
import logging
import re
from enum import StrEnum
from typing import Protocol

from support_agent.llm import ChatModel
from support_agent.prompts import CLASSIFY_SYSTEM

log = logging.getLogger(__name__)


class Category(StrEnum):
    FAQ = "FAQ"
    TECHNICAL = "TECHNICAL"
    COMPLAINT = "COMPLAINT"
    UNCLEAR = "UNCLEAR"


def parse_category(raw: str) -> Category:
    """Find the category word anywhere in the model output.

    LLMs sometimes answer with a sentence ("The category is FAQ.") instead of one
    word. We scan for the first valid label and default to UNCLEAR if none is found.
    """
    for token in re.findall(r"[A-Za-z]+", raw.upper()):
        if token in Category.__members__:
            return Category(token)
    return Category.UNCLEAR


class Classifier(Protocol):
    def classify(self, text: str) -> Category: ...


_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _fuzzy_has(words: list[str], keywords: set[str], cutoff: float = 0.8) -> int:
    """Count words that match a keyword exactly or with a small typo
    ("pasword" -> "password"). Short words must match exactly."""
    hits = 0
    for w in words:
        close = len(w) >= 5 and difflib.get_close_matches(w, keywords, n=1, cutoff=cutoff)
        if w in keywords or close:
            hits += 1
    return hits


def _phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for p in phrases if p in text)


COMPLAINT_WORDS = {
    "unacceptable",
    "worst",
    "terrible",
    "disappointed",
    "frustrated",
    "pathetic",
    "complaint",
    "refund",
    "deducted",
    "overcharged",
    "ignored",
    "useless",
    "fraud",
    "scam",
    "horrible",
    "awful",
    "rude",
}
COMPLAINT_PHRASES = (
    "still not",
    "not been resolved",
    "not resolved",
    "no one",
    "nobody",
    "no reply",
    "no response",
    "charged twice",
    "money was deducted",
    "been waiting",
    "for a week",
    "days",
    "weeks",
    "again and again",
    "never replied",
)
TECH_WORDS = {
    "error",
    "crash",
    "crashes",
    "crashing",
    "login",
    "password",
    "otp",
    "bug",
    "install",
    "loading",
    "upload",
    "download",
    "freeze",
    "freezes",
    "sync",
    "reset",
    "verify",
    "verification",
    "update",
    "blank",
    "timeout",
    "app",
    "website",
    "account",
    "email",
    "notification",
    "notifications",
    "export",
    "connect",
    "screen",
    "page",
    "file",
}
TECH_PHRASES = (
    "not working",
    "doesn't work",
    "does not work",
    "can't",
    "cannot",
    "unable to",
    "won't",
    "log in",
    "sign in",
    "keeps",
    "stuck",
    "failed to",
    "fails",
    "is slow",
    "not loading",
    "not receiving",
    "isn't",
)
FAQ_WORDS = {
    "hours",
    "timings",
    "price",
    "pricing",
    "cost",
    "plans",
    "plan",
    "offer",
    "offers",
    "support",
    "located",
    "location",
    "office",
    "trial",
    "discount",
    "policy",
    "languages",
    "cancel",
    "upgrade",
    "gst",
    "invoice",
    "contact",
    "weekend",
    "weekends",
    "services",
}
FAQ_STARTS = (
    "what",
    "when",
    "where",
    "how much",
    "how many",
    "do you",
    "does",
    "is there",
    "can i",
    "which",
    "are you",
    "how do i",
    "how can i",
)
# Nouns that make a vague problem specific enough to act on.
SPECIFIC_NOUNS = TECH_WORDS - {"app", "page", "screen"}


class KeywordClassifier:
    """Rule-based classifier. Simple, explainable and has no API cost."""

    def classify(self, text: str) -> Category:
        t = " ".join(text.lower().split())
        words = _words(t)
        if len(words) == 0:
            return Category.UNCLEAR

        complaint = _fuzzy_has(words, COMPLAINT_WORDS) + _phrase_hits(t, COMPLAINT_PHRASES)
        tech = _fuzzy_has(words, TECH_WORDS) + _phrase_hits(t, TECH_PHRASES)
        faq = _fuzzy_has(words, FAQ_WORDS) + (2 if t.startswith(FAQ_STARTS) else 0)
        specific = _fuzzy_has(words, SPECIFIC_NOUNS) > 0

        # Rule 1: lost money or an unresolved issue is always a complaint.
        if complaint >= 1 and (complaint >= tech or "deducted" in t or "refund" in t):
            return Category.COMPLAINT
        # Rule 2: very short, vague messages need a question back.
        if len(words) <= 3 and not specific and faq < 2:
            return Category.UNCLEAR
        if tech == 0 and faq == 0:
            return Category.UNCLEAR
        # Rule 3: a question about the service beats a weak technical signal.
        if faq >= 2 and faq >= tech:
            return Category.FAQ
        if tech > 0:
            return Category.TECHNICAL
        return Category.FAQ


class LLMClassifier:
    """Few-shot LLM classifier with a deterministic fallback."""

    def __init__(self, model: ChatModel, fallback: Classifier | None = None) -> None:
        self._model = model
        self._fallback = fallback or KeywordClassifier()

    def classify(self, text: str) -> Category:
        try:
            raw = self._model.invoke(CLASSIFY_SYSTEM, f"Message: {text}")
        except Exception:  # network error, timeout, rate limit...
            log.warning("llm_classify_failed; using keyword fallback", exc_info=True)
            return self._fallback.classify(text)
        return parse_category(raw)


# ---------------------------------------------------------------- sentiment / handoff
ANGRY_WORDS = {
    "angry",
    "furious",
    "unacceptable",
    "worst",
    "terrible",
    "pathetic",
    "useless",
    "ridiculous",
    "fraud",
    "scam",
    "disgusting",
    "horrible",
    "awful",
    "hate",
    "never",
}
HUMAN_PATTERN = re.compile(
    r"\b(human|real person|representative|talk to (?:an? )?(?:agent|person|someone)|"
    r"speak to (?:an? )?(?:agent|person|someone|manager)|manager|supervisor)\b",
    re.IGNORECASE,
)


def detect_sentiment(text: str) -> str:
    """Return 'angry' or 'neutral' from a small lexicon plus tone markers."""
    score = sum(1 for w in _words(text) if w in ANGRY_WORDS)
    score += text.count("!") >= 2
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        score += 1  # SHOUTING
    return "angry" if score >= 2 else "neutral"


def wants_human(text: str) -> bool:
    return bool(HUMAN_PATTERN.search(text))
