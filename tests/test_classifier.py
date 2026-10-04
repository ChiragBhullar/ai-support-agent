import pytest

from support_agent.classifier import (
    Category,
    KeywordClassifier,
    LLMClassifier,
    detect_sentiment,
    parse_category,
    wants_human,
)
from tests.conftest import FakeLLM


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FAQ", Category.FAQ),
        ("technical", Category.TECHNICAL),
        ("The category is COMPLAINT.", Category.COMPLAINT),
        ("  UNCLEAR\n", Category.UNCLEAR),
        ("I think this is a billing question", Category.UNCLEAR),  # no valid label
        ("", Category.UNCLEAR),
    ],
)
def test_parse_category_is_robust_to_messy_output(raw, expected):
    assert parse_category(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What are your working hours?", Category.FAQ),
        ("I can't log in to my account.", Category.TECHNICAL),
        ("pasword reset link is not wrking", Category.TECHNICAL),  # typos
        ("My issue is still not solved after 3 days.", Category.COMPLAINT),
        ("My payment failed but money was deducted", Category.COMPLAINT),  # two types
        ("It's not working.", Category.UNCLEAR),
        ("Help", Category.UNCLEAR),
        ("??", Category.UNCLEAR),
    ],
)
def test_keyword_classifier_report_examples(text, expected):
    assert KeywordClassifier().classify(text) == expected


def test_llm_classifier_uses_model_output():
    llm = FakeLLM(["TECHNICAL"])
    assert LLMClassifier(llm).classify("anything") == Category.TECHNICAL
    assert "Message: anything" in llm.calls[0][1]


def test_llm_classifier_falls_back_when_model_fails():
    clf = LLMClassifier(FakeLLM(fail=True))
    assert clf.classify("What are your working hours?") == Category.FAQ


def test_sentiment_and_human_request():
    assert detect_sentiment("This is the WORST, totally unacceptable!!") == "angry"
    assert detect_sentiment("How do I reset my password?") == "neutral"
    assert wants_human("Can I talk to a human please")
    assert wants_human("let me speak to someone")
    assert not wants_human("my phone number is wrong")
