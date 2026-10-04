"""Redact personal data before it reaches the LLM, the logs or the ticket store.

Customer messages often contain emails, phone numbers or card numbers. Under
India's DPDP Act, 2023 that is personal data, so we mask it at the entry point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


# Order matters: longer / more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
]


@dataclass
class RedactionResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)


def redact(text: str) -> RedactionResult:
    found: dict[str, int] = {}
    for label, pattern in _PATTERNS:

        def _sub(m: re.Match[str], label: str = label) -> str:
            raw = m.group(0)
            if label == "CARD":
                digits = re.sub(r"\D", "", raw)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return raw  # not a real card number; let other patterns try
            found[label] = found.get(label, 0) + 1
            return f"[{label}]"

        text = pattern.sub(_sub, text)
    return RedactionResult(text=text, found=found)
