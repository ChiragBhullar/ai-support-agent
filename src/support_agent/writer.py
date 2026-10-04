"""The final 'write_reply' step: make every reply clear and polite without
changing its facts."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from support_agent.llm import ChatModel
from support_agent.prompts import REPLY_SYSTEM

log = logging.getLogger(__name__)
MAX_REPLY_CHARS = 1200


class ReplyWriter(Protocol):
    def write(self, draft: str, query: str) -> str: ...


class TemplateReplyWriter:
    """Offline writer: tidies whitespace and enforces the length cap."""

    def write(self, draft: str, query: str) -> str:
        text = re.sub(r"[ \t]+", " ", draft).strip()
        return text[:MAX_REPLY_CHARS]


class LLMReplyWriter:
    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self._fallback = TemplateReplyWriter()

    def write(self, draft: str, query: str) -> str:
        try:
            reply = self._model.invoke(
                REPLY_SYSTEM, f"Customer message:\n{query}\n\nDraft reply:\n{draft}"
            )
        except Exception:
            log.warning("llm_reply_failed; returning draft", exc_info=True)
            return self._fallback.write(draft, query)
        # Guardrail: the rewrite must keep any ticket number from the draft.
        for ticket_id in re.findall(r"TKT-[0-9A-F]{8}", draft):
            if ticket_id not in reply:
                reply += f"\n\nYour ticket number is {ticket_id}."
        return self._fallback.write(reply or draft, query)
