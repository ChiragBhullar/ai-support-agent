from __future__ import annotations

import pytest

from support_agent.agent import build_agent
from support_agent.config import Settings
from support_agent.tickets import TicketStore


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(llm_provider="offline", ticket_db_path=tmp_path / "t.db", api_key=None)


@pytest.fixture
def agent(settings):
    store = TicketStore(":memory:")
    yield build_agent(settings, tickets=store)
    store.close()


class FakeLLM:
    """Records calls and returns scripted outputs, so LLM paths are testable offline."""

    def __init__(self, outputs: list[str] | None = None, fail: bool = False) -> None:
        self.outputs = list(outputs or [])
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def invoke(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.fail:
            raise TimeoutError("simulated outage")
        return self.outputs.pop(0) if self.outputs else ""
