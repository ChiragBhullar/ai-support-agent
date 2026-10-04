"""Public entry point: build the agent from settings and handle one message."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver

from support_agent.classifier import KeywordClassifier, LLMClassifier
from support_agent.config import Settings, get_settings
from support_agent.graph import AgentDeps, build_graph
from support_agent.knowledge import KnowledgeBase
from support_agent.llm import build_chat_model
from support_agent.tickets import TicketStore
from support_agent.writer import LLMReplyWriter, TemplateReplyWriter


@dataclass
class AgentResult:
    session_id: str
    reply: str
    category: str
    route: str
    sentiment: str
    ticket_id: str | None
    escalated: bool
    pii_redacted: dict[str, int] = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    latency_ms: int = 0


class SupportAgent:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        # InMemorySaver keeps per-session state for this process. For several
        # replicas, swap in a shared checkpointer (Postgres / Redis).
        self._graph = build_graph(deps, checkpointer=InMemorySaver())

    @property
    def tickets(self) -> TicketStore:
        return self.deps.tickets

    def handle(self, message: str, session_id: str | None = None) -> AgentResult:
        session_id = session_id or uuid.uuid4().hex
        start = time.perf_counter()
        state = self._graph.invoke(
            {"query": message, "session_id": session_id},
            config={"configurable": {"thread_id": session_id}},
        )
        return AgentResult(
            session_id=session_id,
            reply=state["reply"],
            category=state["category"],
            route=state["route"],
            sentiment=state["sentiment"],
            ticket_id=state.get("ticket_id"),
            escalated=state.get("escalated", False),
            pii_redacted=state.get("pii_found", {}),
            sources=state.get("context", []),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def build_agent(
    settings: Settings | None = None, tickets: TicketStore | None = None
) -> SupportAgent:
    settings = settings or get_settings()
    llm = build_chat_model(settings)
    deps = AgentDeps(
        classifier=LLMClassifier(llm) if llm else KeywordClassifier(),
        writer=LLMReplyWriter(llm) if llm else TemplateReplyWriter(),
        faq_kb=KnowledgeBase.from_markdown(settings.data_dir / "faq.md"),
        tech_kb=KnowledgeBase.from_markdown(settings.data_dir / "troubleshooting.md"),
        tickets=tickets or TicketStore(settings.ticket_db_path),
        llm=llm,
        retrieval_threshold=settings.retrieval_threshold,
        top_k=settings.retrieval_top_k,
        max_clarifications=settings.max_clarifications,
    )
    return SupportAgent(deps)
