"""The LangGraph workflow.

    ingest -> classify --(route)--> answer_faq ------\
                               |--> technical_help --\
                               |--> create_ticket ----+--> write_reply -> END
                               |--> ask_clarification /
                               \\--> handoff ---------/

Every node is a plain function that reads the shared state and returns only the
fields it changes. A checkpointer keeps state per session (thread_id), which is
how the agent remembers it already asked a clarifying question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from support_agent.classifier import Category, Classifier, detect_sentiment, wants_human
from support_agent.knowledge import KnowledgeBase
from support_agent.llm import ChatModel
from support_agent.pii import redact
from support_agent.prompts import GROUNDED_ANSWER_SYSTEM
from support_agent.tickets import TicketStore
from support_agent.writer import ReplyWriter

log = logging.getLogger(__name__)


class SupportState(TypedDict, total=False):
    # input
    session_id: str
    query: str
    # set by the agent
    redacted_query: str
    pii_found: dict[str, int]
    category: str
    sentiment: str
    wants_human: bool
    route: str
    context: list[dict[str, Any]]
    draft: str
    reply: str
    ticket_id: str | None
    escalated: bool
    # remembered across turns of one session
    pending_query: str | None
    clarification_attempts: int
    turn: int


@dataclass
class AgentDeps:
    classifier: Classifier
    writer: ReplyWriter
    faq_kb: KnowledgeBase
    tech_kb: KnowledgeBase
    tickets: TicketStore
    llm: ChatModel | None = None
    retrieval_threshold: float = 0.2
    top_k: int = 3
    max_clarifications: int = 1


HANDOFF_MSG = (
    "I'm passing your conversation to a member of our support team so they can help "
    "you directly. Your ticket number is {ticket}. Someone will reply to you on this "
    "conversation."
)


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None):
    # ------------------------------------------------------------------ nodes
    def ingest(state: SupportState) -> SupportState:
        redaction = redact(state["query"])
        text = redaction.text
        # If last turn we asked a clarifying question, read the answer together
        # with the original message so the classifier sees the full picture.
        if state.get("pending_query"):
            text = f"{state['pending_query']} {text}"
        return {
            "redacted_query": text,
            "pii_found": redaction.found,
            "context": [],
            "draft": "",
            "reply": "",
            "ticket_id": None,
            "escalated": False,
            "turn": state.get("turn", 0) + 1,
            "clarification_attempts": state.get("clarification_attempts", 0),
        }

    def classify(state: SupportState) -> SupportState:
        text = state["redacted_query"]
        update: SupportState = {
            "category": deps.classifier.classify(text).value,
            "sentiment": detect_sentiment(text),
            "wants_human": wants_human(text),
        }
        # Store the routing decision in state so it is visible in traces and the API.
        update["route"] = route({**state, **update})
        log.info("classified", extra={"category": update["category"], "route": update["route"]})
        return update

    def route(state: SupportState) -> str:
        if state.get("wants_human"):
            return "handoff"
        category = state["category"]
        if category == Category.UNCLEAR:
            if (
                state.get("clarification_attempts", 0) >= deps.max_clarifications
                or state.get("sentiment") == "angry"
            ):
                return "handoff"
            return "ask_clarification"
        return {
            Category.FAQ: "answer_faq",
            Category.TECHNICAL: "technical_help",
            Category.COMPLAINT: "create_ticket",
        }[Category(category)]

    def _grounded(kb: KnowledgeBase, query: str) -> tuple[str | None, list[dict]]:
        hits = kb.search(query, k=deps.top_k, threshold=deps.retrieval_threshold)
        context = [
            {"id": h.document.id, "title": h.document.title, "score": round(h.score, 3)}
            for h in hits
        ]
        if not hits:
            return None, context
        if deps.llm is None:  # offline: return the best matching entry
            return hits[0].document.text, context
        joined = "\n\n".join(f"### {h.document.title}\n{h.document.text}" for h in hits)
        try:
            answer = deps.llm.invoke(GROUNDED_ANSWER_SYSTEM.format(context=joined), query)
        except Exception:
            log.warning("llm_grounded_answer_failed", exc_info=True)
            return hits[0].document.text, context
        return (None if "I_DONT_KNOW" in answer else answer), context

    def _resolved() -> SupportState:
        return {"pending_query": None, "clarification_attempts": 0}

    def answer_faq(state: SupportState) -> SupportState:
        answer, context = _grounded(deps.faq_kb, state["redacted_query"])
        draft = answer or (
            "I'm sorry, I don't have that information. If you'd like, reply 'talk to a "
            "human' and our team will help you."
        )
        return {"draft": draft, "context": context, **_resolved()}

    def technical_help(state: SupportState) -> SupportState:
        answer, context = _grounded(deps.tech_kb, state["redacted_query"])
        if answer:
            draft = (
                f"Sorry you're having trouble. Please try these steps:\n{answer}\n\n"
                "If this doesn't fix it, reply here and we'll raise a ticket."
            )
            return {"draft": draft, "context": context, **_resolved()}
        ticket = deps.tickets.create(
            session_id=state["session_id"],
            category=Category.TECHNICAL.value,
            summary=state["redacted_query"],
        )
        draft = (
            "Sorry you're having trouble. I couldn't find a guide for this exact "
            f"problem, so I've raised ticket {ticket.id} for our technical team."
        )
        return {"draft": draft, "context": context, "ticket_id": ticket.id, **_resolved()}

    def create_ticket(state: SupportState) -> SupportState:
        priority = "high" if state.get("sentiment") == "angry" else "normal"
        ticket = deps.tickets.create(
            session_id=state["session_id"],
            category=Category.COMPLAINT.value,
            summary=state["redacted_query"],
            priority=priority,
        )
        draft = (
            "I'm sorry about this experience. I've created ticket "
            f"{ticket.id} ({priority} priority) and our team will look into it."
        )
        return {"draft": draft, "ticket_id": ticket.id, **_resolved()}

    def ask_clarification(state: SupportState) -> SupportState:
        return {
            "draft": "Could you tell me a little more? For example, what exactly is not "
            "working, or what you were trying to do when it happened?",
            "pending_query": state["redacted_query"],
            "clarification_attempts": state.get("clarification_attempts", 0) + 1,
        }

    def handoff(state: SupportState) -> SupportState:
        ticket = deps.tickets.create(
            session_id=state["session_id"],
            category=state.get("category", "UNCLEAR"),
            summary=state["redacted_query"],
            priority="high",
            escalated=True,
        )
        return {
            "draft": HANDOFF_MSG.format(ticket=ticket.id),
            "ticket_id": ticket.id,
            "escalated": True,
            **_resolved(),
        }

    def write_reply(state: SupportState) -> SupportState:
        return {"reply": deps.writer.write(state["draft"], state["redacted_query"])}

    # ------------------------------------------------------------------ wiring
    graph = StateGraph(SupportState)
    graph.add_node("ingest", ingest)
    graph.add_node("classify", classify)
    for name, fn in [
        ("answer_faq", answer_faq),
        ("technical_help", technical_help),
        ("create_ticket", create_ticket),
        ("ask_clarification", ask_clarification),
        ("handoff", handoff),
        ("write_reply", write_reply),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify")
    graph.add_conditional_edges(
        "classify",
        lambda s: s["route"],
        ["answer_faq", "technical_help", "create_ticket", "ask_clarification", "handoff"],
    )
    for step in ["answer_faq", "technical_help", "create_ticket", "ask_clarification", "handoff"]:
        graph.add_edge(step, "write_reply")
    graph.add_edge("write_reply", END)
    return graph.compile(checkpointer=checkpointer)
