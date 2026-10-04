"""End-to-end tests of the LangGraph workflow (offline mode)."""

from support_agent.agent import SupportAgent
from support_agent.classifier import KeywordClassifier
from support_agent.config import PROJECT_ROOT
from support_agent.graph import AgentDeps
from support_agent.knowledge import KnowledgeBase
from support_agent.tickets import TicketStore
from support_agent.writer import LLMReplyWriter
from tests.conftest import FakeLLM


def test_faq_is_answered_from_knowledge_base(agent):
    r = agent.handle("What are your working hours?")
    assert r.route == "answer_faq"
    assert "9:00 AM" in r.reply
    assert r.sources and r.ticket_id is None


def test_technical_gets_steps(agent):
    r = agent.handle("I can't log in to my account.")
    assert r.route == "technical_help"
    assert "Forgot password" in r.reply


def test_complaint_creates_ticket_with_priority(agent):
    r = agent.handle("This is the worst service, my issue is still not solved. Unacceptable!!")
    assert r.route == "create_ticket"
    ticket = agent.tickets.get(r.ticket_id)
    assert ticket.priority == "high"
    assert r.ticket_id in r.reply


def test_unclear_asks_then_uses_answer_in_context(agent):
    first = agent.handle("It's not working.", session_id="s1")
    assert first.route == "ask_clarification"
    # The follow-up is read together with the first message.
    second = agent.handle("the password reset link", session_id="s1")
    assert second.route == "technical_help"


def test_still_unclear_after_clarification_goes_to_human(agent):
    agent.handle("Help", session_id="s2")
    r = agent.handle("??", session_id="s2")
    assert r.route == "handoff"
    assert r.escalated and agent.tickets.get(r.ticket_id).escalated


def test_explicit_human_request_hands_off(agent):
    r = agent.handle("I want to talk to a human")
    assert r.route == "handoff"


def test_sessions_are_isolated(agent):
    agent.handle("Help", session_id="a")
    r = agent.handle("Help", session_id="b")
    assert r.route == "ask_clarification"  # 'b' has not been asked yet


def test_pii_never_reaches_ticket_store(agent):
    r = agent.handle("Money was deducted twice! my card 4111 1111 1111 1111, call 9876543210")
    ticket = agent.tickets.get(r.ticket_id)
    assert "4111" not in ticket.summary and "9876543210" not in ticket.summary
    assert r.pii_redacted == {"CARD": 1, "PHONE": 1}


def test_llm_writer_keeps_ticket_number_even_if_model_drops_it():
    llm = FakeLLM(outputs=["We're sorry. Our team will look into it."])
    deps = AgentDeps(
        classifier=KeywordClassifier(),
        writer=LLMReplyWriter(llm),
        faq_kb=KnowledgeBase.from_markdown(PROJECT_ROOT / "data" / "faq.md"),
        tech_kb=KnowledgeBase.from_markdown(PROJECT_ROOT / "data" / "troubleshooting.md"),
        tickets=TicketStore(":memory:"),
    )
    r = SupportAgent(deps).handle("Nobody has replied to me for a week!")
    deps.tickets.close()
    assert r.ticket_id in r.reply
