from support_agent.config import PROJECT_ROOT
from support_agent.knowledge import KnowledgeBase, load_markdown_sections
from support_agent.tickets import TicketStore

DATA = PROJECT_ROOT / "data"


def test_markdown_loader_splits_on_headings():
    docs = load_markdown_sections(DATA / "faq.md")
    assert len(docs) >= 10
    assert docs[0].title == "What are your working hours?"
    assert "9:00 AM" in docs[0].text


def test_search_finds_right_entry_even_with_typos():
    kb = KnowledgeBase.from_markdown(DATA / "troubleshooting.md")
    top = kb.search("pasword reset link not wrking", k=1)[0]
    assert "Password reset" in top.document.title


def test_search_threshold_filters_unrelated_queries():
    kb = KnowledgeBase.from_markdown(DATA / "faq.md")
    assert kb.search("quantum chromodynamics lattice", threshold=0.2) == []


def test_ticket_store_roundtrip():
    store = TicketStore(":memory:")
    t = store.create(session_id="s1", category="COMPLAINT", summary="late", priority="high")
    assert t.id.startswith("TKT-")
    assert store.get(t.id) == t
    assert store.list() == [t]
    assert store.get("TKT-NOPE") is None
