from fastapi.testclient import TestClient
from pydantic import SecretStr

from support_agent.api import create_app


def test_health(agent, settings):
    client = TestClient(create_app(agent, settings))
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_chat_and_ticket_lookup(agent, settings):
    client = TestClient(create_app(agent, settings))
    res = client.post("/v1/chat", json={"message": "My payment failed but money was deducted"})
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == "COMPLAINT"
    assert res.headers["x-request-id"]

    ticket = client.get(f"/v1/tickets/{data['ticket_id']}").json()
    assert ticket["category"] == "COMPLAINT"
    assert client.get("/v1/tickets/TKT-MISSING").status_code == 404


def test_validation_rejects_empty_and_bad_session(agent, settings):
    client = TestClient(create_app(agent, settings))
    assert client.post("/v1/chat", json={"message": ""}).status_code == 422
    bad = {"message": "hi", "session_id": "../etc/passwd"}
    assert client.post("/v1/chat", json=bad).status_code == 422


def test_api_key_is_enforced_when_configured(agent, settings):
    settings.api_key = SecretStr("s3cret")
    client = TestClient(create_app(agent, settings))
    assert client.post("/v1/chat", json={"message": "hours?"}).status_code == 401
    ok = client.post("/v1/chat", json={"message": "hours?"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
