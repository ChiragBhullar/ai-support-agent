"""HTTP API (FastAPI).

POST /v1/chat            send a customer message, get the agent's reply
GET  /v1/tickets         list recent tickets
GET  /v1/tickets/{id}    fetch one ticket
GET  /health             liveness probe
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from support_agent import __version__
from support_agent.agent import SupportAgent, build_agent
from support_agent.config import Settings, get_settings
from support_agent.logging_config import configure_logging

log = logging.getLogger("support_agent.api")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, examples=["I can't log in to my account"])
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    category: str
    route: str
    sentiment: str
    ticket_id: str | None
    escalated: bool
    pii_redacted: dict[str, int]
    sources: list[dict]
    latency_ms: int


def create_app(agent: SupportAgent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    agent = agent or build_agent(settings)
    app = FastAPI(title="AI Customer Support Agent", version=__version__)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        if x_api_key is None or not secrets.compare_digest(
            x_api_key, settings.api_key.get_secret_value()
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status": response.status_code,
                "ms": int((time.perf_counter() - start) * 1000),
            },
        )
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "llm_provider": settings.llm_provider}

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_key)])
    def chat(body: ChatRequest) -> ChatResponse:
        result = agent.handle(body.message, body.session_id)
        log.info(
            "chat_handled",
            extra={
                "session_id": result.session_id,
                "category": result.category,
                "route": result.route,
                "ticket_id": result.ticket_id,
                "escalated": result.escalated,
                "latency_ms": result.latency_ms,
            },
        )
        return ChatResponse(**result.__dict__)

    @app.get("/v1/tickets", dependencies=[Depends(require_key)])
    def list_tickets(limit: int = 50) -> list[dict]:
        return [t.to_dict() for t in agent.tickets.list(limit=min(limit, 200))]

    @app.get("/v1/tickets/{ticket_id}", dependencies=[Depends(require_key)])
    def get_ticket(ticket_id: str) -> dict:
        ticket = agent.tickets.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return ticket.to_dict()

    return app


def app_factory() -> FastAPI:  # used by: uvicorn support_agent.api:app_factory --factory
    return create_app()
