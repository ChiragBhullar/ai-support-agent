"""Ticket storage. SQLite keeps the project self-contained; the TicketStore
interface is what you would point at Zendesk, Freshdesk or Jira in production."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Ticket:
    id: str
    session_id: str
    category: str
    priority: str
    summary: str
    status: str
    escalated: bool
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class TicketStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    escalated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )

    def create(
        self,
        *,
        session_id: str,
        category: str,
        summary: str,
        priority: str = "normal",
        escalated: bool = False,
    ) -> Ticket:
        ticket = Ticket(
            id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
            session_id=session_id,
            category=category,
            priority=priority,
            summary=summary[:500],
            status="open",
            escalated=escalated,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)",
                (
                    ticket.id,
                    ticket.session_id,
                    ticket.category,
                    ticket.priority,
                    ticket.summary,
                    ticket.status,
                    int(ticket.escalated),
                    ticket.created_at,
                ),
            )
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 50) -> list[Ticket]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> Ticket:
        data = dict(row)
        data["escalated"] = bool(data["escalated"])
        return Ticket(**data)
