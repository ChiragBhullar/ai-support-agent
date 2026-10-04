"""Chat with the agent in the terminal:  python -m support_agent.cli"""

from __future__ import annotations

import uuid

from support_agent.agent import build_agent


def main() -> None:
    agent = build_agent()
    session = uuid.uuid4().hex
    print("AI Support Agent - type 'quit' to exit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue
        r = agent.handle(message, session)
        meta = f"[{r.category} -> {r.route}" + (f", ticket {r.ticket_id}" if r.ticket_id else "")
        print(f"agent> {r.reply}\n       {meta}]\n")


if __name__ == "__main__":
    main()
