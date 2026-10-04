"""A thin interface over the chat model so the rest of the code doesn't depend on
one vendor. Tests use a fake that implements the same interface."""

from __future__ import annotations

from typing import Protocol

from support_agent.config import Settings


class ChatModel(Protocol):
    def invoke(self, system: str, user: str) -> str: ...


class OpenAIChatModel:
    """Chat model backed by OpenAI through LangChain (timeouts and retries on)."""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI

        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when SUPPORT_LLM_PROVIDER=openai")
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    def invoke(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        result = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(result.content).strip()


def build_chat_model(settings: Settings) -> ChatModel | None:
    if settings.llm_provider == "openai":
        return OpenAIChatModel(settings)
    return None
