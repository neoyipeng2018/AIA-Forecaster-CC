"""LLM client using langchain-openai."""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aia_forecaster.config import settings

logger = logging.getLogger(__name__)

_ROLE_MAP = {
    "user": HumanMessage,
    "human": HumanMessage,
    "system": SystemMessage,
    "assistant": AIMessage,
}

# Reasoning models use hidden reasoning tokens that count toward max_tokens.
# Apply a floor so short-output calls (e.g. max_tokens=10) still have room.
_REASONING_PREFIXES = ("o1", "o3", "gpt-5")
_REASONING_MIN_TOKENS = 2048


def _to_langchain_messages(messages: list[dict[str, str]]):
    """Convert {"role": ..., "content": ...} dicts to LangChain message objects."""
    return [_ROLE_MAP[m["role"]](content=m["content"]) for m in messages]


class LLMClient:
    """Thin wrapper around langchain-openai for LLM access."""

    def __init__(self, model: str | None = None, temperature: float = 0.7):
        self.model = model or settings.llm_model
        self.temperature = temperature

    def _build_chat(self, temperature: float, max_tokens: int) -> ChatOpenAI:
        # Strip provider prefix (e.g. "openai/gpt-5-mini" → "gpt-5-mini")
        model_name = self.model.split("/", 1)[-1] if "/" in self.model else self.model

        # Reasoning models need headroom for hidden reasoning tokens
        if any(model_name.startswith(p) for p in _REASONING_PREFIXES):
            max_tokens = max(max_tokens, _REASONING_MIN_TOKENS)

        kwargs: dict = dict(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return ChatOpenAI(**kwargs)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the text response."""
        chat = self._build_chat(
            temperature if temperature is not None else self.temperature,
            max_tokens,
        )
        response = await chat.ainvoke(_to_langchain_messages(messages))
        return response.content

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat completion request and parse the response as JSON."""
        response_text = await self.complete(messages, temperature, max_tokens)

        # Extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)
