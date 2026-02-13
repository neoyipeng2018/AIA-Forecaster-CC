"""Multi-provider LLM client using litellm."""

from __future__ import annotations

import json
import logging

import litellm

from aia_forecaster.config import settings

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    """Thin wrapper around litellm for multi-provider LLM access."""

    def __init__(self, model: str | None = None, temperature: float = 0.7):
        self.model = model or settings.llm_model
        self.temperature = temperature

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the text response."""
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat completion request and parse the response as JSON.

        Appends an instruction to respond in JSON to the last user message.
        """
        response_text = await self.complete(messages, temperature, max_tokens)

        # Extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code fence
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)
