"""Pluggable LLM connector.

Uncomment ONE of the examples below (or write your own) and call
``register_llm_connector()`` from ``company/__init__.py``.
"""

from aia_forecaster.llm import set_llm_provider


def register_llm_connector() -> None:
    """Register a custom LLM backend."""

    # ── Example 1: Azure OpenAI ──────────────────────────────────────────
    # from langchain_openai import AzureChatOpenAI
    #
    # def _azure_factory(model_name: str, temperature: float, max_tokens: int):
    #     return AzureChatOpenAI(
    #         azure_deployment=model_name,
    #         api_version="2024-12-01-preview",
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #     )
    #
    # set_llm_provider(_azure_factory)

    # ── Example 2: Anthropic ─────────────────────────────────────────────
    # from langchain_anthropic import ChatAnthropic
    #
    # def _anthropic_factory(model_name: str, temperature: float, max_tokens: int):
    #     return ChatAnthropic(
    #         model_name=model_name,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #     )
    #
    # set_llm_provider(_anthropic_factory)

    # ── Example 3: Ollama (local) ────────────────────────────────────────
    # from langchain_ollama import ChatOllama
    #
    # def _ollama_factory(model_name: str, temperature: float, max_tokens: int):
    #     return ChatOllama(
    #         model=model_name,
    #         temperature=temperature,
    #         num_predict=max_tokens,
    #     )
    #
    # set_llm_provider(_ollama_factory)

    pass  # remove once you uncomment an example above
