"""Application configuration loaded from environment variables."""

import math
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""

    # Cerebras fallback
    cerebras_api_key: str = ""
    cerebras_model: str = "llama-4-scout-17b-16e-instruct"

    # Forecaster
    num_agents: int = 10
    max_search_iterations: int = 5
    platt_alpha: float = math.sqrt(3)

    # Forecast mode
    forecast_mode: str = "hitting"  # "hitting" (barrier touch) or "above" (terminal)

    # Causal reasoning
    regime_weighting_enabled: bool = True

    # Tenor-specific research (Phase 1.5)
    tenor_research_enabled: bool = True
    tenor_research_max_iterations: int = 2

    # Relevance filtering
    relevance_threshold: float = 0.10
    relevance_filtering_enabled: bool = True
    llm_relevance_enabled: bool = True

    # Web search provider
    web_search_provider: str = "duckduckgo"

    # Default currency pair
    default_pair: str = "USDJPY"

    # Storage
    db_path: Path = Path("data/forecasts/forecasts.db")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
