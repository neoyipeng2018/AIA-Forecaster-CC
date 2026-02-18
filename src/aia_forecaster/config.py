"""Application configuration loaded from environment variables."""

import math
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""

    # Forecaster
    num_agents: int = 10
    max_search_iterations: int = 5
    platt_alpha: float = math.sqrt(3)

    # Forecast mode
    forecast_mode: str = "hitting"  # "hitting" (barrier touch) or "above" (terminal)

    # Causal reasoning
    regime_weighting_enabled: bool = True

    # Default currency pair
    default_pair: str = "USDJPY"

    # Storage
    cache_dir: Path = Path("data/cache")
    db_path: Path = Path("data/forecasts/forecasts.db")
    cache_ttl_hours: int = 6

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
