"""Application configuration loaded from environment variables."""

import math
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_model: str = "anthropic/claude-sonnet-4-5-20250929"

    # Forecaster
    num_agents: int = 10
    max_search_iterations: int = 5
    platt_alpha: float = math.sqrt(3)

    # Default currency pair
    default_pair: str = "USDJPY"

    # Storage
    cache_dir: Path = Path("data/cache")
    db_path: Path = Path("data/forecasts/forecasts.db")
    cache_ttl_hours: int = 6

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
