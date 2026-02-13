"""Pydantic data models for the AIA Forecaster pipeline."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Tenor(str, Enum):
    D1 = "1D"
    W1 = "1W"
    M1 = "1M"
    M3 = "3M"
    M6 = "6M"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Search ---


class SearchResult(BaseModel):
    query: str
    title: str
    snippet: str
    url: str
    source: str = ""
    timestamp: datetime | None = None


class FlaggedResult(BaseModel):
    result: SearchResult
    has_foreknowledge: bool
    confidence: Confidence
    evidence: str = ""


# --- Forecasting ---


class ForecastQuestion(BaseModel):
    text: str = Field(description="Binary question, e.g. 'Will USD/JPY be above 155.00 in 1 week?'")
    pair: str = "USDJPY"
    strike: float | None = None
    tenor: Tenor | None = None
    cutoff_date: date = Field(default_factory=date.today)


class AgentForecast(BaseModel):
    agent_id: int
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str
    search_queries: list[str] = Field(default_factory=list)
    evidence: list[SearchResult] = Field(default_factory=list)
    iterations: int = 0


class SupervisorResult(BaseModel):
    reconciled_probability: float | None = Field(None, ge=0.0, le=1.0)
    confidence: Confidence
    reasoning: str
    additional_evidence: list[SearchResult] = Field(default_factory=list)


class EnsembleResult(BaseModel):
    agent_forecasts: list[AgentForecast]
    mean_probability: float
    supervisor: SupervisorResult | None = None
    final_probability: float = Field(
        description="Supervisor probability if high confidence, else mean"
    )


class CalibratedForecast(BaseModel):
    raw_probability: float
    calibrated_probability: float
    alpha: float


# --- FX Surface ---


class SurfaceCell(BaseModel):
    strike: float
    tenor: Tenor
    question: str
    calibrated: CalibratedForecast | None = None


class ProbabilitySurface(BaseModel):
    pair: str
    spot_rate: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cells: list[SurfaceCell] = Field(default_factory=list)


# --- Full Pipeline Output ---


class ForecastRun(BaseModel):
    id: str = ""
    question: ForecastQuestion
    ensemble: EnsembleResult | None = None
    calibrated: CalibratedForecast | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
