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


class SearchMode(str, Enum):
    """Controls which information sources an agent uses."""

    RSS_ONLY = "rss_only"  # RSS feeds only, no web search
    WEB_ONLY = "web_only"  # Web search only, no RSS
    HYBRID = "hybrid"  # Both RSS and web search (original behavior)


# --- Search ---


class CausalFactor(BaseModel):
    """A single causal channel linking an event/condition to an FX impact.

    Agents produce these during research to make their causal reasoning
    explicit.  The supervisor compares them across agents to pinpoint
    exactly *where* forecasts diverge (event vs. channel vs. magnitude).
    """

    event: str = Field(description="Triggering event or condition")
    channel: str = Field(description="Transmission mechanism (e.g. 'rate differential', 'risk appetite')")
    direction: str = Field(description="'bullish' or 'bearish' on the pair")
    magnitude: str = Field(description="'strong', 'moderate', or 'weak'")
    confidence: str = Field(description="'high', 'medium', or 'low'")


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
    spot: float | None = None
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
    search_mode: SearchMode = SearchMode.HYBRID


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


# --- Two-Phase Surface Research ---


class ResearchBrief(BaseModel):
    """Per-agent research output from Phase 1 (shared research)."""

    agent_id: int
    key_themes: list[str] = Field(default_factory=list)
    causal_factors: list[CausalFactor] = Field(default_factory=list)
    evidence: list[SearchResult] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    search_mode: SearchMode = SearchMode.HYBRID
    macro_summary: str = ""
    iterations: int = 0


class SharedResearch(BaseModel):
    """Collection of all agent research briefs for a currency pair."""

    pair: str
    cutoff_date: date
    briefs: list[ResearchBrief] = Field(default_factory=list)


class BatchPricingResult(BaseModel):
    """Per-agent pricing of all strikes for one tenor."""

    agent_id: int
    tenor: Tenor
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Map of strike (as string) to probability",
    )
    reasoning: str = ""
    causal_factors: list[CausalFactor] = Field(default_factory=list)


# --- FX Surface ---


class SurfaceCell(BaseModel):
    strike: float
    tenor: Tenor
    question: str
    calibrated: CalibratedForecast | None = None
    ensemble: EnsembleResult | None = None


class ProbabilitySurface(BaseModel):
    pair: str
    spot_rate: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cells: list[SurfaceCell] = Field(default_factory=list)
    causal_factors: list[CausalFactor] = Field(
        default_factory=list,
        description="Consensus causal factors aggregated from all agent research briefs",
    )
    regime: str = Field(
        default="",
        description="Detected macro regime (risk_on, risk_off, policy_divergence, carry_unwind, mixed)",
    )
    regime_dominant_channels: list[str] = Field(default_factory=list)


# --- Explanation Models ---


class EvidenceItem(BaseModel):
    """A single evidence item with citation frequency across agents."""

    title: str
    snippet: str
    url: str
    source: str = ""
    cited_by_agents: int = 1


class CellExplanation(BaseModel):
    """Per-cell explanation summary for a surface cell."""

    strike: float
    tenor: Tenor
    calibrated_probability: float | None = None
    raw_probability: float | None = None
    num_agents: int = 0
    agent_probabilities: list[float] = Field(default_factory=list)
    top_evidence: list[EvidenceItem] = Field(default_factory=list)
    consensus_summary: str = ""
    disagreement_notes: str = ""
    supervisor_confidence: str | None = None
    supervisor_reasoning: str = ""


class SurfaceExplanation(BaseModel):
    """Full explanation for an entire probability surface."""

    pair: str
    spot_rate: float
    generated_at: datetime
    cells: list[CellExplanation] = Field(default_factory=list)
    causal_factors: list[CausalFactor] = Field(default_factory=list)
    regime: str = ""
    regime_dominant_channels: list[str] = Field(default_factory=list)


# --- Full Pipeline Output ---


class ForecastRun(BaseModel):
    id: str = ""
    question: ForecastQuestion
    ensemble: EnsembleResult | None = None
    calibrated: CalibratedForecast | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
