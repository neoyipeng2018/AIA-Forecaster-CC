"""Pydantic data models for the AIA Forecaster pipeline."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class Tenor(str):
    """Flexible tenor supporting arbitrary horizons like 1D, 3D, 5D, 2W, 1M, 1Y.

    Accepts any ``<integer><unit>`` string where unit is one of:
      D = days, W = weeks, M = months (30 calendar days), Y = years (365 days).

    Predefined constants for common tenors:
      ``Tenor.D1``, ``Tenor.W1``, ``Tenor.W2``, ``Tenor.M1``, ``Tenor.M2``,
      ``Tenor.M3``, ``Tenor.M6``, ``Tenor.M9``, ``Tenor.Y1``

    Examples::

        Tenor("3D")   # 3-day tenor
        Tenor("5D")   # 5-day tenor
        Tenor("2W")   # 2-week tenor
        Tenor("1Y")   # 1-year tenor
    """

    _PATTERN = re.compile(r"^(\d+)([DWMY])$")
    _UNIT_DAYS = {"D": 1, "W": 7, "M": 30, "Y": 365}

    def __new__(cls, value: str) -> Tenor:
        if isinstance(value, cls):
            return value
        value = str(value).upper().strip()
        if not cls._PATTERN.match(value):
            raise ValueError(
                f"Invalid tenor: {value!r}. "
                f"Expected format like '1D', '3D', '2W', '1M', '1Y'."
            )
        return str.__new__(cls, value)

    @property
    def days(self) -> int:
        """Number of calendar days this tenor represents."""
        m = self._PATTERN.match(self)
        assert m is not None
        return int(m.group(1)) * self._UNIT_DAYS[m.group(2)]

    @property
    def trading_days(self) -> int:
        """Approximate number of trading days (for vol calculations)."""
        return max(1, round(self.days * 252 / 365))

    @property
    def value(self) -> str:
        """The tenor string (backward compatibility with Enum interface)."""
        return str(self)

    @property
    def label(self) -> str:
        """Human-readable label like '1 day', '3 months', '1 year'."""
        m = self._PATTERN.match(self)
        assert m is not None
        count, unit = int(m.group(1)), m.group(2)
        unit_names = {"D": "day", "W": "week", "M": "month", "Y": "year"}
        name = unit_names[unit]
        return f"{count} {name}{'s' if count != 1 else ''}"

    def __repr__(self) -> str:
        return f"Tenor({str(self)!r})"

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        return str.__eq__(self, other)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _pydantic_validate(cls, v: Any) -> Tenor:
        if isinstance(v, cls):
            return v
        return cls(v)


# Predefined constants for common tenors
Tenor.D1 = Tenor("1D")
Tenor.W1 = Tenor("1W")
Tenor.W2 = Tenor("2W")
Tenor.M1 = Tenor("1M")
Tenor.M2 = Tenor("2M")
Tenor.M3 = Tenor("3M")
Tenor.M6 = Tenor("6M")
Tenor.M9 = Tenor("9M")
Tenor.Y1 = Tenor("1Y")


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ForecastMode(str, Enum):
    """Controls the probability semantics of the surface.

    ABOVE:   P(price > strike at expiry) — terminal distribution.
    HITTING: P(price touches strike within tenor) — first-passage / barrier.
    """

    ABOVE = "above"
    HITTING = "hitting"


class SearchMode(str, Enum):
    """Controls which information sources an agent uses."""

    RSS_ONLY = "rss_only"  # RSS feeds only, no web search
    WEB_ONLY = "web_only"  # Web search only, no RSS
    HYBRID = "hybrid"  # Both RSS and web search (original behavior)


class SourceConfig(BaseModel):
    """Controls which data sources are active for a pipeline run.

    Allows toggling individual registry sources (RSS, BIS speeches, etc.)
    and web search independently, enabling A/B comparison of source impact.
    """

    registry_sources: list[str] = Field(
        default_factory=lambda: ["rss", "bis_speeches"],
        description="Which passive registry sources to use (e.g. 'rss', 'bis_speeches')",
    )
    web_search_enabled: bool = Field(
        default=True,
        description="Whether agentic web search runs",
    )

    @property
    def label(self) -> str:
        """Short label for filenames, e.g. 'rss+web' or 'bis_speeches+rss+web'."""
        parts = sorted(self.registry_sources)
        if self.web_search_enabled:
            parts.append("web")
        return "+".join(parts) if parts else "none"

    def get_search_mode(self) -> SearchMode:
        """Derive the SearchMode from this config."""
        has_registry = bool(self.registry_sources)
        if has_registry and self.web_search_enabled:
            return SearchMode.HYBRID
        elif has_registry:
            return SearchMode.RSS_ONLY
        elif self.web_search_enabled:
            return SearchMode.WEB_ONLY
        else:
            # No sources at all — fall back to HYBRID (agents will get no data)
            return SearchMode.HYBRID


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
    relevance_score: float | None = None


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
    forecast_mode: ForecastMode = Field(
        default=ForecastMode.ABOVE,
        description="Probability semantics: 'above' (terminal distribution) or 'hitting' (barrier touch)",
    )
    causal_factors: list[CausalFactor] = Field(
        default_factory=list,
        description="Consensus causal factors aggregated from all agent research briefs",
    )
    regime: str = Field(
        default="",
        description="Detected macro regime (risk_on, risk_off, policy_divergence, carry_unwind, mixed)",
    )
    regime_dominant_channels: list[str] = Field(default_factory=list)
    source_config: SourceConfig | None = Field(
        default=None,
        description="Data source configuration used for this surface (None = all sources / default)",
    )


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
    forecast_mode: ForecastMode = ForecastMode.HITTING
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
