"""Historical base rate computation for FX probability anchoring.

Computes statistical base rates for currency pair moves using
log-normal assumptions and dynamically fetched realized volatilities.
Falls back to static estimates when market data is unavailable.
Gives forecasting agents a quantitative anchor to adjust from.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from statistics import NormalDist

from aia_forecaster.models import ForecastMode, Tenor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus provider hook
# ---------------------------------------------------------------------------
# Register a callable to supply consensus forecasts from an external source
# (analyst surveys, internal models, market-implied estimates, etc.).
#
# When set, the consensus rate becomes the center of the probability
# distribution instead of the forward.  The carry-adjusted forward is still
# computed internally and shown to agents for context.
#
# The provider receives (pair, spot, tenor) and should return either:
#   - (consensus_rate, source_label) on success
#   - None to signal "no consensus available for this pair/tenor"
#
# Example:
#
#     from aia_forecaster.fx.base_rates import set_consensus_provider
#
#     def my_consensus(pair: str, spot: float, tenor: Tenor) -> tuple[float, str] | None:
#         rate = internal_model.get_consensus(pair, str(tenor))
#         if rate is None:
#             return None
#         return rate, "analyst_consensus"
#
#     set_consensus_provider(my_consensus)

ConsensusProvider = Callable[[str, float, "Tenor"], tuple[float, str] | None]

_consensus_provider: ConsensusProvider | None = None


def set_consensus_provider(provider: ConsensusProvider | None) -> None:
    """Register (or clear) a consensus-rate provider.

    The provider is called with ``(pair, spot, tenor)`` and should return
    ``(consensus_rate, source_label)`` or ``None``.

    When a consensus rate is available it replaces the forward as the center
    of the probability distribution.  The forward is still computed and shown
    to agents for carry context.

    Pass ``None`` to remove a previously registered provider.
    """
    global _consensus_provider
    _consensus_provider = provider
    if provider is not None:
        logger.info("Consensus provider registered: %s", provider)
    else:
        logger.info("Consensus provider cleared")


def get_consensus(
    pair: str, spot: float, tenor: "Tenor",
) -> tuple[float, str] | None:
    """Query the consensus provider for a point estimate.

    Returns (consensus_rate, source_label) or None if no provider is
    registered or the provider has no data for this pair/tenor.
    """
    if _consensus_provider is None:
        return None
    try:
        return _consensus_provider(pair.upper(), spot, tenor)
    except Exception:
        logger.warning(
            "Consensus provider raised for %s %s; ignoring",
            pair, tenor, exc_info=True,
        )
        return None

# Static fallback volatilities — used when live data is unavailable.
FALLBACK_VOL: dict[str, float] = {
    "USDJPY": 0.10,
    "EURUSD": 0.08,
    "GBPUSD": 0.09,
    "AUDUSD": 0.10,
    "NZDUSD": 0.11,
    "USDCAD": 0.07,
    "USDCHF": 0.08,
}

# Backward-compatible alias: code that imports ANNUALIZED_VOL will still work,
# but the values may be overridden at runtime by dynamic lookups.
ANNUALIZED_VOL = FALLBACK_VOL

# Yahoo Finance ticker format for FX pairs (e.g., "USDJPY=X")
_YF_SUFFIX = "=X"

# Cache: pair -> (annualized_vol, timestamp)
_vol_cache: dict[str, tuple[float, float]] = {}
_CACHE_TTL = 3600  # 1 hour — vol doesn't change dramatically intra-session

_norm = NormalDist(0, 1)

# ---------------------------------------------------------------------------
# Policy rates (central bank target/benchmark rates per currency)
# ---------------------------------------------------------------------------
# Used to compute FX forward rates via covered interest rate parity:
#   Forward = Spot × exp((r_quote - r_base) × T)
#
# These are fallback values for when live rate feeds are unavailable.
# Update periodically or replace with a dynamic source.

_POLICY_RATES_UPDATED = "2026-02"  # YYYY-MM of last update

FALLBACK_POLICY_RATES: dict[str, float] = {
    "USD": 0.0450,  # Federal Reserve — fed funds upper bound
    "JPY": 0.0050,  # Bank of Japan
    "EUR": 0.0275,  # European Central Bank — deposit facility
    "GBP": 0.0425,  # Bank of England
    "AUD": 0.0410,  # Reserve Bank of Australia
    "NZD": 0.0375,  # Reserve Bank of New Zealand
    "CAD": 0.0300,  # Bank of Canada
    "CHF": 0.0050,  # Swiss National Bank
}


def get_policy_rate(currency: str) -> float | None:
    """Return the static fallback policy rate for a currency, or None."""
    return FALLBACK_POLICY_RATES.get(currency.upper())


# ---------------------------------------------------------------------------
# Dynamic interest-rate fetching (Yahoo Finance)
# ---------------------------------------------------------------------------
# Yahoo Finance provides direct yield data only for US Treasuries.
# For other currencies we stay on static fallbacks and log the gap.
#
# To add a new currency:
#   1. Find a Yahoo Finance ticker that reports yield (not price).
#   2. Add an entry to _YIELD_TICKERS below.
#   3. Set is_pct=True if the ticker reports in percentage points
#      (e.g. 4.25 meaning 4.25%), which is standard for Yahoo.

_YIELD_TICKERS: dict[str, list[tuple[str, bool, str]]] = {
    # currency -> [(yahoo_ticker, is_pct, description), ...]
    # Tickers are tried in order; first success wins.
    "USD": [
        ("^IRX", True, "13-week US T-bill"),
    ],
}

# Cache: currency -> (rate_as_decimal, timestamp)
_rate_cache: dict[str, tuple[float, float]] = {}
_RATE_CACHE_TTL = 14400  # 4 hours — short rates move slowly


def _fetch_dynamic_rate(currency: str) -> float | None:
    """Try to fetch a short-term interest rate from Yahoo Finance.

    Returns the annualized rate as a decimal (e.g. 0.0425 for 4.25%),
    or None on any failure.
    """
    currency = currency.upper()
    tickers = _YIELD_TICKERS.get(currency)
    if not tickers:
        return None

    try:
        import yfinance as yf  # noqa: F811 — lazy import
    except ImportError:
        logger.debug("yfinance not installed; skipping dynamic rate for %s", currency)
        return None

    for ticker, is_pct, desc in tickers:
        try:
            data = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if data is None or len(data) == 0:
                logger.debug("No data for %s (%s)", ticker, desc)
                continue

            closes = data["Close"].dropna()
            if hasattr(closes, "squeeze"):
                closes = closes.squeeze()
            if len(closes) == 0:
                continue

            raw_value = float(closes.iloc[-1])

            # Convert from percentage points to decimal if needed
            rate = raw_value / 100.0 if is_pct else raw_value

            # Sanity check: reject implausible rates
            if rate < -0.02 or rate > 0.30:
                logger.warning(
                    "%s rate %.4f from %s (%s) looks implausible; skipping",
                    currency, rate, ticker, desc,
                )
                continue

            logger.info(
                "%s dynamic rate: %.2f%% from %s (%s)",
                currency, rate * 100, ticker, desc,
            )
            return rate

        except Exception:
            logger.debug(
                "Failed to fetch %s for %s rate", ticker, currency, exc_info=True,
            )
            continue

    return None


def get_short_rate(currency: str) -> tuple[float, str]:
    """Return the best available short-term rate for *currency*.

    Priority:
      1. Cached dynamic rate (if fresh)
      2. Freshly fetched dynamic rate from Yahoo Finance
      3. Static fallback from FALLBACK_POLICY_RATES

    Returns:
        Tuple of (rate_as_decimal, source_label).
        source_label is one of: "dynamic:<ticker>", "fallback", "none".

    Raises ValueError if no rate is available at all.
    """
    currency = currency.upper()

    # 1. Check cache
    if currency in _rate_cache:
        rate, ts = _rate_cache[currency]
        if time.time() - ts < _RATE_CACHE_TTL:
            return rate, "dynamic"

    # 2. Try dynamic fetch
    dynamic = _fetch_dynamic_rate(currency)
    if dynamic is not None:
        _rate_cache[currency] = (dynamic, time.time())

        # Log if it differs materially from fallback
        fallback = FALLBACK_POLICY_RATES.get(currency)
        if fallback is not None:
            delta_bp = abs(dynamic - fallback) * 10_000
            if delta_bp > 25:  # > 25bp difference
                logger.info(
                    "%s dynamic rate %.2f%% differs from fallback %.2f%% by %.0fbp",
                    currency, dynamic * 100, fallback * 100, delta_bp,
                )
        return dynamic, "dynamic"

    # 3. Static fallback
    fallback = FALLBACK_POLICY_RATES.get(currency)
    if fallback is not None:
        return fallback, "fallback"

    return 0.0, "none"


def compute_forward_rate(
    pair: str,
    spot: float,
    tenor: Tenor,
) -> tuple[float, float, float, str]:
    """Compute the FX forward rate from interest-rate parity.

    Forward = Spot × exp((r_quote − r_base) × T_years)

    This is pure carry math — no directional view.  Uses dynamically
    fetched rates when available (currently USD via Yahoo Finance ^IRX),
    with static policy-rate fallbacks for other currencies.

    Args:
        pair: Currency pair (e.g. "USDJPY").
        spot: Current spot rate.
        tenor: Forecast horizon.

    Returns:
        Tuple of (forward_rate, r_base, r_quote, source_label).
        Falls back to (spot, 0, 0, "no_rates") if both rates are unknown.
    """
    pair = pair.upper()
    base_ccy, quote_ccy = pair[:3], pair[3:]

    r_base, src_base = get_short_rate(base_ccy)
    r_quote, src_quote = get_short_rate(quote_ccy)

    if src_base == "none" and src_quote == "none":
        logger.debug(
            "No rates for %s/%s — forward defaults to spot",
            base_ccy, quote_ccy,
        )
        return spot, 0.0, 0.0, "no_rates"

    t_years = tenor.days / 365.0
    forward = spot * math.exp((r_quote - r_base) * t_years)

    source_label = f"{base_ccy}:{src_base}|{quote_ccy}:{src_quote}"

    logger.debug(
        "%s forward (%s): %.4f → %.4f  (r_%s=%.2f%% [%s], r_%s=%.2f%% [%s], T=%.3fy)",
        pair, tenor, spot, forward,
        base_ccy, r_base * 100, src_base,
        quote_ccy, r_quote * 100, src_quote,
        t_years,
    )

    return forward, r_base, r_quote, source_label


def _compute_realized_vol(pair: str, lookback_days: int = 60) -> float | None:
    """Fetch recent daily closes from Yahoo Finance and compute annualized vol.

    Uses log-returns standard deviation scaled to annualized.  The lookback
    window defaults to 60 trading days (~3 months) which balances recency
    against noise.

    Returns None on any failure (network, missing data, etc.).
    """
    try:
        import yfinance as yf  # noqa: F811 — lazy import to keep startup fast
    except ImportError:
        logger.debug("yfinance not installed; skipping dynamic vol for %s", pair)
        return None

    ticker = f"{pair.upper()}{_YF_SUFFIX}"
    try:
        data = yf.download(
            ticker,
            period=f"{lookback_days + 10}d",  # small buffer for weekends/holidays
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data is None or len(data) < 10:
            logger.warning("Insufficient price data for %s (%d rows)", pair, len(data) if data is not None else 0)
            return None

        closes = data["Close"].dropna()
        if hasattr(closes, "squeeze"):
            closes = closes.squeeze()
        if len(closes) < 10:
            return None

        # Log returns
        import numpy as np

        log_returns = np.diff(np.log(closes.values.astype(float)))
        daily_vol = float(np.std(log_returns, ddof=1))
        annualized = daily_vol * math.sqrt(252)

        # Sanity check: reject implausible values
        if annualized < 0.01 or annualized > 0.50:
            logger.warning(
                "Computed vol %.4f for %s looks implausible; discarding", annualized, pair
            )
            return None

        return annualized
    except Exception:
        logger.warning("Failed to fetch dynamic vol for %s", pair, exc_info=True)
        return None


def get_annualized_vol(pair: str) -> float:
    """Return the best available annualized volatility for *pair*.

    Priority:
      1. Cached dynamic vol (if fresh)
      2. Freshly computed realized vol from Yahoo Finance
      3. Static fallback from FALLBACK_VOL

    Raises ValueError if the pair has no fallback and dynamic fetch fails.
    """
    pair = pair.upper()

    # 1. Check cache
    if pair in _vol_cache:
        vol, ts = _vol_cache[pair]
        if time.time() - ts < _CACHE_TTL:
            return vol

    # 2. Try dynamic computation
    realized = _compute_realized_vol(pair)
    if realized is not None:
        _vol_cache[pair] = (realized, time.time())
        fallback = FALLBACK_VOL.get(pair)
        if fallback is not None:
            delta = abs(realized - fallback) / fallback
            if delta > 0.25:
                logger.info(
                    "%s realized vol %.1f%% differs from fallback %.1f%% by %.0f%% "
                    "— using realized",
                    pair,
                    realized * 100,
                    fallback * 100,
                    delta * 100,
                )
        else:
            logger.info(
                "%s dynamic vol computed: %.1f%% (no static fallback existed)",
                pair,
                realized * 100,
            )
        return realized

    # 3. Static fallback
    if pair in FALLBACK_VOL:
        logger.debug("Using static fallback vol for %s: %.1f%%", pair, FALLBACK_VOL[pair] * 100)
        return FALLBACK_VOL[pair]

    raise ValueError(
        f"No volatility data for {pair}. "
        f"Dynamic fetch failed and no static fallback available."
    )


def compute_base_rates(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
) -> dict:
    """Compute statistical base rate for P(price > strike at expiry).

    The distribution center is chosen by priority:
      1. Consensus forecast (if a provider is registered and returns a value)
      2. FX forward rate (carry-adjusted, from interest-rate parity)

    The forward is always computed for carry context regardless of whether
    consensus is used as the center.

    The probability is computed under a log-normal model:
        d2 = (ln(C/K) − ½σ_t²) / σ_t
        P(S_T > K) = Φ(d2)
    where C is the distribution center (consensus or forward).

    Args:
        pair: Currency pair (e.g. "USDJPY").
        spot: Current spot rate.
        strike: Target price level.
        tenor: Forecast horizon.

    Returns:
        Dict with keys: sigma_t, move_required, move_pct, z_score,
        base_rate_above, tenor_range_1sigma, annualized_vol, vol_source,
        forward_rate, forward_source, r_base, r_quote,
        center, center_source, consensus_rate, consensus_source.
    """
    pair = pair.upper()
    annual_vol = get_annualized_vol(pair)
    vol_source = "dynamic" if pair in _vol_cache else "fallback"

    # Check for consensus first — if available, skip forward computation
    consensus_result = get_consensus(pair, spot, tenor)
    if consensus_result is not None:
        consensus_rate, consensus_source = consensus_result
        center = consensus_rate
        center_source = consensus_source
        forward = None
        fwd_source = None
        r_base = None
        r_quote = None
    else:
        consensus_rate = None
        consensus_source = None
        forward, r_base, r_quote, fwd_source = compute_forward_rate(pair, spot, tenor)
        center = forward
        center_source = "forward"

    days = tenor.trading_days
    sigma_t = annual_vol * math.sqrt(days / 252)
    tenor_range_1sigma = spot * sigma_t

    # Move from center (consensus or forward)
    move_required = strike - center
    move_pct = move_required / spot

    # Log-normal z-score relative to center
    if sigma_t > 0 and strike > 0 and center > 0:
        d2 = (math.log(center / strike) - 0.5 * sigma_t**2) / sigma_t
        base_rate_above = _norm.cdf(d2)
        z_score = -d2
    else:
        z_score = 0.0
        base_rate_above = 0.5

    return {
        "sigma_t": sigma_t,
        "move_required": move_required,
        "move_pct": move_pct,
        "z_score": z_score,
        "base_rate_above": base_rate_above,
        "tenor_range_1sigma": tenor_range_1sigma,
        "annualized_vol": annual_vol,
        "vol_source": vol_source,
        "forward_rate": forward,
        "forward_source": fwd_source,
        "r_base": r_base,
        "r_quote": r_quote,
        "center": center,
        "center_source": center_source,
        "consensus_rate": consensus_rate,
        "consensus_source": consensus_source,
    }


def _first_passage_probability(
    h: float,
    nu_T: float,
    sigma_t: float,
) -> float:
    """First-passage probability for Brownian motion with drift.

    Computes P(process touches level h before time T) where the log-price
    process has total drift nu_T and tenor-scaled volatility sigma_t.

    For barrier above spot (h > 0):
        P = Φ((nu_T − h)/σ_t) + exp(2·nu_T·h/σ_t²) · Φ((−nu_T − h)/σ_t)

    For barrier below spot (h < 0), uses the minimum of the process:
        P = Φ((−nu_T + h)/σ_t) + exp(2·nu_T·h/σ_t²) · Φ((nu_T + h)/σ_t)

    Reduces to 2·(1 − Φ(|h|/σ_t)) when nu_T = 0 (zero drift / reflection
    principle), which matches the previous implementation.

    Args:
        h: Log-distance to barrier, ln(barrier/spot). Positive = above spot.
        nu_T: Total log-space drift = ln(F/S) − ½·σ_t².
        sigma_t: Tenor-scaled volatility (σ_annual · √(T_trading)).
    """
    if abs(h) < 1e-12:
        return 1.0  # barrier at spot — always touched

    if sigma_t <= 1e-12:
        # No volatility: only hits if drift carries us there
        return 1.0 if (h > 0 and nu_T >= h) or (h < 0 and nu_T <= h) else 0.0

    # Clamp the exponent to avoid overflow (exp(700) ~ 1e304)
    exponent = 2.0 * nu_T * h / (sigma_t**2)
    exponent = max(-500.0, min(500.0, exponent))

    if h > 0:
        # Barrier above spot
        d1 = (nu_T - h) / sigma_t
        d2 = (-nu_T - h) / sigma_t
        p = _norm.cdf(d1) + math.exp(exponent) * _norm.cdf(d2)
    else:
        # Barrier below spot (h < 0)
        abs_h = -h
        d1 = (-nu_T - abs_h) / sigma_t
        d2 = (nu_T - abs_h) / sigma_t
        p = _norm.cdf(d1) + math.exp(exponent) * _norm.cdf(d2)

    return max(0.0, min(1.0, p))


def compute_hitting_base_rate(
    pair: str,
    spot: float,
    barrier: float,
    tenor: Tenor,
) -> dict:
    """Compute the base rate for a barrier/touch probability.

    Uses the drift-adjusted first-passage formula for geometric Brownian
    motion.  The drift target is chosen by priority:
      1. Consensus forecast (if available)
      2. FX forward rate (carry-adjusted)

    When no rate data or consensus is available the drift is zero, which
    is equivalent to the reflection-principle formula.

    Args:
        pair: Currency pair.
        spot: Current spot rate.
        barrier: Barrier/touch level.
        tenor: Forecast horizon.

    Returns:
        Dict with keys: sigma_t, distance_pct, base_rate_hitting, base_rate_above,
        tenor_range_1sigma, annualized_vol, vol_source,
        forward_rate, forward_source, r_base, r_quote,
        center, center_source, consensus_rate, consensus_source.
    """
    pair = pair.upper()
    annual_vol = get_annualized_vol(pair)
    vol_source = "dynamic" if pair in _vol_cache else "fallback"

    # Check for consensus first — if available, skip forward computation
    consensus_result = get_consensus(pair, spot, tenor)
    if consensus_result is not None:
        consensus_rate, consensus_source = consensus_result
        center = consensus_rate
        center_source = consensus_source
        forward = None
        fwd_source = None
        r_base = None
        r_quote = None
    else:
        consensus_rate = None
        consensus_source = None
        forward, r_base, r_quote, fwd_source = compute_forward_rate(pair, spot, tenor)
        center = forward
        center_source = "forward"

    days = tenor.trading_days
    sigma_t = annual_vol * math.sqrt(days / 252)
    tenor_range_1sigma = spot * sigma_t

    base_dict = {
        "sigma_t": sigma_t,
        "tenor_range_1sigma": tenor_range_1sigma,
        "annualized_vol": annual_vol,
        "vol_source": vol_source,
        "forward_rate": forward,
        "forward_source": fwd_source,
        "r_base": r_base,
        "r_quote": r_quote,
        "center": center,
        "center_source": center_source,
        "consensus_rate": consensus_rate,
        "consensus_source": consensus_source,
    }

    if barrier <= 0 or spot <= 0:
        return {
            **base_dict,
            "distance_pct": 0.0,
            "base_rate_hitting": 1.0,
            "base_rate_above": 0.5,
        }

    # Log-distance to barrier and drift toward center
    h = math.log(barrier / spot)
    nu_T = math.log(center / spot) - 0.5 * sigma_t**2

    base_rate_hitting = _first_passage_probability(h, nu_T, sigma_t)

    # P(above at expiry) using center
    if sigma_t > 0 and center > 0:
        d2 = (math.log(center / barrier) - 0.5 * sigma_t**2) / sigma_t
        base_rate_above = _norm.cdf(d2)
    else:
        base_rate_above = 0.5

    distance_pct = (barrier - spot) / spot

    return {
        **base_dict,
        "distance_pct": distance_pct,
        "base_rate_hitting": base_rate_hitting,
        "base_rate_above": base_rate_above,
    }


def _format_market_context(
    stats: dict, base: str, quote: str, spot: float, price_fmt: str, horizon: str,
) -> str:
    """Build the forward + consensus lines for the context block."""
    lines: list[str] = []

    # Consensus line (directional view — shown when provider is set)
    cons = stats.get("consensus_rate")
    cons_src = stats.get("consensus_source")
    if cons is not None and cons_src is not None:
        lines.append(
            f"{horizon} consensus: {base}/{quote} = {cons:{price_fmt}} "
            f"(src: {cons_src})"
        )
    else:
        # Forward line (carry math — only shown when no consensus is available)
        fwd = stats.get("forward_rate")
        fwd_src = stats.get("forward_source", "")
        if fwd is not None and fwd_src != "no_rates":
            r_b = stats.get("r_base", 0)
            r_q = stats.get("r_quote", 0)
            diff = r_q - r_b
            diff_sign = "+" if diff >= 0 else ""

            src_parts = fwd_src.split("|") if "|" in fwd_src else []
            if src_parts:
                src_note = "/".join(p.split(":")[-1] for p in src_parts)
            else:
                src_note = fwd_src

            lines.append(
                f"{horizon} forward: {base}/{quote} = {fwd:{price_fmt}} "
                f"(carry: {base} {r_b:.2%} vs {quote} {r_q:.2%}, "
                f"net {diff_sign}{diff:.2%}, src: {src_note})"
            )

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def format_base_rate_context(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
    forecast_mode: ForecastMode = ForecastMode.ABOVE,
) -> str:
    """Produce a human-readable base rate context block for LLM prompts.

    Returns an empty string if the pair is entirely unsupported
    (no dynamic vol and no fallback).
    """
    pair = pair.upper()

    horizon = tenor.label
    base, quote = pair[:3], pair[3:]
    price_fmt = ".2f" if "JPY" in pair else ".4f"

    if forecast_mode == ForecastMode.HITTING:
        try:
            stats = compute_hitting_base_rate(pair, spot, strike, tenor)
        except ValueError:
            return ""

        center = stats["center"]
        center_src = stats["center_source"]
        direction = "above" if strike >= spot else "below"
        distance_pct = stats["distance_pct"]
        dist_sign = "+" if distance_pct >= 0 else ""

        vol_note = (
            f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
        )
        market_note = _format_market_context(stats, base, quote, spot, price_fmt, horizon)

        drift_note = ""
        if center is not None and center != spot:
            drift_dir = "toward" if (
                (strike > spot and center > spot) or (strike < spot and center < spot)
            ) else "away from"
            drift_note = (
                f"Expected drift ({center_src}) is {drift_dir} this barrier.\n"
            )

        anchor_label = f"the {center_src}" if center_src != "forward" else "the forward"

        return (
            f"BASE RATE CONTEXT (statistical anchor — HITTING/BARRIER mode):\n"
            f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
            f"{market_note}"
            f"{vol_note}"
            f"Barrier: {strike:{price_fmt}} ({direction} spot, "
            f"distance: {dist_sign}{distance_pct:.2%})\n"
            f"{drift_note}"
            f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
            f"Statistical base rate: P(touch {strike:{price_fmt}} within {horizon}) "
            f"= {stats['base_rate_hitting']:.3f} ({stats['base_rate_hitting']:.1%})\n"
            f"For reference, P(above {strike:{price_fmt}} at expiry) "
            f"= {stats['base_rate_above']:.3f}\n"
            f"Note: P(touch) >= P(above) always. P(touch) ~ 1.0 near spot, "
            f"decreasing with distance. Longer tenors increase P(touch).\n"
            f"This is a drift-adjusted first-passage baseline anchored to {anchor_label}. "
            f"Adjust based on current evidence."
        )

    # Default: ABOVE mode
    try:
        stats = compute_base_rates(pair, spot, strike, tenor)
    except ValueError:
        return ""

    center = stats["center"]
    center_src = stats["center_source"]
    direction = "above" if strike >= center else "below"

    # Show move from center (primary) and from spot (for context)
    move_from_center = strike - center
    move_from_spot = strike - spot
    center_sign = "+" if move_from_center >= 0 else ""
    spot_sign = "+" if move_from_spot >= 0 else ""

    center_label = center_src if center_src != "forward" else "forward"

    vol_note = (
        f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
    )
    market_note = _format_market_context(stats, base, quote, spot, price_fmt, horizon)

    return (
        f"BASE RATE CONTEXT (statistical anchor):\n"
        f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
        f"{market_note}"
        f"{vol_note}"
        f"Target: {direction} {strike:{price_fmt}} in {horizon}\n"
        f"  From {center_label}: {center_sign}{move_from_center:{price_fmt}} "
        f"({center_sign}{move_from_center / spot:.2%})\n"
        f"  From spot:    {spot_sign}{move_from_spot:{price_fmt}} "
        f"({spot_sign}{move_from_spot / spot:.2%})\n"
        f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
        f"Required move from {center_label}: {abs(stats['z_score']):.2f} standard deviations\n"
        f"Statistical base rate: P({direction} {strike:{price_fmt}}) "
        f"= {stats['base_rate_above']:.3f} ({stats['base_rate_above']:.1%})\n"
        f"Note: Base rate is anchored to {center_label}"
        f"{' (consensus view)' if center_src != 'forward' else ' (carry-adjusted)'}.\n"
        f"This is a log-normal baseline. Adjust based on current evidence."
    )
