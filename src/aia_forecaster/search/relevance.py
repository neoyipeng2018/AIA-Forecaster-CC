"""Heuristic relevance scoring for search results.

Scores each SearchResult 0.0–1.0 for relevance to a specific currency pair.
Fast, deterministic, zero LLM cost — catches blatantly off-topic articles
(e.g., "Philippines Gold price today" appearing in USDJPY evidence).
"""

from __future__ import annotations

import logging
import re

from aia_forecaster.models import SearchResult
from aia_forecaster.search.rss import CURRENCY_KEYWORDS, GENERAL_FX_KEYWORDS

logger = logging.getLogger(__name__)

# Pairs of (base currency, commodity keywords) where commodity mentions
# should NOT be penalized because the commodity fundamentally drives the currency.
COMMODITY_CURRENCIES: dict[str, list[str]] = {
    "AUD": ["gold", "iron ore", "coal", "copper", "mining"],
    "CAD": ["oil", "crude", "wti", "brent", "energy", "natural gas"],
    "NOK": ["oil", "crude", "brent", "energy", "natural gas"],
    "NZD": ["dairy", "milk", "agriculture"],
    "ZAR": ["gold", "platinum", "mining"],
    "CLP": ["copper", "mining"],
}

# Asset-class keywords that signal the article is about a different market
_UNRELATED_ASSET_KEYWORDS = [
    "gold price", "silver price", "oil price", "crude price",
    "bitcoin", "crypto", "ethereum", "s&p 500", "nasdaq",
    "dow jones", "nifty", "sensex",
]

# Regex patterns for detecting FX pair mentions like "EUR/USD", "EURUSD", "EUR-USD"
_PAIR_PATTERN = re.compile(
    r"\b([A-Z]{3})\s*[/-]?\s*([A-Z]{3})\b"
)


def _normalize_pair_variants(pair: str) -> list[str]:
    """Generate common text representations of a pair for matching."""
    base, quote = pair[:3].upper(), pair[3:].upper()
    return [
        f"{base}/{quote}",
        f"{base}{quote}",
        f"{base}-{quote}",
        f"{base} {quote}",
        f"{base.lower()}/{quote.lower()}",
        f"{base.lower()}{quote.lower()}",
    ]


def _find_other_pairs(text: str, target_base: str, target_quote: str) -> list[str]:
    """Find FX pair mentions in text that are NOT the target pair."""
    other_pairs = []
    known_currencies = set(CURRENCY_KEYWORDS.keys())
    for match in _PAIR_PATTERN.finditer(text.upper()):
        c1, c2 = match.group(1), match.group(2)
        if c1 in known_currencies and c2 in known_currencies:
            if not ({c1, c2} == {target_base, target_quote}):
                other_pairs.append(f"{c1}/{c2}")
    return other_pairs


def score_relevance(result: SearchResult, pair: str) -> float:
    """Score a SearchResult's relevance to a currency pair (0.0–1.0).

    Scoring rubric:
      +0.40  Direct pair mention in title
      +0.25  Direct pair mention in snippet only
      +0.25  Both base AND quote currency keywords present
      +0.15  Only one currency keyword present
      +0.02  Per general FX keyword hit (max +0.15)
      +0.10  Source is pair-specific central bank
      -0.20  Different pair prominently in title
      -0.15  Unrelated asset class in title (unless commodity-currency)
    """
    base = pair[:3].upper()
    quote = pair[3:].upper()

    title = result.title or ""
    snippet = result.snippet or ""
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    combined_lower = f"{title_lower} {snippet_lower}"
    source_lower = (result.source or "").lower()

    score = 0.0

    # --- Positive signals ---

    # Direct pair mention
    pair_variants = _normalize_pair_variants(pair)
    pair_in_title = any(v.lower() in title_lower for v in pair_variants)
    pair_in_snippet = any(v.lower() in snippet_lower for v in pair_variants)

    if pair_in_title:
        score += 0.40
    elif pair_in_snippet:
        score += 0.25

    # Currency keyword presence
    base_keywords = CURRENCY_KEYWORDS.get(base, [base.lower()])
    quote_keywords = CURRENCY_KEYWORDS.get(quote, [quote.lower()])

    has_base = any(kw in combined_lower for kw in base_keywords)
    has_quote = any(kw in combined_lower for kw in quote_keywords)

    if has_base and has_quote:
        score += 0.25
    elif has_base or has_quote:
        score += 0.15

    # General FX keyword density
    fx_hits = sum(1 for kw in GENERAL_FX_KEYWORDS if kw in combined_lower)
    score += min(fx_hits * 0.02, 0.15)

    # Central bank source bonus
    _cb_url_fragments = {
        "USD": ["federalreserve.gov", "treasury.gov"],
        "JPY": ["boj.or.jp"],
        "EUR": ["ecb.europa.eu"],
        "GBP": ["bankofengland.co.uk"],
        "AUD": ["rba.gov.au"],
        "NZD": ["rbnz.govt.nz"],
        "CHF": ["snb.ch"],
        "CAD": ["bankofcanada.ca"],
    }
    for ccy in (base, quote):
        for frag in _cb_url_fragments.get(ccy, []):
            if frag in source_lower or frag in (result.url or "").lower():
                score += 0.10
                break

    # --- Negative signals ---

    # Different pair prominently in title
    other_pairs_in_title = _find_other_pairs(title, base, quote)
    if other_pairs_in_title and not pair_in_title:
        score -= 0.20

    # Unrelated asset class in title
    pair_currencies = {base, quote}
    commodity_exemptions: set[str] = set()
    for ccy in pair_currencies:
        commodity_exemptions.update(COMMODITY_CURRENCIES.get(ccy, []))

    for asset_kw in _UNRELATED_ASSET_KEYWORDS:
        if asset_kw in title_lower:
            # Check if this commodity is exempted for the pair's currencies
            if not any(exempt in asset_kw for exempt in commodity_exemptions):
                score -= 0.15
                break

    return max(0.0, min(1.0, score))


def filter_relevant(
    results: list[SearchResult],
    pair: str,
    threshold: float = 0.20,
) -> list[SearchResult]:
    """Filter search results by relevance score, attaching scores to kept results.

    Args:
        results: List of SearchResult to filter.
        pair: Target currency pair (e.g. "USDJPY").
        threshold: Minimum relevance score to keep (default 0.20).

    Returns:
        Filtered list with relevance_score populated on each result.
    """
    kept: list[SearchResult] = []
    for r in results:
        s = score_relevance(r, pair)
        r.relevance_score = s
        if s >= threshold:
            kept.append(r)
        else:
            logger.debug(
                "Filtered out (score=%.2f < %.2f): %s",
                s, threshold, r.title[:80] if r.title else "(no title)",
            )
    return kept
