"""Web search using DuckDuckGo (free, no API key required)."""

from __future__ import annotations

import logging
from datetime import date, datetime

from duckduckgo_search import DDGS

from aia_forecaster.models import SearchResult
from aia_forecaster.storage.cache import SearchCache

logger = logging.getLogger(__name__)

# Prediction market domains to blacklist (can leak foreknowledge)
BLACKLISTED_DOMAINS = [
    "polymarket.com",
    "metaculus.com",
    "manifold.markets",
    "kalshi.com",
    "predictit.org",
    "smarkets.com",
]

# Utility/tool domains that are never relevant to FX analysis
IRRELEVANT_DOMAINS = [
    "calculator.net",
    "calculateconvert.com",
    "gigacalculator.com",
    "calculatorsoup.com",
    "timeanddate.com",
    "convertunits.com",
    "rapidtables.com",
    "unitconverters.net",
    "mathsisfun.com",
    "daysuntil.net",
    "epochconverter.com",
]

_cache = SearchCache()


def _is_blacklisted(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in BLACKLISTED_DOMAINS) or any(
        domain in url_lower for domain in IRRELEVANT_DOMAINS
    )


async def search_web(
    query: str,
    max_results: int = 10,
    cutoff_date: date | None = None,
) -> list[SearchResult]:
    """Search the web via DuckDuckGo and return structured results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        cutoff_date: If set, filter results to before this date.

    Returns:
        List of SearchResult objects, with blacklisted domains removed.
    """
    # Reject empty or whitespace-only queries
    if not query or not query.strip():
        logger.warning("Empty search query — skipping web search")
        return []

    # Check cache first
    cache_key = f"web:{query}:{cutoff_date}"
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for query: %s", query)
        return [SearchResult(**r) for r in cached]

    # Build query with temporal filter
    search_query = query
    if cutoff_date:
        search_query = f"{query} before:{cutoff_date.isoformat()}"

    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(search_query, max_results=max_results + 5))

        for r in raw_results:
            url = r.get("href", r.get("link", ""))
            if _is_blacklisted(url):
                logger.debug("Filtered blacklisted URL: %s", url)
                continue

            results.append(
                SearchResult(
                    query=query,
                    title=r.get("title", ""),
                    snippet=r.get("body", r.get("snippet", "")),
                    url=url,
                    source="duckduckgo",
                    timestamp=datetime.now(),
                )
            )

            if len(results) >= max_results:
                break

    except Exception:
        logger.exception("DuckDuckGo search failed for query: %s", query)

    # Cache results
    _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
    return results
