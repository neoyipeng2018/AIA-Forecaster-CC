"""Web search using DuckDuckGo (free, no API key required)."""

from __future__ import annotations

import logging
import re
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


def _sanitize_query(query: str) -> str:
    """Sanitize a search query for DuckDuckGo compatibility.

    Strips site: operators, boolean AND/OR, parentheses, and
    truncates to 300 characters at a word boundary.
    """
    q = query
    # Strip site: operators (e.g. site:reuters.com)
    q = re.sub(r'site:\S+', '', q)
    # Strip before:/after: date operators (Google syntax, not DDG)
    q = re.sub(r'(?:before|after):\S+', '', q)
    # Strip boolean operators — convert to spaces
    q = re.sub(r'\b(AND|OR|NOT)\b', ' ', q)
    # Strip parentheses and quotation marks
    q = re.sub(r'[()"\']', ' ', q)
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q).strip()
    # Truncate to 300 chars at word boundary
    if len(q) > 300:
        q = q[:300].rsplit(' ', 1)[0]
    return q


def _compute_timelimit(cutoff_date: date) -> str | None:
    """Compute DDG timelimit parameter from cutoff date.

    Returns 'd' (day), 'w' (week), 'm' (month), 'y' (year), or None.
    """
    days = (date.today() - cutoff_date).days
    if days < 0:
        # Cutoff is in the future — no restriction needed
        return None
    if days <= 1:
        return "d"
    if days <= 7:
        return "w"
    if days <= 31:
        return "m"
    return "y"


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

    # Sanitize query for DDG compatibility
    search_query = _sanitize_query(query)
    if not search_query:
        logger.warning("Query empty after sanitization — skipping web search")
        return []

    # Compute DDG-native time limit from cutoff date
    timelimit = _compute_timelimit(cutoff_date) if cutoff_date else None

    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(
                search_query,
                max_results=max_results + 5,
                timelimit=timelimit,
            ))

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

    # Only cache non-empty results to avoid poisoning cache with failures
    if results:
        _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
    return results
