"""Example: Brave Search web search provider.

This shows how to register a custom web search backend using the
``@web_search_provider`` decorator.  Replace the placeholder logic with
real Brave Search API calls for production use.

Requires a Brave Search API key (https://brave.com/search/api/).
Set the ``BRAVE_SEARCH_API_KEY`` environment variable.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

import httpx

from aia_forecaster.models import SearchResult
from aia_forecaster.search.web_providers import web_search_provider

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


@web_search_provider("brave")
async def search_brave(
    query: str,
    max_results: int = 10,
    cutoff_date: date | None = None,
) -> list[SearchResult]:
    """Search the web via Brave Search API.

    Blacklist filtering is handled by the dispatch layer — this function
    returns raw results.
    """
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        logger.error("BRAVE_SEARCH_API_KEY not set — cannot use Brave search provider")
        return []

    if not query or not query.strip():
        logger.warning("Empty search query — skipping Brave search")
        return []

    params: dict[str, str | int] = {
        "q": query,
        "count": max_results,
    }
    # Brave supports freshness filter: pd (past day), pw (past week), pm (past month), py (past year)
    if cutoff_date:
        days = (date.today() - cutoff_date).days
        if 0 <= days <= 1:
            params["freshness"] = "pd"
        elif days <= 7:
            params["freshness"] = "pw"
        elif days <= 31:
            params["freshness"] = "pm"
        else:
            params["freshness"] = "py"

    results: list[SearchResult] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                BRAVE_API_URL,
                params=params,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    query=query,
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    url=item.get("url", ""),
                    source="brave",
                    timestamp=datetime.now(),
                )
            )

    except Exception:
        logger.exception("Brave search failed for query: %s", query)

    return results
