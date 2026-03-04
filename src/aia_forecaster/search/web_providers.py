"""Pluggable web search provider registry and dispatch.

Mirrors the ``@data_source()`` pattern from ``registry.py`` but for web search
backends.  The ``search_web()`` dispatch function fans out to **all** active
providers in parallel, deduplicates by URL, and applies shared blacklist
filtering on the merged results.

Quick start — registering a new backend::

    from aia_forecaster.search.web_providers import web_search_provider

    @web_search_provider("brave")
    async def search_brave(query, max_results=10, cutoff_date=None):
        ...
        return [SearchResult(...), ...]

Selecting active providers at runtime::

    from aia_forecaster.search.web_providers import set_web_providers
    set_web_providers(["duckduckgo", "brave"])   # agents query both in parallel

    # Or a single provider:
    set_web_provider("brave")
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Awaitable, Callable

from aia_forecaster.models import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

WebSearchFn = Callable[..., Awaitable[list[SearchResult]]]

# ---------------------------------------------------------------------------
# Shared blacklists (applied to ALL providers)
# ---------------------------------------------------------------------------

# Custom blacklisted domains (add via add_blacklisted_domains() or company extensions).
# Prediction markets are intentionally NOT blacklisted -- they provide valuable
# probability signals for forward-looking forecasts.
BLACKLISTED_DOMAINS: list[str] = []

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


def add_blacklisted_domains(domains: list[str]) -> None:
    """Add domains to the blacklist so they are filtered from search results.

    Use this to block company-specific domains (e.g., internal wikis that
    leak information, or additional prediction market sites).

    Example::

        add_blacklisted_domains(["internal-wiki.example.com", "insight.example.com"])
    """
    BLACKLISTED_DOMAINS.extend(domains)


def _is_blacklisted(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in BLACKLISTED_DOMAINS) or any(
        domain in url_lower for domain in IRRELEVANT_DOMAINS
    )


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_providers: dict[str, WebSearchFn] = {}
_active: list[str] = ["duckduckgo"]


def web_search_provider(name: str):
    """Decorator that registers an async function as a web search provider.

    Example::

        @web_search_provider("brave")
        async def search_brave(query, max_results=10, cutoff_date=None):
            ...
            return [SearchResult(...)]
    """

    def decorator(fn: WebSearchFn) -> WebSearchFn:
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"Web search provider '{name}' must be an async function")
        if name in _providers:
            logger.warning("Web search provider '%s' already registered — overwriting", name)
        _providers[name] = fn
        logger.debug("Registered web search provider: %s", name)
        return fn

    return decorator


def set_web_provider(name: str) -> None:
    """Set a single active web search provider.

    Raises ``ValueError`` if the provider has not been registered.
    """
    set_web_providers([name])


def set_web_providers(names: list[str]) -> None:
    """Set one or more active web search providers.

    When multiple providers are active, ``search_web()`` fans out to all
    of them in parallel and merges the results (deduplicated by URL).

    Raises ``ValueError`` if any provider has not been registered.
    """
    _load_providers()
    for name in names:
        if name not in _providers:
            available = ", ".join(sorted(_providers.keys())) or "(none)"
            raise ValueError(
                f"Unknown web search provider '{name}'. Available: {available}"
            )
    global _active
    _active = list(names)
    logger.info("Active web search providers set to %s", _active)


def get_web_providers() -> list[str]:
    """Return the names of all currently active web search providers."""
    return list(_active)


def get_web_provider() -> str:
    """Return the name of the first active web search provider.

    Convenience function for backward compatibility and single-provider usage.
    """
    return _active[0] if _active else "duckduckgo"


def list_web_providers() -> list[str]:
    """Return names of all registered (available) web search providers."""
    _load_providers()
    return sorted(_providers.keys())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def search_web(
    query: str,
    max_results: int = 10,
    cutoff_date: date | None = None,
) -> list[SearchResult]:
    """Search the web via all active providers in parallel, deduplicate, and filter.

    This is the main entry point that agents should call.  When multiple
    providers are active (e.g. ``["duckduckgo", "my_news_api"]``), each is
    queried concurrently and the results are merged, deduplicated by URL,
    then stripped of blacklisted/irrelevant domains.
    """
    _load_providers()

    # Resolve active provider functions
    active_fns: list[tuple[str, WebSearchFn]] = []
    for name in _active:
        fn = _providers.get(name)
        if fn is None:
            logger.error("No web search provider '%s' registered — skipping", name)
            continue
        active_fns.append((name, fn))

    if not active_fns:
        logger.error("No active web search providers available")
        return []

    # Fan out to all active providers in parallel
    async def _call(name: str, fn: WebSearchFn) -> list[SearchResult]:
        try:
            return await fn(query, max_results=max_results, cutoff_date=cutoff_date)
        except Exception:
            logger.exception("Web search provider '%s' failed for query: %s", name, query)
            return []

    provider_results = await asyncio.gather(
        *[_call(name, fn) for name, fn in active_fns]
    )

    # Merge and deduplicate by URL (first occurrence wins)
    seen_urls: set[str] = set()
    merged: list[SearchResult] = []
    for results in provider_results:
        for r in results:
            url_key = r.url.rstrip("/").lower()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            merged.append(r)

    # Apply shared blacklist filtering
    filtered: list[SearchResult] = []
    for r in merged:
        if _is_blacklisted(r.url):
            logger.debug("Filtered blacklisted URL: %s", r.url)
            continue
        filtered.append(r)

    return filtered


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

_providers_loaded = False


def _load_providers() -> None:
    """Import built-in and company web search provider modules."""
    global _providers_loaded
    if _providers_loaded:
        return
    _providers_loaded = True

    # Built-in: DuckDuckGo
    try:
        import aia_forecaster.search.web  # noqa: F401 — triggers @web_search_provider("duckduckgo")
    except Exception:
        logger.debug("Could not load built-in DuckDuckGo web search provider")

    # Company-provided providers
    try:
        import company.search  # noqa: F401 — may register additional @web_search_provider() backends
    except ImportError:
        pass  # No company package — running upstream
    except Exception:
        logger.warning("Failed to load company search extensions for web providers", exc_info=True)
