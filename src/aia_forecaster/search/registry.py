"""Data source registry — add new datasets with a single decorated function.

Usage
-----
    from aia_forecaster.search.registry import data_source, SearchResult

    @data_source("my_feed")
    async def fetch_my_feed(pair: str, cutoff_date, **kwargs) -> list[SearchResult]:
        # your logic here
        return [SearchResult(query=f"my_feed:{pair}", title=..., snippet=..., url="", source="my_feed")]

The decorated function is automatically registered and will be called by every
forecasting agent during its evidence-gathering phase.  No other wiring needed.

Function signature
------------------
Your function MUST accept at least:
    pair        : str           — e.g. "USDJPY"
    cutoff_date : datetime.date — temporal cutoff for foreknowledge filtering

It MAY accept additional **kwargs (max_results, max_age_hours, etc.) which the
agent will pass through when available.

It MUST return ``list[SearchResult]``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import date
from typing import Awaitable, Callable

from aia_forecaster.models import SearchResult

logger = logging.getLogger(__name__)

# Type alias for a data source callable
DataSourceFn = Callable[..., Awaitable[list[SearchResult]]]


class _Registry:
    """Internal registry of data source functions."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSourceFn] = {}

    def register(self, name: str, fn: DataSourceFn) -> None:
        if name in self._sources:
            logger.warning("Data source '%s' already registered — overwriting", name)
        self._sources[name] = fn
        logger.debug("Registered data source: %s", name)

    def unregister(self, name: str) -> None:
        self._sources.pop(name, None)

    @property
    def names(self) -> list[str]:
        return list(self._sources.keys())

    def get(self, name: str) -> DataSourceFn | None:
        return self._sources.get(name)

    def all(self) -> dict[str, DataSourceFn]:
        return dict(self._sources)


_registry = _Registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def data_source(name: str):
    """Decorator that registers an async function as a data source.

    Example::

        @data_source("bloomberg")
        async def fetch_bloomberg(pair: str, cutoff_date, **kwargs):
            ...
            return [SearchResult(...)]
    """

    def decorator(fn: DataSourceFn) -> DataSourceFn:
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"Data source '{name}' must be an async function")
        _registry.register(name, fn)
        # Store metadata on the function for introspection
        fn._data_source_name = name  # type: ignore[attr-defined]
        return fn

    return decorator


def register(name: str, fn: DataSourceFn) -> None:
    """Imperatively register a data source function (alternative to decorator)."""
    _registry.register(name, fn)


def unregister(name: str) -> None:
    """Remove a data source from the registry."""
    _registry.unregister(name)


def list_sources() -> list[str]:
    """Return names of all registered data sources."""
    _load_builtins()
    return _registry.names


def get_source(name: str) -> DataSourceFn | None:
    """Get a specific data source function by name."""
    return _registry.get(name)


_builtins_loaded = False


def _load_builtins() -> None:
    """Import built-in data source modules so their @data_source decorators run."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    try:
        import aia_forecaster.search.rss  # noqa: F401 — triggers @data_source("rss")
    except Exception:
        logger.debug("Could not load built-in RSS data source")


async def fetch_all(
    pair: str,
    cutoff_date: date,
    *,
    source_names: list[str] | None = None,
    **kwargs,
) -> dict[str, list[SearchResult]]:
    """Fetch evidence from all (or selected) registered data sources in parallel.

    Args:
        pair: Currency pair, e.g. "USDJPY".
        cutoff_date: Temporal cutoff date.
        source_names: If given, only fetch from these sources. Otherwise fetch all.
        **kwargs: Passed through to each source (e.g. max_results, max_age_hours).

    Returns:
        Dict mapping source name → list of SearchResult.
    """
    _load_builtins()
    sources = _registry.all()
    if source_names is not None:
        sources = {k: v for k, v in sources.items() if k in source_names}

    async def _safe_fetch(name: str, fn: DataSourceFn) -> tuple[str, list[SearchResult]]:
        try:
            # Pass only kwargs the function accepts
            sig = inspect.signature(fn)
            params = sig.parameters
            call_kwargs: dict = {"pair": pair, "cutoff_date": cutoff_date}
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                call_kwargs.update(kwargs)
            else:
                for k, v in kwargs.items():
                    if k in params:
                        call_kwargs[k] = v
            results = await fn(**call_kwargs)
            logger.info("Data source '%s': returned %d results", name, len(results))
            return name, results
        except Exception:
            logger.exception("Data source '%s' failed", name)
            return name, []

    tasks = [_safe_fetch(name, fn) for name, fn in sources.items()]
    pairs = await asyncio.gather(*tasks)
    return dict(pairs)
