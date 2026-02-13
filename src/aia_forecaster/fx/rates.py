"""Live FX spot rate fetching."""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Simple in-memory cache for spot rates (5-minute TTL)
_rate_cache: dict[str, tuple[float, float]] = {}  # pair -> (rate, timestamp)
_CACHE_TTL = 300  # 5 minutes


def _pair_to_api_format(pair: str) -> tuple[str, str]:
    """Convert 'USDJPY' to ('USD', 'JPY')."""
    return pair[:3].upper(), pair[3:].upper()


async def get_spot_rate(pair: str = "USDJPY") -> float:
    """Fetch the current spot rate for a currency pair.

    Uses exchangerate.host (free, no API key required).

    Args:
        pair: Currency pair string, e.g. 'USDJPY'.

    Returns:
        Current spot rate (e.g., 155.23 for USD/JPY).
    """
    # Check cache
    if pair in _rate_cache:
        rate, ts = _rate_cache[pair]
        if time.time() - ts < _CACHE_TTL:
            return rate

    base, quote = _pair_to_api_format(pair)

    # Try exchangerate.host first (free, no key)
    url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rate = float(data["rates"][quote])
            _rate_cache[pair] = (rate, time.time())
            return rate
    except Exception:
        logger.warning("exchangerate.host failed, trying fallback")

    # Fallback: open.er-api.com (also free)
    url = f"https://open.er-api.com/v6/latest/{base}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rate = float(data["rates"][quote])
            _rate_cache[pair] = (rate, time.time())
            return rate
    except Exception:
        logger.exception("All FX rate APIs failed for %s", pair)
        raise RuntimeError(f"Could not fetch spot rate for {pair}")
