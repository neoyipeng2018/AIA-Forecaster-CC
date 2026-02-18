"""RSS feed aggregation for FX-relevant news."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import feedparser
import httpx

from aia_forecaster.models import SearchResult
from aia_forecaster.storage.cache import SearchCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feed configuration
# ---------------------------------------------------------------------------


@dataclass
class FeedConfig:
    """Configuration for a single RSS feed."""

    url: str
    category: str  # central_bank, fx_specific, macro_data, commodity, geopolitical, general
    currencies: list[str] = field(default_factory=list)  # empty = all pairs


FX_FEEDS: list[FeedConfig] = [
    # === Central Banks ===
    FeedConfig("https://www.federalreserve.gov/feeds/press_all.xml", "central_bank", ["USD"]),
    FeedConfig("https://www.ecb.europa.eu/rss/press.html", "central_bank", ["EUR"]),
    FeedConfig("https://www.boj.or.jp/en/rss/whatsnew.xml", "central_bank", ["JPY"]),
    FeedConfig("https://www.bankofengland.co.uk/rss/news", "central_bank", ["GBP"]),
    FeedConfig("https://www.rba.gov.au/rss/rss-cb-media-releases.xml", "central_bank", ["AUD"]),
    FeedConfig("https://www.rbnz.govt.nz/rss/news.xml", "central_bank", ["NZD"]),
    FeedConfig("https://www.snb.ch/en/mmr/reference/rss_en/source/rss_en.en.xml", "central_bank", ["CHF"]),
    FeedConfig("https://www.bankofcanada.ca/content_type/press-releases/feed/", "central_bank", ["CAD"]),
    # === FX-Specific News ===
    FeedConfig("https://www.fxstreet.com/rss", "fx_specific"),
    FeedConfig("https://www.forexlive.com/feed/", "fx_specific"),
    FeedConfig("https://www.dailyfx.com/feeds/all", "fx_specific"),
    FeedConfig("https://www.investing.com/rss/news_14.rss", "fx_specific"),
    # === Macro Data Sources ===
    FeedConfig("https://www.bls.gov/feed/bls_latest.rss", "macro_data", ["USD"]),
    FeedConfig("https://www.bea.gov/news/feed", "macro_data", ["USD"]),
    FeedConfig("https://ec.europa.eu/eurostat/web/main/news/euro-indicators/feed", "macro_data", ["EUR"]),
    # === Commodity / Energy (affects AUD, CAD, NOK) ===
    FeedConfig("https://oilprice.com/rss/main", "commodity", ["CAD", "NOK", "AUD"]),
    # === Geopolitical / Trade ===
    FeedConfig("https://www.wto.org/english/news_e/news_e.rss", "geopolitical"),
    # === General Financial ===
    FeedConfig("https://feeds.reuters.com/reuters/businessNews", "general"),
    FeedConfig("https://feeds.bbci.co.uk/news/business/rss.xml", "general"),
    FeedConfig("https://www.cnbc.com/id/100727362/device/rss/rss.html", "general"),
    # === Regional ===
    FeedConfig("https://www.japantimes.co.jp/feed/", "general", ["JPY"]),
    FeedConfig("https://english.kyodonews.net/rss/all.xml", "general", ["JPY"]),
    FeedConfig("https://www.theguardian.com/business/rss", "general", ["GBP"]),
    FeedConfig("https://www.smh.com.au/rss/business.xml", "general", ["AUD"]),
]

# Backward compatibility: flat URL list
FX_RSS_FEEDS = [f.url for f in FX_FEEDS]


def register_feed(feed: FeedConfig) -> None:
    """Register a single custom RSS feed.

    Use this to add company-specific feeds without modifying upstream code.

    Example::

        register_feed(FeedConfig(
            "https://internal.example.com/fx-news/rss",
            "fx_specific", ["USD", "EUR"],
        ))
    """
    FX_FEEDS.append(feed)
    FX_RSS_FEEDS.append(feed.url)


def register_feeds(feeds: list[FeedConfig]) -> None:
    """Register multiple custom RSS feeds at once."""
    for feed in feeds:
        register_feed(feed)


# ---------------------------------------------------------------------------
# Currency keyword maps
# ---------------------------------------------------------------------------

CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "JPY": [
        "jpy", "yen", "japan", "boj", "bank of japan", "ueda", "japanese",
        "nikkei", "tokyo",
    ],
    "USD": [
        "usd", "dollar", "fed", "federal reserve", "fomc", "powell", "treasury",
        "nonfarm", "payroll", "cpi", "pce",
    ],
    "EUR": [
        "eur", "euro", "ecb", "european central bank", "lagarde", "eurozone",
        "eurostat", "german",
    ],
    "GBP": [
        "gbp", "pound", "sterling", "bank of england", "boe", "bailey",
        "uk economy", "british",
    ],
    "CHF": [
        "chf", "franc", "swiss", "snb", "swiss national bank", "switzerland",
    ],
    "AUD": [
        "aud", "aussie", "rba", "reserve bank of australia", "australia",
        "iron ore", "bullock",
    ],
    "CAD": [
        "cad", "loonie", "bank of canada", "boc", "canada", "canadian",
        "crude oil", "macklem",
    ],
    "NZD": [
        "nzd", "kiwi", "rbnz", "reserve bank of new zealand", "new zealand",
        "dairy",
    ],
    "NOK": [
        "nok", "norwegian", "norges bank", "krone", "norway", "brent crude",
    ],
    "SEK": [
        "sek", "swedish", "riksbank", "krona", "sweden",
    ],
}

GENERAL_FX_KEYWORDS = [
    "forex", "fx", "exchange rate", "currency", "carry trade",
    "risk-on", "risk-off", "safe haven", "interest rate",
    "inflation", "gdp", "employment", "trade balance",
    "central bank", "monetary policy", "rate hike", "rate cut",
    "hawkish", "dovish", "quantitative", "yield",
    "rate decision", "forward guidance", "tariff", "pmi",
    "bond yield", "vix", "retail sales", "current account",
]


def register_currency_keywords(currency: str, keywords: list[str]) -> None:
    """Extend the keyword map for a currency with additional keywords.

    Adds to existing keywords rather than replacing them.

    Example::

        register_currency_keywords("CNH", ["renminbi", "pboc", "yuan", "china"])
    """
    currency = currency.upper()
    if currency in CURRENCY_KEYWORDS:
        CURRENCY_KEYWORDS[currency].extend(keywords)
    else:
        CURRENCY_KEYWORDS[currency] = list(keywords)


# ---------------------------------------------------------------------------
# Feed health tracking
# ---------------------------------------------------------------------------

_feed_health: dict[str, dict] = {}


def _record_feed_result(url: str, success: bool) -> None:
    if url not in _feed_health:
        _feed_health[url] = {"last_success": None, "failures": 0}
    if success:
        _feed_health[url]["last_success"] = datetime.now(timezone.utc)
        _feed_health[url]["failures"] = 0
    else:
        _feed_health[url]["failures"] += 1


def get_feed_health() -> dict[str, dict]:
    """Return current feed health status for diagnostics."""
    return dict(_feed_health)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_cache = SearchCache()


def _feeds_for_pair(pair: str) -> list[FeedConfig]:
    """Return feeds relevant to a specific currency pair."""
    base = pair[:3].upper()
    quote = pair[3:].upper()
    return [
        f for f in FX_FEEDS
        if not f.currencies or base in f.currencies or quote in f.currencies
    ]


def _pair_keywords(pair: str) -> list[str]:
    """Get all relevant keywords for a currency pair like 'USDJPY'."""
    base = pair[:3].upper()
    quote = pair[3:].upper()
    keywords = list(GENERAL_FX_KEYWORDS)
    keywords.extend(CURRENCY_KEYWORDS.get(base, [base.lower()]))
    keywords.extend(CURRENCY_KEYWORDS.get(quote, [quote.lower()]))
    return keywords


def _headline_matches(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the given keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _entry_hash(title: str, link: str) -> str:
    return hashlib.sha256(f"{title}:{link}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Async feed fetching
# ---------------------------------------------------------------------------

_FEED_TIMEOUT = 10  # seconds per feed


async def _fetch_feed_async(url: str) -> feedparser.FeedParserDict | None:
    """Fetch and parse a single RSS feed asynchronously."""
    try:
        async with httpx.AsyncClient(timeout=_FEED_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.text)
            _record_feed_result(url, success=True)
            return parsed
    except Exception:
        logger.warning("Failed to fetch feed: %s", url)
        _record_feed_result(url, success=False)
        return None


async def fetch_fx_news(
    currency_pair: str = "USDJPY",
    max_age_hours: int = 48,
    max_results: int = 30,
) -> list[SearchResult]:
    """Fetch recent FX-relevant news from RSS feeds.

    Fetches feeds in parallel, filtered by currency pair relevance.

    Args:
        currency_pair: E.g. 'USDJPY'.
        max_age_hours: Only include entries newer than this.
        max_results: Maximum results to return.

    Returns:
        Filtered, deduplicated list of SearchResult.
    """
    cache_key = f"rss:{currency_pair}:{max_age_hours}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return [SearchResult(**r) for r in cached]

    keywords = _pair_keywords(currency_pair)
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    seen_hashes: set[str] = set()
    results: list[SearchResult] = []

    # Fetch only feeds relevant to this pair, in parallel
    feeds = _feeds_for_pair(currency_pair)
    tasks = [_fetch_feed_async(f.url) for f in feeds]
    parsed_feeds = await asyncio.gather(*tasks)

    for feed_config, feed in zip(feeds, parsed_feeds):
        if feed is None:
            continue

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            link = entry.get("link", "")

            # Dedup
            h = _entry_hash(title, link)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Temporal filter
            published = entry.get("published_parsed")
            if published:
                try:
                    pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                    if pub_dt < cutoff_time:
                        continue
                except (TypeError, ValueError):
                    pass

            # Keyword filter
            combined_text = f"{title} {summary}"
            if not _headline_matches(combined_text, keywords):
                continue

            results.append(
                SearchResult(
                    query=f"rss:{currency_pair}",
                    title=title,
                    snippet=summary[:500],
                    url=link,
                    source=feed_config.url,
                    timestamp=datetime.now(),
                )
            )

            if len(results) >= max_results:
                break

        if len(results) >= max_results:
            break

    _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
    logger.info("Fetched %d FX news items for %s from %d feeds", len(results), currency_pair, len(feeds))
    return results


# ---------------------------------------------------------------------------
# Register as a data source
# ---------------------------------------------------------------------------

from aia_forecaster.search.registry import data_source  # noqa: E402


@data_source("rss")
async def _rss_data_source(
    pair: str, cutoff_date: date | None = None, **kwargs
) -> list[SearchResult]:
    """RSS adapter for the data source registry."""
    max_age_hours = kwargs.get("max_age_hours", 48)
    max_results = kwargs.get("max_results", 20)
    return await fetch_fx_news(
        currency_pair=pair,
        max_age_hours=max_age_hours,
        max_results=max_results,
    )
