"""RSS feed aggregation for FX-relevant news."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

import feedparser

from aia_forecaster.models import SearchResult
from aia_forecaster.storage.cache import SearchCache

logger = logging.getLogger(__name__)

# Curated FX-relevant RSS feeds
FX_RSS_FEEDS = [
    # Central bank & macro
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.ecb.europa.eu/rss/press.html",
    # FX news
    "https://www.fxstreet.com/rss",
    "https://www.forexlive.com/feed/",
    # General financial
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    # Japan-specific (for JPY)
    "https://www.japantimes.co.jp/feed/",
    "https://english.kyodonews.net/rss/all.xml",
]

# Keywords mapped by currency for filtering headlines
CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "JPY": ["jpy", "yen", "japan", "boj", "bank of japan", "ueda", "japanese"],
    "USD": ["usd", "dollar", "fed", "federal reserve", "fomc", "powell", "treasury"],
    "EUR": ["eur", "euro", "ecb", "european central bank", "lagarde", "eurozone"],
    "GBP": ["gbp", "pound", "sterling", "bank of england", "boe", "bailey"],
    "CHF": ["chf", "franc", "swiss", "snb", "swiss national bank"],
    "AUD": ["aud", "aussie", "rba", "reserve bank of australia"],
    "CAD": ["cad", "loonie", "bank of canada", "boc"],
    "NZD": ["nzd", "kiwi", "rbnz", "reserve bank of new zealand"],
}

# General FX keywords that match any pair
GENERAL_FX_KEYWORDS = [
    "forex", "fx", "exchange rate", "currency", "carry trade",
    "risk-on", "risk-off", "safe haven", "interest rate",
    "inflation", "gdp", "employment", "trade balance",
    "central bank", "monetary policy", "rate hike", "rate cut",
    "hawkish", "dovish", "quantitative", "yield",
]

_cache = SearchCache()


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


async def fetch_fx_news(
    currency_pair: str = "USDJPY",
    max_age_hours: int = 48,
    max_results: int = 30,
) -> list[SearchResult]:
    """Fetch recent FX-relevant news from RSS feeds.

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

    for feed_url in FX_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            logger.warning("Failed to parse feed: %s", feed_url)
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
                    source=feed_url,
                    timestamp=datetime.now(),
                )
            )

            if len(results) >= max_results:
                break

        if len(results) >= max_results:
            break

    _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
    logger.info("Fetched %d FX news items for %s", len(results), currency_pair)
    return results
