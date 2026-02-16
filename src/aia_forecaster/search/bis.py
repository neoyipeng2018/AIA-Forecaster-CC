"""BIS central bank speeches data source for FX forecasting.

The Bank for International Settlements publishes speeches from central bank
governors and senior officials at https://www.bis.org/doclist/cbspeeches.rss.
These are high-signal for FX: forward guidance, rate signals, and policy
commentary directly influence currency movements.

The feed uses RSS 1.0 (RDF) with a custom ``cb:`` namespace.  The
``cb:institutionAbbrev`` field is always "BIS" — the actual central bank is
extracted from the ``<description>`` text (e.g. "Speech by Mr Andrew Bailey,
Governor of the Bank of England…").
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx

from aia_forecaster.models import SearchResult
from aia_forecaster.search.rss import CURRENCY_KEYWORDS, _headline_matches
from aia_forecaster.storage.cache import SearchCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Institution → currency mapping
# ---------------------------------------------------------------------------

INSTITUTION_CURRENCY_MAP: dict[str, str] = {
    # G10 central banks
    "Federal Reserve": "USD",
    "Board of Governors": "USD",
    "Bank of England": "GBP",
    "Bank of Japan": "JPY",
    "European Central Bank": "EUR",
    "Reserve Bank of Australia": "AUD",
    "Reserve Bank of New Zealand": "NZD",
    "Bank of Canada": "CAD",
    "Swiss National Bank": "CHF",
    "Sveriges Riksbank": "SEK",
    "Norges Bank": "NOK",
    # Eurozone national central banks → EUR
    "Deutsche Bundesbank": "EUR",
    "Bundesbank": "EUR",
    "Banque de France": "EUR",
    "Banca d'Italia": "EUR",
    "Banco de España": "EUR",
    "De Nederlandsche Bank": "EUR",
    "Nationale Bank van België": "EUR",
    "Central Bank of Ireland": "EUR",
    "Bank of Finland": "EUR",
    "Oesterreichische Nationalbank": "EUR",
    "Banco de Portugal": "EUR",
    "Bank of Greece": "EUR",
}

# Fallback: well-known speaker surnames → currency
KNOWN_SPEAKERS: dict[str, str] = {
    "Powell": "USD",
    "Jefferson": "USD",
    "Waller": "USD",
    "Bailey": "GBP",
    "Broadbent": "GBP",
    "Ueda": "JPY",
    "Lagarde": "EUR",
    "de Guindos": "EUR",
    "Bullock": "AUD",
    "Orr": "NZD",
    "Macklem": "CAD",
    "Jordan": "CHF",
    "Schlegel": "CHF",
    "Thedéen": "SEK",
    "Bache": "NOK",
    "Nagel": "EUR",
    "Villeroy": "EUR",
    "Panetta": "EUR",
}

# ---------------------------------------------------------------------------
# Currency extraction
# ---------------------------------------------------------------------------


def _extract_currency(description: str, creator: str = "") -> str | None:
    """Extract the currency code from a speech description or creator field.

    Primary: substring match on description against institution names.
    Fallback: speaker surname match.
    """
    for institution, ccy in INSTITUTION_CURRENCY_MAP.items():
        if institution.lower() in description.lower():
            return ccy

    # Fallback: check speaker surname in description and creator
    combined = f"{description} {creator}"
    for surname, ccy in KNOWN_SPEAKERS.items():
        if surname.lower() in combined.lower():
            return ccy

    return None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rss": "http://purl.org/rss/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "cb": "http://www.cbwiki.net/wiki/index.php/Specification_1.1",
}


@dataclass
class BISSpeechEntry:
    """Parsed BIS speech entry."""

    title: str
    link: str
    description: str
    creator: str
    pub_date: str
    currency: str | None
    simple_title: str
    occurrence_date: str
    speaker_surname: str
    pdf_url: str


def _parse_bis_feed(xml_text: str) -> list[BISSpeechEntry]:
    """Parse BIS RSS 1.0 (RDF) XML into speech entries."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("Failed to parse BIS feed XML")
        return []

    entries: list[BISSpeechEntry] = []
    for item in root.findall("rss:item", _NS):
        title = (item.findtext("rss:title", "", _NS) or "").strip()
        link = (item.findtext("rss:link", "", _NS) or "").strip()
        description = (item.findtext("rss:description", "", _NS) or "").strip()
        creator = (item.findtext("dc:creator", "", _NS) or "").strip()
        pub_date = (item.findtext("dc:date", "", _NS) or "").strip()

        # cb:speech nested fields
        speech = item.find("cb:speech", _NS)
        simple_title = ""
        occurrence_date = ""
        speaker_surname = ""
        pdf_url = ""
        if speech is not None:
            simple_title = (speech.findtext("cb:simpleTitle", "", _NS) or "").strip()
            occurrence_date = (speech.findtext("cb:occurrenceDate", "", _NS) or "").strip()
            person = speech.find("cb:person", _NS)
            if person is not None:
                speaker_surname = (person.findtext("cb:surname", "", _NS) or "").strip()
            resource = speech.find("cb:resource", _NS)
            if resource is not None:
                pdf_url = (resource.findtext("cb:resourceLink", "", _NS) or "").strip()

        currency = _extract_currency(description, creator)

        entries.append(
            BISSpeechEntry(
                title=title,
                link=link,
                description=description,
                creator=creator,
                pub_date=pub_date,
                currency=currency,
                simple_title=simple_title,
                occurrence_date=occurrence_date,
                speaker_surname=speaker_surname,
                pdf_url=pdf_url,
            )
        )

    return entries


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------


def _speech_matches_pair(entry: BISSpeechEntry, pair: str) -> bool:
    """Check whether a speech entry is relevant to a currency pair.

    Level 1: institution-currency direct match (the speech's extracted
    currency is one of the pair's two currencies).
    Level 2: keyword match on the description text.
    """
    base = pair[:3].upper()
    quote = pair[3:].upper()

    # Level 1: direct currency match
    if entry.currency is not None and entry.currency in (base, quote):
        return True

    # Level 2: currency-specific keyword match only (no general FX terms)
    combined = f"{entry.title} {entry.description}"
    keywords: list[str] = []
    keywords.extend(CURRENCY_KEYWORDS.get(base, [base.lower()]))
    keywords.extend(CURRENCY_KEYWORDS.get(quote, [quote.lower()]))
    return _headline_matches(combined, keywords)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

_BIS_FEED_URL = "https://www.bis.org/doclist/cbspeeches.rss"
_FETCH_TIMEOUT = 15  # seconds
_cache = SearchCache()


async def fetch_bis_speeches(
    pair: str,
    max_age_hours: int = 168,
    max_results: int = 15,
) -> list[SearchResult]:
    """Fetch recent BIS central bank speeches relevant to a currency pair.

    Args:
        pair: Currency pair, e.g. "GBPUSD".
        max_age_hours: Only include speeches newer than this (default 7 days).
        max_results: Maximum results to return.

    Returns:
        Filtered list of SearchResult.
    """
    cache_key = f"bis_speeches:{pair}:{max_age_hours}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return [SearchResult(**r) for r in cached]

    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(_BIS_FEED_URL)
            resp.raise_for_status()
            xml_text = resp.text
    except Exception:
        logger.warning("Failed to fetch BIS speeches feed")
        return []

    entries = _parse_bis_feed(xml_text)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    results: list[SearchResult] = []

    for entry in entries:
        # Temporal filter — use occurrence_date or pub_date
        date_str = entry.occurrence_date or entry.pub_date
        if date_str:
            try:
                pub_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            except (ValueError, TypeError):
                pass

        # Pair relevance filter
        if not _speech_matches_pair(entry, pair):
            continue

        snippet = entry.description[:500]
        if entry.simple_title:
            snippet = f"{entry.simple_title} — {snippet}"

        results.append(
            SearchResult(
                query=f"bis_speeches:{pair}",
                title=entry.title,
                snippet=snippet[:500],
                url=entry.pdf_url or entry.link,
                source="bis.org",
                timestamp=datetime.now(),
            )
        )

        if len(results) >= max_results:
            break

    _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
    logger.info(
        "Fetched %d BIS speeches for %s from %d entries",
        len(results), pair, len(entries),
    )
    return results


# ---------------------------------------------------------------------------
# Register as a data source
# ---------------------------------------------------------------------------

from aia_forecaster.search.registry import data_source  # noqa: E402


@data_source("bis_speeches")
async def _bis_data_source(
    pair: str, cutoff_date: date | None = None, **kwargs
) -> list[SearchResult]:
    """BIS speeches adapter for the data source registry."""
    max_age_hours = kwargs.get("max_age_hours", 168)
    max_results = kwargs.get("max_results", 15)
    return await fetch_bis_speeches(
        pair=pair,
        max_age_hours=max_age_hours,
        max_results=max_results,
    )
