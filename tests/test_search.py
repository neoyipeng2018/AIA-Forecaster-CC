"""Tests for search modules."""

import json
import tempfile
import time
from pathlib import Path

from aia_forecaster.search.web import BLACKLISTED_DOMAINS, _is_blacklisted
from aia_forecaster.search.rss import _headline_matches, _pair_keywords
from aia_forecaster.search.bis import (
    _extract_currency,
    _parse_bis_feed,
    _speech_matches_pair,
    BISSpeechEntry,
)
from aia_forecaster.storage.cache import SearchCache


class TestBlacklist:
    def test_blacklisted_domains(self):
        assert _is_blacklisted("https://polymarket.com/market/123")
        assert _is_blacklisted("https://www.metaculus.com/questions/1234")
        assert _is_blacklisted("https://manifold.markets/foo")
        assert _is_blacklisted("https://kalshi.com/event/test")

    def test_non_blacklisted(self):
        assert not _is_blacklisted("https://reuters.com/article/test")
        assert not _is_blacklisted("https://bbc.com/news/test")
        assert not _is_blacklisted("https://fxstreet.com/news/test")


class TestKeywordMatching:
    def test_pair_keywords_usdjpy(self):
        keywords = _pair_keywords("USDJPY")
        assert "jpy" in keywords
        assert "yen" in keywords
        assert "boj" in keywords
        assert "usd" in keywords
        assert "fed" in keywords
        assert "forex" in keywords

    def test_headline_matches(self):
        keywords = _pair_keywords("USDJPY")
        assert _headline_matches("BOJ raises interest rates", keywords)
        assert _headline_matches("Federal Reserve holds steady", keywords)
        assert _headline_matches("Yen strengthens on risk aversion", keywords)
        assert not _headline_matches("Apple releases new iPhone", keywords)


class TestSearchCache:
    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SearchCache(cache_dir=Path(tmpdir), ttl_hours=1)
            data = [{"title": "test", "snippet": "hello"}]
            cache.set("test_key", data)

            result = cache.get("test_key")
            assert result is not None
            assert result[0]["title"] == "test"

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SearchCache(cache_dir=Path(tmpdir), ttl_hours=1)
            assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SearchCache(cache_dir=Path(tmpdir), ttl_hours=1)
            cache.set("key", [{"data": 1}])

            # Manually backdate the timestamp to force expiry
            path = cache._key_path("key")
            data = json.loads(path.read_text())
            data["ts"] = time.time() - 7200  # 2 hours ago (TTL is 1 hour)
            path.write_text(json.dumps(data))

            assert cache.get("key") is None

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SearchCache(cache_dir=Path(tmpdir), ttl_hours=1)
            cache.set("a", [{"x": 1}])
            cache.set("b", [{"y": 2}])
            count = cache.clear()
            assert count == 2
            assert cache.get("a") is None


# ---------------------------------------------------------------------------
# BIS speeches tests
# ---------------------------------------------------------------------------

class TestBISInstitutionMapping:
    def test_federal_reserve(self):
        desc = "Speech by Mr Jerome Powell, Chair of the Board of Governors of the Federal Reserve System"
        assert _extract_currency(desc) == "USD"

    def test_bank_of_england(self):
        desc = "Speech by Mr Andrew Bailey, Governor of the Bank of England"
        assert _extract_currency(desc) == "GBP"

    def test_bank_of_japan(self):
        desc = "Speech by Mr Kazuo Ueda, Governor of the Bank of Japan"
        assert _extract_currency(desc) == "JPY"

    def test_ecb(self):
        desc = "Speech by Ms Christine Lagarde, President of the European Central Bank"
        assert _extract_currency(desc) == "EUR"

    def test_bundesbank_maps_to_eur(self):
        desc = "Speech by Dr Joachim Nagel, President of the Deutsche Bundesbank"
        assert _extract_currency(desc) == "EUR"

    def test_rba(self):
        desc = "Speech by Ms Michele Bullock, Governor of the Reserve Bank of Australia"
        assert _extract_currency(desc) == "AUD"

    def test_boc(self):
        desc = "Speech by Mr Tiff Macklem, Governor of the Bank of Canada"
        assert _extract_currency(desc) == "CAD"

    def test_snb(self):
        desc = "Speech by Mr Martin Schlegel, Chairman of the Swiss National Bank"
        assert _extract_currency(desc) == "CHF"

    def test_unknown_institution(self):
        desc = "Speech by someone at an unknown institution"
        assert _extract_currency(desc) is None

    def test_speaker_fallback(self):
        # No institution name, but known speaker surname
        desc = "Remarks by Powell at the Economic Club"
        assert _extract_currency(desc) == "USD"

    def test_speaker_fallback_in_creator(self):
        desc = "Some speech without institution info"
        assert _extract_currency(desc, creator="Bailey") == "GBP"


class TestBISSpeechPairMatching:
    def _make_entry(self, currency: str | None, description: str = "") -> BISSpeechEntry:
        return BISSpeechEntry(
            title="Test speech",
            link="https://www.bis.org/test",
            description=description,
            creator="",
            pub_date="2026-02-10",
            currency=currency,
            simple_title="",
            occurrence_date="",
            speaker_surname="",
            pdf_url="",
        )

    def test_boe_matches_gbpusd(self):
        entry = self._make_entry("GBP")
        assert _speech_matches_pair(entry, "GBPUSD")

    def test_boe_does_not_match_usdjpy(self):
        entry = self._make_entry("GBP")
        assert not _speech_matches_pair(entry, "USDJPY")

    def test_fed_matches_usdjpy(self):
        entry = self._make_entry("USD")
        assert _speech_matches_pair(entry, "USDJPY")

    def test_fed_matches_eurusd(self):
        entry = self._make_entry("USD")
        assert _speech_matches_pair(entry, "EURUSD")

    def test_keyword_fallback(self):
        # No currency extracted, but description contains relevant keywords
        entry = self._make_entry(None, description="Discussion of yen exchange rate policy")
        assert _speech_matches_pair(entry, "USDJPY")

    def test_general_fx_keywords_do_not_match(self):
        # "interest rate" / "monetary policy" are general FX terms — should NOT match
        entry = self._make_entry(None, description="General remarks on interest rate policy and financial stability")
        assert not _speech_matches_pair(entry, "USDJPY")


_SAMPLE_BIS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns="http://purl.org/rss/1.0/"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:cb="http://www.cbwiki.net/wiki/index.php/Specification_1.1">
  <channel>
    <title>BIS Central bankers' speeches</title>
  </channel>
  <item rdf:about="https://www.bis.org/review/r260210a.htm">
    <title>Andrew Bailey: The economic outlook</title>
    <link>https://www.bis.org/review/r260210a.htm</link>
    <description>Speech by Mr Andrew Bailey, Governor of the Bank of England, at the Treasury Select Committee, London, 10 February 2026.</description>
    <dc:creator>Andrew Bailey</dc:creator>
    <dc:date>2026-02-10</dc:date>
    <cb:speech>
      <cb:simpleTitle>The economic outlook</cb:simpleTitle>
      <cb:occurrenceDate>2026-02-10</cb:occurrenceDate>
      <cb:person>
        <cb:surname>Bailey</cb:surname>
      </cb:person>
      <cb:resource>
        <cb:resourceLink>https://www.bis.org/review/r260210a.pdf</cb:resourceLink>
      </cb:resource>
    </cb:speech>
  </item>
</rdf:RDF>
"""


class TestBISXMLParsing:
    def test_parse_sample_entry(self):
        entries = _parse_bis_feed(_SAMPLE_BIS_XML)
        assert len(entries) == 1
        e = entries[0]
        assert e.title == "Andrew Bailey: The economic outlook"
        assert e.link == "https://www.bis.org/review/r260210a.htm"
        assert "Bank of England" in e.description
        assert e.creator == "Andrew Bailey"
        assert e.pub_date == "2026-02-10"
        assert e.currency == "GBP"
        assert e.simple_title == "The economic outlook"
        assert e.occurrence_date == "2026-02-10"
        assert e.speaker_surname == "Bailey"
        assert e.pdf_url == "https://www.bis.org/review/r260210a.pdf"

    def test_empty_feed(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns="http://purl.org/rss/1.0/">
  <channel><title>Empty</title></channel>
</rdf:RDF>
"""
        entries = _parse_bis_feed(xml)
        assert entries == []

    def test_invalid_xml(self):
        entries = _parse_bis_feed("this is not xml at all")
        assert entries == []
