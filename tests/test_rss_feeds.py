"""Tests for RSS feed configuration and filtering."""

from aia_forecaster.search.rss import (
    CURRENCY_KEYWORDS,
    FX_FEEDS,
    FX_RSS_FEEDS,
    GENERAL_FX_KEYWORDS,
    FeedConfig,
    _feeds_for_pair,
    _headline_matches,
    _pair_keywords,
)


VALID_CATEGORIES = {
    "central_bank", "fx_specific", "macro_data",
    "commodity", "geopolitical", "general",
}


class TestFeedConfig:
    def test_all_feeds_have_valid_category(self):
        for f in FX_FEEDS:
            assert f.category in VALID_CATEGORIES, f"Feed {f.url} has invalid category '{f.category}'"

    def test_all_currencies_recognized(self):
        known = set(CURRENCY_KEYWORDS.keys())
        for f in FX_FEEDS:
            for c in f.currencies:
                assert c in known, f"Feed {f.url} references unknown currency '{c}'"

    def test_backward_compat_flat_list(self):
        assert len(FX_RSS_FEEDS) == len(FX_FEEDS)
        assert all(isinstance(u, str) for u in FX_RSS_FEEDS)

    def test_no_duplicate_urls(self):
        urls = [f.url for f in FX_FEEDS]
        assert len(urls) == len(set(urls)), "Duplicate feed URLs found"

    def test_minimum_feed_count(self):
        assert len(FX_FEEDS) >= 20, f"Expected at least 20 feeds, got {len(FX_FEEDS)}"


class TestFeedsForPair:
    def test_usdjpy_includes_fed_and_boj(self):
        feeds = _feeds_for_pair("USDJPY")
        urls = [f.url for f in feeds]
        assert any("federalreserve" in u for u in urls), "Missing Fed feed for USD"
        assert any("boj" in u for u in urls), "Missing BOJ feed for JPY"

    def test_usdjpy_includes_universal_fx_feeds(self):
        feeds = _feeds_for_pair("USDJPY")
        urls = [f.url for f in feeds]
        assert any("fxstreet" in u for u in urls), "Missing FXStreet (universal)"
        assert any("reuters" in u for u in urls), "Missing Reuters (universal)"

    def test_usdjpy_excludes_rba(self):
        feeds = _feeds_for_pair("USDJPY")
        urls = [f.url for f in feeds]
        assert not any("rba.gov.au" in u for u in urls), "RBA feed should not appear for USDJPY"

    def test_audusd_includes_rba_and_fed(self):
        feeds = _feeds_for_pair("AUDUSD")
        urls = [f.url for f in feeds]
        assert any("rba.gov.au" in u for u in urls), "Missing RBA feed for AUD"
        assert any("federalreserve" in u for u in urls), "Missing Fed feed for USD"

    def test_audusd_includes_commodity(self):
        feeds = _feeds_for_pair("AUDUSD")
        urls = [f.url for f in feeds]
        assert any("oilprice" in u for u in urls), "Missing commodity feed for AUD"

    def test_eurusd_includes_ecb_and_fed(self):
        feeds = _feeds_for_pair("EURUSD")
        urls = [f.url for f in feeds]
        assert any("ecb" in u for u in urls), "Missing ECB feed for EUR"
        assert any("federalreserve" in u for u in urls), "Missing Fed feed for USD"

    def test_gbpusd_includes_boe(self):
        feeds = _feeds_for_pair("GBPUSD")
        urls = [f.url for f in feeds]
        assert any("bankofengland" in u for u in urls), "Missing BOE feed for GBP"


class TestExpandedKeywords:
    def test_usd_macro_keywords(self):
        keywords = CURRENCY_KEYWORDS["USD"]
        assert "nonfarm" in keywords
        assert "cpi" in keywords

    def test_eur_regional_keywords(self):
        keywords = CURRENCY_KEYWORDS["EUR"]
        assert "eurostat" in keywords

    def test_aud_commodity_keywords(self):
        keywords = CURRENCY_KEYWORDS["AUD"]
        assert "iron ore" in keywords

    def test_cad_oil_keywords(self):
        keywords = CURRENCY_KEYWORDS["CAD"]
        assert "crude oil" in keywords

    def test_nok_exists(self):
        assert "NOK" in CURRENCY_KEYWORDS
        assert "norges bank" in CURRENCY_KEYWORDS["NOK"]

    def test_sek_exists(self):
        assert "SEK" in CURRENCY_KEYWORDS
        assert "riksbank" in CURRENCY_KEYWORDS["SEK"]

    def test_general_keywords_expanded(self):
        assert "rate decision" in GENERAL_FX_KEYWORDS
        assert "tariff" in GENERAL_FX_KEYWORDS
        assert "pmi" in GENERAL_FX_KEYWORDS
        assert "bond yield" in GENERAL_FX_KEYWORDS

    def test_headline_matches_new_keywords(self):
        keywords = _pair_keywords("USDJPY")
        assert _headline_matches("US CPI data surprises to the upside", keywords)
        assert _headline_matches("Nonfarm payrolls beat expectations", keywords)
        assert _headline_matches("BOJ rate decision delayed", keywords)
