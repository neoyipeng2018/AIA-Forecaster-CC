"""Tests for relevance scoring and filtering."""

import pytest

from aia_forecaster.models import SearchResult
from aia_forecaster.search.relevance import (
    filter_relevant,
    score_relevance,
)


def _make_result(title: str, snippet: str = "", source: str = "", url: str = "") -> SearchResult:
    return SearchResult(
        query="test",
        title=title,
        snippet=snippet,
        url=url,
        source=source,
    )


# ---------------------------------------------------------------------------
# Scoring tests — USDJPY
# ---------------------------------------------------------------------------


class TestScoreRelevanceUSDJPY:
    """Test scoring rubric with real-world examples from USDJPY forecasts."""

    def test_direct_pair_in_title_scores_high(self):
        r = _make_result("USD/JPY bounces back above 152.00 on risk recovery")
        s = score_relevance(r, "USDJPY")
        assert s >= 0.40, f"Direct pair in title should score >=0.40, got {s}"

    def test_usdjpy_no_slash_in_title(self):
        r = _make_result("USDJPY forecast: Yen weakens after BOJ holds")
        s = score_relevance(r, "USDJPY")
        assert s >= 0.40

    def test_pair_in_snippet_only(self):
        r = _make_result(
            "Yen pulls back on risk appetite",
            snippet="The USD/JPY pair rose 0.3% to 152.50 in Asian trading.",
        )
        s = score_relevance(r, "USDJPY")
        assert s >= 0.25

    def test_both_currency_keywords(self):
        r = _make_result("FOMC Minutes show hawkish tilt as BOJ signals patience")
        s = score_relevance(r, "USDJPY")
        # "fomc"/"fed"→USD, "boj"→JPY, plus "hawkish" general keyword
        assert s >= 0.25

    def test_single_currency_keyword(self):
        r = _make_result("Fed holds rates steady in January meeting")
        s = score_relevance(r, "USDJPY")
        # Only USD keywords, no JPY
        assert 0.15 <= s < 0.40

    def test_japan_macro_data_relevant(self):
        r = _make_result("Japan Machinery Orders rise 2.7% in December")
        s = score_relevance(r, "USDJPY")
        # "japan" → JPY keyword
        assert s >= 0.10

    def test_fomc_minutes_relevant(self):
        r = _make_result(
            "FOMC Minutes February 2026",
            snippet="Federal Reserve officials discussed rate path amid inflation concerns. "
                    "The dollar strengthened on hawkish forward guidance about interest rate policy.",
        )
        s = score_relevance(r, "USDJPY")
        assert s >= 0.20

    # --- Articles that SHOULD be filtered out ---

    def test_philippines_gold_filtered(self):
        r = _make_result("Philippines Gold price today: Gold rises, according to FXStreet data")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20, f"Philippines Gold should score <0.20, got {s}"

    def test_saudi_gold_filtered(self):
        r = _make_result("Saudi Arabia Gold price today: Gold falls, according to FXStreet data")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20, f"Saudi Gold should score <0.20, got {s}"

    def test_eurusd_article_filtered(self):
        r = _make_result("EUR/USD slumps below 1.0800 on strong US data")
        s = score_relevance(r, "USDJPY")
        # Has USD keywords but title is about EUR/USD → penalty
        assert s < 0.20, f"EUR/USD article should score <0.20 for USDJPY, got {s}"

    def test_audnzd_filtered(self):
        r = _make_result("AUD/NZD surges past 1.1100 on RBA hawkishness")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20, f"AUD/NZD article should score <0.20 for USDJPY, got {s}"

    def test_gbpusd_filtered(self):
        r = _make_result("GBP/USD analysis: Pound strengthens on BOE rate hold")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20

    def test_bitcoin_filtered(self):
        r = _make_result("Bitcoin surges past $100,000 as crypto rally continues")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20

    def test_generic_gold_price_filtered(self):
        r = _make_result("Gold price forecast: XAU/USD targets $2100")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20

    def test_sp500_filtered(self):
        r = _make_result("S&P 500 hits record high amid tech rally")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20


class TestScoreRelevanceCommodityCurrencies:
    """Commodity mentions should NOT be penalized for commodity-linked currencies."""

    def test_gold_not_penalized_for_audusd(self):
        r = _make_result("Gold price rises, lifting Australian dollar outlook")
        s = score_relevance(r, "AUDUSD")
        # Gold is relevant for AUD — no penalty, and "australia" keyword hits
        assert s >= 0.10

    def test_oil_not_penalized_for_cadusd(self):
        r = _make_result("Oil price surge boosts Canadian dollar")
        s = score_relevance(r, "USDCAD")
        assert s >= 0.10

    def test_oil_penalized_for_usdjpy(self):
        r = _make_result("Oil price today: WTI crude falls below $70")
        s = score_relevance(r, "USDJPY")
        assert s < 0.20


class TestCentralBankBonus:
    def test_fed_source_bonus_for_usdjpy(self):
        r = _make_result(
            "Press Release",
            snippet="Federal Reserve announces policy decision",
            source="https://www.federalreserve.gov/feeds/press_all.xml",
        )
        s = score_relevance(r, "USDJPY")
        # Central bank bonus + currency keyword
        assert s >= 0.20

    def test_boj_source_bonus(self):
        r = _make_result(
            "Monetary Policy Statement",
            snippet="Bank of Japan maintains ultra-loose policy",
            source="https://www.boj.or.jp/en/rss/whatsnew.xml",
        )
        s = score_relevance(r, "USDJPY")
        assert s >= 0.20


class TestOtherPairPenalty:
    def test_no_penalty_when_target_pair_also_in_title(self):
        r = _make_result("USD/JPY and EUR/USD both react to Fed decision")
        s = score_relevance(r, "USDJPY")
        # Target pair is in title, so no other-pair penalty
        assert s >= 0.40

    def test_penalty_when_only_other_pair_in_title(self):
        r = _make_result("EUR/USD crashes 200 pips after ECB surprise cut")
        s_usdjpy = score_relevance(r, "USDJPY")
        s_eurusd = score_relevance(r, "EURUSD")
        assert s_eurusd > s_usdjpy


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestFilterRelevant:
    def test_filters_below_threshold(self):
        results = [
            _make_result("USD/JPY bounces back above 152"),
            _make_result("Philippines Gold price today"),
            _make_result("FOMC Minutes show hawkish stance on interest rate policy and inflation"),
            _make_result("AUD/NZD surges past 1.1100"),
        ]
        kept = filter_relevant(results, "USDJPY", threshold=0.20)
        titles = [r.title for r in kept]
        assert "USD/JPY bounces back above 152" in titles
        assert "FOMC Minutes show hawkish stance on interest rate policy and inflation" in titles
        assert "Philippines Gold price today" not in titles
        assert "AUD/NZD surges past 1.1100" not in titles

    def test_relevance_scores_attached(self):
        results = [
            _make_result("USD/JPY at 152.00"),
            _make_result("Random article"),
        ]
        kept = filter_relevant(results, "USDJPY", threshold=0.0)
        for r in kept:
            assert r.relevance_score is not None

    def test_empty_input(self):
        assert filter_relevant([], "USDJPY") == []

    def test_threshold_zero_keeps_all(self):
        results = [_make_result("Completely irrelevant article about cooking")]
        kept = filter_relevant(results, "USDJPY", threshold=0.0)
        assert len(kept) == 1

    def test_threshold_one_filters_almost_all(self):
        results = [
            _make_result("USD/JPY bounces back above 152"),
            _make_result("Yen weakens"),
        ]
        kept = filter_relevant(results, "USDJPY", threshold=1.0)
        assert len(kept) == 0


class TestEdgeCases:
    def test_empty_title_and_snippet(self):
        r = _make_result("", snippet="")
        s = score_relevance(r, "USDJPY")
        assert s == 0.0

    def test_score_clamped_to_range(self):
        # Even with maximum positive signals, score should not exceed 1.0
        r = _make_result(
            "USD/JPY forecast: BOJ and Fed rate decision hawkish dovish monetary policy",
            snippet="USD/JPY USDJPY dollar yen japan fed fomc treasury rate cut inflation gdp pmi",
            source="https://www.federalreserve.gov/feeds/press_all.xml",
        )
        s = score_relevance(r, "USDJPY")
        assert 0.0 <= s <= 1.0

    def test_score_clamped_above_zero(self):
        # Multiple penalties should not push below 0
        r = _make_result(
            "EUR/GBP gold price bitcoin S&P 500",
            snippet="",
        )
        s = score_relevance(r, "USDJPY")
        assert s >= 0.0
