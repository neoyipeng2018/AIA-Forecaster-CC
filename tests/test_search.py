"""Tests for search modules."""

import json
import tempfile
import time
from pathlib import Path

from aia_forecaster.search.web import BLACKLISTED_DOMAINS, _is_blacklisted
from aia_forecaster.search.rss import _headline_matches, _pair_keywords
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
