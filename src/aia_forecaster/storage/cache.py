"""File-based search result cache with TTL expiry."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from aia_forecaster.config import settings

logger = logging.getLogger(__name__)


class SearchCache:
    """File-based cache storing search results as JSON with TTL."""

    def __init__(self, cache_dir: Path | None = None, ttl_hours: int | None = None):
        self.cache_dir = cache_dir or settings.cache_dir
        self.ttl_seconds = (ttl_hours or settings.cache_ttl_hours) * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hashed}.json"

    def get(self, key: str) -> list[dict] | None:
        """Return cached data if it exists and is not expired."""
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            if time.time() - data["ts"] > self.ttl_seconds:
                path.unlink(missing_ok=True)
                return None
            return data["results"]
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, results: list[dict]) -> None:
        """Store results with current timestamp."""
        path = self._key_path(key)
        path.write_text(json.dumps({"ts": time.time(), "results": results}))

    def clear(self) -> int:
        """Remove all cached files. Returns count of removed files."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
