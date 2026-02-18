"""Example: Bloomberg data source connector.

Replace this with your actual Bloomberg (or other proprietary) data source.
"""

from __future__ import annotations

import os
from datetime import date

from aia_forecaster.models import SearchResult
from aia_forecaster.search.registry import data_source


@data_source("bloomberg")
async def fetch_bloomberg(
    pair: str, cutoff_date: date | None = None, **kwargs
) -> list[SearchResult]:
    """Fetch FX news from Bloomberg API.

    Requires BLOOMBERG_API_KEY environment variable.
    """
    api_key = os.environ.get("BLOOMBERG_API_KEY")
    if not api_key:
        return []

    # TODO: Replace with actual Bloomberg API calls
    # results = await call_bloomberg_api(pair, cutoff_date, api_key)
    # return [SearchResult(...) for r in results]

    return []
