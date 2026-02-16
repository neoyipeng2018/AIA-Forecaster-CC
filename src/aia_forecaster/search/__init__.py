from .web import search_web
from .rss import fetch_fx_news  # Also triggers RSS data source registration
from .bis import fetch_bis_speeches  # Also triggers BIS data source registration
from .registry import data_source, register, list_sources, fetch_all

__all__ = [
    "search_web",
    "fetch_fx_news",
    "fetch_bis_speeches",
    # Data source registry
    "data_source",
    "register",
    "list_sources",
    "fetch_all",
]
