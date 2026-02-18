"""AIA Forecaster — FX probability forecasting using agentic search and ensemble LLMs."""

__version__ = "0.1.0"


def _load_extensions() -> None:
    """Try to import the ``company`` package to register custom extensions.

    If the ``company/`` directory exists (i.e., we're in a company fork),
    its ``__init__.py`` runs and registers all custom pairs, feeds,
    connectors, etc. via the public registration APIs.

    If it doesn't exist (upstream), this silently does nothing.
    """
    try:
        import company  # noqa: F401
    except ImportError:
        pass  # No company package — running upstream


_load_extensions()
