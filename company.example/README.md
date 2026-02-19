# Company Extensions

This directory is a template for adding company-specific extensions to AIA Forecaster.

## Quick Start

```bash
# 1. Copy this template
cp -r company.example company

# 2. Remove company/ from .gitignore (so YOUR fork tracks it)
#    Edit .gitignore and delete the "company/" line

# 3. Customize the files in company/ for your needs

# 4. Everything auto-loads — no upstream files need modification
```

## Structure

```
company/
├── __init__.py          # Entry point — registers all extensions on import
├── search/
│   ├── __init__.py      # Auto-discovered by the data source registry
│   └── bloomberg.py     # Example: Bloomberg data source connector
├── pairs.py             # Register custom currency pairs (exotics, NDFs)
├── llm.py               # Register custom LLM backend (Azure, Anthropic, Ollama)
└── config.py            # Company-specific configuration overrides
```

## How It Works

When the `aia_forecaster` package initializes, it tries to `import company`.
If the `company/` directory exists (with an `__init__.py`), all registrations
in `company/__init__.py` run automatically:

- **Custom pairs** via `register_pair()` from `aia_forecaster.fx.pairs`
- **Custom RSS feeds** via `register_feed()` from `aia_forecaster.search.rss`
- **Custom keywords** via `register_currency_keywords()` from `aia_forecaster.search.rss`
- **Custom data sources** via `@data_source()` from `aia_forecaster.search.registry`
- **Custom LLM backend** via `set_llm_provider()` from `aia_forecaster.llm`
- **Blacklisted domains** via `add_blacklisted_domains()` from `aia_forecaster.search.web`

## Pulling Upstream Changes

Since all company code lives in `company/` (which upstream `.gitignore` ignores),
you can pull upstream changes with zero merge conflicts:

```bash
git fetch upstream
git merge upstream/main   # clean merge every time
```
