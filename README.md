# AIA Forecaster

FX probability surface forecaster powered by LLM ensembles. Generates P(price > strike) across a grid of strikes and tenors for currency pairs, using agentic search, multi-agent ensembling, and statistical calibration.

Based on the [AIA Forecaster paper](https://arxiv.org/abs/2511.07678) (Alur, Stadie et al., Bridgewater AIA Labs, 2025).

## Installation

```bash
poetry install
poetry shell
```

Set your LLM API key in `.env`:

```
OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

### Generate a probability surface

```bash
# Default: USDJPY, 5 strikes, 5 tenors, today's cutoff
forecast USDJPY

# With a specific cutoff date
forecast USDJPY 2026-02-15

# Fewer strikes for a faster run
forecast USDJPY --strikes 3

# Specific tenors only
forecast surface --tenors 1W,1M

# Show per-cell evidence and reasoning
forecast USDJPY -e

# Verbose mode (full agent traces)
forecast USDJPY -v
```

### Forecast a single question

```bash
forecast question "Will USD/JPY be above 155 in 1 week?"
forecast question "Will EUR/USD be above 1.10 in 3 months?" --pair EURUSD
```

### Evaluate a past forecast

```bash
forecast list                    # see recent runs
forecast evaluate <RUN_ID> 1     # outcome: 1 = yes, 0 = no
```

## Usage from Python

```python
import asyncio
from aia_forecaster.fx.surface import ProbabilitySurfaceGenerator
from aia_forecaster.models import Tenor

async def main():
    gen = ProbabilitySurfaceGenerator(num_agents=3)
    surface = await gen.generate(
        pair="USDJPY",
        num_strikes=3,
        tenors=[Tenor.W1, Tenor.M1],
    )
    for cell in surface.cells:
        p = cell.calibrated.calibrated_probability if cell.calibrated else None
        print(f"  {cell.strike} / {cell.tenor.value}: {p}")
    return surface

surface = asyncio.run(main())
```

### Single question from Python

```python
import asyncio
from aia_forecaster.ensemble.engine import EnsembleEngine
from aia_forecaster.calibration.platt import calibrate
from aia_forecaster.models import ForecastQuestion

async def main():
    engine = EnsembleEngine(num_agents=5)
    question = ForecastQuestion(text="Will USD/JPY be above 155 in 1 week?")
    result = await engine.run(question)
    cal = calibrate(result.final_probability)
    print(f"Calibrated: {cal.calibrated_probability:.4f}")

asyncio.run(main())
```

## Architecture

### Two-Phase Surface Generation

The surface generator uses a shared-research approach to avoid redundant search across cells (~94% fewer LLM calls vs naive per-cell ensembles):

```
Phase 1: Shared Research (M agents, full agentic search)
  Each agent researches the broad pair outlook (not a specific cell)
  Same agentic search loop (up to 5 iterations per agent)
  Output: M ResearchBriefs with unique evidence + macro reasoning

Phase 2: Batch Pricing (M agents x T tenors, one LLM call each)
  Each agent prices ALL strikes for a tenor using its own evidence
  10 agents x 5 tenors = 50 LLM calls (vs ~3,000 with per-cell ensembles)

Phase 3: Aggregation & Calibration
  Per-cell mean of M agent probabilities
  Surface-level supervisor review (anomaly detection, targeted search)
  Platt scaling (alpha = sqrt(3), corrects LLM hedging bias)
  PAVA monotonicity enforcement
```

### Single-Question Pipeline

```
M agents in parallel -> agentic search -> probability estimate
  -> supervisor reconciliation -> Platt calibration
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Forecasting Agent | `agents/forecaster.py` | Agentic search + probability estimation |
| Supervisor Agent | `agents/supervisor.py` | Disagreement resolution, surface review |
| Ensemble Engine | `ensemble/engine.py` | Parallel agent orchestration |
| Surface Generator | `fx/surface.py` | Two-phase surface pipeline |
| Platt Calibration | `calibration/platt.py` | LLM hedging bias correction |
| Monotonicity (PAVA) | `calibration/monotonicity.py` | Strike-monotonicity enforcement |
| Base Rates | `fx/base_rates.py` | Statistical anchoring from vol data |
| Data Source Registry | `search/registry.py` | Pluggable data source framework |

## Adding Custom Data Sources

You can plug any dataset into the forecasting pipeline by writing a single decorated async function. Agents will automatically fetch from it alongside the built-in RSS feeds.

### Minimal example

```python
# my_sources.py  (import this file at startup so the decorator runs)
from aia_forecaster.search.registry import data_source
from aia_forecaster.models import SearchResult

@data_source("my_csv")
async def fetch_csv_data(pair: str, cutoff_date, **kwargs) -> list[SearchResult]:
    """Load headlines from a local CSV file."""
    import csv
    from pathlib import Path

    results = []
    for row in csv.DictReader(open(Path("data") / f"{pair}.csv")):
        results.append(SearchResult(
            query=f"csv:{pair}",
            title=row["headline"],
            snippet=row["body"][:500],
            url=row.get("url", ""),
            source="my_csv",
        ))
    return results[: kwargs.get("max_results", 20)]
```

### Requirements

- The function **must** be `async` and return `list[SearchResult]`.
- Required parameters: `pair` (`str`, e.g. `"USDJPY"`) and `cutoff_date` (`datetime.date`).
- Optional `**kwargs` receives `max_results`, `max_age_hours`, etc. from the agent.
- All registered sources run **in parallel** with error isolation — one failing source won't break the others.

### More examples

```python
# API data source
@data_source("bloomberg_api")
async def fetch_bloomberg(pair: str, cutoff_date, **kwargs):
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/news/{pair}")
        return [
            SearchResult(query=f"bloomberg:{pair}", title=item["title"],
                         snippet=item["summary"], url=item["link"], source="bloomberg")
            for item in resp.json()["articles"]
        ]

# Database source
@data_source("internal_db")
async def fetch_from_db(pair: str, cutoff_date, **kwargs):
    import aiosqlite
    async with aiosqlite.connect("data/research.db") as db:
        rows = await db.execute_fetchall(
            "SELECT title, body, url FROM articles WHERE pair = ? AND date <= ?",
            (pair, cutoff_date.isoformat()),
        )
        return [
            SearchResult(query=f"db:{pair}", title=r[0], snippet=r[1], url=r[2], source="internal_db")
            for r in rows
        ]
```

### Imperative registration

If you prefer not to use the decorator:

```python
from aia_forecaster.search.registry import register

async def my_source(pair, cutoff_date, **kwargs):
    ...

register("my_source", my_source)
```

### Inspecting registered sources

```python
from aia_forecaster.search.registry import list_sources
print(list_sources())  # ['rss', 'my_csv', 'bloomberg_api', ...]
```

## Configuration

Settings are loaded from environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `anthropic/claude-sonnet-4-5-20250929` | LLM model (litellm format) |
| `NUM_AGENTS` | `10` | Number of forecasting agents |
| `MAX_SEARCH_ITERATIONS` | `5` | Max search iterations per agent |
| `PLATT_ALPHA` | `sqrt(3)` | Calibration coefficient |
| `DEFAULT_PAIR` | `USDJPY` | Default currency pair |

Override at the CLI:

```bash
forecast USDJPY --agents 5 --model openai/gpt-4o
```

## Output

Each surface run produces:
- **Console table** with color-coded probabilities
- **Heatmap PNG** saved to `data/forecasts/PAIR_DATE.png`
- **JSON file** with full surface data (probabilities, evidence, reasoning)

## Supported Pairs

- `USDJPY` (USD/JPY)
- `EURUSD` (EUR/USD)
- `GBPUSD` (GBP/USD)
