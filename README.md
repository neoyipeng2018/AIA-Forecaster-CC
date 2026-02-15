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
