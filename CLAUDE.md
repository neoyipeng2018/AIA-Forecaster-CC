# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project adapts the **AIA Forecaster** methodology for **FX (foreign exchange) forecasting**. The reference paper is `reference/AIA Forecaster.pdf` (Alur, Stadie et al., Bridgewater AIA Labs, arXiv:2511.07678, Nov 2025).

### Domain Focus: FX Price Movement Probabilities
The system forecasts **probabilities of price movement** for currency pairs across different:
- **Prices (strikes)**: probability that a pair moves above/below specific price levels (e.g., "P(EUR/USD > 1.10) = 0.35")
- **Tenors**: forecast horizons ranging from intraday to multi-month (e.g., 1D, 1W, 1M, 3M, 6M)
- Output is a **probability surface** over (price, tenor) — analogous to an options-implied probability distribution but derived from news-driven judgmental forecasting

The system combines four techniques adapted from the AIA Forecaster:
1. Agentic, adaptive search over FX-relevant news and macro sources
2. Multi-agent ensembling with a supervisor agent for reconciliation
3. Statistical calibration (Platt scaling) to counter LLM hedging bias
4. Foreknowledge bias detection and mitigation

## System Architecture

The pipeline processes a binary forecasting question `q` through these stages:

### Stage 1: Parallel Forecasting Agents (M agents, typically 10)
Each agent independently:
- Receives question `q`
- Performs **agentic, adaptive search**: iteratively queries a news/search API, conditioning each query on prior results (`q → E1 → E2 → ... → En → (R_i, p_i)`)
- Produces a reasoning trace `R_i` and probability estimate `p_i`

### Stage 2: Supervisor Agent (Reconciliation)
- Reads all M reasoning traces to identify disagreements
- Performs additional targeted search queries to resolve ambiguities (e.g., fact-checking base rates)
- Outputs a reconciled forecast with a confidence level (high/medium/low)
- High-confidence updates replace the simple mean; medium/low are discarded (falls back to mean)

### Stage 3: Statistical Calibration (Platt Scaling)
- Applies Platt scaling with coefficient `α = √3 ≈ 1.73` (from Neyman & Roughgarden, 2022)
- Corrects LLM hedging bias: pushes probabilities away from 0.5 toward 0 or 1
- Mathematically: `p_calibrated = sigmoid(α * log(p / (1 - p)))`
- Equivalent to `p^α / (p^α + (1-p)^α)`

## Key Technical Details

### Search
- Agentic search (LLM controls query strategy) dramatically outperforms non-agentic (fixed 3 queries)
- Search quality is the single most important factor for forecast accuracy
- Must enforce temporal information cutoffs to prevent foreknowledge bias
- Blacklist prediction market domains (Polymarket, Metaculus, Manifold, Kalshi) unless market prices are explicitly provided

#### FX News Ingestion via WorldMonitor Pattern
Agents should learn from the architecture of [worldmonitor](https://github.com/koala73/worldmonitor) for pulling the latest currency-relevant news. Key patterns to adopt:
- **RSS feed aggregation**: WorldMonitor ingests 100+ curated RSS feeds across geopolitics, defense, macro, and energy — we adapt this to FX-relevant feeds (central bank announcements, macro data releases, geopolitical risk, commodity/energy news)
- **Geo-keyword extraction**: WorldMonitor uses a 74-hub strategic location database to infer geography from headlines via keyword matching — we apply a similar approach to map headlines to affected currency pairs
- **Threat/signal classification pipeline**: Hybrid approach combining fast keyword matching (~120 patterns) with async LLM refinement (Groq Llama 3.1) — we adapt this for FX-impact classification (hawkish/dovish, risk-on/risk-off, trade flow disruption)
- **Temporal anomaly detection**: Welford's online algorithm learns per-region baselines; z-score thresholds flag deviations — applicable to detecting unusual news velocity around a currency
- **Multi-source signal fusion**: Convergence requirements (3+ distinct event types within geographic/temporal proximity) prevent false alerts — we use similar convergence logic before shifting FX probability estimates
- **Intelligence gap reporting**: WorldMonitor explicitly surfaces data source outages rather than silencing them — agents should flag when key news sources for a currency are unavailable
- **Redis caching with TTL**: Headline deduplication via hash-keyed cache (24h TTL) prevents re-processing — essential for high-frequency FX news monitoring

### Ensembling
- 10 independent forecasts per question is the standard (diminishing returns beyond ~15)
- Simple mean is a strong baseline; median and trimmed mean perform comparably
- The supervisor agent's value comes from resolving specific disagreements, not from holistic re-evaluation
- Naive LLM aggregation (asking an LLM to evaluate all forecasts and pick the best) performs **worse** than simple averaging -- LLMs overemphasize outlier opinions

### Calibration
- LLMs systematically hedge toward 0.5 due to RLHF training
- Platt scaling and log-odds extremization are mathematically equivalent (shown in Appendix G)
- Statistical correction only helps when initial forecasts are on the correct side of 0.5
- Prompting changes are largely ineffective at fixing hedging behavior; mathematical corrections are more robust

### Foreknowledge Bias
- Search APIs can leak future information via live data widgets, updated Wikipedia pages, and republished articles
- Implement an LLM-as-a-judge pipeline to flag search results that contain post-cutoff information
- ~1.65% of search results contain some form of foreknowledge bias

## Evaluation

### Metrics
- **Brier score**: `(1/n) * Σ(p_i - o_i)²` where `p` is forecast probability, `o ∈ {0,1}` is outcome
- Range: 0 (perfect) to 1; baseline of 0.25 for always predicting 0.5
- Strictly proper scoring rule (incentivizes truthful forecasting)

### Benchmarks
- **ForecastBench** (Karger et al., 2024): ~500-600 questions from prediction markets, dynamic benchmark
- **MarketLiquid**: 1610 questions from liquid prediction markets (politics, economics, AI/tech), harder than ForecastBench

### Target Performance (from paper)
- ForecastBench (FB-7-21): Brier score 0.1076 (vs superforecasters at 0.1110)
- MarketLiquid: Brier score 0.1258 (vs market consensus at 0.1106)

## Base Model Selection
The paper uses OpenAI o3 as the default base model. Claude Sonnet 4 showed the best raw performance but had foreknowledge bias issues due to a more recent training cutoff.
