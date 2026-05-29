# Trading the Production Network

**Research question:** Do machine-readable news shocks to one firm forecast abnormal returns of economically connected firms through point-in-time production-network links?

This repository contains a research-grade, leakage-aware production-network equity pipeline. It builds point-in-time supplier--customer graphs, maps RavenPack Dow Jones firm news into market-usable shock dates, constructs CRSP abnormal-return labels, compares transparent baselines with LightGBM, and evaluates long-short portfolios net of turnover costs.

## Headline result

Status: **research candidate; not a deployed/live trading system**.

Selected research candidate:

```text
Model:                  lightgbm
Forecast horizon:       5 trading days
Rebalance rule:         twice_weekly_mon_thu
One-way cost:           10.0 bps
Out-of-sample period:   2022--2024
Annualized Sharpe:      0.724
Cumulative return:      28.85%
Max drawdown:           -11.80%
HAC t-stat:             1.280
```

This is a positive research result intended to demonstrate a full-stack quantitative research process. It is not presented as a deployed trading system; the next credibility checks are placebo edges, sector/beta neutrality, capacity, slippage, and paper-trading.

## Public contents

- `docs/research_report.md` — paper-style report
- `docs/interviewer_briefing.md` — two-page project pitch
- `docs/figures/` — generated public figures
- `artifacts/release_public/` — aggregate public CSV/JSON summaries
- `examples/synthetic_demo/` — synthetic data smoke test with no vendor records
- `src/production_network_alpha/` — reusable package code

## Data policy

Raw WRDS/vendor data are **not** committed. Protected local files under `data/raw`, `data/interim`, `data/processed`, and heavy model/backtest Parquet outputs are ignored by Git. Public artifacts are aggregate-only or synthetic.

## Quick public demo

```bash
python examples/synthetic_demo/run_synthetic_demo.py
```

## Local dashboard

```bash
python -m production_network_alpha.dashboard.public_results_app
```

## Reproduction overview

The private full pipeline was run phase-by-phase:

1. WRDS schema discovery
2. point-in-time supply-chain graph construction
3. RavenPack news shock extraction
4. CRSP return labels and model matrix
5. LightGBM modeling
6. portfolio/backtest and turnover refinement
7. robustness and production decision

The public repository includes scripts and aggregate artifacts, but not the protected vendor-derived Parquet files required to rerun private extraction.

## Public-release note

This repository is a recruiter-facing, public-safe release. It includes code, documentation, synthetic demo data, aggregate result tables, and generated figures. It intentionally excludes raw WRDS/vendor data, protected Parquet caches, credentials, and local cluster logs. The reported strategy is a research candidate, not a live trading system.
