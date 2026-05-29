# Interview Briefing: Trading the Production Network

## One-sentence pitch

I built a point-in-time supply-chain news-shock engine that tests whether firm-specific news propagates through real economic links into delayed connected-equity returns.

## What is technically hard

1. **Point-in-time graph construction:** supplier--customer links are only used after disclosure/known dates, with CRSP/CCM mappings checked through time.
2. **Timestamp discipline:** RavenPack events are converted into market-usable signal dates, excluding own-news when measuring pure spillovers.
3. **Leakage-aware validation:** labels are forward trading-day abnormal returns with train/validation/test splits locked before model selection.
4. **Economic backtesting:** forecast skill is separated from implementability through turnover, cost, drawdown, and HAC/bootstrapped robustness.

## Current headline result

The best research candidate is **lightgbm / 5d / twice_weekly_mon_thu** with **28.85%** cumulative return, **0.72** Sharpe, and **-11.80%** max drawdown at **10.0 bps** one-way cost over 2022--2024.

## Honest limitation

The HAC t-stat is **1.28**, so the project is a strong research artifact but not yet a live deployment. Next credibility work: placebo edges, sector/beta neutrality, execution capacity, and forward paper-trading.
