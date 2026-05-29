# Methodology

The project combines point-in-time production-network graph construction, entity-level machine-readable news shocks, forward CRSP abnormal-return labels, walk-forward model validation, and net-of-cost portfolio evaluation.

Key anti-leakage controls:

- supply-chain links are lagged to market-known dates;
- news signals are shifted to market-usable signal dates;
- model labels are forward trading-day abnormal returns;
- 2025 is excluded from model-ready data because forward-label coverage was insufficient;
- final backtests use only 2022--2024 out-of-sample predictions.
