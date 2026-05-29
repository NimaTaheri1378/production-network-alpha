# Trading the Production Network: Research Report

Generated: 2026-05-29T04:49:20Z

## Executive summary

This project builds a point-in-time production-network equity pipeline that asks whether machine-readable news shocks to one firm forecast abnormal returns of economically connected firms. The current release is a **research candidate, not a live production trading system**.

The selected research candidate is **lightgbm / 5d / twice_weekly_mon_thu / 10.0 bps one-way cost**. In the 2022--2024 out-of-sample backtest it produced:

- Annualized Sharpe: **0.724**
- Cumulative return: **28.85%**
- Max drawdown: **-11.80%**
- HAC t-stat: **1.280**
- Bootstrap probability of positive Sharpe: **95.0%**
- Decision status: **research_candidate_not_yet_production**

## Data and timing design

The private research pipeline uses WRDS-accessible supply-chain links, RavenPack Dow Jones entity-level news, CRSP stock returns, and CCM linking. Raw vendor records are not included in this repository. Public artifacts are aggregate-only, generated figures, code, docs, and synthetic examples.

The modelling sample uses walk-forward splits:

- Train: 2015--2019
- Validation: 2020--2021
- Test: 2022--2024

2025 was excluded from model-ready labels after forward-label coverage checks.

## Methods

The final model family retained in the backtest is LightGBM on engineered graph-spillover features. Earlier phases benchmarked raw spillover scores, ridge/local-projection-style linear models, and LightGBM across 1, 2, 5, 10, and 20 trading-day horizons.

## Top 10 bps net candidates

|   horizon_days | model    | rebalance_variant     |   ann_sharpe_naive |   cumulative_return |   max_drawdown |   tstat_hac_daily_mean |   avg_daily_turnover |   positive_sharpe_years |
|---------------:|:---------|:----------------------|-------------------:|--------------------:|---------------:|-----------------------:|---------------------:|------------------------:|
|              5 | lightgbm | twice_weekly_mon_thu  |          0.724169  |           0.288452  |      -0.118035 |               1.27971  |            0.135173  |                       3 |
|              5 | lightgbm | weekly_first          |          0.653009  |           0.271273  |      -0.163501 |               1.1314   |            0.123926  |                       2 |
|             10 | lightgbm | twice_weekly_mon_thu  |          0.583389  |           0.197434  |      -0.197521 |               1.00386  |            0.0794357 |                       2 |
|             10 | lightgbm | daily                 |          0.430714  |           0.131646  |      -0.217915 |               0.741858 |            0.0698051 |                       2 |
|              5 | lightgbm | daily                 |          0.412065  |           0.127952  |      -0.131838 |               0.73401  |            0.128605  |                       2 |
|              5 | lightgbm | every_5th_trading_day |          0.229396  |           0.0689637 |      -0.165021 |               0.404667 |            0.115972  |                       2 |
|             10 | lightgbm | every_5th_trading_day |          0.170415  |           0.0406158 |      -0.285413 |               0.296341 |            0.0656671 |                       2 |
|             10 | lightgbm | weekly_first          |          0.0936998 |           0.0124579 |      -0.21914  |               0.168654 |            0.073069  |                       2 |
|             10 | lightgbm | weekly_last           |         -0.0811497 |          -0.0513157 |      -0.261003 |              -0.135259 |            0.0726286 |                       2 |
|              5 | lightgbm | weekly_last           |         -0.233762  |          -0.11157   |      -0.268032 |              -0.385268 |            0.12089   |                       2 |

## Bootstrap robustness

|   horizon_days | model    | rebalance_variant     |   ann_sharpe |   boot_sharpe_p05 |   boot_sharpe_p50 |   boot_sharpe_p95 |   prob_sharpe_positive |
|---------------:|:---------|:----------------------|-------------:|------------------:|------------------:|------------------:|-----------------------:|
|              5 | lightgbm | twice_weekly_mon_thu  |    0.724169  |         0.0120244 |         0.871353  |          1.75013  |                  0.95  |
|              5 | lightgbm | weekly_first          |    0.653009  |        -0.231107  |         0.755644  |          1.70159  |                  0.92  |
|             10 | lightgbm | twice_weekly_mon_thu  |    0.583389  |        -0.367967  |         0.651397  |          1.58479  |                  0.87  |
|             10 | lightgbm | daily                 |    0.430714  |        -0.50759   |         0.46237   |          1.36749  |                  0.789 |
|              5 | lightgbm | daily                 |    0.412065  |        -0.362173  |         0.545539  |          1.44438  |                  0.847 |
|              5 | lightgbm | every_5th_trading_day |    0.229396  |        -0.612069  |         0.267685  |          1.20927  |                  0.692 |
|             10 | lightgbm | every_5th_trading_day |    0.170415  |        -0.730186  |         0.153621  |          1.0808   |                  0.614 |
|             10 | lightgbm | weekly_first          |    0.0936998 |        -0.785108  |         0.135523  |          0.952577 |                  0.601 |
|             10 | lightgbm | weekly_last           |   -0.0811497 |        -0.992005  |        -0.0771184 |          0.943703 |                  0.454 |
|              5 | lightgbm | weekly_last           |   -0.233762  |        -1.19161   |        -0.114998  |          0.838273 |                  0.423 |

## Validation diagnostics

|   horizon_days | model    | rebalance_variant     |   daily_return_rows |   signal_dates_skipped |   avg_daily_positions |   avg_daily_turnover |   max_abs_gross_error |
|---------------:|:---------|:----------------------|--------------------:|-----------------------:|----------------------:|---------------------:|----------------------:|
|              5 | lightgbm | daily                 |                 752 |                      0 |              147.274  |            0.128605  |           4.44089e-16 |
|              5 | lightgbm | twice_weekly_mon_thu  |                 752 |                      0 |              107.969  |            0.135173  |           4.44089e-16 |
|              5 | lightgbm | weekly_first          |                 752 |                      0 |               86.1223 |            0.123926  |           5.55112e-16 |
|              5 | lightgbm | weekly_last           |                 748 |                      0 |               69.0013 |            0.12089   |           6.66134e-16 |
|              5 | lightgbm | every_5th_trading_day |                 750 |                      0 |               69.9933 |            0.115972  |           4.44089e-16 |
|             10 | lightgbm | daily                 |                 752 |                      0 |              186.963  |            0.0698051 |           4.44089e-16 |
|             10 | lightgbm | twice_weekly_mon_thu  |                 752 |                      0 |              146.065  |            0.0794357 |           4.44089e-16 |
|             10 | lightgbm | weekly_first          |                 752 |                      0 |              123.746  |            0.073069  |           4.44089e-16 |
|             10 | lightgbm | weekly_last           |                 748 |                      0 |              102.723  |            0.0726286 |           4.44089e-16 |
|             10 | lightgbm | every_5th_trading_day |                 750 |                      0 |              103.267  |            0.0656671 |           4.44089e-16 |

## Figures

![phase7_0_bootstrap_sharpe_ci.png](figures/phase7_0_bootstrap_sharpe_ci.png)
![phase7_0_cumulative_top_candidates_10bps.png](figures/phase7_0_cumulative_top_candidates_10bps.png)
![phase7_0_phase6_3_10bps_net_sharpe_ranking.png](figures/phase7_0_phase6_3_10bps_net_sharpe_ranking.png)
![phase7_0_top_lgbm_features.png](figures/phase7_0_top_lgbm_features.png)
![phase7_0_yearly_sharpe_top_candidates.png](figures/phase7_0_yearly_sharpe_top_candidates.png)

## Decision

Use this as a research/README headline and interview artifact. Do not call it production-ready until placebo, sector-neutrality, capacity, slippage, and post-cost robustness checks are expanded.
