# Phase 7.0 robustness and production decision

Generated at UTC: 2026-05-29T03:28:49Z

## Decision

- Status: **research_candidate_not_yet_production**
- Selected candidate: **lightgbm / 5d / twice_weekly_mon_thu / 10.0 bps**
- Annualized Sharpe: **0.724**
- Cumulative return: **28.85%**
- Max drawdown: **-11.80%**
- HAC t-stat: **1.280**
- Positive Sharpe years: **3**

## Interpretation

Use as research/README headline, not live production, until bootstrap/placebo/sector-neutral and capacity diagnostics pass.

## Top 10 bps net candidates

|   horizon_days | model    | rebalance_variant     |   ann_sharpe_naive |   cumulative_return |   max_drawdown |   tstat_hac_daily_mean |   avg_daily_turnover |   positive_sharpe_years |   min_year_sharpe |
|---------------:|:---------|:----------------------|-------------------:|--------------------:|---------------:|-----------------------:|---------------------:|------------------------:|------------------:|
|              5 | lightgbm | twice_weekly_mon_thu  |          0.724169  |           0.288452  |      -0.118035 |               1.27971  |            0.135173  |                       3 |         0.0832673 |
|              5 | lightgbm | weekly_first          |          0.653009  |           0.271273  |      -0.163501 |               1.1314   |            0.123926  |                       2 |        -0.12701   |
|             10 | lightgbm | twice_weekly_mon_thu  |          0.583389  |           0.197434  |      -0.197521 |               1.00386  |            0.0794357 |                       2 |        -0.91126   |
|             10 | lightgbm | daily                 |          0.430714  |           0.131646  |      -0.217915 |               0.741858 |            0.0698051 |                       2 |        -1.06093   |
|              5 | lightgbm | daily                 |          0.412065  |           0.127952  |      -0.131838 |               0.73401  |            0.128605  |                       2 |        -0.10675   |
|              5 | lightgbm | every_5th_trading_day |          0.229396  |           0.0689637 |      -0.165021 |               0.404667 |            0.115972  |                       2 |        -0.203505  |
|             10 | lightgbm | every_5th_trading_day |          0.170415  |           0.0406158 |      -0.285413 |               0.296341 |            0.0656671 |                       2 |        -1.39736   |
|             10 | lightgbm | weekly_first          |          0.0936998 |           0.0124579 |      -0.21914  |               0.168654 |            0.073069  |                       2 |        -1.22501   |
|             10 | lightgbm | weekly_last           |         -0.0811497 |          -0.0513157 |      -0.261003 |              -0.135259 |            0.0726286 |                       2 |        -1.37273   |
|              5 | lightgbm | weekly_last           |         -0.233762  |          -0.11157   |      -0.268032 |              -0.385268 |            0.12089   |                       2 |        -1.39256   |

Data policy: this report is aggregate-only and includes no raw WRDS/vendor records.
