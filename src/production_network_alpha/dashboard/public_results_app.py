from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    try:
        from dash import Dash, dash_table, dcc, html
    except Exception as exc:
        raise SystemExit(f"Dash is not installed: {exc}")

    root = Path(__file__).resolve().parents[3]
    public = root / "artifacts" / "release_public"
    ranking_path = public / "phase7_0_10bps_candidate_ranking.csv"
    summary_path = public / "phase7_0_quality_summary.json"

    ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()

    app = Dash(__name__)
    app.title = "Production Network Alpha"
    app.layout = html.Div([
        html.H1("Trading the Production Network"),
        html.P("Aggregate public results. No raw WRDS/vendor records are included."),
        html.H2("10 bps net candidate ranking"),
        dash_table.DataTable(
            data=ranking.head(20).to_dict("records"),
            columns=[{"name": c, "id": c} for c in ranking.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
        ),
        html.H2("Figures"),
        dcc.Markdown("See `docs/figures/` and `artifacts/release_public/figures/` for generated charts."),
    ], style={"fontFamily": "system-ui", "margin": "40px"})
    app.run(debug=False)


if __name__ == "__main__":
    main()
