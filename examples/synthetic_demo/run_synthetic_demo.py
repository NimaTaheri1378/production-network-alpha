from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
matrix = pd.read_csv(ROOT / "synthetic_model_matrix.csv")
X = matrix[["one_hop_signed_spillover", "own_news_shock"]]
y = matrix["label_fwd_abret_5d"]
model = Ridge(alpha=1.0).fit(X, y)
pred = model.predict(X)
print("synthetic_demo_rows", len(matrix))
print("synthetic_demo_r2", round(float(r2_score(y, pred)), 6))
print("synthetic_demo_coef", [round(float(x), 6) for x in model.coef_])
