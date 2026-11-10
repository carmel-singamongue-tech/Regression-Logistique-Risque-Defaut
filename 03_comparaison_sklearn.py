from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
data_file, model_file = OUT / "donnees_preparees.npz", OUT / "modele_manuel.npz"
if not data_file.exists() or not model_file.exists():
    raise SystemExit("Exécute d'abord 01_preparation.py puis 02_regression_manuel.py")

d, m = np.load(data_file), np.load(model_file)
sk = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
sk.fit(d["X_train"], d["y_train"])

names = ["revenu", "endettement", "anciennete", "epargne", "incidents"]
table = pd.DataFrame({
    "variable": names,
    "coefficient_manuel": m["w"],
    "coefficient_sklearn": sk.coef_[0],
    "ecart_absolu": np.abs(m["w"] - sk.coef_[0])
})
table.to_csv(OUT / "03_comparaison_coefficients.csv", index=False, encoding="utf-8-sig")

print("=" * 60)
print("ÉTAPE 3 — COMPARAISON AVEC SCIKIT-LEARN")
print("=" * 60)
print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"\\nBiais manuel   : {float(m['b']):.6f}")
print(f"Biais sklearn  : {float(sk.intercept_[0]):.6f}")
print(f"Écart du biais : {abs(float(m['b']) - sk.intercept_[0]):.6f}")
print("\\nFichier créé : outputs/03_comparaison_coefficients.csv")
