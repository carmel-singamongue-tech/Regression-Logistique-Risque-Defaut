from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from src.logistic_from_scratch import sigmoid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
data_file, model_file = OUT / "donnees_preparees.npz", OUT / "modele_manuel.npz"
if not data_file.exists() or not model_file.exists():
    raise SystemExit("Exécute d'abord les étapes 01 et 02.")

d, m = np.load(data_file), np.load(model_file)
y = d["y_test"]
p = sigmoid(d["X_test"] @ m["w"] + float(m["b"]))

rows = []
for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rows.append({"seuil": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                 "cout_FN": 10 * fn, "cout_FP": fp, "cout_total": 10 * fn + fp})
table = pd.DataFrame(rows)
best = table.sort_values(["cout_total", "seuil"], ascending=[True, False]).iloc[0]
row05 = table.iloc[(table["seuil"] - 0.50).abs().argmin()]
table.to_csv(OUT / "05_seuils_couts.csv", index=False, encoding="utf-8-sig")

print("=" * 60)
print("ÉTAPE 5 — CHOIX DU SEUIL SELON LES COÛTS")
print("=" * 60)
print("Coût retenu = 10 × faux négatifs + 1 × faux positifs")
print(f"Seuil optimal : {best['seuil']:.2f}")
print(f"Coût minimal  : {best['cout_total']:.0f}")
print(f"Coût au seuil 0,50 : {row05['cout_total']:.0f}")
print("\\nDétail au seuil optimal :")
print(f"TN={best['TN']:.0f}, FP={best['FP']:.0f}, FN={best['FN']:.0f}, TP={best['TP']:.0f}")

plt.figure(figsize=(8, 4))
plt.plot(table["seuil"], table["cout_total"])
plt.axvline(best["seuil"], linestyle="--", label=f"Seuil optimal = {best['seuil']:.2f}")
plt.xlabel("Seuil")
plt.ylabel("Coût total")
plt.title("Coût total selon le seuil")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "05_cout_par_seuil.png", dpi=160)
plt.show()

precision, recall, _ = precision_recall_curve(y, p)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision)
plt.xlabel("Rappel")
plt.ylabel("Précision")
plt.title("Courbe précision-rappel")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "05_precision_rappel.png", dpi=160)
plt.show()

fpr, tpr, _ = roc_curve(y, p)
auc = roc_auc_score(y, p)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", label="Aléatoire")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "05_courbe_roc.png", dpi=160)
plt.show()
print(f"\\nAUC ROC : {auc:.4f}")
