from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
model_file = OUT / "modele_manuel.npz"
if not model_file.exists():
    raise SystemExit("Exécute d'abord 02_regression_manuel.py.")

m = np.load(model_file)
names = ["revenu", "endettement", "anciennete", "epargne", "incidents"]
table = pd.DataFrame({"variable": names, "coefficient": m["w"],
                      "rapport_de_cotes_exp_w": np.exp(m["w"])})
table["effet"] = np.where(table["coefficient"] > 0, "augmente les cotes de défaut",
                  np.where(table["coefficient"] < 0, "diminue les cotes de défaut",
                           "effet nul"))
table.to_csv(OUT / "06_rapports_de_cotes.csv", index=False, encoding="utf-8-sig")

print("=" * 60)
print("ÉTAPE 6 — EXPLICATION DU MODÈLE")
print("=" * 60)
print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\\nRègle d'interprétation :")
print("- exp(w) > 1 : une hausse d'un écart-type augmente les cotes de défaut.")
print("- exp(w) < 1 : une hausse d'un écart-type diminue les cotes de défaut.")
print("\\nDeux phrases destinées à un client refusé :")
print("« Votre dossier présente, selon les informations communiquées, une probabilité")
print("estimée de défaut supérieure au seuil de décision retenu. Cette estimation")
print("résulte de plusieurs caractéristiques prises en compte par le modèle et")
print("constitue une aide statistique, non une explication exhaustive de votre situation. »")
print("\\nFichier créé : outputs/06_rapports_de_cotes.csv")
