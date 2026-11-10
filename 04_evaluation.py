from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.logistic_from_scratch import sigmoid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
data_file, model_file = OUT / "donnees_preparees.npz", OUT / "modele_manuel.npz"
if not data_file.exists() or not model_file.exists():
    raise SystemExit("Exécute d'abord les étapes 01 et 02.")

d, m = np.load(data_file), np.load(model_file)
y = d["y_test"]
proba = sigmoid(d["X_test"] @ m["w"] + float(m["b"]))
pred = (proba >= 0.5).astype(int)
naive = np.zeros_like(y)

def show_metrics(label, truth, prediction):
    print(f"\\n{label}")
    print(f"Exactitude : {accuracy_score(truth, prediction):.4f}")
    print(f"Précision  : {precision_score(truth, prediction, zero_division=0):.4f}")
    print(f"Rappel     : {recall_score(truth, prediction, zero_division=0):.4f}")
    print(f"F1         : {f1_score(truth, prediction, zero_division=0):.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(truth, prediction, labels=[0, 1]))

print("=" * 60)
print("ÉTAPE 4 — ÉVALUATION AU SEUIL 0,50")
print("=" * 60)
show_metrics("Modèle logistique", y, pred)
show_metrics("Modèle naïf : toujours « pas de défaut »", y, naive)
print("\\nInterprétation : le modèle naïf peut avoir une exactitude proche de 85 %,")
print("mais son rappel est nul car il ne détecte aucun défaut.")
pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1]),
             index=["Réel_0", "Réel_1"], columns=["Prédit_0", "Prédit_1"]
             ).to_csv(OUT / "04_matrice_confusion.csv", encoding="utf-8-sig")
