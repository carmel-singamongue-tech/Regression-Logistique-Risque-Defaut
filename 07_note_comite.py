from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logistic_from_scratch import sigmoid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
data_file, model_file = OUT / "donnees_preparees.npz", OUT / "modele_manuel.npz"
cost_file = OUT / "05_seuils_couts.csv"
if not all(x.exists() for x in [data_file, model_file, cost_file]):
    raise SystemExit("Exécute les étapes 01 à 06 avant de générer la note.")

d, m = np.load(data_file), np.load(model_file)
table = pd.read_csv(cost_file)
best = table.sort_values(["cout_total", "seuil"], ascending=[True, False]).iloc[0]
p = sigmoid(d["X_test"] @ m["w"] + float(m["b"]))
y = d["y_test"]
pred = (p >= 0.5).astype(int)
cm = confusion_matrix(y, pred, labels=[0, 1])
acc = accuracy_score(y, pred)
prec = precision_score(y, pred, zero_division=0)
rec = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)
auc = roc_auc_score(y, p)
odds = pd.DataFrame({"variable": ["revenu", "endettement", "anciennete", "epargne", "incidents"],
                     "coefficient": m["w"], "exp_w": np.exp(m["w"])}).sort_values("exp_w", ascending=False)
high, low = odds.iloc[0], odds.iloc[-1]

note = f"""# NOTE AU COMITÉ — Détection du défaut de paiement

## 1. Objectif et méthode
Le projet construit un modèle de régression logistique sur 2 000 observations synthétiques. Les données sont séparées en 80 % entraînement et 20 % test avec stratification. La standardisation est ajustée uniquement sur le jeu d'entraînement. Le modèle est programmé par descente de gradient et comparé à scikit-learn.

## 2. Performance au seuil de 0,50
- Exactitude : {acc:.3f}
- Précision : {prec:.3f}
- Rappel : {rec:.3f}
- F1 : {f1:.3f}
- AUC ROC : {auc:.3f}
- Matrice de confusion : `[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]`

Un modèle naïf prédisant toujours « pas de défaut » peut obtenir une exactitude voisine de 85 %, mais son rappel est nul : il ne détecte aucun défaut.

## 3. Seuil recommandé et coût
Avec un coût de 10 unités par faux négatif et 1 unité par faux positif, le seuil minimisant le coût parmi les seuils testés est **{best['seuil']:.2f}**. Le coût associé est de **{best['cout_total']:.0f} unités**. Le choix du seuil traduit donc les priorités métier : éviter un défaut manqué est considéré dix fois plus coûteux qu'un bon client classé à tort comme risqué.

## 4. Variables influentes
Sur les variables standardisées, le rapport de cotes le plus élevé est associé à **{high['variable']}** (exp(w) = {high['exp_w']:.3f}) et le plus faible à **{low['variable']}** (exp(w) = {low['exp_w']:.3f}). Une valeur supérieure à 1 augmente les cotes estimées de défaut pour une hausse d'un écart-type ; une valeur inférieure à 1 les diminue.

## 5. Limites et recommandation
Les données et les noms de variables sont artificiels ; les performances ne peuvent pas être généralisées à une banque ou à une population réelle. Les coûts sont des hypothèses pédagogiques. Pour un projet réel, le seuil doit être choisi sur un jeu de validation séparé, puis le jeu de test doit servir à l'évaluation finale. Il faudrait aussi vérifier la représentativité des données, les biais, l'équité, la conformité et la stabilité du modèle.

**Recommandation :** retenir provisoirement le seuil **{best['seuil']:.2f}** dans le cadre de cette simulation, car il minimise le coût défini dans l'énoncé. Ce prototype ne doit pas être utilisé seul pour prendre une décision réelle de crédit.
"""
(OUT / "07_note_comite.md").write_text(note, encoding="utf-8")
print("=" * 60)
print("ÉTAPE 7 — NOTE AU COMITÉ")
print("=" * 60)
print(note)
print("\\nFichier créé : outputs/07_note_comite.md")
