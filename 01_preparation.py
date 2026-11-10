from pathlib import Path
import numpy as np
from src.data import prepare_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

d = prepare_data()

# Sauvegarde des données préparées
np.savez(
    OUT / "donnees_preparees.npz",
    X_train=d["X_train"],
    X_test=d["X_test"],
    y_train=d["y_train"],
    y_test=d["y_test"]
)

print("=" * 60)
print("ÉTAPE 1 — PRÉPARATION DES DONNÉES")
print("=" * 60)

print(f"Nombre total d'observations : {len(d['y'])}")
print(f"Nombre de variables : {len(d['feature_names'])}")

print(f"Entraînement : {len(d['y_train'])} observations (80 %)")
print(f"Test         : {len(d['y_test'])} observations (20 %)")

print("\nProportion de défauts :")
print(f"Jeu complet  : {d['y'].mean():.2%}")
print(f"Entraînement : {d['y_train'].mean():.2%}")
print(f"Test         : {d['y_test'].mean():.2%}")

print("\nDimensions après standardisation :")
print("X_train :", d["X_train"].shape)
print("X_test  :", d["X_test"].shape)

print("\nMoyennes des variables standardisées (entraînement) :")
print(np.round(d["X_train"].mean(axis=0), 4))

print("\nÉcarts-types (entraînement) :")
print(np.round(d["X_train"].std(axis=0), 4))

print("\nFichier créé : outputs/donnees_preparees.npz")
print("\nÉTAPE 1 TERMINÉE AVEC SUCCÈS !")