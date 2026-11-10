from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.logistic_from_scratch import train_logistic, sigmoid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
data_file = OUT / "donnees_preparees.npz"
if not data_file.exists():
    raise SystemExit("Exécute d'abord 01_preparation.py")

d = np.load(data_file)
model = train_logistic(d["X_train"], d["y_train"])
np.savez(OUT / "modele_manuel.npz", w=model["w"], b=model["b"],
         loss_history=np.array(model["loss_history"]), n_iter=model["n_iter"])

print("=" * 60)
print("ÉTAPE 2 — RÉGRESSION LOGISTIQUE PROGRAMMÉE À LA MAIN")
print("=" * 60)
print("Sigmoïde : sigma(z) = 1 / (1 + exp(-z))")
print("Exemples :")
for z in [-2, 0, 2]:
    print(f"sigma({z}) = {sigmoid(np.array(z)):.4f}")
print("\\nParamètres appris :")
print("Poids w :", np.round(model["w"], 5))
print("Biais b :", round(model["b"], 5))
print("Nombre d'itérations :", model["n_iter"])
print(f"Coût initial : {model['loss_history'][0]:.6f}")
print(f"Coût final   : {model['loss_history'][-1]:.6f}")
print("Le coût a diminué :", model["loss_history"][-1] < model["loss_history"][0])

plt.figure(figsize=(8, 4))
plt.plot(model["loss_history"])
plt.xlabel("Itération")
plt.ylabel("Entropie croisée")
plt.title("Décroissance de la fonction coût")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "02_decroissance_cout.png", dpi=160)
plt.show()
print("\\nFichier créé : outputs/modele_manuel.npz")
