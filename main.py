from pathlib import Path
import matplotlib.pyplot as plt

from src.data import prepare_data
from src.logistic_from_scratch import train_logistic
from src.evaluation import evaluate_all, plot_cost_curve, plot_curves
from src.report import write_outputs

def main():
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    out.mkdir(exist_ok=True)

    data = prepare_data()
    model = train_logistic(data["X_train"], data["y_train"])
    results = evaluate_all(
        model, data["X_train"], data["y_train"],
        data["X_test"], data["y_test"], data["feature_names"]
    )

    print("\n=== TP : RÉGRESSION LOGISTIQUE ===")
    print(f"Taux de défaut global : {data['y'].mean():.2%}")
    print(f"Taux de défaut train : {data['y_train'].mean():.2%}")
    print(f"Taux de défaut test  : {data['y_test'].mean():.2%}")
    print(f"Coût final entraînement : {model['loss_history'][-1]:.6f}")
    print(f"Itérations : {model['n_iter']}")
    print("\nComparaison des coefficients :")
    print(results["comparison"].to_string(index=False))
    print(f"\nÉcart intercept : {results['intercept_difference']:.6f}")
    print("\nMétriques au seuil 0,50 :")
    for name, value in results["metrics_05"].items():
        print(f"{name:>14} : {value:.4f}")
    print("\nClassifieur naïf (toujours classe 0) :")
    for name, value in results["naive_metrics"].items():
        print(f"{name:>14} : {value:.4f}")
    print(f"\nSeuil optimal : {results['best_threshold']:.2f}")
    print(f"Coût optimal : {results['best_cost']}")
    print(f"Coût au seuil 0,50 : {results['cost_at_05']}")
    print(f"AUC ROC : {results['auc']:.4f}")
    print("\nRapports de cotes :")
    print(results["odds_ratios"].to_string(index=False))

    plot_cost_curve(results["threshold_table"], out / "cout_par_seuil.png")
    plot_curves(data["y_test"], results["probabilities"], out)
    plt.figure(figsize=(8, 4))
    plt.plot(model["loss_history"])
    plt.xlabel("Itération")
    plt.ylabel("Entropie croisée")
    plt.title("Décroissance de la fonction coût")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "decroissance_cout.png", dpi=160)
    plt.close()

    write_outputs(results, model, data, out)
    print(f"\nTous les fichiers de résultats sont dans : {out}")

if __name__ == "__main__":
    main()
