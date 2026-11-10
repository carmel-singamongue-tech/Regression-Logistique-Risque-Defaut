# Exécution étape par étape — captures pour le rapport

Exécuter les fichiers depuis le terminal VS Code, dans l'ordre indiqué :

```powershell
python 01_preparation.py
python 02_regression_manuel.py
python 03_comparaison_sklearn.py
python 04_evaluation.py
python 05_optimisation_seuil.py
python 06_interpretation.py
python 07_note_comite.py
```

Chaque script affiche les résultats de son étape. Les graphiques sont affichés puis enregistrés dans `outputs/`. Les fichiers de données et modèles intermédiaires sont également enregistrés dans `outputs/`.

## Captures recommandées

1. `01_preparation.py` : capture du terminal avec les effectifs, proportions de défauts et dimensions.
2. `02_regression_manuel.py` : capture des poids, biais, coût initial/final et du graphique de décroissance.
3. `03_comparaison_sklearn.py` : capture du tableau comparant les coefficients manuels et scikit-learn.
4. `04_evaluation.py` : capture des métriques et des deux matrices de confusion.
5. `05_optimisation_seuil.py` : capture du seuil optimal, du coût et des trois graphiques.
6. `06_interpretation.py` : capture du tableau `exp(w)` et des explications.
7. `07_note_comite.py` : capture de la note affichée dans le terminal ou ouverture de `outputs/07_note_comite.md` dans VS Code.

## Conseils pour de bonnes captures

- Agrandir le terminal et augmenter sa hauteur avant la capture.
- Exécuter une seule étape à la fois.
- Faire apparaître la commande exécutée et les résultats dans la même capture.
- Pour les graphiques, utiliser la fenêtre graphique affichée par Matplotlib.
- Enregistrer les images dans un dossier `captures/` et les numéroter `01_preparation.png`, ..., `07_note_comite.png`.
- Ne pas modifier les résultats : ils proviennent du jeu de données et des paramètres fixés dans l'énoncé.

## Attention à l'interprétation
Le seuil est optimisé sur le jeu de test uniquement pour respecter la consigne pédagogique. Dans un protocole de machine learning rigoureux, le seuil serait choisi sur un jeu de validation indépendant.
