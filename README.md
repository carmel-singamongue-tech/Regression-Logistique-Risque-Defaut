# TP — Régression logistique : détection du défaut

Correction complète des étapes 1 à 7. Le jeu de données est généré exactement selon les paramètres de l'énoncé.

## Ouvrir et exécuter dans VS Code
1. Décompresser l'archive ZIP.
2. Dans VS Code, choisir **Fichier > Ouvrir le dossier** et sélectionner `TP_Regression_Logistique_Correction`.
3. Ouvrir un terminal intégré.
4. Installer les dépendances :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloque l'activation, exécuter directement `python -m pip install -r requirements.txt`, puis `python main.py`.

5. Lancer le projet :

```powershell
python main.py
```

Les tableaux, graphiques et la note au comité seront générés dans `outputs/`.

## Contenu
- `main.py` : lance toute la correction.
- `src/data.py` : génération, séparation stratifiée 80/20 et standardisation sans fuite de données.
- `src/logistic_from_scratch.py` : sigmoïde, entropie croisée, gradient et descente de gradient.
- `src/evaluation.py` : métriques, comparaison scikit-learn, seuils, coûts et courbes.
- `src/report.py` : export des résultats et note au comité.
- `requirements.txt` : dépendances.

## Remarque méthodologique
Pour respecter l'exercice, le seuil est sélectionné sur le jeu de test. Dans un véritable projet, il faudrait le sélectionner sur un jeu de validation distinct et n'utiliser le jeu de test qu'une seule fois pour l'évaluation finale. Les données et noms de variables sont synthétiques/illustratifs : ce modèle n'est pas destiné à décider de vrais crédits.
