# Exploitation de bases de données d'assurance automobile

**Étude réalisée par :** Morel POUNIGUIA

Audit, nettoyage, analyse, modélisation et tarification d'un portefeuille automobile
(100 000 polices sur 2009–2010, 13 300 lignes de sinistres).

## Organisation du dossier

```
Projet AIMAF Actuariat/
├── analyse_resultats.ipynb   ← FICHIER PRINCIPAL : tout le code, les tableaux, les graphiques et les interprétations
├── README.md                 ← ce fichier
├── donnees/                  ← données d'entrée (non publiées sur GitHub, voir plus bas)
│   ├── data_sin.csv            sinistres
│   └── data_ptf_Eco.xlsx       portefeuille et expositions
├── resultats/
│   ├── tableaux/               tableaux CSV de chaque étape (etape1_… à etape9_…), lisibles dans Excel
│   └── figures/                graphiques PNG utilisés dans le rapport
├── rapport/
│   ├── rapport_analyse.pdf     rapport complet
│   └── rapport_analyse.tex     source LaTeX du rapport
├── code/
│   ├── analyse_actuarielle.py  version « script » de l'analyse (étapes 1 à 8 + rapport PDF)
│   ├── creer_notebook.py       reconstruit le notebook à partir du script
│   ├── cellules_extra.py       sections ajoutées au notebook (visualisations avancées, tarification, conclusion)
│   └── reorganiser_notebook.py place le code de chaque étape juste avant ses résultats
└── consignes/                ← sujet du projet (non publié sur GitHub)
```

## Données

Les fichiers de données et le sujet du projet ne sont pas inclus dans ce dépôt. Pour relancer l'analyse,
placer `data_sin.csv` et `data_ptf_Eco.xlsx` dans un dossier `donnees/` à la racine du projet.
Le notebook, le rapport et les dossiers `resultats/` contiennent déjà tous les résultats.

## Organisation du notebook

- **Partie 0 — Préparation** : configuration, outils de mise en forme et d'affichage.
- **Analyse étape par étape** : pour chaque étape, l'explication (objectif, méthode, lecture), puis le **code**
  des fonctions de l'étape, puis son **exécution** : tableaux, graphiques et « points clés » chiffrés s'affichent
  **juste sous le code**. Chaque cellule de code a une sortie (au minimum un message de confirmation).

## Étapes de l'analyse

| Étape | Contenu | Tableaux produits |
|---|---|---|
| 1–2 | Compréhension et audit des données (dictionnaire, manquants, doublons, valeurs aberrantes, règles métier) | `etape1_*`, `etape2_*` |
| 3 | Nettoyage documenté (journal des décisions) et création des variables | `etape3_*` |
| 3 bis | Statistiques détaillées (moyenne, médiane, écart-type, quantiles, asymétrie…) | `stats_detaillees_*` |
| 4 | Jointures des bases et contrôles | `etape4_*` |
| 5 | Analyse descriptive univariée, bivariée, corrélations, GLM de Poisson | `etape5_*` |
| 6 | Segmentation : ACP, k-means, classification hiérarchique | `etape6_*` |
| 7 | Modélisation fréquence et coût : GLM, forêt aléatoire, gradient boosting | `etape7_*` |
| 7 bis | Visualisations avancées : interaction âge × bonus, calibrage, Lorenz, effets partiels | — |
| 8 | Interprétation : concentration des coûts, polices atypiques | `etape8_*` |
| 9 | Application : grille tarifaire GLM (2009), prévision 2010, antisélection | `etape9_*` |

## Comment lancer l'analyse

**Avec le notebook (recommandé)**
1. Ouvrir `analyse_resultats.ipynb` dans VS Code ou Jupyter, depuis ce dossier.
2. Choisir le noyau Python 3, puis « Run All ».
3. Durée totale : environ 2 minutes.

**Avec le script (sans Jupyter)**

```
python code/analyse_actuarielle.py
```

Le script refait les étapes 1 à 8 et le rapport PDF, mais pas les sections 3 bis, 7 bis et 9, qui
n'existent que dans le notebook.

**Bibliothèques nécessaires**

```
pip install pandas numpy scipy scikit-learn matplotlib openpyxl ipykernel
```

Le rapport PDF nécessite en plus une distribution LaTeX (`pdflatex`). Sans elle, seul le fichier `.tex` est produit.

## Points importants

- **Travaillez dans le notebook** : c'est le livrable principal.
- `code/creer_notebook.py` **écrase** le notebook. Ne le lancez que si vous avez modifié
  `code/analyse_actuarielle.py` et voulez reconstruire le notebook à partir du script.
  Lancez ensuite `python3 code/reorganiser_notebook.py` puis « Run All », pour retrouver l'ordre code → résultats.
- Hypothèses **retenues** : chargement commercial de 30 % (étape 9), profils types fictifs (étape 9),
  traitement de la variable *Occupation* (incohérence « Housewife » chez des hommes : lignes conservées et signalées,
  variable gardée dans le tarif).
