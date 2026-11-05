# ==============================================================================
# Projet  : Exploitation de bases de données d'assurance automobile
#           Audit, analyse, modélisation et tarification
# Auteur  : Morel POUNIGUIA
# Fichier : reorganiser_notebook.py
# ==============================================================================

"""Réorganise analyse_resultats.ipynb pour que le résultat de chaque code s'affiche juste en dessous.

Avant : partie A (toutes les fonctions, sans résultat) puis partie B (exécution et résultats).
Après : pour chaque étape -> explication, code des fonctions de l'étape, exécution et résultats.
Chaque cellule de définition se termine par un print de confirmation, pour qu'aucune cellule
de code ne reste sans sortie.

À lancer depuis la racine du projet, sur un notebook produit par code/creer_notebook.py :
    python3 code/reorganiser_notebook.py
puis réexécuter le notebook (Run All) pour régénérer toutes les sorties.
"""
import json
import re
from pathlib import Path

NB = Path("analyse_resultats.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
C = nb["cells"]


def src(i):
    return "".join(C[i]["source"])


def set_src(cell, texte):
    cell["source"] = texte.splitlines(keepends=True)


def md(texte):
    return {"cell_type": "markdown", "metadata": {}, "source": texte.splitlines(keepends=True)}


# Garde-fou : le notebook doit avoir la structure d'origine (partie A puis partie B)
assert src(2).startswith("## Partie A"), "Le notebook semble déjà réorganisé."
assert src(27).startswith("## Partie B")
attendus = {9: "### A.4", 11: "### A.5", 13: "### A.6", 15: "### A.7", 17: "### A.8", 19: "### A.9",
            21: "### A.10", 23: "### A.11", 25: "### A.12", 28: "### Étapes 1 et 2", 30: "### Étape 3 ",
            32: "### Étape 3 bis", 38: "### Étape 4", 40: "### Étape 5", 42: "### Étape 6", 44: "### Étape 7 ",
            46: "### Étape 7 bis", 53: "### Étape 8", 55: "### Étape 9", 61: "### Synthèse", 63: "### Rapport PDF",
            65: "### Conclusion"}
for i, debut in attendus.items():
    assert src(i).startswith(debut), (i, src(i)[:60])

# 1) Ligne de confirmation à la fin de chaque cellule de définition
CONFIRMATIONS = {
    4: 'print(f"Configuration chargée : données dans {DONNEES.name}/, tableaux dans resultats/tableaux, '
       'figures dans resultats/figures, graine aléatoire = {GRAINE}")',
    6: 'print("Outils LaTeX chargés : echapper, nombre, tableau, var_tex")',
}
for i in (10, 12, 14, 16, 18, 20, 22, 24, 26):
    publiques = [f for f in re.findall(r"^def (\w+)", src(i), re.M) if not f.startswith(("_", "s_"))]
    principales = [f for f in publiques if f.startswith(("etape", "stats_", "generer_"))] or publiques
    CONFIRMATIONS[i] = (f'print("Fonctions chargées ({len(publiques)}) ; fonction principale : '
                        f'{", ".join(f + "()" for f in principales[:2])}")')
for i, ligne in CONFIRMATIONS.items():
    set_src(C[i], src(i).rstrip("\n") + "\n\n" + ligne + "  # confirme l'exécution de la cellule\n")


# 2) Titres : les sections de code deviennent des sous-sections de leur étape
def titre_code(i, texte):
    set_src(C[i], "#### " + texte + "\n\nDéfinition des fonctions de l'étape. Le résultat de leur "
            "exécution s'affiche dans la cellule suivante.")


titre_code(9, "Code : chargement, dictionnaire, audit de qualité")
titre_code(11, "Code : nettoyage et préparation")
titre_code(13, "Code : statistiques détaillées")
titre_code(15, "Code : jointures et contrôles")
titre_code(17, "Code : analyse descriptive et GLM de Poisson")
titre_code(19, "Code : ACP, k-means, classification hiérarchique")
titre_code(21, "Code : modèles de fréquence et de coût")
titre_code(23, "Code : concentration des coûts et polices atypiques")
titre_code(25, "Code : génération du rapport LaTeX / PDF")
for i, t in ((3, "### 0.1 Configuration : bibliothèques, chemins et constantes"),
             (5, "### 0.2 Outils de mise en forme du rapport LaTeX"),
             (7, "### 0.3 Fonctions d'affichage du notebook et vérification des données")):
    set_src(C[i], t)

# 3) Textes d'introduction
intro = src(0)
intro = intro.replace(
    "| **A** | **Code de l'analyse** : configuration, outils, fonctions de chaque étape |\n"
    "| **B** | **Résultats et interprétations** : |\n",
    "| **0** | **Préparation** : configuration, outils de mise en forme et d'affichage |\n")
intro = re.sub(
    r"> Le notebook a \*\*deux parties\*\*.*",
    "> Chaque étape suit le même ordre : **explication** (objectif, méthode, lecture), **code** des fonctions de "
    "l'étape, puis **exécution** avec les tableaux, graphiques et points clés affichés **juste en dessous**. "
    "Exécutez le notebook dans l'ordre (« Run All »). Si vous modifiez une fonction, relancez sa cellule puis "
    "la cellule d'exécution qui la suit.",
    intro, flags=re.S)
set_src(C[0], intro)
prep = md("## Partie 0 — Préparation\n\nConfiguration, outils de mise en forme du rapport et fonctions "
          "d'affichage. Chaque cellule confirme son exécution par un message.")
analyse = md("## Analyse étape par étape\n\nChaque étape suit le même plan : **objectif, méthode et lecture** "
             "(texte), **code** des fonctions de l'étape, puis **calcul** (tableaux et graphiques) et "
             "**points clés** (constats chiffrés calculés automatiquement), affichés sous le code.")

# 4) Nouvel ordre : chaque bloc de code est placé juste avant la cellule qui l'exécute
ordre = [0, 1, "prep", 3, 4, 5, 6, 7, 8, "analyse",
         28, 9, 10, 29,                 # étapes 1-2
         30, 11, 12, 31,                # étape 3
         32, 13, 14, 33, 34, 35, 36, 37,  # étape 3 bis
         38, 15, 16, 39,                # étape 4
         40, 17, 18, 41,                # étape 5
         42, 19, 20, 43,                # étape 6
         44, 21, 22, 45,                # étape 7
         46, 47, 48, 49, 50, 51, 52,    # étape 7 bis
         53, 23, 24, 54,                # étape 8
         55, 56, 57, 58, 59, 60,        # étape 9
         61, 62,                        # synthèse
         63, 25, 26, 64,                # rapport PDF
         65]                            # conclusion
pris = [k for k in ordre if isinstance(k, int)]
assert sorted(pris) == [k for k in range(len(C)) if k not in (2, 27)], "Cellule perdue ou dupliquée"
nb["cells"] = [prep if k == "prep" else analyse if k == "analyse" else C[k] for k in ordre]
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Notebook réorganisé : {len(nb['cells'])} cellules. Relancez-le (Run All) pour régénérer les sorties.")
