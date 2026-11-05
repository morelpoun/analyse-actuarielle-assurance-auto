# ==============================================================================
# Projet  : Exploitation de bases de données d'assurance automobile
#           Audit, analyse, modélisation et tarification
# Auteur  : Morel POUNIGUIA
# Fichier : creer_notebook.py
# ==============================================================================

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

from cellules_extra import CONCLUSION, CONFORMITE, TARIF, VIZ

# Générateur du notebook : python code/creer_notebook.py (reconstruit et exécute analyse_resultats.ipynb).
# Attention : il écrase le notebook ; les modifications faites à la main dans le notebook sont perdues.
ICI = Path(__file__).resolve().parent  # dossier code/

DOSSIER = str(ICI.parent)  # racine du projet
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

# ---------------------------------------------------------------- Introduction et préparation
cells += [
    md("# Exploitation de bases de données d'assurance automobile\n"
       "### Audit, analyse, modélisation et tarification\n\n"
       "**Étude réalisée par :** Morel POUNIGUIA  \n"
       "\n---\n\n"
       "Ce notebook exécute le script `Untitled-1.py` **une étape à la fois**. Pour chaque étape on trouve :\n\n"
       "1. **Une explication** : objectif, méthode, et comment lire les résultats ;\n"
       "2. **Le calcul**, avec les tableaux importants et les graphiques ;\n"
       "3. **Les points clés**, chiffrés automatiquement à partir des résultats.\n\n"
       "| Étape | Contenu |\n|---|---|\n"
       "| 1–2 | Compréhension et audit des données |\n"
       "| 3 | Nettoyage et préparation |\n"
       "| 3 bis | Statistiques détaillées |\n"
       "| 4 | Jointure des bases |\n"
       "| 5 | Analyse descriptive et exploratoire |\n"
       "| 6 | Segmentation (ACP, k-means, classification hiérarchique) |\n"
       "| 7 | Modélisation de la fréquence et du coût |\n"
       "| 7 bis | Visualisations avancées |\n"
       "| 8 | Interprétation actuarielle |\n"
       "| 9 | Application : tarification et prévision |\n"
       "| — | Conclusion, limites et recommandations |\n\n"
       "> Lancez toujours la cellule **Préparation** en premier, puis les étapes dans l'ordre (ou « Run All »). "
       "Si vous modifiez `Untitled-1.py`, relancez la cellule Préparation."),
    *CONFORMITE,
    md("## Préparation\nChargement des bibliothèques, du script d'analyse et des fonctions d'affichage."),
    code(r'''import re  # expressions régulières (mise en forme des nombres)
import importlib.util  # outil pour charger un fichier .py dont le nom contient un tiret
import time  # mesure du temps d'exécution de chaque étape
import warnings  # gestion des messages d'avertissement
from pathlib import Path  # manipulation simple des chemins de fichiers

import matplotlib.pyplot as plt  # graphiques
import numpy as np  # calculs numériques
import pandas as pd  # tableaux de données
from IPython.display import Image, Markdown, display  # affichage d'images et de texte dans le notebook

%matplotlib inline

warnings.filterwarnings("ignore")  # masque les avertissements numériques sans gravité (numpy/sklearn)
pd.set_option("display.max_columns", 50)  # affiche jusqu'à 50 colonnes sans les tronquer
pd.set_option("display.max_colwidth", 80)  # coupe les textes trop longs à 80 caractères
pd.set_option("display.float_format", "{:,.4g}".format)  # arrondit les nombres à 4 chiffres significatifs

SCRIPT = Path("/Users/morelpouniguia/Untitled-1.py")  # chemin du script d'analyse principal
spec = importlib.util.spec_from_file_location("analyse", SCRIPT)  # prépare le chargement du script comme module
A = importlib.util.module_from_spec(spec)  # crée le module vide nommé « analyse »
spec.loader.exec_module(A)  # exécute le script : toutes ses fonctions deviennent accessibles via A.xxx
A.preparer_dossiers()  # crée resultats/tableaux, resultats/figures et rapport/ s'ils n'existent pas

for fichier in (A.FICHIER_CSV, A.FICHIER_XLSX):  # parcourt les deux fichiers de données attendus
    assert fichier.exists(), f"Fichier introuvable : {fichier}"  # arrête tout si une donnée manque

VOIR_TOUS_LES_CSV = False  # True : affiche aussi tous les fichiers CSV de chaque étape
LIGNES_MAX = 10  # nombre de lignes affichées pour chaque tableau
duree = {}  # durée de chaque étape, en secondes


def nombres(colonne):  # transforme « 0,25 » en 0.25 quand toute la colonne est numérique
    if colonne.dtype != object:  # colonne déjà numérique : rien à faire
        return colonne  # on la renvoie telle quelle
    essai = pd.to_numeric(colonne.str.replace(",", ".", regex=False), errors="coerce")  # tentative de conversion
    return essai if essai.notna().sum() == colonne.notna().sum() else colonne  # garde le texte si la conversion échoue


def montrer_fig():  # affiche la figure en cours dans le notebook
    fig = plt.gcf()  # figure en cours de construction
    display(fig)  # affichage explicite (le script d'analyse force matplotlib en mode « Agg », sans fenêtre)
    plt.close(fig)  # libère la mémoire et évite un double affichage


def tableau(df, titre, n=LIGNES_MAX):  # affiche un tableau avec un titre et ses dimensions
    display(Markdown(f"**{titre}** — {df.shape[0]} lignes × {df.shape[1]} colonnes"))  # titre
    display(df.head(n))  # premières lignes


def virgule(texte):  # remplace le point décimal par une virgule : 41.1 → 41,1
    return re.sub(r"(?<=\d)\.(?=\d)", ",", texte)  # seulement entre deux chiffres


def points_cles(*phrases):  # affiche une liste à puces de constats
    display(Markdown("#### Points clés\n" + "\n".join(f"- {virgule(p)}" for p in phrases)))  # rendu en Markdown


def nb_fr(x, d=0):  # formate un nombre à la française : 12 345,6
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")  # espace pour les milliers, virgule décimale


def montrer_csv(prefixe):  # affiche tous les fichiers CSV d'une étape (si VOIR_TOUS_LES_CSV)
    for f in sorted(A.SORTIE.glob(f"{prefixe}_*.csv")):  # chaque CSV dont le nom commence par le préfixe
        df = pd.read_csv(f, sep=";", low_memory=False).apply(nombres)  # lecture puis conversion des nombres
        tableau(df, f.stem)  # affichage


def etat_figures():  # date de modification de chaque figure du dossier resultats/figures
    return {f: f.stat().st_mtime for f in A.FIG.glob("*.png")}  # dictionnaire fichier -> date


def montrer_figures(avant):  # affiche les figures créées ou modifiées depuis l'état « avant »
    for f in sorted(A.FIG.glob("*.png")):  # chaque figure du dossier resultats/figures
        if avant.get(f) != f.stat().st_mtime:  # nouvelle figure, ou figure réécrite pendant l'étape
            display(Markdown(f"**Figure : {f.stem.replace('fig_', '').replace('_', ' ')}**"))  # titre
            display(Image(filename=str(f), width=700))  # image


def lancer(nom, fonction, *prefixes):  # exécute une étape, mesure sa durée et affiche ses figures
    debut = time.time()  # heure de départ
    avant = etat_figures()  # figures présentes avant l'étape
    resultat = fonction()  # lance la fonction de l'étape
    duree[nom] = time.time() - debut  # enregistre le temps écoulé
    display(Markdown(f"*{nom} terminée en {duree[nom]:.1f} s*"))  # affiche la durée
    if VOIR_TOUS_LES_CSV:  # option : tous les CSV bruts
        for p in prefixes:  # pour chaque préfixe de fichiers
            montrer_csv(p)  # affichage des CSV
    montrer_figures(avant)  # figures de l'étape uniquement
    return resultat  # renvoie le résultat pour les étapes suivantes


print("Prêt :", SCRIPT.name, "chargé")  # confirme que tout est en place'''),
]

# ---------------------------------------------------------------- Étapes 1 et 2
cells += [
    md("## Étapes 1 et 2 — Compréhension et audit des données\n\n"
       "**Objectif.** Savoir ce que contiennent les trois bases (sinistres, portefeuille, expositions) "
       "et repérer tous les problèmes de qualité **avant** de les corriger.\n\n"
       "**Méthode.**\n"
       "- *Dictionnaire* : type, nature (quantitative, qualitative, identifiant) et signification de chaque variable ;\n"
       "- *Valeurs manquantes* : vraies cases vides et codes textuels (`??`, `NA`…) ;\n"
       "- *Doublons* : lignes strictement identiques et clés `PolNum` répétées ;\n"
       "- *Valeurs aberrantes* : règle de Tukey, hors de [Q1 − 1,5 × IQR ; Q3 + 1,5 × IQR] ;\n"
       "- *Règles métier* : âge < 18 ans, ancienneté supérieure à l'âge, « Housewife » de sexe masculin, etc.\n\n"
       "**Comment lire.** Une valeur aberrante au sens de Tukey n'est pas forcément une erreur : "
       "les gros sinistres sont réels. Les règles « Impossible » sont des erreurs certaines ; "
       "les règles « Suspect » sont à confirmer."),
    code(r'''res = lancer("Étapes 1-2", A.etape1_et_2, "etape1", "etape2")  # audit : res contient tous les résultats

tableau(res["dico"], "Dictionnaire des variables", 20)  # description de chaque variable
tableau(res["stats"], "Statistiques brutes (avant nettoyage)")  # moyenne, quartiles… des données brutes
tableau(res["doublons"], "Doublons par table")  # lignes en double et clés répétées
tableau(res["anomalies_e1"], "Anomalies détectées", 15)  # liste des anomalies
tableau(res["outliers"], "Valeurs aberrantes (règle de Tukey)")  # valeurs hors des bornes
tableau(res["regles"], "Contrôles des règles métier", 15)  # règles impossibles / incohérentes / suspectes
tableau(res["incoherences"], "Incohérences entre variables")  # croisements suspects
tableau(res["formats"], "Problèmes de format")  # décimales, nombres stockés en texte…

st = res["stats"].set_index("variable")  # statistiques brutes indexées par variable
dbl = res["doublons"].set_index("table")  # doublons indexés par table
inc = res["incoherences"]  # incohérences entre variables
points_cles(  # constats chiffrés
    f"Le portefeuille brut compte **{nb_fr(dbl.loc['portefeuille', 'nb_lignes'])} lignes**, "
    f"les sinistres **{nb_fr(dbl.loc['sinistres', 'nb_lignes'])} lignes**.",
    f"Doublons stricts : {dbl['nb_lignes_identiques_en_double'].sum()} lignes ; "
    f"{dbl['nb_cles_repetees'].sum()} clés `PolNum` répétées, dont {dbl['nb_cles_repetees_contenu_different'].sum()} "
    "avec un contenu différent (à arbitrer).",
    f"Valeurs impossibles avant nettoyage : âge de {st.loc['Age', 'min']:.0f} à {st.loc['Age', 'max']:.0f} ans, "
    f"ancienneté de {st.loc['Poldur', 'min']:.0f} à {st.loc['Poldur', 'max']:.0f} ans.",
    f"Incohérence la plus fréquente : « {inc.loc[inc['nb_lignes'].idxmax(), 'regle']} » "
    f"({nb_fr(inc['nb_lignes'].max())} lignes, {inc['pct'].max():.1f} %).",
    f"Montant des sinistres : la moyenne vaut **{st.loc['chg_sin', 'mean'] / st.loc['chg_sin', '50%']:.1f} fois la médiane** "
    f"({nb_fr(st.loc['chg_sin', 'mean'])} € contre {nb_fr(st.loc['chg_sin', '50%'])} €), donc la distribution est très asymétrique.",
)'''),
]

# ---------------------------------------------------------------- Étape 3
cells += [
    md("## Étape 3 — Nettoyage et préparation\n\n"
       "**Objectif.** Corriger les anomalies de l'audit et créer les variables nécessaires à l'analyse.\n\n"
       "**Méthode.** Chaque correction est notée dans un **journal des décisions** (action, nombre de lignes, "
       "justification). Puis on crée les variables actuarielles :\n"
       "- *exposition* = jours couverts / 365 ;\n"
       "- *fréquence annuelle* = nombre de sinistres / exposition ;\n"
       "- *coût moyen par sinistre* = coût total / nombre de sinistres.\n\n"
       "**Comment lire.** Le tableau des vérifications doit être entièrement « OK » : "
       "c'est la garantie que la base est exploitable."),
    code(r'''res["etape3"] = lancer("Étape 3", A.etape3, "etape3")  # base nettoyée, stockée dans res
e3 = res["etape3"]  # raccourci

tableau(pd.DataFrame(e3["journal"]), "Journal des décisions de nettoyage", 30)  # chaque correction et sa raison
tableau(e3["creees"], "Variables créées", 20)  # nouvelles variables
tableau(e3["verif"], "Vérifications après nettoyage", 20)  # contrôles de qualité finaux

aa = e3["avant_apres"]  # effectifs avant et après nettoyage
echecs = (e3["verif"]["resultat"] != "OK").sum()  # nombre de contrôles en échec
points_cles(  # constats chiffrés
    f"Sinistres : {nb_fr(aa['sin_avant'])} → **{nb_fr(aa['sin_apres'])} lignes** "
    f"({nb_fr(aa['sin_avant'] - aa['sin_apres'])} retirées).",
    f"Portefeuille : {nb_fr(aa['ptf_avant'])} → **{nb_fr(aa['ptf_apres'])} polices** ; "
    f"exposition : {nb_fr(aa['expo_avant'])} → {nb_fr(aa['expo_apres'])} lignes.",
    f"{len(e3['journal'])} décisions de nettoyage documentées, {len(e3['creees'])} variables créées.",
    "Tous les contrôles sont OK." if echecs == 0 else f"{echecs} contrôle(s) en échec : voir le tableau ci-dessus.",
)'''),
]

# ---------------------------------------------------------------- Étape 3 bis : statistiques détaillées
cells += [
    md("## Étape 3 bis — Statistiques détaillées (données nettoyées)\n\n"
       "**Objectif.** Décrire précisément chaque variable après nettoyage.\n\n"
       "**Statistiques calculées.**\n\n"
       "| Statistique | Signification |\n|---|---|\n"
       "| moyenne / médiane | valeur centrale ; un grand écart entre les deux signale une distribution asymétrique |\n"
       "| écart-type | dispersion autour de la moyenne |\n"
       "| CV % | écart-type / moyenne : dispersion relative (calculé seulement si la moyenne est > 0) |\n"
       "| Q1, Q3, IQR | quartiles et écart interquartile (dispersion des 50 % centraux) |\n"
       "| P5, P95, P99 | percentiles : P99 est la valeur dépassée par seulement 1 % des observations |\n"
       "| asymétrie | > 0 : queue étirée vers les grandes valeurs |\n"
       "| aplatissement | > 0 : valeurs extrêmes plus fréquentes que dans une loi normale |"),
    code(r'''def stats_detaillees(df, colonnes):  # tableau complet de statistiques descriptives
    num = df[colonnes]  # variables numériques à décrire
    moy = num.mean()  # moyenne de chaque variable
    return pd.DataFrame({  # une ligne par variable, une colonne par statistique
        "N": num.count(),  # nombre de valeurs renseignées
        "manquants": num.isna().sum(),  # nombre de valeurs manquantes
        "moyenne": moy,  # moyenne arithmétique
        "médiane": num.median(),  # valeur centrale
        "écart-type": num.std(),  # dispersion autour de la moyenne
        "CV %": 100 * num.std() / moy.where(moy > 0),  # coefficient de variation (seulement si moyenne > 0)
        "min": num.min(),  # plus petite valeur
        "P5": num.quantile(0.05),  # 5e percentile
        "Q1": num.quantile(0.25),  # premier quartile
        "Q3": num.quantile(0.75),  # troisième quartile
        "IQR": num.quantile(0.75) - num.quantile(0.25),  # écart interquartile
        "P95": num.quantile(0.95),  # 95e percentile
        "P99": num.quantile(0.99),  # 99e percentile
        "max": num.max(),  # plus grande valeur
        "asymétrie": num.skew(),  # coefficient d'asymétrie
        "aplatissement": num.kurt(),  # excès d'aplatissement
    })


base = e3["base"]  # base des polices nettoyée (une ligne par police)
sin = e3["sin"]  # base des sinistres nettoyée (une ligne par sinistre)
VAR_POLICES = ["Age", "Bonus", "Poldur", "Value", "Density", "expo_annees",  # variables clés des polices
               "nb_sinistres", "cout_total", "frequence_annuelle", "cout_moyen_par_sinistre"]  # et leur sinistralité
VAR_POLICES = [v for v in VAR_POLICES if v in base.columns]  # garde seulement celles présentes

st_pol = stats_detaillees(base, VAR_POLICES)  # statistiques des polices
st_sin = stats_detaillees(sin, ["chg_sin"])  # statistiques du montant des sinistres
tableau(st_pol, "Polices", 20)  # affichage
tableau(st_sin, "Sinistres (montant chg_sin)")  # affichage
st_pol.to_csv(A.SORTIE / "stats_detaillees_polices.csv", sep=";", decimal=",")  # sauvegarde pour Excel
st_sin.to_csv(A.SORTIE / "stats_detaillees_sinistres.csv", sep=";", decimal=",")  # (format français)

age = st_pol.loc["Age"]  # statistiques de l'âge
s = st_sin.loc["chg_sin"]  # statistiques des montants de sinistres
nbs = st_pol.loc["nb_sinistres"]  # statistiques du nombre de sinistres par police
asym = st_pol["asymétrie"]  # asymétrie de chaque variable de police
points_cles(  # constats chiffrés
    f"Âge moyen **{age['moyenne']:.1f} ans** (médiane {age['médiane']:.0f}, écart-type {age['écart-type']:.1f}) : "
    f"distribution assez symétrique (asymétrie {age['asymétrie']:.2f}).",
    f"Montant d'un sinistre : moyenne **{nb_fr(s['moyenne'])} €**, médiane **{nb_fr(s['médiane'])} €**, "
    f"P99 {nb_fr(s['P99'])} €, asymétrie {s['asymétrie']:.2f} : quelques gros sinistres tirent la moyenne vers le haut.",
    f"Nombre de sinistres par police : moyenne {nbs['moyenne']:.3f}, variance {nbs['écart-type'] ** 2:.3f}. "
    "Une variance supérieure à la moyenne indique une surdispersion par rapport à une loi de Poisson.",
    f"Variable la plus asymétrique : **{asym.idxmax()}** (asymétrie {asym.max():.1f}).",
)'''),
    md("### Statistiques par catégorie\n"
       "Pour chaque variable qualitative : poids dans le portefeuille, **fréquence** (sinistres par année d'exposition), "
       "**coût moyen** par sinistre et **prime pure** (coût par année d'exposition = fréquence × coût moyen)."),
    code(r'''CATEGORIES = ["Gender", "Type", "Category", "Occupation", "tranche_age", "tranche_anciennete"]  # variables qualitatives
CATEGORIES = [c for c in CATEGORIES if c in base.columns]  # garde seulement celles présentes
prime_globale = base["cout_total"].sum() / base["expo_annees"].sum()  # prime pure moyenne du portefeuille
ecarts = {}  # écart max/min de prime pure par variable

for c in CATEGORIES:  # une table par variable qualitative
    g = base.groupby(c, observed=True)  # regroupe les polices par modalité
    t = pd.DataFrame({  # indicateurs par modalité
        "polices": g.size(),  # nombre de polices
        "part %": 100 * g.size() / len(base),  # part du portefeuille
        "exposition": g["expo_annees"].sum(),  # exposition totale en années
        "sinistres": g["nb_sinistres"].sum(),  # nombre total de sinistres
        "fréquence": g["nb_sinistres"].sum() / g["expo_annees"].sum(),  # sinistres par année d'exposition
        "coût moyen": g["cout_total"].sum() / g["nb_sinistres"].sum().replace(0, np.nan),  # coût par sinistre
        "prime pure": g["cout_total"].sum() / g["expo_annees"].sum(),  # coût par année d'exposition
        "indice prime": 100 * g["cout_total"].sum() / g["expo_annees"].sum() / prime_globale,  # 100 = moyenne
        "âge moyen": g["Age"].mean(),  # âge moyen des assurés
        "âge médian": g["Age"].median(),  # âge médian
        "écart-type âge": g["Age"].std(),  # dispersion des âges
    })
    t_fiable = t[t["polices"] >= 100]  # ignore les modalités trop petites pour l'écart
    ecarts[c] = t_fiable["prime pure"].max() / t_fiable["prime pure"].min()  # rapport prime max / prime min
    tableau(t, c, 20)  # affichage

ecarts = pd.Series(ecarts).sort_values(ascending=False)  # variables triées par pouvoir discriminant
points_cles(  # constats chiffrés
    f"Prime pure moyenne du portefeuille : **{nb_fr(prime_globale, 1)} € par année d'exposition**.",
    "Rapport entre la prime pure la plus haute et la plus basse, par variable : "
    + ", ".join(f"{k} ×{v:.2f}" for k, v in ecarts.items()) + ".",
    f"La variable qui sépare le plus les risques est **{ecarts.index[0]}**.",
    "La colonne *indice prime* se lit par rapport à 100 : 150 veut dire 50 % plus cher que la moyenne.",
)'''),
    md("### Histogrammes\nForme de la distribution des principales variables, avec la moyenne (rouge) et la médiane (vert)."),
    code(r'''VAR_HIST = [v for v in ["Age", "Bonus", "Poldur", "Value", "Density"] if v in base.columns]  # variables des polices
fig, axes = plt.subplots(2, 3, figsize=(14, 7))  # grille de 2 × 3 graphiques
axes = axes.ravel()  # liste simple des 6 emplacements
series = [(v, base[v]) for v in VAR_HIST] + [("Montant sinistre (chg_sin)", sin["chg_sin"])]  # données à tracer
for ax, (nom, x) in zip(axes, series):  # un graphique par variable
    x = x.dropna()  # retire les valeurs manquantes
    ax.hist(x, bins=40, color="#4C72B0", alpha=0.8)  # histogramme en 40 classes
    ax.axvline(x.mean(), color="red", lw=2, label=f"moyenne {x.mean():,.1f}")  # trait de la moyenne
    ax.axvline(x.median(), color="green", lw=2, ls="--", label=f"médiane {x.median():,.1f}")  # trait de la médiane
    ax.set_title(nom)  # titre du graphique
    ax.legend(fontsize=8)  # légende
for ax in axes[len(series):]:  # emplacements inutilisés
    ax.axis("off")  # on les masque
plt.tight_layout()  # ajuste les marges
montrer_fig()  # affiche la figure'''),
]

# ---------------------------------------------------------------- Étape 4
cells += [
    md("## Étape 4 — Jointure des bases\n\n"
       "**Objectif.** Rassembler en une seule base une ligne par police, avec ses caractéristiques, "
       "son exposition et ses sinistres.\n\n"
       "**Méthode.** Trois jointures sur la clé `PolNum`, chacune contrôlée :\n"
       "- portefeuille ↔ exposition (1 à 1) ;\n"
       "- sinistres → polices (plusieurs sinistres pour une police) ;\n"
       "- polices ← sinistres agrégés (0 ou 1 ligne par police).\n\n"
       "**Comment lire.** On vérifie qu'aucune jointure ne crée de doublons et que le coût total est "
       "conservé (somme avant = somme après). Les sinistres « sans police » ne peuvent pas être tarifés et sont exclus."),
    code(r'''res["etape4"] = lancer("Étape 4", lambda: A.etape4(res["etape3"]), "etape4")  # jointure sur la base nettoyée
e4 = res["etape4"]  # raccourci

tableau(e4["controles"], "Contrôles des jointures")  # bilan de chaque jointure
tableau(e4["coherence"], "Contrôles de cohérence")  # conservation des totaux

ok = e4["coherence"]["ok"].all() and (e4["controles"]["resultat"] == "OK").all()  # tous les contrôles passent ?
points_cles(  # constats chiffrés
    f"Base consolidée : **{nb_fr(e4['base_n'])} polices**, dont {nb_fr(e4['sans_sinistre'])} sans sinistre "
    f"({100 * e4['sans_sinistre'] / e4['base_n']:.1f} %).",
    f"{nb_fr(e4['sans_police_lignes'])} lignes de sinistres ({e4['sans_police_polices']} numéros de police) "
    f"n'ont pas de police correspondante, soit {100 * e4['sans_police_lignes'] / e4['sin_consolides_n']:.1f} % "
    "des sinistres : elles sont exclues de la tarification.",
    "Totaux conservés et aucun doublon créé." if ok else "Au moins un contrôle échoue : voir les tableaux.",
)'''),
]

# ---------------------------------------------------------------- Étape 5
cells += [
    md("## Étape 5 — Analyse descriptive et exploratoire\n\n"
       "**Objectif.** Comprendre comment la sinistralité varie selon les caractéristiques des assurés.\n\n"
       "**Méthode.**\n"
       "- *Analyse univariée* : distribution de chaque variable ;\n"
       "- *Analyse bivariée* : fréquence et coût moyen par tranche d'âge, sexe, ancienneté et type ;\n"
       "- *Corrélations de Spearman* : liens monotones entre variables, robustes aux valeurs extrêmes ;\n"
       "- *GLM de Poisson* : effet de chaque facteur sur la fréquence, **toutes choses égales par ailleurs**.\n\n"
       "**Comment lire le GLM.** Le *rapport de taux* (exp du coefficient) se lit par rapport à la modalité de référence : "
       "2,9 veut dire 2,9 fois plus de sinistres. Si l'intervalle de confiance à 95 % ne contient pas 1, l'effet est significatif."),
    code(r'''res["etape5"] = lancer("Étape 5", lambda: A.etape5(res["etape3"]), "etape5")  # analyse exploratoire
e5 = res["etape5"]  # raccourci

tableau(e5["freq_nb"], "Répartition du nombre de sinistres par police")  # combien de polices ont 0, 1, 2… sinistres
tableau(e5["t_age"], "Sinistralité par tranche d'âge")  # fréquence et coût par âge
tableau(e5["t_sexe"], "Sinistralité par sexe")  # fréquence et coût par sexe
tableau(e5["t_anc"], "Sinistralité par ancienneté")  # fréquence et coût par ancienneté
tableau(e5["t_type"], "Sinistralité par type de véhicule")  # fréquence et coût par type
tableau(e5["cible"], "Corrélation (Spearman) avec le nombre de sinistres")  # variables les plus liées aux sinistres
tableau(e5["glm"], "GLM de Poisson : rapports de taux", 20)  # effets ajustés sur la fréquence

ta = e5["t_age"].set_index("tranche_age")  # tableau par âge indexé
ts = e5["t_sexe"].set_index("Gender")  # tableau par sexe indexé
glm = e5["glm"].set_index("facteur")  # GLM indexé par facteur
sig = (e5["glm"]["p_valeur"] < 0.05).sum() - 1  # facteurs significatifs (hors constante)
cible = e5["cible"].iloc[0]  # variable la plus corrélée
points_cles(  # constats chiffrés
    f"{e5['freq_nb'].iloc[0]['pct']:.1f} % des polices n'ont **aucun sinistre** sur la période.",
    f"Les **18-25 ans** ont une fréquence de {ta.loc['18-25', 'frequence_pour_100_ans']:.1f} sinistres pour 100 ans d'exposition, "
    f"contre {ta.loc['36-50', 'frequence_pour_100_ans']:.1f} pour les 36-50 ans "
    f"(×{ta.loc['18-25', 'frequence_pour_100_ans'] / ta.loc['36-50', 'frequence_pour_100_ans']:.1f}).",
    f"Hommes : {ts.loc['Male', 'frequence_pour_100_ans']:.1f} ; femmes : {ts.loc['Female', 'frequence_pour_100_ans']:.1f} sinistres pour 100 ans.",
    f"À caractéristiques égales (GLM), un conducteur de 18-25 ans a **{glm.loc['Âge = 18-25', 'rapport_de_taux']:.2f} fois** "
    "plus de sinistres qu'un 36-50 ans.",
    f"{sig} facteurs du GLM sont significatifs au seuil de 5 %.",
    f"Variable la plus corrélée au nombre de sinistres : **{cible['variable']}** (ρ = {cible['rho_avec_nb_sinistres']:.2f}).",
)'''),
]

# ---------------------------------------------------------------- Étape 6
cells += [
    md("## Étape 6 — Segmentation (ACP, k-means, classification hiérarchique)\n\n"
       "**Objectif.** Regrouper les polices en profils de risque homogènes.\n\n"
       "**Méthode.**\n"
       "- *ACP* (analyse en composantes principales) : résume les 5 variables numériques standardisées "
       "(âge, bonus, ancienneté, valeur, densité) en quelques axes ;\n"
       "- *k-means* : forme k groupes ; k est choisi à l'aide de l'inertie (méthode du coude) et du score de silhouette ;\n"
       "- *Classification hiérarchique* : seconde méthode pour vérifier que les groupes sont stables. "
       "L'accord entre les deux méthodes est mesuré par l'**ARI** (0 = hasard, 1 = groupes identiques).\n\n"
       "**Comment lire.** Une variable avec une charge forte (proche de ±1) sur un axe est bien résumée par cet axe. "
       "Les profils se comparent surtout par leur **fréquence** et leur **charge annuelle**."),
    code(r'''res["etape6"] = lancer("Étape 6", lambda: A.etape6(res["etape3"]), "etape6")  # segmentation
e6 = res["etape6"]  # raccourci

tableau(e6["var"], "ACP : variance expliquée par axe")  # part d'information de chaque axe
tableau(e6["charges"], "ACP : charges des variables sur les deux premiers axes")  # contribution de chaque variable
tableau(e6["choix_k"], "Choix du nombre de groupes (k)")  # inertie et silhouette selon k
tableau(e6["prof"].set_index("profil"), "Profils de risque obtenus")  # description de chaque groupe

pr = e6["prof"].set_index("profil")  # profils indexés par nom
ch = e6["charges"].set_index("variable")  # charges indexées
haut, bas = pr["frequence_pour_100_ans"].idxmax(), pr["frequence_pour_100_ans"].idxmin()  # profils extrêmes
axe1 = ch["CP1"].abs().nlargest(2).index  # deux variables les plus liées à l'axe 1
points_cles(  # constats chiffrés
    f"Les deux premiers axes de l'ACP résument **{e6['var'].iloc[1]['pct_cumule']:.1f} %** de l'information. "
    "C'est peu : les variables sont peu corrélées entre elles, et chacune apporte une information propre.",
    f"Axe 1 : surtout **{axe1[0]}** ({ch.loc[axe1[0], 'CP1']:+.2f}) et **{axe1[1]}** ({ch.loc[axe1[1], 'CP1']:+.2f}) ; "
    f"axe 2 : surtout **{ch['CP2'].abs().idxmax()}**.",
    f"Profil « {haut} » : {pr.loc[haut, 'part_polices']:.1f} % des polices, fréquence "
    f"{pr.loc[haut, 'frequence_pour_100_ans']:.1f} pour 100 ans, âge moyen {pr.loc[haut, 'age']:.0f} ans, bonus moyen {pr.loc[haut, 'bonus']:+.0f}.",
    f"Profil « {bas} » : {pr.loc[bas, 'part_polices']:.1f} % des polices, fréquence "
    f"{pr.loc[bas, 'frequence_pour_100_ans']:.1f} pour 100 ans, soit "
    f"**{pr.loc[haut, 'frequence_pour_100_ans'] / pr.loc[bas, 'frequence_pour_100_ans']:.1f} fois moins** que le profil le plus risqué.",
    f"Accord k-means / hiérarchique : ARI = {e6['ari']:.2f}. C'est un accord modéré : les grandes tendances sont "
    "stables, mais les frontières entre les groupes ne sont pas nettes.",
)'''),
]

# ---------------------------------------------------------------- Étape 7
cells += [
    md("## Étape 7 — Modélisation de la fréquence et du coût\n\n"
       "**Objectif.** Prédire le coût annuel attendu de chaque police (la **prime pure**) "
       "= fréquence attendue × coût moyen attendu.\n\n"
       "**Méthode.**\n"
       "- Séparation **80 % apprentissage / 20 % test** : les modèles sont jugés sur des polices qu'ils n'ont jamais vues ;\n"
       "- *Fréquence* : GLM de Poisson, forêt aléatoire, gradient boosting Poisson, avec l'exposition comme poids ;\n"
       "- *Coût* : GLM Gamma, forêt aléatoire, gradient boosting Gamma ;\n"
       "- Comparaison par **déviance** (plus elle est faible, mieux c'est), en validation croisée et sur le test ; "
       "un modèle constant sert de référence.\n\n"
       "**Comment lire.**\n"
       "- *surapprentissage %* : écart entre la déviance sur le test et sur l'apprentissage ; s'il est élevé, le modèle "
       "« apprend par cœur » ;\n"
       "- *ae_test* : sinistres totaux prédits / observés ; un modèle bien calibré est proche de 1 ;\n"
       "- *Déciles* : on classe les polices de la moins à la plus risquée selon le modèle ; un bon modèle a un coût réel "
       "qui croît d'un décile à l'autre."),
    code(r'''res["etape7"] = lancer("Étape 7", lambda: A.etape7(res["etape3"]), "etape7")  # modèles prédictifs
e7 = res["etape7"]  # raccourci

tableau(e7["split"], "Découpage apprentissage / test")  # taille des échantillons
tableau(e7["res_f"], "Modèles de fréquence (déviance de Poisson)")  # comparaison des modèles de fréquence
tableau(e7["res_s"], "Modèles de coût (déviance Gamma)")  # comparaison des modèles de coût
tableau(e7["total"], "Coût total : modèle retenu contre référence")  # performance de la prime pure
tableau(e7["deciles"], "Coût prédit et réel par décile de risque")  # pouvoir de segmentation
tableau(e7["imp_f"], "Importance des variables : fréquence")  # variables les plus utiles (fréquence)
tableau(e7["imp_s"], "Importance des variables : coût")  # variables les plus utiles (coût)
tableau(e7["glm_rr"], "GLM de fréquence : rapports de taux", 15)  # effet de chaque facteur

rf = e7["res_f"].set_index("modele")  # résultats fréquence indexés
dec = e7["deciles"]  # tableau des déciles
tot = e7["total"]  # comparaison des approches
points_cles(  # constats chiffrés
    f"Fréquence : le modèle retenu est **{e7['best_f']}**. Sa déviance de test est {rf.loc[e7['best_f'], 'deviance_test']:.3f}, "
    f"contre {rf.loc['Constante (référence)', 'deviance_test']:.3f} pour la référence (**gain {e7['gain_f']:.1f} %**). "
    f"Son surapprentissage est de {rf.loc[e7['best_f'], 'surapprentissage_pct']:.1f} %.",
    f"Coût : le modèle retenu est **{e7['best_s']}**. Son gain n'est que de {e7['gain_s']:.1f} % : le montant d'un sinistre "
    "est beaucoup plus difficile à prévoir que leur nombre.",
    f"Calibrage : prédit / observé = **{tot.loc[0, 'ae_total']:.3f}** sur le test (idéal = 1).",
    f"Segmentation : les 10 % de polices jugées les plus risquées représentent **{tot.loc[0, 'part_cout_top10']:.1f} %** du coût réel ; "
    f"le coût réel moyen passe de {nb_fr(dec.iloc[0]['cout_reel_moyen'])} € (décile 1) à {nb_fr(dec.iloc[-1]['cout_reel_moyen'])} € (décile 10).",
    f"Variables les plus utiles pour la fréquence : {', '.join(e7['imp_f']['libelle'].head(3))}.",
)'''),
]

cells += VIZ

# ---------------------------------------------------------------- Étape 8
cells += [
    md("## Étape 8 — Interprétation actuarielle\n\n"
       "**Objectif.** Traduire les résultats en constats utiles pour la tarification et la réassurance.\n\n"
       "**Méthode.**\n"
       "- *Concentration des coûts* : quelle part du coût total vient des 1 %, 5 %, 10 % et 20 % de sinistres les plus chers ;\n"
       "- *Polices atypiques* : polices de l'échantillon test ayant **au moins 2 sinistres** alors que, selon le modèle de "
       "fréquence, la probabilité d'en avoir autant est **inférieure à 0,1 %** (queue de la loi de Poisson).\n\n"
       "**Comment lire.** Une forte concentration justifie un écrêtement des gros sinistres pour la tarification, "
       "ou une couverture de réassurance en excédent de sinistre."),
    code(r'''res["etape8"] = lancer("Étape 8", lambda: A.etape8(res), "etape8")  # interprétation finale
e8 = res["etape8"]  # raccourci

tableau(e8["conc"], "Concentration du coût sur les plus gros sinistres")  # courbe de concentration
atyp = pd.Series(e8["atyp"], name="valeur").to_frame()  # synthèse des polices atypiques
tableau(atyp, "Polices atypiques : synthèse")  # affichage
liste = pd.read_csv(A.SORTIE / "etape8_polices_atypiques.csv", sep=";", decimal=",")  # liste détaillée des polices
tableau(liste, "Polices atypiques : liste (de la plus improbable à la moins improbable)", 20)  # affichage

c = e8["conc"].set_index("part_des_sinistres_pct")["part_du_cout_pct"]  # part du coût selon la part des sinistres
a = e8["atyp"]  # dictionnaire des indicateurs atypiques
points_cles(  # constats chiffrés
    f"Les 1 % de sinistres les plus chers pèsent **{c.loc[1.0]:.1f} %** du coût, les 10 % les plus chers "
    f"**{c.loc[10.0]:.1f} %**, et les 20 % les plus chers **{c.loc[20.0]:.1f} %**.",
    f"{a['n_atyp']} polices atypiques sur {nb_fr(a['n_test'])} ({a['part_polices']:.2f} % des polices) : "
    f"{a['sinistres_observes']:.0f} sinistres observés pour {a['sinistres_attendus']:.1f} attendus. Elles représentent "
    f"{a['part_sinistres']:.1f} % des sinistres et {a['part_cout']:.1f} % du coût de l'échantillon test.",
    "Recommandation : suivre ces polices individuellement (vérification, revue de souscription) "
    "plutôt que de les laisser peser sur le tarif de tous.",
)'''),
]

cells += TARIF

# ---------------------------------------------------------------- Synthèse et rapport
cells += [
    md("## Synthèse générale\nLes chiffres essentiels de toute l'analyse, puis le temps passé sur chaque étape."),
    code(r'''e5, e6, e7, e8 = res["etape5"], res["etape6"], res["etape7"], res["etape8"]  # raccourcis
pr = e6["prof"].set_index("profil")  # profils de risque
synthese = pd.DataFrame({  # tableau récapitulatif
    "Indicateur": [  # libellés
        "Polices analysées",  # taille de la base
        "Polices sans sinistre",  # part des polices sans sinistre
        "Fréquence moyenne (sinistres / an)",  # fréquence globale
        "Coût moyen d'un sinistre",  # sévérité globale
        "Prime pure moyenne (€ / an)",  # coût annuel moyen par police
        "Écart de fréquence entre le profil le plus et le moins risqué",  # pouvoir de la segmentation
        "Modèle de fréquence retenu",  # meilleur modèle de fréquence
        "Modèle de coût retenu",  # meilleur modèle de coût
        "Prédit / observé sur le test (idéal = 1)",  # calibrage
        "Part du coût due au 1 % des plus gros sinistres",  # concentration
    ],
    "Valeur": [  # valeurs correspondantes
        nb_fr(res["etape3"]["base_n"]),  # nombre de polices
        f"{e5['freq_nb'].iloc[0]['pct']:.1f} %",  # part sans sinistre
        f"{base['nb_sinistres'].sum() / base['expo_annees'].sum():.3f}",  # fréquence annuelle
        f"{nb_fr(sin['chg_sin'].mean())} €",  # coût moyen
        f"{nb_fr(prime_globale, 1)} €",  # prime pure
        f"×{pr['frequence_pour_100_ans'].max() / pr['frequence_pour_100_ans'].min():.1f}",  # rapport des fréquences
        e7["best_f"],  # nom du modèle de fréquence
        e7["best_s"],  # nom du modèle de coût
        f"{e7['total'].loc[0, 'ae_total']:.3f}",  # calibrage
        f"{e8['conc'].iloc[0]['part_du_cout_pct']:.1f} %",  # concentration
    ],
}).set_index("Indicateur")  # indicateurs en index pour un affichage plus propre
synthese["Valeur"] = synthese["Valeur"].map(virgule)  # virgule décimale à la française
display(synthese)  # affiche le tableau

d = pd.Series(duree, name="secondes")  # durées des étapes
ax = d.plot.barh(figsize=(7, 3), color="#4C72B0", title="Temps d'exécution par étape (s)")  # graphique des durées
ax.bar_label(ax.containers[0], labels=[f"{v:.1f} s ({100 * v / d.sum():.0f} %)".replace(".", ",") for v in d],  # durée et part
             padding=2, fontsize=7)
ax.margins(x=0.2)  # place pour les étiquettes
plt.gca().invert_yaxis()  # première étape en haut
plt.tight_layout()  # ajuste les marges
montrer_fig()  # affiche la figure'''),
    md("## Rapport PDF\nGénère le rapport LaTeX complet et le compile en PDF (nécessite `pdflatex`)."),
    code(r'''A.generer_rapport(res)  # écrit rapport_analyse.tex puis le compile en PDF
pdf = A.FICHIER_TEX.with_suffix(".pdf")  # chemin du PDF produit
print("Rapport :", pdf if pdf.exists() else "PDF non produit (pdflatex manquant ?)")  # indique où le trouver'''),
]

cells += CONCLUSION

# ================================================================ fusion : tout le code dans le notebook
import re

SRC = open(ICI / "analyse_actuarielle.py", encoding="utf-8").read().split("\n")


def debut_section(titre):  # indice de la ligne « # ==== » qui ouvre la section
    i = next(k for k, l in enumerate(SRC) if l.startswith("# " + titre))
    return i - 1


def lignes(a, b):  # lignes a (incluse) à b (exclue), sans lignes vides en bord
    return "\n".join(SRC[a:b]).strip("\n")


def bornes_def(nom, suivant):  # de « def nom » jusqu'à « def suivant »
    a = next(k for k, l in enumerate(SRC) if l.startswith(f"def {nom}("))
    b = next(k for k, l in enumerate(SRC) if l.startswith(f"def {suivant}("))
    return lignes(a, b)


TITRES = ["CONFIGURATION", "OUTILS LATEX", "ÉTAPE 1", "ÉTAPE 2", "ÉTAPE 3", "ÉTAPE 4", "ÉTAPE 5",
          "ÉTAPE 6", "ÉTAPE 7", "ÉTAPE 8", "PROGRAMME PRINCIPAL"]
D = {t: debut_section(t) for t in TITRES}
SUITE = dict(zip(TITRES, TITRES[1:]))
premiere_import = next(k for k, l in enumerate(SRC) if l.startswith("import "))
config = lignes(premiere_import, D["OUTILS LATEX"])
config = config.replace('matplotlib.use("Agg")  # figures produites sans fenêtre graphique\n', "")
ancien = 'DOSSIER = Path(__file__).resolve().parent.parent  # racine du projet (ce script est dans code/)'
assert ancien in config, "chemin DOSSIER introuvable"
config = config.replace(ancien, 'DOSSIER = Path.cwd()  # dossier du notebook : fonctionne sur tous les ordinateurs')
config = config.replace('DOSSIER = Path.cwd()  # dossier du notebook',
                        'DOSSIER = Path.cwd()  # dossier du notebook = racine du projet')
config += "\n\n\n" + bornes_def("preparer_dossiers", "main")
CODE = {
    "config": config,
    "latex": lignes(D["OUTILS LATEX"], D["ÉTAPE 1"]),
    "e12": lignes(D["ÉTAPE 1"], D["ÉTAPE 3"]) + "\n\n\n" + bornes_def("etape1_et_2", "generer_rapport"),
    "e3": lignes(D["ÉTAPE 3"], D["ÉTAPE 4"]),
    "e4": lignes(D["ÉTAPE 4"], D["ÉTAPE 5"]),
    "e5": lignes(D["ÉTAPE 5"], D["ÉTAPE 6"]),
    "e6": lignes(D["ÉTAPE 6"], D["ÉTAPE 7"]),
    "e7": lignes(D["ÉTAPE 7"], D["ÉTAPE 8"]),
    "e8": lignes(D["ÉTAPE 8"], D["PROGRAMME PRINCIPAL"]),
    "rapport": bornes_def("generer_rapport", "preparer_dossiers"),
}

# 1) les cellules du notebook n'utilisent plus le module « A » ; tableau() devient afficher()
for c in cells:
    src = c.source
    src = re.sub(r'(?<![\w."])A\.(?=[A-Za-z_])', "", src)
    src = re.sub(r"\btableau\(", "afficher(", src)
    if c.cell_type == "code" and "spec_from_file_location" in src:  # cellule Préparation
        garde = [l for l in src.split("\n")
                 if not re.match(r"(import importlib\.util|SCRIPT = |spec = |A = |spec\.loader)", l)]
        src = "\n".join(garde).replace(
            'print("Prêt :", SCRIPT.name, "chargé")  # confirme que tout est en place',
            'print("Prêt : données trouvées, fonctions chargées")  # confirme que tout est en place')
    src = src.replace(
        "Ce notebook exécute le script `Untitled-1.py` **une étape à la fois**.",
        "Ce notebook contient **tout le code** de l'analyse et l'exécute **une étape à la fois**.")
    src = src.replace(
        "> Lancez toujours la cellule **Préparation** en premier, puis les étapes dans l'ordre (ou « Run All »). "
        "Si vous modifiez `Untitled-1.py`, relancez la cellule Préparation.",
        "> Le notebook a **deux parties**. La **partie A — Code** définit toutes les fonctions et ne calcule rien. "
        "La **partie B — Résultats** exécute les étapes, affiche les tableaux et graphiques, et les interprète. "
        "Exécutez la partie A en entier, puis la partie B dans l'ordre (ou simplement « Run All »). "
        "Si vous modifiez une fonction de la partie A, relancez sa cellule, puis l'étape concernée de la partie B.")
    src = src.replace("| Étape | Contenu |\n|---|---|\n",
                      "| Partie / étape | Contenu |\n|---|---|\n"
                      "| **A** | **Code de l'analyse** : configuration, outils, fonctions de chaque étape |\n"
                      "| **B** | **Résultats et interprétations** : |\n")
    c.source = src

# 2) découpage : en-tête (titre, sommaire, consignes) | partie A (code) | partie B (résultats)
i_prep_md = next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## Préparation"))
entete, prep_code, suite = cells[:i_prep_md], cells[i_prep_md + 1], cells[i_prep_md + 2:]
assert prep_code.cell_type == "code" and "def montrer_fig" in prep_code.source

# la fonction stats_detaillees (étape 3 bis) rejoint la partie A ; le calcul reste dans la partie B
c3b = next(c for c in suite if c.cell_type == "code" and c.source.startswith("def stats_detaillees"))
def_stats, calcul_stats = c3b.source.split("\n\n\nbase = e3", 1)
c3b.source = "base = e3" + calcul_stats
CODE["stats"] = def_stats


def demoter(c):  # descend d'un niveau les titres Markdown de la partie B (## → ###)
    if c.cell_type == "markdown":
        c.source = re.sub(r"(?m)^(#{2,5}) ", r"#\1 ", c.source)
    return c


PARTIE_A = [  # (titre, clé du code) dans l'ordre d'exécution
    ("A.1 Configuration : bibliothèques, chemins et constantes", "config"),
    ("A.2 Outils de mise en forme du rapport LaTeX", "latex"),
    ("A.3 Fonctions d'affichage du notebook et vérification des données", None),
    ("A.4 Étapes 1 et 2 : chargement, dictionnaire, audit de qualité", "e12"),
    ("A.5 Étape 3 : nettoyage et préparation", "e3"),
    ("A.6 Étape 3 bis : statistiques détaillées", "stats"),
    ("A.7 Étape 4 : jointures et contrôles", "e4"),
    ("A.8 Étape 5 : analyse descriptive et GLM de Poisson", "e5"),
    ("A.9 Étape 6 : ACP, k-means, classification hiérarchique", "e6"),
    ("A.10 Étape 7 : modèles de fréquence et de coût", "e7"),
    ("A.11 Étape 8 : concentration des coûts et polices atypiques", "e8"),
    ("A.12 Génération du rapport LaTeX / PDF", "rapport"),
]
partie_a = [md("## Partie A — Code de l'analyse\n\n"
               "Cette partie **définit** toutes les fonctions utilisées ensuite (chargement, audit, nettoyage, "
               "jointures, statistiques, segmentation, modèles, rapport). Elle ne lance **aucun calcul** : "
               "son exécution prend quelques secondes.\n\n"
               "Les étapes 7 bis (visualisations avancées) et 9 (tarification) sont écrites directement dans la "
               "partie B, car leur code *est* l'analyse : chaque ligne y est commentée.")]
for titre, cle in PARTIE_A:
    partie_a.append(md(f"### {titre}"))
    partie_a.append(prep_code if cle is None else code(CODE[cle]))
partie_b = [md("## Partie B — Résultats et interprétations\n\n"
               "Chaque étape suit le même plan : **objectif, méthode et lecture** (texte), **calcul** "
               "(tableaux et graphiques), puis **points clés** (constats chiffrés calculés automatiquement).")]
partie_b += [demoter(c) for c in suite]
cells = entete + partie_a + partie_b

assert not any(c.cell_type == "code" and re.search(r'(?<![\w."])A\.(?=[A-Za-z_])', c.source)
               and "def etape" not in c.source and "DOSSIER = " not in c.source
               for c in partie_b), "référence A. restante"
assert sum(1 for c in partie_a if c.cell_type == "code") == len(PARTIE_A)
assert not any(c.cell_type == "code" and c.source.lstrip().startswith(("def ", "import ")) for c in partie_b
               if "partial_dependence" not in c.source and "GammaRegressor" not in c.source), "fonction restée en partie B"

nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": DOSSIER}}).execute()
nbf.write(nb, f"{DOSSIER}/analyse_resultats.ipynb")
print("ok")
