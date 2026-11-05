# ==============================================================================
# Projet  : Exploitation de bases de données d'assurance automobile
#           Audit, analyse, modélisation et tarification
# Auteur  : Morel POUNIGUIA
# Fichier : analyse_actuarielle.py
# ==============================================================================

"""
Analyse actuarielle de trois bases d'assurance : sinistres, portefeuille et expositions.

Étapes :
    1. Compréhension des bases de données
    2. Audit et qualité des données
    3. Nettoyage et préparation
    4. Jointure des bases
    5. Analyse descriptive et exploratoire
    6. Segmentation et réduction de dimension (ACP, K-means, classification hiérarchique)
    7. Modélisation (GLM Poisson et Gamma, forêt aléatoire, gradient boosting)
    8. Interprétation actuarielle et recommandations

Version « script » du notebook analyse_resultats.ipynb (même code, sans affichage interactif).

Utilisation :
    1. Les données sont dans le dossier « donnees » du projet :
           donnees/data_sin.csv
           donnees/data_ptf_Eco.xlsx
    2. Installer les dépendances :  pip install pandas numpy scipy scikit-learn matplotlib openpyxl
    3. Lancer ce fichier depuis n'importe où :  python code/analyse_actuarielle.py

Résultats :
    - resultats/tableaux/ : tableaux CSV de chaque étape
    - resultats/figures/  : graphiques PNG
    - rapport/            : rapport_analyse.tex et rapport_analyse.pdf (le PDF nécessite pdflatex)
"""

import datetime
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures produites sans fenêtre graphique
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats as sstats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error, mean_gamma_deviance,
                             mean_poisson_deviance, silhouette_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DOSSIER = Path(__file__).resolve().parent.parent  # racine du projet (ce script est dans code/)
DONNEES = DOSSIER / "donnees"  # fichiers d'entrée
FICHIER_CSV = DONNEES / "data_sin.csv"
FICHIER_XLSX = DONNEES / "data_ptf_Eco.xlsx"
SORTIE = DOSSIER / "resultats" / "tableaux"  # tableaux CSV
FIG = DOSSIER / "resultats" / "figures"  # graphiques PNG
FICHIER_TEX = DOSSIER / "rapport" / "rapport_analyse.tex"  # rapport LaTeX (et PDF)

GRAINE = 42  # graine commune à toutes les étapes aléatoires
SEUIL_MODALITES = 20
CODES_TEXTE_MANQUANTS = {"", "??", "???", "NA", "N/A", "null", "NULL", "-"}



SIGNIFICATIONS = {
    ("sinistres", "nb_sin"): "Nombre de sinistres déclarés sur la police",
    ("sinistres", "chg_sin"): "Coût total des sinistres (charge), en euros (hypothèse)",
    ("sinistres", "PolNum"): "Identifiant de la police (clé de jointure)",
    ("portefeuille", "PolNum"): "Identifiant de la police (clé de jointure)",
    ("portefeuille", "CalYear"): "Année calendaire de l'observation (2009 ou 2010)",
    ("portefeuille", "Gender"): "Sexe du conducteur ; codes mixtes Male/Female et H/F/h",
    ("portefeuille", "Type"): "Type de véhicule, codes A à F (hypothèse)",
    ("portefeuille", "Category"): "Catégorie de taille (Small/Medium/Large), '???' = inconnu",
    ("portefeuille", "Occupation"): "Catégorie socio-professionnelle du conducteur",
    ("portefeuille", "Age"): "Âge du conducteur en années (valeurs aberrantes présentes)",
    ("portefeuille", "Group1"): "Zone géographique, codes 4 à 20 (hypothèse)",
    ("portefeuille", "Bonus"): "Coefficient bonus-malus, de -50 à 150 (hypothèse)",
    ("portefeuille", "Poldur"): "Ancienneté de la police en années (valeurs négatives présentes)",
    ("portefeuille", "Value"): "Valeur du véhicule en euros (hypothèse) ; valeurs manquantes et '??'",
    ("portefeuille", "Adind"): "Indicateur binaire (0/1), signification à confirmer",
    ("portefeuille", "SubGroup2"): "Sous-groupe géographique, très peu renseigné (88 % de manquants)",
    ("portefeuille", "Group2"): "Groupe géographique à 10 modalités (hypothèse)",
    ("portefeuille", "Density"): "Densité de population de la zone (hypothèse)",
    ("exposition", "PolNum"): "Identifiant de la police (clé de jointure)",
    ("exposition", "Expdays"): "Nombre de jours d'exposition au risque sur l'année (91 à 365)",
}


# ===========================================================================
# OUTILS LATEX ET MISE EN FORME DU RAPPORT
# ===========================================================================


CARACTERES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "∩": r"$\cap$", "–": r"--", "—": r"---", "−": r"$-$", "×": r"$\times$",
    "€": r"\texteuro{}", "«": r"\guillemotleft{}", "»": r"\guillemotright{}",
    "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$",
    "↔": r"$\leftrightarrow$", "←": r"$\leftarrow$",
}


LITTERAL = {
    "é": r"\'e", "è": r"\`e", "ê": r"\^e", "à": r"\`a", "â": r"\^a", "ç": r"\c{c}",
    "ô": r"\^o", "û": r"\^u", "î": r"\^i", "É": r"\'E", "Â": r"\^A",
    "–": "--", "—": "---", "−": r"$-$", "×": r"$\times$", "∩": r"$\cap$",
    "€": r"\texteuro{}", "«": r"\guillemotleft{}", "»": r"\guillemotright{}",
    "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$",
    "À": r"\`A",
    "œ": r"\oe{}", "Œ": r"\OE{}",
    "↔": r"$\leftrightarrow$", "←": r"$\leftarrow$",
}


def echapper(texte) -> str:
    return "".join(CARACTERES.get(c, c) for c in str(texte))


def nombre(x, decimales=0) -> str:
    if pd.isna(x):
        return "--"
    s = f"{x:,.{decimales}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", " ")


def tableau(caption: str, label: str, entetes: list, lignes: list, specs: str) -> str:
    entete = " & ".join(r"\textbf{" + (e if "\\" in e else echapper(e)) + "}" for e in entetes) + r" \\"
    sortie = [
        r"\begin{longtable}{" + specs + "}",
        r"\caption{" + echapper(caption) + r"} \label{" + label + r"} \\",
        r"\toprule", entete, r"\midrule", r"\endfirsthead",
        r"\toprule", entete, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    sortie += [" & ".join(str(c) for c in l) + r" \\" for l in lignes]
    sortie += [r"\bottomrule", r"\end{longtable}", ""]
    return "\n".join(sortie)


def literate() -> str:
    return "literate=" + " ".join("{" + c + "}{{" + v + "}}1" for c, v in LITTERAL.items()) + ","


PREAMBULE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{caption}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{graphicx}

\renewcommand{\contentsname}{Table des matières}
\renewcommand{\tablename}{Tableau}
\renewcommand{\listtablename}{Liste des tableaux}
\renewcommand{\appendixname}{Annexe}
\renewcommand{\refname}{Références}
\newcommand{\dateFR}{__DATE_FR__}

\lstset{
  basicstyle=\ttfamily\footnotesize, breaklines=true, frame=single,
  numbers=left, numberstyle=\tiny\color{gray},
  keywordstyle=\color{blue!60!black}\bfseries,
  commentstyle=\color{green!40!black}\itshape,
  stringstyle=\color{red!50!black}, showstringspaces=false,
  __LITERATE__
  inputencoding=utf8
}

\title{Exploitation de bases de données d'assurance automobile\\[0.3em]\large Audit, analyse, modélisation et tarification}
\author{Morel POUNIGUIA}
\date{\dateFR}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""


FIN = "\n\\end{document}\n"


def var_tex(nom) -> str:
    """Nom de variable en \\texttt avec retour possible après chaque underscore."""
    return "\\_\\allowbreak{}".join(echapper(m) for m in str(nom).split("_"))


# ===========================================================================
# ÉTAPE 1 – COMPRÉHENSION DES BASES DE DONNÉES
# ===========================================================================


def charger_csv_brut() -> pd.DataFrame:
    """Sinistres en texte brut (séparateur ';', pas de conversion automatique)."""
    return pd.read_csv(FICHIER_CSV, sep=";", dtype=str, keep_default_na=False)


def charger_excel() -> dict:
    """Feuilles Excel avec leurs valeurs en cache (data_only)."""
    feuilles = {}
    for ws in load_workbook(FICHIER_XLSX, read_only=True, data_only=True).worksheets:
        rows = list(ws.iter_rows(values_only=True))
        feuilles[ws.title] = pd.DataFrame(rows[1:], columns=rows[0])
    return feuilles


def typer_sinistres(brut: pd.DataFrame) -> pd.DataFrame:
    """Convertit le CSV brut en types numériques (décimale en virgule)."""
    df = brut.copy()
    df["chg_sin"] = pd.to_numeric(df["chg_sin"].str.replace(",", ".", regex=False), errors="coerce")
    df["nb_sin"] = pd.to_numeric(df["nb_sin"], errors="coerce")
    df["PolNum"] = pd.to_numeric(df["PolNum"], errors="coerce")
    return df


def typer_portefeuille(ptf: pd.DataFrame) -> pd.DataFrame:
    df = ptf.copy()
    df["Value_num"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
    return df


def recenser_formules() -> pd.DataFrame:
    """Cellules contenant une formule au lieu d'une valeur, par feuille et par colonne."""
    lignes = []
    for ws in load_workbook(FICHIER_XLSX, read_only=True).worksheets:
        rows = ws.iter_rows(values_only=True)
        entete = next(rows)
        compte = {h: 0 for h in entete}
        exemple = {}
        for r in rows:
            for h, v in zip(entete, r):
                if isinstance(v, str) and v.startswith("="):
                    compte[h] += 1
                    exemple.setdefault(h, v)
        for h, n in compte.items():
            if n:
                lignes.append({"feuille": ws.title, "variable": h, "nb_formules": n,
                               "exemple": exemple[h][:120]})
    return pd.DataFrame(lignes)


def classer_variable(serie: pd.Series) -> str:
    """Retourne 'identifiant', 'qualitative' ou 'quantitative'."""
    if serie.name == "PolNum":
        return "identifiant"
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        if serie.nunique(dropna=True) <= SEUIL_MODALITES:
            return "qualitative"
        return "quantitative"
    return "qualitative"


def dictionnaire(tables: dict) -> pd.DataFrame:
    lignes = []
    for nom, df in tables.items():
        for col in df.columns:
            s = df[col]
            lignes.append({
                "table": nom, "variable": col, "type_pandas": str(s.dtype),
                "nature": classer_variable(s),
                "nb_manquants": int(s.isna().sum()),
                "pct_manquants": round(100 * s.isna().mean(), 2),
                "nb_valeurs_distinctes": int(s.nunique(dropna=True)),
                "signification": SIGNIFICATIONS.get((nom, col), ""),
            })
    return pd.DataFrame(lignes)


def statistiques_quantitatives(dico: pd.DataFrame, tables: dict) -> pd.DataFrame:
    morceaux = []
    for nom, df in tables.items():
        cols = dico[(dico["table"] == nom) & (dico["nature"] == "quantitative")]["variable"]
        if len(cols):
            desc = df[cols].describe().T.round(3)
            desc.insert(0, "table", nom)
            morceaux.append(desc.reset_index().rename(columns={"index": "variable"}))
    return pd.concat(morceaux, ignore_index=True)


def cles_jointure(tables: dict) -> dict:
    """Unicité et recouvrement de PolNum."""
    resultat = {}
    ensembles = {}
    for nom, df in tables.items():
        s = df["PolNum"]
        resultat[nom] = {
            "nb_lignes": int(len(s)), "nb_PolNum_distincts": int(s.nunique()),
            "nb_PolNum_doublons_lignes": int(s.duplicated().sum()),
            "nb_lignes_identiques_completes": int(df.duplicated().sum()),
            "nb_PolNum_manquants": int(s.isna().sum()),
        }
        ensembles[nom] = set(s.dropna())
    noms = list(ensembles)
    recouvrement = {}
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            inter = ensembles[a] & ensembles[b]
            recouvrement[f"{a} ∩ {b}"] = {
                "nb_communs": len(inter),
                "pct_de_" + a: round(100 * len(inter) / len(ensembles[a]), 2),
                "pct_de_" + b: round(100 * len(inter) / len(ensembles[b]), 2),
            }
    resultat["recouvrement"] = recouvrement
    sin_ids = ensembles["sinistres"]
    lignes_sans = int((~tables["sinistres"]["PolNum"].isin(ensembles["portefeuille"])).sum())
    resultat["sinistres_sans_police"] = {
        "nb_PolNum_distincts_sans_correspondance": len(sin_ids - ensembles["portefeuille"]),
        "nb_lignes_sinistres_concernees": lignes_sans,
    }
    return resultat


def anomalies_etape1(tables: dict) -> pd.DataFrame:
    p, e, s = tables["portefeuille"], tables["exposition"], tables["sinistres"]
    regles = [
        ("portefeuille", "Gender", "Codes non normalisés (H, F, h)", p["Gender"].isin(["H", "F", "h"])),
        ("portefeuille", "Gender", "Valeur manquante", p["Gender"].isna()),
        ("portefeuille", "Category", "Modalité '???' (inconnue)", p["Category"].eq("???")),
        ("portefeuille", "Value", "Valeur manquante", p["Value"].isna()),
        ("portefeuille", "Value", "Valeur non numérique '??'",
         pd.to_numeric(p["Value"], errors="coerce").isna() & p["Value"].notna()),
        ("portefeuille", "Age", "Âge < 18 ans", p["Age"] < 18),
        ("portefeuille", "Age", "Âge > 100 ans (aberrant)", p["Age"] > 100),
        ("portefeuille", "Poldur", "Ancienneté négative", p["Poldur"] < 0),
        ("portefeuille", "Poldur", "Ancienneté > âge du conducteur", p["Poldur"] > p["Age"]),
        ("portefeuille", "SubGroup2", "Valeur manquante (taux élevé)", p["SubGroup2"].isna()),
        ("exposition", "Expdays", "Exposition > 365 jours", e["Expdays"] > 365),
        ("sinistres", "chg_sin", "Coût nul", s["chg_sin"] == 0),
        ("sinistres", "PolNum", "Police sans correspondance dans le portefeuille", ~s["PolNum"].isin(p["PolNum"])),
    ]
    return pd.DataFrame([
        {"table": t, "variable": v, "anomalie": libelle, "nb_lignes": int(m.sum()),
         "pct": round(100 * m.mean(), 3)}
        for t, v, libelle, m in regles
    ])


def s_sources(res) -> str:
    noms = {"sinistres": "Sinistres (CSV)", "portefeuille": "Portefeuille PTF (Excel)",
            "exposition": "Exposition Expo (Excel)"}
    lignes = [[echapper(noms[k]), nombre(v["nb_lignes"]), nombre(v["nb_colonnes"])]
              for k, v in res["tables"].items()]
    return ("\\section{Sources de données}\n"
            "Trois jeux de données sont fournis. Le CSV utilise le séparateur \\texttt{;} et la virgule décimale ; "
            "le fichier Excel contient deux feuilles.\n\n"
            + tableau("Dimensions des tables importées", "tab:dimensions", ["Table", "Lignes", "Colonnes"], lignes, "lrr"))


def s_dictionnaire(dico) -> str:
    lignes = [[echapper(r.table), r"\texttt{" + echapper(r.variable) + "}", echapper(r.type_pandas),
               echapper(r.nature), nombre(r.pct_manquants, 2), echapper(r.signification)]
              for r in dico.itertuples()]
    return ("\\section{Dictionnaire des variables}\n"
            "Chaque variable est décrite par son type, sa nature et sa signification. Les significations sont "
            "\\emph{déduites} du nom et des valeurs : elles doivent être validées auprès du producteur.\n\n"
            + tableau("Dictionnaire des variables", "tab:dictionnaire",
                      ["Table", "Variable", "Type", "Nature", "\\% manquants", "Signification"],
                      lignes, "llllrp{6.2cm}"))


def s_nature(dico) -> str:
    def liste(nature):
        sub = dico[dico["nature"] == nature]
        return ", \\allowbreak ".join(
            r"\texttt{" + echapper(r.table) + ".\\allowbreak{}" + echapper(r.variable) + "}" for r in sub.itertuples()
        ) or "aucune"
    return ("\\section{Variables quantitatives et qualitatives}\n\\begin{itemize}\n"
            f"  \\item \\textbf{{Quantitatives}} ({(dico['nature'] == 'quantitative').sum()}) : {liste('quantitative')}.\n"
            f"  \\item \\textbf{{Qualitatives}} ({(dico['nature'] == 'qualitative').sum()}) : {liste('qualitative')}.\n"
            f"  \\item \\textbf{{Identifiants}} ({(dico['nature'] == 'identifiant').sum()}) : {liste('identifiant')}.\n"
            "\\end{itemize}\n"
            "Une variable numérique comptant au plus 20 valeurs distinctes est traitée comme qualitative. "
            "\\textbf{Cas particulier} : \\texttt{portefeuille.Value} est lue comme texte à cause de deux valeurs "
            "\\texttt{??} ; c'est une variable quantitative par nature.\n")


def s_statistiques(stats) -> str:
    lignes = [[echapper(r["table"]), r"\texttt{" + echapper(r["variable"]) + "}", nombre(r["count"]),
               nombre(r["mean"], 2), nombre(r["std"], 2), nombre(r["min"], 2), nombre(r["50%"], 2),
               nombre(r["max"], 2)]
              for r in stats.to_dict("records")]
    return ("\\section{Statistiques descriptives des variables quantitatives}\n"
            "Statistiques calculées sur les valeurs non manquantes.\n\n"
            + tableau("Statistiques descriptives", "tab:stats",
                      ["Table", "Variable", "N", "Moyenne", "Écart-type", "Min", "Médiane", "Max"],
                      lignes, "llrrrrrr"))


def s_anomalies_e1(an) -> str:
    lignes = [[echapper(r.table), r"\texttt{" + echapper(r.variable) + "}", echapper(r.anomalie),
               nombre(r.nb_lignes), nombre(r.pct, 3)] for r in an.itertuples()]
    return ("\\section{Contrôle de qualité de l'étape 1}\n"
            + tableau("Anomalies détectées à l'étape 1", "tab:anomalies_e1",
                      ["Table", "Variable", "Anomalie", "Lignes", "\\% des lignes"], lignes, "llp{6cm}rr"))


def s_cles(cles) -> str:
    lignes = [[echapper(n), nombre(cles[n]["nb_lignes"]), nombre(cles[n]["nb_PolNum_distincts"]),
               nombre(cles[n]["nb_PolNum_doublons_lignes"]), nombre(cles[n]["nb_lignes_identiques_completes"])]
              for n in ["sinistres", "portefeuille", "exposition"]]
    rec = [[echapper(k), nombre(v["nb_communs"]),
            " ; ".join(f"{kk.replace('pct_de_', '')} : {nombre(vv, 2)} \\%" for kk, vv in v.items() if kk.startswith("pct_de_"))]
           for k, v in cles["recouvrement"].items()]
    sans = cles["sinistres_sans_police"]
    return ("\\section{Clés de jointure}\n"
            "La clé candidate est \\texttt{PolNum}, présente dans les trois tables sans valeur manquante. "
            "Elle n'est pas unique, ce qui impose un traitement avant jointure.\n\n"
            + tableau("Unicité de PolNum par table", "tab:doublons_e1",
                      ["Table", "Lignes", "PolNum distincts", "Lignes à PolNum répété", "Lignes identiques en double"],
                      lignes, "lrrrr")
            + "\n" + tableau("Recouvrement entre tables", "tab:recouvrement",
                             ["Jointure", "PolNum communs", "Part couverte"], rec, "lrp{8cm}")
            + f"\n{nombre(sans['nb_PolNum_distincts_sans_correspondance'])} polices de sinistres n'existent pas "
            f"dans le portefeuille ({nombre(sans['nb_lignes_sinistres_concernees'])} lignes).\n")


def s_methode_e1() -> str:
    return ("\\section{Méthode}\n"
            "Les contrôles de l'étape 1 portent sur les dimensions, les types, la classification des variables "
            "(quantitatives, qualitatives, identifiants), les statistiques descriptives et les clés de jointure. "
            "Les significations sont à valider avec le producteur des données.\n")


# ===========================================================================
# ÉTAPE 2 – AUDIT ET QUALITÉ DES DONNÉES
# ===========================================================================


def manquants(brut: pd.DataFrame, nom: str) -> pd.DataFrame:
    n = len(brut)
    lignes = []
    for col in brut.columns:
        s = brut[col]
        vides = int(s.isna().sum())
        textes = int((s.astype(str).str.strip().isin(CODES_TEXTE_MANQUANTS) & s.notna()).sum())
        lignes.append({"table": nom, "variable": col, "nb_NA": vides, "nb_codes_textuels": textes,
                       "nb_total_manquants": vides + textes,
                       "pct_manquants": round(100 * (vides + textes) / n, 2)})
    return pd.DataFrame(lignes)


def doublons(brut: pd.DataFrame, nom: str) -> tuple:
    rep = brut[brut["PolNum"].duplicated(keep=False)]
    diff = rep.groupby("PolNum").filter(lambda g: len(g.drop_duplicates()) > 1)
    resume = {
        "table": nom, "nb_lignes": int(len(brut)),
        "nb_lignes_identiques_en_double": int(brut.duplicated().sum()),
        "nb_cles_repetees": int(rep["PolNum"].nunique()),
        "nb_lignes_cle_repetee": int(len(rep)),
        "nb_cles_repetees_contenu_different": int(diff["PolNum"].nunique()),
    }
    return resume, rep.sort_values("PolNum").head(12).assign(table=nom)


def outliers(df: pd.DataFrame, nom: str, colonnes: list) -> pd.DataFrame:
    lignes = []
    for col in colonnes:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        bas, haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lignes.append({
            "table": nom, "variable": col, "Q1": round(q1, 2), "mediane": round(s.median(), 2),
            "Q3": round(q3, 2), "borne_basse": round(bas, 2), "borne_haute": round(haut, 2),
            "nb_hors_bornes": int(((s < bas) | (s > haut)).sum()),
            "pct_hors_bornes": round(100 * ((s < bas) | (s > haut)).mean(), 2),
            "max": round(s.max(), 2),
        })
    return pd.DataFrame(lignes)


def erreurs_saisie(brut: pd.DataFrame, nom: str) -> pd.DataFrame:
    lignes = []
    for col in brut.columns:
        if col == "PolNum" or not brut[col].map(lambda x: isinstance(x, str)).any():
            continue
        s = brut[col].dropna().astype(str)
        espaces = int((s != s.str.strip()).sum())
        if espaces:
            lignes.append({"table": nom, "variable": col, "anomalie": "Espaces en début/fin de valeur",
                           "nb_lignes": espaces})
        groupes = {}
        for v in s.unique():
            groupes.setdefault(v.strip().upper(), []).append(v)
        for vs in groupes.values():
            if len(vs) > 1:
                lignes.append({"table": nom, "variable": col,
                               "anomalie": "Variantes de casse : " + " / ".join(vs),
                               "nb_lignes": int(s.isin(vs).sum())})
    return pd.DataFrame(lignes)


def regles_metier(ptf: pd.DataFrame, expo: pd.DataFrame, sin: pd.DataFrame) -> pd.DataFrame:
    v = ptf["Value_num"]
    regles = [
        ("Impossible", "portefeuille", "Age", "Âge inférieur à 18 ans (permis de conduire impossible)", ptf["Age"] < 18),
        ("Impossible", "portefeuille", "Age", "Âge supérieur à 100 ans", ptf["Age"] > 100),
        ("Impossible", "portefeuille", "Poldur", "Ancienneté de police négative", ptf["Poldur"] < 0),
        ("Incohérent", "portefeuille", "Poldur / Age", "Ancienneté supérieure à l'âge du conducteur", ptf["Poldur"] > ptf["Age"]),
        ("Suspect", "portefeuille", "Poldur / Age", "Ancienneté > âge − 18 (à confirmer)", ptf["Poldur"] > ptf["Age"] - 18),
        ("Impossible", "portefeuille", "Value", "Valeur du véhicule négative ou nulle", v <= 0),
        ("Suspect", "portefeuille", "Value", "Valeur du véhicule inférieure à 2 000 €", v < 2000),
        ("Impossible", "exposition", "Expdays", "Exposition supérieure à 365 jours", expo["Expdays"] > 365),
        ("Impossible", "exposition", "Expdays", "Exposition nulle ou négative", expo["Expdays"] <= 0),
        ("Impossible", "sinistres", "chg_sin", "Coût de sinistre négatif", sin["chg_sin"] < 0),
        ("Suspect", "sinistres", "chg_sin", "Coût nul malgré au moins un sinistre", (sin["chg_sin"] == 0) & (sin["nb_sin"] > 0)),
        ("Impossible", "sinistres", "nb_sin", "Nombre de sinistres nul ou négatif", sin["nb_sin"] <= 0),
        ("Impossible", "portefeuille", "Density", "Densité de population nulle ou négative", ptf["Density"] <= 0),
    ]
    return pd.DataFrame([
        {"categorie": c, "table": t, "variable": var, "regle": lib,
         "nb_lignes": int(m.fillna(False).sum()), "pct": round(100 * m.fillna(False).mean(), 3)}
        for c, t, var, lib, m in regles
    ])


def incoherences(ptf: pd.DataFrame) -> pd.DataFrame:
    n = len(ptf)
    masques = [
        ("Occupation « Retired » avec un âge inférieur à 50 ans", (ptf["Occupation"] == "Retired") & (ptf["Age"] < 50)),
        ("Occupation « Unemployed » avec un âge inférieur à 18 ans", (ptf["Occupation"] == "Unemployed") & (ptf["Age"] < 18)),
        ("Occupation « Housewife » pour un conducteur de sexe masculin (Male, H ou h)",
         (ptf["Occupation"] == "Housewife") & (ptf["Gender"].isin(["Male", "H", "h"]))),
        ("Sous-groupe SubGroup2 incompatible avec Group2 (1re lettre différente)",
         ptf["SubGroup2"].notna() & (ptf["SubGroup2"].str[0] != ptf["Group2"])),
        ("Catégorie « Small » avec une valeur de véhicule supérieure à 40 000 €",
         (ptf["Category"] == "Small") & (ptf["Value_num"] > 40000)),
        ("Catégorie « Large » avec une valeur de véhicule inférieure à 5 000 €",
         (ptf["Category"] == "Large") & (ptf["Value_num"] < 5000)),
    ]
    return pd.DataFrame([{"regle": lib, "nb_lignes": int(m.fillna(False).sum()),
                          "pct": round(100 * m.fillna(False).mean(), 3)} for lib, m in masques])


def formats(sin_brut: pd.DataFrame, ptf_brut: pd.DataFrame, expo_brut: pd.DataFrame,
            formules: pd.DataFrame) -> pd.DataFrame:
    lignes = [
        {"table": "sinistres", "variable": "chg_sin", "constat": "Décimale en virgule (séparateur ';')",
         "nb_lignes": int(sin_brut["chg_sin"].str.contains(",").sum()), "gravite": "Faible",
         "action": "Lire avec decimal=',' ou remplacer la virgule"},
        {"table": "portefeuille", "variable": "Density", "constat": "Nombre stocké comme texte dans Excel",
         "nb_lignes": int(ptf_brut["Density"].apply(lambda x: isinstance(x, str)).sum()), "gravite": "Moyenne",
         "action": "Convertir en numérique à l'import"},
        {"table": "portefeuille", "variable": "Value", "constat": "Montant stocké comme texte (code '??')",
         "nb_lignes": int(ptf_brut["Value"].apply(lambda x: isinstance(x, str)).sum()), "gravite": "Moyenne",
         "action": "Remplacer '??' par NA puis convertir"},
        {"table": "portefeuille", "variable": "Gender", "constat": "Codes non normalisés (H, F, h)",
         "nb_lignes": int(ptf_brut["Gender"].isin(["H", "F", "h"]).sum()), "gravite": "Moyenne",
         "action": "Recoder en Male / Female"},
    ]
    for nom, df in [("sinistres", sin_brut), ("portefeuille", ptf_brut), ("exposition", expo_brut)]:
        lignes.append({"table": nom, "variable": "PolNum", "constat": "PolNum de longueur différente de 9 chiffres",
                       "nb_lignes": int((df["PolNum"].astype(str).str.len() != 9).sum()),
                       "gravite": "Faible", "action": "Aucune si 0"})
    for _, r in formules.iterrows():
        lignes.append({"table": r["feuille"], "variable": r["variable"],
                       "constat": "Cellule contenant une formule (lien vers un classeur externe absent)",
                       "nb_lignes": int(r["nb_formules"]), "gravite": "Élevée",
                       "action": "Figer les valeurs après vérification, ou retirer le lien"})
    return pd.DataFrame(lignes)


def s_methode_e2() -> str:
    return ("\\section{Méthode de l'audit}\n"
            "Sept familles de contrôles sont appliquées : valeurs manquantes (NA et codes textuels), doublons, "
            "valeurs aberrantes (Tukey, bornes à 1,5 écart interquartile), erreurs de saisie, règles métier, "
            "incohérences entre variables, formats et types (y compris les cellules-formules Excel). "
            "Les contrôles sont lus sur les \\emph{valeurs} des cellules, et non sur leurs formules.\n")


def s_synthese(d) -> str:
    reg = d["regles"].set_index(["table", "variable", "regle"])
    inc = d["incoherences"]
    masc = inc[inc["regle"].str.contains("masculin")].iloc[0]
    formules = int(d["formules"]["nb_formules"].sum())
    sans = d["cles"]["sinistres_sans_police"]
    dbl = d["doublons"].set_index("table")
    age_aberrant = (reg.loc[("portefeuille", "Age", "Âge inférieur à 18 ans (permis de conduire impossible)")]["nb_lignes"]
                    + reg.loc[("portefeuille", "Age", "Âge supérieur à 100 ans")]["nb_lignes"])
    dens = d["formats"].set_index("variable").loc["Density", "nb_lignes"]
    sexe = d["formats"].set_index("variable").loc["Gender", "nb_lignes"]
    constats = [
        ["Élevée", "Formules liées à un classeur externe absent (\\texttt{Category})",
         f"{nombre(formules)} cellules ({nombre(100*formules/100027, 1)} \\%)", "Figer les valeurs après contrôle"],
        ["Élevée", "Sexe et profession quasi indépendants (\\texttt{Housewife} masculin)",
         nombre(masc["nb_lignes"]), "Vérifier le codage source"],
        ["Élevée", "Sinistres dont la police est absente du portefeuille",
         f"{nombre(sans['nb_PolNum_distincts_sans_correspondance'])} polices", "Exclure ou corriger la clé"],
        ["Moyenne", "Nombre décimal stocké comme texte (\\texttt{Density})", nombre(dens), "Convertir à l'import"],
        ["Moyenne", "Doublons exacts dans le portefeuille", nombre(dbl.loc["portefeuille", "nb_lignes_identiques_en_double"]),
         "Supprimer avant jointure"],
        ["Moyenne", "Doublons exacts dans l'exposition", nombre(dbl.loc["exposition", "nb_lignes_identiques_en_double"]),
         "Supprimer avant jointure"],
        ["Moyenne", "Codes de sexe non normalisés (\\texttt{H}, \\texttt{F}, \\texttt{h})", nombre(sexe),
         "Recoder en Male / Female"],
        ["Faible", "Âge aberrant (250 ans) et âges inférieurs à 18 ans", nombre(age_aberrant), "Exclure ou corriger"],
        ["Faible", "Ancienneté de police négative",
         nombre(reg.loc[("portefeuille", "Poldur", "Ancienneté de police négative")]["nb_lignes"]), "Corriger"],
        ["Faible", "Coût de sinistre nul malgré un sinistre déclaré",
         nombre(reg.loc[("sinistres", "chg_sin", "Coût nul malgré au moins un sinistre")]["nb_lignes"]),
         "Confirmer la signification"],
    ]
    return ("\\section{Synthèse des constats}\n"
            "Les anomalies sont classées par gravité, c'est-à-dire par leur effet potentiel sur les analyses.\n\n"
            + tableau("Principaux constats classés par gravité", "tab:synthese",
                      ["Gravité", "Constat", "Volume", "Action recommandée"], constats, "lp{6.8cm}rp{4cm}"))


def s_manquants(d) -> str:
    m = d["manquants"][d["manquants"]["nb_total_manquants"] > 0]
    lignes = [[echapper(r.table), r"\texttt{" + echapper(r.variable) + "}", nombre(r.nb_NA),
               nombre(r.nb_codes_textuels), nombre(r.nb_total_manquants), nombre(r.pct_manquants, 2)]
              for r in m.itertuples()]
    return ("\\section{Valeurs manquantes}\n"
            "Deux natures de manquants : cellules vides (\\texttt{NA}) et codes textuels "
            "(\\texttt{??}, \\texttt{???}), que la recherche de vides ne détecte pas.\n\n"
            + tableau("Valeurs manquantes par variable", "tab:manquants",
                      ["Table", "Variable", "NA", "Codes textuels", "Total", "\\% des lignes"], lignes, "llrrrr"))


def s_doublons(d) -> str:
    lignes = [[echapper(r.table), nombre(r.nb_lignes), nombre(r.nb_lignes_identiques_en_double),
               nombre(r.nb_cles_repetees), nombre(r.nb_lignes_cle_repetee), nombre(r.nb_cles_repetees_contenu_different)]
              for r in d["doublons"].itertuples()]
    ex = d["exemples"][d["exemples"]["table"] == "sinistres"].head(8)
    lignes_ex = [[nombre(float(r["PolNum"])), nombre(float(r["nb_sin"])),
                  nombre(float(str(r["chg_sin"]).replace(",", ".")), 2)]
                 for r in ex.to_dict("records")]
    return ("\\section{Doublons}\n"
            "Deux types : les lignes \\emph{identiques} (suppression sans perte) et les \\emph{clés répétées} "
            "(même \\texttt{PolNum}, contenu différent, à interpréter).\n\n"
            + tableau("Doublons par table", "tab:doublons",
                      ["Table", "Lignes", "Identiques en double", "Clés répétées", "Lignes concernées",
                       "Clés au contenu différent"], lignes, "lrrrrr")
            + "\n" + tableau("Exemples de clés répétées dans les sinistres", "tab:exemples_doublons",
                             ["PolNum", "Nombre de sinistres", "Coût (\\texteuro)"], lignes_ex, "rrr")
            + "\nDans les sinistres, une même police peut légitimement donner lieu à plusieurs déclarations : "
            "ces lignes ne sont pas des doublons à supprimer, contrairement aux lignes identiques.\n")


def s_outliers(d) -> str:
    lignes = [[echapper(r.table), r"\texttt{" + echapper(r.variable) + "}", nombre(r.mediane, 2),
               nombre(r.borne_basse, 2), nombre(r.borne_haute, 2), nombre(r.nb_hors_bornes),
               nombre(r.pct_hors_bornes, 2), nombre(r.max, 2)] for r in d["outliers"].itertuples()]
    return ("\\section{Valeurs aberrantes}\n"
            "Bornes de Tukey : $Q_1 - 1{,}5\\,\\mathrm{IQR}$ et $Q_3 + 1{,}5\\,\\mathrm{IQR}$. Un point hors bornes "
            "n'est pas une erreur en soi.\n\n"
            + tableau("Valeurs hors bornes de Tukey", "tab:outliers",
                      ["Table", "Variable", "Médiane", "Borne basse", "Borne haute", "Hors bornes", "\\%", "Max"],
                      lignes, "llrrrrrr")
            + "\n\\begin{itemize}\n"
            "  \\item \\textbf{Queues longues légitimes} : \\texttt{chg\\_sin}, \\texttt{Value} et \\texttt{Bonus}. "
            "Les points extrêmes sont plausibles.\n"
            "  \\item \\textbf{Variable discrète} : \\texttt{nb\\_sin} a un écart interquartile nul, donc toute "
            "valeur supérieure à 1 est signalée. La méthode n'est pas adaptée.\n"
            "  \\item \\textbf{Exposition partielle} : \\texttt{Expdays} est inférieur à 365 pour 21 \\% des polices, "
            "ce qui est normal pour une police souscrite ou résiliée en cours d'année.\n"
            "\\end{itemize}\n")


def s_saisie(d) -> str:
    lignes = [[echapper(r.table), r"\texttt{" + echapper(r.variable) + "}", echapper(r.anomalie), nombre(r.nb_lignes)]
              for r in d["saisie"].itertuples()]
    return ("\\section{Erreurs de saisie}\n"
            "Contrôles des espaces superflus, des variantes de casse et des codes non normalisés. "
            "Aucun espace parasite n'a été relevé.\n\n"
            + tableau("Erreurs de saisie détectées", "tab:saisie", ["Table", "Variable", "Erreur", "Lignes"],
                      lignes, "llp{7cm}r"))


def s_regles(d) -> str:
    lignes = [[echapper(r.categorie), echapper(r.table) + "." + echapper(r.variable), echapper(r.regle),
               nombre(r.nb_lignes), nombre(r.pct, 3)] for r in d["regles"].itertuples()]
    return ("\\section{Valeurs impossibles et règles métier}\n"
            "Chaque règle est classée en \\emph{impossible}, \\emph{incohérent} ou \\emph{suspect}. "
            "Les règles à zéro sont conservées pour traçabilité.\n\n"
            + tableau("Règles métier appliquées", "tab:regles",
                      ["Catégorie", "Variable", "Règle", "Lignes", "\\%"], lignes, "llp{6.5cm}rr"))


def s_incoherences(d) -> str:
    lignes = [[echapper(r.regle), nombre(r.nb_lignes), nombre(r.pct, 2)] for r in d["incoherences"].itertuples()]
    return ("\\section{Incohérences entre variables}\n"
            + tableau("Incohérences entre variables", "tab:incoherences", ["Règle", "Lignes", "\\%"], lignes, "p{10cm}rr")
            + "\n\\textbf{Lecture sexe × profession} : environ 63 \\% des conducteurs sont de sexe masculin, quelle que "
            "soit la profession. Une vraie corrélation attendrait une répartition très différente pour "
            "\\texttt{Housewife}. Le codage est probablement en cause : cette variable ne doit pas être utilisée "
            "tant que ce point n'est pas expliqué.\n")


def s_formats(d) -> str:
    f = d["formats"]
    lignes = [[echapper(r.table), r"\texttt{" + echapper(r.variable) + "}", echapper(r.constat),
               nombre(r.nb_lignes), echapper(r.gravite)] for r in f.itertuples()]
    return ("\\section{Formats et types}\n"
            "Un constat majeur : une part importante de \\texttt{Category} n'est pas une valeur mais une "
            "\\emph{formule Excel} renvoyant vers un classeur externe dont le chemin est absent. "
            "Les valeurs en cache semblent cohérentes, mais ne sont pas vérifiables. "
            "L'étape 1 ne détectait pas ces formules : ce script les recense.\n\n"
            + tableau("Problèmes de format et de type", "tab:formats",
                      ["Table", "Variable", "Constat", "Lignes", "Gravité"], lignes, "llp{6cm}rl"))


def s_plan(d) -> str:
    return ("\\section{Plan d'action avant l'analyse}\n\\begin{enumerate}\n"
            "  \\item \\textbf{Traçabilité du classeur} : retrouver le fichier source de \\texttt{Category} ou la valider.\n"
            "  \\item \\textbf{Codage du sexe et de la profession} : confirmer \\texttt{Gender} et \\texttt{Occupation}.\n"
            "  \\item \\textbf{Nettoyage déterministe} : supprimer les lignes identiques, convertir \\texttt{Density} et "
            "\\texttt{Value}, normaliser \\texttt{Gender}, remplacer \\texttt{??} et \\texttt{???} par NA.\n"
            "  \\item \\textbf{Valeurs extrêmes} : exclure l'âge de 250 ans et les âges inférieurs à 18 ans ; conserver "
            "les queues longues de \\texttt{chg\\_sin}, \\texttt{Value} et \\texttt{Bonus}.\n"
            "  \\item \\textbf{Jointure} : décider du sort des polices de sinistres absentes du portefeuille, et "
            "dédoublonner l'exposition avant jointure.\n"
            "  \\item \\textbf{Définitions à confirmer} : \\texttt{Poldur}, \\texttt{Adind}, \\texttt{Density} et "
            "l'unité de \\texttt{chg\\_sin} et \\texttt{Value}.\n\\end{enumerate}\n")


# ===========================================================================
# ÉTAPE 3 – NETTOYAGE ET PRÉPARATION
# ===========================================================================


JOURNAL = []


def journaliser(etape, table, variable, action, effectif, justification):
    JOURNAL.append({"etape": etape, "table": table, "variable": variable, "action": action,
                    "nb_lignes": int(effectif), "justification": justification})


def charger_nettoyage():
    sin = pd.read_csv(FICHIER_CSV, sep=";", dtype=str, keep_default_na=False)
    wb = load_workbook(FICHIER_XLSX, read_only=True, data_only=True)
    feuilles = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        feuilles[ws.title] = pd.DataFrame(rows[1:], columns=rows[0])
    # Repère les polices dont la catégorie est une formule Excel (valeur en cache conservée)
    formules = set()
    ws = load_workbook(FICHIER_XLSX, read_only=True)["PTF"]
    rows = ws.iter_rows(values_only=True)
    entete = next(rows)
    i_cat, i_pol = entete.index("Category"), entete.index("PolNum")
    for r in rows:
        if isinstance(r[i_cat], str) and r[i_cat].startswith("="):
            formules.add(int(r[i_pol]))
    return sin, feuilles["PTF"], feuilles["Expo"], formules


def nettoyer_doublons(sin, ptf, expo):
    # Sinistres : lignes identiques = double saisie
    n_avant = len(sin)
    cout_doublons = pd.to_numeric(sin[sin.duplicated()]["chg_sin"].str.replace(",", ".", regex=False)).sum()
    sin = sin.drop_duplicates().reset_index(drop=True)
    journaliser("1. Doublons", "sinistres", "toutes", "Suppression des lignes identiques",
                n_avant - len(sin),
                "Ligne strictement identique à une autre : double saisie, aucune information ajoutée.")

    # Portefeuille : doublons exacts
    n_avant = len(ptf)
    ptf = ptf.drop_duplicates().reset_index(drop=True)
    journaliser("1. Doublons", "portefeuille", "toutes", "Suppression des lignes identiques",
                n_avant - len(ptf), "Ligne identique répétée : une police = une ligne.")

    # Portefeuille : même PolNum, contenu différent -> garder la ligne la plus complète et plausible
    n_conflits = 0
    garder = []
    for pol, grp in ptf.groupby("PolNum", sort=False):
        if len(grp) == 1:
            garder.append(grp.index[0])
            continue
        n_conflits += len(grp) - 1
        completude = grp.notna().sum(axis=1)
        plausible = grp["Poldur"] <= grp["Age"] - 18
        classement = pd.DataFrame({"compl": completude, "plaus": plausible}).sort_values(
            ["plaus", "compl"], ascending=False)
        garder.append(classement.index[0])
        ecarts = grp.loc[:, grp.nunique(dropna=False) > 1].columns.tolist()
        if ecarts == ["SubGroup2"]:
            motif = "Même police, une ligne sans SubGroup2 : on garde la ligne complète."
        else:
            motif = f"Conflit sur {', '.join(ecarts)} : on garde la valeur plausible (ancienneté ≤ âge − 18)."
        journaliser("1. Doublons", "portefeuille", "PolNum " + str(pol), "Conflit résolu : une ligne gardée",
                    len(grp) - 1, motif)
    ptf = ptf.loc[garder].reset_index(drop=True)

    # Exposition : doublons exacts
    n_avant = len(expo)
    expo = expo.drop_duplicates().reset_index(drop=True)
    journaliser("1. Doublons", "exposition", "toutes", "Suppression des lignes identiques",
                n_avant - len(expo), "Ligne identique répétée : une police = une exposition.")
    return sin, ptf, expo, cout_doublons


def corriger_formats(sin, ptf, expo, formules):
    # Sinistres : décimale en virgule, nombres entiers
    sin["chg_sin"] = pd.to_numeric(sin["chg_sin"].str.replace(",", ".", regex=False), errors="coerce")
    sin["nb_sin"] = pd.to_numeric(sin["nb_sin"], errors="coerce").astype("Int64")
    sin["PolNum"] = pd.to_numeric(sin["PolNum"], errors="coerce").astype("Int64")
    journaliser("2. Formats", "sinistres", "chg_sin", "Conversion en nombre (virgule décimale)",
                len(sin), "Le fichier utilise la virgule comme séparateur décimal.")

    # Portefeuille : Density stocké en texte
    n_texte = int(ptf["Density"].apply(lambda x: isinstance(x, str)).sum())
    ptf["Density"] = pd.to_numeric(ptf["Density"], errors="coerce")
    journaliser("2. Formats", "portefeuille", "Density", "Conversion texte vers nombre", n_texte,
                "Densité stockée comme texte dans l'Excel.")

    # Value : code '??' -> manquant
    n_codes = int((ptf["Value"] == "??").sum())
    ptf["Value"] = ptf["Value"].replace({"??": np.nan})
    ptf["Value"] = pd.to_numeric(ptf["Value"], errors="coerce")
    journaliser("2. Formats", "portefeuille", "Value", "Code '??' remplacé par manquant", n_codes,
                "Code textuel signalant une absence de valeur.")

    # Gender : codes normalisés
    avant = ptf["Gender"].copy()
    ptf["Gender"] = ptf["Gender"].replace({"H": "Male", "h": "Male", "F": "Female"})
    ptf["Gender"] = ptf["Gender"].fillna("Inconnu")
    n_norm = int((avant.isin(["H", "h", "F"])).sum())
    journaliser("2. Formats", "portefeuille", "Gender", "Normalisation H/h -> Male, F -> Female", n_norm,
                "Hypothèse : H = homme, F = femme (codes français). À confirmer avec le producteur.")
    journaliser("2. Formats", "portefeuille", "Gender", "Manquant renseigné « Inconnu »",
                int(avant.isna().sum()), "Cinq polices sans sexe : pas d'imputation fiable.")

    # Category : '???' -> 'Inconnue' (pas d'imputation), indicateur de source
    n_inc = int((ptf["Category"] == "???").sum())
    ptf["Category"] = ptf["Category"].replace({"???": "Inconnue"})
    ptf["categorie_source"] = np.where(ptf["PolNum"].isin(formules), "formule_externe", "valeur")
    journaliser("2. Formats", "portefeuille", "Category", "Code '???' remplacé par « Inconnue »", n_inc,
                "Pas d'imputation fiable (0,03 % des lignes). Valeurs issues de formules conservées en cache.")

    # SubGroup2 : indicateur de renseignement, colonne conservée
    ptf["subgroupe_renseigne"] = ptf["SubGroup2"].notna().astype(int)
    journaliser("2. Formats", "portefeuille", "SubGroup2", "Indicateur de renseignement créé",
                len(ptf), "88 % de valeurs manquantes : colonne conservée mais non imputée.")

    ptf["PolNum"] = pd.to_numeric(ptf["PolNum"], errors="coerce").astype("Int64")
    expo["PolNum"] = pd.to_numeric(expo["PolNum"], errors="coerce").astype("Int64")
    expo["Expdays"] = pd.to_numeric(expo["Expdays"], errors="coerce").astype("Int64")
    return sin, ptf, expo


def traiter_valeurs(ptf):
    ptf = ptf.copy()

    # Âges impossibles -> manquants puis imputation par la médiane de la profession
    impossibles = (ptf["Age"] > 100) | (ptf["Age"] < 18)
    n_imp_age = int(impossibles.sum())
    ptf["age_imputee"] = impossibles.astype(int)
    ptf.loc[impossibles, "Age"] = np.nan
    med_age = ptf.groupby("Occupation")["Age"].transform("median")
    ptf["Age"] = ptf["Age"].fillna(med_age)
    journaliser("3. Valeurs impossibles", "portefeuille", "Age",
                "Âge > 100 ou < 18 remplacé par la médiane de la profession", n_imp_age,
                "Âge de 250 ans ou permis impossible à 16 ans : erreur de saisie. Imputation signalée.")

    # Ancienneté négative -> manquant puis médiane
    neg = ptf["Poldur"] < 0
    ptf["poldur_imputee"] = neg.astype(int)
    ptf.loc[neg, "Poldur"] = np.nan
    ptf["Poldur"] = ptf["Poldur"].fillna(ptf["Poldur"].median())
    journaliser("3. Valeurs impossibles", "portefeuille", "Poldur",
                "Ancienneté négative remplacée par la médiane", int(neg.sum()),
                "Une ancienneté négative est impossible.")

    # Ancienneté incohérente avec l'âge : signalée, non modifiée
    ptf["poldur_incoherente"] = (ptf["Poldur"] > ptf["Age"]).astype(int)
    journaliser("3. Valeurs impossibles", "portefeuille", "Poldur",
                "Signalement : ancienneté supérieure à l'âge (non modifiée)",
                int(ptf["poldur_incoherente"].sum()), "Cas peu nombreux, à confirmer avec le producteur.")

    # Valeur du véhicule manquante -> médiane par type et catégorie
    manquante = ptf["Value"].isna()
    ptf["value_imputee"] = manquante.astype(int)
    med = ptf.groupby(["Type", "Category"])["Value"].transform("median")
    ptf["Value"] = ptf["Value"].fillna(med).fillna(ptf["Value"].median())
    journaliser("3. Valeurs manquantes", "portefeuille", "Value",
                "Imputation par la médiane (type, catégorie)", int(manquante.sum()),
                "Valeur manquante pour 0,8 % des polices. Médiane par groupe homogène ; imputation signalée.")

    # Valeurs extrêmes : signalées uniquement
    seuil_value = ptf["Value"].quantile(0.99)
    ptf["value_extreme"] = (ptf["Value"] > seuil_value).astype(int)
    journaliser("4. Valeurs aberrantes", "portefeuille", "Value",
                "Signalement des valeurs au-delà du 99e centile (non modifiées)",
                int(ptf["value_extreme"].sum()), "Véhicule de forte valeur : réel, conservé.")

    # Densité, Bonus : aucune modification
    journaliser("4. Valeurs aberrantes", "portefeuille", "Bonus",
                "Aucune modification", int(ptf["Bonus"].notna().sum()),
                "Coefficient bonus-malus dans une plage plausible (-50 à 150) : queue longue conservée.")
    return ptf


def traiter_sinistres(sin):
    sin = sin.copy()
    # Coût nul : conservé (sinistre clos sans paiement possible), signalé
    sin["cout_nul"] = ((sin["chg_sin"] == 0) & (sin["nb_sin"] > 0)).astype(int)
    journaliser("4. Valeurs aberrantes", "sinistres", "chg_sin",
                "Signalement des coûts nuls (non modifiés)", int(sin["cout_nul"].sum()),
                "Sinistre déclaré sans coût : peut être un dossier clos sans paiement. À confirmer.")

    # Sinistres extrêmes : signalés, non plafonnés
    positifs = sin.loc[sin["chg_sin"] > 0, "chg_sin"]
    p99 = positifs.quantile(0.99)
    sin["sinistre_extreme"] = (sin["chg_sin"] > p99).astype(int)
    journaliser("4. Valeurs aberrantes", "sinistres", "chg_sin",
                "Signalement des sinistres au-delà du 99e centile (non modifiés)",
                int(sin["sinistre_extreme"].sum()),
                "Sinistre exceptionnel potentiellement réel : conservé. Pris en compte dans le coût total.")
    # Version plafonnée à titre de sensibilité uniquement (non utilisée par défaut)
    p995 = positifs.quantile(0.995)
    sin["chg_sin_plafonne_sensibilite"] = sin["chg_sin"].clip(upper=p995)
    journaliser("4. Valeurs aberrantes", "sinistres", "chg_sin_plafonne_sensibilite",
                "Variable de sensibilité créée (plafond au 99,5e centile)", int(len(sin)),
                "Permet de mesurer l'impact des extrêmes. Non utilisée par défaut.")
    return sin


def construire_base(ptf, expo, sin):
    # Agrégation des sinistres par police
    sin_agg = sin.groupby("PolNum").agg(
        nb_sinistres=("nb_sin", "sum"),
        cout_total=("chg_sin", "sum"),
        nb_lignes_sinistres=("PolNum", "size"),
        nb_sinistres_extremes=("sinistre_extreme", "sum"),
    ).reset_index()

    base = ptf.merge(expo, on="PolNum", how="inner", validate="one_to_one")
    base = base.merge(sin_agg, on="PolNum", how="left", validate="one_to_one")
    for c in ["nb_sinistres", "cout_total", "nb_lignes_sinistres", "nb_sinistres_extremes"]:
        base[c] = base[c].fillna(0)
    base["nb_sinistres"] = base["nb_sinistres"].astype(int)
    base["nb_lignes_sinistres"] = base["nb_lignes_sinistres"].astype(int)
    base["nb_sinistres_extremes"] = base["nb_sinistres_extremes"].astype(int)

    # Variables créées
    base["expo_annees"] = base["Expdays"].astype(float) / 365.0
    base["a_sinistre"] = (base["nb_sinistres"] > 0).astype(int)
    base["frequence_annuelle"] = base["nb_sinistres"] / base["expo_annees"]
    base["cout_moyen_par_sinistre"] = np.where(base["nb_sinistres"] > 0,
                                               base["cout_total"] / base["nb_sinistres"].replace(0, np.nan),
                                               np.nan)
    base["tranche_age"] = pd.cut(base["Age"].astype(float), bins=[17, 25, 35, 50, 65, 120],
                                 labels=["18-25", "26-35", "36-50", "51-65", "66+"]).astype(str)
    base["tranche_anciennete"] = pd.cut(base["Poldur"].astype(float), bins=[-1, 1, 4, 9, 1000],
                                        labels=["0-1", "2-4", "5-9", "10+"]).astype(str)
    base["log_valeur"] = np.log(base["Value"].astype(float))
    base["quartile_valeur"] = pd.qcut(base["Value"].astype(float), 4,
                                      labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    base["sexe_renseigne"] = (base["Gender"] != "Inconnu").astype(int)
    return base


def variables_creees() -> pd.DataFrame:
    return pd.DataFrame([
        ("expo_annees", "Exposition en années (Expdays / 365)", "Dénominateur des fréquences"),
        ("nb_sinistres", "Nombre total de sinistres de la police", "Somme de nb_sin"),
        ("cout_total", "Coût total des sinistres de la police", "Somme de chg_sin"),
        ("a_sinistre", "1 si la police a au moins un sinistre", "Variable binaire"),
        ("frequence_annuelle", "Nombre de sinistres par année d'exposition", "nb_sinistres / expo_annees"),
        ("cout_moyen_par_sinistre", "Coût moyen par sinistre", "cout_total / nb_sinistres (si > 0)"),
        ("tranche_age", "Tranche d'âge du conducteur", "18-25, 26-35, 36-50, 51-65, 66+"),
        ("tranche_anciennete", "Tranche d'ancienneté de la police", "0-1, 2-4, 5-9, 10+ ans"),
        ("log_valeur", "Logarithme de la valeur du véhicule", "Pour les modèles multiplicatifs"),
        ("quartile_valeur", "Quartile de valeur du véhicule", "Q1 à Q4"),
        ("sexe_renseigne", "1 si le sexe est renseigné", "Indicateur de qualité"),
        ("nb_sinistres_extremes", "Sinistres au-delà du 99e centile", "Signalement, sans suppression"),
        ("subgroupe_renseigne", "1 si SubGroup2 est renseigné", "Indicateur de qualité"),
        ("age_imputee / poldur_imputee / value_imputee", "1 si la valeur a été imputée", "Traçabilité"),
        ("poldur_incoherente", "1 si ancienneté > âge", "Signalement"),
        ("categorie_source", "formule_externe ou valeur", "Traçabilité de la catégorie"),
    ], columns=["variable", "description", "calcul"])


def verifier(base, sin_clean, sin_raw_total, cout_doublons, ptf_avant, sin_avant, sinistres_sans_police):
    controles = []

    def ajouter(nom, ok, detail):
        controles.append({"controle": nom, "resultat": "OK" if ok else "ÉCHEC", "detail": detail})

    ajouter("Une ligne par police dans la base", base["PolNum"].is_unique, f"{len(base)} polices")
    ajouter("Pas de valeur manquante sur les champs clés",
            base[["PolNum", "Age", "Value", "Expdays", "Gender", "Poldur"]].isna().sum().sum() == 0,
            "PolNum, Age, Value, Expdays, Gender, Poldur")
    ajouter("Expdays entre 1 et 365", base["Expdays"].between(1, 365).all(),
            f"min {int(base['Expdays'].min())}, max {int(base['Expdays'].max())}")
    ajouter("Âge entre 18 et 100 après nettoyage", base["Age"].between(18, 100).all(),
            f"min {base['Age'].min():.0f}, max {base['Age'].max():.0f}")
    ajouter("Ancienneté positive ou nulle", (base["Poldur"] >= 0).all(), f"min {base['Poldur'].min():.0f}")
    ajouter("Valeur du véhicule strictement positive", (base["Value"] > 0).all(),
            f"min {base['Value'].min():,.0f}".replace(",", " "))
    ajouter("Aucun code textuel restant dans Category (hors « Inconnue »)",
            not base["Category"].isin(["???", "??"]).any(), f"{int((base['Category'] == 'Inconnue').sum())} « Inconnue »")
    ajouter("Sexe normalisé", set(base["Gender"].unique()) <= {"Male", "Female", "Inconnu"},
            ", ".join(sorted(base["Gender"].unique())))
    ajouter("Fréquences positives ou nulles", (base["frequence_annuelle"].dropna() >= 0).all(),
            f"max {base['frequence_annuelle'].max():.2f}")
    ajouter("Aucun sinistre perdu : total des nb_sin conservé",
            int(base["nb_sinistres"].sum()) == int(sin_clean.loc[sin_clean["PolNum"].isin(base["PolNum"]), "nb_sin"].sum()),
            f"{int(base['nb_sinistres'].sum())} sinistres dans la base")
    total_base = base["cout_total"].sum()
    total_clean = sin_clean.loc[sin_clean["PolNum"].isin(base["PolNum"]), "chg_sin"].sum()
    ajouter("Coût total rapproché (base = sinistres nettoyés rattachés)",
            abs(total_base - total_clean) < 0.01,
            f"{total_base:,.2f} € dans la base".replace(",", " "))
    ajouter("Coût total rapproché (brut − doublons supprimés = nettoyé)",
            abs((sin_raw_total - cout_doublons) - sin_clean["chg_sin"].sum()) < 0.01,
            f"brut {sin_raw_total:,.2f} − doublons {cout_doublons:,.2f} = nettoyé {sin_clean['chg_sin'].sum():,.2f}"
            .replace(",", " "))
    ajouter("Aucun doublon de police dans l'exposition de la base", base["PolNum"].duplicated().sum() == 0,
            "contrôle d'unicité")
    ajouter("Sinistres hors portefeuille exclus et comptés",
            sinistres_sans_police["nb_lignes"] > 0,
            f"{sinistres_sans_police['polices']} polices, {sinistres_sans_police['nb_lignes']} lignes exclues de la base")
    return pd.DataFrame(controles)


def etape3() -> dict:
    sin, ptf, expo, formules = charger_nettoyage()
    sin_raw_total = pd.to_numeric(sin["chg_sin"].str.replace(",", ".", regex=False)).sum()
    avant = {"sin_avant": len(sin), "ptf_avant": len(ptf), "expo_avant": len(expo)}

    JOURNAL.clear()
    sin, ptf, expo, cout_doublons = nettoyer_doublons(sin, ptf, expo)
    sin, ptf, expo = corriger_formats(sin, ptf, expo, formules)
    ptf = traiter_valeurs(ptf)
    sin = traiter_sinistres(sin)

    hors = sin[~sin["PolNum"].isin(ptf["PolNum"])]
    sinistres_sans_police = {"polices": int(hors["PolNum"].nunique()), "nb_lignes": len(hors)}
    journaliser("5. Jointure", "sinistres", "PolNum",
                "Sinistres sans police dans le portefeuille : conservés dans sinistres_nettoyes, exclus de la base",
                len(hors), "Pas de caractéristiques de risque disponibles pour ces polices. À traiter séparément.")

    base = construire_base(ptf, expo, sin)
    creees = variables_creees()
    verif = verifier(base, sin, sin_raw_total, cout_doublons, len(ptf), len(sin), sinistres_sans_police)

    verif.to_csv(SORTIE / "etape3_verifications.csv", index=False, sep=";")
    pd.DataFrame(JOURNAL).to_csv(SORTIE / "etape3_journal_decisions.csv", index=False, sep=";")
    creees.to_csv(SORTIE / "etape3_variables_creees.csv", index=False, sep=";")
    base.to_csv(SORTIE / "etape3_base_polices.csv", index=False, sep=";", decimal=",")
    sin.to_csv(SORTIE / "etape3_sinistres_nettoyes.csv", index=False, sep=";", decimal=",")

    echec = verif[verif["resultat"] != "OK"]
    print(f"Étape 3 : base de {len(base)} polices, {len(sin)} sinistres nettoyés, "
          f"{len(JOURNAL)} décisions, {len(echec)} contrôle(s) en échec")
    return {"journal": list(JOURNAL), "verif": verif, "creees": creees,
            "base_n": len(base), "base": base, "ptf": ptf, "expo": expo, "sin": sin,
            "avant_apres": {**avant, "sin_apres": len(sin), "ptf_apres": len(ptf), "expo_apres": len(expo)}}


def section_etape3(e3: dict) -> str:
    journal, verif, creees, avant_apres, base_n = (
        e3["journal"], e3["verif"], e3["creees"], e3["avant_apres"], e3["base_n"])
    principes = (
        "\\section{Objectif et principes}\n"
        "L'objectif est d'obtenir une base fiable pour les analyses. Trois principes guident les décisions :\n"
        "\\begin{itemize}\n"
        "  \\item \\textbf{Aucune suppression automatique des valeurs extrêmes.} Un sinistre exceptionnel peut "
        "être réel et actuariellement important : les extrêmes sont signalés, jamais retirés.\n"
        "  \\item \\textbf{Traçabilité.} Chaque décision est journalisée (table, variable, effectif, justification), "
        "et chaque imputation est signalée par une variable indicatrice.\n"
        "  \\item \\textbf{Vérification.} Les totaux sont rapprochés avant et après nettoyage.\n"
        "\\end{itemize}\n")
    lignes_av = [
        ["Sinistres", nombre(avant_apres["sin_avant"]), nombre(avant_apres["sin_apres"])],
        ["Polices du portefeuille", nombre(avant_apres["ptf_avant"]), nombre(avant_apres["ptf_apres"])],
        ["Lignes d'exposition", nombre(avant_apres["expo_avant"]), nombre(avant_apres["expo_apres"])],
        ["Polices dans la base finale", "--", nombre(base_n)],
    ]
    lignes_j = [[echapper(r["etape"]), echapper(r["table"]), r"\texttt{" + var_tex(r["variable"]) + "}",
                 echapper(r["action"]), nombre(r["nb_lignes"]), echapper(r["justification"])]
                for r in journal]
    lignes_v = [[echapper(r["controle"]), echapper(r["resultat"]), echapper(r["detail"])]
                for r in verif.to_dict("records")]
    lignes_c = [[r"\texttt{" + var_tex(r["variable"]) + "}", echapper(r["description"]), echapper(r["calcul"])]
                for r in creees.to_dict("records")]
    return (
        "\n\\part{Étape 3 -- Nettoyage et préparation des données}\n\\section{Objectif}\n"
        "Obtenir une base fiable pour les analyses, sans supprimer les valeurs extrêmes réelles.\n"
        + principes
        + "\\section{Effectifs avant et après nettoyage}\n"
        + tableau("Effectifs avant et après nettoyage", "tab:effectifs", ["Table", "Avant", "Après"],
                  lignes_av, "lrr")
        + "\\section{Journal des décisions}\n"
        "Chaque ligne correspond à une décision. Les effectifs indiquent le nombre de lignes concernées.\n"
        + tableau("Journal des décisions", "tab:journal",
                  ["Étape", "Table", "Variable", "Action", "Lignes", "Justification"], lignes_j,
                  "p{1.8cm}p{1.9cm}p{2.4cm}p{3.2cm}r p{4.0cm}")
        + "\\section{Vérifications après nettoyage}\n"
        + tableau("Contrôles après nettoyage", "tab:verif", ["Contrôle", "Résultat", "Détail"], lignes_v,
                  "p{6.5cm}lp{5.5cm}")
        + "\\section{Variables créées}\n"
        + tableau("Variables créées pour l'analyse", "tab:creees", ["Variable", "Description", "Calcul"],
                  lignes_c, "p{4.2cm}p{5.3cm}p{5cm}")
        + "\\section{Points à valider}\n\\begin{enumerate}\n"
        "  \\item Le sens des codes \\texttt{H}, \\texttt{h} et \\texttt{F} (hypothèse : homme, femme).\n"
        "  \\item L'explication de l'incohérence sexe × profession signalée à l'étape 2.\n"
        "  \\item Le conflit de \\texttt{Poldur} pour une police (une seule ligne retenue selon la plausibilité).\n"
        "  \\item La signification des coûts nuls et de la période d'observation des sinistres.\n"
        "  \\item La version de sensibilité plafonnée au 99,5e centile : à utiliser seulement pour comparer les "
        "résultats, jamais en remplacement des valeurs réelles.\n"
        "\\end{enumerate}\n")


# ===========================================================================
# ÉTAPE 4 – JOINTURE DES BASES
# ===========================================================================


POLICE_ATTRIBUTS = ["PolNum", "CalYear", "Gender", "Type", "Category", "Occupation", "Age", "Group1",
                    "Bonus", "Poldur", "Value", "Adind", "Group2", "Density", "Expdays",
                    "tranche_age", "quartile_valeur"]


def etape4(e3: dict) -> dict:
    """Réalise les jointures une à une, contrôle chaque résultat et produit la base consolidée."""
    ptf, expo, sin = e3["ptf"], e3["expo"], e3["sin"]
    controles = []

    # --- Jointure 1 : portefeuille ↔ exposition (relation 1 à 1) -------------------
    j1 = ptf[["PolNum"]].merge(expo, on="PolNum", how="outer", indicator=True, validate="one_to_one")
    j1_gauche_seules = int((j1["_merge"] == "left_only").sum())
    j1_droite_seules = int((j1["_merge"] == "right_only").sum())
    controles.append({
        "jointure": "Portefeuille ↔ Exposition", "relation": "1 à 1",
        "lignes_gauche": len(ptf), "lignes_droite": len(expo), "lignes_apres": len(j1),
        "correspondances": int((j1["_merge"] == "both").sum()),
        "cles_sans_correspondance_gauche": j1_gauche_seules,
        "cles_sans_correspondance_droite": j1_droite_seules,
        "doublons_generes": len(j1) - int(j1["PolNum"].nunique()),
        "resultat": "OK" if j1_gauche_seules == 0 and j1_droite_seules == 0 and len(j1) == len(ptf) else "À EXAMINER",
    })
    polices = ptf.merge(expo, on="PolNum", how="inner", validate="one_to_one")

    # --- Jointure 2 : sinistres → polices (plusieurs sinistres par police : n à 1) --
    attributs = e3["base"][POLICE_ATTRIBUTS]
    j2 = sin.merge(attributs, on="PolNum", how="left", indicator=True, validate="many_to_one")
    lignes_sans_police = j2[j2["_merge"] == "left_only"]
    controles.append({
        "jointure": "Sinistres → Polices", "relation": "n à 1",
        "lignes_gauche": len(sin), "lignes_droite": len(attributs), "lignes_apres": len(j2),
        "correspondances": int((j2["_merge"] == "both").sum()),
        "cles_sans_correspondance_gauche": int(lignes_sans_police["PolNum"].nunique()),
        "cles_sans_correspondance_droite": None,
        "doublons_generes": len(j2) - len(sin),
        "resultat": "OK" if len(j2) == len(sin) else "ÉCHEC",
    })

    # --- Jointure 3 : polices ← sinistres agrégés (1 à 0..1) ------------------------
    agregat = sin.groupby("PolNum", as_index=False).agg(nb_sinistres=("nb_sin", "sum"),
                                                       cout_total=("chg_sin", "sum"))
    j3 = polices[["PolNum"]].merge(agregat, on="PolNum", how="left", indicator=True, validate="one_to_one")
    sans_sinistre = int((j3["_merge"] == "left_only").sum())
    sinistres_hors_police = int((~agregat["PolNum"].isin(polices["PolNum"])).sum())
    controles.append({
        "jointure": "Polices ← Sinistres agrégés", "relation": "1 à 0 ou 1",
        "lignes_gauche": len(polices), "lignes_droite": len(agregat), "lignes_apres": len(j3),
        "correspondances": int((j3["_merge"] == "both").sum()),
        "cles_sans_correspondance_gauche": sans_sinistre,
        "cles_sans_correspondance_droite": sinistres_hors_police,
        "doublons_generes": len(j3) - len(polices),
        "resultat": "OK" if len(j3) == len(polices) else "ÉCHEC",
    })

    # --- Rapprochement des montants et des effectifs --------------------------------
    base = e3["base"]
    total_sin_rattaches = float(sin.loc[sin["PolNum"].isin(polices["PolNum"]), "chg_sin"].sum())
    total_base = float(base["cout_total"].sum())
    nb_rattaches = int(sin["PolNum"].isin(polices["PolNum"]).sum())
    coherence = [
        ("Coût total : base consolidée = sinistres rattachés", abs(total_base - total_sin_rattaches) < 0.01,
         f"{total_base:,.2f} € = {total_sin_rattaches:,.2f} €".replace(",", " ")),
        ("Nombre de sinistres : base = sinistres rattachés",
         int(base["nb_sinistres"].sum()) == int(sin.loc[sin["PolNum"].isin(polices["PolNum"]), "nb_sin"].sum()),
         f"{int(base['nb_sinistres'].sum())} sinistres"),
        ("Lignes de sinistres : rattachées + non rattachées = total",
         nb_rattaches + int(lignes_sans_police.shape[0]) == len(sin),
         f"{nb_rattaches} + {lignes_sans_police.shape[0]} = {len(sin)}"),
        ("Une ligne par police dans la base consolidée", base["PolNum"].is_unique, f"{len(base)} polices"),
        ("Aucun doublon généré par les jointures", all(c["doublons_generes"] == 0 for c in controles),
         "contrôle sur les trois jointures"),
    ]
    controles_coherence = pd.DataFrame(coherence, columns=["controle", "ok", "detail"])
    controles_coherence["resultat"] = np.where(controles_coherence["ok"], "OK", "ÉCHEC")

    # --- Sorties ---------------------------------------------------------------------
    sin_consolides = j2.drop(columns=["_merge"]).copy()
    sin_consolides["police_trouvee"] = (sin_consolides["Age"].notna()).astype(int)
    controles_df = pd.DataFrame(controles)
    controles_df.to_csv(SORTIE / "etape4_controles_jointure.csv", index=False, sep=";")
    controles_coherence.to_csv(SORTIE / "etape4_coherence.csv", index=False, sep=";")
    pd.DataFrame({"PolNum": lignes_sans_police["PolNum"].drop_duplicates().astype("int64")}).to_csv(
        SORTIE / "etape4_cles_sinistres_sans_police.csv", index=False, sep=";")
    sin_consolides.to_csv(SORTIE / "etape4_sinistres_consolides.csv", index=False, sep=";", decimal=",")
    base.to_csv(SORTIE / "etape4_base_polices_consolidee.csv", index=False, sep=";", decimal=",")

    print(f"Étape 4 : {len(base)} polices consolidées, {len(sin_consolides)} sinistres consolidés, "
          f"{len(lignes_sans_police)} lignes sans police")
    return {"controles": controles_df, "coherence": controles_coherence,
            "sans_police_polices": int(lignes_sans_police["PolNum"].nunique()),
            "sans_police_lignes": len(lignes_sans_police),
            "sans_sinistre": sans_sinistre, "base_n": len(base),
            "sin_consolides_n": len(sin_consolides)}


def section_etape4(e4: dict) -> str:
    c = e4["controles"]
    lignes = [[echapper(r["jointure"]), echapper(r["relation"]), nombre(r["lignes_gauche"]),
               nombre(r["lignes_droite"]), nombre(r["lignes_apres"]), nombre(r["correspondances"]),
               nombre(r["cles_sans_correspondance_gauche"]),
               "--" if pd.isna(r["cles_sans_correspondance_droite"]) else nombre(r["cles_sans_correspondance_droite"]),
               nombre(r["doublons_generes"]), echapper(r["resultat"])]
              for r in c.to_dict("records")]
    coh = [[echapper(r["controle"]), echapper(r["resultat"]), echapper(r["detail"])]
           for r in e4["coherence"].to_dict("records")]
    return (
        "\n\\part{Étape 4 -- Jointure des bases}\n\\section{Objectif}\n"
        "Réunir les informations pour obtenir une vision complète des polices et des sinistres, "
        "en contrôlant chaque jointure.\n\n"
        "\\section{Relations entre les tables}\n"
        "\\begin{itemize}\n"
        "  \\item \\textbf{Portefeuille et exposition} : une police = une ligne dans chaque table (relation 1 à 1).\n"
        "  \\item \\textbf{Sinistres et polices} : une police peut avoir plusieurs sinistres, un sinistre "
        "appartient à une seule police (relation n à 1).\n"
        "  \\item \\textbf{Polices et sinistres agrégés} : chaque police a au plus un total de sinistres "
        "(relation 1 à 0 ou 1).\n"
        "\\end{itemize}\n"
        "\\section{Contrôle des jointures}\n"
        "Pour chaque jointure : lignes avant et après, correspondances, clés sans correspondance et doublons "
        "générés. La validation de la relation (\\texttt{validate}) impose la cardinalité attendue : toute "
        "violation arrête le programme.\n"
        + "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        + tableau("Contrôle des jointures", "tab:jointures",
                  ["Jointure", "Relation", "Gauche", "Droite", "Après", "Corresp.", "Sans corresp. gauche",
                   "Sans corresp. droite", "Doublons", "Résultat"],
                  lignes, "p{2.1cm}p{1.2cm}*{8}{>{\\raggedleft\\arraybackslash}p{1.05cm}}p{1.7cm}")
        + "}\n"
        + "\\section{Cohérence des montants et des effectifs}\n"
        + tableau("Contrôles de cohérence après jointure", "tab:coherence_j",
                  ["Contrôle", "Résultat", "Détail"], coh, "p{7cm}lp{5cm}")
        + "\\section{Résultat : base consolidée}\n"
        f"La base consolidée contient {nombre(e4['base_n'])} polices, une ligne par police. "
        f"Les sinistres consolidés contiennent {nombre(e4['sin_consolides_n'])} lignes, chacune rattachée à ses "
        "caractéristiques de police lorsque la police existe. "
        f"{nombre(e4['sans_police_polices'])} polices de sinistres ({nombre(e4['sans_police_lignes'])} lignes) "
        "ne figurent pas dans le portefeuille : elles sont conservées mais sans caractéristiques de risque.\n"
        "\\section{Points à valider}\n\\begin{enumerate}\n"
        "  \\item Les polices de sinistres absentes du portefeuille : extraction incomplète du portefeuille, "
        "ou polices résiliées avant la période d'observation ?\n"
        f"  \\item Les {nombre(e4['sans_sinistre'])} polices sans sinistre : elles sont conservées dans la base "
        "avec un nombre de sinistres nul, ce qui est attendu pour une analyse de fréquence.\n"
        "  \\item La période des sinistres n'est pas connue : la jointure ne garantit pas qu'un sinistre "
        "correspond à l'exposition de l'année de la police.\n"
        "\\end{enumerate}\n"
    )


# ===========================================================================
# ÉTAPE 5 – ANALYSE DESCRIPTIVE ET EXPLORATOIRE
# ===========================================================================


BANDES_AGE = ["18-25", "26-35", "36-50", "51-65", "66+"]


BANDES_ANC = ["0-1", "2-4", "5-9", "10+"]


def _stats_colonnes(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    lignes = []
    for c in colonnes:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        lignes.append({"variable": c, "N": len(s), "moyenne": s.mean(), "mediane": s.median(),
                       "ecart_type": s.std(), "min": s.min(), "p90": s.quantile(0.9),
                       "p99": s.quantile(0.99), "max": s.max()})
    return pd.DataFrame(lignes)


def _etiqueter(ax, batons, textes, taille=7) -> None:
    """Écrit une étiquette au bout de chaque bâton et laisse de la place au-dessus."""
    ax.bar_label(batons, labels=textes, padding=2, fontsize=taille, fontweight="bold")
    ax.margins(y=0.15)


def _pct(valeurs) -> list:
    """Pourcentages à la française : 12.34 -> '12,3 %'."""
    return [f"{v:.1f} %".replace(".", ",") for v in valeurs]


def _sauver_fig(fig, nom: str) -> str:
    chemin = FIG / f"{nom}.png"
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return (FIG.relative_to(DOSSIER) / f"{nom}.png").as_posix()  # chemin lu par LaTeX depuis la racine


def tableau_groupe(base: pd.DataFrame, col: str, ordre: list) -> pd.DataFrame:
    """Fréquence, coût moyen et part de polices sinistrées par modalité."""
    g = base.groupby(col, observed=True).agg(
        polices=("PolNum", "size"), exposition=("expo_annees", "sum"),
        sinistres=("nb_sinistres", "sum"), cout=("cout_total", "sum"),
        part_sinistrees=("a_sinistre", "mean"), valeur_moyenne=("Value", "mean"),
    ).reindex(ordre).reset_index()
    g["frequence_pour_100_ans"] = 100 * g["sinistres"] / g["exposition"]
    g["cout_moyen_par_sinistre"] = np.where(g["sinistres"] > 0, g["cout"] / g["sinistres"].replace(0, np.nan), np.nan)
    g["part_sinistrees"] = 100 * g["part_sinistrees"]
    return g


def dummies(serie: pd.Series, nom: str, reference: str, ordre: list) -> tuple:
    colonnes, noms = [], []
    for c in ordre:
        if c == reference:
            continue
        colonnes.append((serie == c).astype(float).values)
        noms.append(f"{nom} = {c}")
    return colonnes, noms


def glm_poisson(y, expo, X, noms):
    """Régression de Poisson avec décalage log(exposition), par moindres carrés pondérés itératifs."""
    off = np.log(expo)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / expo.sum())
    for _ in range(100):
        eta = X @ beta + off
        mu = np.exp(eta)
        z = (eta - off) + (y - mu) / mu
        XtW = X.T * mu
        nouveau = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(nouveau - beta)) < 1e-10:
            beta = nouveau
            break
        beta = nouveau
    mu = np.exp(X @ beta + off)
    cov = np.linalg.inv((X.T * mu) @ X)
    se = np.sqrt(np.diag(cov))
    z_val = beta / se
    p = 2 * (1 - sstats.norm.cdf(np.abs(z_val)))
    return pd.DataFrame({
        "facteur": noms, "coefficient": beta, "erreur_type": se,
        "rapport_de_taux": np.exp(beta),
        "ic95_bas": np.exp(beta - 1.96 * se), "ic95_haut": np.exp(beta + 1.96 * se),
        "p_valeur": p,
    })


def etape5(e3: dict) -> dict:
    base = e3["base"].copy()
    sin = e3["sin"]
    base["Bonus_10"] = base["Bonus"].astype(float) / 10.0
    base["Gender"] = base["Gender"].astype(str)
    figures = {}

    # ---- A. Univarié -------------------------------------------------------------------
    uni = _stats_colonnes(base, ["Age", "Value", "Bonus", "Poldur", "expo_annees", "nb_sinistres"])
    uni.to_csv(SORTIE / "etape5_univarie_stats.csv", index=False, sep=";")

    freq_nb = base["nb_sinistres"].value_counts().sort_index().reset_index()
    freq_nb.columns = ["nb_sinistres", "polices"]
    freq_nb["pct"] = 100 * freq_nb["polices"] / len(base)
    freq_nb.to_csv(SORTIE / "etape5_nb_sinistres.csv", index=False, sep=";")

    montants = sin.loc[sin["chg_sin"] > 0, "chg_sin"].astype(float)
    m_stats = pd.DataFrame([{
        "N": len(montants), "moyenne": montants.mean(), "mediane": montants.median(),
        "p90": montants.quantile(0.9), "p99": montants.quantile(0.99), "max": montants.max(),
        "part_cout_top1pct_sinistres": 100 * montants.nlargest(int(len(montants) * 0.01)).sum() / montants.sum(),
    }])
    m_stats.to_csv(SORTIE / "etape5_montants_stats.csv", index=False, sep=";")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(base["Age"].astype(float), bins=np.arange(17, 81, 2), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Âge du conducteur (ans)"); ax.set_ylabel("Polices")
    ax.set_title("Distribution des âges (après nettoyage)")
    figures["age"] = _sauver_fig(fig, "fig_age")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(base["Value"].astype(float), bins=60, color="#55A868", edgecolor="white")
    ax.set_xlabel("Valeur du véhicule (€)"); ax.set_ylabel("Polices")
    ax.set_title("Distribution de la valeur du véhicule (proxy du niveau tarifaire)")
    figures["value"] = _sauver_fig(fig, "fig_value")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b = ax.bar(freq_nb["nb_sinistres"].astype(str), freq_nb["polices"], color="#C44E52")
    ax.set_yscale("log")
    ax.bar_label(b, labels=[f"{v:.2f} %".replace(".", ",") for v in freq_nb["pct"]], padding=2, fontsize=7,
                 fontweight="bold")  # part des polices
    ax.set_ylim(top=freq_nb["polices"].max() * 4); ax.set_xlabel("Nombre de sinistres par police"); ax.set_ylabel("Polices (échelle log)")
    ax.set_title("Fréquence des sinistres par police")
    figures["nb"] = _sauver_fig(fig, "fig_nb_sinistres")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(np.log10(montants), bins=50, color="#8172B2", edgecolor="white")
    ax.set_xlabel("Montant du sinistre (log10 €)"); ax.set_ylabel("Sinistres")
    ax.set_title("Distribution des montants de sinistres (échelle logarithmique)")
    figures["montants"] = _sauver_fig(fig, "fig_montants")

    # ---- B. Bivarié et multivarié ---------------------------------------------------------
    t_age = tableau_groupe(base, "tranche_age", BANDES_AGE)
    t_sexe = tableau_groupe(base, "Gender", ["Male", "Female", "Inconnu"])
    t_anc = tableau_groupe(base, "tranche_anciennete", BANDES_ANC)
    t_type = tableau_groupe(base, "Type", sorted(base["Type"].unique()))
    for nom, t in [("age", t_age), ("sexe", t_sexe), ("anciennete", t_anc), ("type", t_type)]:
        t.to_csv(SORTIE / f"etape5_biv_{nom}.csv", index=False, sep=";")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b = ax.bar(t_age["tranche_age"], t_age["frequence_pour_100_ans"], color="#4C72B0")
    _etiqueter(ax, b, _pct(t_age["frequence_pour_100_ans"]))  # fréquence annuelle en %
    ax.set_xlabel("Tranche d'âge"); ax.set_ylabel("Sinistres pour 100 ans d'exposition")
    ax.set_title("Fréquence des sinistres selon l'âge")
    figures["freq_age"] = _sauver_fig(fig, "fig_frequence_age")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b = ax.bar(t_anc["tranche_anciennete"], t_anc["frequence_pour_100_ans"], color="#DD8452")
    _etiqueter(ax, b, _pct(t_anc["frequence_pour_100_ans"]))  # fréquence annuelle en %
    ax.set_xlabel("Ancienneté de la police (années)"); ax.set_ylabel("Sinistres pour 100 ans d'exposition")
    ax.set_title("Fréquence des sinistres selon l'ancienneté")
    figures["freq_anc"] = _sauver_fig(fig, "fig_frequence_anciennete")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b = ax.bar(t_type["Type"].astype(str), t_type["cout_moyen_par_sinistre"], color="#937860")
    cm = t_type["cout"].sum() / t_type["sinistres"].sum()  # coût moyen de tout le portefeuille
    _etiqueter(ax, b, [f"{v:.0f} €\n({round(100 * (v / cm - 1)) or 0:+d} %)" for v in t_type["cout_moyen_par_sinistre"]])  # écart à la moyenne
    ax.set_xlabel("Type (code)"); ax.set_ylabel("Coût moyen par sinistre (€)")
    ax.set_title("Coût moyen des sinistres selon le type")
    figures["cout_type"] = _sauver_fig(fig, "fig_cout_type")

    # Sexe et niveau tarifaire (valeur du véhicule)
    val_sexe = base.groupby("Gender")["Value"].agg(["count", "mean", "median"]).reindex(
        ["Male", "Female", "Inconnu"]).reset_index()
    val_sexe.to_csv(SORTIE / "etape5_biv_sexe_valeur.csv", index=False, sep=";")

    # Multivarié : Poisson avec décalage log(exposition)
    # Les polices au sexe « Inconnu » sont exclues : aucun sinistre observé, effet non estimable.
    d = base[(base["expo_annees"] > 0) & (base["Gender"] != "Inconnu")].copy()
    y = d["nb_sinistres"].astype(float).values
    expo = d["expo_annees"].astype(float).values
    blocs, noms = [np.ones(len(d))], ["Constante"]
    for col, ref, ordre, nom in [("tranche_age", "36-50", BANDES_AGE, "Âge"),
                                 ("Gender", "Male", ["Male", "Female"], "Sexe"),
                                 ("tranche_anciennete", "5-9", BANDES_ANC, "Ancienneté"),
                                 ("Type", "A", sorted(base["Type"].unique()), "Type")]:
        c, n = dummies(d[col].astype(str), nom, ref, ordre)
        blocs += c
        noms += n
    blocs.append(d["Bonus_10"].values)
    noms.append("Bonus-malus (par +10 points)")
    blocs.append(d["log_valeur"].astype(float).values)
    noms.append("Valeur du véhicule (par +1 log)")
    X = np.column_stack(blocs)
    glm = glm_poisson(y, expo, X, noms)
    glm.to_csv(SORTIE / "etape5_poisson.csv", index=False, sep=";")

    # ---- C. Corrélations -----------------------------------------------------------------
    # a_sinistre et frequence_annuelle sont dérivées de nb_sinistres : leur corrélation serait mécanique.
    cols = ["Age", "Bonus", "Poldur", "Value", "Density", "expo_annees", "nb_sinistres"]
    cd = base[cols].astype(float)
    rho = cd.corr(method="spearman")
    rho.to_csv(SORTIE / "etape5_correlations_spearman.csv", sep=";", decimal=",")
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Corrélations de Spearman")
    figures["corr"] = _sauver_fig(fig, "fig_correlations")

    cible = rho["nb_sinistres"].drop("nb_sinistres").sort_values(key=np.abs, ascending=False).reset_index()
    cible.columns = ["variable", "rho_avec_nb_sinistres"]
    cible.to_csv(SORTIE / "etape5_correlations_cible.csv", index=False, sep=";", decimal=",")

    print(f"Étape 5 : {len(base)} polices analysées ; modèle de Poisson sur {len(d)} polices ; "
          f"{len(figures)} familles de figures")
    return {"uni": uni, "freq_nb": freq_nb, "montants": m_stats, "t_age": t_age, "t_sexe": t_sexe,
            "t_anc": t_anc, "t_type": t_type, "val_sexe": val_sexe, "glm": glm, "rho": rho,
            "cible": cible, "figures": figures, "n_base": len(base), "n_glm": len(d),
            "n_inconnu": int((base["Gender"] == "Inconnu").sum())}


def section_etape5(e5: dict) -> str:
    f = e5["figures"]
    def fig(chemin, legende):
        return ("\\begin{figure}[htbp]\\centering\\includegraphics[width=0.85\\textwidth]{" + chemin + "}\n"
                "\\caption{" + echapper(legende) + "}\\end{figure}\n")

    def tab_groupe(t, nom_col, titre, label):
        lignes = [[echapper(r[nom_col]), nombre(r["polices"]), nombre(r["exposition"], 0),
                    nombre(r["sinistres"]), nombre(r["frequence_pour_100_ans"], 2),
                    nombre(r["part_sinistrees"], 2), nombre(r["cout_moyen_par_sinistre"], 0),
                    nombre(r["valeur_moyenne"], 0)] for r in t.to_dict("records")]
        return ("{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
                + tableau(titre, label, [nom_col.replace("_", " "), "Polices", "Exposition (ans)", "Sinistres",
                                         "Fréquence /100 ans", "\\% sinistrées", "Coût moyen (€)", "Valeur moy. (€)"],
                          lignes, "p{2.4cm}*{7}{>{\\raggedleft\\arraybackslash}p{1.45cm}}")
                + "}\n")

    uni = e5["uni"]
    lignes_u = [[r"\texttt{" + echapper(r["variable"]) + "}", nombre(r["N"]), nombre(r["moyenne"], 2),
                 nombre(r["mediane"], 2), nombre(r["ecart_type"], 2), nombre(r["min"], 2),
                 nombre(r["p90"], 2), nombre(r["max"], 2)] for r in uni.to_dict("records")]
    m = e5["montants"].iloc[0]
    freq = e5["freq_nb"]
    lignes_f = [[nombre(r["nb_sinistres"]), nombre(r["polices"]), nombre(r["pct"], 2)] for r in freq.to_dict("records")]

    glm = e5["glm"]
    lignes_g = []
    for r in glm.to_dict("records"):
        sig = "oui" if r["p_valeur"] < 0.05 else "non"
        lignes_g.append([echapper(r["facteur"]), nombre(r["rapport_de_taux"], 3),
                         nombre(r["ic95_bas"], 3) + " – " + nombre(r["ic95_haut"], 3),
                         "< 0,001" if r["p_valeur"] < 0.001 else nombre(r["p_valeur"], 3), sig])
    rho = e5["cible"]
    lignes_r = [[r"\texttt{" + echapper(r["variable"]) + "}", nombre(r["rho_avec_nb_sinistres"], 3)]
                for r in rho.to_dict("records")]

    return (
        "\n\\part{Étape 5 -- Analyse descriptive et exploratoire}\n\\section{Objectif}\n"
        "Comprendre le comportement des données et identifier les premières relations. "
        "Trois niveaux sont traités : univarié, bivarié et multivarié, puis corrélations.\n\n"
        "\\begin{enumerate}\n"
        "  \\item \\textbf{Limite importante} : le jeu de données ne contient \\emph{pas} de colonne de prime. "
        "Les analyses portant sur la prime sont donc remplacées par la \\textbf{valeur du véhicule} et le "
        "\\textbf{bonus-malus}, qui sont des facteurs tarifaires. Ce sont des substituts, pas des primes.\n"
        "  \\item \\textbf{Type} est interprété comme le type de contrat ou de véhicule (codes A à F, hypothèse à confirmer).\n"
        "  \\item \\textbf{Fréquence} = nombre de sinistres par année d'exposition. Le coût moyen = coût total / nombre de sinistres.\n"
        "\\end{enumerate}\n"
        "\\section{A. Analyse univariée}\n"
        "\\subsection{Distribution des âges et des valeurs}\n"
        + fig(f["age"], "Distribution des âges des conducteurs (base par police, après nettoyage).")
        + fig(f["value"], "Distribution de la valeur du véhicule : substitut du niveau tarifaire, la prime n'étant pas disponible.")
        + tableau("Statistiques descriptives des variables principales", "tab:e5_uni",
                  ["Variable", "N", "Moyenne", "Médiane", "Écart-type", "Min", "P90", "Max"], lignes_u,
                  "lrrrrrrr")
        + "\\subsection{Fréquence des sinistres}\n"
        + fig(f["nb"], "Nombre de sinistres par police (échelle logarithmique).")
        + tableau("Répartition des polices selon leur nombre de sinistres", "tab:e5_nb",
                  ["Sinistres", "Polices", "\\%"], lignes_f, "rrr")
        + "\\subsection{Distribution des montants de sinistres}\n"
        + fig(f["montants"], "Montants des sinistres (chaque sinistre au coût positif). L'échelle logarithmique "
              "montre la queue longue, attendue en assurance.")
        + tableau("Montants des sinistres (coût positif)", "tab:e5_montants",
                  ["N", "Moyenne (€)", "Médiane (€)", "P90 (€)", "P99 (€)", "Max (€)", "Part du coût du top 1 \\%"],
                  [[nombre(m["N"]), nombre(m["moyenne"], 0), nombre(m["mediane"], 0), nombre(m["p90"], 0),
                    nombre(m["p99"], 0), nombre(m["max"], 0), nombre(m["part_cout_top1pct_sinistres"], 1) + " \\%"]],
                  "rrrrrrr")
        + "\\textbf{Lecture} : le top 1 \\% des sinistres représente une part importante du coût total. Ces "
        "sinistres sont réels et ne sont pas retirés : ils comptent fortement dans le coût des garanties.\n"
        "\\section{B. Analyse bivariée et multivariée}\n"
        "\\subsection{Âge et fréquence des sinistres}\n"
        + fig(f["freq_age"], "Fréquence des sinistres selon la tranche d'âge.")
        + tab_groupe(e5["t_age"], "tranche_age", "Fréquence et coût selon la tranche d'âge", "tab:e5_age")
        + "\\subsection{Sexe et niveau tarifaire}\n"
        "Les primes n'étant pas disponibles, le niveau tarifaire est approché par la valeur du véhicule.\n"
        + tab_groupe(e5["t_sexe"], "Gender", "Fréquence, coût et valeur selon le sexe", "tab:e5_sexe")
        + "\\subsection{Ancienneté et sinistralité}\n"
        + fig(f["freq_anc"], "Fréquence des sinistres selon l'ancienneté de la police.")
        + tab_groupe(e5["t_anc"], "tranche_anciennete", "Fréquence et coût selon l'ancienneté", "tab:e5_anc")
        + "\\subsection{Type et coût moyen des sinistres}\n"
        + fig(f["cout_type"], "Coût moyen par sinistre selon le type (codes A à F).")
        + tab_groupe(e5["t_type"], "Type", "Fréquence et coût selon le type", "tab:e5_type")
        + "\\subsection{Analyse multivariée : modèle de Poisson}\n"
        f"Pour mesurer l'effet de chaque facteur toutes choses égales par ailleurs, la fréquence est modélisée "
        f"par une régression de Poisson sur {nombre(e5['n_glm'])} polices, avec décalage logarithmique de "
        "l'exposition. Un rapport de taux supérieur à 1 signifie une fréquence plus élevée ; l'intervalle à 95 \\% "
        "indique la précision. Les polices au sexe inconnu ("
        + str(e5['n_inconnu']) + " polices) sont exclues : aucun sinistre n'y est observé, "
        "l'effet ne peut pas être estimé. Les facteurs de référence sont : âge 36-50, sexe masculin, "
        "ancienneté 5-9 ans, type A.\n"
        + tableau("Modèle de Poisson : rapports de taux (fréquence des sinistres)", "tab:e5_glm",
                  ["Facteur", "Rapport de taux", "IC 95 \\%", "p-valeur", "Significatif (5 \\%)"], lignes_g,
                  "p{6.2cm}rrrl")
        + "\\section{C. Analyse des corrélations}\n"
        "Les corrélations de Spearman sont utilisées : elles ne supposent pas de loi normale et résistent aux valeurs extrêmes.\n"
        + fig(f["corr"], "Matrice de corrélation de Spearman entre les variables numériques.")
        + tableau("Corrélation de chaque variable avec le nombre de sinistres", "tab:e5_rho",
                  ["Variable", "Rho de Spearman"], lignes_r, "lr")
        + "\\subsection{Facteurs potentiellement explicatifs}\n"
        "\\begin{itemize}\n"
        "  \\item Les facteurs \\textbf{significatifs dans le modèle multivarié} sont les candidats les plus solides. "
        "Ils restent à valider : un modèle ne prouve pas la causalité.\n"
        "  \\item Le \\textbf{bonus-malus} est un facteur tarifaire : sa relation avec la sinistralité peut résulter "
        "de la tarification elle-même. Il ne doit pas être interprété comme une cause.\n"
        "  \\item Les corrélations entre variables explicatives (par exemple âge et ancienneté) peuvent masquer "
        "ou exagérer un effet : c'est pourquoi le modèle multivarié est privilégié.\n"
        "\\end{itemize}\n"
        "\\section{Points à valider}\n\\begin{enumerate}\n"
        "  \\item Confirmer l'existence d'une prime dans les données, ou fournir la table des primes.\n"
        "  \\item Confirmer la signification de \\texttt{Type} (contrat, véhicule ou autre).\n"
        "  \\item Connaître la période des sinistres pour vérifier l'alignement avec l'exposition.\n"
        "\\end{enumerate}\n"
    )


# ===========================================================================
# ÉTAPE 6 – SEGMENTATION ET RÉDUCTION DE DIMENSION
# ===========================================================================


VARIABLES_SEG = ["Age", "Bonus", "Poldur", "log_valeur", "log_densite"]


LIBELLES_SEG = {"Age": "Âge", "Bonus": "Bonus-malus", "Poldur": "Ancienneté",
                "log_valeur": "Valeur (log)", "log_densite": "Densité (log)"}


NOMS_PROFILS = ["Faible risque", "Risque intermédiaire", "Risque élevé"]


def etape6(e3: dict) -> dict:

    base = e3["base"].copy()
    figures = {}
    base["Age"] = base["Age"].astype(float)
    base["Bonus"] = base["Bonus"].astype(float)
    base["Poldur"] = base["Poldur"].astype(float)
    base["Value"] = base["Value"].astype(float)
    base["log_valeur"] = np.log(base["Value"].clip(lower=1))
    base["log_densite"] = np.log1p(base["Density"].astype(float).clip(lower=0))
    base["femme"] = (base["Gender"].astype(str) == "Female").astype(float)
    Z = StandardScaler().fit_transform(base[VARIABLES_SEG].to_numpy())
    rng = np.random.default_rng(GRAINE)

    # ---- ACP ---------------------------------------------------------------------------
    acp = PCA(random_state=GRAINE).fit(Z)
    composantes = acp.transform(Z)
    var = pd.DataFrame({
        "composante": [f"CP{i + 1}" for i in range(len(VARIABLES_SEG))],
        "valeur_propre": acp.explained_variance_,
        "pct_variance": 100 * acp.explained_variance_ratio_,
        "pct_cumule": 100 * np.cumsum(acp.explained_variance_ratio_),
    })
    var.to_csv(SORTIE / "etape6_acp_variance.csv", index=False, sep=";", decimal=",")
    # Corrélation variable / composante = vecteur propre × racine de la valeur propre (données standardisées)
    charges = pd.DataFrame(acp.components_[:2].T * np.sqrt(acp.explained_variance_[:2]),
                           columns=["CP1", "CP2"])
    charges.insert(0, "variable", VARIABLES_SEG)
    charges.to_csv(SORTIE / "etape6_acp_charges.csv", index=False, sep=";", decimal=",")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b = ax.bar(var["composante"], var["pct_variance"], color="#4C72B0", label="% de variance")
    ax.bar_label(b, labels=_pct(var["pct_variance"]), label_type="center", fontsize=7, color="white",
                 fontweight="bold")  # % de variance dans le bâton (la courbe passe au-dessus)
    ax.plot(var["composante"], var["pct_cumule"], "o-", color="#C44E52", label="% cumulé")
    ax.set_ylabel("% de la variance"); ax.set_ylim(0, 105); ax.legend(fontsize=8)
    ax.set_title("Éboulis des valeurs propres (ACP)")
    figures["acp_eboulis"] = _sauver_fig(fig, "fig_acp_eboulis")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    for _, r in charges.iterrows():
        ax.arrow(0, 0, r["CP1"], r["CP2"], head_width=0.04, length_includes_head=True, color="#4C72B0")
        ax.text(r["CP1"] * 1.15, r["CP2"] * 1.15, LIBELLES_SEG[r["variable"]], fontsize=8, ha="center")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.set_xlabel(f"CP1 ({var['pct_variance'][0]:.1f} %)"); ax.set_ylabel(f"CP2 ({var['pct_variance'][1]:.1f} %)")
    ax.set_title("Cercle des corrélations")
    figures["acp_cercle"] = _sauver_fig(fig, "fig_acp_cercle")

    # ---- Choix du nombre de groupes (K-means) -------------------------------------------
    idx_sil = rng.choice(len(Z), size=min(20000, len(Z)), replace=False)
    lignes_k = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(Z)
        lignes_k.append({"k": k, "inertie": km.inertia_,
                         "silhouette": silhouette_score(Z[idx_sil], km.labels_[idx_sil])})
    choix_k = pd.DataFrame(lignes_k)
    choix_k.to_csv(SORTIE / "etape6_kmeans_choix_k.csv", index=False, sep=";", decimal=",")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(choix_k["k"], choix_k["inertie"], "o-", color="#4C72B0")
    ax.set_xlabel("Nombre de groupes k"); ax.set_ylabel("Inertie intra-groupes")
    ax2 = ax.twinx()
    ax2.plot(choix_k["k"], choix_k["silhouette"], "s--", color="#C44E52")
    ax2.set_ylabel("Silhouette (rouge)", color="#C44E52")
    ax.set_title("Choix de k : coude de l'inertie et silhouette")
    figures["kmeans_k"] = _sauver_fig(fig, "fig_kmeans_choix_k")

    # ---- Segmentation finale : K-means à 3 groupes --------------------------------------
    km3 = KMeans(n_clusters=3, n_init=10, random_state=GRAINE).fit(Z)
    base["segment_brut"] = km3.labels_

    # Validation : classification hiérarchique de Ward sur un échantillon
    idx_h = rng.choice(len(Z), size=3000, replace=False)
    hier = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Z[idx_h])
    ari = adjusted_rand_score(km3.labels_[idx_h], hier)
    hier_df = pd.DataFrame([{"methode": "K-means (k=3)", "echantillon": 3000,
                             "concordance_ARI_avec_hierarchique": ari}])
    hier_df.to_csv(SORTIE / "etape6_hierarchique_comparaison.csv", index=False, sep=";", decimal=",")

    # ---- Profils : caractéristiques et sinistralité ----------------------------------------
    prof = base.groupby("segment_brut").agg(
        polices=("PolNum", "size"), exposition=("expo_annees", "sum"),
        sinistres=("nb_sinistres", "sum"), cout=("cout_total", "sum"),
        age=("Age", "mean"), bonus=("Bonus", "mean"), poldur=("Poldur", "mean"),
        valeur=("Value", "mean"), part_femmes=("femme", "mean"),
    ).reset_index()
    prof["part_polices"] = 100 * prof["polices"] / prof["polices"].sum()
    prof["frequence_pour_100_ans"] = 100 * prof["sinistres"] / prof["exposition"]
    prof["cout_moyen_par_sinistre"] = prof["cout"] / prof["sinistres"].replace(0, np.nan)
    prof["charge_annuelle"] = prof["cout"] / prof["exposition"]
    prof["part_femmes"] = 100 * prof["part_femmes"]
    # Nommage selon la charge annuelle moyenne (coût total / exposition) : la plus faible = faible risque
    prof = prof.sort_values("charge_annuelle").reset_index(drop=True)
    prof["profil"] = NOMS_PROFILS
    base["profil"] = base["segment_brut"].map(dict(zip(prof["segment_brut"], prof["profil"])))
    prof.to_csv(SORTIE / "etape6_profils.csv", index=False, sep=";", decimal=",")
    base[["PolNum", "segment_brut", "profil"]].to_csv(SORTIE / "etape6_affectation_polices.csv",
                                                       index=False, sep=";")

    # Figures : projection ACP colorée par profil, et sinistralité par profil
    couleurs = {"Faible risque": "#55A868", "Risque intermédiaire": "#DD8452", "Risque élevé": "#C44E52"}
    idx_plot = rng.choice(len(Z), size=5000, replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for nom in NOMS_PROFILS:
        m = idx_plot[base["profil"].values[idx_plot] == nom]
        ax.scatter(composantes[m, 0], composantes[m, 1], s=4, alpha=0.5, color=couleurs[nom], label=nom)
    ax.set_xlabel("CP1"); ax.set_ylabel("CP2"); ax.legend(fontsize=8, markerscale=3)
    ax.set_title("Polices (échantillon de 5 000) projetées sur le premier plan de l'ACP")
    figures["profils_acp"] = _sauver_fig(fig, "fig_profils_acp")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    cols = [couleurs[n] for n in prof["profil"]]
    b1 = a1.bar(prof["profil"], prof["frequence_pour_100_ans"], color=cols)
    _etiqueter(a1, b1, _pct(prof["frequence_pour_100_ans"]))  # fréquence annuelle en %
    a1.set_title("Fréquence (sinistres /100 ans)", fontsize=9)
    b2 = a2.bar(prof["profil"], prof["charge_annuelle"], color=cols)
    part = 100 * prof["cout"] / prof["cout"].sum()  # part du coût total portée par chaque profil
    _etiqueter(a2, b2, [f"{v:.0f} €\n({q:.0f} % du coût)" for v, q in zip(prof["charge_annuelle"], part)])
    a2.set_title("Charge annuelle (€ par an d'exposition)", fontsize=9)
    for a in (a1, a2):
        a.tick_params(axis="x", labelrotation=20, labelsize=7)
    figures["profils_risque"] = _sauver_fig(fig, "fig_profils_risque")

    print(f"Étape 6 : ACP ({var['pct_cumule'][1]:.1f} % de variance sur 2 axes), k-means k=3, "
          f"concordance hiérarchique ARI = {ari:.2f}")
    return {"var": var, "charges": charges, "choix_k": choix_k, "prof": prof, "ari": ari,
            "figures": figures, "n_base": len(base), "variables": VARIABLES_SEG}


def section_etape6(e6: dict) -> str:
    f = e6["figures"]
    prof = e6["prof"]
    var = e6["var"]
    def fig(chemin, legende, largeur="0.85"):
        return ("\\begin{figure}[htbp]\\centering\\includegraphics[width=" + largeur + "\\textwidth]{" + chemin + "}\n"
                "\\caption{" + echapper(legende) + "}\\end{figure}\n")

    lignes_v = [[r["composante"], nombre(r["valeur_propre"], 3), nombre(r["pct_variance"], 1),
                 nombre(r["pct_cumule"], 1)] for r in var.to_dict("records")]
    lignes_k = [[nombre(r["k"]), nombre(r["inertie"], 0), nombre(r["silhouette"], 3)]
                for r in e6["choix_k"].to_dict("records")]
    lignes_p = []
    for r in prof.to_dict("records"):
        lignes_p.append([echapper(r["profil"]), nombre(r["part_polices"], 1), nombre(r["age"], 1),
                         nombre(r["bonus"], 1), nombre(r["poldur"], 1), nombre(r["valeur"], 0),
                         nombre(r["frequence_pour_100_ans"], 2), nombre(r["cout_moyen_par_sinistre"], 0),
                         nombre(r["charge_annuelle"], 0), nombre(r["part_femmes"], 1)])
    p = {r["profil"]: r for r in prof.to_dict("records")}
    faible, moyen, eleve = p["Faible risque"], p["Risque intermédiaire"], p["Risque élevé"]
    return (
        "\n\\part{Étape 6 -- Segmentation et réduction de dimension}\n\\section{Objectif}\n"
        "Identifier des groupes d'assurés aux caractéristiques proches, afin de repérer des profils de risque "
        "(faible, intermédiaire, élevé).\n"
        "\\section{Méthode}\n"
        "\\begin{enumerate}\n"
        "  \\item \\textbf{Variables de segmentation} : âge du conducteur, bonus-malus, ancienneté de la police, "
        "valeur du véhicule (log) et densité de la zone (log). Ce sont des caractéristiques connues à la "
        "souscription. Les sinistres ne sont \\emph{pas} utilisés pour construire les groupes : ils servent "
        "ensuite à vérifier et à nommer les profils, ce qui évite un raisonnement circulaire.\n"
        "  \\item \\textbf{Réduction de dimension (ACP)} : les variables sont centrées et réduites, puis une ACP "
        "résume la structure. Elle sert à lire les groupes et à les visualiser.\n"
        "  \\item \\textbf{Nombre de groupes} : K-means testé de $k=2$ à $k=8$ (inertie et silhouette). "
        "On retient $k=3$, demandé par l'objectif (trois profils).\n"
        "  \\item \\textbf{Validation} : une classification hiérarchique de Ward, sur un échantillon de 3 000 polices, "
        "est comparée au K-means (indice ARI, 1 = concordance parfaite).\n"
        "  \\item \\textbf{Nommage} : les groupes sont classés par charge annuelle moyenne (coût total / exposition). "
        "Le plus faible est « faible risque », le plus élevé « risque élevé ».\n"
        "  \\item \\textbf{Non retenu} : l'analyse factorielle des correspondances multiples (ACM), prévue pour les "
        "variables qualitatives. Le sexe, le type, la catégorie et la profession ne sont donc pas dans la "
        "segmentation. Une ACM pourrait les intégrer dans une étape ultérieure.\n"
        "\\end{enumerate}\n"
        "\\section{A. Réduction de dimension : ACP}\n"
        + fig(f["acp_eboulis"], "Éboulis des valeurs propres : part de variance de chaque composante et cumul.", "0.7")
        + tableau("Valeurs propres de l'ACP", "tab:e6_acp",
                  ["Composante", "Valeur propre", "\\% variance", "\\% cumulé"], lignes_v, "lrrr")
        + fig(f["acp_cercle"], "Cercle des corrélations : contribution des variables aux deux premières composantes.", "0.6")
        + "\\section{B. Choix du nombre de groupes}\n"
        + fig(f["kmeans_k"], "Inertie intra-groupes (bleu) et silhouette (rouge) selon le nombre de groupes $k$.", "0.8")
        + tableau("Inertie et silhouette selon le nombre de groupes", "tab:e6_k",
                  ["$k$", "Inertie", "Silhouette"], lignes_k, "rrr")
        + "\\section{C. Profils d'assurés}\n"
        "Le K-means à $k=3$ produit trois groupes. Concordance avec la classification hiérarchique : ARI = "
        + nombre(e6["ari"], 2) + " (échantillon de 3 000 polices). Cette concordance est \\textbf{modérée} : les "
        "deux méthodes ne tracent pas exactement les mêmes frontières, signe que les groupes se recouvrent.\n"
        + fig(f["profils_acp"], "Polices projetées sur le premier plan de l'ACP, colorées selon leur profil.")
        + tableau("Caractéristiques et sinistralité des trois profils", "tab:e6_profils",
                  ["Profil", "Polices \\%", "Âge", "Bonus", "Ancien.", "Valeur (€)", "Fréq. /100 ans",
                   "Coût moy. (€)", "Charge (€/an)", "\\% femmes"], lignes_p,
                  "p{2.7cm}*{9}{>{\\raggedleft\\arraybackslash}p{1.25cm}}")
        + fig(f["profils_risque"], "Fréquence et charge annuelle moyenne par profil.")
        + "\\section{Interprétation}\n\\begin{itemize}\n"
        f"  \\item \\textbf{{Faible risque}} ({nombre(faible['part_polices'], 1)} \\% des polices) : fréquence de "
        f"{nombre(faible['frequence_pour_100_ans'], 2)} sinistres pour 100 ans, charge annuelle de "
        f"{nombre(faible['charge_annuelle'], 0)} €.\n"
        f"  \\item \\textbf{{Risque intermédiaire}} ({nombre(moyen['part_polices'], 1)} \\% des polices) : fréquence de "
        f"{nombre(moyen['frequence_pour_100_ans'], 2)} sinistres pour 100 ans, charge annuelle de "
        f"{nombre(moyen['charge_annuelle'], 0)} €.\n"
        f"  \\item \\textbf{{Risque élevé}} ({nombre(eleve['part_polices'], 1)} \\% des polices) : fréquence de "
        f"{nombre(eleve['frequence_pour_100_ans'], 2)} sinistres pour 100 ans, charge annuelle de "
        f"{nombre(eleve['charge_annuelle'], 0)} €.\n"
        "\\end{itemize}\n"
        "\\section{Limites}\n\\begin{enumerate}\n"
        "  \\item Le jeu de données ne contient pas de \\emph{prime}. Les profils reflètent donc les caractéristiques "
        "de risque, et non le niveau de tarif réellement appliqué.\n"
        "  \\item Les profils sont des \\emph{groupes statistiques} : ils décrivent des assurés semblables, ils "
        "n'expliquent pas la cause des sinistres.\n"
        "  \\item Le choix $k=3$ est imposé par l'objectif. La silhouette reste faible (environ 0,17 pour "
        "toutes les valeurs de $k$) et la courbe d'inertie n'a pas de coude net : il n'existe pas de nombre de "
        "groupes naturellement meilleur. Les profils sont des découpages utiles, pas des catégories tranchées.\n"
        "  \\item Les variables qualitatives ne sont pas dans la segmentation (voir Méthode).\n"
        "\\end{enumerate}\n"
    )


# ===========================================================================
# ÉTAPE 7 – MODÉLISATION STATISTIQUE ET APPRENTISSAGE AUTOMATIQUE
# ===========================================================================


NUM_MOD = ["Age", "Bonus", "Poldur", "Value", "Density"]


CAT_MOD = ["Gender", "Type", "Category", "Occupation", "Group1"]


LIBELLES_7 = {"Age": "Âge", "Bonus": "Bonus-malus", "Poldur": "Ancienneté", "Value": "Valeur du véhicule",
              "Density": "Densité de la zone", "Gender": "Sexe", "Type": "Type", "Category": "Catégorie",
              "Occupation": "Profession", "Group1": "Zone géographique"}


def _pipe_glm(modele):
    ct = ColumnTransformer([("num", StandardScaler(), NUM_MOD),
                            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_MOD)])
    return Pipeline([("prep", ct), ("modele", modele)])


def _pipe_arbre(categories, modele):
    ct = ColumnTransformer([("cat", OrdinalEncoder(categories=categories), CAT_MOD),
                            ("num", "passthrough", NUM_MOD)])
    return Pipeline([("prep", ct), ("modele", modele)])


def _ajuster(fabrique, X, y, w=None):
    m = fabrique()
    if w is None:
        m.fit(X, y)
    else:
        m.fit(X, y, modele__sample_weight=w)
    return m


def _dev_pois(y, p, w):
    return mean_poisson_deviance(y, p, sample_weight=w)


def _dev_gamma(y, p, w=None):
    return mean_gamma_deviance(y, p, sample_weight=w)


def _cv(fabrique, X, y, w, deviance):
    kf = KFold(n_splits=5, shuffle=True, random_state=GRAINE)
    scores = []
    for a, b in kf.split(X):
        m = _ajuster(fabrique, X.iloc[a], y[a], None if w is None else w[a])
        scores.append(deviance(y[b], m.predict(X.iloc[b]), None if w is None else w[b]))
    return float(np.mean(scores)), float(np.std(scores))


def _importance(m, X, y, w, deviance, rng):
    """Hausse moyenne de la déviance quand on mélange une variable (importance par permutation, test)."""
    ref = deviance(y, m.predict(X), w)
    lignes = []
    for c in NUM_MOD + CAT_MOD:
        hausses = []
        for _ in range(3):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            hausses.append(deviance(y, m.predict(Xp), w) - ref)
        lignes.append({"variable": c, "libelle": LIBELLES_7[c], "hausse_deviance": float(np.mean(hausses)),
                       "hausse_relative_pct": 100 * float(np.mean(hausses)) / ref})
    return pd.DataFrame(lignes).sort_values("hausse_relative_pct", ascending=False).reset_index(drop=True)


def _libelle_coef(nom):
    if nom.startswith("num__"):
        return f"{LIBELLES_7[nom[5:]]} (+1 écart-type)"
    brut = nom[5:]
    for v in CAT_MOD:
        if brut.startswith(v + "_"):
            return f"{LIBELLES_7[v]} = {brut[len(v) + 1:]}"
    return brut


def _metriques_cout(pred, reel):
    ordre = np.argsort(-pred)
    k = int(0.1 * len(pred))
    return {"ae_total": float(pred.sum() / reel.sum()),
            "mae_par_police": float(mean_absolute_error(reel, pred)),
            "part_cout_top10": float(100 * reel[ordre[:k]].sum() / reel.sum())}


def etape7(e3: dict) -> dict:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng(GRAINE)
    figures = {}

    base = e3["base"].copy()
    n_brut = len(base)
    base = base[(base["expo_annees"].astype(float) > 0) & (base["Gender"].astype(str) != "Inconnu")]
    base = base.reset_index(drop=True)
    for c in NUM_MOD:
        base[c] = base[c].astype(float)
    for c in CAT_MOD:
        base[c] = base[c].astype(str)
        # Modalités rares (moins de 100 polices) regroupées : évite des modalités absentes de l'apprentissage
        effectifs = base[c].value_counts()
        rares = effectifs[effectifs < 100].index
        base.loc[base[c].isin(rares), c] = "Autre"
    cats = [sorted(base[c].unique()) for c in CAT_MOD]
    base["expo"] = base["expo_annees"].astype(float)
    base["nb"] = base["nb_sinistres"].astype(float)
    base["cout"] = base["cout_total"].astype(float)

    idx_tr, idx_te = train_test_split(np.arange(len(base)), test_size=0.2, random_state=GRAINE)
    tr, te = base.iloc[idx_tr], base.iloc[idx_te]
    X_tr, X_te = tr[NUM_MOD + CAT_MOD], te[NUM_MOD + CAT_MOD]
    r_tr = (tr["nb"] / tr["expo"]).to_numpy()
    r_te = (te["nb"] / te["expo"]).to_numpy()
    w_tr, w_te = tr["expo"].to_numpy(), te["expo"].to_numpy()
    nb_te = te["nb"].to_numpy()

    # Sinistres à coût positif, rattachés aux caractéristiques de leur police
    sin = e3["sin"]
    # Une ligne de sinistre peut regrouper plusieurs sinistres (nb_sin) : le coût par sinistre est chg_sin / nb_sin,
    # et chaque ligne pèse nb_sin dans l'apprentissage.
    s = sin.loc[(sin["chg_sin"].astype(float) > 0) & (sin["nb_sin"].astype(float) > 0),
                ["PolNum", "chg_sin", "nb_sin"]].copy()
    s["chg_sin"] = s["chg_sin"].astype(float)
    s["nb_sin"] = s["nb_sin"].astype(float)
    s["cout_par_sinistre"] = s["chg_sin"] / s["nb_sin"]
    s = s.merge(base[["PolNum"] + NUM_MOD + CAT_MOD], on="PolNum", how="inner")
    ids_tr, ids_te = set(tr["PolNum"]), set(te["PolNum"])
    s_tr, s_te = s[s["PolNum"].isin(ids_tr)], s[s["PolNum"].isin(ids_te)]
    Xs_tr, ys_tr, ws_tr = s_tr[NUM_MOD + CAT_MOD], s_tr["cout_par_sinistre"].to_numpy(), s_tr["nb_sin"].to_numpy()
    Xs_te, ys_te, ws_te = s_te[NUM_MOD + CAT_MOD], s_te["cout_par_sinistre"].to_numpy(), s_te["nb_sin"].to_numpy()

    split = pd.DataFrame([
        {"jeu": "Polices apprentissage", "effectif": len(tr), "sinistres": int(tr["nb"].sum())},
        {"jeu": "Polices test", "effectif": len(te), "sinistres": int(te["nb"].sum())},
        {"jeu": "Lignes de sinistres à coût positif, apprentissage", "effectif": len(s_tr), "sinistres": int(ws_tr.sum())},
        {"jeu": "Lignes de sinistres à coût positif, test", "effectif": len(s_te), "sinistres": int(ws_te.sum())},
    ])
    split.to_csv(SORTIE / "etape7_split.csv", index=False, sep=";")

    # ---- Fréquence : nombre de sinistres par année d'exposition --------------------------
    cand_f = [
        ("Constante (référence)", lambda: _pipe_arbre(cats, DummyRegressor(strategy="mean"))),
        ("GLM Poisson", lambda: _pipe_glm(PoissonRegressor(alpha=1e-4, max_iter=500))),
        ("Forêt aléatoire", lambda: _pipe_arbre(cats, RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=200, n_jobs=-1, random_state=GRAINE))),
        ("Gradient boosting Poisson", lambda: _pipe_arbre(cats, HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200,
            categorical_features=list(range(len(CAT_MOD))), random_state=GRAINE))),
    ]
    lignes_f, modeles_f = [], {}
    for nom, fab in cand_f:
        cv_m, cv_s = _cv(fab, X_tr, r_tr, w_tr, _dev_pois)
        m = _ajuster(fab, X_tr, r_tr, w_tr)
        d_tr = _dev_pois(r_tr, m.predict(X_tr), w_tr)
        p_te = m.predict(X_te)
        d_te = _dev_pois(r_te, p_te, w_te)
        lignes_f.append({"modele": nom, "deviance_cv": cv_m, "ecart_type_cv": cv_s,
                         "deviance_apprentissage": d_tr, "deviance_test": d_te,
                         "surapprentissage_pct": 100 * (d_te - d_tr) / d_tr,
                         "ae_test": float((p_te * w_te).sum() / nb_te.sum())})
        modeles_f[nom] = m
    res_f = pd.DataFrame(lignes_f)
    res_f.to_csv(SORTIE / "etape7_frequence_modeles.csv", index=False, sep=";", decimal=",")
    best_f = res_f.loc[res_f["deviance_cv"].idxmin(), "modele"]

    # ---- Coût moyen par sinistre : régression gamma -----------------------------------------
    cand_s = [
        ("Constante (référence)", lambda: _pipe_arbre(cats, DummyRegressor(strategy="mean"))),
        ("GLM Gamma", lambda: _pipe_glm(GammaRegressor(alpha=1e-4, max_iter=500))),
        ("Forêt aléatoire", lambda: _pipe_arbre(cats, RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=GRAINE))),
        ("Gradient boosting Gamma", lambda: _pipe_arbre(cats, HistGradientBoostingRegressor(
            loss="gamma", learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=50,
            categorical_features=list(range(len(CAT_MOD))), random_state=GRAINE))),
    ]
    lignes_s, modeles_s = [], {}
    for nom, fab in cand_s:
        cv_m, cv_s = _cv(fab, Xs_tr, ys_tr, ws_tr, _dev_gamma)
        m = _ajuster(fab, Xs_tr, ys_tr, ws_tr)
        d_tr = _dev_gamma(ys_tr, m.predict(Xs_tr), ws_tr)
        p_te = m.predict(Xs_te)
        d_te = _dev_gamma(ys_te, p_te, ws_te)
        lignes_s.append({"modele": nom, "deviance_cv": cv_m, "ecart_type_cv": cv_s,
                         "deviance_apprentissage": d_tr, "deviance_test": d_te,
                         "surapprentissage_pct": 100 * (d_te - d_tr) / d_tr,
                         "mae_test": float(mean_absolute_error(ys_te, p_te, sample_weight=ws_te)),
                         "ae_test": float((p_te * ws_te).sum() / (ys_te * ws_te).sum())})
        modeles_s[nom] = m
    res_s = pd.DataFrame(lignes_s)
    res_s.to_csv(SORTIE / "etape7_severite_modeles.csv", index=False, sep=";", decimal=",")
    best_s = res_s.loc[res_s["deviance_cv"].idxmin(), "modele"]

    # ---- Coût total par police : fréquence × coût moyen ----------------------------------
    m_f, m_s = modeles_f[best_f], modeles_s[best_s]
    c_f, c_s = modeles_f["Constante (référence)"], modeles_s["Constante (référence)"]
    e_te, c_te = te["expo"].to_numpy(), te["cout"].to_numpy()
    pred_modele = m_f.predict(X_te) * e_te * m_s.predict(X_te)
    pred_ref = c_f.predict(X_te) * e_te * c_s.predict(X_te)
    tot = pd.DataFrame([
        {"approche": f"{best_f} × {best_s}", **_metriques_cout(pred_modele, c_te)},
        {"approche": "Référence constante", **_metriques_cout(pred_ref, c_te)},
    ])
    tot.to_csv(SORTIE / "etape7_cout_total_resultats.csv", index=False, sep=";", decimal=",")

    dec = pd.Series(pred_modele).rank(method="first")
    dec = (pd.qcut(dec, 10, labels=False) + 1).to_numpy()
    deciles = pd.DataFrame({"decile": dec, "pred": pred_modele, "reel": c_te}).groupby("decile").agg(
        polices=("pred", "size"), cout_predit_moyen=("pred", "mean"), cout_reel_moyen=("reel", "mean"),
        cout_reel_total=("reel", "sum")).reset_index()
    deciles["part_cout_reel_pct"] = 100 * deciles["cout_reel_total"] / deciles["cout_reel_total"].sum()
    deciles.drop(columns="cout_reel_total").to_csv(SORTIE / "etape7_cout_total_deciles.csv", index=False,
                                                   sep=";", decimal=",")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(deciles["decile"], deciles["cout_reel_moyen"], color="#C7C7C7",
           label="Coût réel moyen (étiquette : part du coût total)")
    haut_max = max(deciles["cout_reel_moyen"].max(), deciles["cout_predit_moyen"].max())
    for d, reel, pred, p in deciles[["decile", "cout_reel_moyen", "cout_predit_moyen", "part_cout_reel_pct"]].itertuples(index=False):
        # étiquette au-dessus du bâton et du point rouge, pour ne pas être masquée par la courbe
        ax.text(d, max(reel, pred) + 0.03 * haut_max, f"{p:.1f} %".replace(".", ","),
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.plot(deciles["decile"], deciles["cout_predit_moyen"], "o-", color="#C44E52", label="Coût prédit moyen")
    ax.set_xlabel("Décile de coût prédit (1 = plus faible, 10 = plus élevé)"); ax.set_ylabel("€ par police et par an")
    ax.set_xticks(range(1, 11)); ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15 * max(deciles["cout_reel_moyen"].max(), deciles["cout_predit_moyen"].max()))  # place pour les étiquettes
    ax.set_title("Coût total annuel : prédit et réel par décile")
    figures["deciles"] = _sauver_fig(fig, "fig_cout_deciles")

    # ---- Importance des variables ---------------------------------------------------------
    imp_f = _importance(m_f, X_te, r_te, w_te, _dev_pois, rng)
    imp_s = _importance(m_s, Xs_te, ys_te, ws_te, _dev_gamma, rng)
    imp_f.to_csv(SORTIE / "etape7_importance_frequence.csv", index=False, sep=";", decimal=",")
    imp_s.to_csv(SORTIE / "etape7_importance_severite.csv", index=False, sep=";", decimal=",")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, imp, titre in [(a1, imp_f, "Fréquence"), (a2, imp_s, "Coût moyen par sinistre")]:
        o = imp.sort_values("hausse_relative_pct")
        ax.barh(o["libelle"], o["hausse_relative_pct"], color="#4C72B0")
        vmax = max(o["hausse_relative_pct"].max(), 0.1)
        for i, v in enumerate(o["hausse_relative_pct"]):
            # étiquette toujours à droite de zéro ; « ≈ 0 % » quand la variable n'apporte rien
            texte = "≈ 0 %" if abs(v) < 0.05 else f"{v:.1f} %".replace(".", ",")
            ax.text(max(v, 0) + 0.02 * vmax, i, texte, va="center", fontsize=6)
        ax.set_xlim(0, 1.25 * vmax)  # place pour les étiquettes
        ax.set_title(titre, fontsize=9); ax.tick_params(labelsize=7)
        ax.set_xlabel("Hausse de déviance (%)", fontsize=8)
    fig.suptitle("Importance des variables par permutation (test)", fontsize=9)
    figures["importance"] = _sauver_fig(fig, "fig_importance_variables")

    # ---- Coefficients du GLM Poisson retenu comme référence interprétable ------------------
    m_glm = modeles_f["GLM Poisson"]
    noms = m_glm.named_steps["prep"].get_feature_names_out()
    coefs = m_glm.named_steps["modele"].coef_
    glm_rr = pd.DataFrame({"facteur": [_libelle_coef(n) for n in noms], "coefficient": coefs,
                           "rapport_de_taux": np.exp(coefs)})
    glm_rr.to_csv(SORTIE / "etape7_glm_frequence_coefficients.csv", index=False, sep=";", decimal=",")
    cats = m_glm.named_steps["prep"].named_transformers_["cat"].categories_
    references = [(LIBELLES_7[v], str(cats[i][0])) for i, v in enumerate(CAT_MOD)]

    def _dev_test(df, nom):
        return float(df.loc[df["modele"] == nom, "deviance_test"].iloc[0])
    gain_f = 100 * (1 - _dev_test(res_f, best_f) / _dev_test(res_f, "Constante (référence)"))
    gain_s = 100 * (1 - _dev_test(res_s, best_s) / _dev_test(res_s, "Constante (référence)"))
    print(f"Étape 7 : {len(tr)} polices d'apprentissage, {len(te)} de test, {int(ws_tr.sum())} sinistres d'apprentissage ; "
          f"fréquence retenue : {best_f} ; coût retenu : {best_s} ; A/E total test = "
          f"{tot.loc[0, 'ae_total']:.3f} ; {n_brut - len(base)} police(s) exclue(s)")
    return {"split": split, "res_f": res_f, "res_s": res_s, "best_f": best_f, "best_s": best_s,
            "total": tot, "deciles": deciles, "imp_f": imp_f, "imp_s": imp_s, "glm_rr": glm_rr,
            "references": references, "figures": figures, "n_exclus": n_brut - len(base),
            "modele_frequence": m_f, "X_te": X_te, "te": te, "gain_f": gain_f, "gain_s": gain_s,
            "n_tr": len(tr), "n_te": len(te), "n_s_tr": int(ws_tr.sum()), "n_s_te": int(ws_te.sum())}


def section_etape7(e7: dict) -> str:
    f = e7["figures"]

    def tab_petit(*args):
        return "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + tableau(*args) + "}\n"

    res_f, res_s, tot = e7["res_f"], e7["res_s"], e7["total"]
    imp_f, imp_s = e7["imp_f"], e7["imp_s"]
    best_f, best_s = e7["best_f"], e7["best_s"]

    def fig(chemin, legende, largeur="0.85"):
        return ("\\begin{figure}[htbp]\\centering\\includegraphics[width=" + largeur + "\\textwidth]{" + chemin + "}\n"
                "\\caption{" + echapper(legende) + "}\\end{figure}\n")

    lignes_f = [[echapper(r["modele"]), nombre(r["deviance_cv"], 4), nombre(r["ecart_type_cv"], 4),
                 nombre(r["deviance_apprentissage"], 4), nombre(r["deviance_test"], 4),
                 nombre(r["surapprentissage_pct"], 1), nombre(r["ae_test"], 3)] for r in res_f.to_dict("records")]
    lignes_s = [[echapper(r["modele"]), nombre(r["deviance_cv"], 3), nombre(r["ecart_type_cv"], 3),
                 nombre(r["deviance_apprentissage"], 3), nombre(r["deviance_test"], 3),
                 nombre(r["surapprentissage_pct"], 1), nombre(r["mae_test"], 0), nombre(r["ae_test"], 3)]
                for r in res_s.to_dict("records")]
    lignes_t = [[echapper(r["approche"]), nombre(r["ae_total"], 3), nombre(r["mae_par_police"], 1),
                 nombre(r["part_cout_top10"], 1)] for r in tot.to_dict("records")]
    imp = imp_f[["variable", "libelle"]].copy()
    imp["freq"] = imp["variable"].map(dict(zip(imp_f["variable"], imp_f["hausse_relative_pct"])))
    imp["sev"] = imp["variable"].map(dict(zip(imp_s["variable"], imp_s["hausse_relative_pct"])))
    imp = imp.sort_values("freq", ascending=False)
    lignes_i = [[echapper(r["libelle"]), nombre(r["freq"], 1), nombre(r["sev"], 1)] for r in imp.to_dict("records")]
    lignes_g = [[echapper(r["facteur"]), nombre(r["rapport_de_taux"], 3)] for r in e7["glm_rr"].to_dict("records")]
    refs = ", ".join(echapper(a) + " : " + echapper(b) for a, b in e7["references"])
    def deviance_test(df, nom):
        return float(df.loc[df["modele"] == nom, "deviance_test"].iloc[0])
    gain_f = 100 * (1 - deviance_test(res_f, best_f) / deviance_test(res_f, "Constante (référence)"))
    gain_s = 100 * (1 - deviance_test(res_s, best_s) / deviance_test(res_s, "Constante (référence)"))
    surappr_gb_f = float(res_f.loc[res_f["modele"] == "Gradient boosting Poisson", "surapprentissage_pct"].iloc[0])
    surappr_gb_s = float(res_s.loc[res_s["modele"] == "Gradient boosting Gamma", "surapprentissage_pct"].iloc[0])
    top3_sev = ", ".join(imp_s["libelle"].head(3).tolist()).lower()
    ligne_modele = tot.iloc[0]
    ligne_ref = tot.iloc[1]
    top3 = ", ".join(imp_f["libelle"].head(3).tolist()).lower()
    return (
        "\n\\part{Étape 7 -- Modélisation statistique et apprentissage automatique}\n\\section{Objectif}\n"
        "Prévoir le coût total annuel attendu par police. Ce coût se décompose en \\textbf{fréquence} "
        "(nombre de sinistres par année d'exposition) et \\textbf{coût moyen par sinistre}, le modèle "
        "« fréquence × coût » utilisé en tarification.\n"
        "\\section{Choix des modèles}\n"
        + tab_petit("Problématique, modèles candidats et métriques", "tab:e7_choix",
                  ["Composante", "Modèles candidats", "Métrique"],
                  [["Fréquence (sinistres / exposition)", "Constante, GLM Poisson, forêt aléatoire, gradient boosting Poisson",
                    "Déviance de Poisson pondérée par l'exposition, ratio A/E"],
                   ["Coût moyen par sinistre", "Constante, GLM Gamma, forêt aléatoire, gradient boosting gamma",
                    "Déviance gamma, erreur absolue moyenne, ratio A/E"],
                   ["Coût total par police", "Fréquence × coût moyen",
                    "Ratio A/E total, erreur absolue par police, part du coût dans le décile supérieur"]],
                  "p{3.4cm}p{5.6cm}p{5.6cm}")
        + "La survenue d'un sinistre n'est pas modélisée à part : le modèle de fréquence la couvre, une police "
        "sans sinistre ayant une fréquence nulle.\n"
        "\\section{Méthode}\n"
        "\\begin{enumerate}\n"
        "  \\item \\textbf{Données} : polices avec exposition positive, hors sexe inconnu ("
        + str(e7["n_exclus"]) + " polices exclues au total). Les sinistres sont ceux à coût positif, sans plafonnement : "
        "les montants extrêmes sont conservés, comme demandé.\n"
        "  \\item \\textbf{Séparation} : " + nombre(e7["n_tr"]) + " polices d'apprentissage et "
        + nombre(e7["n_te"]) + " polices de test (80 / 20, tirage aléatoire à graine fixe). Les "
        + nombre(e7["n_s_tr"]) + " sinistres d'apprentissage et les " + nombre(e7["n_s_te"])
        + " sinistres de test suivent leur police.\n"
        "  \\item \\textbf{Choix des modèles} : validation croisée à 5 blocs sur l'apprentissage uniquement. "
        "Le jeu de test n'a servi ni au choix ni au réglage.\n"
        "  \\item \\textbf{Surapprentissage} : comparaison entre déviance d'apprentissage et déviance de test.\n"
        "  \\item \\textbf{Variables} : elles sont les mêmes pour tous les modèles : âge, bonus-malus, ancienneté, "
        "valeur du véhicule, densité, sexe, type, catégorie, profession et zone géographique. L'année "
        "calendaire n'est pas utilisée (deux années seulement).\n"
        "\\end{enumerate}\n"
        "\\section{A. Modèles de fréquence}\n"
        + tab_petit("Fréquence : validation croisée et test (plus la déviance est faible, mieux c'est)", "tab:e7_freq",
                  ["Modèle", "Déviance CV", "Écart-type CV", "Déviance app.", "Déviance test", "Surappr. (\\%)", "A/E test"],
                  lignes_f, "p{3.8cm}*{6}{>{\\raggedleft\\arraybackslash}p{1.6cm}}")
        + "Modèle retenu : \\textbf{" + echapper(best_f) + "} (déviance de validation croisée la plus faible). "
        "Le ratio A/E compare le nombre de sinistres prédit au nombre observé ; 1 signifie une prévision juste en moyenne.\n"
        "\\section{B. Modèles de coût moyen par sinistre}\n"
        + tab_petit("Coût moyen par sinistre : validation croisée et test", "tab:e7_sev",
                  ["Modèle", "Déviance CV", "Écart-type CV", "Déviance app.", "Déviance test", "Surappr. (\\%)", "MAE test (€)", "A/E test"],
                  lignes_s, "p{3.4cm}*{7}{>{\\raggedleft\\arraybackslash}p{1.4cm}}")
        + "Modèle retenu : \\textbf{" + echapper(best_s) + "}. Le coût moyen d'un sinistre est très dispersé "
        "(queue longue) : l'erreur absolue moyenne reste élevée, même pour le meilleur modèle. Le gain par rapport à "
        "la référence constante est modeste : la déviance de test baisse d'environ " + nombre(gain_s, 1) + " \\%.\n"
        "\\section{C. Coût total annuel par police}\n"
        "Le coût total prédit est le produit : fréquence prédite × exposition × coût moyen prédit. Il est comparé "
        "au coût réel sur le jeu de test, avec une référence constante.\n"
        + tab_petit("Coût total annuel : modèle fréquence × coût contre référence constante", "tab:e7_total",
                  ["Approche", "A/E total", "Erreur abs. par police (€)", "Part coût réel, top décile (\\%)"],
                  lignes_t, "p{5.2cm}*{3}{>{\\raggedleft\\arraybackslash}p{2.6cm}}")
        + fig(f["deciles"], "Coût annuel moyen prédit et réel par décile de coût prédit (jeu de test).")
        + "\\section{D. Surapprentissage}\n"
        "Le surapprentissage se mesure par l'écart entre déviance d'apprentissage et déviance de test. "
        "Les colonnes « Surappr. » des tableaux ci-dessus donnent cet écart en pourcentage. Un écart de quelques "
        "pour cent est normal ; un écart important signalerait un modèle qui mémorise l'apprentissage.\n"
        "Le gradient boosting présente l'écart le plus élevé : " + nombre(surappr_gb_f, 1) + " \\% pour la fréquence et "
        + nombre(surappr_gb_s, 1) + " \\% pour le coût. Pour la fréquence, il reste le meilleur en validation croisée "
        "et il est retenu ; cet écart justifie de le surveiller si le modèle est utilisé en production.\n"
        "\\section{E. Variables importantes et interprétation}\n"
        "L'importance par permutation mesure la hausse de déviance quand la variable est mélangée au hasard dans "
        "le jeu de test : plus la hausse est grande, plus le modèle dépend de la variable.\n"
        + tab_petit("Importance des variables par permutation (hausse de déviance, en %)", "tab:e7_imp",
                  ["Variable", "Fréquence (\\%)", "Coût moyen (\\%)"], lignes_i, "lrr")
        + fig(f["importance"], "Importance des variables par permutation sur le jeu de test.")
        + "Pour lire les effets un par un, le GLM Poisson donne des rapports de taux. Pour les variables numériques, "
        "le rapport correspond à une augmentation d'un écart-type. Pour les variables qualitatives, il est comparé à "
        "la modalité de référence : " + refs + ".\n"
        + tab_petit("GLM Poisson : rapports de taux de fréquence", "tab:e7_glm",
                  ["Facteur", "Rapport de taux"], lignes_g, "p{7cm}r")
        + "\\section{Interprétation}\n\\begin{itemize}\n"
        "  \\item Les variables les plus importantes pour la fréquence sont : " + top3 + ".\n"
        "  \\item Le modèle retenu répartit mieux le coût total que la référence constante : le décile supérieur "
        "concentre " + nombre(ligne_modele["part_cout_top10"], 1) + " \\% du coût réel, contre "
        + nombre(ligne_ref["part_cout_top10"], 1) + " \\% pour la référence.\n"
        "  \\item Pour le coût moyen par sinistre, les variables expliquent peu de choses : la déviance de test ne "
        "baisse que d'environ " + nombre(gain_s, 1) + " \\% par rapport à la référence constante. Les variables les "
        "plus liées au coût sont : " + top3_sev + ".\n"
        "  \\item Pour la fréquence, la déviance de test baisse d'environ " + nombre(gain_f, 1)
        + " \\% par rapport à la référence constante.\n"
        "  \\item Le ratio A/E total du modèle retenu vaut " + nombre(ligne_modele["ae_total"], 3)
        + " sur le jeu de test : le niveau moyen du coût total est bien calé.\n"
        "\\end{itemize}\n"
        "\\section{Limites}\n\\begin{enumerate}\n"
        "  \\item Pas de colonne prime : le modèle prédit un coût, pas un tarif. Le bonus-malus, qui est un facteur "
        "tarifaire, reste dans les variables explicatives.\n"
        "  \\item Les sinistres n'ont pas de date : la validation est aléatoire, pas temporelle.\n"
        "  \\item Le nombre de sinistres est limité (" + nombre(e7["n_s_te"]) + " sinistres de test). Les écarts "
        "entre modèles de coût peuvent donc être sensibles à l'échantillon.\n"
        "  \\item Les montants extrêmes sont conservés : ils pèsent fortement sur la déviance gamma et l'erreur absolue.\n"
        "  \\item Les modèles décrivent des associations, pas des causes.\n"
        "\\end{enumerate}\n"
        "\\section{Points à valider}\n\\begin{enumerate}\n"
        "  \\item Confirmer l'existence d'une prime ou d'un tarif, pour calibrer le coût prédit sur le tarif.\n"
        "  \\item Disposer des dates de sinistre, pour une validation temporelle (apprentissage sur le passé, test sur le futur).\n"
        "\\end{enumerate}\n"
    )


# ===========================================================================
# ÉTAPE 8 – INTERPRÉTATION ACTUARIELLE ET RECOMMANDATIONS
# ===========================================================================


def etape8(res: dict) -> dict:
    e3, e7 = res["etape3"], res["etape7"]
    sin = e3["sin"]
    figures = {}

    # Concentration du coût : une ligne de sinistre regroupe nb_sin sinistres, coût par sinistre = chg_sin / nb_sin
    s = sin.loc[(sin["chg_sin"].astype(float) > 0) & (sin["nb_sin"].astype(float) > 0),
                ["chg_sin", "nb_sin"]].astype(float)
    cout_par_sin = np.repeat(s["chg_sin"].to_numpy() / s["nb_sin"].to_numpy(), s["nb_sin"].astype(int).to_numpy())
    cout_par_sin = np.sort(cout_par_sin)[::-1]
    total = cout_par_sin.sum()
    lignes_c = []
    for p in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(round(p * len(cout_par_sin))))
        lignes_c.append({"part_des_sinistres_pct": 100 * p, "part_du_cout_pct": 100 * cout_par_sin[:k].sum() / total})
    conc = pd.DataFrame(lignes_c)
    conc.to_csv(SORTIE / "etape8_concentration_couts.csv", index=False, sep=";", decimal=",")

    cumul = np.cumsum(cout_par_sin) / total
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(100 * np.arange(1, len(cumul) + 1) / len(cumul), 100 * cumul, color="#4C72B0")
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=0.8)
    ax.set_xlabel("Part des sinistres (les plus coûteux d'abord, %)"); ax.set_ylabel("Part du coût total (%)")
    ax.set_title("Concentration du coût des sinistres (courbe de Lorenz)")
    figures["concentration"] = _sauver_fig(fig, "fig_concentration_couts")

    # Polices atypiques : sinistralité observée très supérieure à la prévision du modèle (jeu de test)
    te = e7["te"]
    mu = e7["modele_frequence"].predict(e7["X_te"]) * te["expo"].to_numpy()
    nb = te["nb"].to_numpy()
    cout = te["cout"].to_numpy()
    p_queue = sstats.poisson.sf(nb - 1, mu)  # P(X >= nb) sous le modèle de Poisson
    atyp = (nb >= 2) & (p_queue < 0.001)
    liste = te.loc[atyp, ["PolNum", "nb", "expo", "cout"]].copy()
    liste["sinistres_attendus"] = mu[atyp]
    liste["probabilite_queue"] = p_queue[atyp]
    liste.sort_values("probabilite_queue").to_csv(SORTIE / "etape8_polices_atypiques.csv", index=False,
                                                  sep=";", decimal=",")
    atyp_synth = {"n_test": len(te), "n_atyp": int(atyp.sum()), "part_polices": 100 * atyp.mean(),
                  "sinistres_observes": float(nb[atyp].sum()), "sinistres_attendus": float(mu[atyp].sum()),
                  "part_sinistres": 100 * nb[atyp].sum() / nb.sum(),
                  "part_cout": 100 * cout[atyp].sum() / cout.sum()}
    pd.DataFrame([atyp_synth]).to_csv(SORTIE / "etape8_atypiques_synthese.csv", index=False, sep=";", decimal=",")

    print(f"Étape 8 : {atyp_synth['n_atyp']} polices atypiques sur {len(te)} polices de test ; "
          f"{conc.loc[0, 'part_du_cout_pct']:.1f} % du coût dans le 1 % des sinistres les plus coûteux")
    return {"conc": conc, "atyp": atyp_synth, "figures": figures}


def section_etape8(res: dict) -> str:
    e5, e6, e7, e8 = res["etape5"], res["etape6"], res["etape7"], res["etape8"]
    f = e8["figures"]
    glm = e5["glm"].set_index("facteur")
    prof = {r["profil"]: r for r in e6["prof"].to_dict("records")}
    faible, moyen, eleve = prof["Faible risque"], prof["Risque intermédiaire"], prof["Risque élevé"]
    t_type = e5["t_type"].set_index("Type")
    t_age = e5["t_age"].set_index("tranche_age")
    t_anc = e5["t_anc"].set_index("tranche_anciennete")
    cible = e5["cible"].set_index("variable")["rho_avec_nb_sinistres"]
    conc = e8["conc"].set_index("part_des_sinistres_pct")["part_du_cout_pct"]
    at = e8["atyp"]
    res_f, res_s, tot = e7["res_f"], e7["res_s"], e7["total"]

    def rr(nom):
        return nombre(float(glm.loc[nom, "rapport_de_taux"]), 2)

    def pct(x, d=1):
        return nombre(x, d) + " \\%"

    rr_double_valeur = float(glm.loc["Valeur du véhicule (par +1 log)", "rapport_de_taux"]) ** np.log(2)
    charge_A = t_type.loc["A", "frequence_pour_100_ans"] * t_type.loc["A", "cout_moyen_par_sinistre"] / 100
    charge_F = t_type.loc["F", "frequence_pour_100_ans"] * t_type.loc["F", "cout_moyen_par_sinistre"] / 100

    lignes_f = [[echapper(r["facteur"]), nombre(r["rapport_de_taux"], 2), nombre(r["ic95_bas"], 2) + " – " + nombre(r["ic95_haut"], 2),
                 "oui" if r["p_valeur"] < 0.05 else "non"]
                for r in e5["glm"].to_dict("records") if r["facteur"] != "Constante"]
    lignes_p = [[echapper(r["profil"]), nombre(r["part_polices"], 1), nombre(r["age"], 1), nombre(r["bonus"], 1),
                 nombre(r["poldur"], 1), nombre(r["frequence_pour_100_ans"], 2), nombre(r["charge_annuelle"], 0)]
                for r in e6["prof"].to_dict("records")]
    lignes_r = [[echapper(v), nombre(x, 3)] for v, x in cible.items()]
    lignes_c = [[nombre(r["part_des_sinistres_pct"], 0) + " \\%", nombre(r["part_du_cout_pct"], 1) + " \\%"]
                for r in e8["conc"].to_dict("records")]
    atyp_txt = nombre(at["n_atyp"])
    refs = ", ".join(echapper(a) + " : " + echapper(b) for a, b in e7["references"])

    def fig(chemin, legende, largeur="0.8"):
        return ("\\begin{figure}[htbp]\\centering\\includegraphics[width=" + largeur + "\\textwidth]{" + chemin + "}\n"
                "\\caption{" + echapper(legende) + "}\\end{figure}\n")

    def tab_petit(*args):
        return "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + tableau(*args) + "}\n"

    return (
        "\n\\part{Étape 8 -- Interprétation actuarielle et recommandations}\n\\section{Objectif}\n"
        "Transformer les résultats statistiques en lecture de risque : quels facteurs comptent, quels profils "
        "sont les plus sinistrés, quelles limites ont les résultats, et comment une compagnie pourrait les utiliser.\n"
        "\\section{Synthèse}\n\\begin{itemize}\n"
        "  \\item \\textbf{Le risque se concentre} : le décile le plus coûteux de polices concentre "
        + pct(float(e7["total"].iloc[0]["part_cout_top10"])) + " du coût réel, contre "
        + pct(float(e7["total"].iloc[1]["part_cout_top10"])) + " pour une tarification uniforme.\n"
        "  \\item \\textbf{La fréquence est expliquée} : bonus-malus, âge et zone géographique sont les facteurs les plus "
        "liés aux sinistres. La déviance de test baisse d'environ " + pct(e7["gain_f"])
        + " par rapport à une prévision uniforme.\n"
        "  \\item \\textbf{Le coût par sinistre est peu prévisible} : les variables disponibles n'expliquent qu'une "
        "faible part de sa dispersion. La tarification du coût moyen devra s'appuyer sur d'autres informations.\n"
        "  \\item \\textbf{Les résultats sont des associations} : ils proviennent d'une seule base, sans prime ni "
        "dates de sinistre. Ils doivent être validés avant tout usage opérationnel.\n"
        "\\end{itemize}\n"
        "\\section{1. Variables qui influencent le risque}\n"
        "Le modèle de Poisson multivarié (étape 5) mesure l'effet de chaque facteur toutes choses égales par ailleurs. "
        "Un rapport de taux de 1,50 signifie 50 % de sinistres en plus. Les facteurs de référence sont : "
        + refs + " ; âge 36-50 ; ancienneté 5-9 ans ; type A.\n"
        + tab_petit("Effet de chaque facteur sur la fréquence (modèle de Poisson multivarié)", "tab:e8_facteurs",
                    ["Facteur", "Rapport de taux", "IC 95 \\%", "Significatif"], lignes_f, "p{6.8cm}rrr")
        + "\\begin{itemize}\n"
        "  \\item \\textbf{Âge} : les 18-25 ans ont une fréquence " + rr("Âge = 18-25") + " fois celle des 36-50 ans. "
        "Après 50 ans, le risque baisse : " + rr("Âge = 66+") + " pour les 66 ans et plus. C'est le facteur le plus marqué "
        "du modèle.\n"
        "  \\item \\textbf{Bonus-malus} : chaque tranche de 10 points supplémentaires multiplie la fréquence par "
        + rr("Bonus-malus (par +10 points)") + ". Voir la mise en garde sur la causalité en section 3.\n"
        "  \\item \\textbf{Type de véhicule} : le type F a une fréquence " + rr("Type = F") + " fois celle du type A, "
        "et le type E " + rr("Type = E") + " fois.\n"
        "  \\item \\textbf{Valeur du véhicule} : doubler la valeur multiplie la fréquence par environ "
        + nombre(rr_double_valeur, 2) + ". C'est un effet modéré.\n"
        "  \\item \\textbf{Ancienneté} : les polices récentes (0-1 an) ont une fréquence " + rr("Ancienneté = 0-1")
        + " fois celle des polices de 5-9 ans ; les polices de 10 ans et plus, " + rr("Ancienneté = 10+") + " fois.\n"
        "  \\item \\textbf{Sexe} : la fréquence féminine est " + rr("Sexe = Female") + " fois la fréquence masculine. "
        "Attention : l'usage du sexe dans la tarification est interdit dans l'Union européenne depuis l'arrêt "
        "\\emph{Test-Achats} (2011). Cet effet est un constat, pas un critère tarifaire.\n"
        "  \\item \\textbf{Zone géographique} : la fréquence varie fortement selon la zone (étape 7). Les zones les plus "
        "sinistrées ont une fréquence jusqu'à environ deux fois et demie celle de la zone de référence.\n"
        "\\end{itemize}\n"
        "\\section{2. Profils à sinistralité plus importante}\n"
        "Les trois profils de l'étape 6 ont été construits sans les sinistres, puis vérifiés par leur sinistralité : "
        "la fréquence et la charge augmentent bien du profil faible risque au profil élevé.\n"
        + tab_petit("Profils d'assurés : caractéristiques et sinistralité", "tab:e8_profils",
                    ["Profil", "Polices \\%", "Âge", "Bonus", "Ancien.", "Fréq. /100 ans", "Charge (€/an)"],
                    lignes_p, "p{3.6cm}*{6}{>{\\raggedleft\\arraybackslash}p{1.6cm}}")
        + "\\begin{itemize}\n"
        "  \\item \\textbf{Risque élevé} (" + pct(eleve["part_polices"]) + " des polices) : conducteurs plus jeunes "
        "(âge moyen " + nombre(eleve["age"], 1) + " ans), bonus-malus moyen de " + nombre(eleve["bonus"], 1)
        + ", et fréquence de " + nombre(eleve["frequence_pour_100_ans"], 1) + " sinistres pour 100 ans. "
        "Sa charge annuelle, de " + nombre(eleve["charge_annuelle"], 0) + " €, est environ quatre fois celle du profil "
        "faible risque.\n"
        "  \\item \\textbf{Risque intermédiaire} : fréquence de " + nombre(moyen["frequence_pour_100_ans"], 1)
        + " pour 100 ans, avec des polices plus anciennes (" + nombre(moyen["poldur"], 1) + " ans en moyenne).\n"
        "  \\item \\textbf{Faible risque} : conducteurs plus âgés (" + nombre(faible["age"], 1) + " ans), bonus-malus "
        "négatif (" + nombre(faible["bonus"], 1) + "), fréquence de " + nombre(faible["frequence_pour_100_ans"], 1)
        + " pour 100 ans.\n"
        "  \\item La valeur du véhicule est presque identique d'un profil à l'autre : elle ne distingue pas le risque "
        "dans ce portefeuille.\n"
        "\\end{itemize}\n"
        "\\section{3. Relations identifiées}\n"
        "\\begin{enumerate}\n"
        "  \\item \\textbf{Fréquence et coût vont en sens inverse selon le type} : la fréquence augmente du type A au "
        "type F (" + nombre(t_type.loc["A", "frequence_pour_100_ans"], 1) + " à "
        + nombre(t_type.loc["F", "frequence_pour_100_ans"], 1) + " pour 100 ans), alors que le coût moyen par sinistre "
        "baisse (" + nombre(t_type.loc["A", "cout_moyen_par_sinistre"], 0) + " € à "
        + nombre(t_type.loc["F", "cout_moyen_par_sinistre"], 0) + " €). La charge annuelle reste du même ordre "
        "(" + nombre(charge_A, 0) + " € contre " + nombre(charge_F, 0) + " €) : les deux effets se compensent en partie. "
        "Une tarification qui ne regarderait que la fréquence ou que le coût se tromperait.\n"
        "  \\item \\textbf{Âge et sinistralité} : la relation est non linéaire. Le coût moyen par sinistre est de "
        + nombre(t_age.loc["18-25", "cout_moyen_par_sinistre"], 0) + " € chez les 18-25 ans, contre "
        + nombre(t_age.loc["36-50", "cout_moyen_par_sinistre"], 0) + " € chez les 36-50 ans : le risque jeune combine "
        "fréquence et coût.\n"
        "  \\item \\textbf{Corrélations univariées} (Spearman, avec le nombre de sinistres) : bonus-malus "
        + nombre(cible["Bonus"], 3) + ", âge " + nombre(cible["Age"], 3) + ", densité " + nombre(cible["Density"], 3)
        + ". Ces coefficients sont faibles : aucune variable ne porte à elle seule la sinistralité.\n"
        "  \\item \\textbf{Bonus-malus : une relation circulaire} : le bonus-malus évolue avec les sinistres passés. "
        "Son association avec la fréquence reflète donc en partie l'historique de sinistres lui-même, et non un risque "
        "intrinsèque. C'est la relation la plus délicate à interpréter.\n"
        "\\end{enumerate}\n"
        + tab_petit("Corrélation de chaque variable avec le nombre de sinistres", "tab:e8_rho",
                    ["Variable", "Rho de Spearman"], lignes_r, "lr")
        + "\\section{4. Limites du modèle}\n\\begin{enumerate}\n"
        "  \\item \\textbf{Pas de prime} : impossible de comparer les résultats au tarif appliqué, ni de calculer un "
        "ratio sinistres sur primes. Le coût prédit est un coût, pas un tarif.\n"
        "  \\item \\textbf{Pas de dates de sinistre} : pas de tendance dans le temps, pas de provisions pour sinistres "
        "non encore déclarés, et une validation aléatoire au lieu d'une validation dans le temps.\n"
        "  \\item \\textbf{Deux années seulement} d'observation (2009 et 2010), donc une stabilité des effets non vérifiée.\n"
        "  \\item \\textbf{Variables incertaines} : la signification de \\texttt{Type}, de la zone géographique et de la "
        "densité est une hypothèse ; la valeur du véhicule comporte des valeurs imputées ; les codes de sexe sont mixtes.\n"
        "  \\item \\textbf{Exclusions} : les sinistres sans police correspondante (étape 4), les polices au sexe inconnu "
        "et les expositions nulles sont écartés.\n"
        "  \\item \\textbf{Coût par sinistre peu prévisible} : la déviance ne baisse que d'environ "
        + pct(e7["gain_s"]) + " par rapport à une prévision uniforme. "
        "Les montants extrêmes restent dans les données : ils dominent la dispersion et rendent les coûts instables.\n"
        "  \\item \\textbf{Surapprentissage} : le gradient boosting apprend davantage son échantillon que les modèles "
        "linéaires. Il est plus difficile à expliquer et à contrôler.\n"
        "  \\item \\textbf{Indépendance fréquence-coût} : le coût total suppose que la fréquence et le coût moyen sont "
        "indépendants. Cette hypothèse n'a pas été testée.\n"
        "  \\item \\textbf{Associations, pas causes} : un facteur significatif peut être un proxy d'un autre facteur "
        "non observé.\n"
        "  \\item \\textbf{Segmentation peu séparée} : la silhouette des groupes est faible (étape 6).\n"
        "\\end{enumerate}\n"
        "\\section{5. Utilisation par une compagnie d'assurance}\n"
        + tab_petit("Applications possibles et conditions", "tab:e8_applications",
                    ["Application", "Ce que disent les résultats", "Recommandation", "Conditions préalables"],
                    [["Amélioration de la tarification",
                      "Bonus-malus, âge, zone et type ont des effets nets sur la fréquence.",
                      "Construire un tarif technique à partir du modèle de Poisson, et le comparer au tarif en vigueur.",
                      "Primes, historique tarifaire, validation dans le temps, contrôle juridique."],
                     ["Identification des profils à risque",
                      "Le profil élevé représente " + pct(eleve["part_polices"]) + " des polices et une charge quatre fois supérieure.",
                      "Renforcer la souscription pour ce profil : conditions, franchises, contrôle des jeunes conducteurs.",
                      "Seuils d'action définis avec les souscripteurs ; suivi annuel."],
                     ["Gestion des sinistres",
                      "Le 1 \\% des sinistres les plus coûteux représente " + pct(conc[1.0], 1) + " du coût ; le coût par sinistre est peu prévisible.",
                      "Suivre les gros dossiers dès l'ouverture, réviser les provisions de ces dossiers, et étudier la réassurance des pertes extrêmes.",
                      "Dates de déclaration, statuts et montants réservés et payés."],
                     ["Détection des comportements atypiques",
                      atyp_txt + " polices du jeu de test (" + pct(at["part_polices"], 2) + ") ont une sinistralité très "
                      "au-dessus de la prévision.",
                      "Revue manuelle ciblée : contrôle de souscription, vérification des déclarations. Aucune décision automatique.",
                      "Seuil calibré, taux de faux positifs mesuré sur un échantillon."],
                     ["Segmentation du portefeuille",
                      "Trois profils ont été identifiés (étape 6), avec une séparation modérée.",
                      "Utiliser la segmentation en complément du tarif, pour le suivi de portefeuille et le marketing.",
                      "Stabilité des groupes dans le temps ; accepter le recouvrement entre profils."]],
                    "p{3.2cm}p{4.3cm}p{4.6cm}p{3.6cm}")
        + fig(f["concentration"], "Concentration du coût des sinistres : part du coût total en fonction de la part des "
              "sinistres les plus coûteux (par sinistre, en tenant compte de nb_sin).", "0.7")
        + tab_petit("Concentration du coût des sinistres", "tab:e8_conc",
                    ["Part des sinistres les plus coûteux", "Part du coût total"], lignes_c, "rr")
        + "Pour la détection, les polices atypiques sont celles dont le nombre de sinistres dépasse nettement la "
        "prévision du modèle de fréquence (probabilité inférieure à 0,1 \\% sous Poisson). Elles représentent "
        + pct(at["part_sinistres"], 1) + " des sinistres du jeu de test et " + pct(at["part_cout"], 1)
        + " de son coût. Leur liste est fournie à titre de revue, pas comme preuve de fraude.\n"
        "\\section{6. Plan de mise en œuvre recommandé}\n\\begin{enumerate}\n"
        "  \\item \\textbf{Données} : obtenir les primes, les dates de sinistre et de souscription, et la signification "
        "des codes (\\texttt{Type}, zone, densité, sexe).\n"
        "  \\item \\textbf{Validation} : entraîner sur une période et tester sur une période ultérieure ; mesurer la "
        "stabilité des effets d'une année sur l'autre.\n"
        "  \\item \\textbf{Juridique} : vérifier qu'aucun facteur interdit (sexe, et leurs substituts) n'entre dans le tarif. "
        "Le sexe et ses effets doivent être exclus des critères tarifaires.\n"
        "  \\item \\textbf{Suivi} : surveiller la calibration (A/E) et la dérive des variables chaque trimestre.\n"
        "\\end{enumerate}\n"
        "\\section{Points à valider}\n\\begin{enumerate}\n"
        "  \\item Confirmer la présence ou l'absence de primes, et leur niveau.\n"
        "  \\item Confirmer la signification de \\texttt{Type}, de la zone et de la densité.\n"
        "  \\item Faire valider juridiquement les facteurs retenus pour la tarification.\n"
        "\\end{enumerate}\n"
    )


# ===========================================================================
# PROGRAMME PRINCIPAL
# ===========================================================================


def etape1_et_2() -> dict:
    sin_brut = charger_csv_brut()
    feuilles = charger_excel()
    ptf_brut, expo_brut = feuilles["PTF"], feuilles["Expo"]
    sin = typer_sinistres(sin_brut)
    ptf = typer_portefeuille(ptf_brut)

    # Étape 1 : tables telles que lues (Value reste en texte pour la détection de '??').
    tables_e1 = {"sinistres": typer_sinistres(sin_brut), "portefeuille": ptf_brut, "exposition": expo_brut}
    dico = dictionnaire(tables_e1)
    stats = statistiques_quantitatives(dico, tables_e1)
    cles = cles_jointure(tables_e1)
    anomalies_e1 = anomalies_etape1(tables_e1)
    etape1_resume = {"tables": {k: {"nb_lignes": len(v), "nb_colonnes": v.shape[1]}
                                for k, v in tables_e1.items()}}
    dico.to_csv(SORTIE / "etape1_dictionnaire.csv", index=False, sep=";")
    stats.to_csv(SORTIE / "etape1_statistiques.csv", index=False, sep=";")
    anomalies_e1.to_csv(SORTIE / "etape1_anomalies.csv", index=False, sep=";")

    # Étape 2
    manq = pd.concat([manquants(sin_brut, "sinistres"), manquants(ptf_brut, "portefeuille"),
                      manquants(expo_brut, "exposition")], ignore_index=True)
    res_dbl, ex_dbl = [], []
    for brut, nom in [(sin_brut, "sinistres"), (ptf_brut, "portefeuille"), (expo_brut, "exposition")]:
        r, e = doublons(brut, nom)
        res_dbl.append(r)
        ex_dbl.append(e)
    out = pd.concat([
        outliers(sin, "sinistres", ["nb_sin", "chg_sin"]),
        outliers(ptf, "portefeuille", ["Age", "Bonus", "Poldur", "Value_num", "Density"]),
        outliers(expo_brut, "exposition", ["Expdays"]),
    ], ignore_index=True)
    saisie = pd.concat([erreurs_saisie(ptf_brut, "portefeuille"), erreurs_saisie(sin_brut, "sinistres")],
                       ignore_index=True)
    regles = regles_metier(ptf, expo_brut, sin)
    inc = incoherences(ptf)
    formules = recenser_formules()
    fmt = formats(sin_brut, ptf_brut, expo_brut, formules)
    for nom, df in [("etape2_manquants", manq), ("etape2_outliers", out), ("etape2_saisie", saisie),
                    ("etape2_regles", regles), ("etape2_incoherences", inc), ("etape2_formats", fmt),
                    ("etape2_formules", formules)]:
        df.to_csv(SORTIE / f"{nom}.csv", index=False, sep=";")

    return {
        "etape1_resume": etape1_resume, "dico": dico, "stats": stats, "cles": cles,
        "anomalies_e1": anomalies_e1,
        "manquants": manq, "doublons": pd.DataFrame(res_dbl), "exemples": pd.concat(ex_dbl, ignore_index=True),
        "outliers": out, "saisie": saisie, "regles": regles, "incoherences": inc,
        "formats": fmt, "formules": formules,
    }


def generer_rapport(res: dict) -> None:
    corps = [
        "\\part{Étape 1 -- Compréhension des bases de données}\n\\section{Objectif}\n"
        "Comprendre les données disponibles : importer, dimensionner, typer, classer les variables et repérer "
        "les clés de jointure.\n",
        s_methode_e1(), s_sources(res["etape1_resume"]), s_dictionnaire(res["dico"]), s_nature(res["dico"]),
        s_statistiques(res["stats"]), s_anomalies_e1(res["anomalies_e1"]), s_cles(res["cles"]),
        "\n\\part{Étape 2 -- Audit et qualité des données}\n\\section{Objectif}\n"
        "Détecter les anomalies avant toute analyse.\n",
        s_methode_e2(), s_synthese(res), s_manquants(res), s_doublons(res), s_outliers(res), s_saisie(res),
        s_regles(res), s_incoherences(res), s_formats(res), s_plan(res),
        section_etape3(res["etape3"]),
        section_etape4(res["etape4"]),
        section_etape5(res["etape5"]),
        section_etape6(res["etape6"]),
        section_etape7(res["etape7"]),
        section_etape8(res),
    ]
    mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
            "septembre", "octobre", "novembre", "décembre"]
    auj = datetime.date.today()
    date_fr = f"{auj.day} {mois[auj.month - 1]} {auj.year}"
    pre = PREAMBULE.replace("__DATE_FR__", date_fr).replace("__LITERATE__", literate())
    FICHIER_TEX.write_text(pre + "\n".join(corps) + FIN, encoding="utf-8")
    if shutil.which("pdflatex") is None:
        print("pdflatex introuvable : le fichier rapport_analyse.tex est produit ; "
              "compilez-le avec votre éditeur LaTeX (ou installez une distribution LaTeX).")
        return
    for _ in range(2):
        # compilé depuis la racine (chemins des figures), fichiers produits dans rapport/
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-output-directory={FICHIER_TEX.parent.name}", FICHIER_TEX.relative_to(DOSSIER).as_posix()],
                       cwd=DOSSIER, check=True, stdout=subprocess.DEVNULL)
    for ext in (".aux", ".log", ".out", ".toc"):  # fichiers techniques de LaTeX, inutiles une fois le PDF produit
        FICHIER_TEX.with_suffix(ext).unlink(missing_ok=True)


def preparer_dossiers() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for dossier in (SORTIE, FIG, FICHIER_TEX.parent):
        dossier.mkdir(parents=True, exist_ok=True)


def main() -> None:
    preparer_dossiers()
    for fichier in (FICHIER_CSV, FICHIER_XLSX):
        if not fichier.exists():
            raise SystemExit(f"Fichier introuvable : {fichier}\n"
                             "Placez data_sin.csv et data_ptf_Eco.xlsx dans le dossier « donnees » du projet.")
    res = etape1_et_2()
    res["etape3"] = etape3()
    res["etape4"] = etape4(res["etape3"])
    res["etape5"] = etape5(res["etape3"])
    res["etape6"] = etape6(res["etape3"])
    res["etape7"] = etape7(res["etape3"])
    res["etape8"] = etape8(res)
    generer_rapport(res)
    print(f"Rapport : {FICHIER_TEX.with_suffix('.pdf')}")
    print(f"Tableaux : {SORTIE}")



if __name__ == "__main__":
    main()
