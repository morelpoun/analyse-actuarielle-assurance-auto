# ==============================================================================
# Projet  : Exploitation de bases de données d'assurance automobile
#           Audit, analyse, modélisation et tarification
# Auteur  : Morel POUNIGUIA
# Fichier : cellules_extra.py
# ==============================================================================

"""Cellules supplémentaires du notebook : correspondance avec les consignes, visualisations avancées,
application (tarification et prévision), conclusion."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

# ------------------------------------------------------------------ correspondance avec les consignes
CONFORMITE = [
    md("### Correspondance avec le processus d'analyse demandé\n\n"
       "| Processus demandé | Où le trouver dans ce notebook |\n|---|---|\n"
       "| 1. Traitement des données : compréhension, complétude, conformité, cohérence, anomalies | Étapes 1–2 (audit) et 3 (nettoyage documenté dans le journal des décisions) |\n"
       "| 2. Jointures de bases et contrôles | Étape 4 |\n"
       "| 3. Analyses descriptives : univariée, multivariée, ASD (réduction de dimension, classification) | Étapes 3 bis, 5 et 6 |\n"
       "| 4. Analyse graphique (data visualisation) | Figures de chaque étape et étape 7 bis (interactions, calibrage, Lorenz, effets partiels) |\n"
       "| 5. Modélisation supervisée paramétrique et non paramétrique | Étape 7 (GLM contre forêt aléatoire et gradient boosting) |\n"
       "| 6. Analyse des résultats : interprétation | « Points clés » de chaque étape et étape 8 |\n"
       "| 7. Application : prévision, tarification | Étape 9 (tarif construit sur 2009, prévision testée sur 2010) |\n"
       "| Conclusion, limites, recommandations | Dernière section |"),
]

# ------------------------------------------------------------------ étape 7 bis : visualisations avancées
VIZ = [
    md("## Étape 7 bis — Visualisations avancées\n\n"
       "**Objectif.** Utiliser les graphiques pour **comprendre** les relations entre variables et vérifier la qualité "
       "du modèle, pas seulement pour décrire.\n\n"
       "| Graphique | Question à laquelle il répond |\n|---|---|\n"
       "| Carte de chaleur âge × bonus | Les deux effets s'additionnent-ils, ou interagissent-ils ? |\n"
       "| Calibrage par vingtile | Les fréquences prédites correspondent-elles aux fréquences observées, pour les bons comme pour les mauvais risques ? |\n"
       "| Courbe de Lorenz et indice de Gini | Le modèle classe-t-il bien les risques ? (Gini 0 = classement au hasard) |\n"
       "| Effets partiels | Quelle est la forme de l'effet de chaque variable (linéaire, seuil, plateau) ? |"),
    md("### Interaction âge × bonus-malus"),
    code(r'''b = base.copy()  # copie de la base des polices
b["bande_bonus"] = pd.cut(b["Bonus"], [-51, -30, 0, 50, 150],  # découpe du bonus-malus en 4 bandes
                          labels=["≤ −30 (bonus fort)", "−29 à 0", "+1 à +50", "> +50 (malus fort)"])
nb_s = b.pivot_table(index="tranche_age", columns="bande_bonus", values="nb_sinistres", aggfunc="sum", observed=False)  # sinistres
expo = b.pivot_table(index="tranche_age", columns="bande_bonus", values="expo_annees", aggfunc="sum", observed=False)  # exposition
freq = (100 * nb_s / expo).reindex(A.BANDES_AGE)  # fréquence pour 100 années d'exposition, âges dans l'ordre

fig, ax = plt.subplots(figsize=(8, 4.5))  # zone de dessin
im = ax.imshow(freq.values, cmap="YlOrRd", aspect="auto")  # carte de chaleur : plus c'est rouge, plus c'est risqué
ax.set_xticks(range(freq.shape[1]), freq.columns, rotation=15)  # étiquettes des bandes de bonus
ax.set_yticks(range(freq.shape[0]), freq.index)  # étiquettes des tranches d'âge
for i in range(freq.shape[0]):  # pour chaque tranche d'âge
    for j in range(freq.shape[1]):  # pour chaque bande de bonus
        v, e = freq.iat[i, j], expo.reindex(A.BANDES_AGE).iat[i, j]  # fréquence et exposition de la case
        if pd.notna(v):  # case non vide
            ax.text(j, i, f"{v:.0f}\n({e:,.0f} ans)".replace(",", " "), ha="center", va="center", fontsize=8)  # valeur
ax.set_xlabel("Bonus-malus")  # titre de l'axe horizontal
ax.set_ylabel("Tranche d'âge")  # titre de l'axe vertical
ax.set_title("Fréquence pour 100 années d'exposition (entre parenthèses : exposition)")  # titre
fig.colorbar(im, ax=ax, label="sinistres / 100 ans")  # échelle de couleurs
plt.tight_layout()  # ajuste les marges
montrer_fig()  # affiche la figure

f_jeune_malus = freq.loc["18-25"].iloc[-1]  # jeunes à fort malus
f_senior_bonus = freq.loc["51-65"].iloc[0]  # 51-65 ans à fort bonus
ratio_age_bonus = freq.iloc[:, 0] / freq.iloc[:, 0].loc["36-50"]  # effet de l'âge parmi les bons conducteurs
points_cles(  # constats chiffrés
    f"Les deux effets se cumulent : un 18-25 ans à fort malus a **{f_jeune_malus:.0f}** sinistres pour 100 ans, "
    f"contre **{f_senior_bonus:.0f}** pour un 51-65 ans à fort bonus (×{f_jeune_malus / f_senior_bonus:.0f}).",
    f"Même parmi les conducteurs à fort bonus, les 18-25 ans restent {ratio_age_bonus.loc['18-25']:.1f} fois plus risqués "
    "que les 36-50 ans : l'âge apporte une information que le bonus ne capte pas encore (peu d'historique).",
    "Les cases à faible exposition (nombre entre parenthèses) sont moins fiables et ne doivent pas être sur-interprétées.",
)'''),
    md("### Calibrage et pouvoir de classement du modèle de fréquence (échantillon test)"),
    code(r'''te = e7["te"]  # polices de l'échantillon test
taux_pred = e7["modele_frequence"].predict(e7["X_te"])  # fréquence annuelle prédite par le modèle retenu
expo_te = te["expo"].to_numpy()  # exposition de chaque police
nb_obs = te["nb"].to_numpy()  # sinistres observés
nb_att = taux_pred * expo_te  # sinistres attendus = fréquence × exposition

ordre = np.argsort(taux_pred)  # polices triées du risque le plus faible au plus élevé
groupe = np.empty(len(ordre), dtype=int)  # numéro de vingtile de chaque police
groupe[ordre] = np.arange(len(ordre)) * 20 // len(ordre)  # 20 groupes de même effectif
cal = pd.DataFrame({"g": groupe, "obs": nb_obs, "att": nb_att, "expo": expo_te}).groupby("g").sum()  # totaux par groupe
cal["freq_obs"] = 100 * cal["obs"] / cal["expo"]  # fréquence observée pour 100 ans
cal["freq_pred"] = 100 * cal["att"] / cal["expo"]  # fréquence prédite pour 100 ans

cum_expo = np.concatenate([[0], np.cumsum(expo_te[ordre]) / expo_te.sum()])  # part cumulée de l'exposition
cum_sin = np.concatenate([[0], np.cumsum(nb_obs[ordre]) / nb_obs.sum()])  # part cumulée des sinistres
gini = 1 - 2 * np.trapz(cum_sin, cum_expo)  # indice de Gini = 1 − 2 × aire sous la courbe de Lorenz

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))  # deux graphiques côte à côte
ax1.plot(cal.index + 1, cal["freq_pred"], "o-", label="prédite", color="#4C72B0")  # fréquence prédite
ax1.plot(cal.index + 1, cal["freq_obs"], "s--", label="observée", color="#DD8452")  # fréquence observée
ax1.set_xlabel("Vingtile de risque prédit (1 = plus faible)")  # axe horizontal
ax1.set_ylabel("Sinistres pour 100 ans")  # axe vertical
ax1.set_title("Calibrage : prédit contre observé")  # titre
ax1.legend()  # légende
ax2.plot(cum_expo, cum_sin, color="#4C72B0", label=f"modèle (Gini = {gini:.2f})")  # courbe de Lorenz
ax2.plot([0, 1], [0, 1], "--", color="grey", label="classement au hasard")  # diagonale de référence
ax2.set_xlabel("Part cumulée de l'exposition (risques faibles d'abord)")  # axe horizontal
ax2.set_ylabel("Part cumulée des sinistres")  # axe vertical
ax2.set_title("Courbe de Lorenz")  # titre
ax2.legend()  # légende
plt.tight_layout()  # ajuste les marges
montrer_fig()  # affiche la figure

ecart = (cal["freq_obs"] / cal["freq_pred"] - 1).abs()  # écart relatif observé / prédit par groupe
points_cles(  # constats chiffrés
    f"La fréquence observée passe de {cal['freq_obs'].iloc[0]:.1f} (vingtile 1) à {cal['freq_obs'].iloc[-1]:.1f} "
    f"(vingtile 20) sinistres pour 100 ans : le modèle sépare les risques d'un facteur ×{cal['freq_obs'].iloc[-1] / cal['freq_obs'].iloc[0]:.0f}.",
    f"Les courbes prédite et observée se suivent : écart médian de {100 * ecart.median():.0f} % par groupe, "
    f"au plus {100 * ecart.max():.0f} %. Le modèle est bien calibré sur toute l'échelle de risque.",
    f"Indice de Gini = **{gini:.2f}** : les 50 % de l'exposition jugés les moins risqués ne portent que "
    f"{100 * np.interp(0.5, cum_expo, cum_sin):.0f} % des sinistres.",
)'''),
    md("### Effets partiels des variables numériques\n"
       "Chaque courbe montre la fréquence moyenne prédite quand on fait varier **une seule** variable en gardant "
       "les autres à leurs valeurs réelles (*partial dependence*). Cela révèle la forme de l'effet, qu'un GLM "
       "linéaire imposerait sinon."),
    code(r'''from sklearn.inspection import partial_dependence  # calcul des effets partiels

echantillon = e7["X_te"].sample(3000, random_state=0)  # 3 000 polices pour un calcul rapide
VAR_PD = {"Age": "Âge", "Bonus": "Bonus-malus", "Density": "Densité de la zone", "Poldur": "Ancienneté du contrat"}  # libellés
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)  # 4 graphiques, même échelle verticale
effets, courbes = {}, {}  # amplitude et courbe de chaque effet
for ax, (v, lib) in zip(axes, VAR_PD.items()):  # une variable par graphique
    pdp = partial_dependence(e7["modele_frequence"], echantillon, [v], grid_resolution=30)  # effet partiel
    x, y = pdp["grid_values"][0], 100 * pdp["average"][0]  # grille de valeurs et fréquence pour 100 ans
    effets[lib] = y.max() / y.min()  # rapport entre l'effet maximal et minimal
    courbes[v] = pd.Series(y, index=x)  # courbe gardée pour les constats
    ax.plot(x, y, color="#4C72B0", lw=2)  # courbe de l'effet
    ax.set_title(lib)  # titre
    ax.set_xlabel(lib)  # axe horizontal
    ax.grid(alpha=0.3)  # grille légère
axes[0].set_ylabel("Fréquence prédite (pour 100 ans)")  # axe vertical commun
plt.tight_layout()  # ajuste les marges
montrer_fig()  # affiche la figure

effets = pd.Series(effets).sort_values(ascending=False)  # variables triées par amplitude
age_c, bon_c = courbes["Age"], courbes["Bonus"]  # courbes de l'âge et du bonus
points_cles(  # constats chiffrés
    "Amplitude de l'effet (fréquence max / min sur la plage observée) : "
    + ", ".join(f"{k} ×{v:.1f}" for k, v in effets.items()) + ".",
    f"Âge : fréquence prédite de {age_c.iloc[0]:.1f} à {age_c.index[0]:.0f} ans, {age_c.asof(40):.1f} à 40 ans et "
    f"{age_c.iloc[-1]:.1f} à {age_c.index[-1]:.0f} ans. L'effet n'est pas linéaire, d'où le découpage en tranches "
    "dans le tarif de l'étape 9.",
    f"Bonus-malus : de {bon_c.iloc[0]:.1f} (bonus {bon_c.index[0]:.0f}) à {bon_c.iloc[-1]:.1f} (malus +{bon_c.index[-1]:.0f}) "
    "sinistres pour 100 ans. Le bonus résume bien l'historique de sinistres du conducteur.",
)'''),
]

# ------------------------------------------------------------------ étape 9 : application
TARIF = [
    md("## Étape 9 — Application : tarification et prévision\n\n"
       "**Objectif.** Construire un **tarif utilisable**, c'est-à-dire une prime pour chaque assuré, et vérifier "
       "qu'il **prévoit** correctement la sinistralité d'une année qu'il n'a jamais vue.\n\n"
       "**Méthode.**\n"
       "1. **Validation temporelle** : le portefeuille contient deux années (2009 et 2010, 50 000 polices chacune). "
       "Le tarif est construit **uniquement sur 2009**, puis appliqué à **2010**. C'est la situation réelle d'un "
       "assureur, qui tarifie l'année suivante avec les données passées.\n"
       "2. **Modèle fréquence × coût** : GLM de Poisson pour la fréquence et GLM Gamma pour le coût moyen. "
       "Un GLM donne un tarif **multiplicatif et lisible** : prime = prime de base × coefficient de chaque facteur.\n"
       "3. **Variables tarifaires** : tranche d'âge, bande de bonus-malus, ancienneté, type de véhicule, profession, densité. "
       "Le **sexe est exclu** : depuis l'arrêt *Test-Achats* de la CJUE (applicable au 21/12/2012), il est interdit "
       "comme critère tarifaire dans l'Union européenne.\n"
       "4. **Équilibrage** : un coefficient garantit que la somme des primes pures égale la somme des coûts 2009.\n"
       "5. **Prime commerciale** = prime pure × (1 + chargement). Le chargement de 30 % couvrant frais, marge et "
       "sécurité est une **hypothèse de travail** (ordre de grandeur usuel en assurance automobile).\n\n"
       "**Comment lire.** Un coefficient de 1,50 veut dire « 50 % plus cher que l'assuré de référence ». "
       "Le ratio **S/P** (sinistres / primes) doit être proche de la même valeur dans tous les groupes : sinon, "
       "certains assurés paient trop et partiront chez un concurrent (**antisélection**)."),
    code(r'''from sklearn.linear_model import GammaRegressor, PoissonRegressor  # GLM Poisson et Gamma

CHARGEMENT = 0.30  # hypothèse de travail : chargement de 30 % pour frais, marge et sécurité
T = base[base["expo_annees"] > 0].copy()  # polices avec une exposition positive
T["bande_bonus"] = pd.cut(T["Bonus"], [-51, -30, 0, 50, 150],  # bonus-malus en 4 bandes
                          labels=["≤ −30", "−29 à 0", "+1 à +50", "> +50"]).astype(str)
T["bande_densite"] = pd.qcut(T["Density"], 4, labels=["D1 faible", "D2", "D3", "D4 forte"]).astype(str)  # densité en quartiles
FACTEURS = {  # variable : libellé
    "tranche_age": "Âge", "bande_bonus": "Bonus-malus", "tranche_anciennete": "Ancienneté",
    "Type": "Type de véhicule", "Occupation": "Profession", "bande_densite": "Densité de la zone",
}
REFS = {f: T.groupby(f)["expo_annees"].sum().idxmax() for f in FACTEURS}  # référence = modalité la plus exposée


def matrice(df):  # transforme les facteurs en variables 0/1, sans la modalité de référence
    X = pd.get_dummies(df[list(FACTEURS)].astype(str), prefix_sep="=")  # une colonne par modalité
    X = X.drop(columns=[f"{f}={r}" for f, r in REFS.items()], errors="ignore")  # retire les références
    return X.reindex(columns=COLONNES, fill_value=0).astype(float)  # mêmes colonnes que l'apprentissage


COLONNES = [c for c in pd.get_dummies(T[list(FACTEURS)].astype(str), prefix_sep="=").columns  # toutes les modalités
            if c not in {f"{f}={r}" for f, r in REFS.items()}]  # sauf les références
app, prev = T[T["CalYear"] == 2009], T[T["CalYear"] == 2010]  # 2009 pour construire, 2010 pour prévoir
X_app = matrice(app)  # facteurs de 2009

glm_f = PoissonRegressor(alpha=1e-6, max_iter=1000)  # GLM de fréquence (pénalité quasi nulle)
glm_f.fit(X_app, app["nb_sinistres"] / app["expo_annees"], sample_weight=app["expo_annees"])  # pondéré par l'exposition
avec_cout = app[(app["nb_sinistres"] > 0) & (app["cout_total"] > 0)]  # polices avec au moins un sinistre payé
glm_c = GammaRegressor(alpha=1e-6, max_iter=1000)  # GLM du coût moyen par sinistre
glm_c.fit(matrice(avec_cout), avec_cout["cout_total"] / avec_cout["nb_sinistres"],  # coût moyen par sinistre
          sample_weight=avec_cout["nb_sinistres"])  # pondéré par le nombre de sinistres


def prime_brute(df):  # prime pure annuelle avant équilibrage = fréquence × coût moyen
    X = matrice(df)  # facteurs de la police
    return glm_f.predict(X) * glm_c.predict(X)  # produit des deux modèles


K = app["cout_total"].sum() / (prime_brute(app) * app["expo_annees"]).sum()  # coefficient d'équilibrage sur 2009
PRIME_BASE = K * np.exp(glm_f.intercept_ + glm_c.intercept_)  # prime pure annuelle de l'assuré de référence

grille = pd.DataFrame({  # coefficients tarifaires par modalité
    "colonne": COLONNES,  # nom technique
    "coef_frequence": np.exp(glm_f.coef_),  # effet sur la fréquence
    "coef_cout": np.exp(glm_c.coef_),  # effet sur le coût moyen
})
grille["coefficient"] = grille["coef_frequence"] * grille["coef_cout"]  # effet total sur la prime
grille[["facteur", "modalité"]] = grille["colonne"].str.split("=", n=1, expand=True)  # facteur et modalité
refs = pd.DataFrame([{"facteur": f, "modalité": f"{r} (référence)", "coef_frequence": 1.0,  # lignes des références
                      "coef_cout": 1.0, "coefficient": 1.0} for f, r in REFS.items()])
grille = pd.concat([refs, grille.drop(columns="colonne")])  # grille complète
grille["facteur"] = grille["facteur"].map(FACTEURS)  # libellé lisible du facteur
grille = grille.sort_values(["facteur", "coefficient"]).set_index(["facteur", "modalité"])  # tri
grille.to_csv(A.SORTIE / "etape9_grille_tarifaire.csv", sep=";", decimal=",")  # sauvegarde

display(Markdown(f"**Prime pure de base** (assuré de référence : {', '.join(f'{FACTEURS[f]} {r}' for f, r in REFS.items())}) : "
                 f"**{nb_fr(PRIME_BASE, 1)} € / an**, soit {nb_fr(PRIME_BASE * (1 + CHARGEMENT), 1)} € de prime commerciale."))
tableau(grille, "Grille tarifaire (coefficients multiplicatifs)", 40)  # affichage de la grille

amplitude = grille.groupby(level=0)["coefficient"].agg(lambda c: c.max() / c.min()).sort_values(ascending=False)  # écart par facteur
points_cles(  # constats chiffrés
    "Écart de prime entre la modalité la plus chère et la moins chère, par facteur : "
    + ", ".join(f"{k} ×{v:.2f}" for k, v in amplitude.items()) + ".",
    f"Le facteur le plus discriminant est **{amplitude.index[0]}**, suivi de **{amplitude.index[1]}**.",
    f"Coefficient d'équilibrage K = {K:.4f} : la somme des primes pures 2009 égale le coût observé 2009 "
    + ("(correction négligible : les GLM sont déjà équilibrés)." if abs(K - 1) < 0.01 else f"(correction de {100 * (K - 1):+.1f} %)."),
)'''),
    md("### Prévision 2010 et test de l'antisélection\n"
       "On applique le tarif 2009 aux polices de 2010, puis on compare avec les sinistres réellement observés en 2010. "
       "On le compare aussi à un **tarif uniforme**, où tout le monde paie la même prime moyenne."),
    code(r'''prev = prev.copy()  # polices 2010
prev["prime_pure"] = K * prime_brute(prev) * prev["expo_annees"]  # prime pure segmentée (exposition réelle)
prime_uniforme = app["cout_total"].sum() / app["expo_annees"].sum()  # prime unique : coût moyen 2009 par année
prev["prime_uniforme"] = prime_uniforme * prev["expo_annees"]  # tarif uniforme

prevu, observe = prev["prime_pure"].sum(), prev["cout_total"].sum()  # coût prévu et observé en 2010
annee = T.groupby("CalYear").agg(sinistres=("nb_sinistres", "sum"), expo=("expo_annees", "sum"),  # sinistralité par année
                                 cout=("cout_total", "sum"))
annee["fréquence (pour 100 ans)"] = 100 * annee["sinistres"] / annee["expo"]  # fréquence annuelle
annee["coût moyen (€)"] = annee["cout"] / annee["sinistres"]  # coût moyen par sinistre
annee["prime pure (€/an)"] = annee["cout"] / annee["expo"]  # coût par année d'exposition
tableau(annee, "Sinistralité observée par année")  # comparaison 2009 / 2010
TENDANCE = annee.loc[2010, "prime pure (€/an)"] / annee.loc[2009, "prime pure (€/an)"]  # évolution de la prime pure
nb_prevu = (glm_f.predict(matrice(prev)) * prev["expo_annees"]).sum()  # nombre de sinistres prévu
display(pd.DataFrame({  # bilan de la prévision
    "prévu (tarif 2009)": [nb_prevu, prevu],  # ce que le tarif annonçait
    "observé 2010": [prev["nb_sinistres"].sum(), observe],  # ce qui s'est passé
    "écart %": [100 * (nb_prevu / prev["nb_sinistres"].sum() - 1), 100 * (prevu / observe - 1)],  # erreur relative
}, index=["Nombre de sinistres", "Coût total (€)"]))  # deux lignes

prev["decile"] = pd.qcut(prev["prime_pure"] / prev["expo_annees"], 10, labels=range(1, 11))  # déciles de prime annuelle
dec9 = prev.groupby("decile", observed=True).agg(  # bilan par décile
    polices=("PolNum", "size"),  # nombre de polices
    prime_annuelle_moyenne=("prime_pure", lambda s: (s / prev.loc[s.index, "expo_annees"]).mean()),  # prime moyenne
    cout=("cout_total", "sum"),  # coût réel
    prime_segmentee=("prime_pure", "sum"),  # primes encaissées avec le tarif segmenté
    prime_uniforme=("prime_uniforme", "sum"),  # primes encaissées avec le tarif uniforme
)
dec9["S/P segmenté"] = dec9["cout"] / dec9["prime_segmentee"]  # ratio sinistres / primes, tarif segmenté
dec9["S/P uniforme"] = dec9["cout"] / dec9["prime_uniforme"]  # ratio sinistres / primes, tarif uniforme
tableau(dec9, "Ratio sinistres / primes 2010 par décile de risque", 10)  # affichage

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))  # deux graphiques
x = np.arange(1, 11)  # numéros de décile
b_u = ax1.bar(x - 0.2, dec9["S/P uniforme"], 0.4, label="tarif uniforme", color="#C44E52")  # S/P avec tarif uniforme
b_s = ax1.bar(x + 0.2, dec9["S/P segmenté"], 0.4, label="tarif segmenté (GLM)", color="#4C72B0")  # S/P avec tarif segmenté
for b_, col in [(b_u, "S/P uniforme"), (b_s, "S/P segmenté")]:  # étiquette en % sur chaque bâton
    ax1.bar_label(b_, labels=[f"{100 * v:.0f} %" for v in dec9[col]], padding=2, fontsize=6, rotation=90)  # S/P en %
ax1.margins(y=0.15)  # place pour les étiquettes
ax1.axhline(1, color="black", lw=1, ls="--")  # S/P = 1 : primes = sinistres
ax1.set_xticks(x)  # un repère par décile
ax1.set_xlabel("Décile de risque (1 = plus faible)")  # axe horizontal
ax1.set_ylabel("Sinistres / primes 2010")  # axe vertical
ax1.set_title("Antisélection : S/P par décile")  # titre
ax1.legend()  # légende
pc = prev["prime_pure"] / prev["expo_annees"] * (1 + CHARGEMENT)  # prime commerciale annuelle
ax2.hist(pc.clip(upper=pc.quantile(0.99)), bins=50, color="#4C72B0", alpha=0.85)  # distribution (1 % extrême regroupé)
ax2.axvline(pc.median(), color="green", ls="--", lw=2, label=f"médiane {pc.median():.0f} €")  # médiane
ax2.axvline(pc.mean(), color="red", lw=2, label=f"moyenne {pc.mean():.0f} €")  # moyenne
ax2.set_xlabel("Prime commerciale annuelle (€)")  # axe horizontal
ax2.set_title("Distribution des primes commerciales 2010")  # titre
ax2.legend()  # légende
plt.tight_layout()  # ajuste les marges
montrer_fig()  # affiche la figure

points_cles(  # constats chiffrés
    f"Prévision 2010 : coût prévu {nb_fr(prevu)} € contre {nb_fr(observe)} € observés "
    f"(écart **{100 * (prevu / observe - 1):+.1f} %**). Le tarif 2009 sous-estime 2010, car la sinistralité a augmenté : "
    f"fréquence {annee.loc[2009, 'fréquence (pour 100 ans)']:.1f} → {annee.loc[2010, 'fréquence (pour 100 ans)']:.1f} pour 100 ans, "
    f"coût moyen {nb_fr(annee.loc[2009, 'coût moyen (€)'])} → {nb_fr(annee.loc[2010, 'coût moyen (€)'])} €.",
    f"En pratique, il faut appliquer au tarif un **coefficient de tendance** (ici ×{TENDANCE:.3f} entre 2009 et 2010). "
    "Avec seulement deux années, ce coefficient est une estimation fragile : il faudra le confirmer sur plus d'historique.",
    f"Avec un **tarif uniforme**, le S/P va de {dec9['S/P uniforme'].iloc[0]:.2f} (décile 1) à "
    f"{dec9['S/P uniforme'].iloc[-1]:.2f} (décile 10) : les bons risques paient trop, partent chez un concurrent, "
    "et l'assureur garde les mauvais risques (antisélection).",
    f"Avec le **tarif segmenté**, le S/P reste entre {dec9['S/P segmenté'].min():.2f} et {dec9['S/P segmenté'].max():.2f} : "
    "il est trop élevé partout, à cause de la hausse de 2010, mais **au même niveau dans tous les déciles**. "
    "La segmentation reste donc valable d'une année sur l'autre : seul le niveau général doit être ajusté.",
    f"Prime commerciale annuelle 2010 : médiane {pc.median():.0f} €, 90 % des assurés entre "
    f"{pc.quantile(0.05):.0f} € et {pc.quantile(0.95):.0f} €.",
)'''),
    md("### Exemples de tarifs pour quatre profils types"),
    code(r'''profils = pd.DataFrame([  # profils d'assurés fictifs (choix retenu), pour illustrer le tarif
    {"profil": "Jeune conducteur, malus, ville", "tranche_age": "18-25", "bande_bonus": "> +50",
     "tranche_anciennete": "0-1", "Type": "A", "Occupation": "Employed", "bande_densite": "D4 forte"},
    {"profil": "Jeune conducteur, sans malus, campagne", "tranche_age": "18-25", "bande_bonus": "−29 à 0",
     "tranche_anciennete": "0-1", "Type": "A", "Occupation": "Employed", "bande_densite": "D1 faible"},
    {"profil": "Conducteur expérimenté, bonus fort", "tranche_age": "36-50", "bande_bonus": "≤ −30",
     "tranche_anciennete": "10+", "Type": "A", "Occupation": "Employed", "bande_densite": "D2"},
    {"profil": "Retraité, bonus fort, campagne", "tranche_age": "66+", "bande_bonus": "≤ −30",
     "tranche_anciennete": "10+", "Type": "A", "Occupation": "Retired", "bande_densite": "D1 faible"},
]).set_index("profil")  # nom du profil en index
profils["fréquence (pour 100 ans)"] = 100 * glm_f.predict(matrice(profils))  # fréquence attendue
profils["coût moyen d'un sinistre (€)"] = K * glm_c.predict(matrice(profils))  # coût moyen attendu (équilibré)
profils["prime pure (€/an)"] = K * prime_brute(profils)  # prime pure annuelle
profils["prime commerciale (€/an)"] = profils["prime pure (€/an)"] * (1 + CHARGEMENT)  # prime commerciale
display(profils)  # affichage

pmax, pmin = profils["prime commerciale (€/an)"].max(), profils["prime commerciale (€/an)"].min()  # extrêmes
points_cles(  # constats chiffrés
    f"Entre le profil le plus cher et le moins cher, la prime commerciale varie de {pmin:.0f} € à {pmax:.0f} € "
    f"(×{pmax / pmin:.1f}).",
    "Le facteur ×K est appliqué au coût moyen pour que les profils restent cohérents avec le tarif équilibré.",
)'''),
]

# ------------------------------------------------------------------ conclusion
CONCLUSION = [
    md("## Conclusion, limites et recommandations\n\n"
       "### Ce qu'il faut retenir\n"
       "- **Qualité des données** : les bases contenaient de nombreuses anomalies (âges impossibles, doublons, codes "
       "non normalisés, nombres stockés en texte). Toutes ont été traitées et documentées dans le journal des décisions.\n"
       "- **Facteurs de risque** : la fréquence dépend surtout du **bonus-malus**, de l'**âge** et de la **densité** de la zone. "
       "Les jeunes conducteurs à fort malus sont de loin les plus risqués.\n"
       "- **Le coût d'un sinistre est peu prévisible** avec les variables disponibles : l'essentiel de la segmentation "
       "vient de la fréquence.\n"
       "- **Concentration** : une petite part des sinistres représente une grande part du coût, ce qui plaide pour un "
       "écrêtement des gros sinistres dans le tarif et une réassurance en excédent de sinistre.\n"
       "- **Application** : le tarif GLM construit sur 2009 garde une segmentation valable en 2010 (S/P homogène entre "
       "déciles, pas d'antisélection, contrairement à un tarif uniforme). En revanche, il **sous-estime le niveau** de 2010, "
       "car la sinistralité a augmenté : un coefficient de tendance est nécessaire.\n\n"
       "### Limites\n"
       "- Une partie des lignes de sinistres n'a **pas de police correspondante** (étape 4). Elles sont exclues, ce qui "
       "peut sous-estimer la sinistralité si elles appartiennent à des polices réelles absentes du portefeuille.\n"
       "- L'incohérence « Housewife » chez des hommes est très fréquente : la variable *Occupation* est probablement mal codée. "
       "Choix retenu : les lignes sont conservées, l'incohérence est signalée et *Occupation* reste dans le tarif ; son effet est à surveiller lors du recalibrage.\n"
       "- La segmentation non supervisée est **modérément stable** (ARI) : les profils servent à décrire le portefeuille, "
       "pas à tarifer.\n"
       "- Deux années seulement : on ne peut pas mesurer de tendance (inflation des coûts, évolution de la fréquence).\n"
       "- Le chargement de 30 % est une hypothèse de travail ; il sera à remplacer par les frais réels de l'insurtech.\n\n"
       "### Recommandations\n"
       "1. Retrouver l'origine des sinistres sans police avant toute mise en production.\n"
       "2. Utiliser le **gradient boosting** pour la sélection des risques et le pilotage, et la **grille GLM** pour le "
       "tarif commercial, qui doit rester explicable aux clients et au régulateur.\n"
       "3. Écrêter les sinistres au-delà d'un seuil (par exemple le P99) dans le tarif, et mutualiser la surcharge.\n"
       "4. Appliquer un coefficient de tendance au tarif, puis le recalibrer chaque année avec les nouvelles données "
       "et suivre le S/P par segment."),
]
