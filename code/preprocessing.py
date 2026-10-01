"""Prétraitement du dataset BRFSS (numpy uniquement)."""

import numpy as np

# --------------------------------------------------------------------------
# 1) Types de colonnes et codes spéciaux à transformer en NaN
# --------------------------------------------------------------------------

# Oui/Non : 1 = oui, 2 = non  ->  1 / 0
BINARY = [
    "HLTHPLN1", "MEDCOST", "BPMEDS", "CVDSTRK3", "CHCSCNCR", "CHCOCNCR",
    "CHCCOPD1", "ADDEPEV2", "CHCKIDNY", "SEX", "NUMHHOL2", "CPDEMO1",
    "VETERAN3", "INTERNET", "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE",
    "DIFFWALK", "DIFFDRES", "DIFFALON", "LMTJOIN3", "ARTHDIS2", "CAREGIV1",
    "CIMEMLOS", "_HCVU651", "_RFHYPE5", "_RFCHOL", "_DRDXAR1", "_RFBING5",
    "_RFDRHV5", "_TOTINDA", "_PASTRNG", "_RFSEAT3", "_FLSHOT6", "_PNEUMO2",
    "_AIDTST3",
]

# Catégories ordonnées : on garde la valeur numérique
ORDINAL = [
    "GENHLTH", "CHECKUP1", "USENOW3", "ARTHSOCL", "JOINPAIN", "_CHOLCHK",
    "_ASTHMS1", "_CHLDCNT", "_EDUCAG", "_INCOMG", "_SMOKER3", "_PACAT1",
    "_LMTACT1", "_LMTWRK1", "_LMTSCL1",
]

# Catégories sans ordre : one-hot encoding
NOMINAL = [
    "_STATE", "PERSDOC2", "MARITAL", "RENTHOM1", "EMPLOY1", "DIABETE3",
    "_RACE", "SXORIENT", "TRNSGNDR", "MSCODE", "WHRTST10", "IMFVPLAC",
]

# Variables continues
CONTINUOUS = [
    "PHYSHLTH", "MENTHLTH", "POORHLTH", "_AGE80", "_BMI5", "_DRNKWEK",
    "_FRUTSUM", "_VEGESUM", "MAXVO2_", "FC60_", "PA1MIN_",
]

# Très asymétriques -> log(1 + x)
LOG_COLS = ["_DRNKWEK", "_FRUTSUM", "_VEGESUM", "PA1MIN_"]

# Codes "ne sait pas / refus / manquant" (cf. codebook), par défaut 7 et 9
DEFAULT_MISSING = [7, 9]
SPECIAL_MISSING = {
    # Le code 7 est une vraie modalité pour ces colonnes
    "EMPLOY1": [9], "MARITAL": [9], "_RACE": [9], "_STATE": [],
    "MSCODE": [], "_AGE80": [], "_BMI5": [], "_FRUTSUM": [], "_VEGESUM": [],
    "PA1MIN_": [],
    "CAREGIV1": [7, 8, 9],
    "PHYSHLTH": [77, 99], "MENTHLTH": [77, 99], "POORHLTH": [77, 99],
    "JOINPAIN": [77, 99], "WHRTST10": [77, 99], "IMFVPLAC": [77, 99],
    "_DRNKWEK": [99900], "MAXVO2_": [999], "FC60_": [999],
    # Pour les variables calculées (_XXX), seul 9 veut dire manquant
    **{c: [9] for c in BINARY + ORDINAL if c.startswith("_")},
}
# Recodages : 88 = "aucun jour" -> 0 ; 8 = "jamais de check-up" -> 5
RECODE = {
    "PHYSHLTH": {88: 0}, "MENTHLTH": {88: 0}, "POORHLTH": {88: 0},
    "CHECKUP1": {8: 5},
}

KEEP = BINARY + ORDINAL + NOMINAL + CONTINUOUS


# --------------------------------------------------------------------------
# 2) Nettoyage colonne par colonne (aucune statistique apprise ici)
# --------------------------------------------------------------------------

def clean_columns(x, header):
    """Sélectionne les colonnes KEEP, remplace les codes spéciaux par NaN,
    applique les recodages. Retourne un array float (N, len(KEEP))."""
    idx = {c: i for i, c in enumerate(header)}
    out = np.empty((x.shape[0], len(KEEP)))
    for j, c in enumerate(KEEP):
        col = x[:, idx[c]].astype(float).copy()
        for code in SPECIAL_MISSING.get(c, DEFAULT_MISSING):
            col[col == code] = np.nan
        for old, new in RECODE.get(c, {}).items():
            col[col == old] = new
        if c in BINARY:
            col = np.where(np.isnan(col), np.nan, (col == 1).astype(float))
        if c in LOG_COLS:
            col = np.log1p(col)
        out[:, j] = col
    return out


# --------------------------------------------------------------------------
# 3) Fit sur le train, puis transform sur train ET test
# --------------------------------------------------------------------------

def fit_preprocessing(x_clean):
    """Apprend sur le TRAIN : médianes, catégories, moyennes, écarts-types."""
    params = {"median": {}, "categories": {}, "nan_cols": []}
    for j, c in enumerate(KEEP):
        col = x_clean[:, j]
        if np.isnan(col).any():
            params["nan_cols"].append(j)
        if c in NOMINAL:
            params["categories"][j] = np.unique(col[~np.isnan(col)])
        else:
            params["median"][j] = np.nanmedian(col)
    X = _build_matrix(x_clean, params)
    params["mean"] = X.mean(axis=0)
    params["std"] = X.std(axis=0)
    params["std"][params["std"] == 0] = 1.0
    return params


def _build_matrix(x_clean, params):
    feats = []
    for j, c in enumerate(KEEP):
        col = x_clean[:, j]
        if c in NOMINAL:
            # Une colonne 0/1 par catégorie ; NaN -> que des zéros
            for cat in params["categories"][j]:
                feats.append((col == cat).astype(float))
        else:
            feats.append(np.where(np.isnan(col), params["median"][j], col))
    # Indicateurs "la valeur était manquante"
    for j in params["nan_cols"]:
        feats.append(np.isnan(x_clean[:, j]).astype(float))
    return np.column_stack(feats)


def transform(x_clean, params):
    """Applique imputation + one-hot + standardisation, ajoute le biais."""
    X = _build_matrix(x_clean, params)
    X = (X - params["mean"]) / params["std"]
    return np.column_stack([np.ones(X.shape[0]), X])


def preprocess(x_train, x_test, header):
    """Pipeline complet. `header` = noms des colonnes de x_train
    (sans 'Id' si on utilise load_csv_data, qui retire la colonne Id)."""
    tr = clean_columns(x_train, header)
    te = clean_columns(x_test, header)
    params = fit_preprocessing(tr)
    return transform(tr, params), transform(te, params)
