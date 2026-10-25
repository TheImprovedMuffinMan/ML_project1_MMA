"""
Preprocessing of the dataset
"""

import numpy as np
import categories as cat
from pathlib import Path
import importlib

importlib.reload(cat)


# ============================================================================ #
# Load feature names
def load_names(x_train_path):
    """
    Load feature names from x_train.csv.

    The Id column is excluded because load_csv_data() already removes it
    from x_train and x_test.
    """
    header = np.genfromtxt(
        x_train_path,
        delimiter=",",
        max_rows=1,
        dtype=str,
    )

    return header[1:]


# ============================================================================ #
# drop unwanted columns


def drop_columns(x_train, x_test, feature_names, cols_to_drop):
    """
    Remove the same columns from train and test.

    Returns:
        x_train_cleaned
        x_test_cleaned
        kept_feature_names
        removed_feature_names
    """

    keep_columns = ~np.isin(feature_names, cols_to_drop)

    x_train_cleaned = x_train[:, keep_columns].copy()
    x_test_cleaned = x_test[:, keep_columns].copy()

    kept_feature_names = feature_names[keep_columns]
    removed_feature_names = feature_names[~keep_columns]

    return (
        x_train_cleaned,
        x_test_cleaned,
        kept_feature_names,
        removed_feature_names,
    )


# ============================================================================ #
# Replace NaN with median


def replace_median(x_train, x_test, feature_names, features_to_impute):
    """
    Replace NaNs in selected continuous features with the training median.

    The median is always calculated from the training set and then applied
    to both training and test data.
    """

    x_train = x_train.copy()
    x_test = x_test.copy()

    medians = {}

    for name in features_to_impute:

        if name not in feature_names:
            raise ValueError(f"{name} not found in feature_names")

        j = np.where(feature_names == name)[0][0]

        median = np.nanmedian(x_train[:, j])

        if np.isnan(median):
            raise ValueError(f"{name}: cannot compute median")

        x_train[np.isnan(x_train[:, j]), j] = median
        x_test[np.isnan(x_test[:, j]), j] = median

        medians[name] = median

    return x_train, x_test, medians


# ============================================================================ #
# Binary variables


def encode_binary_columns(x_train, x_test, feature_names, specs):
    train_cols = []
    test_cols = []
    new_names = []

    for j, name in enumerate(feature_names):
        tr = x_train[:, j]
        te = x_test[:, j]

        if name not in specs:
            train_cols.append(tr)
            test_cols.append(te)
            new_names.append(name)
            continue

        spec = specs[name]

        positive = spec["positive"]
        negative = spec["negative"]
        unknown = spec["unknown"]

        # Check training values
        allowed = positive + negative + unknown
        values = np.unique(tr[~np.isnan(tr)])

        unexpected = values[~np.isin(values, allowed)]

        if len(unexpected) > 0:
            raise ValueError(f"{name}: unexpected values {unexpected}")

        # Positive / negative
        tr_binary = np.isin(tr, positive).astype(float)
        te_binary = np.isin(te, positive).astype(float)

        # Unknown / refused / missing
        tr_unknown = (np.isin(tr, unknown) | np.isnan(tr)).astype(float)

        te_unknown = (np.isin(te, unknown) | np.isnan(te)).astype(float)

        train_cols.extend([tr_binary, tr_unknown])
        test_cols.extend([te_binary, te_unknown])

        new_names.extend(
            [
                f"{name}_positive",
                f"{name}_unknown",
            ]
        )

    return (
        np.column_stack(train_cols),
        np.column_stack(test_cols),
        np.array(new_names),
    )


# ============================================================================ #
# Encode for the cases where there is only yes or no (no unknowns)
def encode_direct_binary_columns(x_train, x_test, feature_names, specs):
    x_train = x_train.copy()
    x_test = x_test.copy()

    for name, spec in specs.items():

        if name not in feature_names:
            continue

        j = np.where(feature_names == name)[0][0]

        x_train[:, j] = np.isin(x_train[:, j], spec["positive"]).astype(float)

        x_test[:, j] = np.isin(x_test[:, j], spec["positive"]).astype(float)

    return x_train, x_test


# ============================================================================ #
# Numerical / ordinal variables with special codes


def encode_numeric_columns(x_train, x_test, feature_names, specs):
    """
    Used for:
      - ordinal variables
      - continuous variables with special BRFSS codes

    Each spec may contain:
      "replace": {old: new}
      "unknown": [...]
      "fill_value": 0
    """

    train_cols = []
    test_cols = []
    new_names = []

    for j, name in enumerate(feature_names):
        tr = x_train[:, j].copy()
        te = x_test[:, j].copy()

        if name not in specs:
            train_cols.append(tr)
            test_cols.append(te)
            new_names.append(name)
            continue

        spec = specs[name]

        replace = spec.get("replace", {})
        unknown = spec.get("unknown", [])
        fill_value = spec.get("fill_value", 0)

        # Meaningful recodings, e.g. 88 -> 0
        for old, new in replace.items():
            tr[tr == old] = new
            te[te == old] = new

        # Unknown indicator
        tr_unknown_mask = np.isin(tr, unknown) | np.isnan(tr)

        te_unknown_mask = np.isin(te, unknown) | np.isnan(te)

        tr_unknown = tr_unknown_mask.astype(float)
        te_unknown = te_unknown_mask.astype(float)

        # Replace unknown code by neutral placeholder
        tr[tr_unknown_mask] = fill_value
        te[te_unknown_mask] = fill_value

        train_cols.extend([tr, tr_unknown])
        test_cols.extend([te, te_unknown])

        new_names.extend(
            [
                name,
                f"{name}_unknown",
            ]
        )

    return (
        np.column_stack(train_cols),
        np.column_stack(test_cols),
        np.array(new_names),
    )


# ============================================================================ #
# Nominal categorical variables


def one_hot_encode_columns(x_train, x_test, feature_names, specs):
    """
    One-hot encode nominal variables.

    - unknown/refused/NaN -> explicit *_unknown column
    - first category is used as reference
    """

    train_cols = []
    test_cols = []
    new_names = []

    for j, name in enumerate(feature_names):
        tr = x_train[:, j]
        te = x_test[:, j]

        if name not in specs:
            train_cols.append(tr)
            test_cols.append(te)
            new_names.append(name)
            continue

        unknown_values = specs[name]["unknown"]

        tr_unknown_mask = np.isin(tr, unknown_values) | np.isnan(tr)

        te_unknown_mask = np.isin(te, unknown_values) | np.isnan(te)

        # Genuine categories only
        tr_valid = tr[~tr_unknown_mask]
        categories = np.unique(tr_valid)

        if len(categories) == 0:
            raise ValueError(f"{name}: no valid training categories")

        # First genuine category = reference
        reference = categories[0]

        for category in categories:
            if category == reference:
                continue

            train_cols.append((tr == category).astype(float))

            test_cols.append((te == category).astype(float))

            new_names.append(f"{name}_{category:g}")

        # Explicit unknown column
        train_cols.append(tr_unknown_mask.astype(float))
        test_cols.append(te_unknown_mask.astype(float))

        new_names.append(f"{name}_unknown")

    return (
        np.column_stack(train_cols),
        np.column_stack(test_cols),
        np.array(new_names),
    )


def scale_features(x_train, x_test, feature_names, features_to_scale):
    """
    Standardise selected features using training statistics.

    If a feature has a corresponding <feature>_unknown column:
        - calculate mean/std using known training observations only
        - scale known observations
        - keep unknown observations at 0

    Otherwise, scale the entire column normally.
    """

    x_train_scaled = x_train.copy().astype(float)
    x_test_scaled = x_test.copy().astype(float)

    means = {}
    stds = {}

    for name in features_to_scale:

        if name not in feature_names:
            continue

        j = np.where(feature_names == name)[0][0]

        unknown_name = f"{name}_unknown"

        # ------------------------------------------------------------
        # Feature has an unknown indicator
        if unknown_name in feature_names:

            j_unknown = np.where(feature_names == unknown_name)[0][0]

            train_known = x_train[:, j_unknown] == 0
            test_known = x_test[:, j_unknown] == 0

            # Learn statistics from known training values only
            mean = np.mean(x_train[train_known, j])
            std = np.std(x_train[train_known, j])

            if std == 0:
                std = 1.0

            # Unknown values remain 0
            x_train_scaled[:, j] = 0.0
            x_test_scaled[:, j] = 0.0

            # Scale only known observations
            x_train_scaled[train_known, j] = (x_train[train_known, j] - mean) / std

            x_test_scaled[test_known, j] = (x_test[test_known, j] - mean) / std

        # ------------------------------------------------------------
        # Normal feature without unknown indicator
        # ------------------------------------------------------------
        else:

            mean = np.mean(x_train[:, j])
            std = np.std(x_train[:, j])

            if std == 0:
                std = 1.0

            x_train_scaled[:, j] = (x_train[:, j] - mean) / std

            x_test_scaled[:, j] = (x_test[:, j] - mean) / std

        means[name] = mean
        stds[name] = std

    return x_train_scaled, x_test_scaled, means, stds


# ============================================================================ #
# Bias term


def add_bias(x_train, x_test, feature_names):
    """
    Prepend a constant column of ones to train and test so that linear and
    logistic models can learn an intercept.
    """
    x_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    x_test = np.column_stack([np.ones(x_test.shape[0]), x_test])
    feature_names = np.concatenate([["bias"], feature_names])

    return x_train, x_test, feature_names


# ============================================================================ #
# Linearly dependent columns
 
 
def drop_dependent_columns(x_train, x_test, feature_names, tol=1e-9):
    """
    Remove every column that is an exact linear combination of the columns before it
    (identical *_unknown indicators, complementary indicator sets, ... ).
    Returns:
        x_train, x_test, kept_feature_names, removed_feature_names
    """
    G = x_train.T @ x_train / x_train.shape[0]
    kept = []
 
    for j in range(G.shape[0]):
        if kept:
            coef = np.linalg.solve(G[np.ix_(kept, kept)], G[kept, j])
            residual = G[j, j] - G[kept, j] @ coef
        else:
            residual = G[j, j]
 
        if G[j, j] > 0 and residual > tol * G[j, j]:
            kept.append(j)
 
    kept = np.array(kept)
    removed = np.setdiff1d(np.arange(G.shape[0]), kept)
 
    return (
        x_train[:, kept],
        x_test[:, kept],
        feature_names[kept],
        feature_names[removed],
    )



# ============================================================================ #
# Complete preprocessing
def clean_data(x_train, x_test):

    # Feature names
    x_train_path = Path(__file__).resolve().parent.parent / "dataset" / "x_train.csv"

    feature_names = load_names(x_train_path)

    # Drop unwanted columns
    cols_to_drop = cat.COLS_ADMIN + cat.COLS_REDUNDANT + cat.COLS_MOSTLY_NAN

    x_train, x_test, feature_names, removed_feature_names = drop_columns(
        x_train, x_test, feature_names, cols_to_drop
    )

    # Median imputation
    x_train, x_test, _ = replace_median(
        x_train, x_test, feature_names, cat.MEDIAN_IMPUTE_FEATURES
    )

    # Binary variables
    x_train, x_test, feature_names = encode_binary_columns(
        x_train, x_test, feature_names, cat.BINARY_SPECS
    )

    # Already binary variables
    x_train, x_test = encode_direct_binary_columns(
        x_train, x_test, feature_names, cat.DIRECT_BINARY_SPECS
    )

    # Numerical / ordinal variables
    x_train, x_test, feature_names = encode_numeric_columns(
        x_train, x_test, feature_names, cat.NUMERIC_SPECS
    )

    # Nominal variables
    x_train, x_test, feature_names = one_hot_encode_columns(
        x_train,
        x_test,
        feature_names,
        cat.NOMINAL_SPECS,
    )

    # Scale selected variables
    x_train, x_test, _, _ = scale_features(
        x_train, x_test, feature_names, cat.FEATURES_TO_SCALE
    )

    # Bias term: constant column of ones (added after scaling, never scaled)
    x_train, x_test, feature_names = add_bias(x_train, x_test, feature_names)


    # Remove linearly dependent columns (done after the bias is added, so that
    # dependencies involving the constant column are detected; the bias is column 0
    # and is therefore always kept)
    x_train, x_test, feature_names, removed = drop_dependent_columns(
        x_train, x_test, feature_names
    )
    if len(removed) > 0:
        print(f"Removed {len(removed)} linearly dependent columns: {list(removed)}")

    return x_train, x_test, feature_names


# """Prétraitement du dataset BRFSS (numpy uniquement)."""

# import numpy as np

# # --------------------------------------------------------------------------
# # 1) Types de colonnes et codes spéciaux à transformer en NaN
# # --------------------------------------------------------------------------

# # Oui/Non : 1 = oui, 2 = non  ->  1 / 0
# BINARY = [
#     "HLTHPLN1",
#     "MEDCOST",
#     "BPMEDS",
#     "CVDSTRK3",
#     "CHCSCNCR",
#     "CHCOCNCR",
#     "CHCCOPD1",
#     "ADDEPEV2",
#     "CHCKIDNY",
#     "SEX",
#     "NUMHHOL2",
#     "CPDEMO1",
#     "VETERAN3",
#     "INTERNET",
#     "QLACTLM2",
#     "USEEQUIP",
#     "BLIND",
#     "DECIDE",
#     "DIFFWALK",
#     "DIFFDRES",
#     "DIFFALON",
#     "LMTJOIN3",
#     "ARTHDIS2",
#     "CAREGIV1",
#     "CIMEMLOS",
#     "_HCVU651",
#     "_RFHYPE5",
#     "_RFCHOL",
#     "_DRDXAR1",
#     "_RFBING5",
#     "_RFDRHV5",
#     "_TOTINDA",
#     "_PASTRNG",
#     "_RFSEAT3",
#     "_FLSHOT6",
#     "_PNEUMO2",
#     "_AIDTST3",
# ]

# # Catégories ordonnées : on garde la valeur numérique
# ORDINAL = [
#     "GENHLTH",
#     "CHECKUP1",
#     "USENOW3",
#     "ARTHSOCL",
#     "JOINPAIN",
#     "_CHOLCHK",
#     "_ASTHMS1",
#     "_CHLDCNT",
#     "_EDUCAG",
#     "_INCOMG",
#     "_SMOKER3",
#     "_PACAT1",
#     "_LMTACT1",
#     "_LMTWRK1",
#     "_LMTSCL1",
# ]

# # Catégories sans ordre : one-hot encoding
# NOMINAL = [
#     "_STATE",
#     "PERSDOC2",
#     "MARITAL",
#     "RENTHOM1",
#     "EMPLOY1",
#     "DIABETE3",
#     "_RACE",
#     "SXORIENT",
#     "TRNSGNDR",
#     "MSCODE",
#     "WHRTST10",
#     "IMFVPLAC",
# ]

# # Variables continues
# CONTINUOUS = [
#     "PHYSHLTH",
#     "MENTHLTH",
#     "POORHLTH",
#     "_AGE80",
#     "_BMI5",
#     "_DRNKWEK",
#     "_FRUTSUM",
#     "_VEGESUM",
#     "MAXVO2_",
#     "FC60_",
#     "PA1MIN_",
# ]

# # Très asymétriques -> log(1 + x)
# LOG_COLS = ["_DRNKWEK", "_FRUTSUM", "_VEGESUM", "PA1MIN_"]

# # Codes "ne sait pas / refus / manquant" (cf. codebook), par défaut 7 et 9
# DEFAULT_MISSING = [7, 9]
# SPECIAL_MISSING = {
#     # Le code 7 est une vraie modalité pour ces colonnes
#     "EMPLOY1": [9],
#     "MARITAL": [9],
#     "_RACE": [9],
#     "_STATE": [],
#     "MSCODE": [],
#     "_AGE80": [],
#     "_BMI5": [],
#     "_FRUTSUM": [],
#     "_VEGESUM": [],
#     "PA1MIN_": [],
#     "CAREGIV1": [7, 8, 9],
#     "PHYSHLTH": [77, 99],
#     "MENTHLTH": [77, 99],
#     "POORHLTH": [77, 99],
#     "JOINPAIN": [77, 99],
#     "WHRTST10": [77, 99],
#     "IMFVPLAC": [77, 99],
#     "_DRNKWEK": [99900],
#     "MAXVO2_": [999],
#     "FC60_": [999],
#     # Pour les variables calculées (_XXX), seul 9 veut dire manquant
#     **{c: [9] for c in BINARY + ORDINAL if c.startswith("_")},
# }
# # Recodages : 88 = "aucun jour" -> 0 ; 8 = "jamais de check-up" -> 5
# RECODE = {
#     "PHYSHLTH": {88: 0},
#     "MENTHLTH": {88: 0},
#     "POORHLTH": {88: 0},
#     "CHECKUP1": {8: 5},
# }

# KEEP = BINARY + ORDINAL + NOMINAL + CONTINUOUS


# # --------------------------------------------------------------------------
# # 2) Nettoyage colonne par colonne (aucune statistique apprise ici)
# # --------------------------------------------------------------------------


# def clean_columns(x, header):
#     """Sélectionne les colonnes KEEP, remplace les codes spéciaux par NaN,
#     applique les recodages. Retourne un array float (N, len(KEEP))."""
#     idx = {c: i for i, c in enumerate(header)}
#     out = np.empty((x.shape[0], len(KEEP)))
#     for j, c in enumerate(KEEP):
#         col = x[:, idx[c]].astype(float).copy()
#         for code in SPECIAL_MISSING.get(c, DEFAULT_MISSING):
#             col[col == code] = np.nan
#         for old, new in RECODE.get(c, {}).items():
#             col[col == old] = new
#         if c in BINARY:
#             col = np.where(np.isnan(col), np.nan, (col == 1).astype(float))
#         if c in LOG_COLS:
#             col = np.log1p(col)
#         out[:, j] = col
#     return out


# # --------------------------------------------------------------------------
# # 3) Fit sur le train, puis transform sur train ET test
# # --------------------------------------------------------------------------


# def fit_preprocessing(x_clean):
#     """Apprend sur le TRAIN : médianes, catégories, moyennes, écarts-types."""
#     params = {"median": {}, "categories": {}, "nan_cols": []}
#     for j, c in enumerate(KEEP):
#         col = x_clean[:, j]
#         if np.isnan(col).any():
#             params["nan_cols"].append(j)
#         if c in NOMINAL:
#             params["categories"][j] = np.unique(col[~np.isnan(col)])
#         else:
#             params["median"][j] = np.nanmedian(col)
#     X = _build_matrix(x_clean, params)
#     params["mean"] = X.mean(axis=0)
#     params["std"] = X.std(axis=0)
#     params["std"][params["std"] == 0] = 1.0
#     return params


# def _build_matrix(x_clean, params):
#     feats = []
#     for j, c in enumerate(KEEP):
#         col = x_clean[:, j]
#         if c in NOMINAL:
#             # Une colonne 0/1 par catégorie ; NaN -> que des zéros
#             for cat in params["categories"][j]:
#                 feats.append((col == cat).astype(float))
#         else:
#             feats.append(np.where(np.isnan(col), params["median"][j], col))
#     # Indicateurs "la valeur était manquante"
#     for j in params["nan_cols"]:
#         feats.append(np.isnan(x_clean[:, j]).astype(float))
#     return np.column_stack(feats)


# def transform(x_clean, params):
#     """Applique imputation + one-hot + standardisation, ajoute le biais."""
#     X = _build_matrix(x_clean, params)
#     X = (X - params["mean"]) / params["std"]
#     return np.column_stack([np.ones(X.shape[0]), X])


# def preprocess(x_train, x_test, header):
#     """Pipeline complet. `header` = noms des colonnes de x_train
#     (sans 'Id' si on utilise load_csv_data, qui retire la colonne Id)."""
#     tr = clean_columns(x_train, header)
#     te = clean_columns(x_test, header)
#     params = fit_preprocessing(tr)
#     return transform(tr, params), transform(te, params)
