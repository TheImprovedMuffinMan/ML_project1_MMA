"""Some helper functions for project 1."""

import csv
import numpy as np
import os

# ---------------------------------------------------------------------------- #
def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# ---------------------------------------------------------------------------- #
def find_binary_like_columns(x, feature_names,max_unique=4,rare_threshold=0.05):
    candidates = []

    for j, name in enumerate(feature_names):
        col = x[:, j]

        # Ignore NaNs
        valid = col[~np.isnan(col)]

        values, counts = np.unique(valid, return_counts=True)

        # skip columns that are not binary (probably)
        if len(values) < 2 or len(values) > max_unique:
            continue

        proportions = counts / counts.sum()

        # Sort from most common to least common
        order = np.argsort(proportions)[::-1]
        values = values[order]
        proportions = proportions[order]

        # Yes/No dominate, the rest are rare (assumption)
        if len(values) == 2 or np.sum(proportions[2:]) <= rare_threshold:
            candidates.append((j, name, values, proportions))

    return candidates

# ---------------------------------------------------------------------------- #
def encode_binary_columns(x_train, x_test, feature_names, binary_specs):
    train_columns = []
    test_columns = []
    new_feature_names = []

    for j, name in enumerate(feature_names):
        train_col = x_train[:, j]
        test_col = x_test[:, j]

        # Leave columns not in BINARY_SPECS unchanged
        if name not in binary_specs:
            train_columns.append(train_col)
            test_columns.append(test_col)
            new_feature_names.append(name)
            continue

        spec = binary_specs[name]

        positive = spec["positive"]
        negative = spec["negative"]
        unknown = spec["unknown"]

        # Check for unexpected values
        allowed = positive + negative + unknown

        train_values = np.unique(train_col[~np.isnan(train_col)])

        unexpected = train_values[~np.isin(train_values, allowed)]

        if len(unexpected) > 0:
            raise ValueError(f"{name}: unexpected values {unexpected}")

        # Main binary column
        train_binary = np.isin(train_col, positive).astype(float)

        test_binary = np.isin(test_col, positive).astype(float)

        # Unknown / refused / missing column
        train_unknown = (np.isin(train_col, unknown) | np.isnan(train_col)
                        ).astype(float)

        test_unknown = (np.isin(test_col, unknown) | np.isnan(test_col)
                       ).astype(float)

        train_columns.extend([train_binary,train_unknown])

        test_columns.extend([test_binary,test_unknown])

        new_feature_names.extend([f"{name}_positive", f"{name}_unknown"])

    return (np.column_stack(train_columns),
            np.column_stack(test_columns),
            np.array(new_feature_names))