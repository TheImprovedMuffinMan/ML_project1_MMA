# -*- coding: utf-8 -*-
# ============================================================================ #
# Authors: Mathilde Aymon, Andras Horkay, Mélusine Plumart
# Submission Date: 29/10/2026
# Contains: implementations of regression algorithms:
#               - Gradient Descent (GD)
#               - Stochastic Gradient Descent (SGD)
#               - Least Squares (LS)
#               - Ridge Regression (RR)
#               - Logistic Regression (LR)
#               - Regularized Logistic Regression (RLR)
#           along with their respective loss functions and gradient
#           computations.
# ============================================================================ #

# ============================================================================ #
# Imports
import numpy as np


# ============================================================================ #
# loss function
def compute_loss(y, tx, w, loss_type="mse"):
    """Compute the MSE or MAE loss."""
    e = y - tx @ w

    if loss_type == "mse":
        return 0.5 * np.mean(e**2)
    elif loss_type == "mae":
        return np.mean(np.abs(e))
    else:
        raise ValueError("Invalid loss_type. Must be 'mse' or 'mae'.")


# ============================================================================ #
# Gradient computation
def compute_gradient(y, tx, w):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    w: shape=(D, ), model parameters

    returns: gradient vector of shape (D, )
    """
    error = y - tx @ w
    N = y.shape[0]
    gradient = -(1 / N) * tx.T @ error
    return gradient


# ============================================================================ #
# Optimisation algorithms
# ============================================================================ #

# ============================================================================ #
# Gradient descent and stochastic gradient descent


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    initial_w: shape=(D, ), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate

    returns: optimal weight and loss value
    """

    w_iter = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w_iter)
        w_iter = w_iter - gamma * gradient

    # We only need final loss
    loss = compute_loss(y, tx, w_iter)

    return w_iter, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    y: shape=(N,), output vector
    tx: shape=(N, D), input matrix
    initial_w: shape=(D,), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate

    returns: optimal weight and loss value
    """
    w_iter = initial_w.copy()
    N = y.shape[0]

    for _ in range(max_iters):
        # the minibatch size is 1, simpler than using batch_iter
        i = np.random.randint(N)
        error = y[i] - tx[i] @ w_iter
        gradient = -tx[i] * error
        w_iter = w_iter - gamma * gradient

    loss = compute_loss(y, tx, w_iter)

    return w_iter, loss


# optional code with stopping criterion for gradient descent
def gradient_descent_stopping(
    y, tx, initial_w, max_iters, gamma, tol, condition="gradient"
):
    """
    NOT REQUIRED FOR PROJECT 1

    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    initial_w: shape=(D, ), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate
    tol: float, tolerance for stopping criterion
    condition: string stopping condition ('gradient', 'weight', 'loss', 'max_iters')
    """
    counter = 0
    w_iter = initial_w.copy()
    loss = compute_loss(y, tx, w_iter)
    while counter < max_iters:
        gradient = compute_gradient(y, tx, w_iter)
        w_iter_new = w_iter - gamma * gradient
        loss_new = compute_loss(y, tx, w_iter_new)

        if condition == "gradient":
            if np.linalg.norm(gradient) < tol:
                break
        elif condition == "weight":
            if np.linalg.norm(w_iter_new - w_iter) < tol:
                break
        elif condition == "loss":
            if abs(loss_new - loss) < tol:
                break
        elif condition == "max_iters":
            if counter >= max_iters:
                break
        else:
            raise ValueError(
                "Invalid stopping condition. Must be 'gradient', 'weight', 'loss', or 'max_iters'."
            )

        w_iter = w_iter_new
        loss = loss_new
        counter += 1

    if counter == max_iters:
        print(
            "Warning: Maximum iterations reached without " "guaranteeing convergence."
        )
    return w_iter, loss


# ============================================================================ #
# Least squares
def least_squares(y, tx):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix

    returns: optimal weight and loss value
    """
    A, b = tx.T @ tx, tx.T @ y       
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    loss = compute_loss(y, tx, w)
    return w, loss


# ============================================================================ #
# Ridge regression
def ridge_regression(y, tx, lambda_):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    lambda_: float, regularization parameter

    returns: optimal weight and loss value
    """
    N, D = tx.shape
    lambda_prime = 2 * N * lambda_

    A, b = tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
        
    loss = compute_loss(y, tx, w)

    return w, loss


# ============================================================================ #
# Logistic regression

def sigmoid(t):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Compute the logistic loss."""
    y_pred = sigmoid(tx @ w)
    # Avoid log(0), we can clip predictions
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss."""
    N = y.shape[0]
    y_pred = sigmoid(tx @ w)
    gradient = tx.T @ (y_pred - y) / N
    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    initial_w: shape=(D, ), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate

    returns: optimal weight and loss value
    """
    w_iter = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w_iter)
        w_iter = w_iter - gamma * gradient

    loss = compute_logistic_loss(y, tx, w_iter)

    return w_iter, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    lambda_: float, regularization parameter
    initial_w: shape=(D, ), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate

    returns: optimal weight and loss value
    """
    w_iter = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w_iter)
        gradient += 2 * lambda_ * w_iter  # adding regularisation term
        w_iter = w_iter - gamma * gradient

    loss = compute_logistic_loss(y, tx, w_iter)  # loss without regularisation

    return w_iter, loss


# ============================================================================ #
# Cross-validation for ridge regression
def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed
    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_ridge(y, tx, k_indices, k, lambda_):
    """
    Return train and test RMSE for the k-th fold.
    """

    val_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis=0).flatten()

    y_train = y[train_indices]
    tx_train = tx[train_indices]

    y_val = y[val_indices]
    tx_val = tx[val_indices]

    w, _ = ridge_regression(y_train, tx_train, lambda_)

    loss_tr = compute_loss(y_train, tx_train, w)
    loss_val = compute_loss(y_val, tx_val, w)

    return loss_tr, loss_val


def cross_validation_ridge_grid(y, tx, k_fold, lambdas, seed):

    k_indices = build_k_indices(y, k_fold, seed)

    mean_train_loss = np.zeros(len(lambdas))
    mean_val_loss = np.zeros(len(lambdas))

    for i, lambda_ in enumerate(lambdas):

        train_losses = []
        val_losses = []

        for k in range(k_fold):

            loss_tr, loss_val = cross_validation_ridge(y, tx, k_indices, k, lambda_)

            train_losses.append(loss_tr)
            val_losses.append(loss_val)

        mean_train_loss[i] = np.mean(train_losses)
        mean_val_loss[i] = np.mean(val_losses)

    id = np.argmin(mean_val_loss)

    best_lambda = lambdas[id]

    return best_lambda, mean_train_loss, mean_val_loss

# ============================================================================ #
# Cross validation for regularized logistic regression
def cross_validation_logistic(y, tx, k_indices, k, lambda_, gamma = 0.01, max_iters = 1000):
    """
    Return train and test loss for the k-th fold.
    """

    val_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis=0).flatten()

    y_train = y[train_indices]
    tx_train = tx[train_indices]

    y_val = y[val_indices]
    tx_val = tx[val_indices]

    initial_w = np.zeros(tx.shape[1])
    w, _ = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters = max_iters, gamma = gamma)

    loss_tr = compute_logistic_loss(y_train, tx_train, w)
    loss_val = compute_logistic_loss(y_val, tx_val, w)

    return loss_tr, loss_val

def cross_validation_logistic_grid(y, tx, k_fold, lambdas, seed, gamma = 0.01, max_iters = 1000):

    k_indices = build_k_indices(y, k_fold, seed)

    mean_train_loss = np.zeros(len(lambdas))
    mean_val_loss = np.zeros(len(lambdas))

    for i, lambda_ in enumerate(lambdas):

        train_losses = []
        val_losses = []

        for k in range(k_fold):

            loss_tr, loss_val = cross_validation_logistic(y, tx, k_indices, k, lambda_, gamma = gamma, max_iters = max_iters)

            train_losses.append(loss_tr)
            val_losses.append(loss_val)

        mean_train_loss[i] = np.mean(train_losses)
        mean_val_loss[i] = np.mean(val_losses)

    id = np.argmin(mean_val_loss)

    best_lambda = lambdas[id]

    return best_lambda, mean_train_loss, mean_val_loss

# ============================================================================ #
def classification_metrics(y, y_pred):

    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == -1) & (y_pred == -1))
    fp = np.sum((y == -1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))

    accuracy = (tp + tn) / len(y)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return accuracy, precision, recall, f1

# ============================================================================ #
# Cross validation for F1

DEFAULT_THRESHOLDS = np.arange(0.05, 0.55, 0.025)

def f1_for_thresholds(y, scores, thresholds=DEFAULT_THRESHOLDS):
    """F1 for the rule 'score >= t' for every t."""
    return np.array(
        [
            classification_metrics(y, np.where(scores >= t, 1, -1))[3]
            for t in thresholds
        ]
    )
 
def tune_threshold(y, scores, thresholds=DEFAULT_THRESHOLDS):
    """Return (best_threshold, best_f1) on the given validation data."""
    f1s = f1_for_thresholds(y, scores, thresholds)
    best = int(np.argmax(f1s))
    return thresholds[best], f1s[best]

def cross_validation_f1(y, tx, folds, fit_fn, score_fn, thresholds=DEFAULT_THRESHOLDS):
    """
    y: {-1, 1} labels, tx: features, folds: list of index arrays
    fit_fn(y_train, tx_train) -> w          (convert labels inside if needed)
    score_fn(tx_val, w)       -> scores
 
    Returns (mean_f1, std_f1): arrays of length len(thresholds), computed over folds.
    The same threshold is used in every fold.
    """
    f1 = np.zeros((len(folds), len(thresholds)))
    for k, val in enumerate(folds):
        train = np.concatenate([f for i, f in enumerate(folds) if i != k])
        w = fit_fn(y[train], tx[train])
        f1[k] = f1_for_thresholds(y[val], score_fn(tx[val], w), thresholds)
    return f1.mean(axis=0), f1.std(axis=0)
 
def grid_search_f1(
    y, tx, folds, params, make_fit_fn, score_fn, thresholds=DEFAULT_THRESHOLDS
):
    """
    Choose a hyper-parameter (lambda, gamma, ...) AND a threshold by mean CV F1.
 
    params:      list of hyper-parameter values
    make_fit_fn: p -> fit_fn(y_train, tx_train) -> w
    Returns (best_param, best_threshold, best_f1, table)
    where table[i, j] = mean CV F1 for params[i] and thresholds[j].
    """
    table = np.zeros((len(params), len(thresholds)))
    for i, p in enumerate(params):
        table[i], _ = cross_validation_f1(
            y, tx, folds, make_fit_fn(p), score_fn, thresholds
        )
    i, j = np.unravel_index(np.argmax(table), table.shape)
    return params[i], thresholds[j], table[i, j], table