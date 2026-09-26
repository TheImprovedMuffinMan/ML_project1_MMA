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
import matplotlib.pyplot as plt

# ============================================================================ #
# loss function
def compute_loss(y, tx, w, loss_type='mse'):
    """Compute the MSE or MAE loss."""
    e = y - tx @ w

    if loss_type == 'mse':
        return 0.5 * np.mean(e**2)
    elif loss_type == 'mae':
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
    gradient = -(1/N) * tx.T @ error
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
def gradient_descent_stopping(y, tx, initial_w, max_iters, gamma, tol, 
                              condition='gradient'):
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

        if condition == 'gradient':
            if np.linalg.norm(gradient) < tol:
                break
        elif condition == 'weight':
            if np.linalg.norm(w_iter_new - w_iter) < tol:
                break
        elif condition == 'loss':
            if abs(loss_new - loss) < tol:
                break
        elif condition == 'max_iters':
            if counter >= max_iters:
                break
        else:
            raise ValueError("Invalid stopping condition. Must be 'gradient', 'weight', 'loss', or 'max_iters'.")

        w_iter = w_iter_new
        loss = loss_new
        counter += 1

    if counter == max_iters:
        print("Warning: Maximum iterations reached without " \
              "guaranteeing convergence.")
    return w_iter, loss

# ============================================================================ #
# Least squares
def least_squares(y, tx):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix

    returns: optimal weight and loss value
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    y_pred = tx @ w
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

    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
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
        gradient += 2 * lambda_ * w_iter # adding regularisation term
        w_iter = w_iter - gamma * gradient

    loss = compute_logistic_loss(y, tx, w_iter) # loss without regularisation

    return w_iter, loss