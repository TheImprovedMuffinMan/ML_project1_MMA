# -*- coding: utf-8 -*-
# ============================================================================ #
# Authors:
# Date :
# Contains:
# ============================================================================ #

# ============================================================================ #
# Imports
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================ #
# loss function
def compute_loss(y, tx, w, loss_type='mse'):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    w: shape=(D, ), model parameters
    loss_type: string, type of loss function to compute ('mse' or 'mae')

    output: loss value (scalar)
    """
    N = y.shape[0]
    if loss_type == 'mse':
        e = y - tx @ w
        loss = (1/(2*N)) * np.sum(e**2)
    elif loss_type == 'mae':
        e = y - tx @ w
        loss = (1/N) * np.sum(np.abs(e))
    else:
        raise ValueError("Invalid loss_type. Must be 'mse' or 'mae'.")
    
    return loss


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
# Batch iterator
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    MISSING
    """

# ============================================================================ #
# Optimisation algorithms
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
        loss = compute_loss(y, tx, w_iter)
        gradient = compute_gradient(y, tx, w_iter)
        w_iter = w_iter - gamma * gradient

    return w_iter, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    initial_w: shape=(D, ), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate
    batch_size: int, size of mini-batches for SGD 
                (default=1 required for project 1)
    
    returns: optimal weight and loss value
    """


def least_squares(y, tx):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix

    returns: optimal weight and loss value
    """

def ridge_regression(y, tx, lambda_):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    lambda_: float, regularization parameter

    returns: optimal weight and loss value
    """

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    y: shape=(N, ), output vector
    tx: shape=(N,D), input matrix
    initial_w: shape=(D, ), initial model parameters
    max_iters: int, number of iterations
    gamma: float, learning rate

    returns: optimal weight and loss value
    """

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