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
    Note: Taken from CS-433 course EPFL.

    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

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
        gradient = compute_gradient(y, tx, w_iter)
        w_iter = w_iter - gamma * gradient

    # We only need final loss
    loss = compute_loss(y, tx, w_iter)

    return w_iter, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size = 1,
                           num_batches = 1):
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
    w_iter = initial_w.copy()
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size = batch_size,
                                            num_batches = num_batches):
            gradient = compute_gradient(y_batch, tx_batch, w_iter)
            w_iter = w_iter - gamma * gradient

    loss = compute_loss(y, tx, w_iter)

    return w_iter, loss


# def stochastic_gradient_descent_stopping(y, tx, initial_w, max_iters, gamma, 
#                                          tol, batch_size, num_batches, 
#                                          condition='gradient'):
#     """
#     NOT REQUIRED FOR PROJECT 1

#     y: shape=(N, ), output vector
#     tx: shape=(N,D), input matrix
#     initial_w: shape=(D, ), initial model parameters
#     max_iters: int, number of iterations
#     gamma: float, learning rate
#     tol: float, tolerance for stopping criterion
#     condition: string stopping condition ('gradient', 'weight', 'loss', 'max_iters')
#     """
#     counter = 0
#     w_iter = initial_w.copy()
#     loss = compute_loss(y, tx, w_iter)

#     while counter < max_iters:
#         for y_batch , tx_batch in batch_iter(y, tx, batch_size = batch_size,
#                                              num_batches=num_batches):
#             gradient = compute_gradient(y, tx, w_iter)
#             w_iter_new = w_iter - gamma * gradient
#             loss_new = compute_loss(y, tx, w_iter_new)

#             if condition == 'gradient':
#                 if np.linalg.norm(gradient) < tol:
#                     break
#             elif condition == 'weight':
#                 if np.linalg.norm(w_iter_new - w_iter) < tol:
#                     break
#             elif condition == 'loss':
#                 if abs(loss_new - loss) < tol:
#                     break
#             elif condition == 'max_iters':
#                 if counter >= max_iters:
#                     break
#             else:
#                 raise ValueError("Invalid stopping condition. Must be 'gradient', 'weight', 'loss', or 'max_iters'.")

#             w_iter = w_iter_new
#             loss = loss_new

#         counter += 1

#     if counter == max_iters:
#         print("Warning: Maximum iterations reached without " \
#               "guaranteeing convergence.")
#     return w_iter, loss



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