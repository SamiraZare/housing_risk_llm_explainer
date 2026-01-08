import numpy as np

def compute_cost_linear(X, y, w, b):
    """
    Compute cost for linear regression using vectorized implementation.
    """
    m = X.shape[0]
    predictions = X @ w + b
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

def gradient_linear(X, y, w, b):
    """
    Compute gradients for linear regression.
    """
    m = X.shape[0]
    predictions = X @ w + b
    error = predictions - y
    dj_dw = (1 / m) * (X.T @ error)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent_linear(X, y, w_in, b_in, alpha, num_iters):
    """
    Gradient descent for linear regression (similar to the Coursera labs).
    """
    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_linear(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost_linear(X, y, w, b)
        J_history.append(cost)

    return w, b, J_history

def predict_linear(X, w, b):
    return X @ w + b
