import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    """
    Logistic regression cost function (binary cross-entropy).
    """
    m = X.shape[0]
    z = X @ w + b
    f_wb = sigmoid(z)
    epsilon = 1e-8  # to avoid log(0)
    cost = -(1/m) * np.sum(y * np.log(f_wb + epsilon) + (1-y) * np.log(1 - f_wb + epsilon))
    return cost

def gradient_logistic(X, y, w, b):
    m = X.shape[0]
    z = X @ w + b
    f_wb = sigmoid(z)
    error = f_wb - y
    dj_dw = (1/m) * (X.T @ error)
    dj_db = (1/m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent_logistic(X, y, w_in, b_in, alpha, num_iters):
    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost_logistic(X, y, w, b)
        J_history.append(cost)

    return w, b, J_history

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def predict_class(X, w, b, threshold=0.5):
    proba = predict_proba(X, w, b)
    return (proba >= threshold).astype(int)
