import numpy as np

def normalize_features(X):
    """
    Mean normalization as in the Coursera course:
    X_norm = (X - mu) / (max - min)
    Returns X_norm, mu, ptp (range)
    """
    mu = np.mean(X, axis=0)
    ptp = np.ptp(X, axis=0)  # max - min
    X_norm = (X - mu) / ptp
    return X_norm, mu, ptp

def apply_normalization(X, mu, ptp):
    """
    Apply stored normalization stats to new data.
    """
    return (X - mu) / ptp

def add_polynomial_features(X, degree=2):
    """
    Simple polynomial feature expansion for scalar feature case,
    or just return X for now and extend later.
    """
    # Example for 1D feature X[:, 0]
    if X.shape[1] == 1 and degree > 1:
        x = X[:, 0:1]
        feats = [x]
        for d in range(2, degree+1):
            feats.append(x**d)
        return np.concatenate(feats, axis=1)
    else:
        # TODO: extend for multi-feature polynomial expansion if desired
        return X
