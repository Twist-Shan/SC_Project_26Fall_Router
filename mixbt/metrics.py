import numpy as np


def logloss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    """Pairwise accuracy on decisive battles (y in {0, 1})."""
    m = (y == 0) | (y == 1)
    return float(np.mean((p[m] > 0.5) == (y[m] == 1)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def evaluate(p, y):
    return dict(logloss=logloss(p, y), accuracy=accuracy(p, y), brier=brier(p, y))
