"""Synthetic data from the mixture model (and from formulation B)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax


def make_params(M, K, p, delta=1.0, gate_scale=1.5, base_sd=1.0, rng=None):
    """True parameters.

    R*_{m,k} = base_m + delta * dev_{m,k}  (columns centred). `delta`
    controls how different the domain-specific rankings are (separation,
    Condition E1); `gate_scale` controls how sharply the features determine
    the domain (Assumption I3 / Condition E5).
    """
    rng = np.random.default_rng(rng)
    base = rng.normal(0, base_sd, M)
    dev = rng.normal(0, 1, (M, K))
    R = base[:, None] + delta * dev
    R -= R.mean(axis=0)
    W = rng.normal(0, gate_scale, (K, p))
    W[:, 0] = 0.0  # zero intercepts -> roughly balanced domain masses
    W[-1] = 0.0
    return W, R


def simulate(n, W, R, rng=None, edges=None, model="A"):
    """Draw n battles. Features: f = (1, N(0, I_{p-1})).

    model='A': mixture of BT (latent z); model='B': capability aggregation.
    edges: optional (L x 2) array restricting which pairs can meet.
    """
    rng = np.random.default_rng(rng)
    K, p = W.shape
    M = R.shape[0]
    F = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    pi = softmax(F @ W.T, axis=1)
    if edges is None:
        a = rng.integers(0, M, n)
        b = (a + rng.integers(1, M, n)) % M  # uniform over b != a
    else:
        e = edges[rng.integers(0, len(edges), n)]
        flip = rng.random(n) < 0.5
        a = np.where(flip, e[:, 1], e[:, 0])
        b = np.where(flip, e[:, 0], e[:, 1])
    if model == "A":
        u = rng.random(n)[:, None]
        z = (u > np.cumsum(pi, axis=1)).sum(axis=1)
        z = np.minimum(z, K - 1)
        prob = expit(R[a, z] - R[b, z])
    else:
        z = None
        prob = expit(np.sum(pi * (R[a] - R[b]), axis=1))
    y = (rng.random(n) < prob).astype(float)
    return dict(F=F, a=a, b=b, y=y, z=z, pi=pi, prob=prob)


def true_proba(F, a, b, W, R, model="A"):
    pi = softmax(F @ W.T, axis=1)
    if model == "A":
        return np.sum(pi * expit(R[a] - R[b]), axis=1)
    return expit(np.sum(pi * (R[a] - R[b]), axis=1))


def simulate_style(n, W, R, B, rng=None):
    """Battles from the style-adjusted mixture: domain-specific style effects
    B (K x q); style differences d_i ~ U(-1, 1)^q."""
    rng = np.random.default_rng(rng)
    K, p = W.shape
    M = R.shape[0]
    q = B.shape[1]
    F = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    D = rng.uniform(-1, 1, size=(n, q))
    pi = softmax(F @ W.T, axis=1)
    a = rng.integers(0, M, n)
    b = (a + rng.integers(1, M, n)) % M
    z = np.minimum((rng.random(n)[:, None] > np.cumsum(pi, axis=1)).sum(1), K - 1)
    eta = R[a, z] - R[b, z] + np.sum(D * B[z], 1)
    y = (rng.random(n) < expit(eta)).astype(float)
    return dict(F=F, a=a, b=b, y=y, D=D, z=z)
