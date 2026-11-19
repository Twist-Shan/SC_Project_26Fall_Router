"""Acceleration and initialisation helpers shared by the EM implementations.

anderson_em : safeguarded Anderson acceleration of a monotone fixed-point map
              (in the spirit of DAAREM, Henderson & Varadhan 2019). Each
              iteration costs one EM map; an extrapolated point is accepted
              only if it does not decrease the penalised log-likelihood,
              otherwise the plain EM step is taken and the history is reset.
              The accepted sequence is therefore monotone, like EM.
"""
from __future__ import annotations

import time

import numpy as np


def anderson_em(em_map, x0, m=5, max_iter=1000, tol=1e-7, reg=1e-8, slack=0.0, history=None, patience=3):
    """Safeguarded Anderson acceleration.

    em_map(x) -> (g, obj_x): one EM update g = M(x) and the penalised
    objective at x (available for free from the E-step at x).
    Returns (x, obj, n_maps, trace) where trace is a list of (time, obj).
    """
    t0 = time.perf_counter()
    x = x0
    g, f = em_map(x)
    n_maps = 1
    X, G = [x], [g]
    trace = [(0.0, f)] if history is None else history
    f_old = f
    small = 0
    for _ in range(max_iter):
        # Anderson (type II) extrapolation from the last m+1 pairs
        if len(X) >= 2:
            Rm = np.array([gi - xi for gi, xi in zip(G, X)]).T  # d x (k+1) residuals
            dR = np.diff(Rm, axis=1)
            dG = np.diff(np.array(G).T, axis=1)
            r = Rm[:, -1]
            A = dR.T @ dR
            A += reg * (np.trace(A) / A.shape[0] + 1e-12) * np.eye(A.shape[0])
            gam = np.linalg.solve(A, dR.T @ r)
            x_new = G[-1] - dG @ gam
        else:
            x_new = g
        g_new, f_new = em_map(x_new)
        n_maps += 1
        if np.isfinite(f_new) and f_new >= f - slack * (abs(f) + 1):
            x, g, f = x_new, g_new, f_new
            X.append(x)
            G.append(g)
            if len(X) > m + 1:
                X.pop(0)
                G.pop(0)
        else:  # safeguard: plain EM step from the current point, reset the history
            x = g
            g, f = em_map(x)
            n_maps += 1
            X, G = [x], [g]
        trace.append((time.perf_counter() - t0, f))
        small = small + 1 if abs(f - f_old) < tol * (abs(f_old) + 1) else 0
        if small >= patience:
            break
        f_old = f
    return x, f, n_maps, trace
