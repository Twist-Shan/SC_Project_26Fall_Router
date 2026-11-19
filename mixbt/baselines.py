"""Baselines that the mixture model is compared against."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from .model import PairData, log_gate, weighted_bt_newton


class GlobalBT:
    """Single Bradley--Terry ranking (equivalent to the mixture with K=1)."""

    def __init__(self, lam_R=1e-2):
        self.lam_R = lam_R

    def fit(self, a, b, y, M, weights=None, prior=None):
        pd = PairData(a, b, y, M)
        w = np.ones((len(a), 1)) if weights is None else np.asarray(weights)[:, None]
        N, Wn = pd.aggregate(w)
        self.r_ = weighted_bt_newton(pd.u, pd.v, N[:, 0], Wn[:, 0], M, lam=self.lam_R, prior=prior)
        self.M = M
        return self

    def predict_proba(self, F, a, b):
        return expit(self.r_[a] - self.r_[b])


class ClusterBT:
    """Two-stage baseline: hard-cluster the prompts (k-means on the
    embedding, or given categories), then fit a separate BT model in each
    cluster. This mimics leaderboards that split by a fixed category set."""

    def __init__(self, K, lam_R=1e-2, seed=0, shrink_to="zero"):
        self.K = K
        self.lam_R = lam_R
        self.seed = seed
        self.shrink_to = shrink_to

    def fit(self, F, a, b, y, M, labels=None):
        from sklearn.cluster import KMeans

        if labels is None:
            self.km_ = KMeans(self.K, n_init=5, random_state=self.seed).fit(F[:, 1:])
            labels = self.km_.labels_
        else:
            self.km_ = None
        self.R_ = np.zeros((M, self.K))
        prior = GlobalBT(0.1).fit(a, b, y, M).r_ if self.shrink_to == "global" else None
        for k in range(self.K):
            idx = labels == k
            if idx.sum() == 0:
                continue
            self.R_[:, k] = GlobalBT(self.lam_R).fit(a[idx], b[idx], y[idx], M, prior=prior).r_
        self.labels_ = labels
        self.M = M
        return self

    def predict_proba(self, F, a, b, labels=None):
        if labels is None:
            labels = self.km_.predict(F[:, 1:])
        return expit(self.R_[a, labels] - self.R_[b, labels])


class CapabilityBT:
    """Formulation (B) of the meeting notes: a single prompt-specific utility

        P(a > b | x) = sigma( sum_k pi_k(x) (r_{a,k} - r_{b,k}) ).

    No latent variable, so it is fitted by direct L-BFGS on the penalised
    log-likelihood (non-convex because of the product pi * R)."""

    def __init__(self, K, lam_W=1e-2, lam_R=1e-2, maxiter=3000, seed=0, shrink_to="zero"):
        self.shrink_to = shrink_to
        self.prior_ = None
        self.K = K
        self.lam_W = lam_W
        self.lam_R = lam_R
        self.maxiter = maxiter
        self.seed = seed

    def _unpack(self, theta, p, M):
        K = self.K
        W = np.vstack([theta[: (K - 1) * p].reshape(K - 1, p), np.zeros((1, p))])
        R = theta[(K - 1) * p :].reshape(M, K)
        return W, R

    def _f_and_g(self, theta, F, a, b, y, M):
        """Negative penalised log-likelihood and its gradient."""
        K, p = self.K, F.shape[1]
        mask = np.ones(p)
        mask[0] = 0
        W, R = self._unpack(theta, p, M)
        lp = log_gate(F, W)
        pi = np.exp(lp)
        D = R[a] - R[b]  # n x K
        d = np.sum(pi * D, axis=1)
        ll = np.sum(y * log_expit(d) + (1 - y) * log_expit(-d))
        pr = np.zeros(M) if self.prior_ is None else self.prior_
        pen = 0.5 * self.lam_W * np.sum((W * mask) ** 2) + 0.5 * self.lam_R * np.sum((R - pr[:, None]) ** 2)
        g = y - expit(d)  # d ll / d d_i
        GR = np.zeros((M, K))
        np.add.at(GR, a, g[:, None] * pi)
        np.add.at(GR, b, -g[:, None] * pi)
        GR -= self.lam_R * (R - pr[:, None])
        # d d_i / d eta_il = pi_il (D_il - d_i)
        GW = (g[:, None] * pi * (D - d[:, None])).T @ F - self.lam_W * W * mask
        grad = np.concatenate([GW[:-1].ravel(), GR.ravel()])
        return -(ll - pen), -grad

    def fit(self, F, a, b, y, M, W0=None, R0=None):
        K, p = self.K, F.shape[1]
        rng = np.random.default_rng(self.seed)
        self.prior_ = GlobalBT(0.1).fit(a, b, y, M).r_ if self.shrink_to == "global" else None
        if W0 is None:
            W0 = rng.normal(scale=0.1, size=(K, p))
            W0[-1] = 0
        if R0 is None:
            r = GlobalBT().fit(a, b, y, M).r_
            R0 = r[:, None] + rng.normal(scale=0.1, size=(M, K))
        t0 = time.perf_counter()

        def f_and_g(theta):
            return self._f_and_g(theta, F, a, b, y, M)

        theta0 = np.concatenate([W0[:-1].ravel(), R0.ravel()])
        res = minimize(f_and_g, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter, "gtol": 1e-6})
        self.W_, self.R_ = self._unpack(res.x, p, M)
        self.R_ -= self.R_.mean(0)
        self.objective_ = -res.fun
        self.fit_time_ = time.perf_counter() - t0
        self.M = M
        return self

    def gate(self, F):
        return np.exp(log_gate(F, self.W_))

    def predict_proba(self, F, a, b):
        pi = self.gate(F)
        return expit(np.sum(pi * (self.R_[a] - self.R_[b]), axis=1))
