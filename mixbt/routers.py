"""Baseline routers used for comparison in the routing experiments.

SWBT        similarity-weighted Bradley--Terry, adapted from RouteLLM's SW ranking (Ong et al. 2024):
            for a new prompt, every training battle i gets weight gamma**(1 + S_i), where S_i is the cosine
            similarity of the prompt embeddings divided by its maximum over the training set, and a ridge-penalised
            BT model is fitted to the weighted battles. Fitted on edge-level sufficient statistics, in batches.
Bilinear    a RouteLLM-inspired matrix-factorisation baseline: logit P(a beats b | x) =
            (u_a - u_b) + (V_a - V_b)^T A f(x), with item scores u, item embeddings V (M x k) and a linear map A of
            the prompt features; ridge on V and all of A, including its intercept column;
            fitted by L-BFGS. This is an adapted implementation, not the official router.
PairLogit   a discriminative router: logit P(i beats j | x) = (r_i - r_j) + theta^T f(x) * s(i, j), where r are global
            BT scores (offset) and s(i, j) = +1 if i is the strong model and -1 otherwise; ridge on theta.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


def _sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


class SWBT:
    def __init__(self, gamma=10.0, lam=2.0, newton_iter=8, batch=256):
        self.gamma, self.lam, self.newton_iter, self.batch = gamma, lam, newton_iter, batch

    def fit(self, E, a, b, y, M):
        """E: n x d prompt embeddings; a, b: item ids; y = 1 if a won."""
        E = np.asarray(E, float)
        self.E_ = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        win_lo = np.where(a < b, y, 1 - y)  # 1 if the lower-index item won
        keys = lo * M + hi
        uk, e = np.unique(keys, return_inverse=True)
        self.edges_ = np.column_stack([uk // M, uk % M])
        L, n = len(uk), len(y)
        self.C_ = sp.csr_matrix((np.ones(n), (np.arange(n), e)), shape=(n, L))      # battle -> edge
        self.Cw_ = sp.csr_matrix((win_lo, (np.arange(n), e)), shape=(n, L))         # wins of lower item
        G = np.zeros((L, M))
        G[np.arange(L), self.edges_[:, 0]] = 1.0
        G[np.arange(L), self.edges_[:, 1]] = -1.0
        self.G_, self.M_ = G, M
        # global fit (all weights equal) as warm start
        self.r0_ = self._solve(np.asarray(self.C_.sum(0)).ravel()[None], np.asarray(self.Cw_.sum(0)).ravel()[None],
                               np.zeros((1, M)), n_scale=n)[0]
        return self

    def _solve(self, N, S, R, n_scale):
        """Batched ridge-penalised BT on edge statistics. N, S: B x L (weighted counts / wins), R: B x M."""
        G, lam = self.G_, self.lam
        for _ in range(self.newton_iter):
            eta = R @ G.T                                  # B x L
            p = _sigmoid(eta)
            grad = (S - N * p) @ G - lam * R               # B x M
            c = N * p * (1 - p)                            # B x L
            H = np.matmul(G.T[None], c[:, :, None] * G[None]) + lam * np.eye(self.M_)[None]
            R = R + np.linalg.solve(H, grad[..., None])[..., 0]
        return R

    def item_scores(self, Et):
        Et = np.asarray(Et, float)
        Et = Et / (np.linalg.norm(Et, axis=1, keepdims=True) + 1e-12)
        out = np.empty((len(Et), self.M_))
        n = self.E_.shape[0]
        for s in range(0, len(Et), self.batch):
            sim = Et[s:s + self.batch] @ self.E_.T                          # B x n
            S = sim / np.maximum(sim.max(axis=1, keepdims=True), 1e-12)
            w = self.gamma ** (1.0 + S)
            w *= n / w.sum(axis=1, keepdims=True)                          # mean weight one
            N = np.asarray(self.C_.T @ w.T).T                              # B x L weighted battle counts
            Sw = np.asarray(self.Cw_.T @ w.T).T                            # B x L weighted wins
            out[s:s + self.batch] = self._solve(N, Sw, np.tile(self.r0_, (len(w), 1)), n)
        return out

    def predict_proba(self, Et, at, bt):
        R = self.item_scores(Et)
        i = np.arange(len(at))
        return _sigmoid(R[i, at] - R[i, bt])


class PairLogit:
    """Logistic router on prompt features for battles between a 'strong' set and the rest, with a global-BT offset."""

    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, X, offset, s, y):
        """X: n x p features (no intercept needed, one is added), offset: global-BT logit of 'a beats b',
        s: +1 if a is the strong model, -1 if b is, y = 1 if a won."""
        X = np.column_stack([np.ones(len(X)), X])
        Z = X * s[:, None]
        lam = self.lam

        def f(th):
            eta = offset + Z @ th
            ll = y * eta - np.logaddexp(0, eta)
            g = Z.T @ (y - _sigmoid(eta))
            return -(ll.sum()) + 0.5 * lam * th[1:] @ th[1:], -(g) + lam * np.r_[0, th[1:]]

        res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.theta_ = res.x
        return self

    def predict_proba(self, X, offset, s):
        X = np.column_stack([np.ones(len(X)), X])
        return _sigmoid(offset + (X * s[:, None]) @ self.theta_)


class Bilinear:
    def __init__(self, k=8, lam=10.0, seed=0, maxiter=500):
        self.k, self.lam, self.seed, self.maxiter = k, lam, seed, maxiter

    def _unpack(self, th):
        M, k, p = self.M_, self.k, self.p_
        u = th[:M]
        V = th[M:M + M * k].reshape(M, k)
        A = th[M + M * k:].reshape(k, p)
        return u, V, A

    def fit(self, X, a, b, y, M):
        X = np.column_stack([np.ones(len(X)), X])
        self.M_, self.p_ = M, X.shape[1]
        rng = np.random.default_rng(self.seed)
        th0 = np.concatenate([np.zeros(M), 0.01 * rng.normal(size=M * self.k), 0.01 * rng.normal(size=self.k * self.p_)])
        lam, n = self.lam, len(y)

        def f(th):
            u, V, A = self._unpack(th)
            Z = X @ A.T                                    # n x k
            dV = V[a] - V[b]                               # n x k
            eta = u[a] - u[b] + np.sum(dV * Z, axis=1)
            r = y - _sigmoid(eta)                          # d loglik / d eta
            ll = np.sum(y * eta - np.logaddexp(0, eta))
            gu = np.bincount(a, r, M) - np.bincount(b, r, M)
            rZ = r[:, None] * Z
            gV = np.zeros((M, self.k))
            np.add.at(gV, a, rZ)
            np.add.at(gV, b, -rZ)
            gA = (r[:, None] * dV).T @ X
            # Penalise the intercept column as well: leaving A[:, 0] free permits
            # V -> epsilon * V, A[:, 0] -> A[:, 0] / epsilon to preserve a
            # low-rank intercept while driving its factor penalty to zero.
            pen = 0.5 * lam * (np.sum(V ** 2) + np.sum(A ** 2)) + 0.5 * 1e-2 * np.sum(u ** 2)
            g = -np.concatenate([gu - 1e-2 * u, (gV - lam * V).ravel(),
                                 (gA - lam * A).ravel()])
            return -ll + pen, g

        res = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=self.maxiter))
        self.theta_ = res.x
        return self

    def predict_proba(self, X, a, b):
        X = np.column_stack([np.ones(len(X)), X])
        u, V, A = self._unpack(self.theta_)
        return _sigmoid(u[a] - u[b] + np.sum((V[a] - V[b]) * (X @ A.T), axis=1))
