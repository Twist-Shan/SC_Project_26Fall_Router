"""
Latent-domain mixture of Bradley--Terry models with response-style covariates.

For battle i with prompt features f_i, models (a_i, b_i), style difference
d_i in R^q (antisymmetric: swapping a and b flips its sign) and outcome
y_i in {0, 1} (1 if a wins; ties are not modelled), the domain-k component is

    eta_ik = r_{a,k} - r_{b,k} + beta_k' d_i,
    P(y_i = 1 | z_i = k) = sigma(eta_ik),

and z_i | x_i ~ softmax(W f_i). With style=None this is exactly the model of
the proposal (formulation A). With style='domain' each latent domain has its
own skills r_k AND its own sensitivity beta_k to presentation (length,
markdown), i.e. its own preference mechanism; style='shared' uses one beta.

Penalised likelihood (all domain parameters shrunk to the K = 1 fit):

    pen = lam_W/2 ||W_{-int}||^2
          + sum_k [ lam_R/2 ||r_k - r0||^2 + lam_B/2 ||beta_k - beta0||^2 ].

EM: the M-step is (i) a gamma-weighted multinomial logistic regression for W
and (ii) for every k a gamma-weighted logistic regression in (r_k, beta_k)
with design (e_a - e_b, d_i). Both are strictly concave (report, Lemmas 2-3);
(ii) is solved exactly by damped Newton. Options: deterministic annealing,
restarts, quasi-Newton polish using the exact score (Fisher's identity),
and conditional penalised-curvature uncertainty approximations.
"""
from __future__ import annotations

import time
from functools import cached_property

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from .model import FitHistory, PairData, log_gate, logsumexp, weighted_bt_newton


def bt_loglik(eta, y):
    """y log sigma(eta) + (1 - y) log sigma(-eta), elementwise (eta n x K)."""
    return log_expit(eta) - (1.0 - y)[:, None] * eta


class BattleData:
    def __init__(self, a, b, y, D, M):
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.y = np.asarray(y, float)
        self.n = len(self.y)
        self.M = M
        self.D = np.zeros((self.n, 0)) if D is None else np.asarray(D, float)
        self.q = self.D.shape[1]
        rows = np.repeat(np.arange(self.n), 2)
        cols = np.column_stack([self.a, self.b]).ravel()
        vals = np.tile([1.0, -1.0], self.n)
        inc = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, M))
        self.X = sp.hstack([inc, sp.csr_matrix(self.D)]).tocsr()  # n x (M + q)

    @cached_property
    def pairs(self):
        """Index unordered candidate pairs once, across all EM steps and restarts."""
        return PairData(self.a, self.b, self.y, self.M)


def component_newton(bd: BattleData, w, r, beta, prior, lam, fit_beta=True, offset=None,
                     max_iter=30, tol=1e-8):
    """argmax over (r, beta) of
        sum_i w_i [y_i log s(eta_i) + (1-y_i) log s(-eta_i)]
        - lam_R/2 ||r - r0||^2 - lam_B/2 ||beta - beta0||^2,
    eta_i = r_{a_i} - r_{b_i} + beta' d_i (+ offset_i).
    Strictly concave for lam_R, lam_B > 0 (Lemma 2), so damped Newton
    converges to the unique maximiser, which satisfies 1'r = 0."""
    M, q = bd.M, bd.q
    r0, b0 = prior
    lam_R, lam_B = lam
    qb = q if fit_beta else 0
    off = np.zeros(bd.n) if offset is None else offset
    Xd = bd.X if qb else bd.X[:, :M]

    def objective(v):
        rr, bb = v[:M], v[M:]
        eta = rr[bd.a] - rr[bd.b] + off + (bd.D @ bb if qb else 0.0)
        val = w @ (log_expit(eta) - (1 - bd.y) * eta) - 0.5 * lam_R * np.sum((rr - r0) ** 2)
        if qb:
            val -= 0.5 * lam_B * np.sum((bb - b0) ** 2)
        return float(val), eta

    v = np.concatenate([r, beta if qb else np.zeros(0)])
    J = np.zeros((len(v), len(v)))
    J[:M, :M] = 1.0 / M  # only matters when lam_R = 0
    f, eta = objective(v)
    for _ in range(max_iter):
        p = expit(eta)
        g = Xd.T @ (w * (bd.y - p))
        g[:M] -= lam_R * (v[:M] - r0)
        if qb:
            g[M:] -= lam_B * (v[M:] - b0)
        if np.max(np.abs(g)) < tol:
            break
        H = ((Xd.multiply((w * p * (1 - p))[:, None])).T @ Xd).toarray()
        H[:M, :M] += lam_R * np.eye(M)
        if qb:
            H[M:, M:] += lam_B * np.eye(qb)
        step = np.linalg.solve(H + J, g)
        step[:M] -= step[:M].mean()
        s = 1.0
        while True:
            v_new = v + s * step
            f_new, eta_new = objective(v_new)
            if f_new >= f - 1e-10 or s < 1e-6:
                break
            s *= 0.5
        v, f, eta = v_new, f_new, eta_new
    r_out = v[:M] - v[:M].mean()
    return r_out, (v[M:] if qb else beta)


class StyleMixtureBT:
    """Mixture of Bradley--Terry models with softmax gating and optional
    domain-specific style covariates.

    style  : None | 'domain' | 'shared'
    aggregate_pairs : use exact weighted edge counts in the skill M-step when
             there are no per-battle response covariates. False uses vote rows.
    anneal : None or increasing temperatures in (0, 1) for deterministic
             annealing EM (tempered E-step gamma propto (pi p)^tau).
    """

    def __init__(self, K, lam_W=3.0, lam_R=50.0, lam_B=10.0, style="domain", max_iter=150,
                 tol=1e-7, n_init=2, init="random", anneal=None, polish=True, seed=0,
                 w_maxiter=30, prior_lam=(0.1, 0.01), verbose=False, accel=None, aa_m=5,
                 n_short=20, short_iter=15, polish_all=False, aggregate_pairs=True):
        self.aggregate_pairs = aggregate_pairs
        self.accel, self.aa_m, self.n_short, self.short_iter = accel, aa_m, n_short, short_iter
        self.polish_all = polish_all
        self.K, self.lam_W, self.lam_R, self.lam_B = K, lam_W, lam_R, lam_B
        self.style, self.max_iter, self.tol, self.n_init = style, max_iter, tol, n_init
        self.init, self.anneal, self.polish, self.seed = init, anneal, polish, seed
        self.w_maxiter, self.prior_lam, self.verbose = w_maxiter, prior_lam, verbose

    # ----------------------------------------------------------- pieces
    def _bd(self, a, b, y, D):
        return BattleData(a, b, y, D if self.style is not None else None, self.M)

    def _eta(self, bd, R, B):
        eta = R[bd.a] - R[bd.b]
        if bd.q:
            eta = eta + bd.D @ B.T
        return eta

    def _penalty(self, W, R, B):
        r0, b0 = self.prior_
        pen = 0.5 * self.lam_W * np.sum(W[:, 1:] ** 2) + 0.5 * self.lam_R * np.sum((R - r0[:, None]) ** 2)
        if self.style == "domain" and B.size:
            pen += 0.5 * self.lam_B * np.sum((B - b0[None, :]) ** 2)
        elif self.style == "shared" and B.size:
            pen += 0.5 * self.lam_B * np.sum((B[0] - b0) ** 2)
        return pen

    def _estep(self, F, bd, W, R, B, tau=1.0):
        L = log_gate(F, W) + bt_loglik(self._eta(bd, R, B), bd.y)
        ll_i = logsumexp(L, axis=1)
        if tau == 1.0:
            return np.exp(L - ll_i[:, None]), ll_i, ll_i
        Lt = tau * L
        lse = logsumexp(Lt, axis=1)
        return np.exp(Lt - lse[:, None]), ll_i, lse / tau

    def _mstep_W(self, F, gamma, W0):
        K, p = W0.shape
        if K == 1:
            return W0
        mask = np.ones(p)
        mask[0] = 0.0
        lam = self.lam_W

        def f_and_g(th):
            Wf = np.vstack([th.reshape(K - 1, p), np.zeros((1, p))])
            lp = log_gate(F, Wf)
            val = -np.sum(gamma * lp) + 0.5 * lam * np.sum((Wf[:-1] * mask) ** 2)
            G = -(gamma - np.exp(lp)).T @ F
            return val, (G[:-1] + lam * Wf[:-1] * mask).ravel()

        res = minimize(f_and_g, W0[:-1].ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.w_maxiter, "gtol": 1e-6})
        return np.vstack([res.x.reshape(K - 1, p), np.zeros((1, p))])

    def _cm_W_observed(self, F, bd, W0, R, B):
        """ECME CM-step: maximise the observed-data objective over W."""
        K, p = W0.shape
        mask = np.ones(p)
        mask[0] = 0.0
        cl = bt_loglik(self._eta(bd, R, B), bd.y)

        def f_and_g(th):
            Wf = np.vstack([th.reshape(K - 1, p), np.zeros((1, p))])
            lp = log_gate(F, Wf)
            L = lp + cl
            ll_i = logsumexp(L, axis=1)
            G = (np.exp(L - ll_i[:, None]) - np.exp(lp)).T @ F - self.lam_W * Wf * mask
            return -(ll_i.sum() - 0.5 * self.lam_W * np.sum(Wf[:, 1:] ** 2)), -G[:-1].ravel()

        res = minimize(f_and_g, W0[:-1].ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.w_maxiter, "gtol": 1e-6})
        return np.vstack([res.x.reshape(K - 1, p), np.zeros((1, p))])

    def _mstep_pref(self, bd, gamma, R, B):
        R, B = R.copy(), B.copy()
        if bd.q == 0 and getattr(self, "aggregate_pairs", True):
            # The E-step uses each prompt separately. Its weights can now be
            # summed because all battles on an oriented edge have one skill logit.
            pairs = bd.pairs
            mass, wins = pairs.aggregate(gamma)
            for k in range(self.K):
                R[:, k] = weighted_bt_newton(
                    pairs.u, pairs.v, mass[:, k], wins[:, k], bd.M,
                    self.lam_R, R[:, k], tol=1e-8, max_iter=30, prior=self.prior_[0],
                )
            return R, B
        lam = (self.lam_R, self.lam_B)
        shared = self.style == "shared" and bd.q
        for k in range(self.K):
            off = bd.D @ B[k] if shared else None
            r, bk = component_newton(bd, gamma[:, k], R[:, k], B[k], self.prior_, lam,
                                     fit_beta=(self.style == "domain" and bd.q > 0), offset=off)
            R[:, k] = r
            if self.style == "domain" and bd.q:
                B[k] = bk
        if shared:
            B = np.tile(self._shared_beta(bd, gamma, R, B[0]), (self.K, 1))
        return R, B

    def _shared_beta(self, bd, gamma, R, beta, iters=15):
        b0 = self.prior_[1]
        base = R[bd.a] - R[bd.b]

        def obj(bv):
            eta = base + (bd.D @ bv)[:, None]
            return np.sum(gamma * bt_loglik(eta, bd.y)) - 0.5 * self.lam_B * np.sum((bv - b0) ** 2)

        f = obj(beta)
        for _ in range(iters):
            p = expit(base + (bd.D @ beta)[:, None])
            g = bd.D.T @ np.sum(gamma * (bd.y[:, None] - p), 1) - self.lam_B * (beta - b0)
            if np.max(np.abs(g)) < 1e-8:
                break
            v = np.sum(gamma * p * (1 - p), 1)
            H = (bd.D * v[:, None]).T @ bd.D + self.lam_B * np.eye(bd.q)
            step = np.linalg.solve(H, g)
            s = 1.0
            while True:
                bn = beta + s * step
                fn = obj(bn)
                if fn >= f - 1e-10 or s < 1e-6:
                    break
                s *= 0.5
            beta, f = bn, fn
        return beta

    def _fit_prior(self, bd):
        lr, lb = self.prior_lam
        r, b = component_newton(bd, np.ones(bd.n), np.zeros(bd.M), np.zeros(bd.q),
                                (np.zeros(bd.M), np.zeros(bd.q)), (lr, lb),
                                fit_beta=bd.q > 0, max_iter=50)
        self.prior_ = (r, b if bd.q else np.zeros(0))

    def _init(self, F, bd, rng):
        K, p = self.K, F.shape[1]
        if self.init == "kmeans":
            from sklearn.cluster import KMeans
            lab = KMeans(K, n_init=3, random_state=int(rng.integers(1 << 31))).fit(F[:, 1:]).labels_
            gamma = np.full((bd.n, K), 0.1 / K)
            gamma[np.arange(bd.n), lab] += 0.9
        else:
            Wr = rng.normal(size=(K, p))
            Wr[-1] = 0
            gamma = np.exp(log_gate(F / np.sqrt(p), Wr))
        r0, b0 = self.prior_
        R = np.tile(r0[:, None], (1, K)) + rng.normal(scale=0.05, size=(bd.M, K))
        R -= R.mean(0)
        B = np.tile(b0, (K, 1))
        W = self._mstep_W(F, gamma, np.zeros((K, p)))
        R, B = self._mstep_pref(bd, gamma, R, B)
        return W, R, B

    # -------------------------------------------------------------- fit
    def fit(self, F, a, b, y, D=None, M=None, init_params=None):
        F = np.asarray(F, float)
        self.M = int(max(np.max(a), np.max(b)) + 1) if M is None else M
        bd = self._bd(a, b, y, D)
        self.q = bd.q
        self._fit_prior(bd)
        rng = np.random.default_rng(self.seed)
        best = None
        starts = None
        if self.init == "emEM" and init_params is None:
            cand = []
            for _ in range(self.n_short):
                W, R, B = self._init(F, bd, rng)
                for _ in range(self.short_iter):
                    gamma = self._estep(F, bd, W, R, B)[0]
                    W = self._mstep_W(F, gamma, W)
                    R, B = self._mstep_pref(bd, gamma, R, B)
                g_, ll_i, _ = self._estep(F, bd, W, R, B)
                cand.append((ll_i.sum() - self._penalty(W, R, B), W, R, B))
            cand.sort(key=lambda c: -c[0])
            starts = [c[1:] for c in cand[: self.n_init]]
        for att in range(self.n_init if init_params is None else 1):
            if starts is not None:
                params = starts[att]
            else:
                params = self._init(F, bd, rng) if init_params is None else tuple(np.copy(x) for x in init_params)
            out = self._run(F, bd, *params)
            if self.polish_all and self.K > 1:
                # hybrid per restart: EM global phase, then quasi-Newton on l_pen
                self.W_, self.R_, self.B_, self.history_ = out
                n_em = len(out[-1].objective)
                self._polish(F, bd)
                out = (self.W_, self.R_, self.B_, self.history_)
                out[-1].n_em = n_em
            if self.verbose:
                print(f"  start {att}: obj {out[-1].objective[-1]:.3f}")
            if best is None or out[-1].objective[-1] > best[-1].objective[-1]:
                best = out
        self.W_, self.R_, self.B_, self.history_ = best
        self.n_iter_em_ = getattr(self.history_, "n_em", len(self.history_.objective))
        self.n_iter_polish_ = 0
        if self.polish and self.K > 1 and not self.polish_all:
            self._polish(F, bd)
        self.loglik_ = self.loglik(F, a, b, y, D)
        return self

    def _run(self, F, bd, W, R, B):
        if self.accel == "anderson":
            return self._run_anderson(F, bd, W, R, B)
        hist = FitHistory()
        t0 = time.perf_counter()
        for tau in list(self.anneal or []) + [1.0]:
            gamma, ll_i, fe = self._estep(F, bd, W, R, B, tau)
            obj_old = fe.sum() - self._penalty(W, R, B)
            tol = self.tol if tau == 1.0 else 10 * self.tol
            for _ in range(self.max_iter if self.K > 1 else 1):
                if self.accel == "ecme" and tau == 1.0 and self.K > 1:
                    W = self._cm_W_observed(F, bd, W, R, B)  # ECME gate step
                    gamma = self._estep(F, bd, W, R, B)[0]
                else:
                    W = self._mstep_W(F, gamma, W)
                R, B = self._mstep_pref(bd, gamma, R, B)
                gamma, ll_i, fe = self._estep(F, bd, W, R, B, tau)
                obj = fe.sum() - self._penalty(W, R, B)  # tempered free energy (= objective at tau=1)
                if tau == 1.0:
                    hist.objective.append(obj)
                    hist.loglik.append(ll_i.sum())
                    hist.time.append(time.perf_counter() - t0)
                if abs(obj - obj_old) < tol * (abs(obj_old) + 1):
                    break
                obj_old = obj
        return W, R, B, hist

    def _run_anderson(self, F, bd, W, R, B):
        from .accel import anderson_em
        p = F.shape[1]

        def em_map(t):
            Wt, Rt, Bt = self._unpack(t, p)
            Rt = Rt - Rt.mean(0)
            gamma, ll_i, _ = self._estep(F, bd, Wt, Rt, Bt)
            W2 = self._mstep_W(F, gamma, Wt)
            R2, B2 = self._mstep_pref(bd, gamma, Rt, Bt)
            return self._pack(W2, R2, B2), ll_i.sum() - self._penalty(Wt, Rt, Bt)

        x, f, n_maps, trace = anderson_em(em_map, self._pack(W, R, B), m=self.aa_m, max_iter=self.max_iter,
                                          tol=self.tol)
        W, R, B = self._unpack(x, p)
        hist = FitHistory()
        for t_, f_ in trace:
            hist.time.append(t_)
            hist.objective.append(f_)
            hist.loglik.append(np.nan)
        self.n_maps_ = n_maps
        return W, R - R.mean(0), B, hist

    # ------------------------------------------------------------ score
    def _pack(self, W, R, B):
        parts = [W[:-1].ravel(), R.ravel()]
        if self.style == "domain":
            parts.append(B.ravel())
        elif self.style == "shared":
            parts.append(B[0])
        return np.concatenate(parts)

    def _unpack(self, th, p):
        K, M, q = self.K, self.M, self.q
        i = (K - 1) * p
        W = np.vstack([th[:i].reshape(K - 1, p), np.zeros((1, p))])
        R = th[i:i + M * K].reshape(M, K)
        i += M * K
        if self.style == "domain":
            B = th[i:i + K * q].reshape(K, q)
        elif self.style == "shared":
            B = np.tile(th[i:i + q], (K, 1))
        else:
            B = np.zeros((K, q))
        return W, R, B

    def score(self, F, bd, W, R, B):
        """Exact gradient of the penalised observed-data log-likelihood via
        Fisher's identity (report, Lemma 1), and its value."""
        K, M = self.K, self.M
        lp = log_gate(F, W)
        eta = self._eta(bd, R, B)
        L = lp + bt_loglik(eta, bd.y)
        ll_i = logsumexp(L, axis=1)
        gamma = np.exp(L - ll_i[:, None])
        mask = np.ones(F.shape[1])
        mask[0] = 0
        gW = (gamma - np.exp(lp)).T @ F - self.lam_W * W * mask
        ge = gamma * (bd.y[:, None] - expit(eta))
        r0, b0 = self.prior_
        gR = np.zeros((M, K))
        np.add.at(gR, bd.a, ge)
        np.add.at(gR, bd.b, -ge)
        gR -= self.lam_R * (R - r0[:, None])
        parts = [gW[:-1].ravel(), gR.ravel()]
        if self.style == "domain":
            parts.append((ge.T @ bd.D - self.lam_B * (B - b0[None, :])).ravel())
        elif self.style == "shared":
            parts.append(ge.sum(1) @ bd.D - self.lam_B * (B[0] - b0))
        return np.concatenate(parts), ll_i.sum() - self._penalty(W, R, B)

    def _polish(self, F, bd, maxiter=3000):
        """L-BFGS-B refinement with gradient and relative-objective tolerances.

        The optimiser can terminate on either tolerance or its iteration
        limit; gtol is a target, not a postcondition on the returned fit.
        """
        p = F.shape[1]
        hist = self.history_
        t0 = time.perf_counter()
        last = hist.time[-1] if hist.time else 0.0

        def f_and_g(th):
            g, val = self.score(F, bd, *self._unpack(th, p))
            return -val, -g

        def cb(th):
            hist.objective.append(self.score(F, bd, *self._unpack(th, p))[1])
            hist.time.append(last + time.perf_counter() - t0)

        res = minimize(f_and_g, self._pack(self.W_, self.R_, self.B_), jac=True, method="L-BFGS-B",
                       callback=cb, options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 20})
        W, R, B = self._unpack(res.x, p)
        self.W_, self.R_, self.B_ = W, R - R.mean(0), B
        self.n_iter_polish_ = res.nit

    # -------------------------------------------------------- inference
    def loglik(self, F, a, b, y, D=None):
        return float(self._estep(np.asarray(F, float), self._bd(a, b, y, D), self.W_, self.R_, self.B_)[1].sum())

    def objective(self, F, a, b, y, D=None):
        return self.loglik(F, a, b, y, D) - self._penalty(self.W_, self.R_, self.B_)

    def posterior(self, F, a, b, y, D=None):
        return self._estep(np.asarray(F, float), self._bd(a, b, y, D), self.W_, self.R_, self.B_)[0]

    def observed_information(self, F, a, b, y, D=None, h=1e-5):
        """Conditional inverse-curvature approximation for all parameters.

        Central differences of the exact penalised score give the Hessian;
        sum-to-zero constraints are handled in an orthonormal basis. Its
        negative inverse is stored in ``cov_`` and square roots of diagonal
        entries are returned as (seW, seR, seB) for API compatibility.
        These are local Laplace quantities conditional on K, penalties,
        features and the fitted global shrinkage centre. They are not, by
        themselves, frequentist standard errors for the full fitting
        procedure or for contrasts against that estimated global centre.
        A full-procedure bootstrap must refit that centre and align domains.
        """
        F = np.asarray(F, float)
        bd = self._bd(a, b, y, D)
        p, K, M = F.shape[1], self.K, self.M
        th0 = self._pack(self.W_, self.R_, self.B_)
        nW = (K - 1) * p
        Q = np.linalg.qr(np.eye(M) - 1.0 / M)[0][:, :M - 1]
        dfull = len(th0)
        T = np.zeros((dfull, dfull - K))
        T[:nW, :nW] = np.eye(nW)
        for k in range(K):
            T[nW + np.arange(M) * K + k, nW + k * (M - 1): nW + (k + 1) * (M - 1)] = Q
        rest = dfull - nW - M * K
        if rest:
            T[nW + M * K:, nW + K * (M - 1):] = np.eye(rest)
        grad = lambda th: self.score(F, bd, *self._unpack(th, p))[0]
        H = np.empty((T.shape[1], T.shape[1]))
        for j in range(T.shape[1]):
            e = h * T[:, j]
            H[:, j] = T.T @ (grad(th0 + e) - grad(th0 - e)) / (2 * h)
        H = 0.5 * (H + H.T)
        cov = T @ np.linalg.inv(-H) @ T.T
        self.cov_ = cov
        return self._unpack(np.sqrt(np.clip(np.diag(cov), 0, None)), p)  # (seW, seR, seB)

    def gate(self, F):
        return np.exp(log_gate(np.asarray(F, float), self.W_))

    def predict_proba(self, F, a, b, D=None):
        """P(a beats b | x, d) = sum_k pi_k(x) sigma(eta_k). With D=None the
        style difference is set to 0. For a model fitted with response
        covariates, this is a neutral-style conditional prediction, not a
        marginal probability before responses exist. Deployment routing
        uses a model fitted with style=None.
        """
        a = np.asarray(a)
        bd = BattleData(a, b, np.zeros(len(a)), D if (self.style is not None and D is not None) else None, self.M)
        eta = self.R_[bd.a] - self.R_[bd.b]
        if bd.q:
            eta = eta + bd.D @ self.B_.T
        return np.sum(self.gate(F) * expit(eta), 1)

    def domain_strength(self, candidates=None):
        """v_{m,k} = mean_{j in C, j != m} sigma(r_mk - r_jk) at neutral style."""
        C = np.arange(self.M) if candidates is None else np.asarray(candidates)
        Rc = self.R_[C]
        P = expit(Rc[:, None, :] - Rc[None, :, :])
        idx = np.arange(len(C))
        P[idx, idx, :] = 0.0
        return P.sum(1) / (len(C) - 1)

    def route(self, F, candidates=None):
        """Select by expected win rate at neutral response covariates.

        This is a deployable prompt-only router when fitted with style=None;
        otherwise the scores describe a conditional, neutral-style contrast.
        """
        C = np.arange(self.M) if candidates is None else np.asarray(candidates)
        S = self.gate(F) @ self.domain_strength(C).T
        return C[np.argmax(S, axis=1)], S


class ClusterStyleBT:
    """Two-stage baseline: k-means clusters of the prompt embedding, then one
    (style-adjusted) BT model per cluster, shrunk to the global fit."""

    def __init__(self, K, lam_R=50.0, lam_B=10.0, style="domain", seed=0):
        self.K, self.style, self.seed, self.lam = K, style, seed, (lam_R, lam_B)

    def fit(self, F, a, b, y, D=None, M=None):
        from sklearn.cluster import KMeans
        self.M = int(max(np.max(a), np.max(b)) + 1) if M is None else M
        bd = BattleData(a, b, y, D if self.style is not None else None, self.M)
        g = StyleMixtureBT(1, style=self.style)
        g.M = self.M
        g._fit_prior(bd)
        self.prior_ = g.prior_
        self.km_ = KMeans(self.K, n_init=5, random_state=self.seed).fit(F[:, 1:])
        lab = self.km_.labels_
        self.R_ = np.zeros((self.M, self.K))
        self.B_ = np.zeros((self.K, bd.q))
        for k in range(self.K):
            r, bk = component_newton(bd, (lab == k).astype(float), self.prior_[0].copy(),
                                     self.prior_[1].copy(), self.prior_, self.lam, fit_beta=bd.q > 0)
            self.R_[:, k] = r
            if bd.q:
                self.B_[k] = bk
        return self

    def predict_proba(self, F, a, b, D=None):
        lab = self.km_.predict(F[:, 1:])
        eta = self.R_[a, lab] - self.R_[b, lab]
        if D is not None and self.style is not None:
            eta = eta + np.sum(D * self.B_[lab], 1)
        return expit(eta)
