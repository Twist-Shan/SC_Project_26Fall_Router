"""
Latent-domain mixture of Bradley--Terry models with a softmax gate.

Model (formulation A in the proposal)
-------------------------------------
For battle i with prompt features f_i (p-vector, first entry = 1 for the
intercept), models (a_i, b_i) and outcome y_i in [0, 1],

    z_i | x_i            ~ Categorical(pi(x_i; W)),  pi_k = softmax(W f_i)_k,
    y_i | z_i = k, a, b  ~ Bernoulli(sigma(r_{a,k} - r_{b,k})).

W is K x p with the last row fixed at 0 (Assumption I1); R is M x K with
centred columns. A fractional y_i (e.g. 0.5 for a tie) is allowed: the
component log-likelihood is y log sigma(d) + (1 - y) log sigma(-d).

Estimation is by (ridge-penalised) EM:

    E-step  gamma_{ik} = posterior P(z_i = k | data)
    M-step  W <- weighted multinomial logistic regression (L-BFGS, warm start)
            R <- K independent weighted Bradley--Terry fits (Newton)

so each iteration increases the penalised observed-data log-likelihood
    l(W, R) - lambda_W/2 ||W_{-int}||^2 - lambda_R/2 ||R||^2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, log_expit, softmax

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def logsumexp(X, axis=1, keepdims=False):
    """Fast row-wise log-sum-exp for an n x K array with small K.

    numpy reductions along a short trailing axis are slow, so the max is
    taken column by column and the sum is done with a mat-vec product."""
    assert axis == 1 and X.ndim == 2
    m = X[:, 0].copy()
    for k in range(1, X.shape[1]):
        np.maximum(m, X[:, k], out=m)
    out = m + np.log(np.exp(X - m[:, None]) @ np.ones(X.shape[1]))
    return out[:, None] if keepdims else out


def log_gate(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    """log pi(x_i; W) for every row of F. Returns n x K."""
    eta = F @ W.T
    return eta - logsumexp(eta, axis=1, keepdims=True)


def component_loglik(R, a, b, y):
    """n x K matrix of log P(y_i | z_i = k)."""
    d = R[a] - R[b]
    ld = log_expit(d)
    # log sigma(-d) = log sigma(d) - d
    return ld - (1.0 - y)[:, None] * d


class PairData:
    """Pre-computed sparse structures for the Bradley--Terry M-step.

    Battles are collapsed onto unordered edges {u, v} (u < v). For each
    component the weighted win count of u over v and the weighted number of
    battles are sufficient statistics for the R-update.
    """

    def __init__(self, a, b, y, M):
        # Form u * M + v in the platform index dtype even for compact input IDs.
        a = np.asarray(a, dtype=np.intp)
        b = np.asarray(b, dtype=np.intp)
        y = np.asarray(y, dtype=float)
        u = np.minimum(a, b)
        v = np.maximum(a, b)
        y_u = np.where(a == u, y, 1.0 - y)  # outcome oriented as "u beats v"
        key = u * M + v
        uniq, edge_id = np.unique(key, return_inverse=True)
        self.M = M
        self.u = uniq // M
        self.v = uniq % M
        self.L = len(uniq)
        self.n = len(a)
        # n x L incidence matrix (battle -> edge)
        self.E = sp.csr_matrix(
            (np.ones(self.n), (np.arange(self.n), edge_id)), shape=(self.n, self.L)
        )
        self.y_u = y_u
        self.edge_id = edge_id

    def aggregate(self, gamma):
        """Weighted battle counts N (L x K) and wins of u (L x K)."""
        N = np.asarray(self.E.T @ gamma)
        Wn = np.asarray(self.E.T @ (gamma * self.y_u[:, None]))
        return N, Wn


def weighted_bt_newton(u, v, N, Wn, M, lam=0.0, r0=None, tol=1e-9, max_iter=100, prior=None):
    """Maximise sum_e [Wn_e log s(r_u - r_v) + (N_e - Wn_e) log s(r_v - r_u)]
    - lam/2 ||r - prior||^2 over r in R^M with 1'r = 0, by damped Newton.
    `prior` (centred, default 0) is the point the ridge penalty shrinks to."""
    r = np.zeros(M) if r0 is None else r0 - r0.mean()
    pr = np.zeros(M) if prior is None else prior - prior.mean()

    def obj(r):
        d = r[u] - r[v]
        return np.sum(Wn * log_expit(d) + (N - Wn) * log_expit(-d)) - 0.5 * lam * (r - pr) @ (r - pr)

    J = np.ones((M, M)) / M
    f = obj(r)
    for _ in range(max_iter):
        d = r[u] - r[v]
        s = expit(d)
        res = Wn - N * s
        g = np.bincount(u, res, M) - np.bincount(v, res, M) - lam * (r - pr)
        if np.max(np.abs(g)) < tol:
            break
        h = N * s * (1 - s)
        H = np.zeros((M, M))
        np.add.at(H, (u, u), h)
        np.add.at(H, (v, v), h)
        np.add.at(H, (u, v), -h)
        np.add.at(H, (v, u), -h)
        H += lam * np.eye(M)
        # H is the negative Hessian (PSD). Add J to fix the null direction 1.
        step = np.linalg.solve(H + J, g)
        step -= step.mean()
        t = 1.0
        while True:
            r_new = r + t * step
            f_new = obj(r_new)
            if f_new >= f - 1e-12 or t < 1e-8:
                break
            t *= 0.5
        r, f = r_new, f_new
    return r - r.mean()


# ----------------------------------------------------------------------------
# Main estimator
# ----------------------------------------------------------------------------


@dataclass
class FitHistory:
    objective: list = field(default_factory=list)  # penalised log-lik
    loglik: list = field(default_factory=list)  # observed-data log-lik
    time: list = field(default_factory=list)


class MixtureBT:
    """Latent-domain mixture of Bradley--Terry models with softmax gating.

    Parameters
    ----------
    K : number of latent domains.
    lam_W, lam_R : ridge penalties for the gate (non-intercept columns) and
        for the skill matrix.
    init : 'kmeans' | 'random' | 'given'.
    n_init : number of random restarts (best penalised objective is kept).
    accel : None (plain EM) or 'squarem' (SQUAREM-accelerated EM).
    anneal : None or increasing temperatures tau_1 < ... < 1 for deterministic
        annealing EM (Ueda & Nakano, 1998): before ordinary EM, run EM with
        the tempered E-step gamma_ik propto (pi_ik p_ik)^tau at each tau.
    polish : if True, the EM solution is refined by direct L-BFGS on the
        observed-data objective (EM for the global phase, quasi-Newton for
        the fast local phase).
    shrink_to : 'zero' (ridge on R) or 'global' (ridge on R - r_global 1',
        i.e. domain skills are shrunk towards the global BT score, so a model
        with little data in a domain falls back to its overall strength).
    """

    def __init__(
        self,
        K,
        lam_W=1e-2,
        lam_R=1e-2,
        max_iter=500,
        tol=1e-7,
        init="kmeans",
        n_init=1,
        seed=0,
        verbose=False,
        w_maxiter=50,
        accel=None,
        shrink_to="zero",
        polish=False,
        anneal=None,
        anneal_jitter=0.1,
        n_short=20,
        short_iter=15,
        aa_m=5,
    ):
        self.n_short, self.short_iter, self.aa_m = n_short, short_iter, aa_m
        self.anneal = anneal
        self.anneal_jitter = anneal_jitter
        self.polish = polish
        self.accel = accel
        self.shrink_to = shrink_to
        self.R_prior_ = None
        self.K = K
        self.lam_W = lam_W
        self.lam_R = lam_R
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.seed = seed
        self.verbose = verbose
        self.w_maxiter = w_maxiter

    # ------------------------------------------------------------------ utils
    def _penalty(self, W, R):
        return 0.5 * self.lam_W * np.sum(W[:, 1:] ** 2) + 0.5 * self.lam_R * np.sum(
            (R - self._prior(R.shape[0])[:, None]) ** 2)

    def _prior(self, M):
        return np.zeros(M) if self.R_prior_ is None else self.R_prior_

    def _set_prior(self, a, b, y, M):
        if self.shrink_to == "global":
            pd = PairData(a, b, y, M)
            N, Wn = pd.aggregate(np.ones((len(a), 1)))
            self.R_prior_ = weighted_bt_newton(pd.u, pd.v, N[:, 0], Wn[:, 0], M, lam=0.1)
        else:
            self.R_prior_ = np.zeros(M)

    def _estep(self, F, a, b, y, W, R):
        L = log_gate(F, W) + component_loglik(R, a, b, y)
        ll_i = logsumexp(L, axis=1)
        gamma = np.exp(L - ll_i[:, None])
        return gamma, ll_i

    def loglik(self, F, a, b, y, W=None, R=None):
        W = self.W_ if W is None else W
        R = self.R_ if R is None else R
        return float(self._estep(F, a, b, y, W, R)[1].sum())

    # ---------------------------------------------------------------- M-steps
    def _mstep_W(self, F, gamma, W0):
        K, p = W0.shape
        if K == 1:
            return W0
        lam = self.lam_W
        mask = np.ones(p)
        mask[0] = 0.0  # do not penalise the intercept

        def f_and_g(theta):
            Wf = np.vstack([theta.reshape(K - 1, p), np.zeros((1, p))])
            lp = log_gate(F, Wf)
            val = -np.sum(gamma * lp) + 0.5 * lam * np.sum((Wf[:-1] * mask) ** 2)
            G = -(gamma - np.exp(lp)).T @ F  # K x p
            G = G[:-1] + lam * Wf[:-1] * mask
            return val, G.ravel()

        res = minimize(
            f_and_g,
            W0[:-1].ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.w_maxiter, "gtol": 1e-6},
        )
        return np.vstack([res.x.reshape(K - 1, p), np.zeros((1, p))])

    def _mstep_R(self, pd: PairData, gamma, R0):
        N, Wn = pd.aggregate(gamma)
        R = np.empty_like(R0)
        for k in range(self.K):
            R[:, k] = weighted_bt_newton(
                pd.u, pd.v, N[:, k], Wn[:, k], pd.M, lam=self.lam_R, r0=R0[:, k],
                prior=self._prior(pd.M),
            )
        return R

    # ----------------------------------------------------------------- inits
    def _initial_params(self, F, a, b, y, M, pd, rng, attempt):
        K, p = self.K, F.shape[1]
        if self.init == "given":
            return self.W_init.copy(), self.R_init.copy()
        if self.init == "kmeans":
            from sklearn.cluster import KMeans

            Z = F[:, 1:]
            km = KMeans(K, n_init=3, random_state=int(rng.integers(1 << 31))).fit(Z)
            gamma = np.full((len(a), K), 0.1 / K)
            gamma[np.arange(len(a)), km.labels_] += 0.9
        elif self.init in ("random", "emEM"):
            # random soft assignment driven by a random gate, so that the
            # initial responsibilities depend on x (avoids symmetric start)
            Wr = rng.normal(scale=1.0, size=(K, p))
            Wr[-1] = 0
            gamma = np.exp(log_gate(F / np.sqrt(p), Wr))
        else:
            raise ValueError(self.init)
        W = np.zeros((K, p))
        R = np.zeros((M, K))
        W = self._mstep_W(F, gamma, W)
        R = self._mstep_R(pd, gamma, R)
        # small jitter on R so that components are not identical when the
        # initial responsibilities are nearly uninformative
        R += rng.normal(scale=0.05, size=R.shape)
        R -= R.mean(axis=0)
        return W, R

    # ------------------------------------------------------------------- fit
    def fit(self, F, a, b, y, M=None, W_init=None, R_init=None):
        F = np.asarray(F, dtype=float)
        a = np.asarray(a)
        b = np.asarray(b)
        y = np.asarray(y, dtype=float)
        M = int(max(a.max(), b.max()) + 1) if M is None else M
        self.M = M
        if W_init is not None:
            self.init = "given"
            self.W_init, self.R_init = W_init, R_init
        pd = PairData(a, b, y, M)
        self._set_prior(a, b, y, M)
        rng = np.random.default_rng(self.seed)
        best = None
        starts = None
        if self.init == "emEM":
            # short-run EM screening (Biernacki et al., 2003): many cheap
            # starts, a few EM iterations each, keep the best n_init.
            cand = []
            for j in range(self.n_short):
                W, R = self._initial_params(F, a, b, y, M, pd, rng, j)
                for _ in range(self.short_iter):
                    gamma, _ = self._estep(F, a, b, y, W, R)
                    W = self._mstep_W(F, gamma, W)
                    R = self._mstep_R(pd, gamma, R)
                obj = self.loglik(F, a, b, y, W, R) - self._penalty(W, R)
                cand.append((obj, W, R))
            cand.sort(key=lambda c: -c[0])
            starts = [(c[1], c[2]) for c in cand[: self.n_init]]
        for attempt in range(self.n_init):
            if starts is not None:
                W, R = starts[attempt]
            else:
                W, R = self._initial_params(F, a, b, y, M, pd, rng, attempt)
            out = self._run_em(F, a, b, y, pd, W, R)
            if self.verbose:
                print(f"  restart {attempt}: obj={out[2].objective[-1]:.3f} iters={len(out[2].objective)}")
            if best is None or out[2].objective[-1] > best[2].objective[-1]:
                best = out
        self.W_, self.R_, self.history_, self.gamma_ = best
        self.n_iter_ = len(self.history_.objective)
        if self.polish:
            hist_em, it_em = self.history_, self.n_iter_
            self.fit_direct(F, a, b, y, self.W_, self.R_, tol=1e-13)
            self.history_em_, self.n_iter_em_ = hist_em, it_em
            self.gamma_ = self.posterior(F, a, b, y)
        self.loglik_ = self.history_.loglik[-1]
        return self

    def _anneal(self, F, a, b, y, pd, W, R, max_iter=300):
        """Deterministic annealing stages. At temperature tau each iteration
        increases the tempered free energy
            F_tau = (1/tau) sum_i log sum_k (pi_ik p_ik)^tau - penalty."""
        rng = np.random.default_rng(self.seed + 7919)
        for tau in list(self.anneal) + [1.0]:
            # symmetry breaking: at low temperature the merged solution (all
            # components equal) is the stable fixed point; when tau increases
            # it can become a saddle that deterministic EM never leaves.
            R = R + rng.normal(scale=self.anneal_jitter, size=R.shape)
            R -= R.mean(0)
            W = W.copy()
            W[:-1] += rng.normal(scale=self.anneal_jitter, size=W[:-1].shape)
            if tau >= 1.0:
                break
            f_old = None
            for _ in range(max_iter):
                L = tau * (log_gate(F, W) + component_loglik(R, a, b, y))
                lse = logsumexp(L, axis=1)
                f = lse.sum() / tau - self._penalty(W, R)
                if f_old is not None and abs(f - f_old) < 10 * self.tol * (abs(f_old) + 1):
                    break
                f_old = f
                gamma = np.exp(L - lse[:, None])
                W = self._mstep_W(F, gamma, W)
                R = self._mstep_R(pd, gamma, R)
        return W, R

    def _run_em(self, F, a, b, y, pd, W, R):
        if self.anneal:
            W, R = self._anneal(F, a, b, y, pd, W, R)
        self._abY = (a, b, y)
        if self.accel in ("squarem", "ecme-squarem"):
            return self._run_squarem(F, a, b, y, pd, W, R)
        if self.accel == "anderson":
            return self._run_anderson(F, a, b, y, pd, W, R)
        hist = FitHistory()
        t0 = time.perf_counter()
        gamma, ll_i = self._estep(F, a, b, y, W, R)
        obj_old = ll_i.sum() - self._penalty(W, R)
        hist.objective.append(obj_old)
        hist.loglik.append(ll_i.sum())
        hist.time.append(0.0)
        ecme = self.accel in ("ecme", "ecme-squarem")
        for it in range(self.max_iter):
            if ecme:
                # ECME (Liu & Rubin, 1994): the gate is updated by maximising the
                # observed-data objective itself, not Q; the skills by the exact
                # Q-maximisation. Both CM-steps increase l_pen.
                W = self._cm_W_observed(F, a, b, y, W, R)
                gamma, _ = self._estep(F, a, b, y, W, R)
            else:
                W = self._mstep_W(F, gamma, W)
            R = self._mstep_R(pd, gamma, R)
            gamma, ll_i = self._estep(F, a, b, y, W, R)
            obj = ll_i.sum() - self._penalty(W, R)
            hist.objective.append(obj)
            hist.loglik.append(ll_i.sum())
            hist.time.append(time.perf_counter() - t0)
            if self.verbose and it % 20 == 0:
                print(f"    iter {it:4d}  obj {obj:.4f}")
            if abs(obj - obj_old) < self.tol * (abs(obj_old) + 1.0):
                break
            obj_old = obj
        self.n_mstep_ = len(hist.objective) - 1
        return W, R, hist, gamma

    def _cm_W_observed(self, F, a, b, y, W0, R):
        """CM-step for the gate on the observed-data objective (ECME):
        maximise l(W, R) - pen over W with R fixed, by warm-started L-BFGS
        using the exact gradient (Fisher's identity). The objective is not
        concave in W, but any increase keeps ECME monotone."""
        K, p = W0.shape
        if K == 1:
            return W0
        mask = np.ones(p)
        mask[0] = 0.0
        cl = component_loglik(R, a, b, y)

        def f_and_g(th):
            Wf = np.vstack([th.reshape(K - 1, p), np.zeros((1, p))])
            lp = log_gate(F, Wf)
            L = lp + cl
            ll_i = logsumexp(L, axis=1)
            gamma = np.exp(L - ll_i[:, None])
            G = (gamma - np.exp(lp)).T @ F - self.lam_W * Wf * mask
            val = ll_i.sum() - 0.5 * self.lam_W * np.sum((Wf[:, 1:]) ** 2)
            return -val, -G[:-1].ravel()

        res = minimize(f_and_g, W0[:-1].ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.w_maxiter, "gtol": 1e-6})
        return np.vstack([res.x.reshape(K - 1, p), np.zeros((1, p))])

    # ----------------------------------------------------------- Anderson
    def _run_anderson(self, F, a, b, y, pd, W, R):
        from .accel import anderson_em
        K, p = W.shape
        M = R.shape[0]
        nW = (K - 1) * p

        def unvec(t):
            return np.vstack([t[:nW].reshape(K - 1, p), np.zeros((1, p))]), t[nW:].reshape(M, K)

        state = {}

        def em_map(t):
            Wt, Rt = unvec(t)
            Rt = Rt - Rt.mean(0)
            gamma, ll_i = self._estep(F, a, b, y, Wt, Rt)
            W2 = self._mstep_W(F, gamma, Wt)
            R2 = self._mstep_R(pd, gamma, Rt)
            return np.concatenate([W2[:-1].ravel(), R2.ravel()]), ll_i.sum() - self._penalty(Wt, Rt)

        x0 = np.concatenate([W[:-1].ravel(), R.ravel()])
        x, f, n_maps, trace = anderson_em(em_map, x0, m=self.aa_m, max_iter=self.max_iter, tol=self.tol)
        W, R = unvec(x)
        R = R - R.mean(0)
        hist = FitHistory()
        for t_, f_ in trace:
            hist.time.append(t_)
            hist.objective.append(f_)
            hist.loglik.append(np.nan)
        hist.loglik[-1] = self.loglik(F, a, b, y, W, R)
        gamma, _ = self._estep(F, a, b, y, W, R)
        self.n_mstep_ = n_maps
        return W, R, hist, gamma

    # ------------------------------------------------------------ SQUAREM
    def _em_map(self, F, pd, gamma, W, R):
        if self.accel == "ecme-squarem":
            a, b, y = self._abY
            W2 = self._cm_W_observed(F, a, b, y, W, R)
            g2 = self._estep(F, a, b, y, W2, R)[0]
            return W2, self._mstep_R(pd, g2, R)
        return self._mstep_W(F, gamma, W), self._mstep_R(pd, gamma, R)

    def _run_squarem(self, F, a, b, y, pd, W, R):
        """SQUAREM (Varadhan & Roland, 2008; scheme S3) around the EM map,
        with a monotonicity safeguard: if the extrapolated point does not
        improve the penalised objective, the step length is halved towards
        the plain double-EM step."""
        K, p = W.shape
        M = R.shape[0]

        def vec(W, R):
            return np.concatenate([W[:-1].ravel(), R.ravel()])

        def unvec(t):
            W = np.vstack([t[: (K - 1) * p].reshape(K - 1, p), np.zeros((1, p))])
            return W, t[(K - 1) * p:].reshape(M, K)

        def obj_of(W, R):
            g, ll = self._estep(F, a, b, y, W, R)
            return ll.sum() - self._penalty(W, R), ll.sum(), g

        hist = FitHistory()
        t0 = time.perf_counter()
        obj0, ll0, gamma = obj_of(W, R)
        hist.objective.append(obj0)
        hist.loglik.append(ll0)
        hist.time.append(0.0)
        n_map = 0
        for it in range(self.max_iter):
            th0 = vec(W, R)
            W1, R1 = self._em_map(F, pd, gamma, W, R)
            g1 = self._estep(F, a, b, y, W1, R1)[0]
            W2, R2 = self._em_map(F, pd, g1, W1, R1)
            n_map += 2
            th1, th2 = vec(W1, R1), vec(W2, R2)
            r = th1 - th0
            v = th2 - th1 - r
            nv = np.linalg.norm(v)
            alpha = -np.linalg.norm(r) / nv if nv > 0 else -1.0
            alpha = min(alpha, -1.0)
            obj2, ll2, gam2 = obj_of(W2, R2)
            accepted = False
            for _try in range(6):
                if alpha > -1.0 - 1e-8:  # alpha = -1 reproduces theta2
                    break
                Wn, Rn = unvec(th0 - 2 * alpha * r + alpha**2 * v)
                Rn = Rn - Rn.mean(0)
                gn = self._estep(F, a, b, y, Wn, Rn)[0]
                Wn, Rn = self._em_map(F, pd, gn, Wn, Rn)  # stabilising EM step
                n_map += 1
                objn, lln, gamn = obj_of(Wn, Rn)
                if np.isfinite(objn) and objn >= obj2:
                    accepted = True
                    break
                alpha = (alpha - 1.0) / 2.0
            if not accepted:
                Wn, Rn, objn, lln, gamn = W2, R2, obj2, ll2, gam2
            W, R, gamma = Wn, Rn, gamn
            hist.objective.append(objn)
            hist.loglik.append(lln)
            hist.time.append(time.perf_counter() - t0)
            if abs(objn - obj0) < self.tol * (abs(obj0) + 1.0):
                break
            obj0 = objn
        self.n_mstep_ = n_map
        return W, R, hist, gamma

    # ------------------------------------------------- direct optimisation
    def score(self, F, a, b, y, W, R, pd=None):
        """Gradient of the penalised observed-data log-likelihood, via
        Fisher's identity: d l / d theta = sum_i sum_k gamma_ik d/dtheta
        [log pi_ik + log p_ik]. Returns (gW (K x p), gR (M x K), value)."""
        M, K = R.shape
        lp = log_gate(F, W)
        cl = component_loglik(R, a, b, y)
        L = lp + cl
        ll_i = logsumexp(L, axis=1)
        gamma = np.exp(L - ll_i[:, None])
        mask = np.ones(F.shape[1])
        mask[0] = 0
        gW = (gamma - np.exp(lp)).T @ F - self.lam_W * W * mask
        gW[-1] = 0
        d = R[a] - R[b]
        res = gamma * (y[:, None] - expit(d))  # n x K
        gR = np.zeros((M, K))
        np.add.at(gR, a, res)
        np.add.at(gR, b, -res)
        gR -= self.lam_R * (R - self._prior(M)[:, None])
        return gW, gR, ll_i.sum() - self._penalty(W, R)

    def fit_direct(self, F, a, b, y, W0, R0, maxiter=5000, tol=1e-10):
        """Maximise the penalised observed-data likelihood directly with
        L-BFGS (used to benchmark EM)."""
        K, p = W0.shape
        M = R0.shape[0]
        self._set_prior(a, b, y, M)
        hist = FitHistory()
        t0 = time.perf_counter()

        def unpack(theta):
            W = np.vstack([theta[: (K - 1) * p].reshape(K - 1, p), np.zeros((1, p))])
            R = theta[(K - 1) * p :].reshape(M, K)
            return W, R

        def f_and_g(theta):
            W, R = unpack(theta)
            gW, gR, val = self.score(F, a, b, y, W, R)
            return -val, -np.concatenate([gW[:-1].ravel(), gR.ravel()])

        def cb(theta):
            W, R = unpack(theta)
            val = self.loglik(F, a, b, y, W, R)
            hist.loglik.append(val)
            hist.objective.append(val - self._penalty(W, R))
            hist.time.append(time.perf_counter() - t0)

        theta0 = np.concatenate([W0[:-1].ravel(), R0.ravel()])
        cb(theta0)
        res = minimize(
            f_and_g, theta0, jac=True, method="L-BFGS-B", callback=cb,
            options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-8, "maxcor": 20},
        )
        W, R = unpack(res.x)
        R -= R.mean(axis=0)
        self.W_, self.R_, self.history_ = W, R, hist
        self.M = M
        self.n_iter_ = res.nit
        return self

    # ------------------------------------------ conditional curvature
    def observed_information(self, F, a, b, y, h=1e-5):
        """Conditional curvature approximation for (W, R).

        The exact score is available through Fisher's identity (method
        `score`), so the Hessian of the penalised observed-data log-lik is
        obtained by central differences of the score. Its negative inverse
        is a local Laplace covariance approximation, conditional on K,
        penalty strengths, feature construction and the fitted shrinkage
        centre. It is not automatically the frequentist sampling covariance
        of the full estimation procedure. Such inference needs additional
        regularity/asymptotic justification or a bootstrap that refits the
        shrinkage centre and aligns component labels.

        The sum-to-zero constraint on each column of R is handled in an
        orthonormal basis Q of {r : 1'r = 0}. The historical method name and
        return signature are retained: (se_W, se_R, cov), where the `se`
        values are square roots of the curvature covariance diagonal and
        cov is mapped back to theta = (vec W[:-1], vec R). In particular,
        these values alone do not give the uncertainty of R - fitted_prior.
        """
        F = np.asarray(F, float)
        y = np.asarray(y, float)
        W, R = self.W_, self.R_
        K, p = W.shape
        M = R.shape[0]
        # basis for centred vectors: Q is M x (M-1)
        Q = np.linalg.qr(np.eye(M) - 1.0 / M)[0][:, : M - 1]
        nW = (K - 1) * p
        # T maps reduced coords (vec W[:-1], s_1..s_K) to full coords
        T = np.zeros((nW + M * K, nW + (M - 1) * K))
        T[:nW, :nW] = np.eye(nW)
        for k in range(K):
            # R.ravel() is row-major: entry (m, k) sits at m*K + k
            rows = nW + np.arange(M) * K + k
            T[rows, nW + k * (M - 1): nW + (k + 1) * (M - 1)] = Q

        def full_grad(theta):
            Wt = np.vstack([theta[:nW].reshape(K - 1, p), np.zeros((1, p))])
            Rt = theta[nW:].reshape(M, K)
            gW, gR, _ = self.score(F, a, b, y, Wt, Rt)
            return np.concatenate([gW[:-1].ravel(), gR.ravel()])

        theta0 = np.concatenate([W[:-1].ravel(), R.ravel()])
        dim = T.shape[1]
        H = np.empty((dim, dim))
        for j in range(dim):
            step = h * T[:, j]
            H[:, j] = T.T @ (full_grad(theta0 + step) - full_grad(theta0 - step)) / (2 * h)
        H = 0.5 * (H + H.T)
        cov_red = np.linalg.inv(-H)
        cov = T @ cov_red @ T.T
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        se_W = np.vstack([se[:nW].reshape(K - 1, p), np.zeros((1, p))])
        se_R = se[nW:].reshape(M, K)
        self.cov_ = cov
        return se_W, se_R, cov

    # ------------------------------------------------------------- predict
    def gate(self, F):
        return np.exp(log_gate(np.asarray(F, float), self.W_))

    def predict_proba(self, F, a, b):
        """P(a beats b | x) under the fitted mixture."""
        pi = self.gate(F)
        return np.sum(pi * expit(self.R_[a] - self.R_[b]), axis=1)

    def posterior(self, F, a, b, y):
        return self._estep(np.asarray(F, float), a, b, np.asarray(y, float), self.W_, self.R_)[0]

    def domain_strength(self, candidates=None):
        """v_{m,k}: average win prob. of m against the other candidates in
        domain k (proposal, Sec. 4.3). Returns |C| x K."""
        C = np.arange(self.M) if candidates is None else np.asarray(candidates)
        Rc = self.R_[C]  # |C| x K
        P = expit(Rc[:, None, :] - Rc[None, :, :])  # |C| x |C| x K
        idx = np.arange(len(C))
        P[idx, idx, :] = 0.0
        return P.sum(axis=1) / (len(C) - 1)

    def route(self, F, candidates=None, cost=None, lam_cost=0.0):
        """Routing score S_m(x) = sum_k pi_k(x) v_{m,k} (minus cost)."""
        C = np.arange(self.M) if candidates is None else np.asarray(candidates)
        S = self.gate(F) @ self.domain_strength(C).T
        if cost is not None:
            S = S - lam_cost * np.asarray(cost)[C][None, :]
        return C[np.argmax(S, axis=1)], S

    def n_params(self, p):
        return (self.K - 1) * p + (self.M - 1) * self.K

    def bic(self, F, a, b, y):
        n = len(a)
        return -2 * self.loglik(F, a, b, y) + self.n_params(F.shape[1]) * np.log(n)


# ----------------------------------------------------------------------------
# Label alignment (needed for simulation / bootstrap)
# ----------------------------------------------------------------------------


def align_labels(R_hat, R_ref):
    """Permutation perm such that R_hat[:, perm] best matches R_ref."""
    from scipy.optimize import linear_sum_assignment

    Rh = R_hat - R_hat.mean(0)
    Rr = R_ref - R_ref.mean(0)
    cost = ((Rh[:, :, None] - Rr[:, None, :]) ** 2).sum(0)  # K_hat x K_ref
    row, col = linear_sum_assignment(cost)
    perm = np.empty(R_ref.shape[1], dtype=int)
    perm[col] = row
    return perm


def permute_params(W, R, perm):
    """Relabel components; re-normalise W so that the last row is 0."""
    Wp = W[perm]
    Wp = Wp - Wp[-1]
    return Wp, R[:, perm]
