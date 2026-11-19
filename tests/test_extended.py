"""Checks for the style-adjusted mixture. Run: python tests/test_extended.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from scipy.optimize import approx_fprime

from mixbt import (BattleData, MixtureBT, StyleMixtureBT, align_labels, bt_loglik, component_newton,
                   make_params, simulate, simulate_style)


def _truth(M=8, K=3, p=4, q=2, seed=0):
    W, R = make_params(M, K, p, delta=1.5, rng=seed)
    B = np.random.default_rng(seed).normal(0, 1.0, (K, q))
    return W, R, B


def test_score_matches_numerical_gradient():
    for style in ["domain", "shared", None]:
        W, R, B = _truth()
        d = simulate_style(400, W, R, B, rng=1)
        m = StyleMixtureBT(K=3, style=style, lam_W=0.3, lam_R=0.5, lam_B=0.5)
        m.M = R.shape[0]
        bd = m._bd(d["a"], d["b"], d["y"], d["D"])
        m.q = bd.q
        m._fit_prior(bd)
        Bm = B if style == "domain" else (np.tile(B[0], (3, 1)) if style == "shared" else np.zeros((3, 0)))
        th = m._pack(W, R + 0.1, Bm)
        g, _ = m.score(d["F"], bd, *m._unpack(th, W.shape[1]))
        num = approx_fprime(th, lambda t: m.score(d["F"], bd, *m._unpack(t, W.shape[1]))[1], 1e-6)
        assert np.max(np.abs(num - g)) < 1e-3 * (1 + np.max(np.abs(g))), style


def test_component_newton_is_optimal_and_centred():
    W, R, B = _truth(K=1)
    d = simulate_style(3000, W, R, B, rng=2)
    bd = BattleData(d["a"], d["b"], d["y"], d["D"], R.shape[0])
    w = np.random.default_rng(0).uniform(0.2, 1.0, bd.n)
    r0 = np.random.default_rng(1).normal(size=bd.M)
    r0 -= r0.mean()
    prior = (r0, np.zeros(bd.q))
    r, b = component_newton(bd, w, np.zeros(bd.M), np.zeros(bd.q), prior, (0.5, 0.5))

    def obj(v):
        rr, bb = v[:bd.M], v[bd.M:]
        eta = rr[bd.a] - rr[bd.b] + bd.D @ bb
        return w @ bt_loglik(eta[:, None], bd.y)[:, 0] - 0.25 * (np.sum((rr - r0) ** 2) + bb @ bb)

    g = approx_fprime(np.concatenate([r, b]), obj, 1e-6)
    assert np.max(np.abs(g)) < 1e-2, np.max(np.abs(g))
    assert abs(r.sum()) < 1e-8  # Lemma 2(iii): the maximiser is centred


def test_no_style_matches_mixturebt_objective():
    """style=None is the proposal's model: same optimum as MixtureBT with
    shrinkage to the global fit (same penalties)."""
    W, R = make_params(8, 2, 4, delta=1.5, rng=7)
    d = simulate(4000, W, R, rng=8)
    m1 = MixtureBT(K=2, lam_W=1.0, lam_R=1.0, shrink_to="global", accel="squarem", tol=1e-10,
                   max_iter=3000, n_init=3, polish=True, seed=0).fit(d["F"], d["a"], d["b"], d["y"], M=8)
    m2 = StyleMixtureBT(K=2, lam_W=1.0, lam_R=1.0, style=None, n_init=3, seed=0, prior_lam=(0.1, 0.01))
    m2.fit(d["F"], d["a"], d["b"], d["y"], M=8)
    o1 = m1.loglik(d["F"], d["a"], d["b"], d["y"]) - m1._penalty(m1.W_, m1.R_)
    o2 = m2.objective(d["F"], d["a"], d["b"], d["y"])
    assert abs(o1 - o2) < 1e-2, (o1, o2)


def test_em_monotone_and_recovery():
    W, R, B = _truth(M=10, K=3, p=5, q=2, seed=3)
    d = simulate_style(30000, W, R, B, rng=4)
    m = StyleMixtureBT(K=3, lam_W=1e-3, lam_R=1e-3, lam_B=1e-3, style="domain", n_init=2, polish=False,
                       tol=1e-9, max_iter=500, seed=0)
    m.fit(d["F"], d["a"], d["b"], d["y"], D=d["D"], M=10)
    obj = np.array(m.history_.objective)
    assert np.all(np.diff(obj) > -1e-6 * np.abs(obj[:-1])), "objective decreased"
    perm = align_labels(m.R_, R)
    assert np.linalg.norm(m.R_[:, perm] - R) / np.linalg.norm(R) < 0.2
    assert np.max(np.abs(m.B_[perm] - B)) < 0.3


def test_daem_runs():
    W, R, B = _truth(seed=5)
    d = simulate_style(5000, W, R, B, rng=6)
    m = StyleMixtureBT(K=3, anneal=[0.5, 0.75], n_init=1).fit(d["F"], d["a"], d["b"], d["y"], D=d["D"], M=8)
    p = m.predict_proba(d["F"], d["a"], d["b"], d["D"])
    assert np.all((p > 0) & (p < 1))




def test_ecme_anderson_monotone_and_hybrid():
    """ECME and safeguarded Anderson EM never decrease the penalised
    log-likelihood (Theorem 2); the per-restart hybrid reaches at least the
    objective of the plain-EM pipeline on a well-separated problem."""
    from mixbt import MixtureBT
    W, R = make_params(10, 3, 5, delta=1.0, rng=11)
    d = simulate(15000, W, R, rng=12)
    for acc in ["ecme", "anderson", "ecme-squarem"]:
        m = MixtureBT(K=3, lam_W=1e-2, lam_R=1e-2, tol=1e-10, max_iter=2000, accel=acc, init="random", seed=1)
        m.fit(d["F"], d["a"], d["b"], d["y"], M=10)
        obj = np.array(m.history_.objective)
        assert np.all(np.diff(obj) >= -1e-7 * np.abs(obj[:-1])), acc
    Wd, Rd, Bd = _truth(M=10, K=3, p=5, q=2, seed=13)
    ds = simulate_style(15000, Wd, Rd, Bd, rng=14)
    kw = dict(K=3, lam_W=1e-2, lam_R=1e-2, lam_B=1e-2, n_init=3, seed=2)
    h = StyleMixtureBT(max_iter=30, polish_all=True, **kw).fit(ds["F"], ds["a"], ds["b"], ds["y"], D=ds["D"], M=10)
    e = StyleMixtureBT(max_iter=300, **kw).fit(ds["F"], ds["a"], ds["b"], ds["y"], D=ds["D"], M=10)
    oh = h.objective(ds["F"], ds["a"], ds["b"], ds["y"], ds["D"])
    oe = e.objective(ds["F"], ds["a"], ds["b"], ds["y"], ds["D"])
    assert oh >= oe - 1e-2, (oh, oe)


if __name__ == "__main__":
    test_score_matches_numerical_gradient()
    test_component_newton_is_optimal_and_centred()
    test_daem_runs()
    test_no_style_matches_mixturebt_objective()
    test_em_monotone_and_recovery()
    test_ecme_anderson_monotone_and_hybrid()
    print("all extended tests passed")
