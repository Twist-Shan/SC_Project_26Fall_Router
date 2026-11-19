"""Correctness checks: run with `python -m pytest tests -q` from the project root."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from scipy.optimize import approx_fprime

from mixbt import (CapabilityBT, MixtureBT, align_labels, make_params, permute_params,
                   simulate, true_proba)


def _data(n=3000, M=8, K=3, p=4, seed=0, delta=1.5):
    W, R = make_params(M, K, p, delta=delta, rng=seed)
    d = simulate(n, W, R, rng=seed + 1)
    return W, R, d


def test_score_matches_numerical_gradient():
    """Fisher-identity score == finite-difference gradient of the penalised log-lik."""
    W, R, d = _data(n=400)
    m = MixtureBT(K=3, lam_W=0.1, lam_R=0.1)
    m.M = R.shape[0]
    rng = np.random.default_rng(1)
    W0 = W + rng.normal(scale=0.3, size=W.shape)
    W0[-1] = 0
    R0 = R + rng.normal(scale=0.3, size=R.shape)
    gW, gR, val = m.score(d["F"], d["a"], d["b"], d["y"], W0, R0)
    K, p = W.shape
    M = R.shape[0]

    def f(theta):
        Wt = np.vstack([theta[: (K - 1) * p].reshape(K - 1, p), np.zeros((1, p))])
        Rt = theta[(K - 1) * p:].reshape(M, K)
        return m.score(d["F"], d["a"], d["b"], d["y"], Wt, Rt)[2]

    theta = np.concatenate([W0[:-1].ravel(), R0.ravel()])
    num = approx_fprime(theta, f, 1e-6)
    ana = np.concatenate([gW[:-1].ravel(), gR.ravel()])
    assert np.max(np.abs(num - ana)) < 1e-3 * (1 + np.max(np.abs(ana)))


def test_capability_gradient():
    W, R, d = _data(n=300)
    cb = CapabilityBT(K=3, lam_W=0.1, lam_R=0.1)
    K, p = W.shape
    M = R.shape[0]
    rng = np.random.default_rng(2)
    theta = rng.normal(scale=0.5, size=(K - 1) * p + M * K)
    f = lambda t: cb._f_and_g(t, d["F"], d["a"], d["b"], d["y"], M)[0]
    ana = cb._f_and_g(theta, d["F"], d["a"], d["b"], d["y"], M)[1]
    num = approx_fprime(theta, f, 1e-6)
    assert np.max(np.abs(num - ana)) < 1e-3 * (1 + np.max(np.abs(ana)))


def test_squarem_matches_em():
    W, R, d = _data(n=5000, delta=1.5)
    kw = dict(K=3, lam_W=1e-2, lam_R=1e-2, tol=1e-10, max_iter=5000, seed=3)
    m1 = MixtureBT(**kw).fit(d["F"], d["a"], d["b"], d["y"], M=R.shape[0])
    m2 = MixtureBT(accel="squarem", **kw).fit(d["F"], d["a"], d["b"], d["y"], M=R.shape[0])
    assert abs(m1.history_.objective[-1] - m2.history_.objective[-1]) < 1e-2
    assert np.all(np.diff(m2.history_.objective) > -1e-8 * abs(m2.history_.objective[0]))


def test_em_monotone_and_recovers():
    W, R, d = _data(n=20000, delta=1.5)
    m = MixtureBT(K=3, lam_W=1e-3, lam_R=1e-3, tol=1e-9, max_iter=300, init="kmeans", n_init=3)
    m.fit(d["F"], d["a"], d["b"], d["y"], M=R.shape[0])
    obj = np.array(m.history_.objective)
    assert np.all(np.diff(obj) > -1e-6 * np.abs(obj[:-1])), "penalised log-lik decreased"
    perm = align_labels(m.R_, R)
    Wh, Rh = permute_params(m.W_, m.R_, perm)
    rel = np.linalg.norm(Rh - R) / np.linalg.norm(R)
    assert rel < 0.25, rel
    # fitted probabilities close to the truth
    p_hat = m.predict_proba(d["F"], d["a"], d["b"])
    p_true = true_proba(d["F"], d["a"], d["b"], W, R)
    assert np.mean(np.abs(p_hat - p_true)) < 0.03


def test_k1_equals_global_bt():
    from mixbt import GlobalBT
    W, R, d = _data(n=2000)
    m = MixtureBT(K=1, lam_R=1e-2).fit(d["F"], d["a"], d["b"], d["y"], M=R.shape[0])
    g = GlobalBT(lam_R=1e-2).fit(d["a"], d["b"], d["y"], R.shape[0])
    assert np.allclose(m.R_[:, 0], g.r_, atol=1e-6)


if __name__ == "__main__":
    test_score_matches_numerical_gradient()
    test_capability_gradient()
    test_squarem_matches_em()
    test_k1_equals_global_bt()
    test_em_monotone_and_recovers()
    print("all tests passed")
