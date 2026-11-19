"""Tests for the baseline routers in mixbt/routers.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
from scipy.optimize import approx_fprime  # noqa: E402

from mixbt import routers  # noqa: E402


def _data(n=400, M=6, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    a = rng.integers(0, M, n)
    b = (a + 1 + rng.integers(0, M - 1, n)) % M
    r = np.linspace(1, -1, M)
    y = (rng.random(n) < 1 / (1 + np.exp(-(r[a] - r[b])))).astype(float)
    return X, a, b, y, M


def test_bilinear_gradient():
    X, a, b, y, M = _data()
    captured = {}

    def fake(f, th0, jac, method, options):
        captured["f"], captured["th0"] = f, th0

        class R:
            x = th0
        return R()

    orig = routers.minimize
    routers.minimize = fake
    try:
        routers.Bilinear(k=2, lam=0.7).fit(X, a, b, y, M)
    finally:
        routers.minimize = orig
    f = captured["f"]
    th = np.random.default_rng(1).normal(size=captured["th0"].shape) * 0.5
    g = f(th)[1]
    gn = approx_fprime(th, lambda t: f(t)[0], 1e-6)
    assert np.max(np.abs(g - gn)) < 1e-3 * max(1.0, np.max(np.abs(g)))


def test_swbt_uniform_weights_is_global_bt():
    X, a, b, y, M = _data(n=2000)
    m = routers.SWBT(gamma=1.0, lam=2.0, newton_iter=20).fit(X, a, b, y, M)   # gamma = 1: equal weights
    R = m.item_scores(X[:5])
    assert np.allclose(R, m.r0_[None], atol=1e-6)
    assert np.corrcoef(m.r0_, np.linspace(1, -1, M))[0, 1] > 0.95


def test_bilinear_intercept_scaling_cannot_escape_penalty():
    """A fixed low-rank intercept must not gain by making one factor unbounded."""
    X, a, b, y, M = _data(n=60, p=2)
    X[:] = 0
    captured = {}

    def fake(f, th0, jac, method, options):
        captured["objective"] = f
        class Result:
            x = th0
        return Result()

    original = routers.minimize
    routers.minimize = fake
    try:
        model = routers.Bilinear(k=1, lam=1.).fit(X, a, b, y, M)
    finally:
        routers.minimize = original
    theta = model.theta_.copy()
    theta[:] = 0
    theta[M:2 * M] = np.linspace(-1, 1, M)
    theta[2 * M] = 1.  # A's intercept column
    escaped = theta.copy()
    escaped[M:2 * M] *= .01
    escaped[2 * M] /= .01
    model.theta_ = theta
    before = model.predict_proba(X, a, b)
    model.theta_ = escaped
    assert np.allclose(before, model.predict_proba(X, a, b))
    f = captured["objective"]
    assert f(escaped)[0] > f(theta)[0] + 1000.


def test_pairlogit_runs():
    X, a, b, y, M = _data()
    s = np.where(a < 2, 1.0, -1.0)
    m = routers.PairLogit(lam=10.0).fit(X, np.zeros(len(y)), s, y)
    p = m.predict_proba(X, np.zeros(len(y)), s)
    assert np.all((p > 0) & (p < 1))


if __name__ == "__main__":
    test_bilinear_gradient()
    test_swbt_uniform_weights_is_global_bt()
    test_bilinear_intercept_scaling_cannot_escape_penalty()
    test_pairlogit_runs()
    print("router tests passed")
