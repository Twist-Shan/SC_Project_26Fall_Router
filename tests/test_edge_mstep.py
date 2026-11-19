"""Regression checks against the row-wise Newton M-step in the Arena fitter."""
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixbt import StyleMixtureBT, make_params, simulate
from mixbt.model import PairData


def test_edge_orientation_with_small_integer_ids():
    a = np.array([99, 101, 99], dtype=np.int8)
    b = np.array([101, 99, 104], dtype=np.int8)
    pairs = PairData(a, b, [1, 0, .5], M=110)
    mass, wins = pairs.aggregate(np.array([[.7, .3], [.2, .8], [.4, .6]]))
    np.testing.assert_array_equal(pairs.u, [99, 99])
    np.testing.assert_array_equal(pairs.v, [101, 104])
    np.testing.assert_allclose(mass, [[.9, 1.1], [.4, .6]])
    np.testing.assert_allclose(wins, [[.9, 1.1], [.2, .3]])


def test_weighted_update_with_disconnected_graph_and_empty_component():
    rng = np.random.default_rng(53)
    a = np.tile([0, 2, 1, 3, 4, 1], 15)
    b = np.tile([1, 1, 0, 4, 3, 2], 15)
    y = rng.choice([0.0, .5, 1.0], len(a))
    p = rng.random(len(a))
    gamma = np.column_stack([p, 1 - p, np.zeros(len(a))])
    model = StyleMixtureBT(K=3, style=None, lam_R=1.3)
    model.M = 6  # candidate five is absent; {0,1,2} and {3,4} are disconnected
    bd = model._bd(a, b, y, None)
    B = np.empty((3, 0))
    R = rng.normal(size=(6, 3))
    R -= R.mean(axis=0)
    prior = rng.normal(size=6)
    model.prior_ = (prior - prior.mean(), np.empty(0))
    fast, _ = model._mstep_pref(bd, gamma, R, B)
    model.aggregate_pairs = False
    reference, _ = model._mstep_pref(bd, gamma, R, B)
    np.testing.assert_allclose(fast, reference, rtol=0, atol=2e-8)
    np.testing.assert_allclose(fast[:, 2], model.prior_[0], atol=1e-10)
    residuals = gamma * (y[:, None] - expit(fast[a] - fast[b]))
    gradient = bd.X.T @ residuals - model.lam_R * (fast - model.prior_[0][:, None])
    np.testing.assert_allclose(gradient, 0., atol=2e-8)


def test_full_em_fit_agrees_at_same_stop():
    W, R = make_params(7, 2, 4, rng=53)
    d = simulate(900, W, R, rng=71)
    fit = dict(K=2, style=None, lam_R=4., n_init=1, seed=89, max_iter=150,
               tol=1e-5, polish=False)
    models = [StyleMixtureBT(**fit, aggregate_pairs=s).fit(d["F"], d["a"], d["b"], d["y"])
              for s in [False, True]]
    first, second = models
    assert first.n_iter_em_ == second.n_iter_em_ < fit["max_iter"]
    np.testing.assert_allclose(first.history_.objective, second.history_.objective, rtol=0, atol=1e-7)
    held = simulate(250, W, R, rng=97)
    predictions = [m.predict_proba(held["F"], held["a"], held["b"]) for m in models]
    np.testing.assert_allclose(*predictions, rtol=0, atol=1e-7)


def test_response_covariates_keep_observation_level_solver():
    rng = np.random.default_rng(53)
    a, b = np.tile([0, 1], 12), np.tile([1, 0], 12)
    y = rng.integers(2, size=len(a))
    D = rng.normal(size=(len(a), 2))
    for style in ("shared", "domain"):
        model = StyleMixtureBT(K=2, style=style)
        model.M = 2
        bd = model._bd(a, b, y, D)
        model._fit_prior(bd)
        gamma = rng.dirichlet([1., 2.], size=len(a))
        params = np.zeros((2, 2)), np.zeros((2, 2))
        fast = model._mstep_pref(bd, gamma, *params)
        assert "pairs" not in bd.__dict__
        model.aggregate_pairs = False
        reference = model._mstep_pref(bd, gamma, *params)
        for left, right in zip(fast, reference):
            np.testing.assert_array_equal(left, right)


def test_polish_iterations_belong_to_the_selected_restart():
    W, R = make_params(7, 2, 4, rng=53)
    d = simulate(900, W, R, rng=71)
    model = StyleMixtureBT(K=2, style=None, n_init=3, seed=89, max_iter=4, polish_all=True)
    model.fit(d["F"], d["a"], d["b"], d["y"])
    assert model.n_iter_polish_ > 0
    assert model.n_iter_em_ + model.n_iter_polish_ == len(model.history_.objective)
