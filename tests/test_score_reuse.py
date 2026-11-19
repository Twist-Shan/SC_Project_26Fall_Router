"""Joint-likelihood acceleration must preserve the previous score and traces."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixbt.extended import StyleMixtureBT, bt_loglik
from mixbt.model import FitHistory, MixtureBT


def test_pair_score_terms_equal_direct_for_binary_soft_and_covariate_outcomes():
    rng = np.random.default_rng(73)
    # Compact integer IDs exercise pair-key arithmetic; both orientations occur.
    a = np.tile(np.array([97, 99, 98, 99], dtype=np.int8), 20)
    b = np.tile(np.array([99, 97, 99, 98], dtype=np.int8), 20)
    R = rng.normal(0, 2, size=(100, 3))
    R[97] = [-1000, 0, 1000]
    y = rng.integers(0, 2, size=len(a)).astype(float)
    soft = y.copy()
    soft[3] = 0.5
    for votes, D, style, repeated in [(y, None, None, True), (soft, None, None, False),
                                      (y, rng.normal(size=(len(a), 1)), "domain", False)]:
        model = StyleMixtureBT(K=3, style=style)
        model.M = 100
        bd = model._bd(a, b, votes, D)
        B = rng.normal(size=(3, bd.q))
        ll, win = model._score_terms(bd, R, B)
        eta = model._eta(bd, R, B)
        assert (bd.score_pair_index is not None) == repeated
        np.testing.assert_array_equal(ll, bt_loglik(eta, votes))
        np.testing.assert_array_equal(win, expit(eta))


def test_score_trace_reuses_only_the_value_at_the_recorded_point():
    model = StyleMixtureBT(K=2, style=None)
    model.M, model.q = 2, 0
    model.W_, model.R_, model.B_ = np.zeros((2, 1)), np.zeros((2, 2)), np.empty((2, 0))
    model.history_ = FitHistory()
    calls = []

    def score(F, bd, W, R, B):
        calls.append(W[0, 0])
        return np.zeros(5), W[0, 0]

    def drive(fun, theta, *, callback, **kwargs):
        fun(theta)
        callback(theta.copy())  # The same parameters need no new score.
        theta[0] = 1.0         # A reused optimiser buffer must not change the cache.
        callback(theta)
        fun(theta)
        callback(theta.copy())
        return SimpleNamespace(x=theta, nit=3)

    with patch.object(model, "score", score), patch("mixbt.extended.minimize", drive):
        model._polish(np.ones((2, 1)), None)
    assert calls == [0.0, 1.0, 1.0]
    assert model.history_.objective == [0.0, 1.0, 1.0]


def test_routing_cache_tracks_candidate_order_and_updated_skills():
    rng = np.random.default_rng(37)
    F = rng.normal(size=(8, 3))
    for cls in (MixtureBT, StyleMixtureBT):
        model = cls(K=2)
        model.M = 4
        model.W_ = rng.normal(size=(2, 3))
        model.R_ = rng.normal(size=(4, 2))
        first = np.array([2, 0, 3])
        with patch("mixbt.model.expit", wraps=expit) as sigmoid:
            for candidates, change, expected_calls in [(first, False, 1), (first, False, 1),
                                                       (first[::-1], False, 2), (first[::-1], True, 3)]:
                if change:
                    model.R_[0, 0] += 0.8
                skills = model.R_[candidates]
                p = expit(skills[:, None, :] - skills[None, :, :])
                p[np.arange(len(skills)), np.arange(len(skills))] = 0
                expected = model.gate(F) @ (p.sum(1) / (len(skills) - 1)).T
                choice, scores = model.route(F, candidates)
                np.testing.assert_array_equal(scores, expected)
                np.testing.assert_array_equal(choice, candidates[expected.argmax(1)])
                model.domain_strength(candidates)[:] = 12
                assert sigmoid.call_count == expected_calls
