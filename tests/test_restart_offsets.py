import numpy as np

from mixbt.restarts import RestartRouter, pair_offsets


def test_pair_offsets_reciprocal_and_unsupported():
    a = np.array([0, 1, 0, 1])
    b = 1 - a
    y = np.array([1, 0, 1, 0])
    p = np.full(4, 0.5)
    offsets = pair_offsets(a, b, y, p, 3)
    assert offsets[0, 1] > 0
    assert offsets[0, 2] == 0
    np.testing.assert_allclose(offsets, pair_offsets(b, a, 1 - y, 1 - p, 3))


def test_restart_router_borda_matches_pair_predictions():
    rng = np.random.default_rng(53)
    X = np.column_stack((np.ones(120), rng.normal(size=120)))
    a = rng.integers(3, size=120)
    b = (a + rng.integers(1, 3, size=120)) % 3
    y = rng.integers(2, size=120)
    router = RestartRouter(dict(K=2, lam_W=2., lam_R=2.), starts=2).fit(X, a, b, y, 3)
    F = X[:5]
    for average in (False, True):
        for correct in (False, True):
            scored = np.zeros((len(F), 3))
            for u in range(3):
                for v in range(u+1,3):
                    left, right = np.full(len(F),u), np.full(len(F),v)
                    p = router.predict_proba(F,left,right,average,correct)
                    np.testing.assert_allclose(1-p,router.predict_proba(F,right,left,average,correct))
                    scored[:,u] += p/2
                    scored[:,v] += (1-p)/2
            winners, scores = router.route(F, [0,1,2],average,correct)
            np.testing.assert_allclose(scores, scored)
            np.testing.assert_array_equal(winners, scored.argmax(axis=1))
