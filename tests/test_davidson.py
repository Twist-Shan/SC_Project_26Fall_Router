from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from mixbt.davidson import DavidsonLexicalBT, davidson_loss_gradient
from tie_refinement import eligible_ties


def test_three_outcome_gradient():
    rng = np.random.default_rng(137)
    x = sparse.csr_matrix(rng.normal(size=(12, 4)))
    z = sparse.csr_matrix(rng.normal(size=(12, 3)))
    theta = rng.normal(size=7)
    y = np.tile([1, 0, 2], 4)
    def objective(t):
        return davidson_loss_gradient(t, x, z, y, 2.3)
    approx = approx_fprime(theta, lambda t: objective(t)[0], 1e-7)
    np.testing.assert_allclose(objective(theta)[1], approx, rtol=1e-5, atol=2e-6)


def test_tie_symmetry_decisive_probability_and_borda():
    model = DavidsonLexicalBT()
    text = ["code this", "write a poem", "why?", "list names"] * 6
    a = np.tile([1, 0, 2, 1, 2, 0], 4)
    b = np.tile([0, 1, 1, 2, 0, 2], 4)
    y = np.tile([1, 1, 0, 0, 2, 2], 4)
    model.fit(text, a, b, y, M=3)
    out = model.predict_outcomes(text, a, b)
    np.testing.assert_allclose(out.sum(1), 1)
    np.testing.assert_allclose(out[:, [1, 0, 2]], model.predict_outcomes(text, b, a))
    np.testing.assert_allclose(out[:, 0] / out[:, :2].sum(1), model.predict_proba(text, a, b))
    scores = model.borda(text, [2, 0, 1])
    for k, m in enumerate([2, 0, 1]):
        expected = np.mean([model.predict_proba(text, np.full(len(a), m), np.full(len(a), j))
                            for j in [2, 0, 1] if j != m], axis=0)
        np.testing.assert_allclose(scores[:, k], expected)


def test_excludes_all_ties_sharing_outer_or_inner_heldout_groups():
    groups = np.array(["outer", "outer", "train", "inner", "inner"])
    outer_train = groups != "outer"
    inner_hold = groups == "inner"
    ok = eligible_ties(np.array(["inner", "only tie", "outer", "train", "inner"]), groups,
                       outer_train, inner_hold)
    np.testing.assert_array_equal(ok, [False, True, False, True, False])
