"""Regression check for the outcome-blind within-pair control."""
import itertools
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from lexical_refinement import losses
from prompt_weight_control import within_pair_scores


def test_exact_pair_permutation_with_orientation_ties_and_singletons():
    a = np.array([0, 1, 0, 0, 1])
    b = np.array([1, 0, 1, 2, 0])
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([.2, .35, .5, .62, .7])
    folds = np.array([0, 0, 0, 0, 1])
    acc, nll = within_pair_scores(a, b, y, p, folds, 3)
    scores, losses_by_draw = [], []
    for perm in itertools.permutations(range(3)):
        source = np.r_[perm, 3, 4]
        new_p = np.where(a == a[source], p[source], 1 - p[source])
        scores.append((new_p > .5) == y)
        losses_by_draw.append(losses(new_p, y))
    np.testing.assert_allclose(acc, np.mean(scores, axis=0))
    np.testing.assert_allclose(nll, np.mean(losses_by_draw, axis=0))
    # Only the first three observations share a fold and an unordered pair.
    labels = np.array([0., 1., 1.])
    probabilities = np.array([.2, .65, .5])
    log_odds = np.log(probabilities / (1 - probabilities))
    covariance = np.mean((labels - labels.mean()) * (log_odds - log_odds.mean()))
    np.testing.assert_allclose(np.mean(nll - losses(p, y)), 3 / 5 * covariance)
