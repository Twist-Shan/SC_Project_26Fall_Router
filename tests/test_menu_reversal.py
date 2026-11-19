"""Check the structural calculation independently of the fitted models."""
import itertools
from fractions import Fraction as F
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from menu_reversal import CONTEXT, evaluate, population, scalar_floor


def test_exact_menu_reversal_floor_and_shared_order_null():
    w = [F(7, 20), F(8, 20), F(5, 20)]
    near, far = F(9, 10), F(81, 82)
    ab = (w[0] + w[2]) * near + w[1] * (1 - far)
    ac = w[0] * far + (w[1] + w[2]) * (1 - near)
    bc = (w[0] + w[1]) * near + w[2] * (1 - far)
    target = population("cyclic")[0]
    np.testing.assert_allclose(CONTEXT[0], [.35, .40, .25])
    np.testing.assert_allclose(target, [float(v) for v in (ab, ac, bc)])
    scores = ({0: ab, 1: 1 - ab},
              {0: (ab + ac) / 2, 1: (1 - ab + bc) / 2, 2: (2 - ac - bc) / 2})
    losses = []
    for order in itertools.permutations(range(3)):
        picks = [next(i for i in order if i in menu) for menu in scores]
        losses.append(sum(max(s.values()) - s[i] for s, i in zip(scores, picks)) / 2)
    assert min(losses) == F(91, 2050)
    np.testing.assert_allclose(scalar_floor(target), float(min(losses)))
    assert evaluate(target, target)["both_correct"]
    assert evaluate(target, target)["choice_ab"] == "A"
    assert evaluate(target, target)["choice_abc"] == "B"
    assert scalar_floor(population("shared-order")[0]) == 0
