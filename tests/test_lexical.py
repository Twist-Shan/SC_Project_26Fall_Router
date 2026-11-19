from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixbt import StyleMixtureBT
from mixbt.lexical import LexicalBT, LexicalMixtureRouter


def test_pair_symmetry_and_borda_equal_literal_opponent_average():
    prompts = ["debug python code", "write a poem", "请翻译成中文", "数学の問題"] * 6
    a = np.tile([1, 0, 2, 1, 2, 0], 4).astype(np.uint8)
    b = np.tile([0, 1, 1, 2, 0, 2], 4).astype(np.uint8)
    y = np.tile([1, 0, 1, 1, 0, 0], 4)
    f = np.column_stack((np.ones(24), np.tile([-1., 0., 1., 2.], 6)))
    lexical = LexicalBT().fit(prompts, a, b, y, M=3)
    assert np.allclose(lexical.predict_proba(prompts, a, b)
                       + lexical.predict_proba(prompts, b, a), 1)
    mix = StyleMixtureBT(K=2, style=None, seed=37, max_iter=4).fit(f, a.astype(int), b.astype(int), y, M=3)
    pooled = LexicalMixtureRouter(mix, lexical)
    with pytest.raises(ValueError, match="feature row"):
        pooled.predict_proba(f[:1], prompts, a, b)
    with pytest.raises(ValueError, match="feature row"):
        pooled.route(f, prompts[:1])
    candidates = [2, 0, 1]
    picks, scores = pooled.route(f, prompts, candidates)
    for col, m in enumerate(candidates):
        expected = np.mean([pooled.predict_proba(f, prompts, np.full(24, m), np.full(24, j))
                            for j in candidates if j != m], axis=0)
        assert np.allclose(scores[:, col], expected)
    assert np.array_equal(picks, np.asarray(candidates)[scores.argmax(1)])


def test_single_class_and_invalid_candidate_sets():
    model = LexicalBT().fit(["a", "b"], [0, 0], [1, 1], [1, 1], M=3)
    assert model.predict_proba(["a"], [0], [1])[0] > 0.5
    with pytest.raises(ValueError):
        model.borda([""], [1, 1])
    with pytest.raises(ValueError):
        model.predict_proba([""], [-1], [1])
