import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixbt import StyleMixtureBT
from mixbt.semantic import SemanticFamilyBT, SemanticMixtureRouter


def test_shared_family_prediction_matches_joint_design_and_pool():
    rng = np.random.default_rng(719)
    n = 90
    ids = ["gpt-4-a", "gpt-4-b", "claude-1", "claude-2", "singleton"]
    a = rng.integers(0, len(ids), n)
    b = (a + rng.integers(1, len(ids), n)) % len(ids)
    embeddings = normalize(rng.normal(size=(n, 8)))
    prompts = ["Explain the integer " + str(i) for i in range(n)]
    y = rng.integers(0, 2, n)
    with threadpool_limits(limits=1):
        model = SemanticFamilyBT(ids, C=.1, semantic_scale=2).fit(prompts, embeddings, a, b, y)
        f = np.c_[np.ones(n), embeddings[:, :2]]
        mix = StyleMixtureBT(K=2, style=None, seed=23, max_iter=5).fit(f, a, b, y, M=5)
    raw = model.lexical.encode(prompts)
    z = model.basis.transform(embeddings)
    d = sparse.hstack((model.lexical.pair_design(raw, a, b, model.M),
                       2 * model.semantic_design(z, a, b)), format="csr")
    p = model.predict_proba(prompts, embeddings, a, b)
    assert np.allclose(model.estimator.predict_proba(d)[:, 1], p, atol=2e-7)
    assert np.allclose(p + model.predict_proba(prompts, embeddings, b, a), 1)
    pooled = SemanticMixtureRouter(mix, model)
    c = [4, 1, 2]
    picks, scores = pooled.route(f, prompts, embeddings, c)
    for j, m in enumerate(c):
        expected = np.mean([pooled.predict_proba(f, prompts, embeddings, np.full(n, m), np.full(n, other))
                            for other in c if other != m], axis=0)
        assert np.allclose(scores[:, j], expected)
    assert np.array_equal(picks, np.asarray(c)[scores.argmax(axis=1)])
    with pytest.raises(ValueError):
        pooled.route(f[:1], prompts, embeddings, c)
    with pytest.raises(ValueError):
        model.predict_proba(prompts, embeddings, np.full(n, -1), b)
