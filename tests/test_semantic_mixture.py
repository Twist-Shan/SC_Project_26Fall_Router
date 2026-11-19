"""A held-out prompt must not affect the semantic EM design on training rows."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from semantic_mixture import semantic_design


def test_pca_scaling_and_collinearity_are_fit_within_training():
    rng = np.random.default_rng(61)
    embeddings = rng.normal(size=(60, 12))
    first_attr = rng.normal(size=60)
    attributes = np.column_stack([first_attr, first_attr, np.ones(60)])
    train = np.arange(60) < 45
    original = semantic_design(embeddings, attributes, train, 4, 97)
    embeddings[~train] += 100
    attributes[~train] *= 20
    changed = semantic_design(embeddings, attributes, train, 4, 97)
    np.testing.assert_array_equal(original[train], changed[train])
    assert original.shape == (60, 6)
    np.testing.assert_array_equal(original[:, 0], 1)
    np.testing.assert_allclose(original[train, 1:].mean(axis=0), 0, atol=1e-12)
    assert not np.allclose(original[~train], changed[~train])
