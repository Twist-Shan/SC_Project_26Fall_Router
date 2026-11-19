"""Regression tests for data leakage and corrected offline evaluation semantics."""
import json
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from arena_pipeline import PromptFeatures, categories, first_prompt, grouped_folds, inner_split, load_data, prompt_group
from arena_nested import curve_metrics, fit_predict, quality_curve, validate_predictions


def test_future_turns_excluded():
    a = json.dumps(["Please explain gravity"])
    b = json.dumps(["Please explain gravity", "Future answer-dependent follow-up"])
    assert first_prompt(a) == first_prompt(b)
    assert categories([first_prompt(a)]).equals(categories([first_prompt(b)]))
    assert "multi_turn" not in categories([first_prompt(a)]).columns


def test_grouped_nested_split():
    groups = np.array([prompt_group(f"prompt {i // 3}") for i in range(150)])
    assert prompt_group("  Hello\nWORLD ") == prompt_group("hello world")
    fid = grouped_folds(groups, seed=1729)
    assert np.array_equal(fid, grouped_folds(groups, seed=1729))
    for fold in range(5):
        train = fid != fold
        assert not set(groups[train]) & set(groups[~train])
        fit, val = inner_split(groups, train, seed=2718 + fold)
        assert not set(groups[fit]) & set(groups[val])
        assert not (fit & ~train).any() and not (val & ~train).any()
        assert np.array_equal(fit | val, train)


def test_train_only_features():
    text = np.array([f"Explain gravity planet orbit sample{i}" if i % 2 else
                     f"Write python code function sample{i}" for i in range(30)])
    transformer = PromptFeatures(n_topics=3, n_lsa=4, min_df=1, seed=37).fit(text)
    vocab = dict(transformer.tfidf_.vocabulary_)
    before = transformer.transform(text)["G"]
    unknown = transformer.transform(["zzzzheldouttoken newvalidationword"])
    assert "zzzzheldouttoken" not in vocab
    assert transformer.tfidf_.vocabulary_ == vocab
    assert np.allclose(before, transformer.transform(text)["G"])
    assert np.isfinite(unknown["G"]).all()
    assert np.allclose(unknown["topics"].sum(1), 1)
    assert np.linalg.matrix_rank(before) == before.shape[1]
    assert all("multi_turn" not in name for name in transformer.gate_names_)


def test_load_raw_first_prompt():
    data = pd.DataFrame(dict(id=[1, 2, 3], model_a=["a"] * 3, model_b=["b"] * 3,
                             prompt=[json.dumps(["Same prompt"]), json.dumps(["Same prompt", "Later"]), '[]'],
                             winner_model_a=[1, 0, 1], winner_model_b=[0, 1, 0], winner_tie=[0, 0, 0]))
    D = load_data(StringIO(data.to_csv(index=False)))
    assert len(D["y"]) == 2 and D["groups"][0] == D["groups"][1]
    assert D["text"].dtype == object
    assert D["text"].nbytes == len(D["text"]) * np.dtype(object).itemsize


def test_ragged_prompt_storage_preserves_long_text():
    long_prompt = "x" * 100000
    data = pd.DataFrame(dict(id=[1, 2], model_a=["a", "a"], model_b=["b", "b"],
                             prompt=[json.dumps(["short"]), json.dumps([long_prompt])],
                             winner_model_a=[1, 0], winner_model_b=[0, 1], winner_tie=[0, 0]))
    D = load_data(StringIO(data.to_csv(index=False)))
    assert D["text"].dtype == object
    assert D["text"].tolist() == ["short", long_prompt]
    assert D["text"].nbytes < 100  # references only, not two 100k-character buffers


def test_quality_score_ties_have_no_order_artifact():
    score = np.ones(10)
    y = np.array([1] * 8 + [0] * 2)
    metrics = curve_metrics(*quality_curve(score, y))
    assert abs(metrics["apgr"] - .5) < 1e-12
    assert abs(metrics["cpt80"] - .8) < 1e-12
    assert metrics == curve_metrics(*quality_curve(score[::-1], y[::-1]))
    weights = np.array([0, 1, 2, 0, 1, 1, 3, 1, 1, 1])
    assert abs(curve_metrics(*quality_curve(score, y, weights))["apgr"] - .5) < 1e-12


def test_offset_has_no_test_label_dependency():
    rng = np.random.default_rng(91)
    n = 80
    D = dict(a=np.zeros(n, int), b=rng.integers(1, 3, n), y=rng.integers(0, 2, n).astype(float),
             models=np.array(["gpt-4-0613", "cheap-a", "cheap-b"]), M=3)
    Z = {"X": rng.normal(size=(n, 3))}
    train = np.arange(n) < 60
    test = np.arange(60, n)
    p = fit_predict("logistic", dict(lam=100.), D, Z, train, test, seed=91)
    D2 = dict(D, y=D["y"].copy())
    D2["y"][test] = 1 - D2["y"][test]
    p2 = fit_predict("logistic", dict(lam=100.), D2, Z, train, test, seed=91)
    assert np.allclose(p, p2)


def test_nonfinite_predictions_rejected_on_task_rows():
    D = dict(a=np.array([0, 1]), b=np.array([1, 2]),
             models=np.array(["gpt-4-0613", "cheap-a", "cheap-b"]))
    test = np.arange(2)
    assert np.array_equal(validate_predictions("logistic", np.array([.6, np.nan]), D, test), [True, False])
    for kind, p in (("mixture", [.6, np.nan]), ("logistic", [np.nan, np.nan]), ("MF", [.5, 1.1])):
        try:
            validate_predictions(kind, np.array(p), D, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid {kind} predictions were silently accepted")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        for name, test in list(globals().items()):
            if name.startswith("test_") and callable(test):
                test()
                print(name, "passed", flush=True)
