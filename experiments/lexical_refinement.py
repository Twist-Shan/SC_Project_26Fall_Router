"""Five grouped outer folds for the fixed 50:50 lexical/BT-mixture probability pool.

Run arena_nested.py first. The reference predictions and splits are checked
against row IDs, group assignments, true outcomes, and the exact first prompts.
Only the new fit's ridge strength is selected, on each outer training fold's
grouped inner holdout. The lexical feature map uses no sample-fitted statistics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import digest, grouped_folds, inner_split, load_data
from mixbt.lexical import LexicalBT


def losses(p, y):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def bootstrap_contrasts(deltas, groups, draws):
    # Reuse each bootstrap draw for all predictions and both paired outcomes.
    _, group = np.unique(groups, return_inverse=True)
    counts = np.bincount(group)
    totals = np.column_stack([np.bincount(group, weights=d) for d in deltas.T])
    sampled = np.empty((draws, totals.shape[1]))
    rng = np.random.default_rng(8191)
    for i in range(draws):
        idx = rng.integers(0, len(counts), len(counts))
        sampled[i] = totals[idx].sum(axis=0) / counts[idx].sum()
    return np.quantile(sampled, [0.025, 0.975], axis=0)


def run(reference, output, draws):
    data = load_data()
    y, groups, M = data["y"], data["groups"], data["M"]
    archive = np.load(reference / "oof_predictions.npz")
    protocol = json.loads((reference / "protocol.json").read_text())
    seed = protocol["seed"]
    folds = grouped_folds(groups, seed=seed)
    text_hash = hashlib.sha256(json.dumps(data["text"].tolist(), ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
    row_hash = digest(zip(data["ids"], data["a"], data["b"], y, groups))
    if (protocol["quick"] or text_hash != protocol["exact_first_prompt_hash"] or
            row_hash != protocol["data_hash"] or not np.array_equal(archive["ids"], data["ids"])
            or not np.array_equal(archive["fold"], folds)):
        raise ValueError("The reference predictions must match the full grouped Arena protocol")
    model = LexicalBT()
    features = model.encode(data["text"])
    design = model.pair_design(features, data["a"], data["b"], M)
    del features
    preds = np.full(len(y), np.nan)
    tuning = []
    for fold in np.unique(folds):
        train = folds != fold
        inner, hold = inner_split(groups, train, seed=seed + 101 + fold)
        val = np.flatnonzero(hold)
        if len(val) > 4000:
            val = np.sort(np.random.default_rng(seed + 211 + fold).choice(val, 4000, replace=False))
        inner_design, val_design = design[inner], design[val]
        trial = []
        for C in (0.03, 0.15, 0.75, 3.0):
            local = LexicalBT(C=C).fit_design(inner_design, y[inner], M)
            score = float(losses(local.model_.predict_proba(val_design)[:, 1], y[val]).mean())
            trial.append((score, C))
            tuning.append(dict(fold=fold, C=C, inner_logloss=score))
        C = min(trial)[1]
        selected = LexicalBT(C=C).fit_design(design[train], y[train], M)
        preds[~train] = selected.model_.predict_proba(design[~train])[:, 1]
        print(f"Fold {fold}: C={C:g} on grouped inner validation", flush=True)
    if not np.isfinite(preds).all():
        raise ValueError("Missing lexical predictions")
    methods = {"equal pool": 0.5 * (archive["mixture"] + preds),
               "lexical BT": preds, "mixture": archive["mixture"],
               "aggregate-score BT": archive["aggregated capability"], "global BT": archive["global BT"]}
    acc = {name: ((p > 0.5) == y).astype(float) for name, p in methods.items()}
    ll = {name: losses(p, y) for name, p in methods.items()}
    summary = pd.DataFrame([dict(method=k, n=len(y), accuracy=acc[k].mean(), logloss=ll[k].mean())
                            for k in methods])
    contrasts, deltas = [], []
    for alt in ("equal pool", "lexical BT"):
        for control in ("mixture", "aggregate-score BT", "global BT"):
            record = dict(method=alt, reference=control)
            for metric, values in (("accuracy", acc), ("logloss", ll)):
                d = values[alt] - values[control]
                deltas.append(d)
                record[metric] = d.mean()
            contrasts.append(record)
    bounds = bootstrap_contrasts(np.column_stack(deltas), groups, draws)
    for i, row in enumerate(contrasts):
        for j, metric in enumerate(("accuracy", "logloss")):
            row[metric + "_lo"], row[metric + "_hi"] = bounds[:, 2*i + j]
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
    per_fold = []
    for fold in np.unique(folds):
        idx = folds == fold
        for name in methods:
            per_fold.append(dict(fold=fold, n=idx.sum(), method=name,
                                 accuracy=acc[name][idx].mean(), logloss=ll[name][idx].mean()))
    pd.DataFrame(per_fold).to_csv(output / "folds.csv", index=False)
    print(summary.round(6).to_string(index=False))
    print(pd.DataFrame(contrasts).round(6).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/nested")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lexical_refinement")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        run(args.reference, args.output, args.bootstrap)
