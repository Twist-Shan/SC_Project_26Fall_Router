"""Evaluate the fixed 50:50 mixture/Davidson router on decisive grouped Arena folds.

Training ties whose first prompt belongs to the outer test or inner validation
groups are excluded. All regularisation and pooling parameters are fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import digest, first_prompt, grouped_folds, inner_split, load_data, prompt_group
from lexical_refinement import bootstrap_contrasts, losses
from mixbt.lexical import LexicalBT
from mixbt.davidson import DavidsonLexicalBT, tie_design


def eligible_ties(tie_groups, decisive_groups, outer_train, inner_holdout):
    excluded_groups = np.unique(decisive_groups[~outer_train | inner_holdout])
    return ~np.isin(tie_groups, excluded_groups)


def load_ties(path, models):
    df = pd.read_csv(path, usecols=["prompt", "model_a", "model_b", "winner_model_a",
                                    "winner_model_b", "winner_tie"])
    df = df.loc[df.winner_tie.eq(1) & df.winner_model_a.eq(0) & df.winner_model_b.eq(0)].copy()
    df["first"] = df.prompt.map(first_prompt)
    df = df.loc[df["first"].str.strip().ne("")]
    index = {m: i for i, m in enumerate(models)}
    if not (df.model_a.isin(index).all() and df.model_b.isin(index).all()):
        raise ValueError("Tie data contains a candidate outside the decisive population")
    return (df["first"].to_numpy(dtype=object), df["first"].map(prompt_group).to_numpy(),
            df.model_a.map(index).to_numpy(int), df.model_b.map(index).to_numpy(int))


def report_metrics(methods, y, groups, folds, output, draws):
    metrics = {k: np.column_stack((((p > .5) == y).astype(float), losses(p, y)))
               for k, p in methods.items()}
    if not all(np.isfinite(x).all() for x in metrics.values()):
        raise ValueError("Nonfinite out-of-fold prediction")
    summary, by_fold = [], []
    for name, values in metrics.items():
        for fold in np.unique(folds):
            mask = folds == fold
            acc, loss = values[mask].mean(axis=0)
            by_fold.append(dict(method=name, fold=fold, n=mask.sum(), accuracy=acc, logloss=loss))
        acc, loss = values.mean(axis=0)
        summary.append(dict(method=name, n=len(y), accuracy=acc, logloss=loss))
    comparisons = [("Davidson pool", "lexical pool"), ("Davidson pool", "mixture"),
                   ("Davidson lexical", "decisive lexical")]
    differences = np.column_stack([metrics[alt] - metrics[control] for alt, control in comparisons])
    bounds = bootstrap_contrasts(differences, groups, draws)
    contrasts = []
    for i, (alt, control) in enumerate(comparisons):
        row = dict(method=alt, reference=control)
        for col, metric in enumerate(("accuracy", "logloss")):
            idx = 2*i + col
            row[metric] = differences[:, idx].mean()
            row[metric+"_lo"], row[metric+"_hi"] = bounds[:, idx]
        contrasts.append(row)
    for filename, rows in (("metrics", summary), ("folds", by_fold), ("contrasts", contrasts)):
        pd.DataFrame(rows).to_csv(output/(filename+".csv"), index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


def run(data_path, reference, output, draws, save_predictions):
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_path)
    y, groups, M = data["y"], data["groups"], data["M"]
    ref = np.load(reference / "oof_predictions.npz")
    protocol = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    seed = protocol["seed"]
    folds = grouped_folds(groups, seed=seed)
    text_hash = hashlib.sha256(json.dumps(data["text"].tolist(), ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
    row_hash = digest(zip(data["ids"], data["a"], data["b"], y, groups))
    if (protocol["quick"] or text_hash != protocol["exact_first_prompt_hash"]
            or row_hash != protocol["data_hash"] or not np.array_equal(ref["ids"], data["ids"])
            or not np.array_equal(ref["fold"], folds)):
        raise ValueError("Incompatible decisive Arena evaluation reference")
    texts, tie_groups, a, b = load_ties(data_path, data["models"])
    all_texts = np.r_[data["text"], texts]
    a, b = np.r_[data["a"], a], np.r_[data["b"], b]
    labels = np.r_[y, np.full(len(texts), 2)]
    lexical = LexicalBT(C=.03)
    f = lexical.encode(all_texts)
    design = lexical.pair_design(f, a, b, M)
    tie_f = tie_design(f, a, b, M)
    del f
    pred_tie = np.full(len(y), np.nan)
    pred_control = np.full(len(y), np.nan)
    recorded = []
    for fold in np.unique(folds):
        train = folds != fold
        _, hold = inner_split(groups, train, seed=seed+101+fold)
        train_ties = eligible_ties(tie_groups, groups, train, hold)
        keep = np.r_[train, train_ties]
        test_rows = np.flatnonzero(~train)
        train_rows = np.flatnonzero(train)
        fit = DavidsonLexicalBT(C=.03).fit_design(design[keep], tie_f[keep], labels[keep], M)
        pred_tie[test_rows] = expit(design[test_rows] @ fit.coef_)
        control = LexicalBT(C=.03).fit_design(design[train_rows], y[train], M)
        pred_control[test_rows] = control.model_.predict_proba(design[test_rows])[:, 1]
        recorded.append(dict(fold=fold, train_decisive=train.sum(), train_ties=train_ties.sum(),
                             total_eligible_source_ties=len(texts), test_decisive=len(test_rows),
                             iterations=fit.n_iter_, scaled_gradient_max=fit.gradient_max_))
        print(recorded[-1], flush=True)
    mixture = ref["mixture"]
    methods = {"Davidson pool": .5*(mixture + pred_tie), "lexical pool": .5*(mixture + pred_control),
               "Davidson lexical": pred_tie, "decisive lexical": pred_control, "mixture": mixture}
    pd.DataFrame(recorded).to_csv(output/"training.csv", index=False)
    report_metrics(methods, y, groups, folds, output, draws)
    if save_predictions:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_predictions, ids=data["ids"], fold=folds, pred=methods["Davidson pool"],
                            control=methods["lexical pool"], decisive=pred_control, davidson=pred_tie)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=ROOT/"data/arena_slim.csv")
    cli.add_argument("--reference", type=Path, default=ROOT/"results/nested")
    cli.add_argument("--output", type=Path, default=ROOT/"results/tie_refinement")
    cli.add_argument("--bootstrap", type=int, default=1000)
    cli.add_argument("--save-predictions", type=Path)
    args = cli.parse_args()
    if args.bootstrap < 1:
        cli.error("--bootstrap must be positive")
    with threadpool_limits(limits=1):
        run(args.data, args.reference, args.output, args.bootstrap, args.save_predictions)
