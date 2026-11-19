"""Isolate the contribution of prompt-specific weights in a fitted Arena EM router.

Compare the same skills and candidate pair under the original prompt and under
the training-fold mean mixing weights. A second, descriptive control averages
over all permutations of prompts within the same held-out fold and unordered
candidate pair. The metrics average per-permutation scores, not probabilities.
All battles and the previously defined 45--55% training-BT stratum are reported.
Prompt-group bootstrap intervals condition on the fitted out-of-fold predictions
and, for the permutation diagnostic, the fold's observed donor prompt pool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from arena_pipeline import categories, load_data
from lexical_refinement import bootstrap_contrasts, losses
from semantic_mixture import semantic_design
from semantic_refinement import check_inputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixbt import StyleMixtureBT


def within_pair_scores(a, b, y, p, test_fold, n_candidates):
    """Expected scores when the same pair receives a uniformly permuted prompt.

    Canonical orientation keeps outcomes and probabilities aligned. For a given
    target, the donor is marginally uniform on all rows of its pair in its fold.
    This gives exact permutation means for both accuracy and negative log loss.
    The control includes original assignments and reproduces all singleton pairs.
    """
    reverse = a > b
    canonical_p = np.where(reverse, 1 - p, p)
    label = np.where(reverse, 1 - y, y)
    keys = (test_fold * n_candidates + np.minimum(a, b)) * n_candidates + np.maximum(a, b)
    _, inv = np.unique(keys, return_inverse=True)
    sizes = np.bincount(inv)

    def group_mean(values):
        return (np.bincount(inv, weights=values) / sizes)[inv]

    # The fitted router resolves exact ties in favour of the original second candidate.
    win = group_mean(canonical_p > .5)
    tie = group_mean(canonical_p == .5)
    choice = win + tie * reverse
    accuracy = np.where(label == 1, choice, 1 - choice)
    nll = np.where(label == 1, group_mean(losses(canonical_p, 1)), group_mean(losses(canonical_p, 0)))
    return accuracy, nll


def fit_fold(fold, folds, seed, data, embeddings, attributes, reference, components):
    train, test = folds != fold, folds == fold
    G = semantic_design(embeddings, attributes, train, components, seed + fold)
    record = json.loads((reference / f"fold_{fold}.json").read_text())
    config = record["chosen"]["mixture"]
    router = StyleMixtureBT(style=None, n_init=3, seed=seed + fold, max_iter=30, polish_all=True, **config)
    a, b, y = (data[k] for k in ("a", "b", "y"))
    router.fit(G[train], a[train], b[train], y[train], M=data["M"])
    pi_mean = router.gate(G[train]).mean(axis=0)
    conditional = router.predict_proba(G[test], a[test], b[test])
    p_domains = expit(router.R_[a[test]] - router.R_[b[test]])
    averaged = p_domains @ pi_mean
    print(f"Finished {components} coordinates, fold {fold}, K={router.K}", flush=True)
    return test, conditional, averaged


def summarize(data, folds, global_bt, variables, output, components):
    p, fixed = variables.T
    y, groups = data["y"], data["groups"]
    actual = ((p > .5) == y).astype(float), losses(p, y)
    constant = ((fixed > .5) == y).astype(float), losses(fixed, y)
    permutation = within_pair_scores(data["a"], data["b"], y, p, folds, data["M"])
    methods = {"train-mean weights": constant, "within-pair permutation (expected)": permutation}
    rows, per_fold = [], []
    for name, mask in (("all", np.ones(len(y), bool)), ("near-even", np.abs(global_bt - .5) <= .05)):
        for ref, (ac, ll) in methods.items():
            delta = np.column_stack([actual[0] - ac, actual[1] - ll])
            change = delta[mask]
            interval = bootstrap_contrasts(change, groups[mask], 1000)
            rows.append(dict(components=components, stratum=name, control=ref, n=int(mask.sum()),
                             groups=len(np.unique(groups[mask])), prompt_accuracy=actual[0][mask].mean(),
                             control_accuracy=ac[mask].mean(), accuracy_gain=change[:, 0].mean(),
                             gain_lo=interval[0, 0], gain_hi=interval[1, 0], net_correct=change[:, 0].sum(),
                             prompt_logloss=actual[1][mask].mean(), control_logloss=ll[mask].mean(),
                             logloss_change=change[:, 1].mean(), loss_lo=interval[0, 1], loss_hi=interval[1, 1]))
            for f in np.unique(folds):
                use = mask & (folds == f)
                per_fold.append(dict(components=components, stratum=name, control=ref, fold=f, n=int(use.sum()),
                                     prompt_accuracy=actual[0][use].mean(), control_accuracy=ac[use].mean(),
                                     accuracy_gain=delta[use, 0].mean(), logloss_change=delta[use, 1].mean()))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"contrasts_pca{components}.csv", index=False)
    pd.DataFrame(per_fold).to_csv(output / f"folds_pca{components}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def run(args):
    data = load_data()
    with np.load(args.reference / "oof_predictions.npz") as base, np.load(args.embeddings) as cache:
        embeddings, folds, seed = check_inputs(data, base, cache, args.reference)
        if cache["max_tokens"].item() != 128:
            raise ValueError("Expected the frozen 128-token encoder")
        global_bt = base["global BT"]
    attributes = categories(data["text"]).to_numpy()
    for components in args.components:
        values = np.full((len(folds), 2), np.nan)
        folder = "semantic_mixture_capacity" if components == 11 else "semantic_mixture"
        matched_file = args.mixture_results / folder / "local_oof.npz"
        def process(fold):
            return fit_fold(fold, folds, seed, data, embeddings, attributes, args.reference, components)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for test, conditional, fixed in pool.map(process, np.unique(folds)):
                values[test] = np.column_stack([conditional, fixed])
        if not np.isfinite(values).all():
            raise ValueError("Missing predictions")
        if matched_file.is_file():
            with np.load(matched_file) as archive:
                if not np.array_equal(data["ids"], archive["ids"]) or not np.array_equal(folds, archive["fold"]):
                    raise ValueError("Reference rows or folds do not match")
                ref = archive["semantic_mixture"]
            distance = np.max(np.abs(values[:, 0] - ref))
            print(f"Largest probability difference from archived router: {distance:.3g}", flush=True)
            if not np.array_equal(values[:, 0] > .5, ref > .5) or distance > 1e-6:
                raise ValueError("Reconstructed fit differs from the archived semantic router")
        summarize(data, folds, global_bt, values, args.output, components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/nested")
    parser.add_argument("--mixture-results", type=Path, default=ROOT / "results",
                        help="Directory containing the optional semantic_mixture*/local_oof.npz parity checks")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/semantic_minilm.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prompt_weight_control")
    parser.add_argument("--components", type=int, nargs="+", default=[11, 32], choices=[11, 32])
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args)
