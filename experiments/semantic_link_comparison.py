"""Compare two compositions of the same 29 features and BT skills on Arena.

The five folds, K, penalties, three initial points and shrinkage centre match
the 11-coordinate semantic EM experiment. The scalar model mixes skills before
the logistic link. Each model selects its start by its own training objective.
Run after semantic_mixture.py --components 11. Only aggregates are published.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import categories, load_data
from lexical_refinement import bootstrap_contrasts, losses
from mixbt import StyleMixtureBT
from mixbt.baselines import CapabilityBT
from semantic_mixture import semantic_design
from semantic_refinement import check_inputs


def scalar_fit(config, train_design, a, b, y, count, seed):
    initial = StyleMixtureBT(style=None, seed=seed, **config)
    initial.M = count
    battles = initial._bd(a, b, y, None)
    initial.q = battles.q
    initial._fit_prior(battles)
    model = CapabilityBT(**config)
    model.prior_ = initial.prior_[0]
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(3):
        weights, skills, _ = initial._init(train_design, battles, rng)
        start = np.r_[weights[:-1].ravel(), skills.ravel()]
        result = minimize(model._f_and_g, start, args=(train_design, a, b, y, count),
                          jac=True, method="L-BFGS-B",
                          options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Scalar fit failed: {result.message}")
        starts.append(result)
    best = min(starts, key=lambda opt: opt.fun)
    model.W_, model.R_ = model._unpack(best.x, train_design.shape[1], count)
    return model, starts


def main(reference):
    data = load_data()
    with (np.load(reference / "oof_predictions.npz") as base,
          np.load(ROOT / "data/semantic_minilm.npz") as cache,
          np.load(ROOT / "results/semantic_mixture_capacity/local_oof.npz") as mixture):
        embeddings, folds, seed = check_inputs(data, base, cache, reference)
        if (int(cache["max_tokens"]) != 128 or not np.array_equal(mixture["ids"], data["ids"])
                or not np.array_equal(mixture["fold"], folds)):
            raise ValueError("The semantic controls must use identical rows and fold assignments")
        mix_p = mixture["semantic_mixture"]
    attributes = categories(data["text"]).to_numpy()
    scalar_p = np.full_like(mix_p, np.nan)
    runs = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        config = json.loads((reference / f"fold_{fold}.json").read_text())["chosen"]["mixture"]
        design = semantic_design(embeddings, attributes, train, 11, seed + fold)
        model, starts = scalar_fit(config, design[train], data["a"][train], data["b"][train],
                                   data["y"][train], data["M"], seed + fold)
        scalar_p[test] = model.predict_proba(design[test], data["a"][test], data["b"][test])
        runs.extend(dict(fold=fold, start=j, objective=-r.fun, iterations=r.nit,
                         grad_max=np.max(np.abs(r.jac)), **config) for j, r in enumerate(starts))
        print(f"Completed scalar fit on fold {fold}", flush=True)
    if not np.all(np.isfinite(scalar_p)):
        raise ValueError("Missing scalar predictions")
    y = data["y"]
    correct = np.column_stack([(p > .5) == y for p in (mix_p, scalar_p)]).astype(float)
    loss = np.column_stack([losses(p, y) for p in (mix_p, scalar_p)])
    deltas = np.column_stack([correct[:, 0] - correct[:, 1], loss[:, 0] - loss[:, 1]])
    intervals = bootstrap_contrasts(deltas, data["groups"], 1000)
    measures = pd.DataFrame(dict(measure=["accuracy", "logloss"],
                                 probability_mixture=[correct[:, 0].mean(), loss[:, 0].mean()],
                                 scalar_score=[correct[:, 1].mean(), loss[:, 1].mean()],
                                 difference=deltas.mean(0), lo=intervals[0], hi=intervals[1]))
    output = ROOT / "results/semantic_link"
    output.mkdir(parents=True, exist_ok=True)
    measures.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(runs).to_csv(output / "fits.csv", index=False)
    print(measures.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/nested",
                        help="Directory containing the original grouped-fold predictions and configurations.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.reference)
