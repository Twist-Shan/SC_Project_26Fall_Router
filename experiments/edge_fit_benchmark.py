"""Time complete prompt-only fits from matched starts on corrected Arena fold 0.

Run python -s experiments/edge_fit_benchmark.py after preparing arena_slim.csv.
The default uses the K=6, lam_W=3, lam_R=50 configuration selected in fold 0.
Use --components 2 for the prespecified smaller comparison. Timing includes the
global prior, EM and the observed-likelihood L-BFGS refinement. Prompt
preprocessing and generation of the common starting parameters are shared.
"""
import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import PromptFeatures, grouped_folds, load_data
from mixbt import StyleMixtureBT


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, choices=[2, 6], default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_name = "edge_fit_benchmark_k6.csv" if args.components == 6 else "edge_fit_benchmark.csv"
    output = args.output or ROOT / "results" / default_name
    data = load_data()
    train = grouped_folds(data["groups"], seed=1729) != 0
    test = ~train
    gate = PromptFeatures(seed=1729).fit(data["text"][train]).transform(data["text"])["G"]
    F, held = gate[train], gate[test]
    a, b, y = (data[k][train] for k in ("a", "b", "y"))
    cfg = dict(K=args.components, style=None, lam_R=50., lam_W=3., n_init=1, tol=1e-7, max_iter=30,
               polish_all=True)
    model = StyleMixtureBT(**cfg, aggregate_pairs=False)
    model.M = data["M"]
    bd = model._bd(a, b, y, None)
    model._fit_prior(bd)
    start = model._init(F, bd, np.random.default_rng(53))
    rows = []
    for rep in range(3):
        fits, seconds, predictions = {}, {}, {}
        for aggregate in ((False, True) if rep % 2 == 0 else (True, False)):
            model = StyleMixtureBT(**cfg, aggregate_pairs=aggregate)
            t0 = perf_counter()
            fits[aggregate] = model.fit(F, a, b, y, M=bd.M, init_params=start)
            seconds[aggregate] = perf_counter() - t0
            predictions[aggregate] = model.predict_proba(held, data["a"][test], data["b"][test])
        row_fit, edge_fit = fits[False], fits[True]
        rows.append(dict(repetition=rep + 1, n_train=len(a), n_test=len(held), K=cfg["K"],
                         row_seconds=seconds[False], edge_seconds=seconds[True],
                         paired_speedup=seconds[False] / seconds[True],
                         row_em_iterations=row_fit.n_iter_em_, edge_em_iterations=edge_fit.n_iter_em_,
                         row_polish_iterations=row_fit.n_iter_polish_,
                         edge_polish_iterations=edge_fit.n_iter_polish_,
                         row_objective=row_fit.objective(F, a, b, y), edge_objective=edge_fit.objective(F, a, b, y),
                         max_prediction_difference=np.abs(predictions[True] - predictions[False]).max()))
        print(rows[-1], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
