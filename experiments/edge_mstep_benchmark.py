"""Matched timing of one Arena skill M-step with vote rows or edge counts.

Run python -s experiments/edge_mstep_benchmark.py after preparing arena_slim.csv.
Both solvers receive the same posterior responsibilities, starting skills,
global ridge prior, 30-step Newton cap and 1e-8 gradient tolerance. Timings
include aggregation, exclude the E-step, and separate one-time edge indexing.
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
from mixbt.extended import StyleMixtureBT


def run_case(data, gate, train_rows, test_rows, case, repeats):
    K = 6  # Selected mixture size in corrected outer fold zero.
    a, b, y = (data[key][train_rows] for key in ("a", "b", "y"))
    F = gate[train_rows]
    model = StyleMixtureBT(K=K, style=None, lam_R=50.0, lam_W=3.0)
    model.M = data["M"]
    bd = model._bd(a, b, y, None)
    model._fit_prior(bd)
    rng = np.random.default_rng(53)
    W = rng.normal(0, 1 / np.sqrt(F.shape[1]), size=(K, F.shape[1]))
    W[-1] = 0.0
    R = model.prior_[0][:, None] + rng.normal(0, 0.3, size=(bd.M, K))
    R -= R.mean(axis=0)
    B = np.empty((K, 0))
    gamma = model._estep(F, bd, W, R, B)[0]
    start = perf_counter()
    pairs = bd.pairs
    index_seconds = perf_counter() - start
    rows = []
    for rep in range(repeats):
        times, skills = {}, {}
        for aggregate in ((False, True) if rep % 2 == 0 else (True, False)):
            model.aggregate_pairs = aggregate
            start = perf_counter()
            skills[aggregate], _ = model._mstep_pref(bd, gamma, R, B)
            times[aggregate] = perf_counter() - start
        predictions, objectives = {}, {}
        for aggregate, fitted in skills.items():
            model.W_, model.R_, model.B_ = W, fitted, B
            predictions[aggregate] = model.predict_proba(
                gate[test_rows], data["a"][test_rows], data["b"][test_rows]
            )
            objectives[aggregate] = model.objective(F, a, b, y)
        rows.append(dict(
            case=case, repetition=rep + 1, M=bd.M, K=K, votes=bd.n, edges=pairs.L,
            index_seconds=index_seconds, row_seconds=times[False], edge_seconds=times[True],
            speedup=times[False] / times[True],
            max_skill_difference=np.abs(skills[False] - skills[True]).max(),
            objective_difference=abs(objectives[True] - objectives[False]),
            max_prediction_difference=np.abs(predictions[True] - predictions[False]).max(),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "edge_mstep_benchmark.csv")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with threadpool_limits(limits=1):
        data = load_data()
        folds = grouped_folds(data["groups"], seed=1729)
        train, test = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
        features = PromptFeatures(seed=1729).fit(data["text"][train])
        gate = features.transform(data["text"])["G"]
        a, b = data["a"][train], data["b"][train]
        keys = np.minimum(a, b) * data["M"] + np.maximum(a, b)
        _, first = np.unique(keys, return_index=True)
        rows = run_case(data, gate, train, test, "all training votes", args.repeats)
        rows += run_case(data, gate, train[np.sort(first)], test, "one vote per pair", args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for label in dict.fromkeys(r["case"] for r in rows):
        selected = [r for r in rows if r["case"] == label]
        print(label, "median paired speedup", np.median([r["speedup"] for r in selected]),
              "max absolute skill difference", max(r["max_skill_difference"] for r in selected))


if __name__ == "__main__":
    main()
