"""Common-budget Arena solve using accepted iterates of the existing fitters.

Fixed design from arena_prepare.py and em_benchmark.py B2, including train/val
row order. The target is the best archived B2 objective before this experiment.
It is a feasible reference on the supplied design, not a known global optimum.
Time starts after forming the common initial parameters; model setup and both
phases of the hybrid are charged. One thread, 3 starts, 3 nested CPU budgets.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
from unittest.mock import patch

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env_var] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixbt.model import FitHistory, MixtureBT, PairData

BUDGETS = (5, 10, 20)
SEEDS = (0, 17, 53)
METHODS = ("EM", "SQUAREM", "ECME", "ECME+SQUAREM", "L-BFGS", "30 EM + L-BFGS")
ACCELERATION = {"SQUAREM": "squarem", "ECME": "ecme", "ECME+SQUAREM": "ecme-squarem"}
REFERENCE = -21952.193679148408
HISTORICAL_START = -22092.73832818427
PER_VOTE_GAPS = (1e-3, 1e-4)


def load_problem(path):
    with np.load(path, allow_pickle=False) as data:
        split, y = data["split"], data["y"]
        idx = np.concatenate([np.flatnonzero((split == s) & (y != 0.5)) for s in ("train", "val")])
        F, a, b, y = (data[name][idx] for name in ("F", "a", "b", "y"))
    if F.shape != (35756, 39):
        raise ValueError(f"The archived objective requires the 35756 x 39 B2 design; got {F.shape}.")
    return F, a, b, y


def array_hash(*arrays):
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(str((arr.shape, arr.dtype)).encode())
        digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()


class EndBudget(Exception):
    pass


class Checkpoints(list):
    """Observe the solver's accepted-objective history without changing its updates."""

    def __init__(self, trace, cpu_start, wall_start):
        super().__init__()
        self.trace = trace
        self.cpu_start = cpu_start
        self.wall_start = wall_start

    def append(self, obj):
        cpu = process_time() - self.cpu_start
        if cpu > max(BUDGETS):
            raise EndBudget
        self.trace.append((cpu, perf_counter() - self.wall_start, float(obj)))
        super().append(obj)


def run_method(method, start, problem, M, initial):
    F, a, b, y = problem
    W, R = start
    model = MixtureBT(K=3, lam_W=3, lam_R=50, shrink_to="global", w_maxiter=30,
                      accel=ACCELERATION.get(method), tol=1e-10,
                      max_iter=30 if method == "30 EM + L-BFGS" else 3000)
    trace = [(0.0, 0.0, initial)]
    cpu_start, wall_start = process_time(), perf_counter()

    def timed_history():
        return FitHistory(objective=Checkpoints(trace, cpu_start, wall_start))

    stopped = "solver_returned"
    with patch("mixbt.model.FitHistory", timed_history):
        try:
            if method != "L-BFGS":
                model.fit(F, a, b, y, M=M, W_init=W, R_init=R)
                W, R = model.W_, model.R_
            if method in ("L-BFGS", "30 EM + L-BFGS"):
                model.fit_direct(F, a, b, y, W, R, tol=1e-14)
        except EndBudget:
            stopped = "cpu_budget"
    if len(trace) > 1 and not np.isclose(trace[1][2], initial, rtol=0, atol=1e-6):
        raise RuntimeError("Solver did not start at the supplied penalised objective.")
    return np.asarray(trace), stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/arena_processed.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/solver_budget")
    args = parser.parse_args()
    problem = load_problem(args.input)
    design_sha = array_hash(*problem)
    F, a, b, y = problem
    n, M = len(F), int(max(a.max(), b.max())) + 1
    assert M == 64
    summary, targets, checkpoints = [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for r, seed in enumerate(SEEDS):
        base = MixtureBT(K=3, lam_W=3, lam_R=50, init="kmeans", seed=seed, shrink_to="global")
        base._set_prior(a, b, y, M)
        W, R = base._initial_params(F, a, b, y, M, PairData(a, b, y, M),
                                    np.random.default_rng(seed), 0)
        initial = base.loglik(F, a, b, y, W, R) - base._penalty(W, R)
        if seed == 0 and abs(initial - HISTORICAL_START) > 1e-3:
            raise ValueError("The seed-zero initial objective does not reproduce the archived B2 design. "
                             "Check the input arrays and the pinned preprocessing environment.")
        meta = dict(seed=seed, n=n, p=F.shape[1], K=3, design_sha256=design_sha,
                    start_sha256=array_hash(W, R), start_objective=initial)
        # Rotate the order between starts so a method does not always run first.
        for method in METHODS[2*r:] + METHODS[:2*r]:
            trace, stopped = run_method(method, (W, R), problem, M, initial)
            cpu, wall, obj = trace.T
            for budget in BUDGETS:
                best = obj[cpu <= budget].max()
                summary.append(dict(**meta, method=method, budget_cpu_s=budget, best_objective=best,
                                    gain_over_start=best-initial, reference_gap=REFERENCE-best,
                                    termination=stopped))
            for gap in PER_VOTE_GAPS:
                threshold = REFERENCE - gap*n
                hits = np.flatnonzero(obj >= threshold)
                first = hits[0] if len(hits) else None
                targets.append(dict(seed=seed, method=method, per_vote_gap=gap,
                                    threshold=threshold, cpu_budget_s=max(BUDGETS),
                                    hit=first is not None,
                                    cpu_seconds=cpu[first] if first is not None else np.nan,
                                    wall_seconds=wall[first] if first is not None else np.nan))
            checkpoints.extend(dict(seed=seed, method=method, accepted_index=i, cpu_s=c, wall_s=w,
                                    objective=v) for i, (c, w, v) in enumerate(trace))
            print(seed, method, stopped, f"reference gap {REFERENCE-obj.max():.2f}", flush=True)
        for name, rows in (("budgets", summary), ("targets", targets), ("checkpoints", checkpoints)):
            pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
