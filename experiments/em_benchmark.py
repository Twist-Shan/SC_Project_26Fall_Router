"""Benchmark of EM variants (run on an otherwise idle machine for fair timings).

B1  simulations: separation delta x 3 replications, common random-gate start;
    EM, SQUAREM, ECME, ECME+SQUAREM, direct L-BFGS
B2  Arena (prompt-only model, K = 3, training+validation split, common k-means
    start as in the first analysis): objective traces of the same algorithms
B3  initialisation: short-run EM screening ("emEM") vs plain random starts on
    the K = 5, delta = 0.5 data set of the initialisation study

Usage: python experiments/em_benchmark.py [B1 B2 B3]
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mixbt import MixtureBT, align_labels, make_params, permute_params, simulate  # noqa: E402
from mixbt.model import PairData  # noqa: E402

RES = os.path.join(HERE, "..", "results")
METHODS = [("EM", None), ("SQUAREM", "squarem"), ("ECME", "ecme"), ("ECME+SQUAREM", "ecme-squarem"),
           ("L-BFGS (direct)", "direct")]


def run_methods(F, a, b, y, M, W0, R0, lam, max_iter=3000, tol=1e-10, R_true=None, shrink="zero"):
    rows, traces = [], {}
    for name, acc in METHODS:
        m = MixtureBT(K=W0.shape[0], lam_W=lam[0], lam_R=lam[1], tol=tol, max_iter=max_iter,
                      accel=None if acc == "direct" else acc, shrink_to=shrink, w_maxiter=30)
        t = time.perf_counter()
        if acc == "direct":
            m.fit_direct(F, a, b, y, W0, R0, tol=1e-14)
        else:
            m.fit(F, a, b, y, M=M, W_init=W0, R_init=R0)
        el = time.perf_counter() - t
        obj = m.loglik(F, a, b, y) - m._penalty(m.W_, m.R_)
        row = dict(method=name, time=el, obj=obj, iters=len(m.history_.objective) - 1)
        if R_true is not None:
            perm = align_labels(m.R_, R_true)
            row["errR"] = np.linalg.norm(m.R_[:, perm] - R_true) / np.linalg.norm(R_true)
        rows.append(row)
        traces[name] = (np.array(m.history_.time), np.array(m.history_.objective))
    return rows, traces


def b1(reps=3):
    rows = []
    for delta in [0.15, 0.3, 0.5, 1.0, 2.0]:
        for rep in range(reps):
            W, R = make_params(10, 3, 6, delta=delta, gate_scale=1.5, rng=3000 + rep)
            d = simulate(20000, W, R, rng=4000 + rep)
            base = MixtureBT(K=3, lam_W=1e-3, lam_R=1e-3, init="random", seed=rep)
            base._set_prior(d["a"], d["b"], d["y"], 10)
            W0, R0 = base._initial_params(d["F"], d["a"], d["b"], d["y"], 10, PairData(d["a"], d["b"], d["y"], 10),
                                          np.random.default_rng(rep), 0)
            r, _ = run_methods(d["F"], d["a"], d["b"], d["y"], 10, W0, R0, (1e-3, 1e-3), R_true=R)
            for x in r:
                x.update(delta=delta, rep=rep)
            rows += r
            print(delta, rep, [(x["method"], round(x["time"], 1), round(x["obj"], 3)) for x in r], flush=True)
    df = pd.DataFrame(rows)
    df["gap"] = df.groupby(["delta", "rep"])["obj"].transform("max") - df["obj"]
    df.to_csv(os.path.join(RES, "em_bench_sim.csv"), index=False)
    print(df.groupby(["method", "delta"])[["time", "iters", "gap", "errR"]].median().round(3))


def b2():
    from arena_analysis import EMKW, LAM_R, LAM_W, load
    D, meta = load()
    tr, va = D["train"], D["val"]
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    M = D["M"]
    base = MixtureBT(K=3, init="kmeans", seed=0, lam_W=LAM_W, lam_R=LAM_R, shrink_to="global")
    base._set_prior(tv["a"], tv["b"], tv["y"], M)
    W0, R0 = base._initial_params(tv["F"], tv["a"], tv["b"], tv["y"], M, PairData(tv["a"], tv["b"], tv["y"], M),
                                  np.random.default_rng(0), 0)
    rows, traces = run_methods(tv["F"], tv["a"], tv["b"], tv["y"], M, W0, R0, (LAM_W, LAM_R), max_iter=3000,
                               shrink="global")
    df = pd.DataFrame(rows)
    df["gap"] = df["obj"].max() - df["obj"]
    df.to_csv(os.path.join(RES, "em_bench_arena.csv"), index=False)
    np.savez(os.path.join(RES, "em_bench_arena_traces.npz"),
             **{f"{k}__t": v[0] for k, v in traces.items()}, **{f"{k}__obj": v[1] for k, v in traces.items()})
    print(df.round(3))


def b3(seeds=10):
    W, R = make_params(10, 5, 6, delta=0.5, gate_scale=1.5, rng=5005)
    d = simulate(20000, W, R, rng=6005)
    s3 = pd.read_csv(os.path.join(RES, "sim_S3S7_init.csv"))
    best = s3[s3.setting == "K=5, delta=0.5"].obj.max()
    rows = []
    for init, kw in [("random start + SQUAREM", dict(init="random", n_init=1, accel="squarem")),
                     ("random start + ECME/SQUAREM", dict(init="random", n_init=1, accel="ecme-squarem")),
                     ("emEM + SQUAREM", dict(init="emEM", n_init=1, n_short=20, short_iter=15, accel="squarem")),
                     ("emEM + ECME/SQUAREM", dict(init="emEM", n_init=1, n_short=20, short_iter=15,
                                                  accel="ecme-squarem"))]:
        for s in range(seeds):
            m = MixtureBT(K=5, lam_W=1e-3, lam_R=1e-3, tol=1e-8, max_iter=3000, seed=20_000 + s, **kw)
            t = time.perf_counter()
            m.fit(d["F"], d["a"], d["b"], d["y"], M=10)
            el = time.perf_counter() - t
            obj = m.loglik(d["F"], d["a"], d["b"], d["y"]) - m._penalty(m.W_, m.R_)
            rows.append(dict(init=init, seed=s, time=el, obj=obj))
            print(init, s, round(el, 1), round(obj, 3), flush=True)
    df = pd.DataFrame(rows)
    best = max(best, df.obj.max())
    df["gap"] = best - df.obj
    df.to_csv(os.path.join(RES, "em_bench_init.csv"), index=False)
    print(df.groupby("init").agg(share_at_best=("gap", lambda g: np.mean(g < 0.5)), median_gap=("gap", "median"),
                                 mean_time=("time", "mean")))


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["B1", "B2", "B3"]):
        t = time.time()
        print("=====", w, flush=True)
        globals()[w.lower()]()
        print(f"[{w} done in {time.time() - t:.0f}s]", flush=True)
