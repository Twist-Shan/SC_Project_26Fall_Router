"""Simulation studies for the mixture Bradley--Terry model.

Usage:  python experiments/sim_experiments.py [S1 S2 ...]   (default: all)
Results are written to results/sim_*.csv; figures come from sim_plots.py.

S1  consistency: estimation error and routing regret vs n, against
    a global BT ranking and a cluster-then-BT two-stage baseline
S2  separation (Condition E1) vs EM convergence speed; EM vs SQUAREM vs
    direct L-BFGS
S3  initialisation and local optima (Condition E6)
S4  choosing K (BIC and held-out log-likelihood)
S5  formulation A (mixture) vs formulation B (aggregated capabilities)
S6  coverage of Wald intervals from the observed information
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.special import expit, softmax  # noqa: E402

from mixbt import (CapabilityBT, ClusterBT, GlobalBT, MixtureBT, align_labels,  # noqa: E402
                   logloss, make_params, permute_params, simulate, true_proba)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
N_JOBS = int(os.environ.get("N_JOBS", "2"))

# default design
M, K, P = 10, 3, 6
DELTA, GATE = 1.0, 1.5
LAM = 1e-3
EMKW = dict(lam_W=LAM, lam_R=LAM, tol=1e-8, max_iter=3000, accel="squarem")


def true_scores(F, W, R):
    """S*_m(x) under the true parameters (all M models as candidates)."""
    Mm = R.shape[0]
    P_ = expit(R[:, None, :] - R[None, :, :])
    P_[np.arange(Mm), np.arange(Mm)] = 0
    v = P_.sum(1) / (Mm - 1)
    return softmax(F @ W.T, axis=1) @ v.T  # n x M


def routing_metrics(choice, S_true):
    best = S_true.max(1)
    got = S_true[np.arange(len(choice)), choice]
    return dict(regret=float(np.mean(best - got)),
                best_pick=float(np.mean(choice == S_true.argmax(1))))


# ----------------------------------------------------------------------- S1
def s1_one(n, rep):
    W, R = make_params(M, K, P, delta=DELTA, gate_scale=GATE, rng=1000 + rep)
    tr = simulate(n, W, R, rng=2000 + 7 * rep + n)
    te = simulate(20000, W, R, rng=999)
    p_or = true_proba(te["F"], te["a"], te["b"], W, R)
    ll_or = logloss(p_or, te["y"])
    S_true = true_scores(te["F"], W, R)
    rows = []

    t = time.perf_counter()
    m = MixtureBT(K=K, n_init=3, seed=rep, **EMKW).fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M)
    t_fit = time.perf_counter() - t
    perm = align_labels(m.R_, R)
    Wh, Rh = permute_params(m.W_, m.R_, perm)
    p = m.predict_proba(te["F"], te["a"], te["b"])
    choice, _ = m.route(te["F"])
    rows.append(dict(method="Mixture EM", n=n, rep=rep, time=t_fit,
                     errR=np.linalg.norm(Rh - R) / np.linalg.norm(R),
                     errW=np.linalg.norm(Wh - W) / np.linalg.norm(W),
                     prob_mae=np.mean(np.abs(p - p_or)), excess_ll=logloss(p, te["y"]) - ll_or,
                     **routing_metrics(choice, S_true)))

    g = GlobalBT(lam_R=LAM).fit(tr["a"], tr["b"], tr["y"], M)
    p = g.predict_proba(te["F"], te["a"], te["b"])
    choice = np.full(len(te["a"]), np.argmax(g.r_))
    rows.append(dict(method="Global BT", n=n, rep=rep, prob_mae=np.mean(np.abs(p - p_or)),
                     excess_ll=logloss(p, te["y"]) - ll_or, **routing_metrics(choice, S_true)))

    c = ClusterBT(K=K, lam_R=LAM, seed=rep).fit(tr["F"], tr["a"], tr["b"], tr["y"], M)
    p = c.predict_proba(te["F"], te["a"], te["b"])
    lab = c.km_.predict(te["F"][:, 1:])
    choice = np.argmax(c.R_[:, lab], axis=0)
    rows.append(dict(method="Cluster-then-BT", n=n, rep=rep, prob_mae=np.mean(np.abs(p - p_or)),
                     excess_ll=logloss(p, te["y"]) - ll_or, **routing_metrics(choice, S_true)))
    return rows


def run_s1(reps=8):
    ns = [1000, 2000, 5000, 10000, 20000, 50000]
    jobs = [(n, r) for n in ns for r in range(reps)]
    res = Parallel(N_JOBS)(delayed(s1_one)(n, r) for n, r in jobs)
    df = pd.DataFrame([row for rows in res for row in rows])
    df.to_csv(os.path.join(OUT, "sim_S1_consistency.csv"), index=False)
    print(df.groupby(["method", "n"])[["errR", "prob_mae", "excess_ll", "regret", "best_pick"]].mean())


# ----------------------------------------------------------------------- S2
def tail_rate(obj, burn=0.5):
    """Empirical linear rate of EM: median ratio of successive increments."""
    inc = np.diff(np.asarray(obj))
    inc = inc[int(len(inc) * burn):]
    inc = inc[inc > 1e-12]
    if len(inc) < 3:
        return np.nan
    return float(np.median(inc[1:] / inc[:-1]))


def s2_one(delta, rep, gate=GATE):
    n = 20000
    W, R = make_params(M, K, P, delta=delta, gate_scale=gate, rng=3000 + rep)
    tr = simulate(n, W, R, rng=4000 + rep)
    base = MixtureBT(K=K, lam_W=LAM, lam_R=LAM, init="kmeans", seed=rep)
    # common starting point: k-means responsibilities -> one M-step
    from mixbt.model import PairData
    pdat = PairData(tr["a"], tr["b"], tr["y"], M)
    W0, R0 = base._initial_params(tr["F"], tr["a"], tr["b"], tr["y"], M, pdat,
                                  np.random.default_rng(rep), 0)
    rows = []
    for name, acc in [("EM", None), ("SQUAREM", "squarem")]:
        m = MixtureBT(K=K, lam_W=LAM, lam_R=LAM, tol=1e-9, max_iter=10000, accel=acc)
        t = time.perf_counter()
        m.fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M, W_init=W0, R_init=R0)
        el = time.perf_counter() - t
        perm = align_labels(m.R_, R)
        _, Rh = permute_params(m.W_, m.R_, perm)
        rows.append(dict(method=name, delta=delta, gate=gate, rep=rep, iters=m.n_iter_ - 1,
                         map_evals=m.n_mstep_, time=el, obj=m.history_.objective[-1],
                         rate=tail_rate(m.history_.objective) if acc is None else np.nan,
                         errR=np.linalg.norm(Rh - R) / np.linalg.norm(R)))
        if acc is None:
            trace = np.array(m.history_.objective)
    m = MixtureBT(K=K, lam_W=LAM, lam_R=LAM)
    t = time.perf_counter()
    m.fit_direct(tr["F"], tr["a"], tr["b"], tr["y"], W0, R0, tol=1e-12)
    el = time.perf_counter() - t
    perm = align_labels(m.R_, R)
    _, Rh = permute_params(m.W_, m.R_, perm)
    rows.append(dict(method="L-BFGS (direct)", delta=delta, gate=gate, rep=rep, iters=m.n_iter_,
                     map_evals=np.nan, time=el, obj=m.history_.objective[-1], rate=np.nan,
                     errR=np.linalg.norm(Rh - R) / np.linalg.norm(R)))
    return rows, (delta, rep, trace)


def run_s2(reps=5):
    deltas = [0.15, 0.3, 0.5, 1.0, 2.0]
    res = Parallel(N_JOBS)(delayed(s2_one)(d, r) for d in deltas for r in range(reps))
    df = pd.DataFrame([row for rows, _ in res for row in rows])
    df.to_csv(os.path.join(OUT, "sim_S2_speed.csv"), index=False)
    traces = {f"{d}_{r}": tr for _, (d, r, tr) in res}
    np.savez_compressed(os.path.join(OUT, "sim_S2_traces.npz"), **traces)
    print(df.groupby(["method", "delta"])[["iters", "time", "obj", "rate", "errR"]].median())


# ----------------------------------------------------------------------- S3
def s3_one(init, start, W, R, tr):
    m = MixtureBT(K=R.shape[1], init=init, n_init=1, seed=10_000 + start, **EMKW)
    m.fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M)
    perm = align_labels(m.R_, R)
    _, Rh = permute_params(m.W_, m.R_, perm)
    return dict(init=init, start=start, obj=m.history_.objective[-1], iters=m.n_iter_,
                errR=np.linalg.norm(Rh - R) / np.linalg.norm(R))


def run_s3(starts=30):
    rows = []
    for setting, (Kt, delta) in {"K=3, delta=0.5": (3, 0.5), "K=5, delta=0.5": (5, 0.5)}.items():
        W, R = make_params(M, Kt, P, delta=delta, gate_scale=GATE, rng=5000 + Kt)
        tr = simulate(20000, W, R, rng=6000 + Kt)
        res = Parallel(N_JOBS)(delayed(s3_one)(init, s, W, R, tr)
                               for init in ["kmeans", "random"] for s in range(starts))
        for r in res:
            r["setting"] = setting
        rows += res
    df = pd.DataFrame(rows)
    df["gap"] = df.groupby("setting")["obj"].transform("max") - df["obj"]
    df.to_csv(os.path.join(OUT, "sim_S3_init.csv"), index=False)
    print(df.groupby(["setting", "init"])[["gap", "errR", "iters"]].describe().T)


# ----------------------------------------------------------------------- S7
def s7_one(start, W, R, tr, anneal):
    m = MixtureBT(K=R.shape[1], init="random", n_init=1, seed=10_000 + start, anneal=anneal, **EMKW)
    m.fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M)
    perm = align_labels(m.R_, R)
    _, Rh = permute_params(m.W_, m.R_, perm)
    return dict(init="random + DAEM", start=start, obj=m.history_.objective[-1], iters=m.n_iter_,
                errR=np.linalg.norm(Rh - R) / np.linalg.norm(R))


def run_s7(starts=30, anneal=(0.5, 0.7, 0.85)):
    """Deterministic annealing EM on exactly the data sets of S3."""
    rows = []
    for setting, (Kt, delta) in {"K=3, delta=0.5": (3, 0.5), "K=5, delta=0.5": (5, 0.5)}.items():
        W, R = make_params(M, Kt, P, delta=delta, gate_scale=GATE, rng=5000 + Kt)
        tr = simulate(20000, W, R, rng=6000 + Kt)
        res = Parallel(N_JOBS)(delayed(s7_one)(s, W, R, tr, list(anneal)) for s in range(starts))
        for r in res:
            r["setting"] = setting
        rows += res
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sim_S7_daem.csv"), index=False)
    s3 = pd.read_csv(os.path.join(OUT, "sim_S3_init.csv"))
    both = pd.concat([s3.drop(columns=["gap"]), df], ignore_index=True)
    both["gap"] = both.groupby("setting")["obj"].transform("max") - both["obj"]
    both.to_csv(os.path.join(OUT, "sim_S3S7_init.csv"), index=False)
    print(both.assign(at_best=both.gap < 0.5).groupby(["setting", "init"]).agg(
        share_at_best=("at_best", "mean"), median_gap=("gap", "median"), median_errR=("errR", "median")))


# ----------------------------------------------------------------------- S4
def s4_one(n, rep):
    W, R = make_params(M, K, P, delta=DELTA, gate_scale=GATE, rng=7000 + rep)
    tr = simulate(n, W, R, rng=8000 + rep)
    va = simulate(n // 4, W, R, rng=9000 + rep)
    rows = []
    for Kf in range(1, 7):
        m = MixtureBT(K=Kf, n_init=3, seed=rep, **EMKW).fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M)
        rows.append(dict(n=n, rep=rep, K=Kf, bic=m.bic(tr["F"], tr["a"], tr["b"], tr["y"]),
                         val_ll=m.loglik(va["F"], va["a"], va["b"], va["y"]) / len(va["a"])))
    return rows


def run_s4(reps=10):
    res = Parallel(N_JOBS)(delayed(s4_one)(n, r) for n in [2000, 5000, 20000] for r in range(reps))
    df = pd.DataFrame([row for rows in res for row in rows])
    df.to_csv(os.path.join(OUT, "sim_S4_selectK.csv"), index=False)
    pick_bic = df.loc[df.groupby(["n", "rep"])["bic"].idxmin()]
    pick_val = df.loc[df.groupby(["n", "rep"])["val_ll"].idxmax()]
    print("BIC choice\n", pd.crosstab(pick_bic["n"], pick_bic["K"]))
    print("validation choice\n", pd.crosstab(pick_val["n"], pick_val["K"]))


# ----------------------------------------------------------------------- S5
def s5_one(dgp, rep):
    n = 20000
    W, R = make_params(M, K, P, delta=DELTA, gate_scale=GATE, rng=11000 + rep)
    if dgp == "B":
        R = R * 1.5  # comparable signal strength
    tr = simulate(n, W, R, rng=12000 + rep, model=dgp)
    te = simulate(20000, W, R, rng=13000 + rep, model=dgp)
    p_or = true_proba(te["F"], te["a"], te["b"], W, R, model=dgp)
    ll_or = logloss(p_or, te["y"])
    rows = []
    mA = MixtureBT(K=K, n_init=3, seed=rep, **EMKW).fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M)
    pA = mA.predict_proba(te["F"], te["a"], te["b"])
    best = None
    for s in range(3):
        mB = CapabilityBT(K=K, lam_W=LAM, lam_R=LAM, seed=s).fit(tr["F"], tr["a"], tr["b"], tr["y"], M)
        if best is None or mB.objective_ > best.objective_:
            best = mB
    pB = best.predict_proba(te["F"], te["a"], te["b"])
    pG = GlobalBT(LAM).fit(tr["a"], tr["b"], tr["y"], M).predict_proba(te["F"], te["a"], te["b"])
    for name, p in [("Fit A (mixture, EM)", pA), ("Fit B (aggregation)", pB), ("Global BT", pG)]:
        rows.append(dict(dgp=f"Data from {dgp}", fit=name, rep=rep,
                         excess_ll=logloss(p, te["y"]) - ll_or, prob_mae=np.mean(np.abs(p - p_or))))
    return rows


def run_s5(reps=6):
    res = Parallel(N_JOBS)(delayed(s5_one)(d, r) for d in ["A", "B"] for r in range(reps))
    df = pd.DataFrame([row for rows in res for row in rows])
    df.to_csv(os.path.join(OUT, "sim_S5_AvsB.csv"), index=False)
    print(df.groupby(["dgp", "fit"])[["excess_ll", "prob_mae"]].agg(["mean", "std"]))


# ----------------------------------------------------------------------- S6
def s6_one(rep, n):
    W, R = make_params(M, K, P, delta=DELTA, gate_scale=GATE, rng=0)  # fixed truth
    tr = simulate(n, W, R, rng=20000 + rep)
    m = MixtureBT(K=K, n_init=2, seed=rep, **EMKW).fit(tr["F"], tr["a"], tr["b"], tr["y"], M=M)
    perm = align_labels(m.R_, R)
    m.W_, m.R_ = permute_params(m.W_, m.R_, perm)
    seW, seR, _ = m.observed_information(tr["F"], tr["a"], tr["b"], tr["y"])
    zR = (m.R_ - R) / seR
    zW = ((m.W_ - W) / np.where(seW > 0, seW, 1))[:-1]
    return dict(rep=rep, n=n, zR=zR.ravel(), zW=zW.ravel(), seR=seR.ravel(),
                errR=(m.R_ - R).ravel())


def run_s6(reps=100):
    rows = []
    for n in [5000, 20000]:
        res = Parallel(N_JOBS)(delayed(s6_one)(r, n) for r in range(reps))
        zR = np.array([r["zR"] for r in res])
        zW = np.array([r["zW"] for r in res])
        err = np.array([r["errR"] for r in res])
        se = np.array([r["seR"] for r in res])
        np.savez_compressed(os.path.join(OUT, f"sim_S6_coverage_n{n}.npz"), zR=zR, zW=zW,
                            errR=err, seR=se)
        rows.append(dict(n=n, cover_R=np.mean(np.abs(zR) < 1.96), cover_W=np.mean(np.abs(zW) < 1.96),
                         mean_se_R=se.mean(), mc_sd_R=err.std(0).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sim_S6_coverage.csv"), index=False)
    print(df)


if __name__ == "__main__":
    # re-import as a module so that joblib workers can unpickle the jobs
    _here = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join([_here, os.path.join(_here, ".."),
                                                os.environ.get("PYTHONPATH", "")])
    sys.path.insert(0, _here)
    import sim_experiments as S

    which = sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5", "S6"]
    for w in which:
        t = time.time()
        print(f"===== {w}", flush=True)
        getattr(S, f"run_{w.lower()}")()
        print(f"[{w} done in {time.time() - t:.0f}s]", flush=True)
