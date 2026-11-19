"""Chatbot Arena analysis with the mixture Bradley--Terry model.

Usage: python experiments/arena_analysis.py [stage ...]
stages: selectK final interpret diagnostics routing bootstrap ties   (default: all but bootstrap)
Requires data/arena_processed.npz from arena_prepare.py.
"""
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from mixbt import (CapabilityBT, ClusterBT, GlobalBT, MixtureBT, evaluate,  # noqa: E402
                   align_labels, permute_params)
from mixbt import plotstyle as ps  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# penalties chosen on the validation split (results/arena_tune.csv)
LAM_W = float(os.environ.get("LAM_W", "3.0"))
LAM_R = float(os.environ.get("LAM_R", "50.0"))
SHRINK = os.environ.get("SHRINK", "global")
LAM_GLOBAL = 0.1  # ridge of the single global BT fit (baseline and shrinkage target)
K_GRID = list(range(1, 9))
EMKW = dict(lam_W=LAM_W, lam_R=LAM_R, tol=1e-7, max_iter=1000, accel="squarem", w_maxiter=30,
            shrink_to=SHRINK)


# --------------------------------------------------------------------- data
def load(include_ties=False):
    z = np.load(os.path.join(DATA, "arena_processed.npz"), allow_pickle=True)
    with open(os.path.join(DATA, "arena_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    F, a, b, y, split = z["F"], z["a"], z["b"], z["y"], z["split"]
    keep = np.ones(len(y), bool) if include_ties else (y != 0.5)
    out = {}
    for s in ["train", "val", "test"]:
        m = keep & (split == s)
        out[s] = dict(F=F[m], a=a[m], b=b[m], y=y[m], idx=np.where(m)[0])
    out["all"] = dict(F=F, a=a, b=b, y=y)
    out["M"] = len(meta["models"])
    return out, meta


def save_obj(obj, name):
    with open(os.path.join(RES, name), "wb") as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(os.path.join(RES, name), "rb") as f:
        return pickle.load(f)


def connected(a, b, M):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    A = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(M, M))
    return connected_components(A, directed=False)[0]


def fit_mix(K, d, M, n_init=3, seed=0, **kw):
    kws = {**EMKW, **kw}
    return MixtureBT(K=K, n_init=n_init, seed=seed, **kws).fit(d["F"], d["a"], d["b"], d["y"], M=M)


# ------------------------------------------------------------------ stages
def stage_selectK(D, meta):
    tr, va = D["train"], D["val"]
    M = D["M"]
    print("connected components of training comparison graph:", connected(tr["a"], tr["b"], M))
    rows = []
    for K in K_GRID:
        t = time.time()
        m = fit_mix(K, tr, M)
        el = time.time() - t
        p_va = m.predict_proba(va["F"], va["a"], va["b"])
        row = dict(K=K, method="Mixture EM", train_ll=m.loglik_ / len(tr["y"]),
                   val_ll=m.loglik(va["F"], va["a"], va["b"], va["y"]) / len(va["y"]),
                   bic=m.bic(tr["F"], tr["a"], tr["b"], tr["y"]), time=el, iters=m.n_iter_,
                   **{f"val_{k}": v for k, v in evaluate(p_va, va["y"]).items()})
        rows.append(row)
        print(row, flush=True)
        save_obj(m, f"arena_mix_K{K}.pkl")
        if K > 1:
            c = ClusterBT(K, lam_R=LAM_R, shrink_to=SHRINK).fit(tr["F"], tr["a"], tr["b"], tr["y"], M)
            p = c.predict_proba(va["F"], va["a"], va["b"])
            rows.append(dict(K=K, method="Cluster-then-BT", val_ll=-evaluate(p, va["y"])["logloss"],
                             **{f"val_{k}": v for k, v in evaluate(p, va["y"]).items()}))
            cb = CapabilityBT(K, lam_W=LAM_W, lam_R=LAM_R, shrink_to=SHRINK).fit(
                tr["F"], tr["a"], tr["b"], tr["y"], M, W0=m.W_, R0=m.R_)
            p = cb.predict_proba(va["F"], va["a"], va["b"])
            rows.append(dict(K=K, method="Aggregation (B)", val_ll=-evaluate(p, va["y"])["logloss"],
                             **{f"val_{k}": v for k, v in evaluate(p, va["y"]).items()}))
            save_obj(cb, f"arena_capB_K{K}.pkl")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "arena_selectK.csv"), index=False)
    print(df[["K", "method", "val_ll", "val_accuracy", "bic"]])


def chosen_K():
    df = pd.read_csv(os.path.join(RES, "arena_selectK.csv"))
    d = df[df.method == "Mixture EM"]
    return int(d.loc[d.val_ll.idxmax(), "K"]), d


def stage_final(D, meta):
    tr, va, te = D["train"], D["val"], D["test"]
    M = D["M"]
    K, _ = chosen_K()
    print("chosen K =", K)
    # refit on train+val with more restarts
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    t = time.time()
    m = fit_mix(K, tv, M, n_init=5, seed=1, polish=True)
    print(f"final fit {time.time() - t:.0f}s, iters {m.n_iter_}")
    save_obj(m, "arena_final.pkl")
    rows = []
    g = GlobalBT(LAM_GLOBAL).fit(tv["a"], tv["b"], tv["y"], M)
    rows.append(dict(method="Global BT (K=1)", **evaluate(g.predict_proba(te["F"], te["a"], te["b"]), te["y"])))
    c = ClusterBT(K, lam_R=LAM_R, shrink_to=SHRINK).fit(tv["F"], tv["a"], tv["b"], tv["y"], M)
    rows.append(dict(method=f"Cluster-then-BT (K={K})", **evaluate(c.predict_proba(te["F"], te["a"], te["b"]), te["y"])))
    cb = CapabilityBT(K, lam_W=LAM_W, lam_R=LAM_R, shrink_to=SHRINK).fit(tv["F"], tv["a"], tv["b"], tv["y"], M, W0=m.W_, R0=m.R_)
    rows.append(dict(method=f"Aggregation B (K={K})", **evaluate(cb.predict_proba(te["F"], te["a"], te["b"]), te["y"])))
    rows.append(dict(method=f"Mixture EM (K={K})", **evaluate(m.predict_proba(te["F"], te["a"], te["b"]), te["y"])))
    # paired bootstrap of the log-loss difference mixture vs global
    p_m = np.clip(m.predict_proba(te["F"], te["a"], te["b"]), 1e-12, 1 - 1e-12)
    p_g = np.clip(g.predict_proba(te["F"], te["a"], te["b"]), 1e-12, 1 - 1e-12)
    yv = te["y"]
    l_m = -(yv * np.log(p_m) + (1 - yv) * np.log(1 - p_m))
    l_g = -(yv * np.log(p_g) + (1 - yv) * np.log(1 - p_g))
    diff = l_g - l_m
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "arena_test_metrics.csv"), index=False)
    print(out)
    print("logloss improvement over global BT: %.5f  95%% CI [%.5f, %.5f]" %
          (diff.mean(), *np.percentile(boots, [2.5, 97.5])))
    with open(os.path.join(RES, "arena_test_improvement.txt"), "w") as f:
        f.write("test battles %d\nlogloss(global) - logloss(mixture) = %.5f, 95%% bootstrap CI [%.5f, %.5f]\n"
                % (len(diff), diff.mean(), *np.percentile(boots, [2.5, 97.5])))


def top_terms(meta, weights, n=12):
    """Terms whose TF-IDF mass is most over-represented under weights."""
    X = meta["tfidf"].transform(pd.Series(meta["first"]).str.slice(0, 3000).str.lower())
    vocab = np.array(meta["tfidf"].get_feature_names_out())
    base = np.asarray(X.mean(0)).ravel() + 1e-6
    out = []
    for k in range(weights.shape[1]):
        w = weights[:, k] / weights[:, k].sum()
        mk = np.asarray(X.T @ w).ravel()
        lift = mk / base
        score = mk * np.log(lift)  # KL-style: frequent *and* over-represented
        out.append(list(vocab[np.argsort(-score)[:n]]))
    return out


def stage_interpret(D, meta):
    m = load_obj("arena_final.pkl")
    Fall = D["all"]["F"]
    pi = m.gate(Fall)
    K = m.K
    z = np.load(os.path.join(DATA, "arena_processed.npz"), allow_pickle=True)
    H = pd.DataFrame(z["raw_hand"], columns=meta["hand_names"])
    terms = top_terms(meta, pi)
    rows = []
    for k in range(K):
        w = pi[:, k] / pi[:, k].sum()
        top = np.argsort(-pi[:, k])[:5]
        rows.append(dict(domain=k + 1, mass=pi[:, k].mean(), hard_share=np.mean(pi.argmax(1) == k),
                         **{f"{c}": float(H[c].values @ w) for c in ["code", "math", "multi_turn", "question"]},
                         length=float(np.expm1(H["log_length"].values) @ w),
                         non_ascii=float(H["non_ascii"].values @ w),
                         terms=", ".join(terms[k]),
                         examples=" || ".join(meta["first"][i][:90].replace("\n", " ") for i in top[:3])))
    df = pd.DataFrame(rows)
    overall = dict(domain="all", mass=1.0, hard_share=1.0, **{c: H[c].mean() for c in ["code", "math", "multi_turn", "question"]},
                   length=float(np.expm1(H["log_length"]).mean()), non_ascii=H["non_ascii"].mean())
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    df.to_csv(os.path.join(RES, "arena_domains.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(df.drop(columns=["examples"]))
    for k in range(K):
        print(f"\n--- domain {k + 1}: {df.terms[k]}\n    e.g. {df.examples[k]}")

    # skill matrix with observed-information standard errors
    tr, va = D["train"], D["val"]
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    t = time.time()
    seW, seR, cov = m.observed_information(tv["F"], tv["a"], tv["b"], tv["y"])
    np.save(os.path.join(RES, "arena_cov.npy"), cov)
    print(f"observed information: {time.time() - t:.0f}s")
    models = meta["models"]
    n_battles = np.bincount(np.concatenate([tv["a"], tv["b"]]), minlength=len(models))
    g = GlobalBT(LAM_GLOBAL).fit(tv["a"], tv["b"], tv["y"], len(models))
    R = pd.DataFrame(m.R_, index=models, columns=[f"r{k + 1}" for k in range(K)])
    S = pd.DataFrame(seR, index=models, columns=[f"se{k + 1}" for k in range(K)])
    tab = pd.concat([R, S], axis=1)
    tab["global"] = g.r_
    tab["battles"] = n_battles
    v = m.domain_strength()
    for k in range(K):
        tab[f"v{k + 1}"] = v[:, k]
    tab.sort_values("global", ascending=False).to_csv(os.path.join(RES, "arena_skills.csv"))
    taus = [kendalltau(m.R_[:, k], g.r_)[0] for k in range(K)]
    print("Kendall tau(domain ranking, global ranking):", np.round(taus, 3))
    pd.DataFrame({"domain": range(1, K + 1), "kendall_tau_vs_global": taus}).to_csv(
        os.path.join(RES, "arena_tau.csv"), index=False)
    plot_interpret(m, tab, df, models)


def plot_interpret(m, tab, dom, models, n_show=20):
    import matplotlib.pyplot as plt
    ps.setup()
    K = m.K
    top = tab.sort_values("battles", ascending=False).head(n_show).sort_values("global", ascending=False)
    Rk = top[[f"r{k + 1}" for k in range(K)]].values
    dev = Rk - top["global"].values[:, None]  # domain-specific deviation from the global score
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ax.barh(np.arange(len(top)), top["global"].values, color=ps.C[0], height=0.6)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("global Bradley–Terry score")
    ax.set_title("Global ranking (K = 1)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax = axes[1]
    lim = np.abs(dev).max()
    im = ax.imshow(dev, cmap=ps.diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    names = domain_names(K)
    labels = [f"{names[k]}\n({dom.mass[k]:.0%} of prompts)" for k in range(K)]
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(False)
    for i in range(dev.shape[0]):
        for k in range(K):
            ax.text(k, i, f"{dev[i, k]:+.2f}", ha="center", va="center", fontsize=6, color=ps.INK)
    ax.set_title("Domain skill minus global score  (r_mk − r_m)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "arena_skill_heatmap.pdf"))
    fig.savefig(os.path.join(FIG, "arena_skill_heatmap.png"))
    plt.close(fig)


def weighted_laplacian_l2(gamma, a, b, M):
    out = []
    for k in range(gamma.shape[1]):
        Lk = np.zeros((M, M))
        w = gamma[:, k]
        np.add.at(Lk, (a, a), w)
        np.add.at(Lk, (b, b), w)
        np.add.at(Lk, (a, b), -w)
        np.add.at(Lk, (b, a), -w)
        out.append(np.sort(np.linalg.eigvalsh(Lk))[1])
    return out


def stage_diagnostics(D, meta):
    """Conditions E2-E4 of the proposal on the fitted model, and EM vs SQUAREM
    vs direct L-BFGS on the real data."""
    m = load_obj("arena_final.pkl")
    tr, va = D["train"], D["val"]
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    M = D["M"]
    gamma = m.posterior(tv["F"], tv["a"], tv["b"], tv["y"])
    nk = gamma.sum(0)
    lam2 = weighted_laplacian_l2(gamma, tv["a"], tv["b"], M)
    edges = np.unique(np.sort(np.column_stack([tv["a"], tv["b"]]), 1), axis=0)
    dmax = [np.abs(m.R_[edges[:, 0], k] - m.R_[edges[:, 1], k]).max() for k in range(m.K)]
    qk = expit(m.R_[edges[:, 0]] - m.R_[edges[:, 1]])  # L x K observable profiles
    gaps = [np.linalg.norm(qk[:, k] - qk[:, l]) for k in range(m.K) for l in range(k + 1, m.K)]
    # I2: affine independence of q_1..q_K -> smallest singular value of [q_k - q_K]
    sv = np.linalg.svd(qk[:, :-1] - qk[:, [-1]], compute_uv=False)
    pi_mean = m.gate(tv["F"]).mean(0)
    diag = pd.DataFrame(dict(domain=np.arange(1, m.K + 1), alpha_k=pi_mean, n_k=nk, lambda2_Lk=lam2,
                             max_abs_logit=dmax))
    diag.to_csv(os.path.join(RES, "arena_conditions.csv"), index=False)
    print(diag)
    print("E1 min pairwise ||q_k - q_l|| = %.3f ; I2 smallest singular value = %.3f; edges L=%d"
          % (min(gaps), sv.min(), len(edges)))
    with open(os.path.join(RES, "arena_conditions.txt"), "w") as f:
        f.write(f"edges L = {len(edges)}\nmin_k<l ||q_k - q_l||_2 = {min(gaps):.4f}\n"
                f"smallest singular value of [q_k - q_K] = {sv.min():.4f}\n")

    # algorithms from a common start
    K = m.K
    base = MixtureBT(K=K, init="kmeans", seed=0, **{k: v for k, v in EMKW.items() if k in ["lam_W", "lam_R", "shrink_to"]})
    base._set_prior(tv["a"], tv["b"], tv["y"], M)
    from mixbt.model import PairData
    pdat = PairData(tv["a"], tv["b"], tv["y"], M)
    W0, R0 = base._initial_params(tv["F"], tv["a"], tv["b"], tv["y"], M, pdat, np.random.default_rng(0), 0)
    traces = {}
    for name, acc, maxit in [("EM", None, 3000), ("SQUAREM", "squarem", 1000)]:
        mm = MixtureBT(K=K, lam_W=LAM_W, lam_R=LAM_R, tol=1e-10, max_iter=maxit, accel=acc, w_maxiter=30,
                       shrink_to=SHRINK)
        mm.fit(tv["F"], tv["a"], tv["b"], tv["y"], M=M, W_init=W0, R_init=R0)
        traces[name] = (np.array(mm.history_.time), np.array(mm.history_.objective))
        print(name, "iters", mm.n_iter_, "time %.0fs" % mm.history_.time[-1], "obj %.3f" % mm.history_.objective[-1])
    mm = MixtureBT(K=K, lam_W=LAM_W, lam_R=LAM_R, shrink_to=SHRINK)
    mm.fit_direct(tv["F"], tv["a"], tv["b"], tv["y"], W0, R0, tol=1e-13)
    traces["L-BFGS (direct)"] = (np.array(mm.history_.time), np.array(mm.history_.objective))
    print("L-BFGS iters", mm.n_iter_, "time %.0fs" % mm.history_.time[-1], "obj %.3f" % mm.history_.objective[-1])
    np.savez(os.path.join(RES, "arena_traces.npz"),
             **{f"{k}__t": v[0] for k, v in traces.items()}, **{f"{k}__obj": v[1] for k, v in traces.items()})


def stage_routing(D, meta):
    m = load_obj("arena_final.pkl")
    te = D["test"]
    tr, va = D["train"], D["val"]
    M = D["M"]
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    g = GlobalBT(LAM_GLOBAL).fit(tv["a"], tv["b"], tv["y"], M)
    # (a) pairwise routing on held-out battles: pick one of the two models
    p_m = m.predict_proba(te["F"], te["a"], te["b"])
    pick_m = p_m > 0.5
    pick_g = g.r_[te["a"]] > g.r_[te["b"]]
    dis = pick_m != pick_g
    win_m = np.mean(pick_m[dis] == (te["y"][dis] == 1))
    nd = dis.sum()
    se = np.sqrt(win_m * (1 - win_m) / nd)
    res = dict(test_battles=len(te["y"]), disagreements=int(nd), share=float(dis.mean()),
               mixture_pick_winrate=float(win_m), ci_lo=float(win_m - 1.96 * se), ci_hi=float(win_m + 1.96 * se))
    print(res)

    # (b) are the prompt-specific adjustments calibrated?  Bin held-out battles
    # by Delta = p_mix - p_global and compare with the realised y - p_global.
    p_g = expit(g.r_[te["a"]] - g.r_[te["b"]])
    delta = p_m - p_g
    resid = te["y"] - p_g
    bins = np.quantile(delta, np.linspace(0, 1, 11))
    lab = np.clip(np.searchsorted(bins, delta, side="right") - 1, 0, 9)
    cal = pd.DataFrame(dict(bin=range(10),
                            mean_delta=[delta[lab == j].mean() for j in range(10)],
                            mean_resid=[resid[lab == j].mean() for j in range(10)],
                            se=[resid[lab == j].std() / np.sqrt((lab == j).sum()) for j in range(10)],
                            n=[(lab == j).sum() for j in range(10)]))
    slope = float(delta @ resid / (delta @ delta))
    slope_se = float(np.sqrt(np.sum((resid - slope * delta) ** 2 * delta**2)) / (delta @ delta))
    cal.to_csv(os.path.join(RES, "arena_calibration.csv"), index=False)
    res.update(calib_slope=slope, calib_slope_se=slope_se)
    print("calibration slope of (y - p_global) on (p_mix - p_global): %.3f (se %.3f)" % (slope, slope_se))

    # (c) strong-vs-cheaper routing curves (model-based): quality of model m on
    # prompt x = expected win rate against a reference field of models.
    models = np.array(meta["models"])
    n_b = np.bincount(np.concatenate([tv["a"], tv["b"]]), minlength=M)
    field = np.where(n_b >= 500)[0]
    Rf = m.R_
    P_ = expit(Rf[:, None, :] - Rf[None, field, :])  # M x |field| x K
    v_field = P_.mean(1)  # M x K (self-match contributes 1/2; negligible)
    pi_te = m.gate(te["F"])
    Q = pi_te @ v_field.T  # n_test x M
    strong = int(np.where(models == "gpt-4-1106-preview")[0][0])
    curves = []
    for cheap_name in ["llama-2-70b-chat", "mixtral-8x7b-instruct-v0.1", "gpt-3.5-turbo-0613", "claude-2.1"]:
        c = int(np.where(models == cheap_name)[0][0])
        gap = Q[:, strong] - Q[:, c]
        order = np.argsort(-gap)  # send the prompts with the largest gap first
        for frac in np.linspace(0, 1, 21):
            kq = int(round(frac * len(gap)))
            use_strong = np.zeros(len(gap), bool)
            use_strong[order[:kq]] = True
            qual = np.where(use_strong, Q[:, strong], Q[:, c]).mean()
            curves.append(dict(cheap=cheap_name, frac_strong=frac, quality=qual,
                               random=(1 - frac) * Q[:, c].mean() + frac * Q[:, strong].mean()))
    cur = pd.DataFrame(curves)
    cur.to_csv(os.path.join(RES, "arena_routing_curves.csv"), index=False)

    # (d) routing inside a pool of open-weight models (proposal Sec. 4.3)
    open_pool = ["llama-2-70b-chat", "mixtral-8x7b-instruct-v0.1", "vicuna-33b", "wizardlm-70b", "yi-34b-chat",
                 "tulu-2-dpo-70b", "openchat-3.5", "zephyr-7b-beta", "starling-lm-7b-alpha"]
    pool = np.array([int(np.where(models == nm)[0][0]) for nm in open_pool])
    choice, S = m.route(te["F"], candidates=pool)
    jbest = np.argmax(S.mean(0))
    gain = float(np.mean(S.max(1) - S[:, jbest]))
    hard = pi_te.argmax(1)
    share = pd.crosstab(pd.Series(models[choice], name="routed_to"), pd.Series(hard + 1, name="domain"),
                        normalize="columns").round(3)
    share.to_csv(os.path.join(RES, "arena_routing_shares.csv"))
    res.update(open_pool=", ".join(open_pool), open_best_fixed=str(models[pool[jbest]]),
               open_model_based_gain=gain,
               open_share_not_best=float(np.mean(choice != pool[jbest])))
    pd.Series(res).to_csv(os.path.join(RES, "arena_routing.csv"))
    print(share)
    print("open pool: best fixed %s, routed elsewhere %.1f%%, model-based gain %.4f"
          % (models[pool[jbest]], 100 * res["open_share_not_best"], gain))


def stage_contrasts(D, meta, min_battles=1000):
    """Within-model domain contrasts r_{m,k} - r_{m,l} with observed-information SEs."""
    m = load_obj("arena_final.pkl")
    cov = np.load(os.path.join(RES, "arena_cov.npy"))
    K, p = m.W_.shape
    nW = (K - 1) * p
    skills = pd.read_csv(os.path.join(RES, "arena_skills.csv"), index_col=0)
    models = meta["models"]
    rows = []
    for i, name in enumerate(models):
        if skills.loc[name, "battles"] < min_battles:
            continue
        for k in range(K):
            for l in range(k + 1, K):
                ik, il = nW + i * K + k, nW + i * K + l
                est = m.R_[i, k] - m.R_[i, l]
                se = np.sqrt(cov[ik, ik] + cov[il, il] - 2 * cov[ik, il])
                rows.append(dict(model=name, contrast=f"D{k + 1}-D{l + 1}", estimate=est, se=se, z=est / se))
    df = pd.DataFrame(rows)
    df["p_bonf"] = np.minimum(1, len(df) * 2 * (1 - __import__("scipy.stats").stats.norm.cdf(np.abs(df.z))))
    df = df.reindex(df.z.abs().sort_values(ascending=False).index)
    df.to_csv(os.path.join(RES, "arena_contrasts.csv"), index=False)
    print(df.head(12).round(3).to_string(index=False))
    print("contrasts significant after Bonferroni:", int((df.p_bonf < 0.05).sum()), "of", len(df))


def stage_bootstrap(D, meta, B=int(os.environ.get("B", "30"))):
    """Nonparametric bootstrap of R (warm start + label alignment), compared
    with observed-information standard errors."""
    m = load_obj("arena_final.pkl")
    tr, va = D["train"], D["val"]
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    n = len(tv["y"])
    rng = np.random.default_rng(123)
    Rs = []
    for bb in range(B):
        i = rng.integers(0, n, n)
        mb = MixtureBT(K=m.K, **EMKW).fit(tv["F"][i], tv["a"][i], tv["b"][i], tv["y"][i], M=D["M"],
                                           W_init=m.W_, R_init=m.R_)
        perm = align_labels(mb.R_, m.R_)
        Rs.append(permute_params(mb.W_, mb.R_, perm)[1])
        print("boot", bb, flush=True)
    Rs = np.array(Rs)
    np.save(os.path.join(RES, "arena_boot_R.npy"), Rs)


def stage_ties(D, meta):
    """Sensitivity: treat ties as half a win (y = 0.5) instead of dropping them."""
    Dt, _ = load(include_ties=True)
    m0 = load_obj("arena_final.pkl")
    tr, va = Dt["train"], Dt["val"]
    tv = {k: np.concatenate([tr[k], va[k]]) for k in ["F", "a", "b", "y"]}
    m1 = MixtureBT(K=m0.K, **EMKW).fit(tv["F"], tv["a"], tv["b"], tv["y"], M=Dt["M"], W_init=m0.W_, R_init=m0.R_)
    perm = align_labels(m1.R_, m0.R_)
    _, R1 = permute_params(m1.W_, m1.R_, perm)
    from scipy.stats import spearmanr
    rho = [spearmanr(R1[:, k], m0.R_[:, k])[0] for k in range(m0.K)]
    print("Spearman (ties as 1/2 vs ties dropped) per domain:", np.round(rho, 3))
    pd.DataFrame({"domain": range(1, m0.K + 1), "spearman": rho}).to_csv(os.path.join(RES, "arena_ties.csv"), index=False)


def domain_names(K):
    p = os.path.join(RES, "domain_names.json")
    if os.path.exists(p):
        import json
        with open(p) as f:
            names = json.load(f)
        if len(names) == K:
            return names
    return [f"D{k + 1}" for k in range(K)]


def stage_plots(D, meta):
    import matplotlib.pyplot as plt
    ps.setup()
    df = pd.read_csv(os.path.join(RES, "arena_selectK.csv"))
    base = df[(df.method == "Mixture EM") & (df.K == 1)].val_ll.values[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    ax = axes[0]
    colors = {"Mixture EM": ps.C[0], "Aggregation (B)": ps.C[6], "Cluster-then-BT": ps.C[1]}
    for meth, c in colors.items():
        d = df[df.method == meth].sort_values("K")
        if meth != "Mixture EM":
            d = pd.concat([df[(df.method == "Mixture EM") & (df.K == 1)], d])
        ax.plot(d.K, (d.val_ll - base) * 1000, marker="o", color=c, label=meth)
    ax.axhline(0, color=ps.AXIS, lw=1)
    ax.set(xlabel="number of domains K", ylabel="validation log-lik gain over K=1\n(×10⁻³ nats / battle)",
           title="(a) Held-out fit")
    ax.legend()
    ax = axes[1]
    d = df[df.method == "Mixture EM"].sort_values("K")
    ax.plot(d.K, d.bic - d.bic.min(), marker="o", color=ps.C[0])
    ax.set(xlabel="number of domains K", ylabel="BIC − min BIC", title="(b) BIC (training data)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "arena_selectK.pdf"))
    fig.savefig(os.path.join(FIG, "arena_selectK.png"))
    plt.close(fig)

    p = os.path.join(RES, "arena_traces.npz")
    if os.path.exists(p):
        z = np.load(p)
        names = ["EM", "SQUAREM", "L-BFGS (direct)"]
        best = max(z[f"{n}__obj"].max() for n in names)
        fig, ax = plt.subplots(figsize=(3.1, 2.3))
        for j, n in enumerate(names):
            t, o = z[f"{n}__t"], z[f"{n}__obj"]
            gap = best - o
            keep = gap > 1e-4
            ax.plot(t[keep] + 1e-2, gap[keep], color=ps.C[j], label=n)
        ax.set(xscale="log", yscale="log", xlabel="seconds", ylabel="objective gap to best",
               title="Convergence on the Arena data")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "arena_traces.pdf"))
        fig.savefig(os.path.join(FIG, "arena_traces.png"))
        plt.close(fig)

    cal_p = os.path.join(RES, "arena_calibration.csv")
    cur_p = os.path.join(RES, "arena_routing_curves.csv")
    if os.path.exists(cal_p) and os.path.exists(cur_p):
        cal = pd.read_csv(cal_p)
        rt = pd.read_csv(os.path.join(RES, "arena_routing.csv"), index_col=0).iloc[:, 0]
        cur = pd.read_csv(cur_p)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
        ax = axes[0]
        lim = max(np.abs(cal.mean_delta).max(), np.abs(cal.mean_resid).max()) * 1.25
        ax.plot([-lim, lim], [-lim, lim], ls="--", color=ps.MUTED, lw=1, label="perfect calibration")
        ax.errorbar(cal.mean_delta, cal.mean_resid, yerr=1.96 * cal.se, marker="o", ls="none",
                    color=ps.C[0], label="test battles, deciles")
        ax.axhline(0, color=ps.AXIS, lw=0.8)
        ax.axvline(0, color=ps.AXIS, lw=0.8)
        ax.set(xlabel=r"predicted adjustment  $\hat p_{mix}-\hat p_{global}$",
               ylabel=r"realised  $y-\hat p_{global}$",
               title=f"(a) Prompt-specific adjustments (slope {float(rt['calib_slope']):.2f})")
        ax.legend(loc="upper left")
        ax = axes[1]
        for j, (cheap, d) in enumerate(cur.groupby("cheap", sort=False)):
            ax.plot(d.frac_strong * 100, d.quality, color=ps.C[j], label=f"{cheap}")
            ax.plot(d.frac_strong * 100, d.random, color=ps.C[j], ls=":", lw=1)
        ax.set(xlabel="% of prompts sent to gpt-4-1106-preview",
               ylabel="model-estimated win rate vs field",
               title="(b) Strong-vs-cheaper routing (model-based)")
        ax.legend(title="cheaper model (dotted: random split)", fontsize=7, title_fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "arena_routing.pdf"))
        fig.savefig(os.path.join(FIG, "arena_routing.png"))
        plt.close(fig)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["selectK", "final", "interpret", "contrasts", "diagnostics", "routing", "ties", "plots"]
    D, meta = load()
    print("train/val/test decisive:", [len(D[s]["y"]) for s in ["train", "val", "test"]], "models", D["M"])
    for s in stages:
        t = time.time()
        print(f"===== {s}", flush=True)
        globals()[f"stage_{s}"](D, meta)
        print(f"[{s} done in {time.time() - t:.0f}s]", flush=True)
