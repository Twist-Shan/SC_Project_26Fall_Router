"""Grouped outer evaluation with train-only preprocessing and inner model selection.

Run: python experiments/arena_nested.py --output results/nested
Smoke: python experiments/arena_nested.py --rows 500 --quick --folds 0 --output results/nested_smoke

Five outer folds group identical normalized first prompts. Each outer training
fold has one grouped 80/20 inner holdout, used for a prespecified staged search.
All preprocessing is refitted inside that inner training subset, then on the
complete outer training subset. No legacy feature/prediction cache is used.
Bootstrap intervals condition on the fitted out-of-fold predictions: they do
not include uncertainty from repeating training or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from arena_pipeline import PromptFeatures, STRONG, digest, grouped_folds, inner_split, load_data
from mixbt import StyleMixtureBT
from mixbt.baselines import CapabilityBT, ClusterBT, GlobalBT
from mixbt.routers import Bilinear, PairLogit, SWBT

SEED = 1729
MODELS = ("global BT", "cluster-then-BT", "aggregated capability", "mixture", "MF", "SW-BT", "logistic")
KS = (1, 2, 3, 4, 5, 6, 8)
PENALTIES = ((3., 50.), (1., 20.), (10., 100.), (30., 200.))


def loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def validate_predictions(kind, p, D, test):
    """Require a prediction for every evaluated row; only out-of-task logistic rows may be NaN."""
    p = np.asarray(p)
    if p.shape != (len(test),):
        raise ValueError(f"{kind}: prediction shape {p.shape} does not match {len(test)} rows")
    expected = np.ones(len(test), dtype=bool)
    if kind == "logistic":
        sa = np.isin(D["models"][D["a"][test]], STRONG)
        sb = np.isin(D["models"][D["b"][test]], STRONG)
        expected = sa ^ sb
    bad = expected & (~np.isfinite(p) | (p < 0) | (p > 1))
    if bad.any():
        raise ValueError(f"{kind}: {int(bad.sum())} invalid predictions on evaluated rows")
    return expected


def fit_predict(kind, config, D, Z, train, test, seed, n_init=1, quick=False):
    a, b, y, M = (D[k] for k in ("a", "b", "y", "M"))
    if kind == "global BT":
        model = GlobalBT(config["lam_R"]).fit(a[train], b[train], y[train], M)
        return model.predict_proba(None, a[test], b[test])
    if kind == "mixture":
        model = StyleMixtureBT(style=None, n_init=n_init, seed=seed, max_iter=10 if quick else 30,
                               polish_all=True, **config)
        model.fit(Z["G"][train], a[train], b[train], y[train], M=M)
        return model.predict_proba(Z["G"][test], a[test], b[test])
    if kind == "cluster-then-BT":
        model = ClusterBT(seed=seed, shrink_to="global", **config)
        model.fit(Z["F"][train], a[train], b[train], y[train], M)
        return model.predict_proba(Z["F"][test], a[test], b[test])
    if kind == "aggregated capability":
        model = CapabilityBT(seed=seed, shrink_to="global", maxiter=300 if quick else 3000, **config)
        model.fit(Z["G"][train], a[train], b[train], y[train], M)
        return model.predict_proba(Z["G"][test], a[test], b[test])
    if kind == "MF":
        model = Bilinear(k=8, seed=seed, maxiter=100 if quick else 500, **config)
        model.fit(Z["X"][train], a[train], b[train], y[train], M)
        return model.predict_proba(Z["X"][test], a[test], b[test])
    if kind == "SW-BT":
        model = SWBT(lam=2., newton_iter=8, batch=32, **config)
        model.fit(Z["E"][train], a[train], b[train], y[train], M)
        return model.predict_proba(Z["E"][test], a[test], b[test])
    if kind == "logistic":
        # The offset is refitted on exactly the current training subset.
        r = GlobalBT(.1).fit(a[train], b[train], y[train], M).r_
        sa, sb = np.isin(D["models"][a], STRONG), np.isin(D["models"][b], STRONG)
        relevant, sign, offset = sa ^ sb, np.where(sa, 1., -1.), r[a] - r[b]
        use = train & relevant
        result = np.full(len(test), np.nan)
        valid = relevant[test]
        if use.sum() < 2 or len(np.unique(y[use])) < 2:
            # Defined fallback for small folds: the training-only global offset.
            from scipy.special import expit
            result[valid] = expit(offset[test][valid])
        else:
            model = PairLogit(**config).fit(Z["X"][use], offset[use], sign[use], y[use])
            result[valid] = model.predict_proba(Z["X"][test][valid], offset[test][valid], sign[test][valid])
        return result
    raise ValueError(kind)


def candidates(kind, quick=False):
    if kind == "global BT":
        return [dict(lam_R=v) for v in ((.1,) if quick else (.01, .1, 1., 10.))]
    if kind == "mixture":
        return [dict(K=k, lam_W=3., lam_R=50.) for k in ((1, 2, 3) if quick else KS)]
    if kind == "cluster-then-BT":
        return [dict(K=k, lam_R=v) for k in ((3,) if quick else (2, 3, 4))
                for v in ((50.,) if quick else (10., 50.))]
    if kind == "aggregated capability":
        return [dict(K=k, lam_W=3., lam_R=50.) for k in ((3,) if quick else (2, 3, 4))]
    if kind == "MF":
        return [dict(lam=v) for v in ((100.,) if quick else (30., 100., 300., 1000.))]
    if kind == "SW-BT":
        return [dict(gamma=v) for v in ((10.,) if quick else (3., 10., 30.))]
    if kind == "logistic":
        return [dict(lam=v) for v in ((1000.,) if quick else (100., 1000., 10000.))]
    raise ValueError(kind)


def tune(kind, D, Z, train, val, seed, quick=False):
    rows = []
    def trial(config):
        t0 = time.perf_counter()
        p = fit_predict(kind, config, D, Z, train, val, seed, quick=quick)
        valid = validate_predictions(kind, p, D, val)
        score = float(loss(D["y"][val][valid], p[valid]).mean()) if valid.any() else float("inf")
        row = dict(config=config, logloss=score, n_validation=int(valid.sum()), seconds=time.perf_counter() - t0)
        rows.append(row)
        print(f"  inner {kind}: {config}, loss={score:.6f}, seconds={row['seconds']:.1f}", flush=True)
    for config in candidates(kind, quick):
        trial(config)
    # Staged search, not an exhaustive K x penalty grid: select K, then penalty.
    if kind == "mixture" and not quick:
        best_k = min(rows, key=lambda r: r["logloss"])["config"]["K"]
        for lw, lr in PENALTIES[1:]:
            trial(dict(K=best_k, lam_W=lw, lam_R=lr))
    best = min(rows, key=lambda r: r["logloss"])
    return best["config"], rows


def run_fold(D, fid, fold, output, quick=False, models=MODELS):
    t0 = time.perf_counter()
    train, test = fid != fold, np.flatnonzero(fid == fold)
    inner_train, inner_val = inner_split(D["groups"], train, seed=SEED + 101 + fold)
    assert not set(D["groups"][train]) & set(D["groups"][test])
    assert not set(D["groups"][inner_train]) & set(D["groups"][inner_val])
    val = np.flatnonzero(inner_val)
    # A fixed, label-independent cap makes SW-BT tuning affordable.
    if len(val) > 4000:
        val = np.sort(np.random.default_rng(SEED + 211 + fold).choice(val, 4000, replace=False))
    feature_config = dict(seed=SEED + fold, max_features=1500 if quick else 30000,
                          min_df=2 if quick else 5, n_topics=4 if quick else 12, n_lsa=8 if quick else 32)
    print(f"Fold {fold}: inner train={inner_train.sum()}, validation={len(val)}, outer test={len(test)}", flush=True)
    inner_features = PromptFeatures(**feature_config).fit(D["text"][inner_train])
    inner_Z = inner_features.transform(D["text"])
    chosen, searches = {}, {}
    for kind in models:
        chosen[kind], searches[kind] = tune(kind, D, inner_Z, inner_train, val, SEED + fold, quick)
    del inner_Z
    outer_features = PromptFeatures(**feature_config).fit(D["text"][train])
    Z = outer_features.transform(D["text"])
    predictions, fit_times = {}, {}
    for kind in models:
        start = time.perf_counter()
        predictions[kind] = fit_predict(kind, chosen[kind], D, Z, train, test, SEED + fold,
                                        n_init=1 if quick else 3, quick=quick)
        validate_predictions(kind, predictions[kind], D, test)
        fit_times[kind] = time.perf_counter() - start
        print(f"Fold {fold}: outer {kind}, seconds={fit_times[kind]:.1f}", flush=True)
    info = dict(fold=fold, n_train=int(train.sum()), n_test=len(test), n_inner_train=int(inner_train.sum()),
                n_inner_validation=len(val), training_ids_hash=digest(D["ids"][train]),
                test_ids_hash=digest(D["ids"][test]), inner_training_ids_hash=digest(D["ids"][inner_train]),
                inner_validation_ids_hash=digest(D["ids"][val]), outer_feature_names=outer_features.gate_names_,
                chosen=chosen, searches=searches, fit_seconds=fit_times, seconds=time.perf_counter() - t0)
    np.savez_compressed(output / f"fold_{fold}.npz", idx=test, ids=D["ids"][test], **predictions)
    (output / f"fold_{fold}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def quality_curve(score, strong_won, weights=None):
    """Tie-aware curve; all equal scores imply the random-routing straight line."""
    w = np.ones(len(score)) if weights is None else np.asarray(weights, float)
    order = np.argsort(-score, kind="stable")
    score, y, w = np.asarray(score)[order], np.asarray(strong_won)[order], w[order]
    end = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    total = w.sum()
    q = np.r_[0., np.cumsum(w)[end] / total]
    performance = (w * (1 - y)).sum() / total + np.r_[0., np.cumsum(w * (2 * y - 1))[end] / total]
    # Zero-weight score blocks arise during bootstrap and carry no horizontal area.
    keep = np.r_[True, np.diff(q) > 0]
    return q[keep], performance[keep]


def curve_metrics(q, performance):
    gap = performance[-1] - performance[0]
    if gap <= 1e-12:
        return dict(apgr=np.nan, cpt80=np.nan)
    recovered = (performance - performance[0]) / gap
    trapz = getattr(np, "trapezoid", np.trapz)
    found = np.flatnonzero(recovered >= .8)
    j = int(found[0]) if len(found) else None
    cpt = np.nan if j is None else (0. if j == 0 else q[j - 1] + (.8 - recovered[j - 1]) *
                                   (q[j] - q[j - 1]) / (recovered[j] - recovered[j - 1]))
    return dict(apgr=float(trapz(recovered, q)), cpt80=float(cpt))


def settings(D):
    models, a, b = D["models"], D["a"], D["b"]
    proprietary = ("gpt-", "claude", "mistral-medium", "gemini", "palm", "pplx")
    weak_sets = {"open-weight": [m for m in models if not m.startswith(proprietary)],
                 "all others": [m for m in models if m not in STRONG]}
    result = {}
    for name, weak in weak_sets.items():
        sa = np.isin(models[a], STRONG) & np.isin(models[b], weak)
        sb = np.isin(models[b], STRONG) & np.isin(models[a], weak)
        idx = np.flatnonzero(sa | sb)
        result[name] = dict(idx=idx, flip=sb[idx], strong_won=np.where(sb[idx], 1 - D["y"][idx], D["y"][idx]))
    return result


def evaluate(D, fid, P, output, bootstrap=1000):
    y = D["y"]
    pg = np.clip(P["global BT"], 1e-12, 1 - 1e-12)
    cg = ((pg > .5) == (y == 1)).astype(float)
    gap = np.abs(np.log(pg / (1 - pg)))
    # Cluster bootstrap of fixed OOF predictions, paired across methods.
    _, group = np.unique(D["groups"], return_inverse=True)
    n_groups = group.max() + 1
    rng = np.random.default_rng(31415)
    accuracy, by_gap, by_fold = [], [], []
    for kind, p in P.items():
        if not np.isfinite(p).all():
            continue
        c = ((p > .5) == (y == 1)).astype(float)
        dl = c - cg
        row = dict(router=kind, n=len(y), accuracy=c.mean(), gain=dl.mean(), logloss=loss(y, p).mean())
        grouped_sum = np.bincount(group, weights=dl, minlength=n_groups)
        grouped_n = np.bincount(group, minlength=n_groups)
        boots = []
        local_rng = np.random.default_rng(31415)
        for _ in range(bootstrap):
            sample = local_rng.integers(0, n_groups, n_groups)
            boots.append(grouped_sum[sample].sum() / grouped_n[sample].sum())
        if boots:
            row.update(gain_lo=np.quantile(boots, .025), gain_hi=np.quantile(boots, .975), se=np.std(boots, ddof=1))
        accuracy.append(row)
        for lo, hi in ((0., .1), (.1, .2), (.2, .4), (.4, np.inf)):
            mask = (gap >= lo) & (gap < hi)
            by_gap.append(dict(router=kind, lo=lo, hi=hi, n=int(mask.sum()),
                               accuracy=c[mask].mean(), acc_global=cg[mask].mean(), gain=dl[mask].mean()))
        for fold in range(5):
            mask = fid == fold
            by_fold.append(dict(router=kind, fold=fold, n=int(mask.sum()), accuracy=c[mask].mean(),
                                gain=dl[mask].mean(), logloss=loss(y[mask], p[mask]).mean()))
    pd.DataFrame(accuracy).to_csv(output / "pairwise_accuracy.csv", index=False)
    pd.DataFrame(by_gap).to_csv(output / "accuracy_by_gap.csv", index=False)
    pd.DataFrame(by_fold).to_csv(output / "metrics_by_fold.csv", index=False)
    routing, curves = [], {}
    for setting, s in settings(D).items():
        idx, flip, sw = s["idx"], s["flip"], s["strong_won"]
        if not len(idx):
            continue
        _, gi = np.unique(D["groups"][idx], return_inverse=True)
        ng = gi.max() + 1
        scores = {kind: np.where(flip, 1 - p[idx], p[idx]) for kind, p in P.items()}
        boot = {kind: [] for kind in P}
        for _ in range(bootstrap):
            counts = np.bincount(rng.integers(0, ng, ng), minlength=ng)
            w = counts[gi]
            for kind, score in scores.items():
                boot[kind].append(curve_metrics(*quality_curve(score, sw, w))["apgr"])
        ref_point = curve_metrics(*quality_curve(scores["global BT"], sw))["apgr"]
        for kind, score in scores.items():
            q, perf = quality_curve(score, sw)
            sm = curve_metrics(q, perf)
            row = dict(setting=setting, router=kind, n=len(idx), strong_win=sw.mean(), **sm,
                       gain_vs_global=sm["apgr"] - ref_point)
            if bootstrap:
                delta = np.array(boot[kind]) - np.array(boot["global BT"])
                row.update(apgr_lo=np.nanquantile(boot[kind], .025), apgr_hi=np.nanquantile(boot[kind], .975),
                           gain_lo=np.nanquantile(delta, .025), gain_hi=np.nanquantile(delta, .975))
            routing.append(row)
            grid = np.linspace(0, 1, 201)
            curves[f"{setting}|{kind}"] = np.column_stack([grid, np.interp(grid, q, perf)])
    pd.DataFrame(routing).to_csv(output / "routing_summary.csv", index=False)
    np.savez_compressed(output / "routing_curves.npz", **curves)
    if "mixture" in P and "MF" in P:
        cm = ((P["mixture"] > .5) == (y == 1)).astype(float)
        cf = ((P["MF"] > .5) == (y == 1)).astype(float)
        pd.Series(dict(acc_diff=(cm - cf).mean(), logloss_diff=(loss(y, P["mixture"]) - loss(y, P["MF"])).mean())).to_csv(output / "mixture_vs_mf.csv")
    print(pd.DataFrame(accuracy).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "nested")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true", help="Fit/checkpoint requested folds without pooled evaluation")
    args = parser.parse_args()
    if args.fit_only and args.evaluate_only:
        parser.error("--fit-only and --evaluate-only are mutually exclusive")
    if any(f < 0 or f >= 5 for f in args.folds):
        parser.error("folds must be in 0..4")
    if (args.rows or args.quick) and args.output.resolve() == (ROOT / "results" / "nested").resolve():
        parser.error("Smoke/quick runs require a separate --output directory")
    D = load_data()
    if args.rows:
        ix = np.sort(np.random.default_rng(SEED).choice(len(D["y"]), min(args.rows, len(D["y"])), replace=False))
        for key in ("text", "groups", "ids", "a", "b", "y"):
            D[key] = D[key][ix]
    fid = grouped_folds(D["groups"], seed=SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = sorted((ROOT / "mixbt").glob("*.py")) + [Path(__file__).resolve(),
                    ROOT / "experiments" / "arena_pipeline.py", ROOT / "experiments" / "arena_features_v2.py"]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_files}
    search_config = {kind: candidates(kind, args.quick) for kind in MODELS}
    search_config.update(staged_mixture_penalties=PENALTIES, inner_restarts=1,
                         outer_restarts=1 if args.quick else 3, quick=args.quick, threads=args.threads)
    config_hash = hashlib.sha256(json.dumps(search_config, sort_keys=True).encode()).hexdigest()
    protocol = dict(version=2, seed=SEED, n=len(D["y"]), groups=len(np.unique(D["groups"])),
                    data_hash=digest(zip(D["ids"], D["a"], D["b"], D["y"], D["groups"])),
                    exact_first_prompt_hash=hashlib.sha256(json.dumps(D["text"].tolist(), ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest(), source_hashes=source_hashes,
                    config_hash=config_hash, search_config=search_config,
                    fold_hash=digest(fid), quick=args.quick, outer_folds=5, inner="one grouped 80/20 split per outer fold",
                    inner_validation_cap=4000, mixture_K_grid=KS, staged_penalties=PENALTIES,
                    inner_restarts=1, outer_restarts=1 if args.quick else 3,
                    first_prompt_only=True, target="Final recorded battle preference, not isolated first-answer quality",
                    uncertainty="Prompt-group bootstrap of fixed OOF predictions; excludes repeated training and selection uncertainty",
                    evaluation="APGR/CPT use preference and strong-call-share proxies, not monetary cost or correctness")
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Output directory contains a different protocol/data set; use a new directory")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "split_manifest.npz", ids=D["ids"], groups=D["groups"], fold=fid, y=D["y"])
    with threadpool_limits(limits=args.threads):
        if not args.evaluate_only:
            for fold in args.folds:
                if (args.output / f"fold_{fold}.npz").exists():
                    print(f"Reusing completed fold {fold}", flush=True)
                    continue
                run_fold(D, fid, fold, args.output, quick=args.quick)
        if args.fit_only:
            print("Requested folds complete; --fit-only skips pooled evaluation.", flush=True)
        elif all((args.output / f"fold_{f}.npz").exists() for f in range(5)):
            P = {kind: np.full(len(D["y"]), np.nan) for kind in MODELS}
            for fold in range(5):
                saved = np.load(args.output / f"fold_{fold}.npz")
                if not np.array_equal(D["ids"][saved["idx"]], saved["ids"]):
                    raise ValueError("Cached fold ID mismatch")
                for kind in MODELS:
                    P[kind][saved["idx"]] = saved[kind]
            np.savez_compressed(args.output / "oof_predictions.npz", ids=D["ids"], fold=fid, **P)
            evaluate(D, fid, P, args.output, args.bootstrap)
        else:
            print("Partial fold run; pooled metrics are generated only after all five folds finish.", flush=True)


if __name__ == "__main__":
    main()
