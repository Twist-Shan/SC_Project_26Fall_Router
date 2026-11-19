"""Descriptive full-data fit using the corrected first-prompt feature pipeline.

Run after the five outer folds of ``arena_nested.py`` have completed::

    python experiments/arena_descriptive.py

The modal outer-fold K is used; ties favour the smaller K. Within that K,
the modal penalty pair is used, with deterministic lexicographic tie-breaking.
This refit describes all observed decisive battles. It supplies no test-set
performance estimate, confidence intervals or significance tests.

Outputs live in results/nested/descriptive/, separate from legacy results.
For a smoke run, specify --max-rows and a separate --output-dir.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from arena_pipeline import PromptFeatures, load_data  # noqa: E402
from mixbt import StyleMixtureBT  # noqa: E402


def choose_configuration(results_dir, K=None, lam_W=None, lam_R=None):
    """Return a documented descriptive configuration, without outcome tuning."""
    if K is not None:
        return {"K": K, "lam_W": 3.0 if lam_W is None else lam_W,
                "lam_R": 50.0 if lam_R is None else lam_R}, {
                    "rule": "explicit CLI configuration; descriptive only"}
    records = []
    paths = [Path(results_dir) / f"fold_{fold}.json" for fold in range(5)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Wait for all five outer folds or supply --K. Missing: " + ", ".join(missing))
    for p in paths:
        with p.open(encoding="utf-8") as stream:
            records.append(json.load(stream)["chosen"]["mixture"])
    counts = Counter(int(r["K"]) for r in records)
    K = min(counts, key=lambda k: (-counts[k], k))
    pairs = Counter((float(r["lam_W"]), float(r["lam_R"])) for r in records if int(r["K"]) == K)
    w, r = min(pairs, key=lambda item: (-pairs[item], item))
    configuration = {"K": K, "lam_W": w if lam_W is None else lam_W,
                     "lam_R": r if lam_R is None else lam_R}
    return configuration, {
        "rule": "modal outer K, then modal penalty pair within K; smallest lexicographic tie",
        "outer_choices": records,
        "penalty_override": lam_W is not None or lam_R is not None,
    }


def describe(model, features, transformed, data, output_dir):
    """Save descriptive quantities with no uncertainty-based annotations."""
    output_dir = Path(output_dir)
    gate = model.gate(transformed["G"])
    mass = gate.mean(axis=0)
    # Reporting order only: the stored model keeps its original parameterisation.
    order = np.argsort(-mass, kind="stable")
    gate, mass = gate[:, order], mass[order]
    categories = transformed["categories"]
    if not isinstance(categories, pd.DataFrame):
        categories = pd.DataFrame(categories, columns=features.category_names_)
    rows = []
    for name in categories.columns:
        if name == "log_length":
            continue  # a continuous length mean is not a category prevalence
        value = categories[name].to_numpy(dtype=float)
        prevalence = float(value.mean())
        if prevalence <= 0:
            continue
        weighted_mean = gate.T @ value / gate.sum(axis=0)
        rows.append({"feature": name, "kind": "category", "prevalence": prevalence,
                     **{f"lift{k + 1}": float(weighted_mean[k] / prevalence) for k in range(model.K)}})
    topics = np.asarray(transformed["topics"])
    for j in range(topics.shape[1]):
        value = topics[:, j]
        prevalence = float(value.mean())
        if prevalence <= 0:
            continue
        words = features.topic_words_[j]
        words = words if isinstance(words, str) else ", ".join(words[:5])
        weighted_mean = gate.T @ value / gate.sum(axis=0)
        rows.append({"feature": f"topic {j + 1}: {words}", "kind": "topic", "prevalence": prevalence,
                     **{f"lift{k + 1}": float(weighted_mean[k] / prevalence) for k in range(model.K)}})
    lifts = pd.DataFrame(rows)
    lifts.to_csv(output_dir / "lifts.csv", index=False)
    skills = pd.DataFrame(model.R_[:, order], index=data["models"],
                          columns=[f"r{k + 1}" for k in range(model.K)])
    skills.index.name = "model"
    skills["global"] = model.prior_[0]
    skills["battles"] = np.bincount(np.r_[data["a"], data["b"]], minlength=data["M"])
    for k in range(model.K):
        skills[f"difference{k + 1}"] = skills[f"r{k + 1}"] - skills["global"]
    skills.to_csv(output_dir / "skills.csv")
    domains = [{"domain": f"D{k + 1}", "model_component": int(component), "mass": float(mass[k])}
               for k, component in enumerate(order)]
    with (output_dir / "domains.json").open("w", encoding="utf-8") as stream:
        json.dump(domains, stream, indent=2)
    np.savez_compressed(output_dir / "gate.npz", gate=gate, ids=data["ids"],
                        report_component_order=order)
    return lifts, skills, domains


def make_figure(lifts, skills, domains, output_dir):
    """A descriptive heat map with neutral domain labels and no test stars."""
    from mixbt import plotstyle as ps
    ps.setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    K = len(domains)
    cats = lifts[(lifts.kind == "category") & (lifts.prevalence >= 0.003)].copy()
    top = skills.sort_values("battles", ascending=False).head(16).sort_values("global", ascending=False)
    labels = {
        "creative": "creative writing", "roleplay": "role-play", "translate": "translation",
        "rewrite": "rewrite / summarise", "factual": "factual question", "advice": "advice / how-to",
        "puzzle": "reasoning puzzle", "chitchat": "greeting / joke", "explain": "explanation",
        "brainstorm": "list / brainstorm", "sensitive": "sensitive request",
        "constraints": "format constraints", "long_input": "long input", "non_english": "non-English",
    }
    vertical = K > 4
    height = max(3.0, 0.16 * max(len(cats), len(top)) + 0.7)
    fig, axes = plt.subplots(2 if vertical else 1, 1 if vertical else 2,
                             figsize=(ps.TEXTWIDTH, height * (2 if vertical else 1)))
    ax, bx = np.ravel(axes)
    cols = [f"lift{k + 1}" for k in range(K)]
    values = cats[cols].to_numpy(float)
    log_lift = np.log2(np.clip(values, 1e-6, None))
    limit = max(1.0, min(3.0, float(np.abs(log_lift).max())))
    ps.heatmap(ax, np.clip(log_lift, -limit, limit), ps.diverging_cmap(), TwoSlopeNorm(0, -limit, limit))
    for i in range(len(cats)):
        for k in range(K):
            ax.text(k, i, f"{values[i, k]:.1f}", ha="center", va="center", fontsize=6.6)
    ax.set_yticks(range(len(cats)), [labels.get(s, s) for s in cats.feature], fontsize=7)
    ax.set_xticks(range(K), [f"D{k + 1}\n({100 * d['mass']:.0f}%)" for k, d in enumerate(domains)], fontsize=7)
    ax.set_title("(a) Prompt-category lift")
    deviations = top[[f"difference{k + 1}" for k in range(K)]].to_numpy(float)
    limit = max(0.01, float(np.abs(deviations).max()))
    ps.heatmap(bx, deviations, ps.diverging_cmap(), Normalize(-limit, limit))
    for i in range(len(top)):
        for k in range(K):
            bx.text(k, i, f"{deviations[i, k]:+.2f}", ha="center", va="center", fontsize=6.6)
    bx.set_yticks(range(len(top)), top.index, fontsize=7)
    bx.set_xticks(range(K), [f"D{k + 1}" for k in range(K)], fontsize=7)
    bx.set_title("(b) Skill minus global skill (descriptive)")
    for panel in (ax, bx):
        panel.grid(False)
        panel.tick_params(axis="both", length=0)
        for spine in panel.spines.values():
            spine.set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(Path(output_dir) / f"fig_nested_domains.{suffix}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "nested")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--K", type=int)
    parser.add_argument("--lam-W", type=float)
    parser.add_argument("--lam-R", type=float)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--feature-seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-rows", type=int, help="smoke run only; requires a separate --output-dir")
    args = parser.parse_args(argv)
    if args.n_init < 1 or args.threads < 1 or (args.K is not None and args.K < 1):
        parser.error("K, n-init and threads must be positive")
    if args.max_rows is not None and (args.output_dir is None or args.max_rows < 20):
        parser.error("--max-rows requires a separate --output-dir and at least 20 rows")
    config, source = choose_configuration(args.results_dir, args.K, args.lam_W, args.lam_R)
    if config["lam_W"] < 0 or config["lam_R"] <= 0:
        parser.error("lam-W must be nonnegative and lam-R positive")
    output_dir = args.output_dir or args.results_dir / "descriptive"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        data = load_data(args.data)
        if args.max_rows is not None:
            ix = np.arange(min(args.max_rows, len(data["y"])))
            for key in ("text", "groups", "ids", "a", "b", "y"):
                data[key] = np.asarray(data[key])[ix]
        print("descriptive configuration:", config, flush=True)
        features = PromptFeatures(seed=args.feature_seed).fit(data["text"])
        transformed = features.transform(data["text"])
        model = StyleMixtureBT(style=None, n_init=args.n_init if config["K"] > 1 else 1,
                               seed=args.seed, max_iter=30, polish_all=True, verbose=True, **config)
        fit_start = time.perf_counter()
        model.fit(transformed["G"], data["a"], data["b"], data["y"], M=data["M"])
        fit_seconds = time.perf_counter() - fit_start
        joblib.dump({"model": model, "features": features, "models": data["models"],
                     "configuration": config}, output_dir / "fit.joblib")
        lifts, skills, domains = describe(model, features, transformed, data, output_dir)
        make_figure(lifts, skills, domains, output_dir)
        bd = model._bd(data["a"], data["b"], data["y"], None)
        gradient, objective = model.score(transformed["G"], bd, model.W_, model.R_, model.B_)
    summary = {
        "purpose": "smoke" if args.max_rows is not None else "descriptive full-data refit; no test claims",
        "configuration": config, "selection": source, "n": int(len(data["y"])),
        "n_groups": int(len(np.unique(data["groups"]))), "M": int(data["M"]),
        "p": int(transformed["G"].shape[1]), "gate_features": list(features.gate_names_),
        "seed": args.seed, "feature_seed": args.feature_seed, "n_init": args.n_init,
        "ids_sha256": hashlib.sha256("\n".join(map(str, data["ids"])).encode()).hexdigest(),
        "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - start,
        "objective": float(objective), "gradient_inf_norm": float(np.abs(gradient).max()),
        "domains": domains,
        "uncertainty": "None reported; skill-minus-global differences are descriptive.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
