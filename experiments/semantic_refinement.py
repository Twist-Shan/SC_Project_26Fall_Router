"""Outer-fold evaluation of a frozen multilingual encoder for preference routing.

Run the original grouped Arena evaluation and encode the first prompts first.
The primary router is a fixed equal-probability pool of the semantic-family BT
and original EM mixture. Ridge and semantic scale are selected by each outer
training set's inner validation log loss. The previous lexical pool is refitted
with its saved inner choices for a direct paired comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import digest, grouped_folds, inner_split, load_data
from lexical_refinement import bootstrap_contrasts, losses
from mixbt.lexical import LexicalBT
from mixbt.semantic import SemanticBasis, SemanticFamilyBT

EXPECTED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
METHOD = "semantic pool"
CONTROL = "lexical pool"


def check_inputs(data, baseline, embeddings, reference):
    protocol = json.loads((reference / "protocol.json").read_text())
    ids = data["ids"]
    exact = hashlib.sha256(json.dumps(data["text"].tolist(), ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
    fold = grouped_folds(data["groups"], seed=protocol["seed"])
    row_hash = digest(zip(ids, data["a"], data["b"], data["y"], data["groups"]))
    if (protocol["quick"] or protocol["exact_first_prompt_hash"] != exact or
            protocol["data_hash"] != row_hash or not np.array_equal(fold, baseline["fold"])
            or not np.array_equal(baseline["ids"], ids)):
        raise ValueError("Baseline OOF predictions do not match this grouped Arena dataset")
    x = embeddings["features"]
    if (not np.array_equal(embeddings["ids"], ids) or str(embeddings["exact_first_prompt_hash"]) != exact
            or str(embeddings["revision"]) != EXPECTED_REVISION or x.shape != (len(ids), 384)
            or not np.isfinite(x).all()):
        raise ValueError("Embedding cache is inconsistent with the text, revision, or IDs")
    return x, fold, protocol["seed"]


def summary(methods, data, folds, output):
    correct = {k: ((p > .5) == data["y"]).astype(float) for k, p in methods.items()}
    nll = {k: losses(p, data["y"]) for k, p in methods.items()}
    metrics = pd.DataFrame([dict(method=k, n=len(v), accuracy=correct[k].mean(), logloss=nll[k].mean())
                            for k, v in methods.items()])
    rows, delta = [], []
    for alt in (METHOD, "semantic BT"):
        for ref in (CONTROL, "lexical BT", "mixture", "aggregate-score BT", "global BT"):
            ad = correct[alt] - correct[ref]
            ld = nll[alt] - nll[ref]
            rows.append(dict(method=alt, reference=ref, accuracy=ad.mean(), logloss=ld.mean()))
            delta.extend((ad, ld))
    bounds = bootstrap_contrasts(np.column_stack(delta), data["groups"], 1000)
    for i, row in enumerate(rows):
        for j, name in enumerate(("accuracy", "logloss")):
            row[name + "_lo"], row[name + "_hi"] = bounds[:, 2 * i + j]
    per_fold = [dict(fold=f, method=k, n=int((folds == f).sum()), accuracy=correct[k][folds == f].mean(),
                     logloss=nll[k][folds == f].mean()) for f in np.unique(folds) for k in methods]
    contrasts = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    contrasts.to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(per_fold).to_csv(output / "folds.csv", index=False)
    write_difficulty(methods, data, output)
    print(metrics.round(6).to_string(index=False), flush=True)
    print(contrasts[contrasts.method.eq(METHOD)].round(6).to_string(index=False), flush=True)


def write_difficulty(methods, data, output):
    gap = np.abs(methods["global BT"] - .5)
    new_correct = (methods[METHOD] > .5) == data["y"]
    old_correct = (methods[CONTROL] > .5) == data["y"]
    change = new_correct.astype(float) - old_correct
    logloss_change = losses(methods[METHOD], data["y"]) - losses(methods[CONTROL], data["y"])
    strata = (("0 to 0.05", gap <= .05), ("over 0.05 to 0.15", (gap > .05) & (gap <= .15)),
              ("over 0.15", gap > .15))
    by_gap = []
    for label, mask in strata:
        if mask.any():
            interval = bootstrap_contrasts(np.column_stack([change[mask], logloss_change[mask]]),
                                           data["groups"][mask], 1000)
            by_gap.append(dict(global_probability_gap=label, n=mask.sum(),
                               semantic_accuracy=new_correct[mask].mean(),
                               lexical_pool_accuracy=old_correct[mask].mean(),
                               net_additional_correct=int(change[mask].sum()), accuracy_gain=change[mask].mean(),
                               accuracy_lo=interval[0, 0], accuracy_hi=interval[1, 0],
                               logloss_difference=logloss_change[mask].mean(),
                               logloss_lo=interval[0, 1], logloss_hi=interval[1, 1]))
    pd.DataFrame(by_gap).to_csv(output / "difficulty.csv", index=False)


def run(args):
    data = load_data()
    a, b, y, groups, names = (data[k] for k in ("a", "b", "y", "groups", "models"))
    with np.load(args.reference / "oof_predictions.npz") as oof, np.load(args.embeddings) as embeds:
        embeddings, folds, seed = check_inputs(data, oof, embeds, args.reference)
        methods = {dest: oof[src] for dest, src in (("mixture", "mixture"),
                   ("aggregate-score BT", "aggregated capability"), ("global BT", "global BT"))}
    prior = pd.read_csv(args.lexical / "tuning.csv")
    lexical = LexicalBT()
    words = lexical.encode(data["text"])
    word_width = words.shape[1]
    word_design = lexical.pair_design(words, a, b, len(names))
    del words
    fresh, old = np.full(len(y), np.nan), np.full(len(y), np.nan)
    cfg_rows, timings = [], []
    for fold in np.unique(folds):
        start = perf_counter()
        train, test = folds != fold, folds == fold
        inner, val = inner_split(groups, train, seed + 101 + fold)
        iv = np.flatnonzero(val)
        if len(iv) > 4000:
            iv = np.sort(np.random.default_rng(seed + 211 + fold).choice(iv, 4000, replace=False))
        template = SemanticFamilyBT(names)
        z = SemanticBasis().fit(embeddings[inner]).transform(embeddings)
        semantic = template.semantic_design(z, a, b)
        pool = []
        for scale in (2., 6.):
            joined = sparse.hstack([word_design, scale * semantic], format="csr")
            training, validation = joined[inner], joined[iv]
            del joined
            for C in (.01, .03, .09):
                model = SemanticFamilyBT(names, C, scale).fit_joint(training, y[inner], word_width, z.shape[1])
                score = losses(model.estimator.predict_proba(validation)[:, 1], y[iv]).mean()
                pool.append((score, C, scale))
                cfg_rows.append(dict(fold=fold, C=C, semantic_scale=scale, inner_logloss=score))
            del training, validation
        del semantic, z
        _, C, scale = min(pool)
        selected = perf_counter()
        model = SemanticFamilyBT(names, C, scale)
        model.basis = SemanticBasis().fit(embeddings[train])
        z = model.basis.transform(embeddings)
        joined = sparse.hstack((word_design, scale * model.semantic_design(z, a, b)), format="csr")
        model.fit_joint(joined[train], y[train], word_width, z.shape[1])
        fresh[test] = model.estimator.predict_proba(joined[test])[:, 1]
        del joined, z
        outer_fit = perf_counter() - selected
        prior_rows = prior.loc[prior.fold.eq(fold)].sort_values(["inner_logloss", "C"])
        baseline = LexicalBT(C=float(prior_rows.iloc[0].C)).fit_design(word_design[train], y[train], len(names))
        old[test] = baseline.model_.predict_proba(word_design[test])[:, 1]
        timings.append(dict(fold=fold, inner_seconds=selected-start, final_fit_predict_seconds=outer_fit,
                            C=C, semantic_scale=scale))
        print(f"Completed fold {fold}: C={C:g}, semantic scale={scale:g}, outer stage={outer_fit:.1f}s", flush=True)
    if not np.isfinite(fresh).all() or not np.isfinite(old).all():
        raise ValueError("Missing held-out predictions")
    methods[METHOD] = .5 * (methods["mixture"] + fresh)
    methods[CONTROL] = .5 * (methods["mixture"] + old)
    methods["semantic BT"], methods["lexical BT"] = fresh, old
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cfg_rows).to_csv(args.output / "tuning.csv", index=False)
    pd.DataFrame(timings).to_csv(args.output / "timings.csv", index=False)
    summary(methods, data, folds, args.output)
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_predictions, ids=data["ids"], fold=folds, pred=methods[METHOD],
                            semantic_only=fresh, lexical_pool=methods[CONTROL])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=ROOT / "results/nested")
    p.add_argument("--lexical", type=Path, default=ROOT / "results/lexical_refinement")
    p.add_argument("--embeddings", type=Path, default=ROOT / "data/semantic_minilm.npz")
    p.add_argument("--output", type=Path, default=ROOT / "results/semantic_refinement")
    p.add_argument("--save-predictions", type=Path)
    with threadpool_limits(limits=1):
        run(p.parse_args())
