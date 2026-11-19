"""Five-fold representation comparison in the original EM mixture.

The frozen MiniLM features replace NMF topics. For each outer fold, retain
the original fit's training-selected K and penalties, random seed, three
starts and solver settings. PCA32 and standardization are fit on training
rows. The secondary equal-probability pool is fixed before evaluation.
Use --components 11 for a supplementary capacity-matched comparison with the
eleven free NMF topic proportions in the original router.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_nested import fit_predict
from arena_pipeline import categories, load_data
from lexical_refinement import bootstrap_contrasts, losses
from semantic_refinement import check_inputs


def semantic_design(embeddings, attributes, train, components, seed):
    pca = PCA(n_components=components, svd_solver="randomized", iterated_power=7, random_state=seed)
    pca.fit(embeddings[train])
    columns = np.column_stack([attributes, pca.transform(embeddings)])
    mean, sd = columns[train].mean(0), columns[train].std(0)
    eligible = np.flatnonzero(sd > 1e-10)
    z = (columns[:, eligible] - mean[eligible]) / sd[eligible]
    _, r, order = qr(z[train], mode="economic", pivoting=True)
    rank = np.sum(np.abs(np.diag(r)) > 1e-8 * max(abs(r[0, 0]), 1.))
    kept = np.sort(order[:rank])
    return np.column_stack([np.ones(len(z)), z[:, kept]])


def run(reference, embedding_path, output, workers, components):
    data = load_data()
    with np.load(reference / "oof_predictions.npz") as original, np.load(embedding_path) as cache:
        embeddings, folds, seed = check_inputs(data, original, cache, reference)
        if cache["max_tokens"].item() != 128:
            raise ValueError("Expected the prespecified 128-token encoder")
        old, global_bt = original["mixture"], original["global BT"]
    attributes = categories(data["text"]).to_numpy()
    predictions = np.full(len(folds), np.nan)
    configs = []

    def process(fold):
        train, test = folds != fold, np.flatnonzero(folds == fold)
        record = json.loads((reference / f"fold_{fold}.json").read_text())
        config = record["chosen"]["mixture"]
        G = semantic_design(embeddings, attributes, train, components, seed + fold)
        pred = fit_predict("mixture", config, data, {"G": G}, train, test, seed + fold, n_init=3)
        print(f"Completed fold {fold}, K={config['K']}, design={G.shape[1]}", flush=True)
        return test, pred, dict(fold=fold, features=G.shape[1], components=components,
                                n_train=train.sum(), n_test=len(test), **config)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for test, pred, cfg in pool.map(process, range(5)):
            predictions[test] = pred
            configs.append(cfg)
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete held-out predictions")
    methods = {"semantic mixture": predictions, "equal mixture pool": (old + predictions) / 2,
               "original mixture": old, "global BT": global_bt}
    y = data["y"]
    acc = {k: ((p > .5) == y).astype(float) for k, p in methods.items()}
    ll = {k: losses(p, y) for k, p in methods.items()}
    rows, deltas = [], []
    for m in ("semantic mixture", "equal mixture pool"):
        for ref in ("original mixture", "global BT"):
            rows.append(dict(method=m, reference=ref))
            for label, labels in (("accuracy", acc), ("logloss", ll)):
                delta = labels[m] - labels[ref]
                rows[-1][label] = delta.mean()
                deltas.append(delta)
    bounds = bootstrap_contrasts(np.column_stack(deltas), data["groups"], 1000)
    for i, row in enumerate(rows):
        for j, label in enumerate(("accuracy", "logloss")):
            row[label + "_lo"], row[label + "_hi"] = bounds[:, 2 * i + j]
    output.mkdir(exist_ok=True, parents=True)
    summary = pd.DataFrame([dict(method=m, n=len(y), accuracy=acc[m].mean(), logloss=ll[m].mean()) for m in methods])
    summary.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(configs).to_csv(output / "config.csv", index=False)
    by_fold = [dict(fold=f, method=m, n=int((folds == f).sum()), accuracy=acc[m][folds == f].mean(),
                    logloss=ll[m][folds == f].mean()) for f in range(5) for m in methods]
    pd.DataFrame(by_fold).to_csv(output / "folds.csv", index=False)
    np.savez_compressed(output / "local_oof.npz", ids=data["ids"], fold=folds, semantic_mixture=predictions)
    print(summary.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/nested")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/semantic_minilm.npz")
    parser.add_argument("--components", type=int, choices=[11, 32], default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        parser.error("Use between one and five workers")
    suffix = "semantic_mixture" if args.components == 32 else "semantic_mixture_capacity"
    output = args.output or ROOT / "results" / suffix
    with threadpool_limits(1):
        run(args.reference, args.embeddings, output, args.workers, args.components)
