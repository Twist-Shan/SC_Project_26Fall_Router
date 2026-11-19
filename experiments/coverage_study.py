"""How comparison coverage and skill separation affect candidate-set routing."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixbt import GlobalBT, StyleMixtureBT, simulate, true_proba
from sim_experiments import routing_metrics, true_scores


def run():
    m, n_train, n_test = 8, 2400, 4000
    all_edges = np.array(list(combinations(range(m), 2)))
    weights = np.array([[0.0, 1.7, -0.8], [0.0, 0.0, 0.0]])
    # Two specialists, shared average strengths, and opposite domain contrasts.
    base = np.array([0.35, 0.30, 0.18, 0.10, 0.03, -0.15, -0.23, -0.34])
    contrast = np.array([1.0, -1.0, 0.55, -0.55, 0.25, -0.25, 0.1, -0.1])
    rows = []
    for rep, stream in enumerate(np.random.SeedSequence(731).spawn(20)):
        graph_seed, train_seed, test_seed, fit_seed = map(int, stream.generate_state(4))
        order = np.random.default_rng(graph_seed).permutation(m)
        path = np.sort(np.column_stack((order[:-1], order[1:])), axis=1)
        for delta in (0.4, 1.4):
            skills = base[:, None] + delta * np.column_stack((contrast, -contrast))
            skills -= skills.mean(axis=0)
            test = simulate(n_test, weights, skills, rng=test_seed, edges=all_edges)
            f = test["F"]
            target = true_scores(f, weights, skills)
            pair_target = true_proba(f, test["a"], test["b"], weights, skills)
            for graph, edges in (("path", path), ("complete", all_edges)):
                train = simulate(n_train, weights, skills, rng=train_seed, edges=edges)
                args = (train["a"], train["b"], train["y"])
                mixture = StyleMixtureBT(K=2, style=None, lam_W=1.0, lam_R=2.0,
                                         n_init=3, seed=fit_seed, max_iter=30, polish_all=True)
                mixture.fit(train["F"], *args, M=m)
                global_bt = GlobalBT(lam_R=0.1).fit(*args, M=m)
                designs = (("mixture", mixture, mixture.route(f)[0]),
                           ("global_BT", global_bt, np.full(n_test, np.argmax(global_bt.r_))))
                for method, model, choice in designs:
                    p = model.predict_proba(f, test["a"], test["b"])
                    rows.append(dict(rep=rep, train_seed=train_seed, graph=graph,
                                     edges=len(edges), delta=delta, method=method,
                                     pair_mae=np.abs(p - pair_target).mean(),
                                     **routing_metrics(choice, target)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        results = run()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    results.to_csv(output / "coverage_study_runs.csv", index=False)
    groups = results.groupby(["delta", "graph", "method"])
    means = groups[["regret", "best_pick", "pair_mae"]].mean()
    means["regret_mcse"] = groups["regret"].sem()
    means.to_csv(output / "coverage_study_summary.csv")
    print(means.round(4).to_string())
