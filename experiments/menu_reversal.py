"""Can a fitted pairwise router reverse its choice when the candidate set changes?

A structural stress test at one target prompt type. Each replicate generates
3600 independent training votes (six prompt types x three pairs x 200). See
DESIGNS, WEIGHTS, and MENUS for the fixed experimental specification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixbt import CapabilityBT, StyleMixtureBT

DESIGNS = np.array([[1, 0, 0], [1, 2, 0], [1, 0, 2], [1, -2, -2], [1, 1, -1], [1, -1, 1]], float)
WEIGHTS = np.array([[np.log(7/5), 2, -.5], [np.log(8/5), -.5, 2], [0, 0, 0]])
CONTEXT = softmax(DESIGNS @ WEIGHTS.T, axis=1)
EDGES = np.array(list(combinations(range(3), 2)))  # AB, AC, BC
MENUS = (np.array([0, 1]), np.array([0, 1, 2]))
VOTES = 200
REP = 20


def population(condition):
    h = np.log(9)
    if condition == "cyclic":
        skills = h * np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], float)
    elif condition == "shared-order":
        skills = h * np.outer([1, 0, -1], [.7, 1., 1.3])
    else:
        raise ValueError(condition)
    return CONTEXT @ expit(skills[EDGES[:, 0]] - skills[EDGES[:, 1]]).T


def menu_scores(pair_p):
    table = np.full((3, 3), .5)
    a, b = EDGES.T
    table[a, b], table[b, a] = pair_p, 1 - pair_p
    return [(table[np.ix_(menu, menu)].sum(1) - .5) / (len(menu) - 1) for menu in MENUS]


def evaluate(pred, target):
    scores = menu_scores(target)
    route = [menu[np.argmax(s)] for menu, s in zip(MENUS, menu_scores(pred))]
    # Menus are sorted prefixes of [A, B, C], with local and global indices identical.
    loss = [(s.max() - s[idx]) for s, idx in zip(scores, route)]
    return dict(choice_ab="ABC"[route[0]], choice_abc="ABC"[route[1]], mean_regret=np.mean(loss),
                both_correct=bool(np.allclose(loss, 0., atol=1e-12, rtol=0)),
                target_pair_mae=np.abs(pred - target).mean())


def scalar_floor(target):
    scores = menu_scores(target)
    costs = []
    for order in permutations(range(3)):
        picks = [next(i for i in order if i in menu) for menu in MENUS]
        costs.append(np.mean([s.max() - s[pick] for s, pick in zip(scores, picks)]))
    return min(costs)


def replicate(spec):
    rep, condition, seed = spec
    probs = population(condition)
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(len(DESIGNS)), len(EDGES) * VOTES)
    e = np.tile(np.repeat(np.arange(len(EDGES)), VOTES), len(DESIGNS))
    F, (a, b) = DESIGNS[t], EDGES[e].T
    y = (rng.random(len(t)) < probs[t, e]).astype(float)
    target_x = np.repeat(DESIGNS[:1], len(EDGES), axis=0)
    params = dict(K=3, lam_W=.2, lam_R=.2)
    models = {"probability mixture": [], "scalar BT": []}
    for _ in range(3):
        init_w = rng.normal(scale=.6, size=(3, F.shape[1]))
        init_w[-1] = 0
        init_r = rng.normal(scale=.6, size=(3, 3))
        init_r -= init_r.mean(0)
        mix = StyleMixtureBT(style=None, max_iter=20, polish_all=True, n_init=1, **params)
        mix.fit(F, a, b, y, M=3, init_params=(init_w, init_r, np.empty((3, 0))))
        scalar = CapabilityBT(shrink_to="global", **params)
        scalar.fit(F, a, b, y, M=3, W0=init_w, R0=init_r)
        for name, obj, model in (("probability mixture", mix.history_.objective[-1], mix),
                                 ("scalar BT", scalar.objective_, scalar)):
            pred = model.predict_proba(target_x, *EDGES.T)
            models[name].append((obj, pred))
    rows = []
    predictions = {name: max(starts, key=lambda z: z[0])[1] for name, starts in models.items()}
    wins = np.bincount(e[t == 0], weights=y[t == 0])
    predictions["target pair frequencies"] = (wins + .5) / (VOTES + 1)
    for name, p in predictions.items():
        rows.append(dict(rep=rep, condition=condition, seed=seed, method=name, **evaluate(p, probs[0])))
    return rows


def run(workers, output):
    specs = []
    for rep, stream in enumerate(np.random.SeedSequence(577).spawn(REP)):
        seed = int(stream.generate_state(1)[0])
        specs.extend((rep, condition, seed) for condition in ("cyclic", "shared-order"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for result in pool.map(replicate, specs):
            rows.extend(result)
            print(f"Completed {result[0]['condition']} replication {result[0]['rep']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(output / "replicates.csv", index=False)
    summary = runs.groupby(["condition", "method"]).agg(
        replications=("rep", "size"), mean_regret=("mean_regret", "mean"), regret_mcse=("mean_regret", "sem"),
        both_correct_runs=("both_correct", "sum"), both_correct_fraction=("both_correct", "mean"),
        pair_mae=("target_pair_mae", "mean")).reset_index()
    summary["any_fixed_order_floor"] = summary.condition.map(lambda c: scalar_floor(population(c)[0]))
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--output", type=Path, default=ROOT / "results/menu_reversal")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.workers, args.output)
