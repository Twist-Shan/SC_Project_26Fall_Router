"""Separate the value of observing the prompt from the choice of BT formulation.

The five settings use the eight-candidate coverage study's skill profiles and
the same training and testing distribution. Both fitted prompt-aware methods
receive the same training battles, prompt features, component count, penalties
and two starting parameter vectors. The oracle uses all eight candidates.
"""
import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, softmax, roots_hermitenorm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixbt import CapabilityBT, GlobalBT, StyleMixtureBT


BASE = np.array([0.35, 0.30, 0.18, 0.10, 0.03, -0.15, -0.23, -0.34])
CONTRAST = np.array([1.0, -1.0, 0.55, -0.55, 0.25, -0.25, 0.1, -0.1])
W = np.array([[0.0, 1.7, -0.8], [0.0, 0.0, 0.0]])
EDGES = np.array(list(combinations(range(len(BASE)), 2)))
SETTINGS = (("homogeneous", 0.0), ("probability", 0.4), ("utility", 0.4),
            ("probability", 1.4), ("utility", 1.4))


def skills(delta):
    values = BASE[:, None] + delta * np.column_stack((CONTRAST, -CONTRAST))
    return values - values.mean(axis=0)


def pair_probabilities(pi, r, generator):
    """Probability of the first candidate winning on all unordered edges."""
    difference = r[EDGES[:, 0]] - r[EDGES[:, 1]]
    if generator == "utility":
        return expit(pi @ difference.T)
    return pi @ expit(difference).T


def borda_scores(p):
    scores = np.zeros((len(p), len(BASE)))
    for column, (a, b) in enumerate(EDGES):
        scores[:, a] += p[:, column]
        scores[:, b] += 1 - p[:, column]
    return scores / (len(BASE) - 1)


def best_fixed_candidate(r, generator):
    # The random softmax logit is one-dimensional Gaussian under this design.
    # Integrate over that distribution before maximising over constant choices.
    nodes, mass = roots_hermitenorm(96)
    pi1 = expit(nodes * np.linalg.norm(W[0, 1:]))
    pi = np.column_stack((pi1, 1 - pi1))
    scores = borda_scores(pair_probabilities(pi, r, generator))
    return int((mass @ scores).argmax())


def fit_methods(f, a, b, y, seed):
    m = len(BASE)
    rng = np.random.default_rng(seed)
    fixed = GlobalBT(lam_R=0.1).fit(a, b, y, m)
    best_em = best_scalar = None
    for _ in range(2):
        w0 = rng.normal(scale=0.5, size=W.shape)
        w0[-1] = 0
        r0 = fixed.r_[:, None] + rng.normal(scale=0.3, size=(m, 2))
        r0 -= r0.mean(0)
        em = StyleMixtureBT(K=2, style=None, lam_W=1, lam_R=2, n_init=1,
                            max_iter=30, polish_all=True).fit(
                                f, a, b, y, M=m, init_params=(w0, r0, np.empty((2, 0))))
        scalar = CapabilityBT(K=2, lam_W=1, lam_R=2, shrink_to="global").fit(
            f, a, b, y, m, W0=w0.copy(), R0=r0.copy())
        if best_em is None or em.history_.objective[-1] > best_em.history_.objective[-1]:
            best_em = em
        if best_scalar is None or scalar.objective_ > best_scalar.objective_:
            best_scalar = scalar
    return fixed, best_em, best_scalar


def sample_prompts(n, rng):
    return np.column_stack((np.ones(n), rng.normal(size=(n, W.shape[1] - 1))))


def replicate(rep, stream, n_train, n_test):
    train_stream, test_stream, fit_stream = stream.spawn(3)
    train_rng = np.random.default_rng(train_stream)
    f = sample_prompts(n_train, train_rng)
    fit_seed = int(fit_stream.generate_state(1)[0])
    edge = train_rng.integers(len(EDGES), size=n_train)
    a, b = EDGES[edge].T
    uniform = train_rng.random(n_train)
    test = sample_prompts(n_test, np.random.default_rng(test_stream))
    train_pi = softmax(f @ W.T, axis=1)
    test_pi = softmax(test @ W.T, axis=1)
    rows = []
    for generator, delta in SETTINGS:
        r = skills(delta)
        p = pair_probabilities(train_pi, r, generator)[np.arange(n_train), edge]
        y = (uniform < p).astype(float)
        target_p = pair_probabilities(test_pi, r, generator)
        target = borda_scores(target_p)
        best = target.max(axis=1)
        fixed_oracle = best_fixed_candidate(r, generator)
        value_of_prompt = float(np.mean(best - target[:, fixed_oracle]))
        fixed, em, scalar = fit_methods(f, a, b, y, fit_seed)
        predictions = {
            "Global BT": np.broadcast_to(fixed.predict_proba(None, EDGES[:, 0], EDGES[:, 1]), target_p.shape),
            "EM mixture": pair_probabilities(em.gate(test), em.R_, "probability"),
            "Scalar BT": pair_probabilities(scalar.gate(test), scalar.R_, "utility"),
        }
        for method, predicted in predictions.items():
            selection = borda_scores(predicted).argmax(axis=1)
            if method == "EM mixture" and not np.array_equal(selection, em.route(test)[0]):
                raise ValueError("Explicit all-pair scores disagree with the deployed EM rule")
            chosen_truth = target[np.arange(n_test), selection]
            regret = np.mean(best - chosen_truth)
            kl = target_p * (np.log(target_p) - np.log(predicted))
            kl += (1 - target_p) * (np.log1p(-target_p) - np.log1p(-predicted))
            rows.append(dict(rep=rep, generator=generator, delta=delta, method=method,
                             n_train=n_train, n_test=n_test, fixed_oracle=fixed_oracle,
                             prompt_value=value_of_prompt, regret=regret,
                             gain_over_best_fixed=value_of_prompt - regret,
                             correct_choice=np.mean(selection == target.argmax(axis=1)),
                             excess_logloss=kl.mean()))
    return rows


def summarise(runs):
    metrics = ["prompt_value", "regret", "gain_over_best_fixed", "correct_choice", "excess_logloss"]
    by_method = runs.groupby(["generator", "delta", "method"], as_index=True)[metrics]
    summary = by_method.agg(["mean", "sem"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    rows = []
    for (generator, delta), group in runs.groupby(["generator", "delta"]):
        regret = group.pivot(index="rep", columns="method", values="regret")
        for baseline in ["Global BT", "Scalar BT"]:
            gain = regret[baseline] - regret["EM mixture"]
            rows.append(dict(generator=generator, delta=delta, baseline=baseline,
                             paired_reps=len(gain), regret_reduction=gain.mean(),
                             regret_reduction_mcse=gain.sem()))
    return summary, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--train", type=int, default=2400)
    parser.add_argument("--test", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=593)
    parser.add_argument("--output", type=Path, default=ROOT / "results/prompt_value")
    args = parser.parse_args()
    if min(args.reps, args.train, args.test) < 1:
        parser.error("replications, training size and test size must be positive")
    rows = []
    with threadpool_limits(limits=1):
        for rep, stream in enumerate(np.random.SeedSequence(args.seed).spawn(args.reps)):
            rows.extend(replicate(rep, stream, args.train, args.test))
            print(f"Finished replication {rep + 1}/{args.reps}", flush=True)
    runs = pd.DataFrame(rows)
    summary, contrasts = summarise(runs)
    args.output.mkdir(exist_ok=True, parents=True)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv")
    contrasts.to_csv(args.output / "contrasts.csv", index=False)
    print(contrasts.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
