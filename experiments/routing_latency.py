"""Measure single-prompt routing on supplied embeddings and a fixed pool."""
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixbt import StyleMixtureBT, make_params


def uncached_route(model, F, candidates):
    skills = model.R_[candidates]
    probs = expit(skills[:, None, :] - skills[None, :, :])
    i = np.arange(len(skills))
    probs[i, i] = 0.0
    strength = probs.sum(axis=1) / (len(skills) - 1)
    scores = model.gate(F) @ strength.T
    return candidates[np.argmax(scores, axis=1)], scores


def benchmark():
    rng = np.random.default_rng(61)
    M, K, p = 64, 6, 29
    model = StyleMixtureBT(K=K, style=None)
    model.M = M
    model.W_, model.R_ = make_params(M, K, p, rng=rng)
    F = rng.normal(size=(500, p))
    F[:, 0] = 1.0
    candidates = np.arange(M)
    rows = []
    for size in (2, 8, 64):
        pool = candidates[:size]
        # Check all requests and warm the one-pool cache before measuring.
        for left, right in zip(model.route(F, pool), uncached_route(model, F, pool)):
            np.testing.assert_array_equal(left, right)
        for rep in range(7):
            times = {}
            variants = (("uncached", uncached_route), ("cached", StyleMixtureBT.route))
            for name, route in (variants if rep % 2 == 0 else reversed(variants)):
                start = perf_counter()
                for i in range(len(F)):
                    route(model, F[i:i + 1], pool)
                times[name] = (perf_counter() - start) * 1e6 / len(F)
            rows.append(dict(repetition=rep + 1, pool=size, K=K, features=p, prompts=len(F),
                             uncached_us=times["uncached"], cached_us=times["cached"],
                             speedup=times["uncached"] / times["cached"]))
    output = ROOT / "results" / "routing_latency.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for size in (2, 8, 64):
        selected = [r for r in rows if r["pool"] == size]
        print(size, {k: np.median([r[k] for r in selected]) for k in ("uncached_us", "cached_us", "speedup")})


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        benchmark()
