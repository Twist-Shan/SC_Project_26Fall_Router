"""Benchmark the joint-score optimizations against the previous edge-based fitter.

Run python experiments/score_reuse_benchmark.py after preparing the Arena input.
Each timed fit includes global BT, EM and joint L-BFGS. The training-only text
transform and fixed common initialization run once outside the timing window.
"""
import csv
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from edge_fit_benchmark import ROOT, load_fold_problem
from mixbt.extended import StyleMixtureBT, bt_loglik


class PreviousFit(StyleMixtureBT):
    """Vote-level score and L-BFGS trace callback from release d0572e0."""

    def _score_terms(self, bd, R, B):
        eta = self._eta(bd, R, B)
        return bt_loglik(eta, bd.y), expit(eta)

    def _polish(self, F, bd, maxiter=3000):
        hist = self.history_
        p = F.shape[1]
        t0 = perf_counter()
        last = hist.time[-1] if hist.time else 0.0

        def f_and_g(theta):
            g, objective = self.score(F, bd, *self._unpack(theta, p))
            return -objective, -g

        def record(theta):
            hist.objective.append(self.score(F, bd, *self._unpack(theta, p))[1])
            hist.time.append(last + perf_counter() - t0)

        result = minimize(f_and_g, self._pack(self.W_, self.R_, self.B_), jac=True,
                          method="L-BFGS-B", callback=record,
                          options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 20})
        W, R, B = self._unpack(result.x, p)
        self.W_, self.R_, self.B_ = W, R - R.mean(0), B
        self.n_iter_polish_ = result.nit


def time_fit(cls, cfg, start, F, train, M):
    model = cls(**cfg)
    calls = 0
    evaluate = model.score

    def counted_score(*args):
        nonlocal calls
        calls += 1
        return evaluate(*args)

    model.score = counted_score
    t0 = perf_counter()
    model.fit(F, *train, M=M, init_params=start)
    return model, perf_counter() - t0, calls


def benchmark():
    cfg, start, F, train, held, test_pairs, M = load_fold_problem(6)
    rows = []
    for rep in range(3):
        variants = (("previous", PreviousFit), ("optimized", StyleMixtureBT))
        seconds, scores, fits = {}, {}, {}
        for label, cls in (variants if rep % 2 == 0 else reversed(variants)):
            fits[label], seconds[label], scores[label] = time_fit(cls, cfg, start, F, train, M)
        old, new = fits["previous"], fits["optimized"]
        old_probs = old.predict_proba(held, *test_pairs)
        new_probs = new.predict_proba(held, *test_pairs)
        np.testing.assert_array_equal(old.history_.objective, new.history_.objective)
        np.testing.assert_array_equal(old_probs, new_probs)
        rows.append(dict(rep=rep + 1, K=cfg["K"], train=len(F), test=len(held),
                         previous_seconds=seconds["previous"], optimized_seconds=seconds["optimized"],
                         speedup=seconds["previous"] / seconds["optimized"],
                         previous_score_calls=scores["previous"], optimized_score_calls=scores["optimized"],
                         em_iterations=new.n_iter_em_, polish_iterations=new.n_iter_polish_,
                         objective=new.history_.objective[-1],
                         max_trace_difference=np.max(np.abs(np.subtract(old.history_.objective,
                                                                         new.history_.objective))),
                         max_prediction_difference=np.max(np.abs(old_probs - new_probs))))
        print(rows[-1], flush=True)
    output = ROOT / "results" / "score_reuse_benchmark.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        benchmark()
