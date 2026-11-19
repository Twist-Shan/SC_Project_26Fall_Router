"""Prompt-only BT routing with restart averaging and optional shrunk pair corrections."""
import numpy as np
from scipy.special import expit, logit
from .extended import StyleMixtureBT


def pair_offsets(a, b, y, p, count, ridge=12.5):
    """Unit log-odds offsets oriented toward the smaller ID; unseen edges stay zero."""
    edge = np.minimum(a,b)*count + np.maximum(a,b)
    low_first = a < b
    base = logit(np.clip(np.where(low_first,p,1-p),1e-12,1-1e-12))
    target = np.where(low_first,y,1-y)
    delta = np.zeros(count*count)
    for _ in range(50):
        fitted = expit(base+delta[edge])
        score = np.bincount(edge,weights=target-fitted,minlength=delta.size)-ridge*delta
        curvature = np.bincount(edge,weights=fitted*(1-fitted),minlength=delta.size)+ridge
        step = np.clip(score/curvature,-1,1)
        delta += step
        if np.max(np.abs(step))<1e-9:
            return delta.reshape(count,count)
    raise RuntimeError("Pair offsets did not converge")


class RestartRouter:
    """Fit the same initial points as the best-start estimator; keep all component fits."""

    def __init__(self, config, seed=1729, starts=3):
        if starts < 1:
            raise ValueError("At least one start is required")
        self.config = dict(config)
        self.seed = seed
        self.starts = starts

    def fit(self, F, a, b, y, M):
        self.M = M
        self.models = []
        rng = np.random.default_rng(self.seed)
        for _ in range(self.starts):
            model = StyleMixtureBT(**self.config, style=None, polish_all=True, max_iter=30, seed=self.seed)
            model.M = M
            bd = model._bd(a, b, y, None)
            model.q = bd.q
            model._fit_prior(bd)
            initial = model._init(F, bd, rng)
            model.fit(F, a, b, y, M=M, init_params=initial)
            self.models.append(model)
        self.best = max(self.models, key=lambda fitted: fitted.history_.objective[-1])
        self.offsets = {}
        for average in (False, True):
            p = self.predict_proba(F, a, b, average=average)
            self.offsets[average] = pair_offsets(a, b, y, p, M)
        return self

    def predict_proba(self, F, a, b, average=True, correct=False):
        models = self.models if average else [self.best]
        prob = np.mean([model.predict_proba(F, a, b) for model in models], axis=0)
        if correct:
            offset = self.offsets[average]
            prob = expit(logit(np.clip(prob, 1e-12, 1-1e-12)) + offset[a,b] - offset[b,a])
        return prob

    def route(self, F, candidates=None, average=True, correct=False):
        pool = np.arange(self.M) if candidates is None else np.asarray(candidates)
        if (pool.ndim != 1 or len(pool) < 2 or len(np.unique(pool)) != len(pool)
                or not np.issubdtype(pool.dtype, np.integer) or np.any((pool < 0) | (pool >= self.M))):
            raise ValueError("Provide at least two distinct fitted candidate IDs")
        if correct:
            scores = np.zeros((len(F), len(pool)))
            for i, u in enumerate(pool):
                for j in range(i+1, len(pool)):
                    p = self.predict_proba(F, np.full(len(F),u), np.full(len(F),pool[j]), average, True)
                    scores[:,i] += p
                    scores[:,j] += 1-p
            scores /= len(pool)-1
        else:
            models = self.models if average else [self.best]
            scores = np.mean([model.route(F, pool)[1] for model in models], axis=0)
        return pool[np.argmax(scores, axis=1)], scores
