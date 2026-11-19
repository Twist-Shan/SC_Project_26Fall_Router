"""Three-outcome lexical BT model with a symmetric, prompt-dependent tie weight."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .lexical import LexicalBT


def outcome_logits(d, t):
    return np.column_stack((0.5 * d, -0.5 * d, t))


def tie_design(features, a, b, M):
    """Tie propensity: intercept, log prompt length, and the two candidate identities."""
    n = features.shape[0]
    item = sparse.csr_matrix((np.ones(2 * n), (np.repeat(np.arange(n), 2),
                             np.column_stack((a, b)).ravel())), shape=(n, M))
    return sparse.hstack((np.ones((n, 1)), features[:, 1:2], item), format="csr")


def davidson_loss_gradient(theta, design, tie_features, y, precision):
    """Summed negative log likelihood plus ridge, outcomes 1=A, 0=B, 2=tie."""
    split = design.shape[1]
    logits = outcome_logits(design @ theta[:split], tie_features @ theta[split:])
    normalizer = logsumexp(logits, axis=1)
    p = np.exp(logits - normalizer[:, None])
    target = np.where(y == 2, 2, 1 - y)
    loss = (normalizer - logits[np.arange(len(y)), target]).sum()
    loss += 0.5 * precision * (theta @ theta)
    d_res = 0.5 * (p[:, 0] - p[:, 1] - (y == 1) + (y == 0))
    grad = np.r_[design.T @ d_res, tie_features.T @ (p[:, 2] - (y == 2))]
    return loss, grad + precision * theta


class DavidsonLexicalBT(LexicalBT):
    """Learn lexical win contrasts from wins, losses and ties.

    predict_proba and borda use P(A wins | decisive). predict_outcomes returns
    unconditional probabilities of A winning, B winning and a tie, in that order.
    """

    def __init__(self, C=0.03):
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be positive and finite")
        super().__init__(C)

    def fit_design(self, design, tie_features, y, M):
        y = np.asarray(y)
        if (y.ndim != 1 or len(y) == 0 or len(y) != design.shape[0] or
                len(y) != tie_features.shape[0] or not np.isin(y, [0, 1, 2]).all()):
            raise ValueError("Provide one outcome in {0,1,2} for every design row")
        y = y.astype(np.intp, copy=False)
        precision = 1 / self.C
        # Rescale coordinates by a diagonal curvature approximation so common
        # candidate intercepts and rare n-grams share an effective step size.
        squared = np.concatenate([np.asarray(x.power(2).sum(axis=0)).ravel()
                                  for x in (design, tie_features)])
        scale = 1 / np.sqrt(precision + 0.25 * squared)

        def objective(v):
            value, grad = davidson_loss_gradient(v * scale, design, tie_features, y, precision)
            return value, grad * scale

        fit = minimize(objective, np.zeros_like(scale), jac=True, method="L-BFGS-B",
                       options={"ftol": 1e-12, "gtol": 1e-5, "maxiter": 500})
        if not fit.success:
            raise RuntimeError(f"Davidson optimisation did not converge: {fit.message}")
        self.n_iter_ = fit.nit
        self.gradient_max_ = float(np.max(np.abs(fit.jac)))
        theta = fit.x * scale
        self.coef_, self.tie_coef_ = np.split(theta, [design.shape[1]])
        self._set_coefficients(self.coef_, M)
        return self

    def fit(self, prompts, a, b, y, M):
        features = self.encode(prompts)
        design = self.pair_design(features, a, b, M)
        return self.fit_design(design, tie_design(features, a, b, M), y, M)

    def predict_proba(self, prompts, a, b):
        design = self.pair_design(self.encode(prompts), a, b, self.M_)
        return expit(design @ self.coef_)

    def predict_outcomes(self, prompts, a, b):
        f = self.encode(prompts)
        d = self.pair_design(f, a, b, self.M_) @ self.coef_
        t = tie_design(f, a, b, self.M_) @ self.tie_coef_
        logits = outcome_logits(d, t)
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
