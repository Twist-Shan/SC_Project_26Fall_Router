"""A regularised pairwise text predictor and its equal-weight BT-mixture router."""
import re

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression


# Capped character counts keep an occasional quotation from dominating a script flag.
_SCRIPTS = [
    re.compile(pattern) for pattern in (
        r"[\u3400-\u9fff]", r"[\u3040-\u30ff]", r"[\uac00-\ud7af]",
        r"[\u0400-\u052f]", r"[\u0600-\u06ff]", r"[\u0900-\u097f]",
        r"[\u00c0-\u024f]", r"[\u0e00-\u0e7f]",
    )
]
_REQUEST = [
    re.compile(pattern, re.I) for pattern in (
        r"\b(python|javascript|java|sql|function|programming|code|debug|compiler)\b|#[ ]*include|\x60{3}|~~~",
        r"\\frac|\\sum|\\int|\b(solve|equation|probability|algebra|calculus|proof|integral)\b",
        r"\b(translate|translation|language|english|chinese|japanese|korean|french|spanish)\b",
        r"\b(story|poem|creative|fiction|lyrics|roleplay|character)\b",
        r"\b(rewrite|summarize|summarise|format|list|json|table|markdown)\b",
    )
]


class LexicalBT:
    """Candidate-specific text scores and antisymmetric, ridge-shrunk pair offsets.

    The feature map is fixed so one can encode prompts before grouping folds.
    A larger C means weaker ridge regularisation. The model uses only the prompt
    and the candidate identities when producing a preference probability.
    """

    def __init__(self, C=0.3):
        self.C = C
        self.vectorizer = HashingVectorizer(
            n_features=2048, ngram_range=(1, 2), alternate_sign=True, norm="l2",
            token_pattern=r"(?u)\b\w+\b",
        )

    def encode(self, prompts):
        text = [s[:3000] for s in prompts]
        named = []
        for s in text:
            scripts = [min(len(p.findall(s)), 8) / 8 for p in _SCRIPTS]
            request = [float(p.search(s) is not None) for p in _REQUEST]
            named.append([6., np.log1p(len(s)) / 4, min(s.count("\n"), 12) / 12,
                          min(s.count("    "), 4) / 4, *scripts, *request])
        return sparse.hstack([sparse.csr_matrix(named), self.vectorizer.transform(text)],
                             format="csr")

    @staticmethod
    def pair_design(features, a, b, M):
        features = sparse.csr_matrix(features)
        a, b = np.asarray(a), np.asarray(b)
        n, width = features.shape
        if a.shape != (n,) or b.shape != (n,) or np.any(a == b):
            raise ValueError("Provide two different candidate IDs per prompt")
        if not (np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)
                and np.all((a >= 0) & (a < M) & (b >= 0) & (b < M))):
            raise ValueError("Candidate IDs must be integer indices in [0, M)")
        a, b = a.astype(np.intp, copy=False), b.astype(np.intp, copy=False)
        row_lengths = np.diff(features.indptr)

        def item_block(ids):
            indices = features.indices + np.repeat(ids * width, row_lengths)
            return sparse.csr_matrix((features.data, indices, features.indptr),
                                     shape=(n, M * width))

        lo, hi = np.minimum(a, b), np.maximum(a, b)
        pair = sparse.csr_matrix((0.5 * np.sign(b - a), (np.arange(n), lo * M + hi)),
                                 shape=(n, M * M))
        return sparse.hstack((item_block(a) - item_block(b), pair), format="csr")

    def fit_design(self, design, y, M):
        """Fit a precomputed pair design; useful for paired validation fits."""
        y = np.asarray(y)
        if len(y) == 0 or len(y) != design.shape[0] or not np.isin(y, (0, 1)).all():
            raise ValueError("One binary outcome is required per row")
        weights = None
        if np.unique(y).size == 1:
            # Mirroring one row preserves its loss and admits both class labels.
            design = sparse.vstack([design, -design[0]], format="csr")
            weights = np.ones(len(y) + 1)
            weights[[0, -1]] = 0.5
            y = np.r_[y, 1 - y[0]]
        self.model_ = LogisticRegression(
            C=self.C, fit_intercept=False, solver="liblinear", dual=True,
            max_iter=1500, tol=1e-5, random_state=43,
        ).fit(design, y, sample_weight=weights)
        self.M_ = M
        coef = self.model_.coef_.ravel()
        offset_start = len(coef) - M * M
        self.items_ = coef[:offset_start].reshape(M, -1)
        offsets = 0.5 * coef[offset_start:].reshape(M, M)
        self.offsets_ = offsets - offsets.T
        return self

    def fit(self, prompts, a, b, y, M):
        return self.fit_design(self.pair_design(self.encode(prompts), a, b, M), y, M)

    def predict_proba(self, prompts, a, b):
        design = self.pair_design(self.encode(prompts), a, b, self.M_)
        return self.model_.predict_proba(design)[:, 1]

    def borda(self, prompts, candidates):
        """Return one uniform-opponent mean win rate per prompt and candidate."""
        c = np.asarray(candidates)
        if c.ndim != 1 or len(c) < 2 or len(np.unique(c)) != len(c):
            raise ValueError("Provide at least two distinct candidates")
        if not np.issubdtype(c.dtype, np.integer) or np.any((c < 0) | (c >= self.M_)):
            raise ValueError("Candidate ID outside the fitted set")
        logits = self.encode(prompts) @ self.items_[c].T
        offset = self.offsets_[np.ix_(c, c)]
        out = np.empty_like(logits)
        for i in range(0, len(logits), 128):
            block = logits[i:i + 128]
            p = expit(block[:, :, None] - block[:, None, :] + offset)
            out[i:i + len(block)] = (p.sum(axis=2) - 0.5) / (len(c) - 1)
        return out


class LexicalMixtureRouter:
    """Equal-weight linear pool of a fitted prompt-only mixture and a fitted LexicalBT."""

    def __init__(self, mixture, lexical):
        if mixture.style is not None or mixture.M != lexical.M_:
            raise ValueError("Use a prompt-only mixture and matching candidate indices")
        self.mixture, self.lexical = mixture, lexical

    def predict_proba(self, features, prompts, a, b):
        if len(features) != len(prompts):
            raise ValueError("Provide a prompt for each mixture feature row")
        return 0.5 * (self.mixture.predict_proba(features, a, b)
                      + self.lexical.predict_proba(prompts, a, b))

    def route(self, features, prompts, candidates=None):
        if len(features) != len(prompts):
            raise ValueError("Provide a prompt for each mixture feature row")
        c = np.arange(self.lexical.M_) if candidates is None else np.asarray(candidates)
        text_scores = self.lexical.borda(prompts, c)
        _, mixture_scores = self.mixture.route(features, c)
        scores = 0.5 * (mixture_scores + text_scores)
        return c[np.argmax(scores, axis=1)], scores
