"""Hierarchical, prompt-conditional Bradley--Terry scores using frozen embeddings."""
import re

import numpy as np
from scipy import sparse
from scipy.special import expit, softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from .lexical import LexicalBT


def family_name(name):
    name = name.lower()
    for prefix in ("gpt-3.5", "gpt-4", "chatglm", "qwen", "llama"):
        if name.startswith(prefix):
            return prefix
    return re.split(r"[-.]", name, maxsplit=1)[0]


class SemanticBasis:
    """Training-only linear directions and soft memberships in semantic clusters."""

    def fit(self, embeddings):
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
            raise ValueError("A finite training matrix of at least three embeddings is required")
        d = min(48, len(x) - 1, x.shape[1])
        self.pca = PCA(d, svd_solver="randomized", random_state=43).fit(x)
        proj = self.project(x)
        self.centres = normalize(MiniBatchKMeans(
            n_clusters=min(24, len(x)), n_init=3, batch_size=512, random_state=43,
        ).fit(proj).cluster_centers_)
        return self

    def project(self, embeddings):
        components = self.pca.transform(np.asarray(embeddings, dtype=np.float32))
        # Partial whitening preserves large semantic directions without letting
        # the leading component dominate the cosine similarities.
        return normalize(components / np.maximum(self.pca.explained_variance_, 1e-7) ** 0.25)

    def transform(self, embeddings):
        z = self.project(embeddings)
        memberships = softmax(8.0 * (z @ self.centres.T), axis=1)
        return np.column_stack([z, memberships - 1.0 / memberships.shape[1]])


def interaction_design(z, a, b, n_items):
    """One sparse block per item with opposite signs for the two competitors."""
    z = sparse.csr_matrix(z)
    lengths = np.diff(z.indptr)
    n, width = z.shape

    def block(ids):
        cols = z.indices + np.repeat(np.asarray(ids, dtype=np.intp) * width, lengths)
        return sparse.csr_matrix((z.data, cols, z.indptr), shape=(n, width * n_items))

    return block(a) - block(b)


class SemanticFamilyBT:
    """A joint lexical, semantic and family-pooled preference predictor.

    For a model m, the semantic score is the sum of a model-specific term and
    a shared term for its family. Singleton families use the model-specific
    term alone. An antisymmetric pair offset captures residual pair preferences.
    """

    def __init__(self, model_names, C=0.03, semantic_scale=3.0):
        self.names = tuple(model_names)
        self.M = len(self.names)
        families = [family_name(m) for m in self.names]
        repeated = sorted({f for f in families if families.count(f) > 1})
        self.family = np.asarray([repeated.index(f) if f in repeated else len(repeated)
                                  for f in families], dtype=np.intp)
        self.n_family = len(repeated)
        self.C, self.semantic_scale = C, semantic_scale
        self.lexical = LexicalBT()

    def semantic_design(self, z, a, b):
        items = interaction_design(z, a, b, self.M)
        fam = interaction_design(z, self.family[a], self.family[b], self.n_family + 1)
        # Drop the final block, which represents the singleton-family sentinel.
        return sparse.hstack([items, 1.5 * fam[:, :self.n_family * z.shape[1]]], format="csr")

    def fit_joint(self, design, y, lexical_width, semantic_width):
        y = np.asarray(y)
        if len(y) != design.shape[0] or not np.isin(y, (0, 1)).all() or len(y) == 0:
            raise ValueError("One binary outcome is required per row")
        weight = None
        if np.unique(y).size == 1:
            design = sparse.vstack((design, -design[0]), format="csr")
            y = np.r_[y, 1 - y[0]]
            weight = np.ones(len(y))
            weight[[0, -1]] = 0.5
        self.estimator = LogisticRegression(
            C=self.C, fit_intercept=False, dual=True, solver="liblinear",
            max_iter=1000, tol=1e-5, random_state=43,
        ).fit(design, y, sample_weight=weight)
        coef = self.estimator.coef_[0]
        pos = self.M * lexical_width
        self.words = coef[:pos].reshape(self.M, lexical_width)
        stop = pos + self.M ** 2
        offset = 0.5 * coef[pos:stop].reshape(self.M, self.M)
        self.offset = offset - offset.T
        sem = self.semantic_scale * coef[stop:].reshape(self.M + self.n_family, semantic_width)
        shared = np.vstack([1.5 * sem[self.M:], np.zeros((1, semantic_width))])
        self.semantic = sem[:self.M] + shared[self.family]
        return self

    def fit(self, prompts, embeddings, a, b, y):
        if len(prompts) != len(embeddings):
            raise ValueError("Provide one embedding per prompt")
        a, b = np.asarray(a), np.asarray(b)
        words = self.lexical.encode(prompts)
        lexical = self.lexical.pair_design(words, a, b, self.M)
        self.basis = SemanticBasis().fit(embeddings)
        z = self.basis.transform(embeddings)
        design = sparse.hstack([lexical, self.semantic_scale * self.semantic_design(z, a, b)],
                               format="csr")
        return self.fit_joint(design, y, words.shape[1], z.shape[1])

    def item_scores(self, prompts, embeddings):
        if len(prompts) != len(embeddings):
            raise ValueError("Provide one embedding per prompt")
        return self.lexical.encode(prompts) @ self.words.T + self.basis.transform(embeddings) @ self.semantic.T

    def candidates(self, candidates):
        c = np.asarray(candidates)
        if c.ndim != 1 or not np.issubdtype(c.dtype, np.integer) or np.any((c < 0) | (c >= self.M)):
            raise ValueError("Candidates must be valid model indices")
        return c

    def predict_proba(self, prompts, embeddings, a, b):
        a, b = self.candidates(a), self.candidates(b)
        if len(a) != len(prompts) or a.shape != b.shape or np.any(a == b):
            raise ValueError("Provide two distinct candidates per prompt")
        scores = self.item_scores(prompts, embeddings)
        ix = np.arange(len(a))
        return expit(scores[ix, a] - scores[ix, b] + self.offset[a, b])

    def borda(self, prompts, embeddings, candidates):
        c = self.candidates(candidates)
        if len(c) < 2 or len(np.unique(c)) != len(c):
            raise ValueError("Provide at least two distinct candidates")
        scores = self.item_scores(prompts, embeddings)[:, c]
        out = np.empty_like(scores)
        for i in range(0, len(scores), 128):
            x = scores[i:i + 128]
            probs = expit(x[:, :, None] - x[:, None, :] + self.offset[np.ix_(c, c)])
            out[i:i + len(x)] = (probs.sum(axis=2) - .5) / (len(c) - 1)
        return out


class SemanticMixtureRouter:
    """Equal-weight probability pool with a fitted prompt-only BT mixture."""

    def __init__(self, mixture, semantic):
        if mixture.style is not None or mixture.M != semantic.M:
            raise ValueError("Both routers must share the prompt-only candidate indexing")
        self.mixture, self.semantic = mixture, semantic

    def predict_proba(self, mixture_features, prompts, embeddings, a, b):
        if len(mixture_features) != len(prompts):
            raise ValueError("Provide one prompt per mixture feature row")
        p = self.semantic.predict_proba(prompts, embeddings, a, b)
        return .5 * (p + self.mixture.predict_proba(mixture_features, a, b))

    def route(self, mixture_features, prompts, embeddings, candidates=None):
        if len(mixture_features) != len(prompts):
            raise ValueError("Provide one prompt per mixture feature row")
        c = np.arange(self.semantic.M) if candidates is None else np.asarray(candidates)
        conditional = self.semantic.borda(prompts, embeddings, c)
        _, mixture_scores = self.mixture.route(mixture_features, c)
        scores = .5 * (conditional + mixture_scores)
        return c[scores.argmax(axis=1)], scores
