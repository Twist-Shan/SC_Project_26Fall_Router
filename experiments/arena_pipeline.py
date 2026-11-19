"""First-prompt data and train-only feature transforms for corrected Arena evaluation.

The target remains the recorded final battle preference. Only the first user
prompt is visible: subsequent user turns and both responses are excluded.
No fitted objects in the legacy cached feature files are loaded here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from arena_features_v2 import CATS, I

ROOT = Path(__file__).resolve().parents[1]
STRONG = ("gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-0314", "gpt-4-0613")


def first_prompt(value):
    """Extract the first recorded user turn; never expose later turns."""
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        parsed = value
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else ""
    return parsed if isinstance(parsed, str) else ""


def prompt_group(text):
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()


def load_data(path=None):
    df = pd.read_csv(path or ROOT / "data" / "arena_slim.csv")
    first = df["prompt"].map(first_prompt)
    keep = first.str.strip().ne("") & df["winner_tie"].eq(0)
    keep &= df["winner_model_a"].add(df["winner_model_b"]).eq(1)
    df, first = df.loc[keep].reset_index(drop=True), first.loc[keep].reset_index(drop=True)
    models = np.array(sorted(set(df.model_a) | set(df.model_b)))
    index = {m: i for i, m in enumerate(models)}
    # Ragged Python strings: fixed-width NumPy Unicode would allocate every row
    # at the longest prompt's width, which can inflate this corpus to gigabytes.
    return dict(text=first.to_numpy(dtype=object), groups=np.array([prompt_group(x) for x in first]),
                ids=df.id.to_numpy(), a=df.model_a.map(index).to_numpy(int),
                b=df.model_b.map(index).to_numpy(int), y=df.winner_model_a.to_numpy(float),
                models=models, M=len(models))


def grouped_folds(groups, n_splits=5, seed=1729):
    """Assign whole prompt groups to approximately equal-sized folds, without labels."""
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < n_splits:
        raise ValueError("Fewer prompt groups than folds")
    rng = np.random.default_rng(seed)
    # Randomise equally sized groups, then greedily balance numbers of battles.
    perm = rng.permutation(len(unique))
    order = perm[np.argsort(-counts[perm], kind="stable")]
    load = np.zeros(n_splits, int)
    assigned = np.empty(len(unique), int)
    for j in order:
        eligible = np.flatnonzero(load == load.min())
        f = int(rng.choice(eligible))
        assigned[j] = f
        load[f] += counts[j]
    return assigned[inverse]


def inner_split(groups, train, seed=2718, fraction=0.2):
    unique = np.unique(groups[train])
    if len(unique) < 2:
        raise ValueError("Need at least two training prompt groups")
    rng = np.random.default_rng(seed)
    n_val = min(len(unique) - 1, max(1, int(np.ceil(fraction * len(unique)))))
    validation_groups = rng.permutation(unique)[:n_val]
    val = train & np.isin(groups, validation_groups)
    return train & ~val, val


def digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def categories(text):
    """Fixed, predeclared rules on the currently visible first prompt only."""
    text = list(text)
    out = {name: np.array([bool(re.search(pattern, s[:2000], I)) for s in text], float)
           for name, pattern in CATS.items()}
    lengths = np.array([len(s) for s in text])
    out["long_input"] = (lengths > 1000).astype(float)
    out["non_english"] = np.array([sum(ord(c) > 127 for c in s) / max(len(s), 1) > .2 for s in text], float)
    out["log_length"] = np.log1p(lengths)
    return pd.DataFrame(out)


class PromptFeatures:
    """Fit vocabulary, IDF, SVD, NMF, scaling and column selection on training text.

    Zero NMF rows receive uniform topic proportions. One topic is omitted from
    the gate, and further exact training collinearities are removed by pivoted QR.
    Selection retains original columns, so feature names remain interpretable.
    """
    def __init__(self, seed=1729, n_topics=12, n_lsa=32, max_features=30000, min_df=5):
        self.seed, self.n_topics, self.n_lsa = seed, n_topics, n_lsa
        self.max_features, self.min_df = max_features, min_df

    @staticmethod
    def _text(text):
        return [s[:3000].lower() for s in text]

    def _raw(self, text):
        cat = categories(text)
        X = self.tfidf_.transform(self._text(text))
        Z = self.svd_.transform(X)
        T = self.nmf_.transform(X)
        total = T.sum(1, keepdims=True)
        T = np.divide(T, total, out=np.full_like(T, 1 / T.shape[1]), where=total > 1e-14)
        named = np.column_stack([cat.to_numpy(), T[:, :-1]])
        return cat, T, Z, named

    def fit(self, text):
        text = np.asarray(text, dtype=object)
        if len(text) < 3:
            raise ValueError("Need at least three training prompts")
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_df=.5,
                                    max_features=self.max_features, sublinear_tf=True, stop_words="english")
        try:
            X = self.tfidf_.fit_transform(self._text(text))
        except ValueError:
            # Small smoke-test corpora can have no term meeting min_df.
            self.tfidf_.set_params(min_df=1, max_df=1.0)
            X = self.tfidf_.fit_transform(self._text(text))
        if X.shape[1] < 2:
            raise ValueError("Training prompts must contain at least two vocabulary terms")
        d = min(self.n_lsa, X.shape[0] - 1, X.shape[1] - 1)
        k = min(self.n_topics, X.shape[0], X.shape[1])
        self.svd_ = TruncatedSVD(d, random_state=self.seed).fit(X)
        self.nmf_ = NMF(k, init="nndsvda", random_state=self.seed, max_iter=400).fit(X)
        cat, _, Z, named = self._raw(text)
        self.category_names_ = list(cat.columns)
        all_names = self.category_names_ + [f"topic_{j + 1}" for j in range(k - 1)]
        self.named_mean_, self.named_sd_ = named.mean(0), named.std(0)
        eligible = np.flatnonzero(self.named_sd_ > 1e-10)
        self.named_sd_ = np.maximum(self.named_sd_, 1e-10)
        normalized = (named - self.named_mean_) / self.named_sd_
        if eligible.size:
            _, r, pivots = qr(normalized[:, eligible], mode="economic", pivoting=True)
            rank = int(np.count_nonzero(np.abs(np.diag(r)) > 1e-8 * max(1.0, abs(r[0, 0]))))
            self.columns_ = np.sort(eligible[pivots[:rank]])
        else:
            self.columns_ = np.array([], int)
        self.gate_names_ = ["intercept"] + [all_names[i] for i in self.columns_]
        self.lsa_mean_, self.lsa_sd_ = Z.mean(0), np.maximum(Z.std(0), 1e-10)
        vocab = self.tfidf_.get_feature_names_out()
        self.topic_words_ = [list(vocab[np.argsort(-c)[:8]]) for c in self.nmf_.components_]
        self.training_text_hash_ = digest(text)
        return self

    def transform(self, text):
        cat, topics, Z, named = self._raw(text)
        named = ((named - self.named_mean_) / self.named_sd_)[:, self.columns_]
        E = (Z - self.lsa_mean_) / self.lsa_sd_
        one = np.ones((len(named), 1))
        return dict(G=np.column_stack([one, named]), F=np.column_stack([one, E, named]),
                    X=np.column_stack([named, E]), E=E, categories=cat, topics=topics)

    def fit_transform(self, text):
        return self.fit(text).transform(text)
