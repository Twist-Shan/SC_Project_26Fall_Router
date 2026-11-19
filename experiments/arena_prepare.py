"""Prepare Chatbot Arena (lmarena-ai/arena-human-preference-55k) for the model.

Input : data/train.csv  (columns id, model_a, model_b, prompt, response_a,
        response_b, winner_model_a, winner_model_b, winner_tie)
Output: data/arena_processed.npz, data/arena_meta.pkl, data/arena_slim.csv

Prompt features f(x) (all computed from the *prompt only*, so they are
available at routing time):
  * 32-dim LSA embedding: TF-IDF (uni+bigrams) of the first user turn,
    reduced by truncated SVD -- fitted on the training split only;
  * 6 interpretable features: log length, multi-turn, code markers,
    math markers, share of non-ASCII characters, question mark.
All features are standardised (training mean/sd) and an intercept is added.
"""
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
D_LSA = int(os.environ.get("D_LSA", "32"))
SEED = 2026

CODE_RE = re.compile(
    r"```|\bdef |\bimport |#include|public static|function\s*\(|console\.log|"
    r"\bSELECT\b.*\bFROM\b|=>|;\s*$|^\s*(for|while|if)\s*\(.*\)\s*\{|"
    r"\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|html|css|bash|regex|"
    r"code|coding|function|script|api|compile|debug)\b", re.M | re.I)
MATH_RE = re.compile(
    r"\d+\s*[\+\-\*/\^=<>]\s*\d+|\\frac|\\sqrt|\$[^$]+\$|\bsolve\b|\bequation\b|"
    r"\bintegral\b|\bderivative\b|\bprobability\b|\bprove\b|\bcalculate\b|\bmath\b|"
    r"\bequals?\b|\bsquare root\b|\bpercent", re.I)


def parse_turns(s):
    try:
        t = json.loads(s)
        if isinstance(t, list):
            return [x if isinstance(x, str) else "" for x in t]
    except Exception:
        pass
    return [str(s)]


def clean(s):
    return s.encode("utf-8", "replace").decode("utf-8", "replace")


def handcrafted(first, n_turns):
    L = np.array([len(s) for s in first])
    non_ascii = np.array([sum(ord(c) > 127 for c in s) / max(len(s), 1) for s in first])
    return pd.DataFrame({
        "log_length": np.log1p(L),
        "multi_turn": (np.asarray(n_turns) > 1).astype(float),
        "code": np.array([bool(CODE_RE.search(s)) for s in first], float),
        "math": np.array([bool(MATH_RE.search(s)) for s in first], float),
        "non_ascii": non_ascii,
        "question": np.array(["?" in s for s in first], float),
    })


def main(path=os.path.join(DATA, "train.csv")):
    df = pd.read_csv(path)
    print("raw rows", len(df))
    # prompts and votes without the (large) response columns
    df.drop(columns=[c for c in ["response_a", "response_b"] if c in df.columns]).to_csv(
        os.path.join(DATA, "arena_slim.csv"), index=False)
    turns = df["prompt"].map(parse_turns)
    df["n_turns"] = turns.map(len)
    df["first"] = turns.map(lambda t: clean(t[0]) if t else "")
    df["y"] = np.where(df.winner_model_a == 1, 1.0, np.where(df.winner_model_b == 1, 0.0, 0.5))
    df = df[df["first"].str.strip().str.len() > 0].reset_index(drop=True)

    models = sorted(set(df.model_a) | set(df.model_b))
    idx = {m: i for i, m in enumerate(models)}
    df["a"] = df.model_a.map(idx)
    df["b"] = df.model_b.map(idx)

    rng = np.random.default_rng(SEED)
    u = rng.random(len(df))
    split = np.where(u < 0.8, "train", np.where(u < 0.9, "val", "test"))
    df["split"] = split
    tr = split == "train"

    text = df["first"].str.slice(0, 3000).str.lower()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=30000,
                            sublinear_tf=True, stop_words="english")
    X_tr = tfidf.fit_transform(text[tr])
    svd = TruncatedSVD(D_LSA, random_state=SEED).fit(X_tr)
    X = tfidf.transform(text)
    Z = svd.transform(X)
    H = handcrafted(df["first"].tolist(), df["n_turns"].values)
    feats = np.column_stack([Z, H.values])
    names = [f"lsa{j + 1}" for j in range(D_LSA)] + list(H.columns)
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-12
    F = np.column_stack([np.ones(len(df)), (feats - mu) / sd])

    np.savez_compressed(os.path.join(DATA, "arena_processed.npz"), F=F, a=df.a.values,
                        b=df.b.values, y=df.y.values, split=split,
                        raw_hand=H.values)
    meta = dict(models=models, feature_names=["intercept"] + names, tfidf=tfidf, svd=svd,
                hand_names=list(H.columns), first=df["first"].tolist(), ids=df["id"].tolist(),
                n_turns=df["n_turns"].values)
    with open(os.path.join(DATA, "arena_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    dec = df.y != 0.5
    print(f"battles {len(df)}  decisive {dec.sum()}  ties {(~dec).sum()}  models {len(models)}")
    print("split sizes", pd.Series(split).value_counts().to_dict())
    print("feature dim p =", F.shape[1], " explained var (LSA) = %.3f" % svd.explained_variance_ratio_.sum())
    print(H.mean().round(3).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:])
