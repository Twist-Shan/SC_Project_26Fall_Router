"""Interpretable gate features for the Arena prompts.

Two blocks, both computed from the prompt only (so they are available at
routing time):

1. Named prompt categories (regular expressions, in the spirit of Arena's own
   category tags): coding, maths, creative writing, role-play, translation /
   non-English, rewrite & summarise, factual question, advice / how-to,
   reasoning puzzle, greeting & chit-chat, explanation, list / brainstorm,
   sensitive request, instruction constraints, long input, multi-turn, plus
   log length.
2. Topic proportions from non-negative matrix factorisation (NMF) of the
   TF-IDF matrix; each topic is named by its top words.

Output: data/arena_gate_v2.npz (cat, nmf, names, topic words)
"""
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
N_TOPICS = int(os.environ.get("N_TOPICS", "12"))

I = re.I | re.M
CATS = {
    "coding": r"```|\bdef |\bimport |#include|function\s*\(|console\.log|\bSELECT\b.*\bFROM\b|=>|"
              r"\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|html|css|bash|regex|code|coding|"
              r"function|script|api|compile|debug|programming|algorithm|class|json)\b",
    "maths": r"\d+\s*[\+\-\*/\^=<>]\s*\d+|\\frac|\\sqrt|\$[^$]+\$|\b(solve|equation|integral|derivative|"
             r"probability|prove|calculate|math|equals?|square root|percent|sum of|product of|matrix)\b",
    "creative": r"\b(write|compose|create)\b.{0,40}\b(story|poem|song|lyrics|haiku|essay|novel|fiction|limerick|"
                r"sonnet|rap|screenplay|script|tale|fable|joke)\b|\b(story|poem|haiku|lyrics)\b",
    "roleplay": r"\b(act as|pretend|you are a|you're a|role-?play|imagine you|stay in character|as if you were)\b",
    "translate": r"\b(translate|translation|in (french|spanish|german|chinese|japanese|russian|italian|"
                 r"portuguese|korean|arabic))\b",
    "rewrite": r"\b(summari[sz]e|summary|rewrite|re-write|paraphrase|proofread|rephrase|improve (this|the)|"
               r"correct (this|the)|edit (this|the)|tl;?dr)\b",
    "factual": r"^\s*(what|who|when|where|which|how many|how much|is|are|was|were|did|does)\b",
    "advice": r"\b(how (do|can|should) i|how to|should i|tips|recommend|best way|advice|help me)\b",
    "puzzle": r"\b(riddle|puzzle|logic|brain ?teaser|trick question|sisters?|brothers?|apples?|"
              r"how many .* (left|have)|if .* then)\b",
    "chitchat": r"^\s*(hi|hello|hey|yo|sup|good (morning|evening)|how are you)\b|\b(tell me a joke|joke)\b",
    "explain": r"\b(explain|difference between|what is the difference|define|definition|meaning of|"
               r"what does .* mean|why (is|are|do|does))\b",
    "brainstorm": r"\b(list|give me \d+|name \d+|generate|ideas?|suggest|brainstorm|examples? of)\b",
    "sensitive": r"\b(kill|murder|bomb|weapon|gun|hack|hacking|drugs?|cocaine|meth|sex|sexual|nsfw|porn|"
                 r"illegal|steal|racist|suicide|violent|violence|nazi|terror)\w*\b",
    "constraints": r"\b(in \d+ words|exactly \d+|only (answer|respond|reply)|without using|must not|"
                   r"do not use|word count|format|in json|bullet points|one sentence|single word)\b",
}
NICE = {"coding": "coding", "maths": "maths", "creative": "creative writing", "roleplay": "role-play",
        "translate": "translation", "rewrite": "rewrite / summarise", "factual": "factual question",
        "advice": "advice / how-to", "puzzle": "reasoning puzzle", "chitchat": "greeting / joke",
        "explain": "explanation", "brainstorm": "list / brainstorm", "sensitive": "sensitive request",
        "constraints": "format constraints", "long_input": "long input (>1000 chars)",
        "multi_turn": "multi-turn", "non_english": "non-English", "log_length": "log prompt length"}


def categories(first, n_turns):
    rows = {}
    for name, pat in CATS.items():
        rx = re.compile(pat, I)
        rows[name] = np.array([bool(rx.search(s[:2000])) for s in first], float)
    L = np.array([len(s) for s in first])
    rows["long_input"] = (L > 1000).astype(float)
    rows["multi_turn"] = (np.asarray(n_turns) > 1).astype(float)
    rows["non_english"] = np.array([sum(ord(c) > 127 for c in s) / max(len(s), 1) > 0.2 for s in first], float)
    rows["log_length"] = np.log1p(L)
    return pd.DataFrame(rows)


def main():
    with open(os.path.join(DATA, "arena_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    z = np.load(os.path.join(DATA, "arena_processed.npz"), allow_pickle=True)
    split = z["split"]
    cat = categories(meta["first"], meta["n_turns"])
    print("category prevalence:\n", cat.drop(columns="log_length").mean().round(3).to_string())
    text = pd.Series(meta["first"]).str.slice(0, 3000).str.lower()
    X = meta["tfidf"].transform(text)
    nmf = NMF(N_TOPICS, init="nndsvda", random_state=0, max_iter=400).fit(X[split == "train"])
    T = nmf.transform(X)
    T = T / (T.sum(1, keepdims=True) + 1e-12)  # topic proportions
    vocab = np.array(meta["tfidf"].get_feature_names_out())
    words = [list(vocab[np.argsort(-c)[:8]]) for c in nmf.components_]
    for j, w in enumerate(words):
        print(f"topic {j + 1:2d}: {', '.join(w)}")
    np.savez_compressed(os.path.join(DATA, "arena_gate_v2.npz"), cat=cat.values, cat_names=np.array(cat.columns),
                        nmf=T, topic_words=np.array([", ".join(w) for w in words]))


if __name__ == "__main__":
    main()
