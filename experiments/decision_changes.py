"""Decompose full-coverage selection gains into corrections and new errors."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_probabilities(path, reference, key):
    with np.load(path, allow_pickle=False) as saved:
        for name in ("ids", "fold"):
            if not np.array_equal(saved[name], reference[name]):
                raise ValueError(f"{path} has incompatible {name}")
        p = saved[key].copy()
    if p.shape != reference["ids"].shape or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError(f"Invalid probability vector in {path}")
    return p


def compare(p, baseline, y, group, draws):
    choice = p > 0.5
    changed = choice != (baseline > 0.5)
    corrected = changed & (choice == y)
    lost = changed & (choice != y)
    delta = corrected.astype(int) - lost.astype(int)
    _, group_idx = np.unique(group, return_inverse=True)
    totals = np.bincount(group_idx)
    grouped = np.column_stack([np.bincount(group_idx, weights=v) for v in (changed, corrected, delta)])
    rng = np.random.default_rng(859)
    gains = np.empty(draws)
    wins = []
    for i in range(draws):
        picked = rng.integers(len(totals), size=len(totals))
        switches, right, gain = grouped[picked].sum(axis=0)
        gains[i] = gain / totals[picked].sum()
        if switches:
            wins.append(right / switches)
    gain_lo, gain_hi = np.quantile(gains, [0.025, 0.975])
    win_lo, win_hi = np.quantile(wins, [0.025, 0.975]) if wins else (np.nan, np.nan)
    return dict(n=len(y), changed=int(changed.sum()), corrected=int(corrected.sum()), new_errors=int(lost.sum()),
                change_share=changed.mean(), accuracy_among_changes=corrected.sum() / changed.sum() if changed.any() else np.nan,
                win_lo=win_lo, win_hi=win_hi, overall_gain=delta.mean(), gain_lo=gain_lo, gain_hi=gain_hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/nested")
    parser.add_argument("--semantic-oof", type=Path)
    parser.add_argument("--tie-oof", type=Path)
    parser.add_argument("--semantic-mixture-oof", type=Path)
    parser.add_argument("--capacity-matched-oof", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/decision_changes.csv")
    args = parser.parse_args()
    with (np.load(args.reference / "oof_predictions.npz", allow_pickle=False) as reference,
          np.load(args.reference / "split_manifest.npz", allow_pickle=False) as split):
        if any(not np.array_equal(reference[k], split[k]) for k in ("ids", "fold")):
            raise ValueError("The split manifest and baseline probabilities do not match")
        methods = {key: reference[key] for key in ("global BT", "mixture")}
        if args.semantic_oof:
            methods["lexical pool"] = read_probabilities(args.semantic_oof, reference, "lexical_pool")
            methods["semantic pool"] = read_probabilities(args.semantic_oof, reference, "pred")
        if args.tie_oof:
            lexical = read_probabilities(args.tie_oof, reference, "control")
            if "lexical pool" in methods and not np.allclose(methods["lexical pool"], lexical, rtol=0, atol=1e-8):
                raise ValueError("The two ablations do not reproduce the same lexical control")
            methods["lexical pool"] = lexical
            methods["tie-assisted pool"] = read_probabilities(args.tie_oof, reference, "pred")
        if args.semantic_mixture_oof:
            semantic = read_probabilities(args.semantic_mixture_oof, reference, "semantic_mixture")
            methods["semantic-gated mixture"] = semantic
            methods["equal mixture pool"] = 0.5 * (semantic + methods["mixture"])
        if args.capacity_matched_oof:
            methods["matched-dimension EM"] = read_probabilities(args.capacity_matched_oof, reference, "semantic_mixture")
        rows = []
        for method, p in methods.items():
            if method == "global BT":
                continue
            comparators = ["global BT"]
            if method != "mixture":
                comparators.append("mixture")
            if method not in ("mixture", "lexical pool") and "lexical pool" in methods:
                comparators.append("lexical pool")
            if method == "semantic pool" and "matched-dimension EM" in methods:
                comparators.append("matched-dimension EM")
            for base in comparators:
                row = compare(p, methods[base], split["y"], split["groups"], 1000)
                rows.append(dict(method=method, baseline=base, **row))
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
