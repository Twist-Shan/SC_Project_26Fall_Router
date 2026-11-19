"""Cost of adding the lexical branch to an existing mixture on Arena fold zero.

The cost includes tokenisation and sparse design construction. The fit uses the
ridge selected by lexical_refinement.py on fold zero's inner validation set.
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import grouped_folds, load_data
from mixbt.lexical import LexicalBT


def run(output, C, repeats=3, batch_size=256):
    data = load_data()
    folds = grouped_folds(data["groups"], seed=1729)
    train, test = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    prompts, a, b, y, M = (data[k] for k in ("text", "a", "b", "y", "M"))
    model = LexicalBT(C=C)
    start = perf_counter()
    features = model.encode(prompts[train])
    encoded = perf_counter()
    design = model.pair_design(features, a[train], b[train], M)
    assembled = perf_counter()
    model.fit_design(design, y[train], M)
    fitted = perf_counter()
    rows = []
    def record(stage, seconds, n, replicate):
        rows.append(dict(fold=0, C=C, operation=stage, batch_size=batch_size, n=n,
                         repetition=replicate, seconds=seconds, ms_per_prompt=1000 * seconds / n))
    record("fit: encode text", encoded-start, len(train), 1)
    record("fit: assemble pairs", assembled-encoded, len(train), 1)
    record("fit: solve ridge", fitted-assembled, len(train), 1)
    record("fit: total", fitted-start, len(train), 1)
    del design, features
    # Both operations process all held-out prompts in the same input order.
    candidates = np.arange(M)
    warm = test[:32]
    model.predict_proba(prompts[warm], a[warm], b[warm])
    model.borda(prompts[warm], candidates)
    for rep in range(repeats):
        tasks = ("predict logged pair", "score all candidates")
        for task in tasks if rep % 2 == 0 else reversed(tasks):
            start = perf_counter()
            for first in range(0, len(test), batch_size):
                idx = test[first:first + batch_size]
                if task == tasks[0]:
                    model.predict_proba(prompts[idx], a[idx], b[idx])
                else:
                    model.borda(prompts[idx], candidates)
            record(task, perf_counter() - start, len(test), rep + 1)
    results = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "branch_timing.csv", index=False)
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/lexical_refinement")
    parser.add_argument("--C", type=float, help="Override the stored inner-validation selection")
    args = parser.parse_args()
    if args.C is None:
        trials = pd.read_csv(args.output / "tuning.csv")
        fold = trials.loc[trials.fold.eq(0)].sort_values(["inner_logloss", "C"])
        C = float(fold.iloc[0].C)
    else:
        C = args.C
    with threadpool_limits(limits=1):
        run(args.output, C)
