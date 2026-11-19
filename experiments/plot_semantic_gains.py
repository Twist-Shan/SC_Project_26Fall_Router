"""Paired accuracy gains by predeclared global-BT matchup difficulty."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def summarize(folder):
    contrasts = pd.read_csv(folder / "contrasts.csv")
    full = contrasts[(contrasts.method == "semantic pool") & (contrasts.reference == "lexical pool")].iloc[0]
    strata = pd.read_csv(folder / "difficulty.csv").set_index("global_probability_gap")
    rows = [dict(stratum="All pairs", n=int(strata.n.sum()), gain=full.accuracy,
                 lo=full.accuracy_lo, hi=full.accuracy_hi)]
    for key, label in [("0 to 0.05", "Near even"), ("over 0.05 to 0.15", "Moderate"),
                       ("over 0.15", "Clear favourite")]:
        row = strata.loc[key]
        rows.append(dict(stratum=label, n=int(row.n), gain=row.accuracy_gain,
                         lo=row.accuracy_lo, hi=row.accuracy_hi))
    return pd.DataFrame(rows)


def draw(table, path):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.45, 1.92))
    fig.subplots_adjust(left=.25, right=.97, bottom=.26, top=.96)
    ax.axvline(0, color="#7b858b", lw=.9, zorder=0)
    for i, row in table.iterrows():
        y = len(table) - 1 - i
        color = "#176659" if i in (0, 1) else "#536d80"
        ax.plot([100 * row.lo, 100 * row.hi], [y, y], color=color, lw=1.6, solid_capstyle="round")
        ax.scatter(100 * row.gain, y, s=35, marker="D" if i == 0 else "o", c=color, zorder=3)
    labels = [f"{r.stratum}  ({r.n:,})" for r in table.itertuples()]
    ax.set_yticks(np.arange(len(table))[::-1], labels)
    ax.set_ylim(-.48, len(table) - .45)
    ax.set_xlim(min(-.55, 100 * table.lo.min() - .12), 100 * table.hi.max() + .16)
    ax.set_xlabel("Accuracy gain over the lexical pool (percentage points)")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#909a9e")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#909a9e")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    table = summarize(ROOT / "results/semantic_refinement")
    draw(table, ROOT / "figures/fig_semantic_gains.pdf")
