"""Draw the simulation and Arena comparisons used in the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK = "#2D2D32"
CORAL = "#F28E8E"
SKY = "#8EC5E8"


def main():
    simulation = pd.read_csv(ROOT / "results/prompt_value/summary.csv")
    simulation = simulation.loc[
        (simulation.method == "EM mixture") & simulation.generator.isin(["homogeneous", "probability"])
    ].set_index("delta").loc[[0.0, 0.4, 1.4]]
    arena = pd.read_csv(ROOT / "results/prompt_weight_control/contrasts_pca11.csv")
    arena = arena.loc[arena.control == "train-mean weights"].set_index("stratum").loc[["all", "near-even"]]
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 8.5, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "axes.linewidth": .65})
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 2.35),
                                     gridspec_kw={"width_ratios": [1.02, 1]})
    fig.subplots_adjust(left=.08, bottom=.24, top=.73, right=.98, wspace=.62)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#98a9ad")
        ax.tick_params(length=3, width=.6)
        ax.set_axisbelow(True)
    left.set_title("(a)  Simulation", loc="left", fontsize=11, fontweight="bold", pad=26)
    left.text(0, 1.13, "Gain over the best fixed LLM", transform=left.transAxes, fontsize=8.7)
    x = np.arange(len(simulation))
    error = dict(ecolor=INK, capsize=2, elinewidth=.8, capthick=.8)
    left.bar(x - .17, 100 * simulation.prompt_value_mean, width=.31,
             color=SKY, edgecolor=INK, linewidth=.65, label="Oracle",
             yerr=100 * simulation.prompt_value_sem, error_kw=error)
    left.bar(x + .17, 100 * simulation.gain_over_best_fixed_mean, width=.31,
             color=CORAL, edgecolor=INK, linewidth=.65, label="Fitted EM",
             yerr=100 * simulation.gain_over_best_fixed_sem, error_kw=error)
    left.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=1, frameon=False,
                borderaxespad=.1, handlelength=1.25, labelspacing=.22, fontsize=8.2)
    left.axhline(0, color="#70868d", linewidth=.65)
    left.set_xlim(-.47, 2.49)
    left.set_ylim(-2.35, 19.)
    left.set_yticks([0, 5, 10, 15])
    left.set_ylabel("Extra wins per 100")
    left.set_xticks(x, ["None", "Weak", "Strong"])
    left.tick_params(axis="x", length=0)
    left.set_xlabel("Domain specialisation")
    left.grid(axis="y", color="#e3eaec", linewidth=.5)

    right.set_title("(b)  Arena", loc="left", fontsize=11, fontweight="bold", pad=26)
    right.text(0, 1.13, "Gain over fixed domain weights", transform=right.transAxes, fontsize=8.7)
    y = np.array([1.0, 0.0])
    points = 100 * arena.accuracy_gain.to_numpy()
    ends = 100 * arena[["gain_lo", "gain_hi"]].to_numpy().T
    right.errorbar(points, y, xerr=[points - ends[0], ends[1] - points],
                   fmt="o", color=INK, markerfacecolor=CORAL, markersize=5.8, capsize=3., elinewidth=1.2)
    for value, level in zip(points, y):
        right.annotate(f"+{value:.2f}", (value, level), xytext=(0, 6),
                       textcoords="offset points", ha="center", color=INK, fontsize=8.8)
    labels = [f"{name}\n(n = {n:,})" for name, n in zip(["All pairs", "Near-even"], arena.n)]
    right.set_yticks(y, labels, linespacing=1.06)
    right.tick_params(axis="y", length=0, pad=5)
    right.set_ylim(-.51, 1.60)
    right.set_xlim(0, 5.1)
    right.set_xticks(np.arange(6))
    right.set_xlabel("Accuracy gain (percentage points)")
    right.grid(axis="x", color="#e3eaec", linewidth=.5)
    fig.text(.081, .035, "Whiskers: ±1 Monte Carlo SE", fontsize=7.7)
    fig.text(right.get_position().x0, .035, "Intervals: 95% prompt-group bootstrap", fontsize=7.7)
    figure_path = ROOT / "figures/fig_prompt_value.pdf"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
