"""Tables for the simulation studies (reads results/sim_*.csv). Figures: experiments/make_figures.py."""
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def table_s1():
    df = pd.read_csv(os.path.join(RES, "sim_S1_consistency.csv"))
    tab = df.groupby(["method", "n"])[["errR", "excess_ll", "regret", "best_pick"]].mean().round(4)
    tab.to_csv(os.path.join(RES, "table_S1.csv"))
    return tab


def table_s2():
    df = pd.read_csv(os.path.join(RES, "sim_S2_speed.csv"))
    tab = df.groupby(["method", "delta"])[["iters", "time", "errR", "rate"]].median().round(3)
    best = df.groupby(["delta", "rep"])["obj"].transform("max")
    df["obj_gap"] = best - df.obj
    tab["obj_gap_median"] = df.groupby(["method", "delta"]).obj_gap.median().round(4)
    tab.to_csv(os.path.join(RES, "table_S2.csv"))
    return tab


def table_s3():
    df = pd.read_csv(os.path.join(RES, "sim_S3_init.csv"))
    tab = df.assign(at_best=df.gap < 0.5).groupby(["setting", "init"]).agg(
        share_at_best=("at_best", "mean"), median_gap=("gap", "median"), median_errR=("errR", "median"),
        median_iters=("iters", "median")).round(3)
    tab.to_csv(os.path.join(RES, "table_S3.csv"))
    return tab


def table_s4():
    df = pd.read_csv(os.path.join(RES, "sim_S4_selectK.csv"))
    pb = df.loc[df.groupby(["n", "rep"])["bic"].idxmin()]
    pv = df.loc[df.groupby(["n", "rep"])["val_ll"].idxmax()]
    t = pd.concat({"BIC": pd.crosstab(pb.n, pb.K), "validation": pd.crosstab(pv.n, pv.K)}, axis=1).fillna(0).astype(int)
    t.to_csv(os.path.join(RES, "table_S4.csv"))
    return t


def table_s5():
    df = pd.read_csv(os.path.join(RES, "sim_S5_AvsB.csv"))
    t = df.groupby(["dgp", "fit"]).excess_ll.agg(["mean", "std"]).round(5)
    t.to_csv(os.path.join(RES, "table_S5.csv"))
    return t


def table_s6():
    return pd.read_csv(os.path.join(RES, "sim_S6_coverage.csv"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for f in [table_s1, table_s2, table_s3, table_s4, table_s5, table_s6]:
        try:
            print(f"--- {f.__name__}\n", f())
        except FileNotFoundError as e:
            print(f"--- {f.__name__}: missing {e.filename}")
