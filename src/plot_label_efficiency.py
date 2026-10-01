"""Summarise and plot the label-efficiency grid (A0 / A1 / A2, plus A2 zero-shot).

Reads results/label_efficiency/{A0,A1,A2,A2zs}_*.csv, prints mean ± sd over seeds
(clean protocol) and writes results/label_efficiency/curves.png and summary.csv.

Usage:
    python src/plot_label_efficiency.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = Path("results/label_efficiency")
GRADES = ["EXP", "ICM", "TE"]
TITLES = {"EXP": "Expansion", "ICM": "Inner cell mass", "TE": "Trophectoderm"}
XCEPTION = {"EXP": 0.785, "ICM": 0.723, "TE": 0.670}       # clean protocol, all labels

# Reference categorical palette, slots 1-3 in fixed order (validated: all checks pass;
# aqua < 3:1 contrast -> direct labels + distinct markers as relief).
STYLE = {
    "A0": dict(color="#2a78d6", marker="o", label="A0 frozen"),
    "A1": dict(color="#eb6834", marker="s", label="A1 captions only"),
    "A2": dict(color="#1baf7a", marker="^", label="A2 captions + pseudo-captions"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load() -> pd.DataFrame:
    frames = []
    for cond, pattern in (("A0", "A0_*.csv"), ("A1", "A1_*.csv"), ("A2", "A2_*.csv"), ("A2-zeroshot", "A2zs_*.csv")):
        for f in DIR.glob(pattern):
            d = pd.read_csv(f).assign(condition=cond)
            if cond == "A2-zeroshot":                    # n stored as "50" / "all"
                d["n"] = d["n"].astype(str).replace({"all": "2042"}).astype(int)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df[(df["protocol"] == "clean") & df["target"].isin(GRADES)]


def main():
    df = load()
    summ = (df.groupby(["condition", "target", "n"])
              .agg(acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1_macro", "mean"), seeds=("acc", "size"))
              .reset_index())
    summ.to_csv(DIR / "summary.csv", index=False)
    for metric, name in (("acc", "accuracy"), ("f1", "macro F1")):
        table = summ.pivot_table(index="n", columns=["target", "condition"], values=metric)
        print(f"\n=== clean protocol, {name}, mean over seeds ===")
        print(table.round(3).to_string())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    xs = [50, 100, 200, 500, 2042]
    for ax, g in zip(axes, GRADES):
        ax.set_facecolor(SURFACE)
        ax.axhline(XCEPTION[g], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(52, XCEPTION[g] + 0.006, f"Xception {XCEPTION[g]:.3f}", color=MUTED, fontsize=8.5, va="bottom")
        for cond, st in STYLE.items():
            s = summ[(summ.condition == cond) & (summ.target == g)].sort_values("n")
            ax.fill_between(s["n"], s["acc"] - s["acc_sd"].fillna(0), s["acc"] + s["acc_sd"].fillna(0),
                            color=st["color"], alpha=0.12, lw=0, zorder=2)
            ax.plot(s["n"], s["acc"], color=st["color"], lw=2, marker=st["marker"], ms=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=st["label"])
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["50", "100", "200", "500", "all\n2,042"])
        ax.minorticks_off()
        ax.set_xlim(42, 2600)
        ax.set_title(TITLES[g], color=INK, fontsize=11, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel("labelled training images", color=MUTED)
    axes[0].set_ylabel("test accuracy (clean protocol)", color=MUTED)
    axes[0].set_ylim(0.45, 0.85)

    # direct labels at the right end of the last panel (selective: one per series)
    last = summ[summ.target == "TE"]
    for cond, st in STYLE.items():
        y = last[(last.condition == cond) & (last.n == 2042)]["acc"].iloc[0]
        offset = {"A0": -0.012, "A1": -0.03, "A2": 0.012}[cond]
        axes[2].annotate(cond, (2042, y), xytext=(2150, y + offset), color=INK, fontsize=9,
                         va="center", annotation_clip=False)

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02),
               labelcolor=INK)
    fig.text(0.01, -0.04, "Mean of 3 seeds (shaded: ± 1 sd); 'all' is 1 seed. 300-image gold test set; "
             "test accuracy has ~±5-point 95% CI.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = DIR / "curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out} and {DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
