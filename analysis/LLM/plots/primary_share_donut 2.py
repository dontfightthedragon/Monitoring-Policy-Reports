#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

CSV = "analysis/LLM/artifacts/full/compare_primary_shares.csv"
OUT = "analysis/LLM/figures"; os.makedirs(OUT, exist_ok=True)

def make_donut(ax, labels, shares, title):
    # stable color mapping per label
    palette = {
        "FORESTS_LAND":   "#1f77b4",
        "WATER_MARINE":   "#2ca02c",
        "WILDLIFE_SPECIES":"#ff7f0e",
        "CROSS_CUTTING":  "#9467bd",
        "NONE":           "#7f7f7f",
    }
    colors = [palette.get(l, "#bbbbbb") for l in labels]
    wedges, _ = ax.pie(
        shares, labels=None, startangle=90, counterclock=False,
        colors=colors, wedgeprops=dict(width=0.35, edgecolor="white")
    )
    # add center circle text
    ax.text(0, 0, title, ha="center", va="center", fontsize=10)
    ax.set_aspect("equal")
    # add external legend with percentages
    pct = [f"{l}: {s*100:.1f}%" for l,s in zip(labels, shares)]
    ax.legend(wedges, pct, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

def main(path: str):
    df = pd.read_csv(path)
    labels = df["llm_primary_label"].tolist()
    g4 = df["gpt4"].astype(float).tolist()
    g3 = df["gpt35"].astype(float).tolist()

    # single-model donuts
    for name, data in [("gpt4", g4), ("gpt35", g3)]:
        fig, ax = plt.subplots(figsize=(6, 5))
        make_donut(ax, labels, data, name.upper())
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"primary_shares_donut_{name}.pdf"), bbox_inches="tight")
        plt.close(fig)

    # side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    make_donut(axes[0], labels, g4, "GPT-4o")
    make_donut(axes[1], labels, g3, "GPT-3.5")
    fig.suptitle("Primary label distribution (donut)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "primary_shares_donut_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV, help="Path to compare_primary_shares.csv")
    args = ap.parse_args()
    main(args.csv)

