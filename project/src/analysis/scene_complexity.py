#!/usr/bin/env python3
"""
Scene Visual Complexity (BLIP-only)
Accuracy by (semantic_type × scene_complexity_bin)

Usage: python3 src/analysis/scene_complexity.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "scene_complexity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 200
# Thresholds from Section 9 exploration
OBJ_BINS   = [0, 12, 19, 999]  # low ≤12, med 13-19, high ≥20
DENS_BINS  = [0, 1.64, 3.00, 999]  # low ≤1.64, med 1.65-3.00, high >3.00

def acc_ci(correct, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = correct / n
    margin = z * np.sqrt(p * (1 - p) / n)
    return (p, p - margin, p + margin)

print("Loading data...")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("", "_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)

df["obj_bin"]  = pd.cut(df["n_obj_in_scene"],  bins=OBJ_BINS,
                         labels=["low", "medium", "high"])
df["dens_bin"] = pd.cut(df["rel_density_in_scene"], bins=DENS_BINS,
                         labels=["low", "medium", "high"])

SEM_TYPES  = ["rel", "attr", "obj", "cat", "global"]
COMP_BINS  = ["low", "medium", "high"]
SEM_COLORS = {"rel":"#4C72B0","attr":"#DD8452","obj":"#55A868",
              "cat":"#C44E52","global":"#8172B2"}

rows = []
for metric, col in [("object_count", "obj_bin"), ("rel_density", "dens_bin")]:
    for sem in SEM_TYPES:
        sub_sem = df[df["semantic"] == sem]
        for cb in COMP_BINS:
            bucket = sub_sem[sub_sem[col] == cb]
            n = len(bucket); c = bucket["correct"].sum()
            p, lo, hi = acc_ci(c, n)
            rows.append({"metric": metric, "semantic": sem, "complexity": cb,
                         "n": n, "accuracy": p, "ci_lo": lo, "ci_hi": hi})

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "complexity_accuracy.csv", index=False)

print("\n=== EXP 5: BLIP accuracy by semantic type × scene complexity ===")
for metric in ["object_count", "rel_density"]:
    print(f"\n  Metric: {metric}")
    print(f"  {'semantic':8} {'complexity':10} {'n':>8} {'accuracy':>9}")
    print("  " + "-" * 42)
    sub = df_out[df_out["metric"] == metric]
    for _, r in sub.iterrows():
        if r["n"] < MIN_N: continue
        print(f"  {r['semantic']:8} {r['complexity']:10} {r['n']:>8,} {r['accuracy']:>9.3f}")

# Plots: one per metric
for metric, col, xlabel in [
    ("object_count", "obj_bin",
     "Object count (low≤12, medium 13–19, high≥20)"),
    ("rel_density",  "dens_bin",
     "Relation density (low≤1.64, medium 1.65–3.0, high>3.0)")
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df_out[df_out["metric"] == metric]
    for sem in ["rel", "attr", "cat"]:
        s2 = sub[sub["semantic"] == sem]
        accs = [s2[s2["complexity"]==cb]["accuracy"].values[0]
                if s2[s2["complexity"]==cb]["n"].values[0] >= MIN_N else np.nan
                for cb in COMP_BINS]
        ax.plot(COMP_BINS, accs, marker="o", label=sem, color=SEM_COLORS[sem])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BLIP accuracy")
    ax.set_title(f"Exp 5: Accuracy vs scene complexity ({metric})")
    ax.legend(title="Semantic type")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 0.8)
    plt.tight_layout()
    fname = f"{metric}.png"
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {fname}")

print(f"\nAll outputs in: {OUT_DIR}")
