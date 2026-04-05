#!/usr/bin/env python3
"""
Answer Frequency Analysis (BLIP-only)
Accuracy by (semantic_type × answer_frequency_bin) for query questions

Usage: python3 src/analysis/answer_frequency.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "answer_frequency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 50

FREQ_BINS   = ["1-10", "11-100", "101-500", "501+"]
FREQ_COLORS = {"rel": "#4C72B0", "attr": "#DD8452", "cat": "#55A868", "global": "#8172B2"}

def freq_bin(rank):
    if rank <= 10:  return "1-10"
    if rank <= 100: return "11-100"
    if rank <= 500: return "101-500"
    return "501+"

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
df["freq_bin"] = df["answer_rank_train"].apply(freq_bin)

query = df[df["structural"] == "query"].copy()
SEM_TYPES = ["rel", "attr", "cat", "global"]

rows = []
for sem in SEM_TYPES:
    sub_sem = query[query["semantic"] == sem]
    for fb in FREQ_BINS:
        bucket = sub_sem[sub_sem["freq_bin"] == fb]
        n = len(bucket); c = bucket["correct"].sum()
        p, lo, hi = acc_ci(c, n)
        rows.append({"semantic": sem, "freq_bin": fb,
                     "n": n, "accuracy": p, "ci_lo": lo, "ci_hi": hi})

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "freq_accuracy.csv", index=False)

print("\n=== EXP 7: BLIP accuracy by semantic type × answer frequency bin (query only) ===")
print(f"{'semantic':8} {'freq_bin':10} {'n':>8} {'accuracy':>9}")
print("-" * 42)
for _, r in df_out.iterrows():
    if r["n"] < MIN_N: continue
    print(f"{r['semantic']:8} {r['freq_bin']:10} {r['n']:>8,} {r['accuracy']:>9.3f}")

# Frequency slope: linear regression of accuracy ~ log(rank) per semantic type
print("\n  Frequency slope (linear regression accuracy ~ log10(rank)):")
for sem in SEM_TYPES:
    sub = query[query["semantic"] == sem]
    valid = sub[sub["answer_rank_train"].notna() & sub["correct"].notna()]
    if len(valid) < 100: continue
    x = np.log10(valid["answer_rank_train"].clip(lower=1))
    y = valid["correct"].astype(float)
    slope, intercept, r, p_val, _ = scipy_stats.linregress(x, y)
    print(f"  {sem:8}: slope={slope:+.4f}  r²={r**2:.3f}  p={p_val:.2e}")

# Plot: line per semantic type, x=freq_bin, y=accuracy
fig, ax = plt.subplots(figsize=(8, 5))
for sem in SEM_TYPES:
    sub = df_out[df_out["semantic"] == sem]
    accs = [sub[sub["freq_bin"] == fb]["accuracy"].values[0]
            if sub[sub["freq_bin"] == fb]["n"].values[0] >= MIN_N else np.nan
            for fb in FREQ_BINS]
    ns   = [sub[sub["freq_bin"] == fb]["n"].values[0] for fb in FREQ_BINS]
    ax.plot(FREQ_BINS, accs, marker="o", label=sem,
            color=FREQ_COLORS.get(sem, "gray"))
    for i, (a, n) in enumerate(zip(accs, ns)):
        if not np.isnan(a) and n >= MIN_N:
            ax.annotate(f"n={n:,}", (FREQ_BINS[i], a),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7)

ax.set_xlabel("Answer frequency rank bin (training data)")
ax.set_ylabel("BLIP accuracy (normalized EM)")
ax.set_title("Exp 7: Accuracy vs answer frequency (query questions only)")
ax.legend(title="Semantic type")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 0.85)
plt.tight_layout()
plt.savefig(OUT_DIR / "freq_accuracy.png", dpi=150)
plt.close()
print(f"\nPlot saved: freq_accuracy.png")
print(f"All outputs in: {OUT_DIR}")
