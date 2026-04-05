#!/usr/bin/env python3
"""
Answer-Type Regime Analysis (BLIP-only)
Binary / Constrained / Open-vocabulary

Usage: python3 src/analysis/answer_regime.py
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "answer_regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 50

def acc_ci(correct, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = correct / n
    margin = z * np.sqrt(p * (1 - p) / n)
    return (p, p - margin, p + margin)

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r'^(a |an |the )', '', s)
    return s.strip('.,!?;:"\'()-')

print("Loading data...")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("", "_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)

def regime(s):
    if s in {"verify", "logical"}: return "binary"
    if s in {"choose", "compare"}: return "constrained"
    return "open"

df["regime"] = df["structural"].apply(regime)

# ══════════════════════════════════════════════════════════════════════════════
# BINARY REGIME
# ══════════════════════════════════════════════════════════════════════════════
binary = df[df["regime"] == "binary"].copy()
binary["blip_says_yes"] = binary["blip_answer"].apply(lambda x: norm(x) == "yes")
binary["gt_is_yes"]     = binary["gt_answer"].apply(lambda x: x.lower() == "yes")

print("\n" + "="*65)
print("BINARY REGIME")
print("="*65)

rows_bin = []
for s in ["verify", "logical"]:
    for sem in sorted(binary[binary["structural"]==s]["semantic"].unique()):
        sub = binary[(binary["structural"]==s) & (binary["semantic"]==sem)]
        if len(sub) < MIN_N: continue
        gt_yes_rate  = sub["gt_is_yes"].mean()
        pred_yes_rate = sub["blip_says_yes"].mean()
        yes_acc = sub[sub["gt_is_yes"]]["correct"].mean()
        no_acc  = sub[~sub["gt_is_yes"]]["correct"].mean()
        p, lo, hi = acc_ci(sub["correct"].sum(), len(sub))
        rows_bin.append({"structural": s, "semantic": sem, "n": len(sub),
                         "overall_acc": p, "gt_yes_rate": gt_yes_rate,
                         "pred_yes_rate": pred_yes_rate,
                         "acc_on_yes": yes_acc, "acc_on_no": no_acc,
                         "bias_gap": yes_acc - no_acc})

df_bin = pd.DataFrame(rows_bin)
df_bin.to_csv(OUT_DIR / "binary.csv", index=False)

print(f"\n{'struct×sem':20} {'n':>7} {'acc':>6} {'GT_yes':>7} {'pred_yes':>9} "
      f"{'acc|GT=yes':>11} {'acc|GT=no':>10} {'bias_gap':>9}")
print("-" * 85)
for _, r in df_bin.iterrows():
    key = f"{r['structural']}×{r['semantic']}"
    print(f"{key:20} {r['n']:>7,} {r['overall_acc']:>6.3f} {r['gt_yes_rate']:>7.3f} "
          f"{r['pred_yes_rate']:>9.3f} {r['acc_on_yes']:>11.3f} {r['acc_on_no']:>10.3f} "
          f"{r['bias_gap']:>+9.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTRAINED REGIME
# ══════════════════════════════════════════════════════════════════════════════
constrained = df[df["regime"] == "constrained"].copy()
constrained["choice_extracted"] = constrained["choice_extracted"].fillna(False)

# Check if BLIP answer matches one of the two extracted candidates
def candidate_match(row):
    if not row["choice_extracted"]: return np.nan
    a, b = norm(str(row["choice_a"])), norm(str(row["choice_b"]))
    pred  = norm(str(row["blip_answer"]))
    return int(pred in {a, b})

constrained["cand_match"] = constrained.apply(candidate_match, axis=1)

print("\n" + "="*65)
print("CONSTRAINED REGIME")
print("="*65)

rows_con = []
for s in ["choose", "compare"]:
    sub = constrained[constrained["structural"] == s]
    sub_ext = sub[sub["choice_extracted"]]
    n_ext   = len(sub_ext)
    match_rate = sub_ext["cand_match"].mean() if n_ext > 0 else np.nan
    # accuracy when BLIP stays in-constraint vs out
    if n_ext > 0:
        acc_in  = sub_ext[sub_ext["cand_match"]==1]["correct"].mean()
        acc_out = sub_ext[sub_ext["cand_match"]==0]["correct"].mean()
        n_in    = (sub_ext["cand_match"]==1).sum()
        n_out   = (sub_ext["cand_match"]==0).sum()
    else:
        acc_in = acc_out = n_in = n_out = np.nan
    p, lo, hi = acc_ci(sub["correct"].sum(), len(sub))
    rows_con.append({"structural": s, "n_total": len(sub),
                     "n_extractable": n_ext, "overall_acc": p,
                     "cand_match_rate": match_rate,
                     "acc_in_constraint": acc_in, "acc_out_constraint": acc_out,
                     "n_in": n_in, "n_out": n_out})

df_con = pd.DataFrame(rows_con)
df_con.to_csv(OUT_DIR / "constrained.csv", index=False)

for _, r in df_con.iterrows():
    print(f"\n  {r['structural']} (n={r['n_total']:,}, extractable={r['n_extractable']:,})")
    print(f"    Overall accuracy         : {r['overall_acc']:.3f}")
    print(f"    Candidate match rate      : {r['cand_match_rate']:.3f}")
    print(f"    Acc when in-constraint    : {r['acc_in_constraint']:.3f}  (n={r['n_in']:,})")
    print(f"    Acc when out-of-constraint: {r['acc_out_constraint']:.3f}  (n={r['n_out']:,})")

# ══════════════════════════════════════════════════════════════════════════════
# OPEN REGIME
# ══════════════════════════════════════════════════════════════════════════════
open_q = df[df["regime"] == "open"].copy()

print("\n" + "="*65)
print("OPEN REGIME (query questions)")
print("="*65)

rows_open = []
for sem in ["rel", "attr", "cat", "global"]:
    sub = open_q[open_q["semantic"] == sem]
    p, lo, hi = acc_ci(sub["correct"].sum(), len(sub))
    rows_open.append({"semantic": sem, "n": len(sub),
                      "accuracy": p, "ci_lo": lo, "ci_hi": hi})

df_open = pd.DataFrame(rows_open)
df_open.to_csv(OUT_DIR / "open.csv", index=False)

print(f"\n{'semantic':8} {'n':>8} {'accuracy':>9} {'95% CI':>16}")
print("-" * 46)
for _, r in df_open.iterrows():
    ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
    print(f"{r['semantic']:8} {r['n']:>8,} {r['accuracy']:>9.3f} {ci:>16}")

# ── Summary figure: overall accuracy by regime + semantic ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Binary
ax = axes[0]
labels = [f"{r['structural']}×{r['semantic']}" for _, r in df_bin.iterrows()]
acc_yes = df_bin["acc_on_yes"].values
acc_no  = df_bin["acc_on_no"].values
x = np.arange(len(labels))
ax.bar(x - 0.2, acc_yes, 0.35, label="GT=yes", color="#4C72B0", alpha=0.85)
ax.bar(x + 0.2, acc_no,  0.35, label="GT=no",  color="#DD8452", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_title("Binary: acc on yes vs no GT"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)

# Constrained: match rate
ax = axes[1]
structs = df_con["structural"].values
match_r = df_con["cand_match_rate"].values
acc_in_ = df_con["acc_in_constraint"].values
acc_out_= df_con["acc_out_constraint"].values
x2 = np.arange(len(structs))
ax.bar(x2 - 0.25, match_r,  0.25, label="In-constraint rate", color="#55A868", alpha=0.85)
ax.bar(x2,         acc_in_,  0.25, label="Acc | in-constraint", color="#4C72B0", alpha=0.85)
ax.bar(x2 + 0.25, acc_out_, 0.25, label="Acc | out-of-constraint", color="#C44E52", alpha=0.85)
ax.set_xticks(x2); ax.set_xticklabels(structs)
ax.set_title("Constrained regime"); ax.legend(fontsize=7)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)

# Open: accuracy by semantic
ax = axes[2]
sems = df_open["semantic"].values
accs = df_open["accuracy"].values
colors = ["#4C72B0","#DD8452","#55A868","#8172B2"]
bars = ax.bar(sems, accs, color=colors[:len(sems)], alpha=0.85)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, a + 0.01,
            f"{a:.2f}", ha="center", fontsize=9)
ax.set_title("Open regime (query)"); ax.set_ylabel("BLIP accuracy")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 0.8)

plt.suptitle("Exp 8: Answer-Type Regime Analysis (BLIP)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSummary plot saved: summary.png")
print(f"All outputs in: {OUT_DIR}")
