#!/usr/bin/env python3
"""
Attribute Type Breakdown (BLIP-only)
Accuracy by (structural_type × canonical_attr_type)

Usage: python3 src/analysis/attribute_type.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "attribute_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 100

CANONICAL = {
    "color": "color", "colour": "color",
    "hposition": "position", "vposition": "position",
    "material": "material",
    "size": "size",
    "activity": "activity",
    "pose": "pose",
    "shape": "shape",
    "length": "length",
    "height": "height",
    "age": "age",
    "weather": "weather",
    "sportActivity": "sport",
    "pattern": "pattern",
    "cleanliness": "cleanliness",
    "common": "common",
}

def canonical(t):
    if pd.isna(t): return "other"
    return CANONICAL.get(str(t).strip(), "other")

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

attr = df[df["semantic"] == "attr"].copy()
attr["canon_type"] = attr["attr_type"].apply(canonical)

STRUCTS = ["query", "verify", "choose", "compare"]

rows = []
for s in STRUCTS:
    sub_s = attr[attr["structural"] == s]
    for ct in sorted(attr["canon_type"].unique()):
        bucket = sub_s[sub_s["canon_type"] == ct]
        n = len(bucket); c = bucket["correct"].sum()
        p, lo, hi = acc_ci(c, n)
        rows.append({"structural": s, "attr_type": ct,
                     "n": n, "accuracy": p, "ci_lo": lo, "ci_hi": hi})

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "attr_type_accuracy.csv", index=False)

print("\n=== EXP 3: BLIP accuracy by structural × canonical attribute type ===")
print(f"(only cells with n ≥ {MIN_N} shown)\n")
for s in STRUCTS:
    sub = df_out[df_out["structural"] == s].sort_values("accuracy", ascending=False)
    header_printed = False
    for _, r in sub.iterrows():
        if r["n"] < MIN_N: continue
        if not header_printed:
            print(f"  {s}:")
            header_printed = True
        ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"    {r['attr_type']:14} n={r['n']:6,}  acc={r['accuracy']:.3f}  {ci}")
    if header_printed: print()

# Heatmap: structural × attr_type
TYPES_ORDERED = ["color", "position", "material", "size", "activity",
                 "pose", "shape", "length", "height", "weather", "common", "other"]
pivot_acc = df_out.pivot(index="structural", columns="attr_type", values="accuracy")
pivot_n   = df_out.pivot(index="structural", columns="attr_type", values="n")

# Keep only cols present in data
cols = [c for c in TYPES_ORDERED if c in pivot_acc.columns]
pivot_acc = pivot_acc[cols].reindex(STRUCTS)
pivot_n   = pivot_n[cols].reindex(STRUCTS)

# Mask low-n cells
mask = pivot_n < MIN_N

annot = pivot_acc.copy().astype(object)
for r in pivot_acc.index:
    for c in pivot_acc.columns:
        n_val = pivot_n.loc[r, c]
        if n_val < MIN_N:
            annot.loc[r, c] = "—"
        else:
            annot.loc[r, c] = f"{pivot_acc.loc[r, c]:.2f}"

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(pivot_acc.where(~mask), annot=annot, fmt="s",
            vmin=0.3, vmax=0.8, cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "BLIP accuracy"})
ax.set_title("Exp 3: BLIP accuracy by structural type × attribute type")
ax.set_xlabel("Attribute type"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT_DIR / "attr_type_heatmap.png", dpi=150)
plt.close()
print(f"Heatmap saved: attr_type_heatmap.png")
print(f"All outputs in: {OUT_DIR}")
