#!/usr/bin/env python3
"""
Relation Type & Spatial Hop Analysis (BLIP-only)
Sub-part A: query×rel accuracy by (queried_rel_type × depth_bin)
Sub-part B: accuracy by (semantic_type × n_spatial_hops_bin)

Usage: python3 src/analysis/relation_type.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "relation_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 50
COLORS = {"spatial":"#4C72B0","action":"#DD8452","comparative":"#55A868","other":"#C44E52"}
SEM_COLORS = {"rel":"#4C72B0","attr":"#DD8452","cat":"#55A868"}

def acc_ci(correct, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = correct / n
    m = z * np.sqrt(p*(1-p)/n)
    return (p, p-m, p+m)

print("Loading data...")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("","_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)

# ── Sub-part A ────────────────────────────────────────────────────────────────
qrel = df[(df["structural"]=="query") & (df["semantic"]=="rel")].copy()
qrel["depth_bin"] = pd.cut(qrel["program_depth"], bins=[0,2,3,4,99],
                            labels=["≤2","3","4","≥5"])

REL_TYPES  = ["spatial","action","comparative","other"]
DEPTH_BINS = ["≤2","3","4","≥5"]

rows_a = []
for rt in REL_TYPES:
    sub = qrel[qrel["queried_rel_type"]==rt]
    for db in DEPTH_BINS:
        bucket = sub[sub["depth_bin"]==db]
        n = len(bucket); c = bucket["correct"].sum()
        p,lo,hi = acc_ci(c,n)
        rows_a.append({"rel_type":rt,"depth_bin":db,"n":n,"accuracy":p,"ci_lo":lo,"ci_hi":hi})

df_a = pd.DataFrame(rows_a)
df_a.to_csv(OUT_DIR/"rel_type_depth.csv", index=False)

print("\n=== SUB-PART A: query×rel accuracy by relation type × depth ===")
print(f"{'rel_type':12} {'depth':5} {'n':>7} {'accuracy':>9}")
print("-"*40)
for _,r in df_a.iterrows():
    if r["n"] < MIN_N: continue
    print(f"{r['rel_type']:12} {r['depth_bin']:5} {r['n']:>7,} {r['accuracy']:>9.3f}")

print("\n  Marginal by relation type:")
for rt in REL_TYPES:
    sub = qrel[qrel["queried_rel_type"]==rt]
    n = len(sub); c = sub["correct"].sum()
    p,lo,hi = acc_ci(c,n)
    print(f"  {rt:12} n={n:6,}  acc={p:.3f}  CI=[{lo:.3f},{hi:.3f}]")

# Plot A
fig, ax = plt.subplots(figsize=(8,5))
for rt in ["spatial","action"]:
    sub = df_a[(df_a["rel_type"]==rt)&(df_a["n"]>=MIN_N)]
    ax.plot(sub["depth_bin"].astype(str), sub["accuracy"],
            marker="o", label=rt.capitalize(), color=COLORS[rt])
    ax.fill_between(sub["depth_bin"].astype(str),
                    sub["ci_lo"], sub["ci_hi"], alpha=0.15, color=COLORS[rt])
ax.set_xlabel("Program depth bin"); ax.set_ylabel("BLIP accuracy")
ax.set_title("Exp 1A: query×rel accuracy by relation type and depth")
ax.legend(); ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0,0.8); plt.tight_layout()
plt.savefig(OUT_DIR/"rel_type_plot.png", dpi=150); plt.close()
print("\n  Plot saved: rel_type_plot.png")

# ── Sub-part B ────────────────────────────────────────────────────────────────
def hop_bin(n):
    if n==0: return "0"
    if n==1: return "1"
    return "2+"

df["hop_bin"] = df["n_spatial_hops"].apply(hop_bin)
HOP_LABELS = ["0","1","2+"]
SEM_TYPES  = ["rel","attr","cat"]

rows_b = []
for sem in SEM_TYPES:
    sub_sem = df[df["semantic"]==sem]
    for hb in HOP_LABELS:
        bucket = sub_sem[sub_sem["hop_bin"]==hb]
        n = len(bucket); c = bucket["correct"].sum()
        p,lo,hi = acc_ci(c,n)
        rows_b.append({"semantic":sem,"hop_bin":hb,"n":n,"accuracy":p,"ci_lo":lo,"ci_hi":hi})

df_b = pd.DataFrame(rows_b)
df_b.to_csv(OUT_DIR/"spatial_hops.csv", index=False)

print("\n=== SUB-PART B: accuracy by semantic type × spatial hop count ===")
print(f"{'semantic':8} {'hops':5} {'n':>8} {'accuracy':>9}")
print("-"*38)
for _,r in df_b.iterrows():
    if r["n"]<MIN_N: continue
    print(f"{r['semantic']:8} {r['hop_bin']:5} {r['n']:>8,} {r['accuracy']:>9.3f}")

# drop in accuracy from 0→1→2+ hops
print("\n  Accuracy drop from 0 to 1 spatial hop:")
for sem in SEM_TYPES:
    sub = df_b[df_b["semantic"]==sem]
    a0 = sub[sub["hop_bin"]=="0"]["accuracy"].values[0]
    a1 = sub[sub["hop_bin"]=="1"]["accuracy"].values[0]
    a2 = sub[sub["hop_bin"]=="2+"]["accuracy"].values[0]
    print(f"  {sem}: 0hops={a0:.3f}  1hop={a1:.3f}  2+hops={a2:.3f}  "
          f"drop(0→1)={a1-a0:+.3f}  drop(0→2+)={a2-a0:+.3f}")

# Plot B
x = np.arange(len(HOP_LABELS)); w = 0.25
fig, ax = plt.subplots(figsize=(7,5))
for i,sem in enumerate(SEM_TYPES):
    sub = df_b[df_b["semantic"]==sem]
    accs = [sub[sub["hop_bin"]==hb]["accuracy"].values[0] for hb in HOP_LABELS]
    ax.bar(x+i*w, accs, w, label=sem, color=SEM_COLORS[sem], alpha=0.85)
ax.set_xticks(x+w); ax.set_xticklabels(["0 hops","1 hop","2+ hops"])
ax.set_ylabel("BLIP accuracy"); ax.set_title("Exp 1B: accuracy vs spatial hop count")
ax.legend(title="Semantic type"); ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0,0.8); plt.tight_layout()
plt.savefig(OUT_DIR/"spatial_hops_plot.png", dpi=150); plt.close()
print("  Plot saved: spatial_hops_plot.png")
print(f"\nAll outputs in: {OUT_DIR}")
