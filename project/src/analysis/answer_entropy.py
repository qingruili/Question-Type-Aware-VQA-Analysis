#!/usr/bin/env python3
"""
Answer Entropy Decomposition (BLIP-only)
Accuracy by (semantic_type × answer_entropy_bin) for query questions.
Entropy computed from training answer distribution per groups_global.

Usage: python3 src/analysis/answer_entropy.py
"""
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
STATS_CSV     = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV      = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
TRAIN_Q_PATH  = PROJECT_ROOT / "data" / "questions1.2" / "train_balanced_questions.json"
OUT_DIR       = PROJECT_ROOT / "results" / "analysis" / "answer_entropy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 50

def acc_ci(correct, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = correct / n
    margin = z * np.sqrt(p * (1 - p) / n)
    return (p, p - margin, p + margin)

def entropy(answers):
    c = Counter(answers)
    total = sum(c.values())
    return -sum((v/total) * math.log2(v/total) for v in c.values() if v > 0)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading val stats + features...")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("", "_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)

print("Loading training questions (for entropy)...")
with open(TRAIN_Q_PATH) as f:
    train_q = json.load(f)

# Build group → list of answers from training query questions
group_answers = defaultdict(list)
for q in train_q.values():
    if q["types"]["structural"] != "query": continue
    g = q.get("groups", {}).get("global", "")
    if g:
        group_answers[g].append(q["answer"])

# Compute entropy per group (min 10 training examples)
group_entropy = {g: entropy(ans) for g, ans in group_answers.items() if len(ans) >= 10}
print(f"  Groups with entropy: {len(group_entropy):,}")

# ── Assign entropy to val query questions ─────────────────────────────────────
query = df[df["structural"] == "query"].copy()
query["group_entropy"] = query["groups_global"].map(group_entropy)

covered = query["group_entropy"].notna().sum()
total   = len(query)
print(f"  Val query questions with entropy assigned: {covered:,}/{total:,} ({100*covered/total:.1f}%)")

# Drop uncovered (mostly query×attr where groups_global is missing)
q_ent = query[query["group_entropy"].notna()].copy()

# Entropy tercile bins (based on full distribution)
p33, p67 = q_ent["group_entropy"].quantile([0.333, 0.667])
print(f"  Entropy percentiles: p33={p33:.3f}  p67={p67:.3f}")

def ent_bin(e):
    if e <= p33: return "low"
    if e <= p67: return "medium"
    return "high"

q_ent["ent_bin"] = q_ent["group_entropy"].apply(ent_bin)

SEM_TYPES = ["rel", "attr", "cat", "global"]
ENT_BINS  = ["low", "medium", "high"]

rows = []
for sem in SEM_TYPES:
    sub_sem = q_ent[q_ent["semantic"] == sem]
    for eb in ENT_BINS:
        bucket = sub_sem[sub_sem["ent_bin"] == eb]
        n = len(bucket); c = bucket["correct"].sum()
        p, lo, hi = acc_ci(c, n)
        rows.append({"semantic": sem, "ent_bin": eb,
                     "n": n, "accuracy": p, "ci_lo": lo, "ci_hi": hi})

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "entropy_accuracy.csv", index=False)

print("\n=== BLIP accuracy by semantic type × entropy bin (query only) ===")
print(f"  Entropy bins:  low ≤ {p33:.2f}  |  medium {p33:.2f}–{p67:.2f}  |  high > {p67:.2f}")
print(f"{'semantic':8} {'ent_bin':8} {'n':>7} {'accuracy':>9}")
print("-" * 38)
for _, r in df_out.iterrows():
    if r["n"] < MIN_N: continue
    print(f"{r['semantic']:8} {r['ent_bin']:8} {r['n']:>7,} {r['accuracy']:>9.3f}")

# Pivot heatmap
pivot_acc = df_out.pivot(index="semantic", columns="ent_bin", values="accuracy")[ENT_BINS]
pivot_n   = df_out.pivot(index="semantic", columns="ent_bin", values="n")[ENT_BINS]
pivot_acc = pivot_acc.reindex(SEM_TYPES)
pivot_n   = pivot_n.reindex(SEM_TYPES)

annot = pivot_acc.copy().astype(object)
for r in pivot_acc.index:
    for c in pivot_acc.columns:
        n_val = pivot_n.loc[r, c]
        annot.loc[r, c] = f"{pivot_acc.loc[r,c]:.2f}\nn={n_val:,}" if n_val >= MIN_N else "—"

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pivot_acc.where(pivot_n >= MIN_N), annot=annot, fmt="s",
            vmin=0.3, vmax=0.75, cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "BLIP accuracy"})
ax.set_title("BLIP Accuracy by Answer Entropy (Query Questions)")
ax.set_xlabel("Entropy bin"); ax.set_ylabel("Semantic type")
plt.tight_layout()
plt.savefig(OUT_DIR / "entropy_heatmap.png", dpi=150)
plt.close()
print(f"\nHeatmap saved: entropy_heatmap.png")

# Within-entropy comparison: is rel still harder than attr at same entropy?
print("\n  Within-entropy comparison (rel vs attr at same entropy bin):")
for eb in ENT_BINS:
    rel_row  = df_out[(df_out["semantic"]=="rel") & (df_out["ent_bin"]==eb)]
    attr_row = df_out[(df_out["semantic"]=="attr") & (df_out["ent_bin"]==eb)]
    if rel_row.empty or attr_row.empty: continue
    rel_acc  = rel_row["accuracy"].values[0]
    attr_acc = attr_row["accuracy"].values[0]
    gap = rel_acc - attr_acc
    print(f"  {eb:6}: rel={rel_acc:.3f}  attr={attr_acc:.3f}  gap={gap:+.3f}")

print(f"\nAll outputs in: {OUT_DIR}")
