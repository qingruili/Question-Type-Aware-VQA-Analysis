#!/usr/bin/env python3
"""
Error Typology (BLIP-only)
Classify wrong predictions into: wrong_type / wrong_value / compound_truncation /
person_issue / near_miss / binary_flip

Usage: python3 src/analysis/error_typology.py
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "error_typology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Answer ontology (from Section 10 exploration) ─────────────────────────────
# Per-semantic answer vocabularies (simplified sets for type classification)
COLOR_ANSWERS    = {"white","black","blue","red","green","brown","gray","yellow",
                    "orange","purple","pink","beige","tan","grey","silver","gold",
                    "dark","light","multicolored","colorful","striped","spotted"}
POSITION_ANSWERS = {"left","right","top","bottom","middle","center","front","back",
                    "side","corner","edge"}
MATERIAL_ANSWERS = {"wood","wooden","metal","plastic","glass","leather","stone",
                    "concrete","paper","fabric","cloth","rubber","brick","steel"}
BINARY_ANSWERS   = {"yes","no"}
PERSON_ANSWERS   = {"man","woman","boy","girl","person","people","lady","guy",
                    "child","kid","children","men","women","girls","boys"}

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r'^(a |an |the )', '', s)
    return s.strip('.,!?;:"\'()-')

def answer_type(s):
    s = norm(s)
    if s in BINARY_ANSWERS:   return "binary"
    if s in COLOR_ANSWERS:    return "color"
    if s in POSITION_ANSWERS: return "position"
    if s in MATERIAL_ANSWERS: return "material"
    if s in PERSON_ANSWERS:   return "person"
    return "other"

def classify_error(gt_raw, pred_raw, semantic):
    gt, pred = norm(gt_raw), norm(pred_raw)
    if gt == pred: return "correct"  # shouldn't happen

    # Binary flip
    if gt in BINARY_ANSWERS and pred in BINARY_ANSWERS:
        return "binary_flip"

    # Compound truncation: one is prefix/suffix of other
    gt_w, pred_w = gt.split(), pred.split()
    if (len(gt_w) > 1 and (pred_w == gt_w[:len(pred_w)] or pred_w == gt_w[-len(pred_w):])) or \
       (len(pred_w) > 1 and (gt_w == pred_w[:len(gt_w)] or gt_w == pred_w[-len(gt_w):])):
        return "compound_truncation"

    gt_t, pred_t = answer_type(gt), answer_type(pred)
    # Person ambiguity: one side is a person label
    if gt_t == "person" or pred_t == "person":
        return "person_ambiguity"

    # Wrong type: predicted a different semantic category of answer
    if gt_t != pred_t and gt_t != "other" and pred_t != "other":
        return "wrong_type"

    # Near miss: edit distance ≤ 2 (very similar strings)
    if abs(len(gt) - len(pred)) <= 2:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, gt, pred).ratio()
        if ratio > 0.75: return "near_miss"

    return "wrong_value"

print("Loading data...")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("", "_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)
df["norm_gt"]   = df["gt_answer"].apply(norm)
df["norm_pred"] = df["blip_answer"].apply(norm)

wrong = df[~df["correct"]].copy()
wrong["error_type"] = wrong.apply(
    lambda r: classify_error(r["gt_answer"], r["blip_answer"], r["semantic"]), axis=1)

print(f"\nTotal wrong predictions: {len(wrong):,}")
print(f"Total correct predictions: {df['correct'].sum():,}")

rows = []
STRUCTS = ["query","verify","logical","choose","compare"]
SEMS    = ["rel","attr","obj","cat","global"]

print("\n=== EXP 10: ERROR TYPE DISTRIBUTION BY STRUCTURAL × SEMANTIC ===")
all_error_types = sorted(wrong["error_type"].unique())

# Overall distribution
overall = wrong["error_type"].value_counts()
print("\nOverall error distribution:")
for et, n in overall.items():
    print(f"  {et:22} {n:>7,}  ({100*n/len(wrong):.1f}%)")

# By semantic type
print("\nBy semantic type:")
for sem in SEMS:
    sub = wrong[wrong["semantic"] == sem]
    if len(sub) == 0: continue
    print(f"\n  {sem} (n={len(sub):,} wrong):")
    for et, n in sub["error_type"].value_counts().items():
        print(f"    {et:22} {n:>7,}  ({100*n/len(sub):.1f}%)")

# By structural type
print("\nBy structural type:")
for s in STRUCTS:
    sub = wrong[wrong["structural"] == s]
    if len(sub) == 0: continue
    print(f"\n  {s} (n={len(sub):,} wrong):")
    for et, n in sub["error_type"].value_counts().items():
        print(f"    {et:22} {n:>7,}  ({100*n/len(sub):.1f}%)")

# Save per-question error types
wrong_out = wrong[["qid","structural","semantic","gt_answer","blip_answer","error_type"]].copy()
wrong_out.to_csv(OUT_DIR / "wrong_predictions.csv", index=False)

# Pivot: structural × semantic → dominant error type
print("\nDominant error type per structural × semantic cell:")
for s in STRUCTS:
    for sem in SEMS:
        sub = wrong[(wrong["structural"]==s) & (wrong["semantic"]==sem)]
        if len(sub) < 50: continue
        dom = sub["error_type"].value_counts().index[0]
        dom_pct = sub["error_type"].value_counts().iloc[0] / len(sub)
        print(f"  {s}×{sem}: {dom} ({dom_pct:.0%})")

# ── Plot: stacked bar per semantic type ───────────────────────────────────────
ERROR_TYPES_ORDERED = ["compound_truncation","near_miss","person_ambiguity",
                       "wrong_type","wrong_value","binary_flip"]
COLORS = {"compound_truncation":"#AED6F1","near_miss":"#A9DFBF",
          "person_ambiguity":"#FAD7A0","wrong_type":"#F1948A",
          "wrong_value":"#C0392B","binary_flip":"#8E44AD"}

sem_data = {}
for sem in SEMS:
    sub = wrong[wrong["semantic"] == sem]
    total = len(sub)
    if total == 0: continue
    sem_data[sem] = {et: sub[sub["error_type"]==et].shape[0]/total
                     for et in ERROR_TYPES_ORDERED}

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(sem_data))
bottoms = np.zeros(len(sem_data))
for et in ERROR_TYPES_ORDERED:
    vals = [sem_data[sem].get(et, 0) for sem in sem_data]
    ax.bar(x, vals, bottom=bottoms, label=et,
           color=COLORS.get(et, "gray"), alpha=0.85)
    bottoms += np.array(vals)

ax.set_xticks(x); ax.set_xticklabels(list(sem_data.keys()))
ax.set_ylabel("Fraction of wrong predictions")
ax.set_title("Exp 10: Error type composition by semantic type")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(OUT_DIR / "error_types.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: error_types.png")
print(f"All outputs in: {OUT_DIR}")
