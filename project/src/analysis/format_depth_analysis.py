#!/usr/bin/env python3
"""
Format & Depth Analysis (BLIP-only)

Covers:
  (1) Depth-controlled heatmaps (depth=2 and depth=3, annotated with n)
  (2) Structural format comparison (verify vs query vs choose at same attribute)
  (3) Why verify is easier: format collapse analysis with yes/no balance check
  (4) Attribute difficulty ranking + language-prior hypothesis
  (5) Choose mismatch: is BLIP's wrong answer semantically close to a candidate?

Usage: python3 src/analysis/format_depth_analysis.py
"""
import re, math
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "format_depth_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 80

CANONICAL = {
    "color":"color","colour":"color",
    "hposition":"position","vposition":"position",
    "material":"material","size":"size","activity":"activity",
    "pose":"pose","shape":"shape","length":"length","height":"height",
    "age":"age","weather":"weather","sportActivity":"sport",
    "pattern":"pattern","cleanliness":"cleanliness","common":"common",
}
def canonical(t):
    if pd.isna(t): return "other"
    return CANONICAL.get(str(t).strip(), "other")

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r'^(a |an |the )', '', s)
    return s.strip('.,!?;"\'()-')

def acc_ci(correct, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = correct / n
    m = z * np.sqrt(p * (1-p) / n)
    return (p, p-m, p+m)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("","_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)
attr = df[df["semantic"] == "attr"].copy()
attr["canon_type"] = attr["attr_type"].apply(canonical)

STRUCTS = ["query","verify","choose","compare"]
TYPES_ORDERED = ["color","position","material","size","activity",
                 "pose","shape","length","weather","common","other"]

# ═══════════════════════════════════════════════════════════════════════════════
# (1) DEPTH-CONTROLLED HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("(1) DEPTH-CONTROLLED HEATMAPS")
print("="*65)

def make_heatmap(attr_sub, title, fname):
    rows = []
    for s in STRUCTS:
        for ct in TYPES_ORDERED:
            bucket = attr_sub[(attr_sub["structural"]==s) & (attr_sub["canon_type"]==ct)]
            n = len(bucket); c = int(bucket["correct"].sum())
            p, _, _ = acc_ci(c, n)
            rows.append({"structural":s,"attr_type":ct,"n":n,"accuracy":p})
    df_h = pd.DataFrame(rows)
    pivot_acc = df_h.pivot(index="structural",columns="attr_type",values="accuracy")
    pivot_n   = df_h.pivot(index="structural",columns="attr_type",values="n")
    cols = [c for c in TYPES_ORDERED if c in pivot_acc.columns]
    pivot_acc = pivot_acc[cols].reindex(STRUCTS)
    pivot_n   = pivot_n[cols].reindex(STRUCTS)

    annot = pivot_acc.copy().astype(object)
    for r in pivot_acc.index:
        for c in pivot_acc.columns:
            n_v = pivot_n.loc[r,c]
            a_v = pivot_acc.loc[r,c]
            if n_v < MIN_N or np.isnan(a_v):
                annot.loc[r,c] = f"—\nn={n_v}"
            else:
                annot.loc[r,c] = f"{a_v:.2f}\nn={n_v}"

    fig, ax = plt.subplots(figsize=(14,4))
    mask = (pivot_n < MIN_N) | pivot_acc.isna()
    sns.heatmap(pivot_acc.where(~mask), annot=annot, fmt="s",
                vmin=0.15, vmax=0.85, cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label":"BLIP accuracy"})
    ax.set_title(title)
    ax.set_xlabel("Attribute type"); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUT_DIR/fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")
    return df_h

print("\n  Building heatmap: ALL depths")
df_all = make_heatmap(attr, "Exp 3: BLIP accuracy by structural × attr type (all depths, n annotated)",
                      "format_heatmap_all.png")
print("\n  Building heatmap: depth = 2 only")
df_d2  = make_heatmap(attr[attr["program_depth"]==2],
                      "Exp 3: depth=2 only", "format_heatmap_depth2.png")
print("\n  Building heatmap: depth = 3 only")
df_d3  = make_heatmap(attr[attr["program_depth"]==3],
                      "Exp 3: depth=3 only", "format_heatmap_depth3.png")

# Print depth=2 vs depth=3 comparison for key types
print("\n  Depth effect within structural types (key attr types):")
print(f"  {'struct':8} {'attr_type':10} {'d=2 acc':>9} {'d=3 acc':>9} {'d=2 n':>7} {'d=3 n':>7} {'diff':>7}")
print("  " + "-"*60)
for s in STRUCTS:
    for ct in ["color","position","material"]:
        r2 = df_d2[(df_d2["structural"]==s)&(df_d2["attr_type"]==ct)]
        r3 = df_d3[(df_d3["structural"]==s)&(df_d3["attr_type"]==ct)]
        if r2.empty or r3.empty: continue
        a2,n2 = r2["accuracy"].values[0], r2["n"].values[0]
        a3,n3 = r3["accuracy"].values[0], r3["n"].values[0]
        if n2 < MIN_N and n3 < MIN_N: continue
        diff = a3 - a2 if not (np.isnan(a2) or np.isnan(a3)) else float("nan")
        a2s = f"{a2:.3f}" if n2 >= MIN_N else "   —  "
        a3s = f"{a3:.3f}" if n3 >= MIN_N else "   —  "
        print(f"  {s:8} {ct:10} {a2s:>9} {a3s:>9} {n2:>7} {n3:>7} {diff:>+7.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# (2+3) STRUCTURAL FORMAT COMPARISON + WHY VERIFY IS EASIER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("(2+3) STRUCTURAL FORMAT COMPARISON — SAME ATTRIBUTE TYPE")
print("="*65)

print("\n  Accuracy at depth=2 (most common, fairest comparison):")
print(f"  {'attr_type':12} {'query':>8} {'verify':>8} {'choose':>8} {'compare':>8}  (verify−query gap)")
print("  " + "-"*65)
for ct in TYPES_ORDERED:
    row = {}
    for s in STRUCTS:
        bucket = attr[(attr["structural"]==s)&(attr["canon_type"]==ct)&(attr["program_depth"]==2)]
        n = len(bucket)
        row[s] = (bucket["correct"].mean() if n >= MIN_N else float("nan"), n)
    vals = {s: row[s][0] for s in STRUCTS}
    ns   = {s: row[s][1] for s in STRUCTS}
    if all(np.isnan(v) for v in vals.values()): continue
    gap = vals["verify"] - vals["query"]
    def fmt(v, n): return f"{v:.3f}" if n >= MIN_N else "  — "
    print(f"  {ct:12} {fmt(vals['query'],ns['query']):>8} {fmt(vals['verify'],ns['verify']):>8} "
          f"{fmt(vals['choose'],ns['choose']):>8} {fmt(vals['compare'],ns['compare']):>8}  "
          f"{f'{gap:+.3f}' if not np.isnan(gap) else '  —  ':>8}")

# Why verify is easier: yes/no balance check + accuracy decomposition
print("\n  Why verify is easier — yes/no balance and bias:")
print(f"  {'struct×attr':20} {'n':>7} {'GT_yes%':>9} {'BLIP_yes%':>10} {'acc|GT=yes':>12} {'acc|GT=no':>11} {'gap':>6}")
print("  " + "-"*80)
binary_structs = ["verify","logical"]
for s in binary_structs:
    for ct in ["color","position","material","size"]:
        bucket = attr[(attr["structural"]==s)&(attr["canon_type"]==ct)]
        n = len(bucket)
        if n < MIN_N: continue
        gt_yes    = (bucket["gt_answer"].str.lower() == "yes").mean()
        blip_yes  = (bucket["blip_answer"].str.lower() == "yes").mean()
        acc_yes   = bucket[bucket["gt_answer"].str.lower()=="yes"]["correct"].mean()
        acc_no    = bucket[bucket["gt_answer"].str.lower()=="no"]["correct"].mean()
        gap       = acc_yes - acc_no
        print(f"  {s+'×'+ct:20} {n:>7,} {gt_yes:>9.3f} {blip_yes:>10.3f} "
              f"{acc_yes:>12.3f} {acc_no:>11.3f} {gap:>+6.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# (4) ATTRIBUTE DIFFICULTY — LANGUAGE PRIOR HYPOTHESIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("(4) ATTRIBUTE DIFFICULTY — ANSWER CONCENTRATION ANALYSIS")
print("="*65)

# For query×attr: how concentrated are the answers? (entropy proxy)
query_attr = attr[attr["structural"]=="query"].copy()
print("\n  query×attr: answer concentration per canonical type")
print(f"  {'type':12} {'n':>7} {'unique_ans':>11} {'top1_pct':>10} {'top5_pct':>10} {'entropy':>9} {'accuracy':>10}")
print("  " + "-"*72)
for ct in TYPES_ORDERED:
    sub = query_attr[query_attr["canon_type"]==ct]
    n = len(sub)
    if n < 50: continue
    ans_counts = sub["gt_answer"].str.lower().value_counts()
    unique = len(ans_counts)
    top1   = ans_counts.iloc[0] / n
    top5   = ans_counts.iloc[:5].sum() / n
    total  = n
    probs  = ans_counts.values / total
    ent    = -sum(p * math.log2(p) for p in probs if p > 0)
    acc    = sub["correct"].mean()
    print(f"  {ct:12} {n:>7,} {unique:>11,} {top1:>10.3f} {top5:>10.3f} {ent:>9.3f} {acc:>10.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# (5) CHOOSE MISMATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("(5) CHOOSE MISMATCH ANALYSIS")
print("="*65)

choose = df[(df["structural"]=="choose")&(df["choice_extracted"]==True)].copy()
choose["norm_pred"]  = choose["blip_answer"].apply(norm)
choose["norm_gt"]    = choose["gt_answer"].apply(norm)
choose["norm_ca"]    = choose["choice_a"].apply(norm)
choose["norm_cb"]    = choose["choice_b"].apply(norm)

# Positional words must not be detected as near_miss via substring:
# "top" appearing in "in the top part" is a directional pick, not a semantic near-miss.
POSITIONAL = {
    "left","right","top","bottom","front","back","middle","center",
    "above","below","upper","lower","inside","outside","side","near","far",
}

def categorize_choose(row):
    pred = row["norm_pred"]
    gt   = row["norm_gt"]
    ca   = row["norm_ca"]
    cb   = row["norm_cb"]
    if pred == gt:
        return "exact_correct"
    if pred == ca or pred == cb:
        return "exact_wrong_candidate"
    # near miss: pred is substring of a candidate or vice versa.
    # Only apply when the candidate is a short clean string (< 25 chars).
    # Long fragments like "Does the white bottle look small" are extraction
    # artifacts — any short word will match as a substring, which is meaningless.
    # Also skip positional words regardless of candidate length.
    for cand in [ca, cb]:
        if pred in cand or cand in pred:
            if pred in POSITIONAL:
                return "exact_wrong_candidate"
            if len(cand) > 25:          # extraction artifact — long fragment
                return "exact_wrong_candidate"
            match_label = "near_miss_correct" if cand == gt else "near_miss_wrong"
            return match_label
    # semantic similarity via SequenceMatcher
    sim_ca = SequenceMatcher(None, pred, ca).ratio()
    sim_cb = SequenceMatcher(None, pred, cb).ratio()
    best_cand = ca if sim_ca >= sim_cb else cb
    best_sim  = max(sim_ca, sim_cb)
    is_correct_cand = (best_cand == gt)
    if best_sim > 0.6:
        return "fuzzy_correct" if is_correct_cand else "fuzzy_wrong"
    return "free_form"

choose["mismatch_type"] = choose.apply(categorize_choose, axis=1)

print("\n  Choose mismatch category distribution:")
mc = choose["mismatch_type"].value_counts()
for cat, n in mc.items():
    acc = choose[choose["mismatch_type"]==cat]["correct"].mean()
    print(f"  {cat:28} {n:>7,} ({100*n/len(choose):5.1f}%)  acc={acc:.3f}")

# Save examples of each mismatch type
print("\n  Examples by mismatch type:")
for mtype in ["near_miss_correct","near_miss_wrong","fuzzy_correct","fuzzy_wrong","free_form"]:
    sub = choose[choose["mismatch_type"]==mtype].head(5)
    if len(sub)==0: continue
    print(f"\n  --- {mtype} ---")
    for _,r in sub.iterrows():
        print(f"    Q: {r['question'][:70]}")
        print(f"       GT={r['gt_answer']!r}  BLIP={r['blip_answer']!r}  "
              f"cands=[{r['choice_a']!r},{r['choice_b']!r}]")

# Save choose examples to file for report
choose_out = choose[["qid","question","gt_answer","blip_answer",
                      "choice_a","choice_b","mismatch_type","correct","imageId"]].copy()
choose_out.to_csv(OUT_DIR/"choose_mismatch.csv", index=False)
print(f"\n  Choose mismatch CSV saved.")
print(f"\nAll outputs in: {OUT_DIR}")
