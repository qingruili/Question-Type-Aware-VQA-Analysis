#!/usr/bin/env python3
"""
Qualitative Example Visualizations (individual images)

Generates examples across four categories, each saved as a separate PNG
in its own folder so you can pick which ones to use in the report.

Folders:
  qualitative_examples/A_verify_vs_query/    — same attr, verify correct vs query wrong
  qualitative_examples/B_choose_mismatch/    — choose question mismatch types
  qualitative_examples/C_query_wrong/        — open query wrong predictions (visual inspection)
  qualitative_examples/D_position_confusion/ — left/right/top/bottom directional errors

Quality filter: examples where the conclusion is clear (NOT annotation artifacts).

Usage: python3 src/examples/qualitative_examples.py
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from viz_examples import save_individual_examples

STATS_CSV  = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV   = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
MISMATCH_CSV = PROJECT_ROOT / "results" / "analysis" / "format_depth_analysis" / "choose_mismatch.csv"
OUT_BASE   = PROJECT_ROOT / "results" / "analysis" / "qualitative_examples"

# ── helpers ───────────────────────────────────────────────────────────────────
CANONICAL = {
    "color":"color","colour":"color",
    "hposition":"position","vposition":"position",
    "material":"material","size":"size","activity":"activity",
    "pose":"pose","shape":"shape","length":"length","height":"height",
    "weather":"weather","sportActivity":"sport",
}
def canon(t):
    if pd.isna(t): return "other"
    return CANONICAL.get(str(t).strip(), "other")

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r'^(a |an |the )', '', s)
    return s.strip('.,!?;:"\'()-')

def is_annotation_artifact(r):
    """Return True if wrong answer looks like a compound_truncation / near_miss."""
    gt   = norm(str(r.get("gt_answer", "")))
    pred = norm(str(r.get("blip_answer", "")))
    if not gt or not pred:
        return True
    # compound truncation: pred is last word of GT (e.g., GT="computer mouse", pred="mouse")
    if gt.endswith(" " + pred) or pred.endswith(" " + gt):
        return True
    # near-miss: one is a substring of the other
    if pred in gt or gt in pred:
        return True
    # spelling variants that are both correct
    NEAR_SYNONYMS = {frozenset(["gray","grey"]), frozenset(["blond","blonde"]),
                     frozenset(["lying","laying"]), frozenset(["sofa","couch"]),
                     frozenset(["bookshelf","bookcase"])}
    if frozenset([gt, pred]) in NEAR_SYNONYMS:
        return True
    return False

def to_row(r, label=""):
    return {
        "image_id":    int(r["image_id"]),
        "question":    r["question"],
        "gt_answer":   r["gt_answer"],
        "blip_answer": r["blip_answer"],
        "correct":     bool(r["correct"]),
        "label":       label,
    }

def pick_diverse(df, n, image_col="image_id", filter_fn=None):
    """Pick up to n rows from df ensuring no repeated image_id."""
    seen, result = set(), []
    for _, r in df.iterrows():
        if filter_fn and filter_fn(r):
            continue
        if r[image_col] not in seen:
            result.append(r)
            seen.add(r[image_col])
            if len(result) >= n:
                break
    return result

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("","_f"))
df["correct"]    = df["blip_correct_norm"].astype(bool)
df["canon_type"] = df["attr_type"].apply(canon)
attr = df[df["semantic"] == "attr"].copy()

# ─────────────────────────────────────────────────────────────────────────────
# A) verify=correct vs query=wrong — same attr type, optionally same image
#    Target: ~100 total images (~50 pairs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("A) verify=correct vs query=wrong")
print("="*60)

out_A = OUT_BASE / "A_verify_vs_query"

# Clear existing examples so the folder only contains fresh output
import shutil
if out_A.exists():
    shutil.rmtree(out_A)

# color only, depth=2 only, same-image pairs only
verify_ok  = attr[(attr["structural"]=="verify") & (attr["canon_type"]=="color") &
                   attr["correct"] & (attr["program_depth"]==2)].copy()
query_fail = attr[(attr["structural"]=="query")  & (attr["canon_type"]=="color") &
                   ~attr["correct"] & (attr["program_depth"]==2)].copy()
query_fail = query_fail[~query_fail.apply(is_annotation_artifact, axis=1)]

# only images that have BOTH a depth-2 verify correct AND a depth-2 query wrong
shared_imgs = set(verify_ok["image_id"]) & set(query_fail["image_id"])

rows_A = []
for img_id in list(shared_imgs):
    v = verify_ok[verify_ok["image_id"]==img_id].iloc[0]
    q = query_fail[query_fail["image_id"]==img_id].iloc[0]
    rows_A.append(to_row(v, label="verify × color"))
    rows_A.append(to_row(q, label="query × color"))

pairs_found = len(rows_A) // 2
print(f"  color, depth=2, same-image pairs: {pairs_found} pairs → {len(rows_A)} images")

save_individual_examples(rows_A, out_dir=out_A, prefix="A_")
print(f"  → {len(rows_A)} individual images in {out_A}")

# ─────────────────────────────────────────────────────────────────────────────
# B) Choose mismatch — focus on:
#      near_miss_wrong : genuine semantic near-misses (GT="light brown", BLIP="brown")
#      free_form       : BLIP ignores candidates (GT="sofa", BLIP="couch")
#    Positional wrong-direction cases are now correctly in exact_wrong_candidate
#    after the fix in format_depth_analysis.py; we no longer include them here.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("B) Choose mismatch examples")
print("="*60)

import shutil as _shutil
out_B = OUT_BASE / "B_choose_mismatch"
if out_B.exists():
    _shutil.rmtree(out_B)

rows_B = []

if MISMATCH_CSV.exists():
    cm = pd.read_csv(MISMATCH_CSV, dtype={"qid": str})
    cm = cm.rename(columns={"imageId": "image_id"})
    if "image_id" not in cm.columns:
        cm = cm.merge(feats[["question_id","image_id"]], left_on="qid",
                      right_on="question_id", how="left")

    def is_clean_choose(r):
        """Return True if candidates look like short answer strings (not full sentence fragments)."""
        ca, cb = str(r.get("choice_a","")), str(r.get("choice_b",""))
        if len(ca) > 40 or len(cb) > 40: return False
        for prefix in ("Is ", "Are ", "Does ", "Do ", "Was ", "Were "):
            if ca.startswith(prefix) or cb.startswith(prefix): return False
        return True

    for mtype, label_str, n_want in [
        ("near_miss_correct", "choose: near-miss correct (BLIP compressed the right answer)", 10),
        ("free_form",         "choose: free-form (ignores candidates)", 10),
    ]:
        sub = cm[cm["mismatch_type"] == mtype].dropna(subset=["image_id"])
        clean_sub = sub[sub.apply(is_clean_choose, axis=1)]
        picked = pick_diverse(clean_sub, n_want)
        for r in picked:
            rows_B.append({
                "image_id":    int(r["image_id"]),
                "question":    r["question"],
                "gt_answer":   r["gt_answer"],
                "blip_answer": r["blip_answer"],
                "correct":     bool(r["correct"]),
                "label":       label_str,
            })
        print(f"  {mtype}: {len(picked)} examples (from {len(sub)} total, {len(clean_sub)} clean)")

    save_individual_examples(rows_B, out_dir=out_B, prefix="B_")
    print(f"  → {len(rows_B)} individual images in {out_B}")
else:
    print("  MISMATCH CSV not found — run format_depth_analysis.py first")

# ─────────────────────────────────────────────────────────────────────────────
# C) Open query wrong predictions — visual inspection by attribute type
#    Filter: non-trivial BLIP answers, no annotation artifacts
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("C) Open query wrong predictions")
print("="*60)

out_C = OUT_BASE / "C_query_wrong"
rows_C = []

for ct, n_want in [
    ("activity", 30),
    ("pose",      5),
    ("color",     6),
    ("position",  5),
    ("size",      4),
    ("material",  4),
]:
    wrong = attr[
        (attr["structural"] == "query") &
        (attr["canon_type"] == ct) &
        (~attr["correct"]) &
        attr["blip_answer"].notna() &
        (attr["blip_answer"].astype(str).str.strip() != "")
    ].copy()
    picked = pick_diverse(wrong, n_want, filter_fn=is_annotation_artifact)
    for r in picked:
        rows_C.append(to_row(r, label=f"query×{ct} ✗"))
    print(f"  {ct}: {len(picked)} examples")

save_individual_examples(rows_C, out_dir=out_C, prefix="C_")
print(f"  → {len(rows_C)} individual images in {out_C}")

# ─────────────────────────────────────────────────────────────────────────────
# D) Position confusion — left/right/top/bottom directional errors
#    Include both query×position and verify×position cases
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("D) Position confusion examples")
print("="*60)

import shutil as _shutil_d
out_D = OUT_BASE / "D_position_confusion"
if out_D.exists():
    _shutil_d.rmtree(out_D)
rows_D = []

POS_WORDS = {"left","right","top","bottom","middle","center","front","back","side"}

def is_pos_confusion(r):
    gt_n   = norm(str(r.get("gt_answer","")))
    pred_n = norm(str(r.get("blip_answer","")))
    return gt_n in POS_WORDS and pred_n in POS_WORDS and gt_n != pred_n

# query×position wrong — directional confusion only
q_pos = attr[
    (attr["structural"] == "query") &
    (attr["canon_type"] == "position") &
    (~attr["correct"])
].copy()
q_pos_conf = q_pos[q_pos.apply(is_pos_confusion, axis=1)]
picked = pick_diverse(q_pos_conf, 13)
for r in picked:
    rows_D.append(to_row(r, label="query×position"))
print(f"  query×position confusion: {len(picked)} examples")

save_individual_examples(rows_D, out_dir=out_D, prefix="D_")
print(f"  → {len(rows_D)} individual images in {out_D}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  A_verify_vs_query:    {len(rows_A)} images → {out_A}")
print(f"  B_choose_mismatch:    {len(rows_B)} images → {out_B}")
print(f"  C_query_wrong:        {len(rows_C)} images → {out_C}")
print(f"  D_position_confusion: {len(rows_D)} images → {out_D}")
print(f"\nBase directory: {OUT_BASE}")
