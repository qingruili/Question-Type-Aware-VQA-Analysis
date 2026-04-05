#!/usr/bin/env python3
"""
src/analysis/build_question_features.py

Pre-compute per-question features for all 132,062 GQA val questions.
Outputs a single flat CSV: results/analysis/question_features/question_features.csv

Each experiment script should load this CSV and join with model predictions
on question_id, rather than recomputing these fields independently.

Columns produced
----------------
question_id       : str   — GQA question ID (primary key)
structural        : str   — query / verify / logical / choose / compare
semantic          : str   — rel / attr / obj / cat / global
image_id          : str   — GQA image ID (links to scene graph)
question          : str   — natural language question text
answer            : str   — ground-truth answer string
program_depth     : int   — number of operations in q["semantic"] list
n_relate_ops      : int   — total relate operations in program
n_spatial_hops    : int   — relate ops whose label classifies as "spatial"
queried_rel_label : str   — for query×rel: relation label being asked about
                            (last relate op before final query op); else ""
queried_rel_type  : str   — taxonomy class of queried_rel_label; else ""
attr_type         : str   — for semantic=="attr": canonical attribute type
                            (e.g. "color", "hposition", "material"); else ""
n_obj_in_scene    : int   — objects in image's scene graph (-1 if missing)
n_rel_in_scene    : int   — total relation edges in scene graph (-1 if missing)
rel_density_in_scene : float — n_rel / n_obj; -1.0 if missing or n_obj==0
answer_rank_train : int   — frequency rank of GT answer in training split
                            (1 = most common; len(vocab)+1 if unseen in train)
binary_answer     : bool  — True if answer in {"yes", "no"}
groups_global     : str   — q["groups"]["global"] for query questions; else ""
choice_extracted  : bool  — True if choice candidates successfully extracted
choice_a          : str   — first candidate for choose/compare; else ""
choice_b          : str   — second candidate for choose/compare; else ""
has_entailment    : bool  — True if q["entailed"] is non-empty
n_entailed        : int   — len(q["entailed"])

Usage
-----
  python src/analysis/build_question_features.py

Run time: ~2–3 minutes on a laptop (dominated by loading train_questions).
"""

import json
import re
import csv
from collections import Counter
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
VAL_Q_PATH    = DATA_DIR / "questions1.2" / "val_balanced_questions.json"
TRAIN_Q_PATH  = DATA_DIR / "questions1.2" / "train_balanced_questions.json"
VAL_SG_PATH   = DATA_DIR / "sceneGraphs" / "val_sceneGraphs.json"
TAXONOMY_PATH = DATA_DIR / "relation_taxonomy.json"
OUT_CSV       = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    print(f"  Loading {path.name} ...", flush=True)
    with open(path) as f:
        return json.load(f)

print("Loading data files...")
val_questions   = load_json(VAL_Q_PATH)
train_questions = load_json(TRAIN_Q_PATH)
val_graphs      = load_json(VAL_SG_PATH)

print("Loading relation taxonomy...")
with open(TAXONOMY_PATH) as f:
    _taxonomy = json.load(f)

_spatial_set     = set(_taxonomy["spatial"]["labels"])
_action_set      = set(_taxonomy["action"]["labels"])
_comparative_set = set(_taxonomy["comparative"]["labels"])
_other_set       = set(_taxonomy["other"]["labels"])
_kw              = _taxonomy["_keywords"]


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def classify_relation_label(label):
    """Return 'spatial', 'action', 'comparative', or 'other'."""
    if label in _comparative_set: return "comparative"
    if label in _action_set:      return "action"
    if label in _spatial_set:     return "spatial"
    if label in _other_set:       return "other"
    l = label.lower()
    for kw in _kw["comparative"]:
        if kw in l: return "comparative"
    for kw in _kw["action"]:
        if kw in l: return "action"
    for kw in _kw["spatial"]:
        if kw in l: return "spatial"
    return "other"


def extract_relate_label_from_op(op):
    """
    Extract the relation label from a single relate operation dict.
    Argument format: "role,relation_label,direction (objectId)"
    Returns the label string, or None if unparseable.
    """
    arg = op.get("argument", "")
    arg = re.sub(r"\s*\(\d+\)\s*$", "", arg).strip()
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) >= 2:
        return parts[1]
    return None


def count_relate_ops(q):
    """Return (total_relate_ops, n_spatial_hops) for a question."""
    total   = 0
    spatial = 0
    for op in q.get("semantic", []):
        if op["operation"] == "relate":
            total += 1
            label = extract_relate_label_from_op(op)
            if label and classify_relation_label(label) == "spatial":
                spatial += 1
    return total, spatial


def get_queried_relation(q):
    """
    For query×rel questions: the relation label from the last relate op
    that precedes the final query op.  Returns "" if not found.
    """
    ops = q.get("semantic", [])
    last_query_idx = None
    for i in range(len(ops) - 1, -1, -1):
        if ops[i]["operation"] == "query":
            last_query_idx = i
            break
    if last_query_idx is None:
        return ""
    for i in range(last_query_idx - 1, -1, -1):
        if ops[i]["operation"] == "relate":
            label = extract_relate_label_from_op(ops[i])
            return label if label else ""
    return ""


def extract_attr_type(q):
    """
    Extract attribute type from an attr-semantic question.
    Handles compound operation names (e.g. "verify color") and bare forms.
    Returns a type string, or "" if not determinable.
    """
    for op in reversed(q.get("semantic", [])):
        name = op["operation"]
        arg  = op.get("argument", "")
        parts = name.split(None, 1)
        base  = parts[0]

        if base == "query":
            t = re.sub(r"\s*\[.*?\]\s*$", "", arg).strip()
            t = t.split(":")[0].strip()
            if t:
                return t

        elif base in {"verify", "filter", "choose"}:
            if len(parts) == 2:
                if parts[1] == "rel":
                    continue          # "verify rel" is not an attribute op
                return parts[1]
            else:
                t = re.sub(r"\s*\[.*?\]\s*$", "", arg).strip()
                t = t.split(":")[0].strip()
                if t:
                    return t

        elif base in {"same", "different"}:
            return parts[1] if len(parts) == 2 else base

        elif name == "common":
            return "common"

    return ""


def extract_choice_candidates(question_str):
    """
    Extract two candidate answers from a choose/compare question.
    Returns (a, b) strings or ("", "") on failure.
    """
    m = re.search(
        r",?\s*([\w][\w\s]*?)\s+or\s+([\w][\w\s]*?)\s*\??$",
        question_str.strip(),
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", ""


# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute answer rank from training split
# ══════════════════════════════════════════════════════════════════════════════

print("Building training answer frequency rank table...")
train_answer_counts = Counter(q["answer"] for q in train_questions.values())
# rank 1 = most frequent answer
rank_lookup = {
    ans: rank + 1
    for rank, (ans, _) in enumerate(train_answer_counts.most_common())
}
OOV_RANK = len(train_answer_counts) + 1   # rank for answers not in training


# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute scene graph stats per image
# ══════════════════════════════════════════════════════════════════════════════

print("Building scene graph stats per image...")
sg_stats = {}   # image_id -> (n_obj, n_rel, rel_density)
for img_id, graph in val_graphs.items():
    objs  = graph.get("objects", {})
    n_obj = len(objs)
    n_rel = sum(len(obj.get("relations", [])) for obj in objs.values())
    density = round(n_rel / n_obj, 4) if n_obj > 0 else -1.0
    sg_stats[img_id] = (n_obj, n_rel, density)


# ══════════════════════════════════════════════════════════════════════════════
# Build feature rows
# ══════════════════════════════════════════════════════════════════════════════

COLUMNS = [
    "question_id",
    "structural",
    "semantic",
    "image_id",
    "question",
    "answer",
    "program_depth",
    "n_relate_ops",
    "n_spatial_hops",
    "queried_rel_label",
    "queried_rel_type",
    "attr_type",
    "n_obj_in_scene",
    "n_rel_in_scene",
    "rel_density_in_scene",
    "answer_rank_train",
    "binary_answer",
    "groups_global",
    "choice_extracted",
    "choice_a",
    "choice_b",
    "has_entailment",
    "n_entailed",
]

print(f"Building features for {len(val_questions):,} questions...")

rows = []
n_processed = 0

for qid, q in val_questions.items():
    structural = q["types"]["structural"]
    semantic   = q["types"]["semantic"]
    image_id   = q.get("imageId", "")
    question   = q.get("question", "")
    answer     = q.get("answer", "")

    # Program features
    program_depth             = len(q.get("semantic", []))
    n_relate_ops, n_spatial   = count_relate_ops(q)

    # Queried relation (query×rel only, but compute for all — empty string otherwise)
    queried_rel_label = ""
    queried_rel_type  = ""
    if semantic == "rel":
        queried_rel_label = get_queried_relation(q)
        if queried_rel_label:
            queried_rel_type = classify_relation_label(queried_rel_label)

    # Attribute type (attr semantic only)
    attr_type = extract_attr_type(q) if semantic == "attr" else ""

    # Scene graph
    sg = sg_stats.get(image_id)
    if sg:
        n_obj_in_scene, n_rel_in_scene, rel_density = sg
    else:
        n_obj_in_scene, n_rel_in_scene, rel_density = -1, -1, -1.0

    # Answer rank in training
    answer_rank_train = rank_lookup.get(answer, OOV_RANK)

    # Binary answer flag
    binary_answer = answer.lower() in {"yes", "no"}

    # Answer group (query questions only)
    groups_global = q.get("groups", {}).get("global", "") or ""

    # Choice candidates (choose / compare only)
    choice_extracted = False
    choice_a         = ""
    choice_b         = ""
    if structural in {"choose", "compare"}:
        a, b = extract_choice_candidates(question)
        if a and b:
            choice_extracted = True
            choice_a         = a
            choice_b         = b

    # Entailment flags
    entailed      = q.get("entailed") or []
    has_entailment = len(entailed) > 0
    n_entailed     = len(entailed)

    rows.append({
        "question_id":           qid,
        "structural":            structural,
        "semantic":              semantic,
        "image_id":              image_id,
        "question":              question,
        "answer":                answer,
        "program_depth":         program_depth,
        "n_relate_ops":          n_relate_ops,
        "n_spatial_hops":        n_spatial,
        "queried_rel_label":     queried_rel_label,
        "queried_rel_type":      queried_rel_type,
        "attr_type":             attr_type,
        "n_obj_in_scene":        n_obj_in_scene,
        "n_rel_in_scene":        n_rel_in_scene,
        "rel_density_in_scene":  rel_density,
        "answer_rank_train":     answer_rank_train,
        "binary_answer":         binary_answer,
        "groups_global":         groups_global,
        "choice_extracted":      choice_extracted,
        "choice_a":              choice_a,
        "choice_b":              choice_b,
        "has_entailment":        has_entailment,
        "n_entailed":            n_entailed,
    })

    n_processed += 1
    if n_processed % 20000 == 0:
        print(f"  ... {n_processed:,} / {len(val_questions):,}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Write CSV
# ══════════════════════════════════════════════════════════════════════════════

print(f"Writing {len(rows):,} rows to {OUT_CSV} ...")
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. File size: {OUT_CSV.stat().st_size / 1e6:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# Sanity checks — print to stdout for review
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

total = len(rows)
print(f"\nTotal rows: {total:,}")

# Check structural / semantic distribution
from collections import defaultdict
struct_counts = Counter(r["structural"] for r in rows)
sem_counts    = Counter(r["semantic"]    for r in rows)
print("\nStructural distribution:")
for k, v in struct_counts.most_common():
    print(f"  {k:10}: {v:>7,}  ({100*v/total:.1f}%)")
print("\nSemantic distribution:")
for k, v in sem_counts.most_common():
    print(f"  {k:10}: {v:>7,}  ({100*v/total:.1f}%)")

# Check attr_type coverage
attr_rows = [r for r in rows if r["semantic"] == "attr"]
attr_empty = sum(1 for r in attr_rows if not r["attr_type"])
print(f"\nattr_type: {len(attr_rows):,} attr questions, "
      f"{attr_empty} with empty attr_type ({100*attr_empty/len(attr_rows):.1f}%)")

# Check queried_rel_label coverage for query×rel
qrel_rows  = [r for r in rows if r["structural"] == "query" and r["semantic"] == "rel"]
qrel_empty = sum(1 for r in qrel_rows if not r["queried_rel_label"])
print(f"\nqueried_rel_label: {len(qrel_rows):,} query×rel questions, "
      f"{qrel_empty} with empty label ({100*qrel_empty/len(qrel_rows):.1f}%)")
# Show distribution of queried_rel_type
qrel_type_counts = Counter(r["queried_rel_type"] for r in qrel_rows if r["queried_rel_type"])
print("  queried_rel_type distribution:")
for t, n in qrel_type_counts.most_common():
    print(f"    {t:15}: {n:>6,}  ({100*n/len(qrel_rows):.1f}%)")

# Check choice extraction rate
choice_rows = [r for r in rows if r["structural"] in {"choose", "compare"}]
choice_ok   = sum(1 for r in choice_rows if r["choice_extracted"])
print(f"\nchoice_extracted: {choice_ok:,} / {len(choice_rows):,} "
      f"({100*choice_ok/len(choice_rows):.1f}%) choose/compare questions")

# Check scene graph coverage
sg_missing = sum(1 for r in rows if r["n_obj_in_scene"] == -1)
print(f"\nScene graph missing: {sg_missing:,} / {total:,} questions "
      f"({100*sg_missing/total:.1f}%)")

# Check binary answer distribution
binary_count = sum(1 for r in rows if r["binary_answer"])
print(f"\nbinary_answer=True: {binary_count:,} / {total:,} ({100*binary_count/total:.1f}%)")

# Check groups_global coverage for query questions
query_rows     = [r for r in rows if r["structural"] == "query"]
query_has_grp  = sum(1 for r in query_rows if r["groups_global"])
print(f"\ngroups_global coverage (query only): {query_has_grp:,} / {len(query_rows):,} "
      f"({100*query_has_grp/len(query_rows):.1f}%)")
# Break down by semantic type
for sem in ["rel", "attr", "cat", "global"]:
    sub      = [r for r in query_rows if r["semantic"] == sem]
    sub_has  = sum(1 for r in sub if r["groups_global"])
    if sub:
        print(f"  query×{sem}: {sub_has:,}/{len(sub):,} ({100*sub_has/len(sub):.1f}%)")

# Check answer rank distribution
rank_bins = [(1, 10), (11, 100), (101, 500), (501, 1000), (1001, float("inf"))]
print("\nanswer_rank_train distribution:")
for lo, hi in rank_bins:
    n     = sum(1 for r in rows if lo <= r["answer_rank_train"] <= hi)
    label = f"rank {lo}–{int(hi)}" if hi != float("inf") else f"rank {lo}+"
    print(f"  {label:>20}: {n:>7,}  ({100*n/total:.1f}%)")

# Spot-check: 5 sample rows
print("\nSample rows (first 5):")
for r in rows[:5]:
    print(f"  {r['question_id']:>10}  {r['structural']:8} {r['semantic']:7} "
          f"depth={r['program_depth']} n_rel={r['n_relate_ops']} "
          f"spatial={r['n_spatial_hops']} "
          f"attr={r['attr_type'] or '-':15} "
          f"qrel={r['queried_rel_label'][:20] or '-'}")

print(f"\nCSV saved to: {OUT_CSV}")
