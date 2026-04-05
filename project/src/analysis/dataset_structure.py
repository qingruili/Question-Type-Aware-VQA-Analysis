#!/usr/bin/env python3
"""
src/analysis/dataset_structure.py

Phase 0: Data exploration of the GQA dataset (all 10 sections).

Outputs are saved to results/analysis/dataset_structure/ and printed to stdout.

Sections:
  1. Overall dataset structure (cell counts + depth distribution)
  2. Relation label frequency table
  3. Attribute type vocabulary
  4. Answer group coverage (groups.global) for query questions
  5. Yes/No ratio per verify cell
  6. Choice candidate extractability test
  7. Answer frequency distribution (training split)
  8. Entailment annotation coverage
  9. Scene graph basic statistics
  10. Answer ontology — vocabulary by semantic category

Usage:
  python src/analysis/dataset_structure.py
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
VAL_Q_PATH   = DATA_DIR / "questions1.2" / "val_balanced_questions.json"
TRAIN_Q_PATH = DATA_DIR / "questions1.2" / "train_balanced_questions.json"
VAL_SG_PATH  = DATA_DIR / "sceneGraphs" / "val_sceneGraphs.json"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "dataset_structure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURAL_ORDER = ["query", "verify", "logical", "choose", "compare"]
SEMANTIC_ORDER   = ["rel", "attr", "obj", "cat", "global"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    print(f"Loading {path} ...", flush=True)
    with open(path) as f:
        return json.load(f)


def get_depth(q):
    """Number of operations = number of items in semantic list."""
    return len(q.get("semantic", []))


def get_program_str(q):
    """Return the program as a string (semanticStr field)."""
    return q.get("semanticStr", "")


def extract_relate_labels(program_str):
    """
    Extract all relation labels from relate: operations in semanticStr.
    Format: relate: role,relation_label,direction (objId) [dep]
    The relation label is the second comma-separated part of the argument.
    """
    labels = []
    # Match 'relate: arg' where arg = 'role,relation,direction (objId) [dep]'
    for m in re.finditer(r'relate:\s*([^\[>]+?)(?:\s*\[|\s*->|$)', program_str):
        arg = m.group(1).strip()
        # remove trailing object ref like '(1234567)'
        arg = re.sub(r'\s*\(\d+\)\s*$', '', arg).strip()
        parts = [p.strip() for p in arg.split(',')]
        if len(parts) >= 2:
            labels.append(parts[1])  # relation label is the middle part
    return labels


def tee(lines, out_path):
    """Write lines to file and print to stdout."""
    text = "\n".join(lines) + "\n"
    print(text)
    out_path.write_text(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════

val_questions   = load_json(VAL_Q_PATH)
train_questions = load_json(TRAIN_Q_PATH)
val_graphs      = load_json(VAL_SG_PATH)

print(f"\nValidation questions : {len(val_questions):,}")
print(f"Training questions   : {len(train_questions):,}")
print(f"Validation images    : {len(val_graphs):,}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Overall Dataset Structure
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 1: STRUCTURAL × SEMANTIC CELL COUNTS (validation)",
         "=" * 60]

cell_counts = Counter()
for q in val_questions.values():
    s = q["types"]["structural"]
    m = q["types"]["semantic"]
    cell_counts[(s, m)] += 1

header = f"{'':12}" + "".join(f"{m:>10}" for m in SEMANTIC_ORDER) + f"{'TOTAL':>10}"
lines.append(header)
for s in STRUCTURAL_ORDER:
    row_total = sum(cell_counts.get((s, m), 0) for m in SEMANTIC_ORDER)
    row = f"{s:12}" + "".join(
        f"{cell_counts.get((s, m), 0):>10}" for m in SEMANTIC_ORDER
    ) + f"{row_total:>10}"
    lines.append(row)
col_totals = [sum(cell_counts.get((s, m), 0) for s in STRUCTURAL_ORDER) for m in SEMANTIC_ORDER]
grand_total = sum(col_totals)
lines.append(f"{'TOTAL':12}" + "".join(f"{t:>10}" for t in col_totals) + f"{grand_total:>10}")

lines += ["",
          "=" * 60,
          "SECTION 1b: PROGRAM DEPTH DISTRIBUTION PER STRUCTURAL TYPE",
          "=" * 60]

depth_by_struct = defaultdict(Counter)
for q in val_questions.values():
    s = q["types"]["structural"]
    d = get_depth(q)
    depth_by_struct[s][d] += 1

for s in STRUCTURAL_ORDER:
    depths = sorted(depth_by_struct[s].items())
    total_s = sum(n for _, n in depths)
    mean_d  = sum(d * n for d, n in depths) / total_s if total_s else 0
    lines.append(f"\n{s} (n={total_s:,}, mean depth={mean_d:.2f}):")
    for d, n in depths:
        bar = "█" * (n // 500)
        lines.append(f"  depth {d}: {n:6,}  {bar}")

tee(lines, OUT_DIR / "section1_cell_counts.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Relation Label Frequency Table
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 2: ALL RELATE LABELS BY FREQUENCY (validation)",
         "=" * 60]

label_counter = Counter()
for q in val_questions.values():
    prog = get_program_str(q)
    for label in extract_relate_labels(prog):
        label_counter[label] += 1

lines.append(f"Total unique relate labels: {len(label_counter)}")
lines.append(f"{'Count':>8}  Label")
lines.append("-" * 50)
for label, count in label_counter.most_common():
    lines.append(f"{count:>8}  {label}")

tee(lines, OUT_DIR / "section2_relation_labels.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Attribute Type Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 3: ATTRIBUTE TYPES IN PROGRAMS (attr semantic, validation)",
         "=" * 60]

def extract_final_op_type(q):
    """
    For attribute questions, extract the attribute type label from the
    final query/verify/choose/filter/same/different operation.

    Two formats exist in GQA:
      (A) Compound operation name: "verify color", "filter hposition",
          "choose material" — the attribute type IS the second word of the
          operation name; the argument is just the value ("black", "left", …).
      (B) Bare operation name: "verify", "filter", "choose", "query" —
          the argument encodes "type: value" or just "type".
    """
    ops = q.get("semantic", [])
    for op in reversed(ops):
        name = op["operation"]
        arg  = op.get("argument", "")
        parts = name.split(None, 1)   # split on first whitespace at most once
        base  = parts[0]

        if base == "query":
            # Always bare: argument IS the type (e.g. "name", "color")
            t = re.sub(r'\s*\[.*?\]\s*$', '', arg).strip()
            t = t.split(":")[0].strip()
            if t:
                return t

        elif base in {"verify", "filter", "choose"}:
            if len(parts) == 2:
                # Compound: "verify color", "filter hposition", "choose rel", …
                attr_type = parts[1]
                # "verify rel" means it's a relation check, not an attribute
                if attr_type == "rel":
                    continue
                return attr_type
            else:
                # Bare: argument is "type: value" or just "type"
                t = re.sub(r'\s*\[.*?\]\s*$', '', arg).strip()
                t = t.split(":")[0].strip()
                if t:
                    return t

        elif base in {"same", "different"}:
            # e.g. "same color", "different material"
            if len(parts) == 2:
                return parts[1]
            return base   # generic "same" / "different" — rarely hits

        elif name == "common":
            # "What do X and Y have in common?" — no specific attribute type
            return "common"

    return None

attr_type_counter = Counter()
# Also per structural type
attr_type_by_struct = defaultdict(Counter)

for q in val_questions.values():
    if q["types"]["semantic"] != "attr":
        continue
    s = q["types"]["structural"]
    t = extract_final_op_type(q)
    key = t if t else "[unextracted]"
    attr_type_counter[key] += 1
    attr_type_by_struct[s][key] += 1

lines.append(f"Total unique attribute types: {len(attr_type_counter)}")
lines.append(f"{'Count':>8}  Attribute type")
lines.append("-" * 40)
for t, count in attr_type_counter.most_common():
    lines.append(f"{count:>8}  {t}")

lines += ["", "Per structural type (top 10 each):"]
for s in STRUCTURAL_ORDER:
    if not attr_type_by_struct[s]:
        continue
    lines.append(f"\n  {s}:")
    for t, n in attr_type_by_struct[s].most_common(10):
        lines.append(f"    {n:>7,}  {t}")

tee(lines, OUT_DIR / "section3_attribute_types.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Answer Group Coverage
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 4: ANSWER GROUP (groups.global) COVERAGE FOR QUERY QUESTIONS",
         "=" * 60]

query_qs = {qid: q for qid, q in val_questions.items()
            if q["types"]["structural"] == "query"}

total = len(query_qs)
has_group = sum(1 for q in query_qs.values()
                if q.get("groups", {}).get("global"))
lines.append(f"Overall: {has_group:,} / {total:,} ({100*has_group/total:.1f}%)")

lines.append("\nBy semantic type:")
for sem in ["rel", "attr", "cat", "global"]:
    sub = {qid: q for qid, q in query_qs.items()
           if q["types"]["semantic"] == sem}
    if not sub:
        lines.append(f"  query×{sem}: 0 / 0")
        continue
    sub_has = sum(1 for q in sub.values()
                  if q.get("groups", {}).get("global"))
    pct = 100 * sub_has / len(sub)
    lines.append(f"  query×{sem}: {sub_has:,} / {len(sub):,} ({pct:.1f}%)")

all_groups = [q["groups"]["global"] for q in query_qs.values()
              if q.get("groups", {}).get("global")]
lines.append(f"\nUnique groups.global values: {len(set(all_groups)):,}")
lines.append("Top 30 most common groups:")
for g, c in Counter(all_groups).most_common(30):
    lines.append(f"  {c:>6}  {g}")

tee(lines, OUT_DIR / "section4_group_coverage.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Yes/No Ratio Per Verify Cell
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 5: YES/NO RATIO PER VERIFY CELL (validation ground truth)",
         "=" * 60]

for sem in ["rel", "attr", "obj", "global"]:
    qs = [q for q in val_questions.values()
          if q["types"]["structural"] == "verify"
          and q["types"]["semantic"] == sem]
    if not qs:
        lines.append(f"verify×{sem}: no questions")
        continue
    yes_count = sum(1 for q in qs if q["answer"].lower() == "yes")
    no_count  = len(qs) - yes_count
    lines.append(
        f"verify×{sem}  n={len(qs):6,}  "
        f"yes={yes_count:,} ({100*yes_count/len(qs):.1f}%)  "
        f"no={no_count:,} ({100*no_count/len(qs):.1f}%)"
    )

# Also logical (which has yes/no answers)
lines.append("")
for sem in SEMANTIC_ORDER:
    qs = [q for q in val_questions.values()
          if q["types"]["structural"] == "logical"
          and q["types"]["semantic"] == sem]
    if not qs:
        continue
    yes_count = sum(1 for q in qs if q["answer"].lower() == "yes")
    no_count  = len(qs) - yes_count
    lines.append(
        f"logical×{sem}  n={len(qs):6,}  "
        f"yes={yes_count:,} ({100*yes_count/len(qs):.1f}%)  "
        f"no={no_count:,} ({100*no_count/len(qs):.1f}%)"
    )

tee(lines, OUT_DIR / "section5_yesno_ratio.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Choice Candidate Extractability Test
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 6: CHOICE CANDIDATE EXTRACTION TEST",
         "=" * 60]

def extract_choice_candidates(question_str):
    """
    Extract the two candidate answers from a choose/compare question.
    Handles 'X or Y' near the end of the question.
    """
    m = re.search(r',?\s*([\w][\w\s]*?)\s+or\s+([\w][\w\s]*?)\s*\??$',
                  question_str.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None

choose_qs = [q for q in val_questions.values()
             if q["types"]["structural"] in {"choose", "compare"}]

success, fail = 0, 0
fail_examples = []
success_examples = []
for q in choose_qs:
    result = extract_choice_candidates(q["question"])
    if result:
        success += 1
        if len(success_examples) < 5:
            success_examples.append((q["question"], result))
    else:
        fail += 1
        if len(fail_examples) < 15:
            fail_examples.append(q["question"])

total_choose = len(choose_qs)
lines.append(f"choose/compare questions: {total_choose:,}")
lines.append(f"  extraction success: {success:,} ({100*success/total_choose:.1f}%)")
lines.append(f"  extraction failure: {fail:,} ({100*fail/total_choose:.1f}%)")
lines.append("\nExample successes (first 5):")
for q_str, (a, b) in success_examples:
    lines.append(f"  Q: {q_str}")
    lines.append(f"     → candidates: '{a}' | '{b}'")
lines.append("\nExample failures (first 15):")
for ex in fail_examples:
    lines.append(f"  {ex}")

tee(lines, OUT_DIR / "section6_choice_extraction.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Answer Frequency Distribution (Training Split)
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 7: TRAINING ANSWER FREQUENCY DISTRIBUTION",
         "=" * 60]

train_answer_counts = Counter()
for q in train_questions.values():
    train_answer_counts[q["answer"]] += 1

total_train = len(train_questions)
lines.append(f"Training questions   : {total_train:,}")
lines.append(f"Unique answers       : {len(train_answer_counts):,}")
lines.append("\nTop 30 answers by frequency:")
for ans, count in train_answer_counts.most_common(30):
    pct = 100 * count / total_train
    lines.append(f"  {count:>7,}  ({pct:5.2f}%)  {ans}")

# Rank lookup
rank_lookup = {ans: rank + 1 for rank, (ans, _)
               in enumerate(train_answer_counts.most_common())}

val_answers = [q["answer"] for q in val_questions.values()]
val_ranks   = [rank_lookup.get(ans, len(train_answer_counts) + 1)
               for ans in val_answers]
oov_val     = sum(1 for r in val_ranks if r > len(train_answer_counts))

lines.append(f"\nValidation GT answers not in training: "
             f"{oov_val:,} / {len(val_answers):,} "
             f"({100*oov_val/len(val_answers):.1f}%)")

bins = [(1, 10), (11, 100), (101, 500), (501, 1000), (1001, float('inf'))]
lines.append("\nDistribution of val GT answer ranks:")
for lo, hi in bins:
    n   = sum(1 for r in val_ranks if lo <= r <= hi)
    label = f"rank {lo}–{int(hi)}" if hi != float('inf') else f"rank {lo}+"
    lines.append(f"  {label:>20}: {n:>7,}  ({100*n/len(val_ranks):.1f}%)")

tee(lines, OUT_DIR / "section7_answer_frequency.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Entailment Annotation Coverage
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 8: ENTAILMENT ANNOTATION COVERAGE (validation)",
         "=" * 60]

has_entailed = sum(1 for q in val_questions.values()
                   if q.get("entailed") and len(q["entailed"]) > 0)
total        = len(val_questions)
lines.append(f"Questions with entailed field: {has_entailed:,} / {total:,} "
             f"({100*has_entailed/total:.1f}%)")
lines.append("\nPer structural × semantic cell:")

for s in STRUCTURAL_ORDER:
    for sem in SEMANTIC_ORDER:
        qs = [q for q in val_questions.values()
              if q["types"]["structural"] == s
              and q["types"]["semantic"] == sem]
        if not qs:
            continue
        with_ent  = [q for q in qs if q.get("entailed") and len(q["entailed"]) > 0]
        n_with    = len(with_ent)
        avg_ent   = (sum(len(q["entailed"]) for q in with_ent) / n_with
                     if n_with else 0)
        pct       = 100 * n_with / len(qs)
        lines.append(f"  {s}×{sem}: {n_with:,}/{len(qs):,} ({pct:.1f}%) "
                     f"have entailments (avg {avg_ent:.1f} entailed per source)")

# Check broken links
all_val_ids   = set(val_questions.keys())
broken_links  = 0
total_links   = 0
for q in val_questions.values():
    for eid in q.get("entailed", []):
        total_links += 1
        if eid not in all_val_ids:
            broken_links += 1

lines.append(f"\nTotal entailment links : {total_links:,}")
lines.append(f"Broken links (target not in val split): {broken_links:,} "
             f"({100*broken_links/total_links:.1f}% of links)" if total_links else
             "No entailment links found.")

tee(lines, OUT_DIR / "section8_entailment_coverage.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Scene Graph Basic Statistics
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 9: SCENE GRAPH STATISTICS (validation images)",
         "=" * 60]

obj_counts    = []
rel_counts    = []
rel_densities = []

for img_id, graph in val_graphs.items():
    objs  = graph.get("objects", {})
    n_obj = len(objs)
    n_rel = sum(len(obj.get("relations", [])) for obj in objs.values())
    obj_counts.append(n_obj)
    rel_counts.append(n_rel)
    if n_obj > 0:
        rel_densities.append(n_rel / n_obj)

def distribution_summary(name, values):
    out = []
    values = sorted(values)
    n = len(values)
    out.append(f"\n{name} (n={n:,}):")
    for pct in [0, 10, 25, 33, 50, 67, 75, 90, 95, 100]:
        idx = min(int(pct / 100 * n), n - 1)
        out.append(f"  p{pct:3d}: {values[idx]:.2f}")
    out.append(f"  mean: {sum(values)/n:.2f}")
    return out

lines += distribution_summary("Objects per image",   obj_counts)
lines += distribution_summary("Relations per image",  rel_counts)
lines += distribution_summary("Relation density (relations/objects)", rel_densities)

lines.append("\nObject count histogram (capped at 30):")
obj_hist = Counter(min(c, 30) for c in obj_counts)
for k in sorted(obj_hist):
    label = f"{k}" if k < 30 else "30+"
    bar   = "█" * (obj_hist[k] // 30)
    lines.append(f"  {label:>4} objects: {obj_hist[k]:5,}  {bar}")

tee(lines, OUT_DIR / "section9_scene_graph_stats.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Answer Ontology — Vocabulary by Semantic Category
# ═══════════════════════════════════════════════════════════════════════════════

lines = ["=" * 60,
         "SECTION 10: ANSWER VOCABULARY BY SEMANTIC TYPE (validation GT answers)",
         "=" * 60]

answers_by_sem = defaultdict(Counter)
for q in val_questions.values():
    sem = q["types"]["semantic"]
    answers_by_sem[sem][q["answer"]] += 1

for sem in SEMANTIC_ORDER:
    counter  = answers_by_sem[sem]
    unique   = len(counter)
    total_q  = sum(counter.values())
    top15    = counter.most_common(15)
    lines.append(f"\n{sem}: {unique:,} unique answers across {total_q:,} questions")
    lines.append(f"  Top 15: {[f'{ans}({n})' for ans, n in top15]}")

lines += ["", "=" * 60,
          "ANSWER OVERLAP BETWEEN SEMANTIC TYPES",
          "=" * 60]
sem_vocabs = {sem: set(answers_by_sem[sem].keys()) for sem in SEMANTIC_ORDER}
for i, s1 in enumerate(SEMANTIC_ORDER):
    for s2 in SEMANTIC_ORDER[i + 1:]:
        overlap   = sem_vocabs[s1] & sem_vocabs[s2]
        examples  = list(overlap)[:8]
        lines.append(f"  {s1} ∩ {s2}: {len(overlap):,} shared answers  "
                     f"e.g. {examples}")

binary_answers = {"yes", "no"}
lines.append(f"\nQuestions with binary GT answers: "
             f"{sum(1 for q in val_questions.values() if q['answer'] in binary_answers):,}")
lines.append("Structural types of binary-answer questions:")
binary_struct = Counter(q["types"]["structural"] for q in val_questions.values()
                        if q["answer"] in binary_answers)
for s, n in binary_struct.most_common():
    lines.append(f"  {s}: {n:,}")

# Also: answers unique to each semantic type
lines.append("\nAnswers unique to each semantic type (not in any other):")
for sem in SEMANTIC_ORDER:
    others = set().union(*(sem_vocabs[s] for s in SEMANTIC_ORDER if s != sem))
    unique_to = sem_vocabs[sem] - others
    examples  = list(unique_to)[:10]
    lines.append(f"  {sem}: {len(unique_to):,} unique answers  e.g. {examples}")

tee(lines, OUT_DIR / "section10_answer_ontology.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nAll outputs saved to: {OUT_DIR}")
