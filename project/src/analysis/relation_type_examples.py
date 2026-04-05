#!/usr/bin/env python3
"""
src/analysis/relation_type_examples.py

Generate human-readable examples for query×rel questions (spatial & action),
showing: question text, GT, BLIP answer, correctness, error type,
and the full functional program annotated with where relate operations sit.

Outputs saved to: results/analysis/relation_type/error_examples.txt

Usage:
  python3 src/analysis/relation_type_examples.py
"""

import json, re, random
from pathlib import Path
from collections import Counter
import pandas as pd

random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
VAL_Q_PATH   = PROJECT_ROOT / "data" / "questions1.2" / "val_balanced_questions.json"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "relation_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EXAMPLES = 12   # per category

# ── Normalization (inline, no imports from seaborn-dependent files) ───────────
_NUM_MAP = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
            'six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}

def norm(s):
    s = str(s).strip().lower()
    if s in _NUM_MAP: s = _NUM_MAP[s]
    s = re.sub(r'^(a |an |the )', '', s)
    s = s.strip('.,!?;:"\'()-')
    return s

# ── Error type classifier ─────────────────────────────────────────────────────
GENERIC_PERSON = {
    'man','woman','boy','girl','person','people','lady','guy',
    'child','kid','children','men','women','girls','boys','no one','nobody'
}

def classify_error(gt_raw, pred_raw):
    gt, pred = norm(gt_raw), norm(pred_raw)
    gt_w, pred_w = gt.split(), pred.split()
    if len(gt_w) > 1 and pred_w == [gt_w[0]]:   return 'compound_truncation'
    if len(gt_w) > 1 and pred_w == [gt_w[-1]]:  return 'compound_truncation'
    if len(pred_w) > 1 and gt_w == [pred_w[0]]: return 'compound_truncation'
    if pred in GENERIC_PERSON and gt not in GENERIC_PERSON:
        return 'person_genericity'
    if gt in GENERIC_PERSON and pred not in GENERIC_PERSON:
        return 'person_specificity'
    return 'wrong_object'

# ── Annotate program: highlight relate ops and show chain structure ────────────
def annotate_program(semantic_list, semanticStr):
    """
    Return a multi-line string showing each operation in the program,
    marking relate ops and their position index.
    """
    lines = []
    n_ops = len(semantic_list)
    for i, op in enumerate(semantic_list):
        name = op['operation']
        arg  = op.get('argument', '')
        deps = op.get('dependencies', [])
        tag  = ''
        if name == 'relate':
            parts = re.sub(r'\s*\(\d+\)\s*$', '', arg).strip().split(',')
            rel_label = parts[1].strip() if len(parts) >= 2 else arg
            tag = f'  ◀ RELATE op #{i}: "{rel_label}"'
        elif name == 'query':
            tag = f'  ◀ QUERY (final answer)'
        elif name == 'select':
            tag = f'  ◀ SELECT (start)'
        dep_str = f'[deps:{deps}]' if deps else ''
        lines.append(f'    [{i}] {name}: {arg} {dep_str}{tag}')
    return '\n'.join(lines)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data…")
stats = pd.read_csv(STATS_CSV, dtype={'qid': str})
feats = pd.read_csv(FEAT_CSV,  dtype={'question_id': str})
df = stats.merge(feats, left_on='qid', right_on='question_id',
                 how='left', suffixes=('', '_feat'))

print("Loading question JSON for program strings…")
with open(VAL_Q_PATH) as f:
    val_q = json.load(f)

# Restrict to query×rel
qrel = df[(df['structural'] == 'query') & (df['semantic'] == 'rel')].copy()
qrel['correct'] = qrel['blip_correct_norm'].astype(bool)
qrel['error_type'] = qrel.apply(
    lambda r: 'correct' if r['correct']
              else classify_error(r['gt_answer'], r['blip_answer']),
    axis=1
)

print(f"\nquery×rel total: {len(qrel):,}")
print("Error type distribution:")
print(qrel['error_type'].value_counts().to_string())

# ── Example builder ───────────────────────────────────────────────────────────
def make_example_block(row, idx):
    qid  = str(row['qid'])
    q_data = val_q.get(qid, {})
    sem_list   = q_data.get('semantic', [])
    sem_str    = q_data.get('semanticStr', '')

    lines = []
    lines.append(f"  Example {idx}")
    lines.append(f"  Question  : {row['question']}")
    lines.append(f"  GT answer : {row['gt_answer']}")
    lines.append(f"  BLIP answer: {row['blip_answer']}")
    lines.append(f"  Result    : {'✓ CORRECT' if row['correct'] else '✗ WRONG'}"
                 + (f"  [{row['error_type']}]" if not row['correct'] else ''))
    lines.append(f"  Rel type  : {row['queried_rel_type']}   depth={row['program_depth']}  n_relate_ops={row['n_relate_ops']}")
    lines.append(f"  Program   : {sem_str}")
    lines.append(f"  Annotated :")
    lines.append(annotate_program(sem_list, sem_str))
    return '\n'.join(lines)

# ── Generate examples ─────────────────────────────────────────────────────────

output_lines = []
output_lines.append("=" * 78)
output_lines.append("EXPERIMENT 1 — QUERY×REL EXAMPLES: SPATIAL vs ACTION")
output_lines.append("Program annotation: ◀ RELATE marks each relation hop; ◀ QUERY = final answer")
output_lines.append("=" * 78)

SECTIONS = [
    # (title, rel_type_filter, error_type_filter)
    ("SPATIAL — CORRECT predictions",    "spatial",     "correct"),
    ("SPATIAL — WRONG: wrong object",    "spatial",     "wrong_object"),
    ("SPATIAL — WRONG: compound trunc.", "spatial",     "compound_truncation"),
    ("ACTION  — CORRECT predictions",    "action",      "correct"),
    ("ACTION  — WRONG: wrong object",    "action",      "wrong_object"),
    ("ACTION  — WRONG: person genericity\n"
     "  (BLIP says 'man/woman/person' instead of role-specific GT)",
                                         "action",      "person_genericity"),
    ("ACTION  — WRONG: person specificity\n"
     "  (GT is generic 'man/woman', BLIP gives more specific role)",
                                         "action",      "person_specificity"),
    ("ACTION  — WRONG: compound trunc.", "action",      "compound_truncation"),
]

for title, rel_type, etype in SECTIONS:
    pool = qrel[(qrel['queried_rel_type'] == rel_type) &
                (qrel['error_type']       == etype)]
    output_lines.append("")
    output_lines.append("─" * 78)
    output_lines.append(f"  {title}  (pool n={len(pool):,})")
    output_lines.append("─" * 78)

    if len(pool) == 0:
        output_lines.append("  (no examples)")
        continue

    sample = pool.sample(min(N_EXAMPLES, len(pool)), random_state=42)
    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        output_lines.append("")
        output_lines.append(make_example_block(row, idx))

text = '\n'.join(output_lines) + '\n'
out_path = OUT_DIR / "error_examples.txt"
out_path.write_text(text)
print(text)
print(f"\nSaved to {out_path}")
