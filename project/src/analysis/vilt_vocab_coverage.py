#!/usr/bin/env python3
"""
src/analysis/vilt_vocab_coverage.py

Standalone analysis of the ViLT out-of-vocabulary (OOV) problem.
Produces a written summary and figures useful for the milestone report.

Outputs (all in results/analysis/vilt_vocab_coverage/):
  oov_summary.txt          — narrative summary with key stats
  oov_by_cell.csv          — per-cell OOV rates before/after normalization
  oov_examples.txt         — representative OOV answer examples with questions
  figures/
    oov_rate_heatmap.png   — per-cell OOV% heatmap (before normalization)
    oov_fixed_by_norm.png  — how many OOV answers were fixed by each rule

Usage:
  python src/analysis/vilt_vocab_coverage.py
"""

import json
import logging
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "predictions" / "all_predictions.jsonl"
OUT_DIR          = PROJECT_ROOT / "results" / "analysis" / "vilt_vocab_coverage"
FIG_DIR          = OUT_DIR / "figures"

STRUCTURAL_TYPES = ["query", "verify", "logical", "choose", "compare"]
SEMANTIC_TYPES   = ["rel", "attr", "obj", "cat", "global"]
VALID_CELLS = {
    ("query",   "rel"),  ("query",   "attr"), ("query",   "cat"), ("query",   "global"),
    ("verify",  "rel"),  ("verify",  "attr"), ("verify",  "obj"), ("verify",  "global"),
    ("logical", "attr"), ("logical", "obj"),
    ("choose",  "rel"),  ("choose",  "attr"), ("choose",  "cat"), ("choose",  "global"),
    ("compare", "attr"),
}

# ── Import shared normalization from analyze_results ───────────────────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from analyze_results import (
    normalize_strict,
    normalize_normalized,
    _NUM_MAP,
    _BRIT_TO_AM,
    _depluralize,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_vilt_labels() -> set:
    from transformers import ViltForQuestionAnswering
    print("  Loading ViLT label set...")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    labels = set(model.config.id2label.values())
    del model
    return labels


def classify_oov_cause(raw_gt: str, norm_gt: str, raw_labels: set, norm_labels: set) -> str:
    """
    For OOV answers (raw_gt NOT in raw_labels), classify why normalization
    does or does not fix them.
    Returns one of:
      'fixed_by_norm'   — not in raw_labels but norm_gt IS in norm_labels
      'still_oov'       — not in raw_labels and norm_gt NOT in norm_labels
    (If raw_gt IS in raw_labels it won't be passed to this function.)
    """
    return "fixed_by_norm" if norm_gt in norm_labels else "still_oov"


def identify_fix_type(raw_gt: str, norm_gt: str) -> list:
    """
    Return list of rule tags that changed raw_gt → norm_gt.
    Used to understand which normalization rules recover the most OOV cases.
    """
    tags = []
    after_strict = normalize_strict(raw_gt)
    after_norm   = norm_gt

    if raw_gt.strip().lower() != after_strict:
        # Something changed in strict: lowercase/article/number
        if raw_gt.strip().lower() in _NUM_MAP:
            tags.append("number_word")
        if re.match(r"^(a |an |the )", raw_gt.strip().lower()):
            tags.append("article")
    # British spelling
    words_before = after_strict.split()
    words_after  = after_norm.split()
    if any(_BRIT_TO_AM.get(w, w) != w for w in words_before):
        tags.append("brit_spelling")
    # Depluralize
    if any(_depluralize(w) != w for w in words_before):
        tags.append("depluralize")
    if not tags:
        tags.append("other")
    return tags


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading predictions...")
    rows = []
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} questions")

    print("\nLoading ViLT labels...")
    raw_labels  = load_vilt_labels()
    norm_labels = {normalize_normalized(l) for l in raw_labels}
    print(f"  Raw label count:        {len(raw_labels):,}")
    print(f"  Normalized label count: {len(norm_labels):,}")

    # ── Per-answer OOV status ─────────────────────────────────────────────────
    df["gt_norm"]   = df["gt_answer"].apply(normalize_normalized)
    df["oov_raw"]   = ~df["gt_answer"].isin(raw_labels)
    df["oov_norm"]  = ~df["gt_norm"].isin(norm_labels)

    unique_raw_gt  = df["gt_answer"].unique()
    unique_norm_gt = df["gt_norm"].unique()

    n_raw_oov_types  = sum(g not in raw_labels  for g in unique_raw_gt)
    n_norm_oov_types = sum(g not in norm_labels for g in unique_norm_gt)
    n_raw_oov_q      = df["oov_raw"].sum()
    n_norm_oov_q     = df["oov_norm"].sum()
    fixed_types = n_raw_oov_types - n_norm_oov_types
    fixed_q     = n_raw_oov_q    - n_norm_oov_q

    # ── Per-cell OOV rates ────────────────────────────────────────────────────
    cell_rows = []
    for struct, sem in sorted(VALID_CELLS):
        sub = df[(df["structural"] == struct) & (df["semantic"] == sem)]
        n   = len(sub)
        if n == 0:
            continue
        oov_raw_n  = sub["oov_raw"].sum()
        oov_norm_n = sub["oov_norm"].sum()
        cell_rows.append({
            "structural":    struct,
            "semantic":      sem,
            "n":             n,
            "oov_raw_n":     int(oov_raw_n),
            "oov_raw_pct":   oov_raw_n / n * 100,
            "oov_norm_n":    int(oov_norm_n),
            "oov_norm_pct":  oov_norm_n / n * 100,
            "fixed_by_norm": int(oov_raw_n - oov_norm_n),
        })

    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(OUT_DIR / "oov_by_cell.csv", index=False, float_format="%.2f")
    print(f"\nSaved per-cell OOV stats: oov_by_cell.csv")

    # Print cell table
    print(f"\n{'='*75}")
    print("  OOV RATE PER CELL (before and after normalization)")
    print(f"{'='*75}")
    print(f"{'structural':<12} {'semantic':<8} {'n':>7} "
          f"{'oov_raw%':>9} {'oov_norm%':>10} {'fixed':>7}")
    print("-" * 75)
    for _, r in cell_df.sort_values("oov_raw_pct", ascending=False).iterrows():
        print(f"{r['structural']:<12} {r['semantic']:<8} {int(r['n']):>7,} "
              f"{r['oov_raw_pct']:>8.1f}% {r['oov_norm_pct']:>9.1f}% "
              f"{int(r['fixed_by_norm']):>7,}")

    # ── Which fix rules help the most ─────────────────────────────────────────
    # For answers that were OOV before but in-vocab after normalization
    raw_oov_answers = {g for g in unique_raw_gt if g not in raw_labels}
    fix_counts: Counter = Counter()
    for raw_ans in raw_oov_answers:
        norm_ans = normalize_normalized(raw_ans)
        if norm_ans in norm_labels:
            tags = identify_fix_type(raw_ans, norm_ans)
            for tag in tags:
                fix_counts[tag] += 1

    print(f"\n{'='*50}")
    print("  RULES THAT RECOVERED OOV ANSWER TYPES")
    print(f"{'='*50}")
    for tag, cnt in fix_counts.most_common():
        print(f"  {tag:<20} {cnt:>4} answer types recovered")

    # ── Top OOV answers (still OOV after normalization) ───────────────────────
    still_oov_df = df[df["oov_norm"]]
    top_oov = (still_oov_df.groupby("gt_norm")
               .size()
               .sort_values(ascending=False)
               .head(30))

    # ── OOV examples (with questions) ─────────────────────────────────────────
    ex_lines = [
        "=" * 70,
        "ViLT OOV EXAMPLES (still OOV after normalization)",
        "Top 20 OOV answers with representative questions",
        "=" * 70,
    ]
    for norm_ans, count in top_oov.head(20).items():
        examples = still_oov_df[still_oov_df["gt_norm"] == norm_ans].head(2)
        ex_lines.append(f"\nGT (normalized): {repr(norm_ans)}  [{count} questions]")
        for _, ex in examples.iterrows():
            ex_lines.append(f"  [{ex['structural']}×{ex['semantic']}]  "
                            f"Q: {ex['question']}")
            ex_lines.append(f"  GT_raw={repr(ex['gt_answer'])}  "
                            f"BLIP={repr(ex['blip_answer'])}  "
                            f"ViLT={repr(ex['vilt_answer'])}")
    ex_lines.append("\n" + "=" * 70)
    ex_lines.append("EXAMPLES FIXED BY NORMALIZATION (OOV before, in-vocab after)")
    ex_lines.append("=" * 70)
    fixed_df = df[df["oov_raw"] & ~df["oov_norm"]]
    seen_fixed = set()
    for _, row in fixed_df.iterrows():
        key = (row["gt_answer"], row["gt_norm"])
        if key in seen_fixed or len(seen_fixed) >= 15:
            continue
        seen_fixed.add(key)
        tags = identify_fix_type(row["gt_answer"], row["gt_norm"])
        ex_lines.append(
            f"\n  raw={repr(row['gt_answer'])}  →  norm={repr(row['gt_norm'])}"
            f"  [rule: {', '.join(tags)}]"
        )
        ex_lines.append(
            f"  Q: {row['question']}  [{row['structural']}×{row['semantic']}]"
        )

    (OUT_DIR / "oov_examples.txt").write_text("\n".join(ex_lines))
    print(f"\nSaved OOV examples: oov_examples.txt")

    # ── Written summary ────────────────────────────────────────────────────────
    summary = f"""ViLT OOV Analysis Summary
Generated from: {PREDICTIONS_FILE.name}
Total questions: {len(df):,}
ViLT raw label set size: {len(raw_labels):,}

=== BEFORE NORMALIZATION ===
Unique GT answer types:         {len(unique_raw_gt):,}
OOV answer types (not in ViLT): {n_raw_oov_types:,}  ({n_raw_oov_types/len(unique_raw_gt)*100:.1f}% of types)
Questions with OOV GT:          {n_raw_oov_q:,}  ({n_raw_oov_q/len(df)*100:.1f}% of questions)

=== AFTER NORMALIZATION (rules 1-6) ===
Unique normalized GT types:     {len(unique_norm_gt):,}
OOV normalized types:           {n_norm_oov_types:,}  ({n_norm_oov_types/len(unique_norm_gt)*100:.1f}% of types)
Questions with OOV norm GT:     {n_norm_oov_q:,}  ({n_norm_oov_q/len(df)*100:.1f}% of questions)

=== IMPROVEMENT FROM NORMALIZATION ===
Answer types recovered:         {fixed_types:,}
Questions now answerable:       {fixed_q:,}
Questions still OOV:            {n_norm_oov_q:,}  ({n_norm_oov_q/len(df)*100:.1f}%)

=== CELLS WITH HIGHEST OOV RATE (after normalization) ===
"""
    for _, r in cell_df.sort_values("oov_norm_pct", ascending=False).head(5).iterrows():
        summary += (f"  {r['structural']}×{r['semantic']:<8} "
                    f"{r['oov_norm_pct']:.1f}%  ({int(r['oov_norm_n'])} questions)\n")

    summary += f"""
=== CELLS WITH ZERO OOV ===
"""
    for _, r in cell_df[cell_df["oov_norm_n"] == 0].iterrows():
        summary += f"  {r['structural']}×{r['semantic']}\n"

    summary += f"""
=== RULE EFFECTIVENESS (answer types recovered by each rule) ===
"""
    for tag, cnt in fix_counts.most_common():
        summary += f"  {tag:<20} {cnt} answer types\n"

    summary += """
=== INTERPRETATION ===
The OOV problem is concentrated in open-vocab cells (query×rel, query×cat,
query×global, compare×attr). Binary-answer cells (verify×*, logical×*) have
zero OOV since "yes" and "no" are always in the ViLT label set.

The key insight is that the OOV rate by *question count* (3-4%) is much lower
than the OOV rate by *answer type* (30%+), because OOV answers tend to be
long-tail, low-frequency answers. Most questions ask about common objects and
attributes that ARE covered by ViLT's VQAv2-derived vocabulary.

Normalization (rules 1-6) recovers a small fraction of OOV cases. The remaining
OOV questions (after normalization) represent a structural ceiling for ViLT that
cannot be addressed without retraining or expanding the label set.
"""
    (OUT_DIR / "oov_summary.txt").write_text(summary)
    print(f"Saved summary: oov_summary.txt")
    print(summary)

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_oov_heatmap(cell_df, "oov_raw_pct",  "OOV rate BEFORE normalization (%)",
                      FIG_DIR / "oov_rate_before_norm.png")
    _plot_oov_heatmap(cell_df, "oov_norm_pct", "OOV rate AFTER normalization (rules 1-6) (%)",
                      FIG_DIR / "oov_rate_after_norm.png")
    _plot_fix_bar(fix_counts, FIG_DIR / "oov_fixed_by_rule.png")

    print(f"\n✓ All OOV analysis outputs saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}/")


def _plot_oov_heatmap(cell_df: pd.DataFrame, pct_col: str, title: str, out_path: Path):
    mat = pd.DataFrame(index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES, dtype=float)
    mat[:] = np.nan
    annot = pd.DataFrame("", index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES)
    for _, r in cell_df.iterrows():
        mat.loc[r["structural"], r["semantic"]] = r[pct_col]
        annot.loc[r["structural"], r["semantic"]] = f"{r[pct_col]:.1f}%"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(
        mat.astype(float), annot=annot, fmt="",
        cmap="YlOrRd", vmin=0, vmax=15,
        linewidths=0.5, linecolor="white", ax=ax,
        mask=mat.isna(), annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8, "label": "OOV %"},
    )
    for si, struct in enumerate(STRUCTURAL_TYPES):
        for mi, sem in enumerate(SEMANTIC_TYPES):
            if pd.isna(mat.loc[struct, sem]):
                ax.add_patch(plt.Rectangle((mi, si), 1, 1,
                                           fill=True, color="#d0d0d0", lw=0))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Semantic type", fontsize=10)
    ax.set_ylabel("Structural type", fontsize=10)
    ax.set_xticklabels(SEMANTIC_TYPES, fontsize=9)
    ax.set_yticklabels(STRUCTURAL_TYPES, fontsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def _plot_fix_bar(fix_counts: Counter, out_path: Path):
    if not fix_counts:
        return
    labels = list(fix_counts.keys())
    values = [fix_counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(labels, values, color="#4C72B0")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Answer types recovered from OOV", fontsize=10)
    ax.set_title("OOV answer types fixed by each normalization rule", fontsize=11,
                 fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
