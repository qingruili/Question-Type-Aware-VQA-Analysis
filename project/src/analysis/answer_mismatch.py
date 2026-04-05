#!/usr/bin/env python3
"""
src/analysis/answer_mismatch.py

Finds all BLIP "correct-ish" failures — cases where exact-match reports wrong
but the prediction is semantically close to the ground truth.

Categories detected:
  plural          — singular/plural mismatch only  (zebra / zebras)
  blip_superset   — GT is a substring of BLIP pred (park / skate park)
  blip_subset     — BLIP pred is a substring of GT  (train / train car)
  article_only    — differ only by leading a/an/the  (a dog / dog)
  prep_prefix     — BLIP adds a preposition prefix   (snow / on snow, park / in park)
  synonym_overlap — share ≥1 content word but neither contains the other

Output:
  results/analysis/answer_mismatch/answer_mismatch.csv   — all cases, one row per question
  results/analysis/answer_mismatch/summary.txt           — counts and accuracy impact per category
"""

import json
import re
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "predictions" / "all_predictions.jsonl"
RESULTS_DIR      = PROJECT_ROOT / "results" / "analysis" / "answer_mismatch"

# ── Normalization (same as inference script) ───────────────────────────────────
_NUM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_PREPS = {"in", "on", "at", "near", "by", "of", "to", "into", "onto",
          "over", "under", "from", "with", "behind", "above", "below"}
_STOP  = {"a", "an", "the", "is", "are", "it", "this", "that"} | _PREPS


def normalize(s: str) -> str:
    s = s.strip().lower()
    if s in _NUM_MAP:
        s = _NUM_MAP[s]
    s = re.sub(r"^(a |an |the )", "", s)
    return s


# ── Mismatch detectors ─────────────────────────────────────────────────────────

def is_plural_only(pred: str, gt: str) -> bool:
    """Differ only by a trailing 's' or 'es'."""
    if pred == gt:
        return False
    return (pred + "s" == gt or pred + "es" == gt or
            gt + "s" == pred or gt + "es" == pred)


def is_article_only(pred_raw: str, gt_raw: str) -> bool:
    """Differ only by a leading article that normalize() already strips."""
    p = normalize(pred_raw)
    g = normalize(gt_raw)
    return p == g and pred_raw != gt_raw


def is_prep_prefix(pred: str, gt: str) -> bool:
    """BLIP adds a prepositional phrase prefix: 'on snow' vs 'snow'."""
    words = pred.split()
    if len(words) >= 2 and words[0] in _PREPS:
        return " ".join(words[1:]) == gt
    return False


def is_blip_superset(pred: str, gt: str) -> bool:
    """GT is a strict substring of BLIP prediction (BLIP adds extra words)."""
    return gt in pred and gt != pred


def is_blip_subset(pred: str, gt: str) -> bool:
    """BLIP prediction is a strict substring of GT (BLIP drops words)."""
    return pred in gt and pred != gt


def is_synonym_overlap(pred: str, gt: str) -> bool:
    """Share at least one meaningful content word but neither contains the other."""
    p_words = {w for w in pred.split() if len(w) > 2 and w not in _STOP}
    g_words = {w for w in gt.split()   if len(w) > 2 and w not in _STOP}
    if not p_words or not g_words:
        return False
    return bool(p_words & g_words)


def classify(pred_raw: str, gt_raw: str):
    """
    Return a list of mismatch categories that apply.
    Returns [] if no match (genuinely wrong).
    """
    pred = normalize(pred_raw)
    gt   = normalize(gt_raw)

    if pred == gt:   # already counted correct by exact-match
        return []

    cats = []

    if is_article_only(pred_raw, gt_raw):
        cats.append("article_only")
    if is_plural_only(pred, gt):
        cats.append("plural")
    if is_prep_prefix(pred, gt):
        cats.append("prep_prefix")
    if is_blip_superset(pred, gt):
        cats.append("blip_superset")
    if is_blip_subset(pred, gt):
        cats.append("blip_subset")
    # synonym_overlap only if no stronger category matched
    if not cats and is_synonym_overlap(pred, gt):
        cats.append("synonym_overlap")

    return cats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions
    rows = []
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} predictions")

    # Focus on BLIP-wrong rows
    blip_wrong = df[df["blip_correct"] == False].copy()
    print(f"BLIP wrong: {len(blip_wrong):,} ({len(blip_wrong)/len(df):.1%})")

    # Classify each wrong prediction
    blip_wrong["mismatch_cats"] = blip_wrong.apply(
        lambda r: classify(str(r["blip_answer"]), str(r["gt_answer"])), axis=1
    )
    blip_wrong["mismatch_label"] = blip_wrong["mismatch_cats"].apply(
        lambda cats: " + ".join(cats) if cats else "genuinely_wrong"
    )
    blip_wrong["is_correctish"] = blip_wrong["mismatch_cats"].apply(bool)

    # ── Build output CSV ───────────────────────────────────────────────────────
    out_cols = [
        "qid", "structural", "semantic", "program_depth",
        "question", "gt_answer", "blip_answer",
        "mismatch_label", "vilt_answer", "vilt_correct",
    ]
    correctish = blip_wrong[blip_wrong["is_correctish"]].sort_values(
        ["mismatch_label", "structural", "semantic"]
    )[out_cols]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "answer_mismatch.csv"
    correctish.to_csv(out_csv, index=False)
    print(f"\nSaved {len(correctish):,} correct-ish cases → {out_csv.name}")

    # ── Summary by category ────────────────────────────────────────────────────
    # Count each category independently (a row can have multiple)
    cat_names = ["plural", "blip_superset", "blip_subset",
                 "prep_prefix", "article_only", "synonym_overlap"]
    cat_counts = {}
    for cat in cat_names:
        mask = blip_wrong["mismatch_cats"].apply(lambda cs: cat in cs)
        cat_counts[cat] = mask.sum()

    total_wrong    = len(blip_wrong)
    total_correct  = (df["blip_correct"] == True).sum()
    total          = len(df)

    lines = []
    lines.append("=" * 65)
    lines.append("BLIP CORRECT-ISH MISMATCH ANALYSIS")
    lines.append("=" * 65)
    lines.append(f"Total questions        : {total:>8,}")
    lines.append(f"BLIP correct           : {total_correct:>8,}  ({total_correct/total:.3%})")
    lines.append(f"BLIP wrong             : {total_wrong:>8,}  ({total_wrong/total:.3%})")
    lines.append(f"  ↳ correct-ish        : {len(correctish):>8,}  ({len(correctish)/total_wrong:.3%} of wrong)")
    lines.append(f"  ↳ genuinely wrong    : {total_wrong - len(correctish):>8,}")
    lines.append("")
    lines.append("Breakdown by mismatch category (can overlap):")
    lines.append(f"  {'Category':<20}  {'Count':>7}  {'% of wrong':>10}  {'Impact on acc':>14}")
    lines.append("  " + "-" * 55)
    for cat in cat_names:
        n = cat_counts[cat]
        pct_wrong  = n / total_wrong
        acc_impact = n / total         # if all counted correct, acc gains this much
        lines.append(f"  {cat:<20}  {n:>7,}  {pct_wrong:>10.2%}  {acc_impact:>+14.3%}")

    lines.append("")
    lines.append("Accuracy if all correct-ish counted as correct:")
    adjusted = (total_correct + len(correctish)) / total
    lines.append(f"  Strict exact-match : {total_correct/total:.4f}")
    lines.append(f"  Relaxed (all fixed): {adjusted:.4f}  (+{adjusted - total_correct/total:.4f})")

    lines.append("")
    lines.append("Correct-ish breakdown by structural type:")
    for struct in ["query", "verify", "logical", "choose", "compare"]:
        sub = blip_wrong[blip_wrong["structural"] == struct]
        n_ci = sub["is_correctish"].sum()
        n_w  = len(sub)
        lines.append(f"  {struct:<10}  wrong={n_w:>6,}  correct-ish={n_ci:>5,}  ({n_ci/n_w:.2%} of wrong)" if n_w else f"  {struct:<10}  (no data)")

    lines.append("")
    lines.append("Correct-ish breakdown by semantic type:")
    for sem in ["rel", "attr", "obj", "cat", "global"]:
        sub = blip_wrong[blip_wrong["semantic"] == sem]
        n_ci = sub["is_correctish"].sum()
        n_w  = len(sub)
        lines.append(f"  {sem:<10}  wrong={n_w:>6,}  correct-ish={n_ci:>5,}  ({n_ci/n_w:.2%} of wrong)" if n_w else f"  {sem:<10}  (no data)")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = RESULTS_DIR / "summary.txt"
    summary_path.write_text(summary_text + "\n")
    print(f"\nSaved summary → {summary_path.name}")


if __name__ == "__main__":
    main()
