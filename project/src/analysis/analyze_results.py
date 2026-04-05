#!/usr/bin/env python3
"""
src/analysis/analyze_results.py

Runs three evaluation modes on results/predictions/all_predictions.jsonl:

  strict               — exact match after standard normalization (lowercase,
                         number words→digits, strip leading article)
  normalized           — extends strict with British→American spelling and
                         plural→singular (rules 1-6)
  normalized_OOV_excluded — normalized, restricted to questions where the
                         normalized GT answer is in ViLT's label set
                         (both models evaluated on the same subset)

Output directories:
  results/strict/
  results/normalized/
  results/normalized_OOV_excluded/

Each contains:
  accuracy_5x5_blip.csv / accuracy_5x5_vilt.csv / accuracy_5x5_gap.csv
  accuracy_heatmap_blip.png / accuracy_heatmap_vilt.png / accuracy_heatmap_gap.png
  accuracy_by_group.csv
  summary.txt

Additional outputs at results/analysis/blip_vilt_comparison/:
  comparison_table.csv       — all modes side-by-side per cell
  comparison_heatmaps.png    — 5-panel figure for the report

Usage:
  python src/analysis/analyze_results.py
  python src/analysis/analyze_results.py --predictions results/predictions/all_predictions.jsonl
  python src/analysis/analyze_results.py --skip-oov   # skip loading ViLT model (faster)
"""

import argparse
import json
import logging
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# suppress transformers noise when loading ViLT labels
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "predictions" / "all_predictions.jsonl"
RESULTS_DIR      = PROJECT_ROOT / "results"
ANALYSIS_DIR     = RESULTS_DIR / "analysis"

# ── GQA 5×5 matrix definition ──────────────────────────────────────────────────
STRUCTURAL_TYPES = ["query", "verify", "logical", "choose", "compare"]
SEMANTIC_TYPES   = ["rel", "attr", "obj", "cat", "global"]

VALID_CELLS = {
    ("query",   "rel"),  ("query",   "attr"), ("query",   "cat"), ("query",   "global"),
    ("verify",  "rel"),  ("verify",  "attr"), ("verify",  "obj"), ("verify",  "global"),
    ("logical", "attr"), ("logical", "obj"),
    ("choose",  "rel"),  ("choose",  "attr"), ("choose",  "cat"), ("choose",  "global"),
    ("compare", "attr"),
}

# ── Capability group definitions ────────────────────────────────────────────────
STRUCTURAL_GROUPS = {
    "S1 Open-ended retrieval (query)":          lambda s, _: s == "query",
    "S2 Binary perception (verify + logical)":  lambda s, _: s in ("verify", "logical"),
    "S3 Constrained choice (choose + compare)": lambda s, _: s in ("choose", "compare"),
}
SEMANTIC_GROUPS = {
    "V1 Relational/spatial (rel)":     lambda _, m: m == "rel",
    "V2 Attribute recognition (attr)": lambda _, m: m == "attr",
    "V3 Object detection (obj)":       lambda _, m: m == "obj",
    "V4 Categorization (cat)":         lambda _, m: m == "cat",
    "V5 Scene understanding (global)": lambda _, m: m == "global",
}

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

_NUM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Rule 5: British → American spelling (word-level, applied after lowercase)
_BRIT_TO_AM = {
    "grey": "gray", "greys": "grays",
    "blonde": "blond",                      # feminine French form → standard
    "colour": "color", "colours": "colors", "coloured": "colored",
    "behaviour": "behavior", "behaviours": "behaviors",
    "centre": "center", "centres": "centers",
    "favourite": "favorite", "favourites": "favorites",
    "organised": "organized", "organisation": "organization",
    "realise": "realize", "recognised": "recognized",
    "catalogue": "catalog", "licence": "license",
    "theatre": "theater", "fibre": "fiber",
    "aeroplane": "airplane",
}

# Safe exceptions for depluralize — do NOT strip the trailing 's'
_PLURAL_EXCEPTIONS = frozenset({
    "yes", "bus", "gas", "class", "grass", "glass", "mass", "pass", "kiss",
    "dress", "stress", "press", "cross", "moss", "loss", "boss",
})


def normalize_strict(s: str) -> str:
    """
    Standard normalization — the conventional VQA baseline.
    Rules applied:
      1. strip whitespace + lowercase
      3. strip leading article (a / an / the)
      4. number words → digits (single-word only)
    """
    s = s.strip().lower()
    # Rule 4: number word → digit (only if the entire string is a number word)
    if s in _NUM_MAP:
        s = _NUM_MAP[s]
    # Rule 3: strip leading article
    s = re.sub(r"^(a |an |the )", "", s)
    return s


def _depluralize(word: str) -> str:
    """
    Conservative plural → singular (Rule 6).
    Only strips when the pattern is unambiguous.
    """
    if word in _PLURAL_EXCEPTIONS or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"          # berries → berry
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"          # leaves → leaf
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]                # buses → bus
    if word.endswith("xes") and len(word) > 4:
        return word[:-2]                # boxes → box
    if word.endswith("zes") and len(word) > 4:
        return word[:-2]                # buzzes → buzz
    if word.endswith("ches") and len(word) > 5:
        return word[:-2]                # benches → bench
    if word.endswith("shes") and len(word) > 5:
        return word[:-2]                # dishes → dish
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]                # chairs → chair, dogs → dog
    return word


def normalize_normalized(s: str) -> str:
    """
    Extended normalization — rules 1-6.
    Builds on normalize_strict, adding:
      2. strip punctuation at start/end of full string
      5. British → American spelling (word-level)
      6. plural → singular (word-level, conservative)
    """
    s = normalize_strict(s)
    # Rule 2: strip leading/trailing punctuation from full string
    s = s.strip(".,!?;:\"'()-")
    # Rule 5: British → American (word by word)
    words = s.split()
    words = [_BRIT_TO_AM.get(w, w) for w in words]
    # Rule 6: depluralize each word
    words = [_depluralize(w) for w in words]
    return " ".join(words)


def is_correct(pred: str, gt: str, norm_fn) -> bool:
    """Return True if pred matches gt under norm_fn (symmetric exact match)."""
    return norm_fn(str(pred)) == norm_fn(str(gt))


# ══════════════════════════════════════════════════════════════════════════════
# ViLT OOV UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_vilt_norm_labels(norm_fn) -> set:
    """
    Load ViLT's 3,129-label set and return the set of *normalized* label strings.
    Normalized labels are used for OOV checking — a GT answer is answerable iff
    its normalized form appears in this set.
    """
    from transformers import ViltForQuestionAnswering
    print("  Loading ViLT label set...")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    raw_labels = set(model.config.id2label.values())
    del model  # free memory
    return {norm_fn(l) for l in raw_labels}


def flag_vilt_oov(df: pd.DataFrame, norm_fn, norm_labels: set) -> pd.DataFrame:
    """
    Add a boolean column 'vilt_oov':
      True  → normalized GT is NOT in ViLT's normalized label set
               (ViLT structurally cannot answer this question correctly)
      False → answerable
    """
    df = df.copy()
    df["vilt_oov"] = df["gt_answer"].apply(
        lambda gt: norm_fn(str(gt)) not in norm_labels
    )
    n_oov = df["vilt_oov"].sum()
    pct   = n_oov / len(df) * 100
    print(f"  ViLT OOV questions (after normalization): {n_oov:,} / {len(df):,} ({pct:.1f}%)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} predictions from {path.name}")
    return df


def add_correctness_columns(df: pd.DataFrame, norm_fn, suffix: str) -> pd.DataFrame:
    """Add blip_correct_{suffix} and vilt_correct_{suffix} columns using norm_fn."""
    df = df.copy()
    for model, ans_col in [("blip", "blip_answer"), ("vilt", "vilt_answer")]:
        col = f"{model}_correct_{suffix}"
        df[col] = df.apply(
            lambda r: is_correct(r[ans_col], r["gt_answer"], norm_fn)
            if pd.notna(r[ans_col]) else None,
            axis=1,
        )
    return df


def build_matrix(df: pd.DataFrame, correct_col: str) -> pd.DataFrame:
    """Build 5×5 accuracy matrix from a (possibly pre-filtered) DataFrame."""
    matrix = pd.DataFrame(index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES, dtype=float)
    matrix[:] = np.nan
    valid = df[correct_col].notna()
    for (struct, sem) in VALID_CELLS:
        mask = (df["structural"] == struct) & (df["semantic"] == sem) & valid
        sub  = df.loc[mask, correct_col]
        if len(sub) > 0:
            matrix.loc[struct, sem] = sub.mean()
    return matrix


def build_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES, dtype=float)
    matrix[:] = np.nan
    for (struct, sem) in VALID_CELLS:
        n = ((df["structural"] == struct) & (df["semantic"] == sem)).sum()
        if n > 0:
            matrix.loc[struct, sem] = n
    return matrix


def print_matrix(matrix: pd.DataFrame, title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")
    print(f"{'':12s}" + "".join(f"{c:>10s}" for c in SEMANTIC_TYPES))
    print("-" * 70)
    for struct in STRUCTURAL_TYPES:
        row = f"{struct:12s}"
        for sem in SEMANTIC_TYPES:
            v = matrix.loc[struct, sem]
            row += f"{'—':>10s}" if pd.isna(v) else f"{v:>10.3f}"
        print(row)
    print("-" * 70)
    row = f"{'SEM AVG':12s}"
    for sem in SEMANTIC_TYPES:
        vals = [matrix.loc[s, sem] for s in STRUCTURAL_TYPES
                if not pd.isna(matrix.loc[s, sem])]
        row += f"{np.mean(vals):>10.3f}" if vals else f"{'—':>10s}"
    print(row)
    print("\nSTRUCT AVG:")
    for struct in STRUCTURAL_TYPES:
        vals = [matrix.loc[struct, s] for s in SEMANTIC_TYPES
                if not pd.isna(matrix.loc[struct, s])]
        if vals:
            print(f"  {struct:12s}  {np.mean(vals):.3f}  (over {len(vals)} cells)")


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path,
                 cmap: str = "RdYlGn", vmin=0.0, vmax=1.0,
                 center=None, fmt_fn=None):
    annot = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    for struct in STRUCTURAL_TYPES:
        for sem in SEMANTIC_TYPES:
            v = matrix.loc[struct, sem]
            if not pd.isna(v):
                annot.loc[struct, sem] = fmt_fn(v) if fmt_fn else f"{v:.3f}"

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        matrix.astype(float), annot=annot, fmt="",
        cmap=cmap, vmin=vmin, vmax=vmax, center=center,
        linewidths=0.5, linecolor="white", ax=ax,
        mask=matrix.isna(), annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8},
    )
    for si, struct in enumerate(STRUCTURAL_TYPES):
        for mi, sem in enumerate(SEMANTIC_TYPES):
            if pd.isna(matrix.loc[struct, sem]):
                ax.add_patch(plt.Rectangle((mi, si), 1, 1,
                                           fill=True, color="#d0d0d0", lw=0))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Semantic type", fontsize=11)
    ax.set_ylabel("Structural type", fontsize=11)
    ax.set_xticklabels(SEMANTIC_TYPES, fontsize=10)
    ax.set_yticklabels(STRUCTURAL_TYPES, fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def compute_group_stats(df: pd.DataFrame, blip_col: str, vilt_col: str) -> pd.DataFrame:
    rows = []
    for group_name, fn in {**STRUCTURAL_GROUPS, **SEMANTIC_GROUPS}.items():
        mask = df.apply(lambda r: fn(r["structural"], r["semantic"]), axis=1)
        sub  = df[mask]
        if len(sub) == 0:
            continue
        row = {"group": group_name, "n": len(sub)}
        for model, col in [("BLIP", blip_col), ("ViLT", vilt_col)]:
            valid      = sub[col].notna()
            row[f"{model}_acc"] = sub.loc[valid, col].mean() if valid.any() else float("nan")
            row[f"{model}_n"]   = valid.sum()
        rows.append(row)
    return pd.DataFrame(rows)


def run_evaluation(df: pd.DataFrame, blip_col: str, vilt_col: str,
                   label: str, out_dir: Path):
    """
    Full evaluation pipeline for one normalization mode.
    df       : DataFrame for BOTH models (already filtered if needed)
    blip_col : correctness column for BLIP
    vilt_col : correctness column for ViLT
    label    : mode name used in titles / filenames
    out_dir  : output directory
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*70}")
    print(f"  EVALUATION MODE: {label.upper()}")
    print(f"{'#'*70}")
    print(f"  Questions in this evaluation: {len(df):,}")

    has_blip = df[blip_col].notna().any()
    has_vilt = df[vilt_col].notna().any()

    summary_lines = [
        f"Evaluation mode: {label}",
        f"Questions processed: {len(df):,}",
    ]

    blip_matrix = vilt_matrix = None

    if has_blip:
        blip_matrix = build_matrix(df, blip_col)
        print_matrix(blip_matrix, f"BLIP ACCURACY [{label}]")
        blip_matrix.to_csv(out_dir / "accuracy_5x5_blip.csv")
        overall = df[blip_col].dropna().mean()
        summary_lines.append(f"BLIP overall accuracy: {overall:.4f}")

    if has_vilt:
        vilt_matrix = build_matrix(df, vilt_col)
        print_matrix(vilt_matrix, f"ViLT ACCURACY [{label}]")
        vilt_matrix.to_csv(out_dir / "accuracy_5x5_vilt.csv")
        overall = df[vilt_col].dropna().mean()
        summary_lines.append(f"ViLT overall accuracy: {overall:.4f}")

    if has_blip and has_vilt:
        gap_matrix = blip_matrix - vilt_matrix
        print_matrix(gap_matrix, f"GAP (BLIP − ViLT) [{label}]")
        gap_matrix.to_csv(out_dir / "accuracy_5x5_gap.csv")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    print(f"\nSaving heatmaps [{label}]:")
    if has_blip:
        plot_heatmap(blip_matrix,
                     f"BLIP Accuracy ({label}) — GQA Val Balanced",
                     out_dir / "accuracy_heatmap_blip.png")
    if has_vilt:
        plot_heatmap(vilt_matrix,
                     f"ViLT Accuracy ({label}) — GQA Val Balanced",
                     out_dir / "accuracy_heatmap_vilt.png")
    if has_blip and has_vilt:
        plot_heatmap(gap_matrix,
                     f"Accuracy Gap BLIP−ViLT ({label}) — positive = BLIP wins",
                     out_dir / "accuracy_heatmap_gap.png",
                     cmap="RdBu_r", vmin=-0.3, vmax=0.3, center=0.0,
                     fmt_fn=lambda v: f"{v:+.3f}")

    # ── Capability groups ──────────────────────────────────────────────────────
    group_df = compute_group_stats(df, blip_col, vilt_col)
    print(f"\n{'='*70}\n  CAPABILITY GROUPS [{label}]\n{'='*70}")
    print(group_df.to_string(index=False))
    group_df.to_csv(out_dir / "accuracy_by_group.csv", index=False)

    # ── Marginals ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  MARGINALS [{label}]\n{'='*70}")
    for axis_name, type_list in [("structural", STRUCTURAL_TYPES), ("semantic", SEMANTIC_TYPES)]:
        print(f"\nBy {axis_name}:")
        for val in type_list:
            sub = df[df[axis_name] == val]
            if len(sub) == 0:
                continue
            row = f"  {val:12s}  n={len(sub):>7,}"
            if has_blip:
                b = sub[blip_col].dropna().mean()
                row += f"  BLIP={b:.3f}"
            if has_vilt:
                v2 = sub[vilt_col].dropna().mean()
                row += f"  ViLT={v2:.3f}"
            if has_blip and has_vilt:
                row += f"  gap={b - v2:+.3f}"
            print(row)

    # ── Summary file ───────────────────────────────────────────────────────────
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nAll [{label}] results saved to: {out_dir.relative_to(PROJECT_ROOT)}/")

    return blip_matrix, vilt_matrix


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE — all modes side by side
# ══════════════════════════════════════════════════════════════════════════════

def save_comparison_table(df_full: pd.DataFrame,
                          df_answerable: pd.DataFrame,
                          out_dir: Path):
    """
    Build and save a per-cell comparison table with columns:
      cell | n_full | n_answerable |
      blip_strict | blip_norm |
      vilt_strict | vilt_norm | vilt_answerable |
      gap_strict  | gap_norm  | gap_answerable
    Also saves a 5-panel comparison heatmap figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        ("blip_strict",      df_full,       "blip_correct_strict"),
        ("blip_norm",        df_full,       "blip_correct_norm"),
        ("vilt_strict",      df_full,       "vilt_correct_strict"),
        ("vilt_norm",        df_full,       "vilt_correct_norm"),
        ("vilt_answerable",  df_answerable, "vilt_correct_norm"),
    ]

    # Build per-cell rows
    cell_rows = []
    for struct, sem in sorted(VALID_CELLS):
        row = {
            "structural":   struct,
            "semantic":     sem,
            "n_full":       int(((df_full["structural"] == struct) &
                                 (df_full["semantic"] == sem)).sum()),
            "n_answerable": int(((df_answerable["structural"] == struct) &
                                 (df_answerable["semantic"] == sem)).sum()),
        }
        for col_name, df_, corr_col in modes:
            mask = (df_["structural"] == struct) & (df_["semantic"] == sem)
            sub  = df_.loc[mask, corr_col].dropna()
            row[col_name] = sub.mean() if len(sub) > 0 else float("nan")

        row["gap_strict"]     = row["blip_strict"]     - row["vilt_strict"]
        row["gap_norm"]       = row["blip_norm"]       - row["vilt_norm"]
        row["gap_answerable"] = row["blip_norm"]       - row["vilt_answerable"]
        cell_rows.append(row)

    tbl = pd.DataFrame(cell_rows)
    tbl_path = out_dir / "comparison_table.csv"
    tbl.to_csv(tbl_path, index=False, float_format="%.4f")
    print(f"\nSaved comparison table: {tbl_path.relative_to(PROJECT_ROOT)}")

    # Pretty-print to console
    print(f"\n{'='*100}")
    print("  FULL COMPARISON TABLE (per cell)")
    print(f"{'='*100}")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(tbl.to_string(index=False))

    # ── Overall summary row ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  OVERALL ACCURACY SUMMARY")
    print(f"{'='*70}")
    for col_name, df_, corr_col in modes:
        acc = df_[corr_col].dropna().mean()
        n   = len(df_)
        print(f"  {col_name:20s}  acc={acc:.4f}  (n={n:,})")

    # ── 5-panel heatmap figure ─────────────────────────────────────────────────
    _save_comparison_figure(tbl, out_dir / "comparison_heatmaps.png")


def _save_comparison_figure(tbl: pd.DataFrame, out_path: Path):
    """5-panel heatmap: BLIP strict | BLIP norm | ViLT strict | ViLT norm | ViLT answerable."""
    panels = [
        ("blip_strict",    "BLIP\nStrict"),
        ("blip_norm",      "BLIP\nNormalized"),
        ("vilt_strict",    "ViLT\nStrict"),
        ("vilt_norm",      "ViLT\nNormalized"),
        ("vilt_answerable","ViLT\nAnswerable"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle("Accuracy by cell — all evaluation modes", fontsize=13,
                 fontweight="bold", y=1.01)

    for ax, (col, title) in zip(axes, panels):
        # Pivot tbl into 5×5 matrix
        mat = pd.DataFrame(index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES, dtype=float)
        mat[:] = np.nan
        for _, r in tbl.iterrows():
            mat.loc[r["structural"], r["semantic"]] = r[col]

        annot = pd.DataFrame("", index=mat.index, columns=mat.columns)
        for struct in STRUCTURAL_TYPES:
            for sem in SEMANTIC_TYPES:
                v = mat.loc[struct, sem]
                if not pd.isna(v):
                    annot.loc[struct, sem] = f"{v:.2f}"

        sns.heatmap(
            mat.astype(float), annot=annot, fmt="",
            cmap="RdYlGn", vmin=0.0, vmax=1.0,
            linewidths=0.4, linecolor="white", ax=ax,
            mask=mat.isna(), annot_kws={"size": 7},
            cbar=(ax is axes[-1]),
            cbar_kws={"shrink": 0.8} if ax is axes[-1] else {},
        )
        for si, struct in enumerate(STRUCTURAL_TYPES):
            for mi, sem in enumerate(SEMANTIC_TYPES):
                if pd.isna(mat.loc[struct, sem]):
                    ax.add_patch(plt.Rectangle((mi, si), 1, 1,
                                               fill=True, color="#d0d0d0", lw=0))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Semantic", fontsize=8)
        ax.set_ylabel("Structural" if ax is axes[0] else "", fontsize=8)
        ax.set_xticklabels(SEMANTIC_TYPES, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(STRUCTURAL_TYPES, fontsize=7, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 5-panel figure: {out_path.relative_to(PROJECT_ROOT)}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--skip-oov", action="store_true",
                        help="Skip loading ViLT model for OOV flagging (faster, "
                             "skips normalized_OOV_excluded mode)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_predictions(args.predictions)

    # ── Compute correctness columns ───────────────────────────────────────────
    print("\nComputing strict correctness (rules 1, 3, 4)...")
    df = add_correctness_columns(df, normalize_strict, suffix="strict")

    print("Computing normalized correctness (rules 1-6)...")
    df = add_correctness_columns(df, normalize_normalized, suffix="norm")

    # ── ViLT OOV flagging ─────────────────────────────────────────────────────
    if not args.skip_oov:
        print("\nFlagging ViLT OOV questions...")
        norm_labels = load_vilt_norm_labels(normalize_normalized)
        df = flag_vilt_oov(df, normalize_normalized, norm_labels)
        df_answerable = df[~df["vilt_oov"]].copy()
        print(f"  Answerable subset: {len(df_answerable):,} questions "
              f"({len(df_answerable)/len(df)*100:.1f}% of full set)")
    else:
        print("\n[--skip-oov] Skipping ViLT OOV flagging. "
              "normalized_OOV_excluded mode will not be run.")
        df["vilt_oov"] = False
        df_answerable  = df.copy()

    # ── Save per-question CSV ─────────────────────────────────────────────────
    pq_path = RESULTS_DIR / "per_question_stats.csv"
    df.to_csv(pq_path, index=False)
    print(f"\nSaved per-question stats ({len(df):,} rows): {pq_path.name}")

    # ── Mode 1: strict ────────────────────────────────────────────────────────
    run_evaluation(df,
                   blip_col="blip_correct_strict",
                   vilt_col="vilt_correct_strict",
                   label="strict",
                   out_dir=RESULTS_DIR / "strict")

    # ── Mode 2: normalized ────────────────────────────────────────────────────
    run_evaluation(df,
                   blip_col="blip_correct_norm",
                   vilt_col="vilt_correct_norm",
                   label="normalized",
                   out_dir=RESULTS_DIR / "normalized")

    # ── Mode 3: normalized-OOV-excluded ───────────────────────────────────────
    if not args.skip_oov:
        run_evaluation(df_answerable,
                       blip_col="blip_correct_norm",
                       vilt_col="vilt_correct_norm",
                       label="normalized_OOV_excluded",
                       out_dir=RESULTS_DIR / "normalized_OOV_excluded")

    # ── Side-by-side lift summary ─────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("  STRICT → NORMALIZED LIFT")
    print(f"{'#'*70}")
    for model, strict_col, norm_col in [
        ("BLIP", "blip_correct_strict", "blip_correct_norm"),
        ("ViLT", "vilt_correct_strict", "vilt_correct_norm"),
    ]:
        s       = df[strict_col].dropna().mean()
        n       = df[norm_col].dropna().mean()
        flipped = (df[norm_col] & ~df[strict_col]).sum()
        print(f"  {model}:  strict={s:.4f}  normalized={n:.4f}  "
              f"(+{n-s:.4f})  questions flipped correct: {flipped:,}")

    # ── Comparison table & 5-panel figure ─────────────────────────────────────
    if not args.skip_oov:
        print(f"\n{'#'*70}")
        print("  COMPARISON TABLE (all modes)")
        print(f"{'#'*70}")
        save_comparison_table(df, df_answerable, out_dir=ANALYSIS_DIR / "blip_vilt_comparison")


if __name__ == "__main__":
    main()
