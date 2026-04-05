#!/usr/bin/env python3
"""
src/exploration/explore_answer_structure.py

Explores the answer structure of GQA val_balanced_questions.json.

Questions answered:
  - How many GT answers does each question have?
  - What does answer vs fullAnswer vs annotations look like per cell?
  - How does answer vocabulary size vary across the 5x5 matrix?
  - What are the top answers in each cell?
  - How do answer lengths differ across structural types?
  - How much do answer vocabularies overlap across cells?

Outputs (in results/exploration/answer_structure/):
  figures/
    answer_vocab_size_heatmap.png   — unique answer count per 5x5 cell
    answer_length_by_structural.png — word-length histogram per structural type
    answer_overlap_heatmap.png      — pairwise Jaccard overlap between cell vocabs
  answer_examples.txt               — annotated examples from all 15 cells
  answer_vocab_stats.csv            — per-cell vocab sizes and top answers
"""

import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "questions1.2" / "val_balanced_questions.json"
OUT_DIR      = PROJECT_ROOT / "results" / "exploration" / "answer_structure"
FIG_DIR      = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── GQA taxonomy ──────────────────────────────────────────────────────────────
STRUCTURAL = ["query", "verify", "logical", "choose", "compare"]
SEMANTIC   = ["rel", "attr", "obj", "cat", "global"]
VALID_CELLS = {
    ("query",   "rel"),  ("query",   "attr"), ("query",   "cat"), ("query",   "global"),
    ("verify",  "rel"),  ("verify",  "attr"), ("verify",  "obj"), ("verify",  "global"),
    ("logical", "attr"), ("logical", "obj"),
    ("choose",  "rel"),  ("choose",  "attr"), ("choose",  "cat"), ("choose",  "global"),
    ("compare", "attr"),
}


def load_data() -> dict:
    print("Loading val_balanced_questions.json …")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"  {len(data):,} questions loaded")
    return data


def bucket_by_cell(data: dict) -> dict:
    """Return dict[(structural, semantic)] -> list of question dicts."""
    cells = defaultdict(list)
    for q in data.values():
        key = (q["types"]["structural"], q["types"]["semantic"])
        cells[key].append(q)
    return cells


# ── Analysis 1: confirm single GT answer ──────────────────────────────────────
def check_answer_multiplicity(data: dict):
    print("\n── GT Answer Multiplicity ─────────────────────────────────────────")
    list_answers  = sum(1 for q in data.values() if isinstance(q["answer"], list))
    empty_answers = sum(1 for q in data.values() if q["answer"] == "")
    multi_ann     = sum(1 for q in data.values() if len(q["annotations"]["answer"]) > 1)
    print(f"  Questions with list-type answer:           {list_answers}")
    print(f"  Questions with empty answer string:        {empty_answers}")
    print(f"  Questions with >1 answer annotation node:  {multi_ann}")
    print(f"  → Every question has exactly 1 GT answer string.")


# ── Analysis 2: vocabulary stats per cell ─────────────────────────────────────
def compute_vocab_stats(cells: dict) -> pd.DataFrame:
    rows = []
    for (st, sem), qs in cells.items():
        answers = [q["answer"] for q in qs]
        counter = Counter(answers)
        top5 = "; ".join(f"{a}({c})" for a, c in counter.most_common(5))
        rows.append({
            "structural": st,
            "semantic":   sem,
            "n_questions": len(qs),
            "n_unique_answers": len(counter),
            "top5_answers": top5,
            "answer_entropy": _entropy(counter),
        })
    df = pd.DataFrame(rows).sort_values(["structural", "semantic"])
    return df


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    probs = [c / total for c in counter.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def plot_vocab_heatmap(df: pd.DataFrame):
    matrix = pd.DataFrame(index=STRUCTURAL, columns=SEMANTIC, dtype=float)
    for _, row in df.iterrows():
        if (row["structural"], row["semantic"]) in VALID_CELLS:
            matrix.loc[row["structural"], row["semantic"]] = row["n_unique_answers"]

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = matrix.isnull()
    annot = matrix.copy().astype(object)
    for r in matrix.index:
        for c in matrix.columns:
            v = matrix.loc[r, c]
            annot.loc[r, c] = "" if pd.isna(v) else f"{int(v):,}"

    sns.heatmap(
        matrix.astype(float), mask=mask, annot=annot, fmt="",
        cmap="YlOrRd", linewidths=0.5, ax=ax,
        cbar_kws={"label": "Unique answers"},
    )
    ax.set_title("Unique Answer Vocabulary Size per 5×5 Cell", fontsize=13, pad=12)
    ax.set_xlabel("Semantic type"); ax.set_ylabel("Structural type")
    plt.tight_layout()
    path = FIG_DIR / "answer_vocab_size_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Analysis 3: answer word-length distribution by structural type ─────────────
def plot_answer_length_distribution(cells: dict):
    by_struct = defaultdict(list)
    for (st, _), qs in cells.items():
        for q in qs:
            by_struct[st].append(len(q["answer"].split()))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for ax, st in zip(axes, STRUCTURAL):
        lengths = by_struct[st]
        counts = Counter(lengths)
        max_len = max(counts)
        xs = list(range(1, max_len + 1))
        ys = [counts.get(x, 0) for x in xs]
        ax.bar(xs, ys, color="steelblue", edgecolor="white")
        ax.set_title(st, fontsize=11)
        ax.set_xlabel("Word count")
        ax.set_ylabel("# questions")
        mean = np.mean(lengths)
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.2, label=f"mean={mean:.2f}")
        ax.legend(fontsize=8)
        # annotate unique answers
        n_unique = len(set(q["answer"] for qs_list in cells.values() for q in qs_list
                           if q["types"]["structural"] == st))
        ax.set_xlabel(f"Word count  (unique answers: {n_unique:,})")

    fig.suptitle("Answer Word-Length Distribution by Structural Type", fontsize=13, y=1.01)
    plt.tight_layout()
    path = FIG_DIR / "answer_length_by_structural.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Analysis 4: pairwise vocabulary Jaccard overlap ───────────────────────────
def plot_vocab_overlap(cells: dict):
    cell_keys = sorted(VALID_CELLS, key=lambda x: (STRUCTURAL.index(x[0]), SEMANTIC.index(x[1])))
    cell_labels = [f"{st[:3]}×{sem}" for st, sem in cell_keys]
    vocabs = {k: set(q["answer"] for q in cells[k]) for k in cell_keys if k in cells}

    n = len(cell_keys)
    matrix = np.zeros((n, n))
    for i, k1 in enumerate(cell_keys):
        for j, k2 in enumerate(cell_keys):
            v1, v2 = vocabs.get(k1, set()), vocabs.get(k2, set())
            if v1 and v2:
                matrix[i, j] = len(v1 & v2) / len(v1 | v2)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=cell_labels, yticklabels=cell_labels,
                cmap="Blues", annot=True, fmt=".2f", linewidths=0.3,
                ax=ax, cbar_kws={"label": "Jaccard similarity"})
    ax.set_title("Pairwise Answer Vocabulary Overlap (Jaccard) Between Cells", fontsize=12, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = FIG_DIR / "answer_overlap_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Analysis 5: annotated examples from all 15 cells ─────────────────────────
def write_examples(cells: dict, n_per_cell: int = 3):
    lines = []
    lines.append("=" * 80)
    lines.append("GQA ANSWER STRUCTURE: ANNOTATED EXAMPLES PER CELL")
    lines.append("=" * 80)
    lines.append(
        "\nEach question has exactly ONE ground-truth answer (a short string).\n"
        "Fields shown per example:\n"
        "  question    : the natural language question\n"
        "  answer      : the GT answer (what models must match)\n"
        "  fullAnswer  : a full sentence form of the answer\n"
        "  annotations : scene-graph object IDs referenced by answer/question\n"
        "  program     : the functional program (semantic field)\n"
    )

    cell_order = sorted(VALID_CELLS, key=lambda x: (STRUCTURAL.index(x[0]), SEMANTIC.index(x[1])))

    for (st, sem) in cell_order:
        qs = cells.get((st, sem), [])
        if not qs:
            continue
        counter  = Counter(q["answer"] for q in qs)
        n_unique = len(counter)
        top3     = counter.most_common(3)
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  CELL: {st.upper()} × {sem.upper()}   "
                     f"(n={len(qs):,}, unique_answers={n_unique:,})")
        lines.append(f"  Top-3 answers: {top3}")
        lines.append(f"{'─' * 70}")

        # show diverse examples: one common answer, one rare answer, one mid
        chosen = []
        by_answer = defaultdict(list)
        for q in qs:
            by_answer[q["answer"]].append(q)
        # most common answer
        chosen.append(by_answer[counter.most_common(1)[0][0]][0])
        # least common answer (unique)
        rare_ans = [a for a, c in counter.items() if c == 1]
        if rare_ans:
            chosen.append(by_answer[rare_ans[0]][0])
        # longest answer (by word count)
        longest = max(qs, key=lambda q: len(q["answer"].split()))
        if longest not in chosen:
            chosen.append(longest)
        chosen = chosen[:n_per_cell]

        for i, q in enumerate(chosen, 1):
            prog_str = " → ".join(
                f"{step['operation']}({step['argument']!r})"
                for step in q["semantic"]
            )
            ann_summary = {k: list(v.keys()) for k, v in q["annotations"].items() if v}
            lines.append(f"\n  Example {i}:")
            lines.append(f"    question   : {q['question']}")
            lines.append(f"    answer     : {q['answer']!r}")
            lines.append(f"    fullAnswer : {q['fullAnswer']!r}")
            lines.append(f"    imageId    : {q['imageId']}")
            lines.append(f"    annotations: {ann_summary}")
            prog_wrapped = textwrap.fill(prog_str, width=70,
                                         initial_indent="    program    : ",
                                         subsequent_indent="                 ")
            lines.append(prog_wrapped)

    out_path = OUT_DIR / "answer_examples.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ── Save vocab stats CSV ───────────────────────────────────────────────────────
def save_vocab_stats(df: pd.DataFrame):
    path = OUT_DIR / "answer_vocab_stats.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")

    print("\n── Per-cell vocabulary summary ────────────────────────────────────")
    print(df[["structural", "semantic", "n_questions", "n_unique_answers",
              "answer_entropy", "top5_answers"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    data  = load_data()
    cells = bucket_by_cell(data)

    print("\n── Step 1: GT answer multiplicity check ───────────────────────────")
    check_answer_multiplicity(data)

    print("\n── Step 2: Vocabulary stats per cell ──────────────────────────────")
    df = compute_vocab_stats(cells)
    save_vocab_stats(df)
    plot_vocab_heatmap(df)

    print("\n── Step 3: Answer word-length distributions ───────────────────────")
    plot_answer_length_distribution(cells)

    print("\n── Step 4: Pairwise vocabulary overlap ────────────────────────────")
    plot_vocab_overlap(cells)

    print("\n── Step 5: Annotated examples ─────────────────────────────────────")
    write_examples(cells)

    print("\n✓ Done. Outputs in results/exploration/answer_structure/")


if __name__ == "__main__":
    main()
