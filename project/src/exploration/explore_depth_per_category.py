#!/usr/bin/env python3
"""
src/exploration/explore_depth_per_category.py

Explores program depth distribution across question categories.

Questions answered:
  - What is the depth distribution for each structural type?
  - What is the mean depth per 5x5 cell (heatmap)?
  - What do questions look like at each depth level within each structural type?
  - Which operations appear at which depths?
  - How do the operations used differ by depth?

Program depth = len(q['semantic']) = number of reasoning steps.

Outputs (in results/exploration/depth_distribution/):
  figures/
    depth_hist_by_structural.png   — depth histogram per structural type (5 subplots)
    depth_mean_heatmap.png         — mean program depth per 5x5 cell
    depth_operations_heatmap.png   — which operations appear at each depth level
  depth_examples.txt               — example Q+A+program per structural type per depth bin
  depth_stats.csv                  — per-cell depth statistics
"""

import json
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
OUT_DIR      = PROJECT_ROOT / "results" / "exploration" / "depth_distribution"
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

DEPTH_COLORS = {2: "#4e9af1", 3: "#3ec97c", 4: "#f5a623", 5: "#e74c3c",
                6: "#9b59b6", 7: "#1abc9c", 8: "#e67e22", 9: "#c0392b"}


def load_data() -> dict:
    print("Loading val_balanced_questions.json …")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"  {len(data):,} questions loaded")
    return data


def get_depth(q: dict) -> int:
    return len(q["semantic"])


# ── Analysis 1: depth histograms per structural type ─────────────────────────
def plot_depth_histograms(data: dict):
    by_struct = defaultdict(list)
    for q in data.values():
        by_struct[q["types"]["structural"]].append(get_depth(q))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, st in zip(axes, STRUCTURAL):
        depths = by_struct[st]
        counter = Counter(depths)
        xs = sorted(counter)
        ys = [counter[x] for x in xs]
        colors = [DEPTH_COLORS.get(x, "#888888") for x in xs]
        bars = ax.bar([str(x) for x in xs], ys, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{st}\n(n={len(depths):,})", fontsize=10)
        ax.set_xlabel("Program depth")
        ax.set_ylabel("# questions")
        mean = np.mean(depths)
        ax.set_title(f"{st}\nn={len(depths):,}, mean={mean:.2f}", fontsize=10)

        # annotate bars with percentage
        total = sum(ys)
        for bar, y in zip(bars, ys):
            pct = 100 * y / total
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Program Depth Distribution by Structural Type", fontsize=13, y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "depth_hist_by_structural.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Analysis 2: mean depth heatmap per 5x5 cell ──────────────────────────────
def plot_depth_heatmap(data: dict):
    sums   = defaultdict(list)
    for q in data.values():
        key = (q["types"]["structural"], q["types"]["semantic"])
        sums[key].append(get_depth(q))

    mean_mat = pd.DataFrame(index=STRUCTURAL, columns=SEMANTIC, dtype=float)
    std_mat  = pd.DataFrame(index=STRUCTURAL, columns=SEMANTIC, dtype=float)
    for (st, sem), depths in sums.items():
        if (st, sem) in VALID_CELLS:
            mean_mat.loc[st, sem] = np.mean(depths)
            std_mat.loc[st, sem]  = np.std(depths)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mask = mean_mat.isnull()

    for ax, mat, title, fmt in [
        (axes[0], mean_mat, "Mean Program Depth per Cell", ".2f"),
        (axes[1], std_mat,  "Std Dev of Program Depth per Cell", ".2f"),
    ]:
        annot = mat.copy().astype(object)
        for r in mat.index:
            for c in mat.columns:
                v = mat.loc[r, c]
                annot.loc[r, c] = "" if pd.isna(v) else f"{v:.2f}"
        sns.heatmap(mat.astype(float), mask=mask, annot=annot, fmt="",
                    cmap="YlOrRd", linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Depth"})
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Semantic type")
        ax.set_ylabel("Structural type")

    plt.tight_layout()
    path = FIG_DIR / "depth_mean_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Analysis 3: operations by depth level ────────────────────────────────────
def plot_operations_by_depth(data: dict):
    """For each depth level, which operations appear (summed across positions)?"""
    # collect op counts at each depth level
    ops_at_depth = defaultdict(Counter)
    for q in data.values():
        d = get_depth(q)
        for step in q["semantic"]:
            # simplify compound ops like "filter color" -> "filter color"
            ops_at_depth[d][step["operation"]] += 1

    depths = sorted(ops_at_depth)
    # find top-15 operations overall
    total_ops = Counter()
    for c in ops_at_depth.values():
        total_ops.update(c)
    top_ops = [op for op, _ in total_ops.most_common(15)]

    matrix = pd.DataFrame(index=top_ops, columns=[str(d) for d in depths], dtype=float)
    for d in depths:
        total = sum(ops_at_depth[d].values())
        for op in top_ops:
            matrix.loc[op, str(d)] = 100 * ops_at_depth[d].get(op, 0) / total

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.3, ax=ax, cbar_kws={"label": "% of ops at that depth"})
    ax.set_title("Top-15 Operations as % of All Operations at Each Depth Level",
                 fontsize=11, pad=10)
    ax.set_xlabel("Program depth")
    ax.set_ylabel("Operation")
    plt.tight_layout()
    path = FIG_DIR / "depth_operations_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Analysis 4: example questions per structural type per depth bin ────────────
def write_depth_examples(data: dict, n_examples: int = 3):
    # group by (structural, depth)
    groups = defaultdict(list)
    for q in data.values():
        st = q["types"]["structural"]
        d  = get_depth(q)
        groups[(st, d)].append(q)

    lines = []
    lines.append("=" * 80)
    lines.append("PROGRAM DEPTH EXAMPLES BY STRUCTURAL TYPE")
    lines.append("=" * 80)
    lines.append(
        "\nProgram depth = number of steps in the functional program (q['semantic']).\n"
        "Each example shows the question, answer, full answer, and step-by-step program.\n"
    )

    for st in STRUCTURAL:
        depth_keys = sorted(d for (s, d) in groups if s == st)
        lines.append(f"\n{'═' * 70}")
        lines.append(f"  STRUCTURAL TYPE: {st.upper()}")
        depth_dist = Counter(get_depth(q) for q in data.values()
                             if q["types"]["structural"] == st)
        dist_str = "  |  ".join(f"depth {d}: {c:,} ({100*c/sum(depth_dist.values()):.0f}%)"
                                for d, c in sorted(depth_dist.items()))
        lines.append(f"  Depth distribution: {dist_str}")
        lines.append(f"{'═' * 70}")

        for d in depth_keys:
            qs = groups[(st, d)]
            lines.append(f"\n  ── depth {d}  ({len(qs):,} questions) ──────────────────────")

            # pick diverse examples: short q, long q, interesting semantic type
            seen_sem = set()
            chosen = []
            for q in qs:
                sem = q["types"]["semantic"]
                if sem not in seen_sem:
                    chosen.append(q)
                    seen_sem.add(sem)
                if len(chosen) >= n_examples:
                    break
            # fill up if needed
            for q in qs:
                if q not in chosen and len(chosen) < n_examples:
                    chosen.append(q)

            for i, q in enumerate(chosen, 1):
                sem = q["types"]["semantic"]
                lines.append(f"\n    Example {i} [{st}×{sem}]:")
                lines.append(f"      question   : {q['question']}")
                lines.append(f"      answer     : {q['answer']!r}")
                lines.append(f"      fullAnswer : {q['fullAnswer']!r}")
                lines.append(f"      imageId    : {q['imageId']}")
                lines.append(f"      program ({d} steps):")
                for step_i, step in enumerate(q["semantic"]):
                    deps = step["dependencies"]
                    dep_str = f" [deps: {deps}]" if deps else " [root]"
                    lines.append(
                        f"        step {step_i}: {step['operation']}"
                        f"({step['argument']!r}){dep_str}"
                    )

    out_path = OUT_DIR / "depth_examples.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ── Analysis 5: per-cell depth stats CSV ──────────────────────────────────────
def save_depth_stats(data: dict):
    rows = []
    cell_data = defaultdict(list)
    for q in data.values():
        key = (q["types"]["structural"], q["types"]["semantic"])
        cell_data[key].append(get_depth(q))

    for (st, sem), depths in sorted(cell_data.items()):
        if (st, sem) not in VALID_CELLS:
            continue
        counter = Counter(depths)
        rows.append({
            "structural": st, "semantic": sem,
            "n_questions": len(depths),
            "depth_min": min(depths), "depth_max": max(depths),
            "depth_mean": round(np.mean(depths), 3),
            "depth_std":  round(np.std(depths), 3),
            "depth_dist": dict(sorted(counter.items())),
        })

    df = pd.DataFrame(rows)
    path = OUT_DIR / "depth_stats.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")

    print("\n── Per-cell depth statistics ───────────────────────────────────────")
    print(df[["structural", "semantic", "n_questions",
              "depth_min", "depth_max", "depth_mean", "depth_std",
              "depth_dist"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    data = load_data()

    print("\n── Step 1: Depth histograms per structural type ───────────────────")
    plot_depth_histograms(data)

    print("\n── Step 2: Mean depth heatmap per 5×5 cell ────────────────────────")
    plot_depth_heatmap(data)

    print("\n── Step 3: Operations by depth level ──────────────────────────────")
    plot_operations_by_depth(data)

    print("\n── Step 4: Example questions per structural type per depth ─────────")
    write_depth_examples(data)

    print("\n── Step 5: Per-cell depth stats CSV ────────────────────────────────")
    save_depth_stats(data)

    print("\n✓ Done. Outputs in results/exploration/depth_distribution/")


if __name__ == "__main__":
    main()
