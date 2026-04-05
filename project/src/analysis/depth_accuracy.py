#!/usr/bin/env python3
"""
src/analysis/analyze_depth_accuracy.py

Analyses accuracy as a function of program (logic) depth.

Produces:
  results/analysis/depth_accuracy/
    depth_accuracy_overall.png        — aggregate line plot, both models
    depth_accuracy_by_structural.png  — 5-panel per structural type
    depth_accuracy_logical_detail.png — zoom on logical bimodal (depths 4-7)
    depth_accuracy_heatmap.png        — structural × depth bin accuracy heatmap
    depth_accuracy_table.csv          — all numbers: n / BLIP / ViLT / gap per cell
    depth_accuracy_summary.txt        — key observations

Usage:
  python src/analysis/analyze_depth_accuracy.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "depth_accuracy"

STRUCTURAL_TYPES = ["query", "verify", "logical", "choose", "compare"]

# Colour palette — consistent across figures
STRUCT_COLORS = {
    "query":   "#4C72B0",
    "verify":  "#55A868",
    "logical": "#C44E52",
    "choose":  "#8172B2",
    "compare": "#CCB974",
}
BLIP_COLOR = "#2196F3"
VILT_COLOR = "#FF9800"

# Minimum n to draw a solid data point (below this: dashed + open marker)
MIN_N_SOLID = 200


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with one row per (structural, program_depth) combination,
    columns: n, blip_acc, vilt_acc, gap.
    Uses blip_correct_norm / vilt_correct_norm (rules 1-6 normalization).
    """
    df = pd.read_csv(csv_path)

    rows = []
    for struct in STRUCTURAL_TYPES:
        sub = df[df["structural"] == struct]
        for depth in sorted(sub["program_depth"].unique()):
            d_sub = sub[sub["program_depth"] == depth]
            n      = len(d_sub)
            blip   = d_sub["blip_correct_norm"].mean()
            vilt   = d_sub["vilt_correct_norm"].mean()
            rows.append({
                "structural":    struct,
                "program_depth": int(depth),
                "n":             n,
                "blip_acc":      blip,
                "vilt_acc":      vilt,
                "gap":           blip - vilt,
            })

    # Overall (all structural types pooled)
    for depth in sorted(df["program_depth"].unique()):
        d_sub = df[df["program_depth"] == depth]
        n     = len(d_sub)
        blip  = d_sub["blip_correct_norm"].mean()
        vilt  = d_sub["vilt_correct_norm"].mean()
        rows.append({
            "structural":    "overall",
            "program_depth": int(depth),
            "n":             n,
            "blip_acc":      blip,
            "vilt_acc":      vilt,
            "gap":           blip - vilt,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Overall accuracy vs depth (all types pooled)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overall(agg: pd.DataFrame, out_path: Path):
    ov = agg[agg["structural"] == "overall"].copy()

    fig, (ax_acc, ax_n) = plt.subplots(
        2, 1, figsize=(9, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    depths = ov["program_depth"].values
    blip   = ov["blip_acc"].values
    vilt   = ov["vilt_acc"].values
    ns     = ov["n"].values

    ax_acc.plot(depths, blip, "o-", color=BLIP_COLOR, lw=2, ms=7,
                label="BLIP (normalized)")
    ax_acc.plot(depths, vilt, "s--", color=VILT_COLOR, lw=2, ms=7,
                label="ViLT (normalized)")

    # Annotate accuracy values
    for d, b, v in zip(depths, blip, vilt):
        ax_acc.annotate(f"{b:.3f}", (d, b), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5,
                        color=BLIP_COLOR)
        ax_acc.annotate(f"{v:.3f}", (d, v), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7.5,
                        color=VILT_COLOR)

    ax_acc.set_ylabel("Accuracy (normalized)", fontsize=11)
    ax_acc.set_ylim(0.25, 0.85)
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_acc.legend(fontsize=9, loc="upper right")
    ax_acc.set_title("Overall Accuracy vs Logic Depth — GQA Val Balanced",
                     fontsize=12, fontweight="bold")
    ax_acc.grid(axis="y", alpha=0.3)

    # Bottom panel — sample size bar chart
    ax_n.bar(depths, ns, color="#78909C", width=0.6)
    for d, n in zip(depths, ns):
        ax_n.text(d, n + 300, f"{n:,}", ha="center", va="bottom", fontsize=7.5,
                  color="#37474F")
    ax_n.set_ylabel("Number of Questions", fontsize=9)
    ax_n.set_xlabel("Logic Depth", fontsize=11)
    ax_n.set_xticks(depths)
    ax_n.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))
    ax_n.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per structural type (5 panels)
# ══════════════════════════════════════════════════════════════════════════════

def plot_by_structural(agg: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)
    fig.suptitle("Accuracy vs Logic Depth — Per Structural Type",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, struct in zip(axes, STRUCTURAL_TYPES):
        sub = agg[agg["structural"] == struct].sort_values("program_depth")
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        depths = sub["program_depth"].values
        blip   = sub["blip_acc"].values
        vilt   = sub["vilt_acc"].values
        ns     = sub["n"].values
        color  = STRUCT_COLORS[struct]

        # Split solid vs dashed by sample size
        for i in range(len(depths)):
            ls_b = "-"  if ns[i] >= MIN_N_SOLID else "--"
            ls_v = "-"  if ns[i] >= MIN_N_SOLID else "--"
            mk_b = "o"  if ns[i] >= MIN_N_SOLID else "o"
            alpha = 1.0 if ns[i] >= MIN_N_SOLID else 0.5

            if i < len(depths) - 1:
                # Draw segment between i and i+1 using the smaller n to decide style
                min_n = min(ns[i], ns[i+1])
                ls = "-" if min_n >= MIN_N_SOLID else "--"
                ax.plot([depths[i], depths[i+1]], [blip[i], blip[i+1]],
                        ls, color=BLIP_COLOR, lw=2, alpha=0.9 if min_n>=MIN_N_SOLID else 0.5)
                ax.plot([depths[i], depths[i+1]], [vilt[i], vilt[i+1]],
                        ls, color=VILT_COLOR, lw=2, alpha=0.9 if min_n>=MIN_N_SOLID else 0.5)

            ax.scatter(depths[i], blip[i], color=BLIP_COLOR, zorder=5, s=50,
                       marker="o", alpha=alpha)
            ax.scatter(depths[i], vilt[i], color=VILT_COLOR, zorder=5, s=50,
                       marker="s", alpha=alpha)

        # Annotate n under x-axis ticks
        ax.set_xticks(depths)
        ax.set_xticklabels([f"{d}\n(n={n:,})" if n < 1000 else f"{d}\n({n//1000}K)"
                            for d, n in zip(depths, ns)], fontsize=7.5)

        # Accuracy annotations above/below points (only for solid points)
        for d, b, v, n in zip(depths, blip, vilt, ns):
            if n >= MIN_N_SOLID:
                ax.annotate(f"{b:.2f}", (d, b), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=BLIP_COLOR)
                ax.annotate(f"{v:.2f}", (d, v), xytext=(0, -12),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=VILT_COLOR)

        ax.set_title(struct, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Logic Depth", fontsize=9)
        ax.set_ylabel("Accuracy" if struct == "query" else "", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.25, 1.02)
        ax.grid(axis="y", alpha=0.3)

        # Legend only in first panel
        if struct == "query":
            from matplotlib.lines import Line2D
            legend_els = [
                Line2D([0], [0], color=BLIP_COLOR, lw=2, marker="o", label="BLIP"),
                Line2D([0], [0], color=VILT_COLOR, lw=2, marker="s",
                       linestyle="--", label="ViLT"),
            ]
            ax.legend(handles=legend_els, fontsize=8, loc="lower left")

    # Dashed-line note
    fig.text(0.5, -0.03,
             "Dashed segments / faded points: n < 200 (sparse — interpret with caution)",
             ha="center", fontsize=8.5, color="#666666", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Logical bimodal detail
# ══════════════════════════════════════════════════════════════════════════════

def plot_logical_detail(df_raw: pd.DataFrame, out_path: Path):
    """
    Zoom into logical×obj and logical×attr separately to show the bimodal
    depth-5 vs depth-7 pattern in logical×obj.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Logical Questions: Accuracy vs Logic Depth\n"
                 "(logical×attr depth 4–6 | logical×obj depth 5–9)",
                 fontsize=12, fontweight="bold")

    for ax, (sem, title) in zip(axes, [
        ("attr", "logical × attr"),
        ("obj",  "logical × obj"),
    ]):
        sub = df_raw[(df_raw["structural"] == "logical") &
                     (df_raw["semantic"] == sem)]
        rows = []
        for depth in sorted(sub["program_depth"].unique()):
            d_sub = sub[sub["program_depth"] == depth]
            n     = len(d_sub)
            if n < 5:
                continue
            rows.append({
                "depth": int(depth), "n": n,
                "blip":  d_sub["blip_correct_norm"].mean(),
                "vilt":  d_sub["vilt_correct_norm"].mean(),
            })
        if not rows:
            ax.set_visible(False)
            continue
        tbl = pd.DataFrame(rows)

        ax.plot(tbl["depth"], tbl["blip"], "o-", color=BLIP_COLOR, lw=2.5,
                ms=8, label="BLIP")
        ax.plot(tbl["depth"], tbl["vilt"], "s--", color=VILT_COLOR, lw=2.5,
                ms=8, label="ViLT")

        for _, r in tbl.iterrows():
            lbl = f"n={r['n']:,}" if r["n"] < 1000 else f"n={r['n']//1000}K"
            ax.annotate(lbl, (r["depth"], min(r["blip"], r["vilt"])),
                        xytext=(0, -20), textcoords="offset points",
                        ha="center", fontsize=8, color="#555555")
            ax.annotate(f"{r['blip']:.3f}", (r["depth"], r["blip"]),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=BLIP_COLOR)
            ax.annotate(f"{r['vilt']:.3f}", (r["depth"], r["vilt"]),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color=VILT_COLOR)

        # Highlight bimodal gap for logical×obj
        if sem == "obj" and len(tbl) >= 2:
            d5 = tbl[tbl["depth"] == 5]
            d7 = tbl[tbl["depth"] == 7]
            if len(d5) and len(d7):
                ax.axvspan(5.3, 6.7, alpha=0.06, color="grey",
                           label="sparse (n≈85)")
                ax.annotate("Depth-5: single\nrelate+verify chain",
                            xy=(5, d5["blip"].values[0]),
                            xytext=(4.6, d5["blip"].values[0] + 0.06),
                            fontsize=8, color="#555",
                            arrowprops=dict(arrowstyle="->", color="#999", lw=1),
                            ha="right")
                ax.annotate("Depth-7: compound\nlogical (AND/OR)",
                            xy=(7, d7["blip"].values[0]),
                            xytext=(7.2, d7["blip"].values[0] + 0.04),
                            fontsize=8, color="#555",
                            arrowprops=dict(arrowstyle="->", color="#999", lw=1),
                            ha="left")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Logic Depth", fontsize=10)
        ax.set_ylabel("Accuracy (normalized)", fontsize=10)
        ax.set_xticks(tbl["depth"])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.3, 0.85)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap: structural × depth bin
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(agg: pd.DataFrame, out_path: Path):
    """
    3-panel heatmap: BLIP acc | ViLT acc | Gap (BLIP-ViLT)
    Rows = structural type, Columns = depth bin (2,3,4,5,6+)
    Only show cells with n ≥ 30.
    """
    structs    = STRUCTURAL_TYPES
    # Bin 6+ = depths 6,7,8,9 pooled
    depth_bins = [2, 3, 4, 5, 6]
    bin_labels = ["2", "3", "4", "5", "6+"]

    def get_bin(row):
        return min(row["program_depth"], 6)

    agg2 = agg[agg["structural"] != "overall"].copy()
    agg2["depth_bin"] = agg2["program_depth"].apply(lambda d: min(d, 6))

    # Weighted average within bin (re-aggregate using per_question_stats)
    df_raw = pd.read_csv(STATS_CSV)
    df_raw["depth_bin"] = df_raw["program_depth"].apply(lambda d: min(d, 6))

    blip_mat = pd.DataFrame(np.nan, index=structs, columns=depth_bins)
    vilt_mat = pd.DataFrame(np.nan, index=structs, columns=depth_bins)

    for struct in structs:
        for db in depth_bins:
            mask = (df_raw["structural"] == struct) & (df_raw["depth_bin"] == db)
            sub  = df_raw[mask]
            if len(sub) < 30:
                continue
            blip_mat.loc[struct, db] = sub["blip_correct_norm"].mean()
            vilt_mat.loc[struct, db] = sub["vilt_correct_norm"].mean()

    gap_mat = blip_mat - vilt_mat

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    fig.suptitle("Accuracy by Structural Type × Logic Depth Bin",
                 fontsize=12, fontweight="bold", y=1.03)

    panel_info = [
        (blip_mat, "BLIP accuracy (normalized)", "RdYlGn", 0.3, 1.0, None,
         lambda v: f"{v:.2f}"),
        (vilt_mat, "ViLT accuracy (normalized)", "RdYlGn", 0.3, 1.0, None,
         lambda v: f"{v:.2f}"),
        (gap_mat,  "Gap: BLIP − ViLT",           "RdBu_r", -0.15, 0.15, 0.0,
         lambda v: f"{v:+.2f}"),
    ]

    for ax, (mat, title, cmap, vmin, vmax, center, fmt) in zip(axes, panel_info):
        annot = mat.copy().astype(object)
        for s in structs:
            for db in depth_bins:
                v = mat.loc[s, db]
                annot.loc[s, db] = "" if pd.isna(v) else fmt(v)

        sns.heatmap(
            mat.rename(columns=dict(zip(depth_bins, bin_labels))).astype(float),
            annot=annot.rename(columns=dict(zip(depth_bins, bin_labels))),
            fmt="", cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            linewidths=0.5, linecolor="white", ax=ax,
            mask=mat.rename(columns=dict(zip(depth_bins, bin_labels))).isna(),
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.8},
        )
        # Grey-out empty cells
        for si, s in enumerate(structs):
            for di, db in enumerate(depth_bins):
                if pd.isna(mat.loc[s, db]):
                    ax.add_patch(plt.Rectangle((di, si), 1, 1,
                                               fill=True, color="#d0d0d0", lw=0))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Logic Depth bin", fontsize=9)
        ax.set_ylabel("Structural type" if ax is axes[0] else "", fontsize=9)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_yticklabels(structs, fontsize=9, rotation=0)

    fig.text(0.5, -0.04, "Grey cells: n < 30 (insufficient data)",
             ha="center", fontsize=8.5, color="#666666", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE + SUMMARY TEXT
# ══════════════════════════════════════════════════════════════════════════════

def save_table_and_summary(agg: pd.DataFrame, out_dir: Path):
    # CSV
    tbl_path = out_dir / "depth_accuracy_table.csv"
    agg.round(4).to_csv(tbl_path, index=False)
    print(f"  Saved: {tbl_path.relative_to(PROJECT_ROOT)}")

    # Console + text summary
    lines = []
    lines.append("=" * 70)
    lines.append("ACCURACY VS LOGIC DEPTH — KEY NUMBERS")
    lines.append("(normalized exact-match, rules 1-6)")
    lines.append("=" * 70)

    lines.append("\n── OVERALL (all types pooled) ──────────────────────────────")
    ov = agg[agg["structural"] == "overall"].sort_values("program_depth")
    lines.append(f"{'depth':>7}  {'n':>8}  {'BLIP':>7}  {'ViLT':>7}  {'gap':>7}  note")
    lines.append("-" * 65)
    for _, r in ov.iterrows():
        d = int(r["program_depth"])
        note = ""
        if d == 5:
            note = "⚠ 83% logical×obj (binary yes/no) — inflated"
        elif d >= 6:
            note = "mostly logical×obj compound questions"
        lines.append(f"{d:>7}  {int(r['n']):>8,}  {r['blip_acc']:>7.3f}"
                     f"  {r['vilt_acc']:>7.3f}  {r['gap']:>+7.3f}  {note}")

    lines.append("\n── PER STRUCTURAL TYPE ─────────────────────────────────────")
    for struct in STRUCTURAL_TYPES:
        sub = agg[agg["structural"] == struct].sort_values("program_depth")
        lines.append(f"\n  {struct.upper()}")
        lines.append(f"  {'depth':>5}  {'n':>8}  {'BLIP':>7}  {'ViLT':>7}  {'gap':>7}")
        lines.append("  " + "-" * 45)
        for _, r in sub.iterrows():
            sparse = "  (sparse)" if r["n"] < MIN_N_SOLID else ""
            lines.append(f"  {int(r['program_depth']):>5}  {int(r['n']):>8,}"
                         f"  {r['blip_acc']:>7.3f}  {r['vilt_acc']:>7.3f}"
                         f"  {r['gap']:>+7.3f}{sparse}")

    lines.append("\n── KEY OBSERVATIONS ────────────────────────────────────────")
    lines += _key_observations(agg)

    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "depth_accuracy_summary.txt").write_text(summary + "\n")
    print(f"  Saved: {(out_dir / 'depth_accuracy_summary.txt').relative_to(PROJECT_ROOT)}")


def _key_observations(agg: pd.DataFrame) -> list:
    ov = agg[agg["structural"] == "overall"].sort_values("program_depth")
    obs = []

    # Depth-5 inflation
    d5 = ov[ov["program_depth"] == 5]
    d3 = ov[ov["program_depth"] == 3]
    if len(d5) and len(d3):
        obs.append(
            f"1. DEPTH-5 ARTEFACT: overall accuracy at depth-5 "
            f"({d5['blip_acc'].values[0]:.3f} BLIP) is HIGHER than depth-3 "
            f"({d3['blip_acc'].values[0]:.3f} BLIP). This is not because models "
            f"handle complex questions better — it is because 83% of depth-5 "
            f"questions are logical×obj (always binary yes/no), which inflates accuracy."
        )

    # ViLT > BLIP at depth 6+
    d6 = ov[ov["program_depth"] >= 6]
    if len(d6):
        for _, r in d6.iterrows():
            if r["gap"] < 0:
                obs.append(
                    f"2. VILT > BLIP AT DEPTH {int(r['program_depth'])}: "
                    f"ViLT ({r['vilt_acc']:.3f}) leads BLIP ({r['blip_acc']:.3f}) "
                    f"at depth {int(r['program_depth'])}. These are all logical×obj "
                    f"compound questions (AND/OR of two sub-questions). BLIP may "
                    f"over-generate at high complexity; ViLT's classifier is less "
                    f"prone to surface-form errors on binary questions."
                )
                break

    # logical bimodal
    log = agg[agg["structural"] == "logical"].sort_values("program_depth")
    d5_log = log[log["program_depth"] == 5]
    d7_log = log[log["program_depth"] == 7]
    if len(d5_log) and len(d7_log):
        obs.append(
            f"3. LOGICAL BIMODAL: logical questions split at depth-5 (n={int(d5_log['n'].values[0]):,}, "
            f"BLIP={d5_log['blip_acc'].values[0]:.3f}) and depth-7 (n={int(d7_log['n'].values[0]):,}, "
            f"BLIP={d7_log['blip_acc'].values[0]:.3f}). "
            f"Depth-5 = single relate→verify chain; depth-7 = compound AND/OR of two "
            f"sub-questions. Both models drop substantially at depth-7."
        )

    # verify is mostly flat
    ver = agg[agg["structural"] == "verify"].sort_values("program_depth")
    if len(ver) >= 2:
        acc_range = ver["blip_acc"].max() - ver["blip_acc"].min()
        obs.append(
            f"4. VERIFY IS MOSTLY FLAT: BLIP accuracy ranges only "
            f"{acc_range:.3f} across depths {int(ver['program_depth'].min())}–"
            f"{int(ver['program_depth'].max())} for verify questions. "
            f"Binary yes/no format dominates; depth adds little difficulty."
        )

    # query shows meaningful decline
    qry = agg[agg["structural"] == "query"].sort_values("program_depth")
    qry_main = qry[qry["n"] >= MIN_N_SOLID]
    if len(qry_main) >= 2:
        drop = qry_main["blip_acc"].iloc[0] - qry_main["blip_acc"].iloc[-1]
        obs.append(
            f"5. QUERY DECLINES WITH DEPTH: BLIP accuracy drops from "
            f"{qry_main['blip_acc'].iloc[0]:.3f} (depth {int(qry_main['program_depth'].iloc[0])}) "
            f"to {qry_main['blip_acc'].iloc[-1]:.3f} (depth {int(qry_main['program_depth'].iloc[-1])}), "
            f"a decline of {drop:.3f}. Longer reasoning chains genuinely hurt "
            f"open-vocabulary retrieval."
        )

    return obs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {STATS_CSV.name}...")
    df_raw = pd.read_csv(STATS_CSV)
    print(f"  {len(df_raw):,} questions")

    print("\nAggregating by structural × depth...")
    agg = load_and_aggregate(STATS_CSV)

    print("\nGenerating figures...")
    plot_overall(agg, OUT_DIR / "depth_accuracy_overall.png")
    plot_by_structural(agg, OUT_DIR / "depth_accuracy_by_structural.png")
    plot_logical_detail(df_raw, OUT_DIR / "depth_accuracy_logical_detail.png")
    plot_heatmap(agg, OUT_DIR / "depth_accuracy_heatmap.png")

    print("\nSaving table and summary...")
    save_table_and_summary(agg, OUT_DIR)

    print(f"\n✓ All outputs saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
