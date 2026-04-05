#!/usr/bin/env python3
"""
src/analysis/yesno_balance.py

Yes/No Prediction Balance Analysis

For all binary questions (verify + logical), measures whether BLIP and ViLT
have a systematic yes/no prediction bias — i.e., they predict "yes" (or "no")
more often than the ground-truth distribution warrants, and whether that bias
causes asymmetric accuracy on yes-questions vs. no-questions.

Analyses:
  A. Global bias table: per structural×semantic cell, GT yes-rate vs.
     BLIP/ViLT prediction yes-rate.
  B. Asymmetric accuracy: accuracy on yes-GT vs no-GT questions per cell.
  C. Bias gap = accuracy_yes − accuracy_no (a large gap = strong bias exploitation).
  D. Figures:
       prediction_yesrate.png  — grouped bar chart of yes-rates
       asymmetric_accuracy.png — grouped bar chart of acc_yes vs acc_no
       bias_gap.png            — diverging bar chart of bias gaps

Outputs saved to: results/analysis/yesno_balance/

Usage:
  python3 src/analysis/yesno_balance.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "yesno_balance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLIP_COLOR  = "#2196F3"
VILT_COLOR  = "#FF9800"
GT_COLOR    = "#555555"
MIN_N       = 50   # minimum cell size to report

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data…")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})

# Merge on question id
df = stats.merge(feats[["question_id", "binary_answer"]],
                 left_on="qid", right_on="question_id", how="left")

# Restrict to binary questions (verify + logical)
binary = df[df["structural"].isin({"verify", "logical"})].copy()
print(f"Binary questions: {len(binary):,}  "
      f"(verify={len(binary[binary.structural=='verify']):,}, "
      f"logical={len(binary[binary.structural=='logical']):,})")

# ── Accuracy metric: normalized exact match ──────────────────────────────────
# blip_correct_norm / vilt_correct_norm already computed in per_question_stats
BLIP_CORR = "blip_correct_norm"
VILT_CORR = "vilt_correct_norm"

# ── Helper: confidence interval for proportion ───────────────────────────────
def acc_ci(n_correct, n_total, z=1.96):
    if n_total == 0:
        return float("nan"), float("nan"), float("nan")
    p = n_correct / n_total
    margin = z * np.sqrt(p * (1 - p) / n_total)
    return p, p - margin, p + margin


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis A: GT yes-rate vs. BLIP/ViLT prediction yes-rate
# ═══════════════════════════════════════════════════════════════════════════════

cells = []
for (struct, sem), grp in binary.groupby(["structural", "semantic"]):
    n = len(grp)
    if n < MIN_N:
        continue
    gt_yes_rate   = (grp["gt_answer"].str.lower() == "yes").mean()
    blip_yes_rate = (grp["blip_answer"].str.lower() == "yes").mean()
    vilt_yes_rate = (grp["vilt_answer"].str.lower() == "yes").mean()

    # Accuracy on yes-GT vs no-GT subsets
    yes_grp = grp[grp["gt_answer"].str.lower() == "yes"]
    no_grp  = grp[grp["gt_answer"].str.lower() == "no"]

    blip_acc_yes = yes_grp[BLIP_CORR].mean() if len(yes_grp) >= MIN_N else float("nan")
    blip_acc_no  = no_grp[BLIP_CORR].mean()  if len(no_grp)  >= MIN_N else float("nan")
    vilt_acc_yes = yes_grp[VILT_CORR].mean() if len(yes_grp) >= MIN_N else float("nan")
    vilt_acc_no  = no_grp[VILT_CORR].mean()  if len(no_grp)  >= MIN_N else float("nan")

    cells.append({
        "cell":           f"{struct}×{sem}",
        "structural":     struct,
        "semantic":       sem,
        "n":              n,
        "n_yes_gt":       len(yes_grp),
        "n_no_gt":        len(no_grp),
        "gt_yes_rate":    gt_yes_rate,
        "blip_yes_rate":  blip_yes_rate,
        "vilt_yes_rate":  vilt_yes_rate,
        "blip_acc_yes":   blip_acc_yes,
        "blip_acc_no":    blip_acc_no,
        "vilt_acc_yes":   vilt_acc_yes,
        "vilt_acc_no":    vilt_acc_no,
        "blip_bias_gap":  blip_acc_yes - blip_acc_no if not np.isnan(blip_acc_yes) else float("nan"),
        "vilt_bias_gap":  vilt_acc_yes - vilt_acc_no if not np.isnan(vilt_acc_yes) else float("nan"),
        # overall accuracy
        "blip_acc":       grp[BLIP_CORR].mean(),
        "vilt_acc":       grp[VILT_CORR].mean(),
    })

results = pd.DataFrame(cells)
results.to_csv(OUT_DIR / "bias_table.csv", index=False)


# ── Print results ─────────────────────────────────────────────────────────────
def fmt(x):
    return f"{x:.1%}" if not np.isnan(x) else "  n/a  "

lines = []
lines += ["=" * 80, "EXPERIMENT 4: YES/NO PREDICTION BIAS", "=" * 80, ""]

lines += ["─" * 80,
          "PANEL A: YES-RATE COMPARISON (GT vs. BLIP vs. ViLT)",
          "─" * 80]
header = f"{'Cell':18} {'n':>7}  {'GT yes%':>9}  {'BLIP yes%':>10}  {'ViLT yes%':>10}  {'BLIP Δ':>8}  {'ViLT Δ':>8}"
lines.append(header)
for _, r in results.iterrows():
    blip_d = r.blip_yes_rate - r.gt_yes_rate
    vilt_d = r.vilt_yes_rate - r.gt_yes_rate
    sign_b = "+" if blip_d >= 0 else ""
    sign_v = "+" if vilt_d >= 0 else ""
    lines.append(
        f"{r.cell:18} {int(r.n):>7}  {fmt(r.gt_yes_rate):>9}  "
        f"{fmt(r.blip_yes_rate):>10}  {fmt(r.vilt_yes_rate):>10}  "
        f"{sign_b}{blip_d:+.1%}{'':<3}  {sign_v}{vilt_d:+.1%}"
    )

lines += ["", "─" * 80,
          "PANEL B: ACCURACY ON YES-GT vs NO-GT QUESTIONS",
          "─" * 80]
header2 = (f"{'Cell':18} {'n_yes':>6}  {'n_no':>6}  "
           f"{'BLIP acc_yes':>13}  {'BLIP acc_no':>12}  {'BLIP gap':>10}  "
           f"{'ViLT acc_yes':>13}  {'ViLT acc_no':>12}  {'ViLT gap':>10}")
lines.append(header2)
for _, r in results.iterrows():
    bg = r.blip_bias_gap
    vg = r.vilt_bias_gap
    bg_str = f"{bg:+.1%}" if not np.isnan(bg) else "  n/a"
    vg_str = f"{vg:+.1%}" if not np.isnan(vg) else "  n/a"
    lines.append(
        f"{r.cell:18} {int(r.n_yes_gt):>6}  {int(r.n_no_gt):>6}  "
        f"{fmt(r.blip_acc_yes):>13}  {fmt(r.blip_acc_no):>12}  {bg_str:>10}  "
        f"{fmt(r.vilt_acc_yes):>13}  {fmt(r.vilt_acc_no):>12}  {vg_str:>10}"
    )

lines += ["", "─" * 80,
          "PANEL C: OVERALL ACCURACY (for context)",
          "─" * 80]
header3 = f"{'Cell':18} {'n':>7}  {'BLIP acc':>10}  {'ViLT acc':>10}  {'Gap (B-V)':>11}"
lines.append(header3)
for _, r in results.iterrows():
    gap = r.blip_acc - r.vilt_acc
    lines.append(
        f"{r.cell:18} {int(r.n):>7}  {fmt(r.blip_acc):>10}  "
        f"{fmt(r.vilt_acc):>10}  {gap:>+.1%}"
    )

# Also compute aggregate over all binary questions
lines += ["", "─" * 80,
          "AGGREGATE (all binary questions)",
          "─" * 80]
for struct in ["verify", "logical", "all binary"]:
    subset = binary if struct == "all binary" else binary[binary.structural == struct]
    n = len(subset)
    gt_yr   = (subset["gt_answer"].str.lower() == "yes").mean()
    blip_yr = (subset["blip_answer"].str.lower() == "yes").mean()
    vilt_yr = (subset["vilt_answer"].str.lower() == "yes").mean()
    blip_acc = subset[BLIP_CORR].mean()
    vilt_acc = subset[VILT_CORR].mean()
    yes_sub = subset[subset["gt_answer"].str.lower() == "yes"]
    no_sub  = subset[subset["gt_answer"].str.lower() == "no"]
    blip_ay = yes_sub[BLIP_CORR].mean()
    blip_an = no_sub[BLIP_CORR].mean()
    vilt_ay = yes_sub[VILT_CORR].mean()
    vilt_an = no_sub[VILT_CORR].mean()
    lines.append(f"\n{struct} (n={n:,}):")
    lines.append(f"  GT yes-rate:   {gt_yr:.1%}")
    lines.append(f"  BLIP yes-rate: {blip_yr:.1%}  (Δ from GT: {blip_yr-gt_yr:+.1%})")
    lines.append(f"  ViLT yes-rate: {vilt_yr:.1%}  (Δ from GT: {vilt_yr-gt_yr:+.1%})")
    lines.append(f"  BLIP acc overall: {blip_acc:.1%}  acc_yes: {blip_ay:.1%}  acc_no: {blip_an:.1%}  gap: {blip_ay-blip_an:+.1%}")
    lines.append(f"  ViLT acc overall: {vilt_acc:.1%}  acc_yes: {vilt_ay:.1%}  acc_no: {vilt_an:.1%}  gap: {vilt_ay-vilt_an:+.1%}")

text = "\n".join(lines) + "\n"
print(text)
(OUT_DIR / "summary.txt").write_text(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

cell_labels = results["cell"].tolist()
x = np.arange(len(cell_labels))
w = 0.25  # bar width

# ── Figure 1: Prediction yes-rate vs GT ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w,   results["gt_yes_rate"],   width=w, label="GT",   color=GT_COLOR,   alpha=0.85)
ax.bar(x,       results["blip_yes_rate"], width=w, label="BLIP", color=BLIP_COLOR, alpha=0.85)
ax.bar(x + w,   results["vilt_yes_rate"], width=w, label="ViLT", color=VILT_COLOR, alpha=0.85)
ax.axhline(0.5, ls="--", lw=1, color="gray", label="50% baseline")
ax.set_xticks(x)
ax.set_xticklabels(cell_labels, rotation=30, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylabel("Yes-prediction rate")
ax.set_title("Experiment 4A: Yes-Rate — GT vs. BLIP vs. ViLT\n(binary questions: verify + logical)")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(OUT_DIR / "prediction_yesrate.png", dpi=150)
plt.close()
print("Saved prediction_yesrate.png")

# ── Figure 2: Accuracy on yes-GT vs no-GT ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (acc_yes_col, acc_no_col, c, title) in zip(axes, [
    ("blip_acc_yes", "blip_acc_no", BLIP_COLOR, "BLIP"),
    ("vilt_acc_yes", "vilt_acc_no", VILT_COLOR, "ViLT"),
]):
    acc_yes = results[acc_yes_col].values
    acc_no  = results[acc_no_col].values
    ax.bar(x - w/2, acc_yes, width=w, label="Acc on yes-GT", color=c,  alpha=0.85)
    ax.bar(x + w/2, acc_no,  width=w, label="Acc on no-GT",  color=c,  alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(f"{title}: Accuracy by GT Answer Direction")
    ax.set_ylabel("Accuracy (normalized)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    for i, (ay, an) in enumerate(zip(acc_yes, acc_no)):
        if not (np.isnan(ay) or np.isnan(an)):
            gap = float(ay) - float(an)
            ypos = max(float(ay), float(an)) + 0.02
            ax.text(i, min(ypos, 0.95), f"{gap:+.0%}", ha="center", fontsize=7.5,
                    color="red" if abs(gap) > 0.1 else "dimgray")

plt.suptitle("Experiment 4B: Asymmetric Accuracy on Yes vs. No Questions", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "asymmetric_accuracy.png", dpi=150)
plt.close()
print("Saved asymmetric_accuracy.png")

# ── Figure 3: Bias gap diverging bar chart ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (bias_col, c, title) in zip(axes, [
    ("blip_bias_gap", BLIP_COLOR, "BLIP"),
    ("vilt_bias_gap", VILT_COLOR, "ViLT"),
]):
    gaps = results[bias_col].values
    colors_bar = ["firebrick" if g > 0.1 else "steelblue" if g < -0.1 else "gray"
                  for g in gaps]
    ax.barh(cell_labels, gaps, color=colors_bar, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline( 0.1, color="gray", lw=0.7, ls="--")
    ax.axvline(-0.1, color="gray", lw=0.7, ls="--")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(f"{title}: Bias Gap\n(acc_yes − acc_no)")
    ax.set_xlabel("Accuracy gap (positive = yes-biased)")
    for i, g in enumerate(gaps):
        if not np.isnan(g):
            ax.text(g + (0.005 if g >= 0 else -0.005), i,
                    f"{g:+.1%}", va="center",
                    ha="left" if g >= 0 else "right", fontsize=8)

plt.suptitle("Experiment 4C: Yes/No Bias Gap per Cell", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "bias_gap.png", dpi=150)
plt.close()
print("Saved bias_gap.png")

print(f"\nAll outputs saved to {OUT_DIR}")
