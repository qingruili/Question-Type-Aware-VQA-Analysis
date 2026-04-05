#!/usr/bin/env python3
"""
Experiment: Target Object Size vs Accuracy

Hypothesis: questions about small objects are harder because BLIP's ViT
patch encoder has limited resolution for small regions.

Method:
  - Parse the `select` operation in each question's semanticStr to get
    the target object's scene-graph ID.
  - Look up the object's bounding box (x, y, w, h) in val_sceneGraphs.json.
  - Compute relative area = (w * h) / (img_W * img_H).
  - Bin into terciles: small / medium / large.
  - Analyse accuracy by: size × structural, size × semantic, size × attr_type,
    size × program_depth.

Output: results/analysis/object_size/
"""
import re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_CSV    = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV     = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
QUES_JSON    = PROJECT_ROOT / "data" / "questions1.2" / "val_balanced_questions.json"
SG_JSON      = PROJECT_ROOT / "data" / "sceneGraphs" / "val_sceneGraphs.json"
OUT_DIR      = PROJECT_ROOT / "results" / "analysis" / "object_size"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helper ────────────────────────────────────────────────────────────────────
def parse_select_id(sem_str):
    """Return the scene-graph object ID from a 'select: <name> (<id>)' token."""
    m = re.search(r'select:\s*\S+\s*\((\d+)\)', str(sem_str))
    return m.group(1) if m else None

CANONICAL = {
    "color":"color","colour":"color",
    "hposition":"position","vposition":"position",
    "material":"material","size":"size","activity":"activity",
    "pose":"pose","shape":"shape","length":"length","height":"height",
    "weather":"weather","sportActivity":"sport",
}
def canon(t):
    if pd.isna(t): return "other"
    return CANONICAL.get(str(t).strip(), "other")

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading CSV data …")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("","_f"))
df["correct"] = df["blip_correct_norm"].astype(bool)
df["canon_type"] = df["attr_type"].apply(canon)

print("Loading questions JSON …")
with open(QUES_JSON) as f:
    qs = json.load(f)

print("Loading scene graphs …")
with open(SG_JSON) as f:
    sg = json.load(f)

# ── extract target object bounding box ────────────────────────────────────────
print("Extracting object bounding boxes …")
records = []
for _, row in df.iterrows():
    qid  = row["qid"]
    q    = qs.get(qid)
    if q is None:
        records.append(None)
        continue
    obj_id = parse_select_id(q.get("semanticStr", ""))
    if obj_id is None:
        records.append(None)
        continue
    img_key = str(row["image_id"])
    graph   = sg.get(img_key)
    if graph is None:
        records.append(None)
        continue
    obj = graph["objects"].get(obj_id)
    if obj is None:
        records.append(None)
        continue
    img_W = graph.get("width",  500)
    img_H = graph.get("height", 333)
    obj_w = max(obj.get("w", 0), 1)
    obj_h = max(obj.get("h", 0), 1)
    rel_area = (obj_w * obj_h) / (img_W * img_H)
    records.append(rel_area)

df["rel_obj_area"] = records
covered = df["rel_obj_area"].notna().sum()
print(f"  Coverage: {covered}/{len(df)} ({covered/len(df)*100:.1f}%)")

# save extended features
df[["qid","rel_obj_area"]].dropna().to_csv(
    OUT_DIR / "obj_area_per_question.csv", index=False)

df_valid = df.dropna(subset=["rel_obj_area"]).copy()

# Tercile thresholds
p33 = df_valid["rel_obj_area"].quantile(0.333)
p67 = df_valid["rel_obj_area"].quantile(0.667)
print(f"\nArea tercile thresholds: p33={p33:.4f}  p67={p67:.4f}")
print(f"  (small < {p33:.3f} | medium {p33:.3f}–{p67:.3f} | large > {p67:.3f})")

def size_bin(v):
    if v < p33: return "small"
    elif v < p67: return "medium"
    return "large"

df_valid["size_bin"] = df_valid["rel_obj_area"].apply(size_bin)
size_order = ["small", "medium", "large"]

print("\n" + "="*60)
print("(1) Accuracy by size bin — OVERALL")
print("="*60)
ov = df_valid.groupby("size_bin")["correct"].agg(["mean","count"]).reindex(size_order)
ov.columns = ["accuracy","n"]
print(ov.to_string())

print("\n" + "="*60)
print("(2) Accuracy by size × structural type")
print("="*60)
piv_struct = (
    df_valid.groupby(["structural","size_bin"])["correct"]
    .mean().unstack("size_bin").reindex(columns=size_order)
    .round(3)
)
print(piv_struct.to_string())

print("\n" + "="*60)
print("(3) Accuracy by size × semantic type")
print("="*60)
piv_sem = (
    df_valid.groupby(["semantic","size_bin"])["correct"]
    .mean().unstack("size_bin").reindex(columns=size_order)
    .round(3)
)
print(piv_sem.to_string())

print("\n" + "="*60)
print("(4) Accuracy by size × attr_type (query×attr only)")
print("="*60)
attr_sub = df_valid[(df_valid["structural"]=="query") & (df_valid["semantic"]=="attr")]
focus_types = ["color","position","material","size","activity","pose"]
piv_attr = (
    attr_sub[attr_sub["canon_type"].isin(focus_types)]
    .groupby(["canon_type","size_bin"])["correct"]
    .agg(["mean","count"]).unstack("size_bin")
    .round(3)
)
print(piv_attr.to_string())

print("\n" + "="*60)
print("(5) Accuracy by size × depth bins (2, 3, 4, ≥5)")
print("="*60)
df_valid["depth_bin"] = df_valid["program_depth"].apply(
    lambda d: str(d) if d <= 4 else "≥5"
)
depth_order = ["2","3","4","≥5"]
piv_depth = (
    df_valid.groupby(["depth_bin","size_bin"])["correct"]
    .mean().unstack("size_bin").reindex(index=depth_order, columns=size_order)
    .round(3)
)
print(piv_depth.to_string())

# ── quick regression: area → accuracy (logistic-like, using pearson) ──────────
from scipy.stats import pearsonr, spearmanr
r_p, p_p = pearsonr(df_valid["rel_obj_area"], df_valid["correct"].astype(float))
r_s, p_s = spearmanr(df_valid["rel_obj_area"], df_valid["correct"].astype(float))
print(f"\nCorrelation: Pearson r={r_p:.4f} p={p_p:.4g}  |  Spearman rho={r_s:.4f} p={p_s:.4g}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: accuracy vs size bin overall
ax = axes[0]
bar_df = df_valid.groupby("size_bin")["correct"].agg(["mean","count"]).reindex(size_order).reset_index()
bars = ax.bar(bar_df["size_bin"], bar_df["mean"], color=["#E74C3C","#F39C12","#2ECC71"])
for bar, (_, row) in zip(bars, bar_df.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Accuracy")
ax.set_title("Overall accuracy by object size")
ax.axhline(df_valid["correct"].mean(), color="black", linestyle="--", linewidth=1, label="overall")
ax.legend(fontsize=8)

# Plot 2: size × structural
ax = axes[1]
for struct, row in piv_struct.iterrows():
    ax.plot(size_order, row.values, marker="o", label=struct)
ax.set_ylim(0.3, 0.95)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by size × structural type")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: size × semantic
ax = axes[2]
for sem, row in piv_sem.iterrows():
    ax.plot(size_order, row.values, marker="o", label=sem)
ax.set_ylim(0.3, 0.95)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by size × semantic type")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "objsize_accuracy_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: objsize_accuracy_overview.png")

# Plot 4: heatmap — attr type × size bin
fig, ax = plt.subplots(figsize=(7, 5))
heat_data = (
    attr_sub[attr_sub["canon_type"].isin(focus_types)]
    .groupby(["canon_type","size_bin"])["correct"]
    .mean().unstack("size_bin").reindex(columns=size_order)
)
heat_n = (
    attr_sub[attr_sub["canon_type"].isin(focus_types)]
    .groupby(["canon_type","size_bin"])["correct"]
    .count().unstack("size_bin").reindex(columns=size_order)
)
annot = heat_data.round(3).astype(str) + "\n(n=" + heat_n.astype(str) + ")"
sns.heatmap(heat_data, annot=annot, fmt="", cmap="RdYlGn", vmin=0.2, vmax=0.8,
            ax=ax, linewidths=0.5)
ax.set_title("query×attr accuracy by attr type × object size")
ax.set_xlabel("object size bin")
ax.set_ylabel("attr type")
plt.tight_layout()
plt.savefig(OUT_DIR / "objsize_attr_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: objsize_attr_heatmap.png")

# ── save summary table ────────────────────────────────────────────────────────
summary = {
    "overall_by_size": ov,
    "structural_x_size": piv_struct,
    "semantic_x_size": piv_sem,
    "depth_x_size": piv_depth,
}
with open(OUT_DIR / "objsize_summary.txt", "w") as f:
    f.write(f"Target Object Size vs Accuracy\n")
    f.write(f"Coverage: {covered}/{len(df)} ({covered/len(df)*100:.1f}%)\n")
    f.write(f"Tercile thresholds: p33={p33:.4f}  p67={p67:.4f}\n\n")
    f.write(f"Correlation: Pearson r={r_p:.4f} p={p_p:.4g}  |  Spearman rho={r_s:.4f} p={p_s:.4g}\n\n")
    for name, tbl in summary.items():
        f.write(f"--- {name} ---\n{tbl.to_string()}\n\n")

print(f"\nAll outputs in: {OUT_DIR}")
