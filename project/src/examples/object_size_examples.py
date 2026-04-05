#!/usr/bin/env python3
"""
Object Size Example Visualizations (Folder E)

Generates individual images showing the target object's bounding box,
illustrating how object size affects BLIP's accuracy.

Three sub-sets:
  E_small_wrong/   — small objects where BLIP fails (with bbox drawn)
  E_large_correct/ — same attr type, large objects where BLIP succeeds
  E_size_contrast/ — side-by-side JSON info for matched small-wrong vs large-correct pairs
                     (each pair saved as two separate PNGs named pair_XX_a/b)

The bounding box is drawn directly onto the image before display so the
queried region is clearly visible.

Usage: python3 src/examples/object_size_examples.py
"""
import re, json, io, sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))

STATS_CSV   = PROJECT_ROOT / "results" / "per_question_stats.csv"
FEAT_CSV    = PROJECT_ROOT / "results" / "analysis" / "question_features" / "question_features.csv"
AREA_CSV    = PROJECT_ROOT / "results" / "analysis" / "object_size" / "obj_area_per_question.csv"
QUES_JSON   = PROJECT_ROOT / "data" / "questions1.2" / "val_balanced_questions.json"
SG_JSON     = PROJECT_ROOT / "data" / "sceneGraphs" / "val_sceneGraphs.json"
IMAGES_ZIP  = PROJECT_ROOT / "data" / "images.zip"
OUT_BASE    = PROJECT_ROOT / "results" / "analysis" / "qualitative_examples" / "E_object_size"

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

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r'^(a |an |the )', '', s)
    return s.strip('.,!?;:"\'()-')

def parse_select_id(sem_str):
    m = re.search(r'select:\s*\S+\s*\((\d+)\)', str(sem_str))
    return m.group(1) if m else None

def _wrap(text, max_chars=52):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return "\n".join(lines)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading CSV data …")
stats = pd.read_csv(STATS_CSV, dtype={"qid": str})
feats = pd.read_csv(FEAT_CSV,  dtype={"question_id": str})
df = stats.merge(feats, left_on="qid", right_on="question_id", suffixes=("","_f"))
df["correct"]    = df["blip_correct_norm"].astype(bool)
df["canon_type"] = df["attr_type"].apply(canon)
area_df = pd.read_csv(AREA_CSV, dtype={"qid": str})
df = df.merge(area_df, on="qid", how="left")

p33 = df["rel_obj_area"].quantile(0.333)
p67 = df["rel_obj_area"].quantile(0.667)
df["size_bin"] = df["rel_obj_area"].apply(
    lambda v: "small" if v < p33 else ("large" if v >= p67 else "medium")
    if pd.notna(v) else None
)

print("Loading questions JSON …")
with open(QUES_JSON) as f:
    qs = json.load(f)

print("Loading scene graphs …")
with open(SG_JSON) as f:
    sg = json.load(f)

print("Opening images zip …")
_zip = zipfile.ZipFile(IMAGES_ZIP, "r")

def load_image_raw(image_id):
    path = f"images/{image_id}.jpg"
    try:
        data = _zip.read(path)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except KeyError:
        return None

def get_bbox(qid, image_id):
    """Return (x, y, w, h) bounding box of the select object, or None."""
    q = qs.get(str(qid))
    if q is None: return None
    obj_id = parse_select_id(q.get("semanticStr",""))
    if obj_id is None: return None
    graph = sg.get(str(image_id))
    if graph is None: return None
    obj = graph["objects"].get(obj_id)
    if obj is None: return None
    return obj.get("x"), obj.get("y"), obj.get("w"), obj.get("h")

def draw_bbox_on_image(pil_img, bbox, color="#FF4444", width=4):
    """Draw a rectangle on the image; return new PIL image."""
    if bbox is None or any(v is None for v in bbox):
        return pil_img
    x, y, w, h = bbox
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for i in range(width):
        draw.rectangle([x-i, y-i, x+w+i, y+h+i], outline=color)
    return img

def save_single(row_data, out_path, draw_box=True):
    """Save one example image with optional bounding box."""
    img_id  = row_data["image_id"]
    correct = bool(row_data.get("correct", False))
    border  = "#2ECC71" if correct else "#E74C3C"
    mark    = "✓" if correct else "✗"

    pil_img = load_image_raw(img_id)
    if pil_img and draw_box:
        bbox = get_bbox(row_data["qid"], img_id)
        pil_img = draw_bbox_on_image(pil_img, bbox,
                                      color="#FF4444" if not correct else "#2ECC71")

    fig, ax = plt.subplots(figsize=(5, 4.8))
    if pil_img is not None:
        ax.imshow(pil_img)
    else:
        ax.set_facecolor("#dddddd")
        ax.text(0.5, 0.5, "image not found", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(3)

    q_text = _wrap(str(row_data.get("question", "")), 52)
    label  = row_data.get("label", "")
    area_pct = row_data.get("rel_obj_area", float("nan"))
    area_str = f"obj area={area_pct*100:.1f}% of image" if pd.notna(area_pct) else ""
    caption = f"[{label}]  {area_str}\nQ: {q_text}\nGT: {row_data.get('gt_answer','')}    BLIP: {row_data.get('blip_answer','')} {mark}"
    ax.set_xlabel(caption, fontsize=7.5, loc="left", color="black", labelpad=5)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def pick_diverse(sub_df, n, filter_fn=None):
    seen, result = set(), []
    for _, r in sub_df.iterrows():
        if filter_fn and filter_fn(r): continue
        if r["image_id"] not in seen:
            result.append(r)
            seen.add(r["image_id"])
            if len(result) >= n: break
    return result

def is_annotation_artifact(r):
    gt   = norm(str(r.get("gt_answer", "")))
    pred = norm(str(r.get("blip_answer", "")))
    if not gt or not pred: return True
    if gt.endswith(" " + pred) or pred.endswith(" " + gt): return True
    if pred in gt or gt in pred: return True
    SYNONYMS = {frozenset(["gray","grey"]), frozenset(["blond","blonde"]),
                frozenset(["lying","laying"]), frozenset(["sofa","couch"]),
                frozenset(["bookshelf","bookcase"])}
    return frozenset([gt, pred]) in SYNONYMS

# ── sub-set 1: small objects BLIP gets wrong ──────────────────────────────────
print("\n" + "="*60)
print("E1) Small objects — BLIP wrong (bbox drawn)")
print("="*60)

out_small = OUT_BASE / "E1_small_wrong"
out_small.mkdir(parents=True, exist_ok=True)

# Focus on query questions with a clear attr type (not 'other')
# where size_bin=small and BLIP is wrong
small_wrong = df[
    (df["size_bin"] == "small") &
    (~df["correct"]) &
    (df["structural"] == "query") &
    (df["semantic"] == "attr") &
    (~df["canon_type"].isin(["other"])) &
    df["blip_answer"].notna() &
    (df["blip_answer"].astype(str).str.strip() != "")
].copy()
small_wrong = small_wrong[~small_wrong.apply(is_annotation_artifact, axis=1)]
small_wrong = small_wrong.sort_values("rel_obj_area")  # smallest first

TARGET_TYPES = ["color","material","size","activity","pose","shape"]
count = 0
for ct in TARGET_TYPES:
    sub = small_wrong[small_wrong["canon_type"] == ct]
    picked = pick_diverse(sub, 5)
    for r in picked:
        fname = out_small / f"E1_{count:02d}_{r['image_id']}_wrong.png"
        save_single(r, fname, draw_box=True)
        count += 1
    print(f"  {ct}: {len(picked)} examples (area range {sub['rel_obj_area'].min():.4f}–{sub['rel_obj_area'].quantile(0.1):.4f})")

print(f"  → {count} images in {out_small}")

# ── sub-set 2: large objects BLIP gets right ──────────────────────────────────
print("\n" + "="*60)
print("E2) Large objects — BLIP correct (bbox drawn)")
print("="*60)

out_large = OUT_BASE / "E2_large_correct"
out_large.mkdir(parents=True, exist_ok=True)

large_correct = df[
    (df["size_bin"] == "large") &
    (df["correct"]) &
    (df["structural"] == "query") &
    (df["semantic"] == "attr") &
    (~df["canon_type"].isin(["other"]))
].copy()

count = 0
for ct in TARGET_TYPES:
    sub = large_correct[large_correct["canon_type"] == ct]
    sub = sub.sort_values("rel_obj_area", ascending=False)  # largest first
    picked = pick_diverse(sub, 5)
    for r in picked:
        fname = out_large / f"E2_{count:02d}_{r['image_id']}_ok.png"
        save_single(r, fname, draw_box=True)
        count += 1
    print(f"  {ct}: {len(picked)} examples")

print(f"  → {count} images in {out_large}")

# ── sub-set 3: matched contrast pairs (same attr type) ────────────────────────
print("\n" + "="*60)
print("E3) Contrast pairs: small-wrong vs large-correct (same attr type)")
print("="*60)

out_contrast = OUT_BASE / "E3_contrast_pairs"
out_contrast.mkdir(parents=True, exist_ok=True)

pair_count = 0
for ct in ["color", "material", "size", "activity"]:
    sw = small_wrong[small_wrong["canon_type"] == ct].copy()
    lc = large_correct[large_correct["canon_type"] == ct].copy()
    sw = sw.sort_values("rel_obj_area")
    lc = lc.sort_values("rel_obj_area", ascending=False)
    n_pairs = min(len(sw), len(lc), 4)
    sw_picked = pick_diverse(sw, n_pairs)
    lc_picked = pick_diverse(lc, n_pairs)
    for i, (s_row, l_row) in enumerate(zip(sw_picked, lc_picked)):
        fa = out_contrast / f"E3_pair{pair_count:02d}a_{s_row['image_id']}_small_wrong.png"
        fb = out_contrast / f"E3_pair{pair_count:02d}b_{l_row['image_id']}_large_ok.png"
        save_single(s_row, fa, draw_box=True)
        save_single(l_row, fb, draw_box=True)
        pair_count += 1
    print(f"  {ct}: {min(n_pairs, pair_count)} pairs")

print(f"  → {pair_count*2} images ({pair_count} pairs) in {out_contrast}")

print(f"\nAll E examples in: {OUT_BASE}")
