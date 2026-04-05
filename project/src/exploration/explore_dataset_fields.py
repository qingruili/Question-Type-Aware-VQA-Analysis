#!/usr/bin/env python3
"""
src/exploration/explore_dataset_fields.py

A comprehensive field guide to the GQA dataset files.

Questions answered:
  - What fields exist in val_balanced_questions.json and what do they mean?
  - What does the scene graph provide per image?
  - How do the 'annotations' in questions link to scene graph objects?
  - What are 'entailed' and 'equivalent' questions (useful for dedup/consistency)?
  - What is the 'groups' field (global/local semantic categories)?
  - What are the 102 'detailed' subtypes beyond the 5×5 matrix?
  - Scene graph statistics: objects per image, attributes, relations, bbox sizes.

Outputs (in results/exploration/dataset_fields/):
  figures/
    scene_graph_stats.png         — distributions of objects/attrs/relations per image
    detailed_type_distribution.png — top-30 detailed subtypes
    global_group_distribution.png  — global group counts
    entailed_equivalent_dist.png   — cluster size distributions
  field_guide.txt                   — annotated field-by-field reference
  detailed_types.csv                — all 102 detailed subtypes with counts
  scene_graph_summary.csv           — per-image scene graph statistics
"""

import json
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
Q_PATH       = PROJECT_ROOT / "data" / "questions1.2" / "val_balanced_questions.json"
SG_PATH      = PROJECT_ROOT / "data" / "sceneGraphs" / "val_sceneGraphs.json"
OUT_DIR      = PROJECT_ROOT / "results" / "exploration" / "dataset_fields"
FIG_DIR      = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURAL = ["query", "verify", "logical", "choose", "compare"]
SEMANTIC   = ["rel", "attr", "obj", "cat", "global"]


def load_data():
    print("Loading val_balanced_questions.json …")
    with open(Q_PATH) as f:
        questions = json.load(f)
    print(f"  {len(questions):,} questions")

    print("Loading val_sceneGraphs.json …")
    with open(SG_PATH) as f:
        scene_graphs = json.load(f)
    print(f"  {len(scene_graphs):,} scene graphs")
    return questions, scene_graphs


# ── Field guide text ──────────────────────────────────────────────────────────
def write_field_guide(questions: dict, scene_graphs: dict):
    sample_q = list(questions.values())[5]   # pick an interesting one
    sample_sg_id = sample_q["imageId"]
    sample_sg = scene_graphs.get(sample_sg_id, {})

    lines = []
    lines.append("=" * 80)
    lines.append("GQA DATASET FIELD GUIDE")
    lines.append("=" * 80)

    # ── Questions file ──
    lines.append("\n\n── val_balanced_questions.json ────────────────────────────────────────")
    lines.append("Structure: { question_id: { ...fields... }, ... }")
    lines.append(f"Total questions: {len(questions):,}")
    lines.append(f"Keys per question: {list(sample_q.keys())}")

    fields = [
        ("question",    "str",  "The natural language question text."),
        ("answer",      "str",  "The single ground-truth answer string (always 1 GT). "
                                "Short: usually 1 word. Models must match this exactly."),
        ("fullAnswer",  "str",  "A full sentence form of the answer (e.g., 'The cat is white.'). "
                                "Not used for evaluation — informational only."),
        ("imageId",     "str",  "The GQA image ID (matches filename in data/images/ and key in "
                                "val_sceneGraphs.json)."),
        ("isBalanced",  "bool", "Always True in the balanced split (False entries were filtered out). "
                                "Balanced = answer distribution is artificially equalized."),
        ("types",       "dict", "Three classification axes:\n"
                                "  structural (str): reasoning form — query/verify/logical/choose/compare\n"
                                "  semantic   (str): visual target  — rel/attr/obj/cat/global\n"
                                "  detailed   (str): 102 fine-grained subtypes (e.g. 'existRelS', "
                                "'colorQuery')"),
        ("semantic",    "list", "The functional program: a list of reasoning steps (the 'program').\n"
                                "  Each step: { operation: str, argument: str, dependencies: [int] }\n"
                                "  'operation': the reasoning op (select, relate, query, filter color, ...)\n"
                                "  'argument': the target value/attribute (e.g., 'color', 'cat (329774)')\n"
                                "  'dependencies': indices of prior steps this step depends on\n"
                                "  len(semantic) = program depth = primary complexity measure."),
        ("semanticStr", "str",  "Human-readable version of the semantic program (arrow-separated). "
                                "Redundant with 'semantic' but easier to read."),
        ("annotations", "dict", "Maps question text / answer / fullAnswer to the scene graph object "
                                "IDs they reference.\n"
                                "  Structure: { 'answer': {token_pos: sg_obj_id}, "
                                "'question': {token_pos: sg_obj_id}, 'fullAnswer': {...} }\n"
                                "  token_pos is the word index in the question/answer string.\n"
                                "  sg_obj_id can be looked up in val_sceneGraphs.json.\n"
                                "  NOTE: many questions have empty annotations (no grounding)."),
        ("groups",      "dict", "Semantic group labels:\n"
                                "  global (str|None): high-level category (e.g. 'color', 'person', "
                                "'animal', 'furniture') — 112 values, None for ~48% of questions\n"
                                "  local  (str):      fine-grained template group (e.g. '10q-helmet_color')"),
        ("entailed",    "list", "Question IDs that are logically entailed by this question "
                                "(if this Q is true, those Qs are also true). "
                                "Useful for measuring logical consistency of model predictions."),
        ("equivalent",  "list", "Question IDs that ask the same thing (same answer, similar program). "
                                "Includes this question's own ID. "
                                "Useful for measuring surface-form robustness."),
    ]

    for name, ftype, desc in fields:
        lines.append(f"\n  [{ftype}] {name}")
        for line in desc.split("\n"):
            lines.append(f"         {line}")

    # ── Sample question ──
    lines.append(f"\n\nSAMPLE QUESTION (id: {list(questions.keys())[5]}):")
    import pprint
    lines.append(pprint.pformat(sample_q, width=80))

    # ── Scene graph file ──
    lines.append("\n\n── val_sceneGraphs.json ───────────────────────────────────────────────")
    lines.append("Structure: { image_id: { width, height, objects: { obj_id: {...} } } }")
    lines.append(f"Total images: {len(scene_graphs):,}")

    sg_fields = [
        ("width / height", "int", "Image dimensions in pixels."),
        ("objects",        "dict", "Dict of {object_id: object_record}. Object IDs match those in "
                                    "question annotations."),
        ("  name",         "str",  "Object category label (e.g., 'person', 'car', 'helmet')."),
        ("  x, y, w, h",   "int",  "Bounding box: top-left x/y, width, height in pixels."),
        ("  attributes",   "list", "List of attribute strings (e.g., ['blue', 'large']). "
                                    "Can be empty."),
        ("  relations",    "list", "List of {name: str, object: obj_id} dicts. "
                                    "Directed edges to other objects. "
                                    "name = spatial/functional relation (e.g., 'to the left of', "
                                    "'wearing', 'of')."),
    ]

    for name, ftype, desc in sg_fields:
        lines.append(f"\n  [{ftype}] {name}")
        for line in desc.split("\n"):
            lines.append(f"         {line}")

    # ── Sample scene graph ──
    if sample_sg:
        lines.append(f"\n\nSAMPLE SCENE GRAPH (image_id: {sample_sg_id}, "
                     f"linked to question above):")
        # truncate to first 2 objects for readability
        truncated = dict(list(sample_sg.get("objects", {}).items())[:2])
        sample_sg_trunc = {**sample_sg, "objects": truncated,
                           "...": f"({len(sample_sg.get('objects', {}))} objects total)"}
        lines.append(pprint.pformat(sample_sg_trunc, width=80))

    # ── annotations linkage example ──
    lines.append("\n\n── HOW annotations LINKS QUESTION TO SCENE GRAPH ──────────────────────")
    lines.append(
        "annotations.question maps word-position → scene graph object ID.\n"
        "You can look up that object ID in val_sceneGraphs.json to get:\n"
        "  - the object's name, bounding box, attributes, and relations\n\n"
        "Example workflow:\n"
        "  q = questions['05515938']   # 'What is this bird called?'\n"
        "  ann = q['annotations']['question']   # {'3': '329774'} → word 3 = 'bird' → obj 329774\n"
        "  sg = scene_graphs[q['imageId']]      # scene graph for image 2405722\n"
        "  obj = sg['objects']['329774']         # name='parrot', x=..., attributes=[...]\n"
    )

    out_path = OUT_DIR / "field_guide.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ── Scene graph statistics ────────────────────────────────────────────────────
def compute_sg_stats(scene_graphs: dict) -> pd.DataFrame:
    rows = []
    for img_id, sg in scene_graphs.items():
        objects = sg.get("objects", {})
        n_obj   = len(objects)
        n_attrs = sum(len(o.get("attributes", [])) for o in objects.values())
        n_rels  = sum(len(o.get("relations", []))  for o in objects.values())
        rows.append({
            "image_id": img_id,
            "width": sg.get("width", 0), "height": sg.get("height", 0),
            "n_objects": n_obj,
            "n_attributes": n_attrs,
            "n_relations": n_rels,
            "attrs_per_obj": round(n_attrs / n_obj, 3) if n_obj else 0,
            "rels_per_obj":  round(n_rels  / n_obj, 3) if n_obj else 0,
        })
    return pd.DataFrame(rows)


def plot_sg_stats(sg_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    plots = [
        ("n_objects",    "Objects per image",     "steelblue"),
        ("n_attributes", "Attributes per image",  "darkorange"),
        ("n_relations",  "Relations per image",   "seagreen"),
        ("attrs_per_obj","Attributes per object", "mediumpurple"),
        ("rels_per_obj", "Relations per object",  "crimson"),
    ]

    for ax, (col, title, color) in zip(axes, plots):
        vals = sg_df[col].values
        ax.hist(vals, bins=40, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("# images")
        stats = f"mean={vals.mean():.1f}  median={np.median(vals):.0f}  max={vals.max():.0f}"
        ax.set_title(f"{title}\n{stats}", fontsize=9)

    # Image dimensions scatter
    ax = axes[5]
    sample = sg_df.sample(min(2000, len(sg_df)), random_state=42)
    ax.scatter(sample["width"], sample["height"], alpha=0.2, s=5, color="slategray")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Image height (px)")
    ax.set_title(f"Image Dimensions (n={len(sg_df):,} images)\n"
                 f"mean: {sg_df['width'].mean():.0f}×{sg_df['height'].mean():.0f}", fontsize=9)

    fig.suptitle("Scene Graph Statistics (val split)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = FIG_DIR / "scene_graph_stats.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Detailed type distribution ────────────────────────────────────────────────
def plot_detailed_types(questions: dict):
    counter = Counter(q["types"]["detailed"] for q in questions.values())

    # save full CSV
    df = pd.DataFrame(counter.most_common(), columns=["detailed_type", "count"])
    df["structural"] = df["detailed_type"].apply(
        lambda t: next((q["types"]["structural"] for q in questions.values()
                        if q["types"]["detailed"] == t), ""))
    df.to_csv(OUT_DIR / "detailed_types.csv", index=False)
    print(f"  Saved: {(OUT_DIR / 'detailed_types.csv').relative_to(PROJECT_ROOT)}")
    print(f"  Total detailed types: {len(counter)}")

    # plot top-30
    top30 = counter.most_common(30)
    labels, counts = zip(*top30)

    # color by structural type
    struct_colors = {"query": "#4e9af1", "verify": "#3ec97c", "logical": "#f5a623",
                     "choose": "#e74c3c", "compare": "#9b59b6"}
    colors = []
    for label in labels:
        st = next((q["types"]["structural"] for q in questions.values()
                   if q["types"]["detailed"] == label), "query")
        colors.append(struct_colors.get(st, "gray"))

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top-30 Detailed Question Subtypes (colored by structural type)", fontsize=11)

    # legend
    for st, c in struct_colors.items():
        ax.bar(0, 0, color=c, label=st)
    ax.legend(title="Structural", loc="lower right", fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "detailed_type_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Global group distribution ─────────────────────────────────────────────────
def plot_global_groups(questions: dict):
    counter = Counter(q["groups"]["global"] for q in questions.values())
    none_count = counter.pop(None, 0)

    top20 = counter.most_common(20)
    labels, counts = zip(*top20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(labels)), counts, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(
        f"Top-20 Global Semantic Groups (q['groups']['global'])\n"
        f"(None = {none_count:,} questions have no global group)", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "global_group_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Entailed/equivalent distributions ─────────────────────────────────────────
def plot_entailed_equivalent(questions: dict):
    entailed_counts   = [len(q["entailed"])   for q in questions.values()]
    equivalent_counts = [len(q["equivalent"]) for q in questions.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, counts, title, color, note in [
        (axes[0], entailed_counts,   "Entailed cluster size",  "steelblue",
         "Q's whose truth is implied by this Q"),
        (axes[1], equivalent_counts, "Equivalent cluster size", "darkorange",
         "Q's asking the same thing (rephrased)"),
    ]:
        counter = Counter(counts)
        xs = sorted(counter)
        ys = [counter[x] for x in xs]
        ax.bar([str(x) for x in xs[:20]], ys[:20], color=color, edgecolor="white")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("# questions")
        pct_zero = 100 * counter.get(0, 0) / len(counts)
        ax.set_title(f"{title}\n({note})\n"
                     f"mean={np.mean(counts):.1f}, {pct_zero:.0f}% have none", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "entailed_equivalent_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ── Print a comprehensive console summary ────────────────────────────────────
def print_summary(questions: dict, sg_df: pd.DataFrame):
    print("\n── DATASET SUMMARY ────────────────────────────────────────────────")
    print(f"  Total questions:       {len(questions):,}")
    print(f"  Unique images:         {len(set(q['imageId'] for q in questions.values())):,}")
    print(f"  Unique answers:        {len(set(q['answer'] for q in questions.values())):,}")
    print(f"  Unique detailed types: {len(set(q['types']['detailed'] for q in questions.values()))}")
    print(f"  Unique global groups:  "
          f"{len(set(q['groups']['global'] for q in questions.values()))}")

    print("\n── SCENE GRAPH SUMMARY ────────────────────────────────────────────")
    print(f"  Images with scene graphs:  {len(sg_df):,}")
    print(f"  Avg objects per image:     {sg_df['n_objects'].mean():.1f}  "
          f"(range {sg_df['n_objects'].min()}–{sg_df['n_objects'].max()})")
    print(f"  Avg attributes per image:  {sg_df['n_attributes'].mean():.1f}")
    print(f"  Avg relations per image:   {sg_df['n_relations'].mean():.1f}")
    print(f"  Avg attrs per object:      {sg_df['attrs_per_obj'].mean():.2f}")
    print(f"  Avg rels per object:       {sg_df['rels_per_obj'].mean():.2f}")
    print(f"  Typical image size:        "
          f"{sg_df['width'].median():.0f}×{sg_df['height'].median():.0f} px (median)")

    print("\n── ANNOTATIONS COVERAGE ───────────────────────────────────────────")
    has_q_ann   = sum(1 for q in questions.values() if q["annotations"]["question"])
    has_ans_ann = sum(1 for q in questions.values() if q["annotations"]["answer"])
    print(f"  Questions with question annotations: {has_q_ann:,} "
          f"({100*has_q_ann/len(questions):.1f}%)")
    print(f"  Questions with answer annotations:   {has_ans_ann:,} "
          f"({100*has_ans_ann/len(questions):.1f}%)")

    print("\n── ENTAILED / EQUIVALENT ──────────────────────────────────────────")
    has_entailed   = sum(1 for q in questions.values() if q["entailed"])
    has_equivalent = sum(1 for q in questions.values()
                         if len(q["equivalent"]) > 1)   # >1 because includes self
    print(f"  Questions with ≥1 entailed question:    {has_entailed:,} "
          f"({100*has_entailed/len(questions):.1f}%)")
    print(f"  Questions with ≥1 equivalent question:  {has_equivalent:,} "
          f"({100*has_equivalent/len(questions):.1f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    questions, scene_graphs = load_data()

    print("\n── Step 1: Write field guide ───────────────────────────────────────")
    write_field_guide(questions, scene_graphs)

    print("\n── Step 2: Scene graph statistics ─────────────────────────────────")
    sg_df = compute_sg_stats(scene_graphs)
    sg_df.to_csv(OUT_DIR / "scene_graph_summary.csv", index=False)
    print(f"  Saved: {(OUT_DIR / 'scene_graph_summary.csv').relative_to(PROJECT_ROOT)}")
    plot_sg_stats(sg_df)

    print("\n── Step 3: Detailed subtype distribution ───────────────────────────")
    plot_detailed_types(questions)

    print("\n── Step 4: Global group distribution ──────────────────────────────")
    plot_global_groups(questions)

    print("\n── Step 5: Entailed / equivalent distributions ─────────────────────")
    plot_entailed_equivalent(questions)

    print("\n── Step 6: Console summary ─────────────────────────────────────────")
    print_summary(questions, sg_df)

    print("\n✓ Done. Outputs in results/exploration/dataset_fields/")


if __name__ == "__main__":
    main()
