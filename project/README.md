# Question-Type-Aware VQA Analysis

**What Drives BLIP's Accuracy on GQA? A Question-Type-Aware Analysis**

---

## Project Overview

This project conducts a fine-grained, question-type-stratified analysis of BLIP
(`Salesforce/blip-vqa-base`) on the GQA balanced validation split (132,062 questions).
Using GQA's two-axis taxonomy — **structural type** (query, verify, logical, choose,
compare) × **semantic type** (rel, attr, obj, cat, global) — we analyse BLIP across all
15 populated cells and identify three orthogonal accuracy drivers:

1. **Answer format** — verify outperforms query by up to +23.9 pp at the same reasoning depth.
2. **Reasoning demand** — relational/spatial questions lag attribute questions by 7.8 pp even after controlling for answer entropy.
3. **Visual grounding quality** — perceptual attributes (color, material) degrade sharply with small objects; spatial position does not, revealing a conceptual bottleneck.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/                                  ← GQA dataset (not tracked, see Setup)
│   ├── questions1.2/
│   │   └── val_balanced_questions.json
│   ├── images/
│   └── sceneGraphs/
├── src/
│   ├── inference/
│   │   └── run_inference.py               ← Step 1: run BLIP on the full dataset
│   ├── analysis/                          ← all analyses (paper + exploratory)
│   │   ├── build_question_features.py     ← Step 2: extract auxiliary features
│   │   ├── analyze_results.py
│   │   ├── relation_type.py
│   │   ├── answer_entropy.py
│   │   ├── attribute_type.py
│   │   ├── format_depth_analysis.py
│   │   ├── yesno_balance.py
│   │   ├── object_size.py
│   │   ├── error_typology.py
│   │   ├── scene_complexity.py
│   │   ├── answer_frequency.py
│   │   ├── answer_regime.py
│   │   ├── depth_accuracy.py
│   │   ├── vilt_vocab_coverage.py
│   │   ├── answer_mismatch.py
│   │   ├── dataset_structure.py
│   │   ├── relation_type_examples.py
│   │   └── viz_examples.py                ← shared image rendering utility
│   ├── examples/                          ← qualitative figure generation
│   │   ├── qualitative_examples.py
│   │   └── object_size_examples.py
│   └── exploration/                       ← early dataset exploration
│       ├── explore_answer_structure.py
│       ├── explore_dataset_fields.py
│       └── explore_depth_per_category.py
└── results/
    ├── per_question_stats.csv             ← main inference output [gitignored]
    ├── predictions/                       ← raw model outputs [gitignored]
    ├── normalized/                        ← 5×5 accuracy (report metric)
    ├── normalized_OOV_excluded/           ← 5×5 accuracy excluding ViLT OOV answers
    ├── strict/                            ← 5×5 accuracy with strict string match
    └── analysis/
        ├── question_features/             ← auxiliary features [gitignored]
        ├── blip_vilt_comparison/
        ├── relation_type/
        ├── answer_entropy/
        ├── attribute_type/
        ├── format_depth_analysis/
        ├── yesno_balance/
        ├── object_size/
        ├── error_typology/
        ├── scene_complexity/
        ├── answer_frequency/
        ├── answer_regime/
        ├── depth_accuracy/
        ├── vilt_vocab_coverage/
        ├── answer_mismatch/
        ├── dataset_structure/
        └── qualitative_examples/          ← annotated images [gitignored, 420 MB]
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Models** are loaded from HuggingFace on first run:
- `Salesforce/blip-vqa-base`
- `dandelin/vilt-b32-finetuned-vqa` (preliminary comparison only)

**Data**: Download the GQA dataset from https://cs.stanford.edu/people/dorarad/gqa/download.html
and place files under `data/` as shown above.

---

## Evaluation metric variants

Three variants of the 5×5 accuracy tables are stored under `results/`:

| Folder | Metric |
|---|---|
| `results/strict/` | Raw exact string match, no normalization |
| `results/normalized/` | Lowercase + number words → digits ← **used in the report** |
| `results/normalized_OOV_excluded/` | Same normalization; ViLT answers outside its 3,129-label vocabulary excluded from its denominator |

---

## All Experiments

Every script in `src/analysis/` and `src/examples/` is listed below. Scripts marked ✓ contributed directly to the report (table or figure number shown); others were exploratory.

### Core pipeline

| Script | Description | In report |
|---|---|---|
| `src/inference/run_inference.py` | Run BLIP (and ViLT) on all 132,062 GQA val questions; save per-question predictions | all results |
| `src/analysis/build_question_features.py` | Parse attribute subtypes from `semanticStr`, retrieve bbox areas from scene graphs, compute logic depth | all analyses |
| `src/analysis/analyze_results.py` | Compute 5×5 per-cell accuracy for BLIP and ViLT; render accuracy heatmaps | **Figure 1** |

### Analyses used in the report

| Script | Description | In report |
|---|---|---|
| `src/analysis/format_depth_analysis.py` | Format effect at controlled logic depth; choose mismatch outcome breakdown | **Table 2**, **Table 3** |
| `src/analysis/answer_entropy.py` | Accuracy by answer entropy bin for query questions; rel vs. attr at equal entropy | **Figure 3** |
| `src/analysis/relation_type.py` | Accuracy by relation subtype (spatial, action, comparative) for query×rel | **Table 4** |
| `src/analysis/attribute_type.py` | Accuracy by attribute subtype (color, material, position, …) for query×attr | **Table 5** |
| `src/analysis/yesno_balance.py` | Verify question GT-yes ratio in train and val splits | inline stat §4.1 |
| `src/analysis/object_size.py` | Accuracy by target object size bin (small/medium/large) × attribute subtype | **Figure 6**, **Table 6** |
| `src/analysis/error_typology.py` | Classify all 60,828 wrong predictions into error types | **Table 7** |
| `src/examples/qualitative_examples.py` | Render annotated example images for verify-vs-query, choose mismatch, relational failures | **Figure 2**, **Figure 4**, **Figure 5** |
| `src/examples/object_size_examples.py` | Render bbox-annotated examples for small-object failures and large-object successes | **Figure 7** |

### Exploratory analyses (not in report)

| Script | Description | Key finding / why not included |
|---|---|---|
| `src/analysis/scene_complexity.py` | Accuracy vs. scene object count and relation density | No significant effect beyond question type |
| `src/analysis/answer_frequency.py` | Accuracy stratified by answer token frequency in training data | Collinear with entropy; superseded by `answer_entropy.py` |
| `src/analysis/answer_regime.py` | Binary / constrained / open-ended regime breakdown | Reframed as format analysis in §4.1 |
| `src/analysis/depth_accuracy.py` | Accuracy vs. logic depth per structural type | Depth effect small vs. format; noted in §4.1 text |
| `src/analysis/vilt_vocab_coverage.py` | ViLT out-of-vocabulary rate per cell before/after normalization | Confirmed vocab not a barrier (≥93% coverage); motivated `normalized_OOV_excluded/` |
| `src/analysis/answer_mismatch.py` | BLIP prediction–GT mismatch analysis by cell | Subsumed by `error_typology.py` |
| `src/analysis/dataset_structure.py` | Structural exploration of GQA fields, answer ontology, scene graph stats | Used to understand dataset before designing experiments |
| `src/analysis/relation_type_examples.py` | Render annotated error examples for relation-type failures | Superseded by `qualitative_examples.py` |
| `src/exploration/explore_answer_structure.py` | Answer vocabulary size and overlap across cells | Early dataset characterisation |
| `src/exploration/explore_dataset_fields.py` | GQA JSON field inventory and type distributions | Early dataset characterisation |
| `src/exploration/explore_depth_per_category.py` | Logic depth distribution per structural × semantic cell | Led to `depth_accuracy.py` |

---

## Report Traceability

Each number in the report is traceable to a script and output file below.

### Figure 1 — BLIP accuracy heatmap (`fig:heatmap`)
**File:** `results/normalized/accuracy_heatmap_blip.png`

```bash
python src/analysis/analyze_results.py
```
Key outputs: `results/normalized/accuracy_5x5_blip.csv`, `results/analysis/blip_vilt_comparison/comparison_table.csv`

---

### Table 2 — Format effect at controlled depth (`tab:format`)
**Numbers:** color +23.9 pp, size +19.9 pp, position +5.9 pp, material +2.2 pp (verify − query at depth = 2)

```bash
python src/analysis/format_depth_analysis.py
```
Key outputs: `results/analysis/format_depth_analysis/`

---

### Table 3 — Choose outcome breakdown (`tab:choose`)
**Numbers:** exact correct 61.6%, exact wrong 31.8%, free-form 3.8%, near-miss 0.7%

```bash
python src/analysis/format_depth_analysis.py
```
Key outputs: `results/analysis/format_depth_analysis/choose_mismatch.csv` *(gitignored; rerun to regenerate)*

---

### Figure 2 — Example 1: verify vs. query (`fig:ex1`)
**File:** `docs/figures/report/figure_example1.png`

```bash
python src/examples/qualitative_examples.py
```
Key outputs: `results/analysis/qualitative_examples/A_verify_vs_query/` *(gitignored)*

---

### Figure 3 — Answer entropy heatmap (`fig:entropy`)
**File:** `results/analysis/answer_entropy/entropy_heatmap.png`

```bash
python src/analysis/answer_entropy.py
```
Key outputs: `results/analysis/answer_entropy/entropy_accuracy.csv`

The 7.8 pp rel/attr gap at equal low entropy is read from this CSV.

---

### Figure 4 — Example 2: choose mismatch (`fig:ex2`)
**File:** `docs/figures/report/figure_example2.png`

```bash
python src/examples/qualitative_examples.py
```
Key outputs: `results/analysis/qualitative_examples/B_choose_mismatch/` *(gitignored)*

---

### Table 4 — Relation subtypes (`tab:reltype`)
**Numbers:** spatial 41.5%, action 40.6%, comparative 32.2%

```bash
python src/analysis/relation_type.py
```
Key outputs: `results/analysis/relation_type/rel_type_depth.csv`

---

### Table 5 — Attribute subtypes (`tab:attrtype`)
**Numbers:** material 69.5%, color 52.8%, position 48.4%, size 46.9%, pose 30.1%, activity 26.4%

```bash
python src/analysis/attribute_type.py
```
Key outputs: `results/analysis/attribute_type/attr_type_accuracy.csv`

---

### Figure 5 — Example 3: relational failure (`fig:ex3`)
**File:** `docs/figures/report/figure_example3.png`

```bash
python src/examples/qualitative_examples.py
```
Key outputs: `results/analysis/qualitative_examples/C_query_wrong/`, `D_position_confusion/` *(gitignored)*

---

### Inline stat — yes/no balance (§4.1 body text)
**Numbers:** GT-yes 49.6% (train), 49.5% (val)

```bash
python src/analysis/yesno_balance.py
```
Key outputs: `results/analysis/yesno_balance/bias_table.csv`

---

### Figure 6 — Object size heatmap (`fig:objsize`)
**File:** `results/analysis/object_size/objsize_attr_heatmap.png`

```bash
python src/analysis/object_size.py
```
Key outputs: `results/analysis/object_size/objsize_attr_heatmap.png`

---

### Table 6 — Size breakdown by attr subtype (`tab:size`)
**Numbers:** material +17.0 pp, color +10.1 pp, position +2.2 pp (large − small)

```bash
python src/analysis/object_size.py
```
Key outputs: `results/analysis/object_size/obj_area_per_question.csv` *(gitignored; rerun to regenerate)*

---

### Figure 7 — Example 4: object size contrast (`fig:ex4`)
**File:** `docs/figures/report/figure_example4.png`

```bash
python src/examples/object_size_examples.py
```
Key outputs: `results/analysis/qualitative_examples/E_object_size/` *(gitignored)*

---

### Table 7 — Error typology (`tab:errors`)
**Numbers:** wrong_value 67.9%, binary_flip 20.0%, person_ambiguity 7.1%, compound_truncation 4.5%, near_miss 0.5%, wrong_type 0.1%

```bash
python src/analysis/error_typology.py
```
Key outputs: `results/analysis/error_typology/wrong_predictions.csv` *(gitignored; rerun to regenerate)*

---

## Quick Reference

| Script | Report output |
|---|---|
| `src/inference/run_inference.py` | all downstream analyses |
| `src/analysis/build_question_features.py` | all analyses using auxiliary features |
| `src/analysis/analyze_results.py` | **Figure 1** |
| `src/analysis/format_depth_analysis.py` | **Table 2**, **Table 3** |
| `src/analysis/answer_entropy.py` | **Figure 3** |
| `src/analysis/relation_type.py` | **Table 4** |
| `src/analysis/attribute_type.py` | **Table 5** |
| `src/analysis/yesno_balance.py` | inline yes/no balance stat |
| `src/analysis/object_size.py` | **Figure 6**, **Table 6** |
| `src/analysis/error_typology.py` | **Table 7** |
| `src/examples/qualitative_examples.py` | **Figure 2**, **Figure 4**, **Figure 5** |
| `src/examples/object_size_examples.py` | **Figure 7** |

