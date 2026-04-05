# Workload Distribution

**Project:** Question-Type-Aware Evaluation of Vision-Language Models for VQA.  
**Team:** Sophia Yang (xya134) · Qingrui Li (qingruil)

---

## Sophia Yang (xya134)

**Setup & Infrastructure**
- Environment setup, dependency management, and repository organization
- GQA dataset download, verification, and data preprocessing pipeline

**Inference & Feature Extraction**
- BLIP inference pipeline (`src/inference/run_inference.py`) on GQA balanced validation split
- Auxiliary question feature extraction (`src/analysis/build_question_features.py`)

**Analysis**
- Answer entropy analysis (`src/analysis/answer_entropy.py`)
- Attribute type analysis (`src/analysis/attribute_type.py`)
- Object size analysis (`src/analysis/object_size.py`) and qualitative examples (`src/examples/object_size_examples.py`)
- Yes/no balance analysis (`src/analysis/yesno_balance.py`)
- Dataset structure exploration (`src/exploration/`)

**Report Writing**
- Introduction and related work sections
- Visual grounding quality findings (object size / perceptual attributes)
- Figure preparation and caption writing for attribute/object-size plots
- Proofreading and final formatting

---

## Qingrui Li (qingruil)

**Setup & Infrastructure**
- ViLT baseline setup and preliminary BLIP–ViLT comparison
- Results directory organization and output validation scripts

**Inference & Feature Extraction**
- ViLT inference and OOV vocabulary coverage analysis (`src/analysis/vilt_vocab_coverage.py`)
- Answer mismatch and error typology analysis (`src/analysis/answer_mismatch.py`, `src/analysis/error_typology.py`)

**Analysis**
- Format × depth interaction analysis (`src/analysis/format_depth_analysis.py`)
- Relation type analysis (`src/analysis/relation_type.py`) and relation examples (`src/analysis/relation_type_examples.py`)
- Answer frequency and answer regime analyses (`src/analysis/answer_frequency.py`, `src/analysis/answer_regime.py`)
- Scene complexity and depth accuracy analyses (`src/analysis/scene_complexity.py`, `src/analysis/depth_accuracy.py`)
- Qualitative example generation (`src/examples/qualitative_examples.py`)

**Report Writing**
- Methodology and experimental setup sections
- Answer format and reasoning demand findings
- Results tables (5×5 accuracy, normalized/strict variants)
- Conclusion and limitations sections
- Proofreading and final formatting
