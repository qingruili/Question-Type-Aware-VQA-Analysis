# CMPT 713 Natural Language Processing

**Course:** CMPT 713 — Natural Language Processing
**Team:** curious-minds.  
**Members:** Sophia Yang (xya134) · Qingrui Li (qingruil)

---

## Table of Contents

| # | Directory | Topic |
|---|-----------|-------|
| 0 | [hw0/](hw0/) | Text Segmentation |
| 1 | [hw1/](hw1/) | Spell Checking |
| 2 | [hw2/](hw2/) | BERT Finetuning for Robust Phrasal Chunking |
| 3 | [hw3/](hw3/) | Neural Machine Translation and Transformers |
| 4 | [hw4/](hw4/) | Prefix Tuning for Text Generation |
| — | [project/](project/) | Final Project: Question-Type-Aware VQA Analysis |

---

## Homeworks

### [HW0 — Text Segmentation](hw0/)
Word segmentation using unigram language model probabilities.

### [HW1 — Spell Checking](hw1/)
Noisy-channel spell correction with edit distance and language model scoring.

### [HW2 — BERT Finetuning for Robust Phrasal Chunking](hw2/)
Sequence labeling for NER evaluated with CoNLL F1 metric.

### [HW3 — Neural Machine Translation and Transformers](hw3/)
Attention-based encoder–decoder NMT evaluated with BLEU score.

### [HW4 — Prefix Tuning for Text Generation](hw4/)
Further NMT and sequence labeling experiments, evaluated with BLEU.

---

## Final Project

### [Project — Question-Type-Aware Evaluation of Vision-Language Models for VQA](project/)

A fine-grained analysis of `Salesforce/blip-vqa-base` on the GQA balanced validation split (132,062 questions). Using GQA's two-axis taxonomy — **structural type** (query, verify, logical, choose, compare) × **semantic type** (rel, attr, obj, cat, global) — we identify three orthogonal accuracy drivers:

1. **Answer format** — verify outperforms query by up to +23.9 pp at the same reasoning depth.
2. **Reasoning demand** — relational/spatial questions lag attribute questions by 7.8 pp even after controlling for answer entropy.
3. **Visual grounding quality** — perceptual attributes (color, material) degrade sharply with small objects; spatial position does not, revealing a conceptual bottleneck.

See [`project/README.md`](project/README.md) for full setup and reproduction instructions.
