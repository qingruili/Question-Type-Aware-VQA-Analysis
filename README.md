# Step2: Question-Type-Aware Evaluation of Vision-Language Models for VQA

### [Link to Step 1](https://github.com/qingruili/PEFT-Methods-for-BLIP-2-VQA)

A fine-grained analysis of `Salesforce/blip-vqa-base` on the GQA balanced validation split (132,062 questions). Using GQA's two-axis taxonomy — **structural type** (query, verify, logical, choose, compare) × **semantic type** (rel, attr, obj, cat, global) — we identify three orthogonal accuracy drivers:

1. **Answer format** — verify outperforms query by up to +23.9 pp at the same reasoning depth.
2. **Reasoning demand** — relational/spatial questions lag attribute questions by 7.8 pp even after controlling for answer entropy.
3. **Visual grounding quality** — perceptual attributes (color, material) degrade sharply with small objects; spatial position does not, revealing a conceptual bottleneck.

See [`project/README.md`](project/README.md) for full setup and reproduction instructions.
