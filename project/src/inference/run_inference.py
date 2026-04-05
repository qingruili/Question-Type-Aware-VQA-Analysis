#!/usr/bin/env python3
"""
src/inference/run_inference.py

Full inference run of BLIP and ViLT on GQA balanced validation split.

Features:
  - Resumable: skips already-processed questions on restart
  - Writes predictions incrementally to JSONL (safe against interruption)
  - Reads images from data/images/ (extracted) or data/images.zip (fallback)
  - Sorts questions by imageId → each image loaded once (cache-friendly)
  - Saves per-question metadata needed for all downstream analyses

Usage:
  python run_inference.py                 # full run, both models
  python run_inference.py --dry-run 200   # test on first 200 questions
  python run_inference.py --skip-blip     # ViLT only
  python run_inference.py --skip-vilt     # BLIP only

Output:
  results/predictions/all_predictions.jsonl   — one JSON object per line
"""

import argparse
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
QUESTIONS_PATH   = PROJECT_ROOT / "data" / "questions1.2" / "val_balanced_questions.json"
IMAGES_DIR       = PROJECT_ROOT / "data" / "images"       # extracted directory (preferred)
IMAGES_ZIP       = PROJECT_ROOT / "data" / "images.zip"   # fallback
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "predictions" / "all_predictions.jsonl"

# ── Answer normalization ───────────────────────────────────────────────────────
_NUM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

def normalize(s: str) -> str:
    s = s.strip().lower()
    if s in _NUM_MAP:
        s = _NUM_MAP[s]
    s = re.sub(r"^(a |an |the )", "", s)
    return s

# ── Image loading (with single-image cache keyed on imageId) ──────────────────
_zip_handle: zipfile.ZipFile | None = None
_zip_names: set[str] | None = None
_cache_id: str | None = None
_cache_img: Image.Image | None = None


def _open_zip() -> zipfile.ZipFile:
    global _zip_handle, _zip_names
    if _zip_handle is None:
        _zip_handle = zipfile.ZipFile(IMAGES_ZIP)
        _zip_names = set(_zip_handle.namelist())
    return _zip_handle


def load_image(image_id: str) -> Image.Image | None:
    """Return PIL image for image_id, using a one-slot cache (effective when
    questions are sorted by imageId)."""
    global _cache_id, _cache_img

    if image_id == _cache_id:
        return _cache_img

    img = None

    # 1. Try extracted directory
    path = IMAGES_DIR / f"{image_id}.jpg"
    if path.exists():
        img = Image.open(path).convert("RGB")

    # 2. Fall back to zip
    elif IMAGES_ZIP.exists():
        zf = _open_zip()
        key = f"images/{image_id}.jpg"
        if key in _zip_names:
            with zf.open(key) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")

    _cache_id, _cache_img = image_id, img
    return img


# ── Helpers ────────────────────────────────────────────────────────────────────
def program_depth(q: dict) -> int:
    return len(q.get("semantic", []))


def load_done_qids(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["qid"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run BLIP + ViLT inference on GQA val balanced")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="Process only the first N questions (for testing)")
    parser.add_argument("--skip-blip", action="store_true", help="Skip BLIP inference")
    parser.add_argument("--skip-vilt", action="store_true", help="Skip ViLT inference")
    args = parser.parse_args()

    run_blip = not args.skip_blip
    run_vilt = not args.skip_vilt

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Load questions ─────────────────────────────────────────────────────────
    print("Loading questions...")
    with open(QUESTIONS_PATH) as f:
        data = json.load(f)
    print(f"Total questions in val_balanced: {len(data):,}")

    # ── Resume: find already-processed qids ───────────────────────────────────
    done_qids = load_done_qids(PREDICTIONS_FILE)
    if done_qids:
        print(f"Resuming — already done: {len(done_qids):,} | remaining: {len(data) - len(done_qids):,}")

    # ── Build to-do list, sort by imageId for cache efficiency ────────────────
    todo = [(qid, q) for qid, q in data.items() if qid not in done_qids]
    todo.sort(key=lambda x: x[1]["imageId"])

    if args.dry_run > 0:
        todo = todo[: args.dry_run]
        print(f"Dry-run mode: processing {len(todo)} questions")

    if not todo:
        print("Nothing to process — all questions already done.")
        return

    # ── Load models ────────────────────────────────────────────────────────────
    if run_blip:
        print("\nLoading BLIP (Salesforce/blip-vqa-base)...")
        blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        blip_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(device).eval()

    if run_vilt:
        print("Loading ViLT (dandelin/vilt-b32-finetuned-vqa)...")
        vilt_proc  = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        ).to(device).eval()

    # ── Inference loop ─────────────────────────────────────────────────────────
    n_missing = 0
    n_errors  = 0
    t_run_start = time.time()

    with open(PREDICTIONS_FILE, "a", buffering=1) as out_f, torch.no_grad():
        for qid, q in tqdm(todo, desc="Inference", unit="q", dynamic_ncols=True):
            image_id      = q["imageId"]
            question_text = q["question"]
            gt_answer     = q["answer"]
            structural    = q["types"]["structural"]
            semantic      = q["types"]["semantic"]
            depth         = program_depth(q)

            image = load_image(image_id)
            if image is None:
                n_missing += 1
                continue

            # ── BLIP ──────────────────────────────────────────────────────────
            blip_answer = blip_time = None
            if run_blip:
                t0 = time.perf_counter()
                try:
                    inputs = blip_proc(image, question_text, return_tensors="pt").to(device)
                    out = blip_model.generate(**inputs)
                    blip_answer = blip_proc.decode(out[0], skip_special_tokens=True)
                    blip_time = round(time.perf_counter() - t0, 4)
                except Exception as e:
                    blip_answer = ""
                    n_errors += 1

            # ── ViLT ──────────────────────────────────────────────────────────
            vilt_answer = vilt_time = None
            if run_vilt:
                t0 = time.perf_counter()
                try:
                    inputs = vilt_proc(image, question_text, return_tensors="pt").to(device)
                    out = vilt_model(**inputs)
                    pred_id = out.logits.argmax(-1).item()
                    vilt_answer = vilt_model.config.id2label[pred_id]
                    vilt_time = round(time.perf_counter() - t0, 4)
                except Exception as e:
                    vilt_answer = ""
                    n_errors += 1

            # ── Correctness ───────────────────────────────────────────────────
            blip_correct = (
                normalize(blip_answer) == normalize(gt_answer)
                if blip_answer is not None else None
            )
            vilt_correct = (
                normalize(vilt_answer) == normalize(gt_answer)
                if vilt_answer is not None else None
            )

            row = {
                "qid":           qid,
                "imageId":       image_id,
                "question":      question_text,
                "gt_answer":     gt_answer,
                "structural":    structural,
                "semantic":      semantic,
                "program_depth": depth,
                "blip_answer":   blip_answer,
                "vilt_answer":   vilt_answer,
                "blip_correct":  blip_correct,
                "vilt_correct":  vilt_correct,
                "blip_time":     blip_time,
                "vilt_time":     vilt_time,
            }
            out_f.write(json.dumps(row) + "\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_run_start
    total_processed = len(todo) - n_missing
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.2f} h ({elapsed:.0f} s)")
    print(f"Processed : {total_processed:,} questions")
    print(f"Missing images : {n_missing}")
    print(f"Inference errors : {n_errors}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
