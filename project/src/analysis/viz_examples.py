#!/usr/bin/env python3
"""
Image Visualization Utility
----------------------------
Reads images on-demand from data/images.zip and renders them as
annotated matplotlib figures (image + question/GT/BLIP).

Core function:
    show_examples(rows, title, out_path, cols=3)

  rows  : list of dicts with keys:
            image_id   - int or str image ID
            question   - question text
            gt_answer  - ground truth answer
            blip_answer- BLIP prediction
            correct    - bool
            label      - (optional) extra subtitle text

Additional function:
    save_individual_examples(rows, out_dir, prefix="")

      Saves one PNG per example into out_dir, named
      "{prefix}{index:02d}_{imageId}_{correct}.png".
      Each image shows just one example (larger, cleaner layout).

Usage:
    from viz_examples import show_examples, save_individual_examples
    show_examples(rows, title="My examples", out_path="out.png")
    save_individual_examples(rows, out_dir="results/analysis/qualitative_examples/A_verify_vs_query/")
"""
import io
import zipfile
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_ZIP   = PROJECT_ROOT / "data" / "images.zip"

# open once at import time — fast for repeated reads
_zip = None

def _get_zip():
    global _zip
    if _zip is None:
        _zip = zipfile.ZipFile(IMAGES_ZIP, "r")
    return _zip


def load_image(image_id):
    """Return a PIL Image for the given image_id, or None on error."""
    z = _get_zip()
    path = f"images/{image_id}.jpg"
    try:
        data = z.read(path)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except KeyError:
        return None


def _wrap(text, max_chars=60):
    """Simple word-wrap to max_chars per line."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def show_examples(rows, title="", out_path=None, cols=3, img_size=(3, 2.8)):
    """
    Render a grid of examples.

    Parameters
    ----------
    rows      : list of dicts (keys: image_id, question, gt_answer,
                               blip_answer, correct, label [optional])
    title     : overall figure title
    out_path  : path to save (str or Path); if None, plt.show() is called
    cols      : number of columns in the grid
    img_size  : (width, height) in inches per cell
    """
    n = len(rows)
    n_rows = (n + cols - 1) // cols
    fig_w = img_size[0] * cols
    fig_h = img_size[1] * n_rows + (0.4 if title else 0)

    fig, axes = plt.subplots(n_rows, cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    for idx, row in enumerate(rows):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        img = load_image(row["image_id"])
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#eeeeee")
            ax.text(0.5, 0.5, "image\nnot found",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=8)

        ax.set_xticks([]); ax.set_yticks([])

        # border colour: green = correct, red = wrong
        colour = "#2ECC71" if row.get("correct") else "#E74C3C"
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.5)

        # caption
        q_text   = _wrap(row.get("question", ""), 48)
        gt_text  = str(row.get("gt_answer", ""))
        pred_text = str(row.get("blip_answer", ""))
        label_extra = row.get("label", "")

        mark = "✓" if row.get("correct") else "✗"
        caption = f"Q: {q_text}\nGT: {gt_text}  BLIP: {pred_text} {mark}"
        if label_extra:
            caption = f"[{label_extra}]\n" + caption

        ax.set_xlabel(caption, fontsize=6.5, loc="left",
                      color="black", labelpad=3)

    # hide unused cells
    for idx in range(n, n_rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)

    plt.tight_layout(pad=0.3, h_pad=1.2, w_pad=0.5)

    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def save_individual_examples(rows, out_dir, prefix=""):
    """
    Save one PNG per example as a standalone, larger image.

    File name pattern: {prefix}{index:02d}_{imageId}_{ok|wrong}.png
    Each figure: image on top, caption block below.

    Parameters
    ----------
    rows    : list of dicts (same schema as show_examples)
    out_dir : directory path (str or Path); created if absent
    prefix  : optional string prepended to every filename
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows):
        img = load_image(row["image_id"])
        correct = bool(row.get("correct", False))
        border_colour = "#2ECC71" if correct else "#E74C3C"
        mark = "✓" if correct else "✗"

        fig, ax = plt.subplots(figsize=(5, 4.8))

        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#dddddd")
            ax.text(0.5, 0.5, "image not found",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=10)

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colour)
            spine.set_linewidth(3)

        # detailed caption
        q_text    = _wrap(row.get("question", ""), 52)
        gt_text   = str(row.get("gt_answer", ""))
        pred_text = str(row.get("blip_answer", ""))
        label_extra = row.get("label", "")

        caption = f"Q: {q_text}\nGT: {gt_text}    BLIP: {pred_text} {mark}"
        if label_extra:
            caption = f"[{label_extra}]\n" + caption

        ax.set_xlabel(caption, fontsize=8, loc="left",
                      color="black", labelpad=5)

        plt.tight_layout(pad=0.5)

        ok_str  = "ok" if correct else "wrong"
        fname   = f"{prefix}{idx:02d}_{row['image_id']}_{ok_str}.png"
        fpath   = out_dir / fname
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(rows)} individual images to {out_dir}")
