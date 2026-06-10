#!/usr/bin/env python
"""
Mask2Former inference + query-saturation diagnostic.

Point it at a trained checkpoint (a dir produced by train_mask2former.py,
e.g. runs/.../checkpoint-best) and one or more images. For each image it
reports the things that actually answer "are my queries the bottleneck?":

  * num_queries .............. the hard ceiling on instances the model can emit
  * active queries ........... how many queries predict a REAL class (not the
                               no-object class) at zero threshold. This is the
                               direct, threshold-free saturation test.
  * threshold sweep .......... detected-instance counts per class as the
                               confidence threshold drops (0.0 -> 0.9). Shows
                               how many of the misses are just low-confidence
                               (recoverable) vs genuinely absent.
  * score distribution ....... confidence spread of the active queries.
  * overlay PNG .............. masks drawn on the image for eyeballing quality.

How to read it
--------------
  active ~= num_queries on your dense frames  -> SATURATED, raise num_queries.
  active well below num_queries               -> queries are NOT the limit;
                                                 your ceiling is conf/IoU, not slots.

Because the sweep also shows you the conf-threshold effect, it doubles as a
sanity check on the eval-threshold issue: if instance counts keep climbing as
the threshold drops toward 0, your fixed-0.5 eval was discarding real
detections.

Usage
-----
    python m2f_infer_diag.py \
        --model runs/m2f_swin_t_parts_v1/checkpoint-best \
        --image path/to/frame.png \
        --out  diag_out

    # whole folder (great for finding where saturation kicks in across density):
    python m2f_infer_diag.py --model ... --image path/to/frames/ --out diag_out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SWEEP = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> np.ndarray:
    """Match training-time loading: grayscale -> 3ch, RGBA -> RGB, ->uint8."""
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:  # e.g. 16-bit phase contrast
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = (255.0 * (img - lo) / max(hi - lo, 1.0)).astype(np.uint8)
    return img


def collect_images(p: Path):
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
    return [p]


def palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(60, 256, size=(max(n, 1), 3), dtype=np.uint8)


def overlay(image: np.ndarray, masks, alpha: float = 0.5) -> np.ndarray:
    out = image.astype(np.float32).copy()
    cols = palette(len(masks))
    for i, m in enumerate(masks):
        m = np.asarray(m, dtype=bool)
        out[m] = (1 - alpha) * out[m] + alpha * cols[i]
    return out.clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
@torch.no_grad()
def post_process(processor, outputs, hw, thr):
    """Wrapper that tolerates older transformers without return_binary_maps."""
    try:
        return processor.post_process_instance_segmentation(
            outputs, target_sizes=[hw], threshold=thr,
            return_binary_maps=True)[0], True
    except TypeError:
        return processor.post_process_instance_segmentation(
            outputs, target_sizes=[hw], threshold=thr)[0], False


def masks_from_result(res, hw, binary):
    """Return list of (label_id, score, HxW bool mask)."""
    seg = res.get("segmentation", None)
    info = res.get("segments_info", [])
    out = []
    if seg is None or len(info) == 0:
        return out
    seg_t = seg if isinstance(seg, torch.Tensor) else torch.as_tensor(seg)
    if binary and seg_t.dim() == 3:                      # (N, H, W) binaries
        for k, s in enumerate(info):
            out.append((s["label_id"], float(s["score"]),
                        seg_t[k].bool().cpu().numpy()))
    else:                                                # (H, W) id map
        for s in info:
            out.append((s["label_id"], float(s["score"]),
                        (seg_t == s["id"]).bool().cpu().numpy()))
    return out


@torch.no_grad()
def analyse(model, processor, image, device, no_object_idx):
    H, W = image.shape[:2]
    enc = processor(images=[image], return_tensors="pt")
    outputs = model(pixel_values=enc["pixel_values"].to(device))

    # ---- threshold-free saturation test, straight from the class logits ----
    class_logits = outputs.class_queries_logits[0]        # (Q, C+1)
    probs = class_logits.softmax(-1)
    top_label = probs.argmax(-1)
    top_score = probs.max(-1).values
    active = top_label != no_object_idx
    n_active = int(active.sum().item())
    active_scores = top_score[active].detach().cpu().numpy()

    # ---- confidence-threshold sweep via the official post-processor ----
    sweep = {}
    for thr in SWEEP:
        res, _ = post_process(processor, outputs, (H, W), thr)
        info = res.get("segments_info", [])
        per_cls = {}
        for s in info:
            per_cls[s["label_id"]] = per_cls.get(s["label_id"], 0) + 1
        sweep[thr] = (len(info), per_cls)

    return outputs, n_active, active_scores, sweep, (H, W)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_classes(per_cls, id2label):
    if not per_cls:
        return "-"
    return ", ".join(f"{id2label.get(c, c)}={n}"
                     for c, n in sorted(per_cls.items()))


def report(name, num_queries, n_active, active_scores, sweep, id2label):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    pct = 100.0 * n_active / max(num_queries, 1)
    print(f"num_queries (ceiling) : {num_queries}")
    print(f"active queries (thr=0): {n_active}  ({pct:.0f}% of ceiling)")
    if n_active >= 0.95 * num_queries:
        print("  >>> SATURATED: active queries are at the ceiling. The model "
              "physically cannot emit more instances here -- raise num_queries.")
    elif n_active >= 0.75 * num_queries:
        print("  >>> NEAR saturation (>=75%). Some dense frames likely clip; "
              "more queries would help on the busiest images.")
    else:
        print("  >>> Headroom remains; queries are NOT the bottleneck on this "
              "image. A low instance count here points at conf/IoU, not slots.")

    if active_scores.size:
        qs = np.quantile(active_scores, [0.0, 0.25, 0.5, 0.75, 1.0])
        print("active-query scores   : "
              f"min={qs[0]:.2f} q25={qs[1]:.2f} med={qs[2]:.2f} "
              f"q75={qs[3]:.2f} max={qs[4]:.2f}")
        print(f"  of {n_active} active, {(active_scores >= 0.5).sum()} are >=0.50 "
              f"and {(active_scores >= 0.25).sum()} are >=0.25")

    print("threshold sweep (instances kept):")
    print(f"  {'thr':>5} | {'total':>6} | per-class")
    for thr in SWEEP:
        total, per_cls = sweep[thr]
        print(f"  {thr:>5.2f} | {total:>6} | {fmt_classes(per_cls, id2label)}")
    print("  (if total keeps rising as thr->0, a fixed-0.5 eval was dropping "
          "real detections)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="checkpoint dir (contains config.json + processor).")
    ap.add_argument("--base-model",
                    default="facebook/mask2former-swin-tiny-coco-instance",
                    help="fallback processor source when the checkpoint dir has "
                         "no preprocessor_config.json (i.e. a Trainer "
                         "checkpoint-<step>, not a checkpoint-best). Set this to "
                         "the same --model id you trained from.")
    ap.add_argument("--image", type=Path, required=True,
                    help="image file OR a folder of images.")
    ap.add_argument("--out", type=Path, default=Path("diag_out"),
                    help="dir for overlay PNGs (created if missing).")
    ap.add_argument("--viz-threshold", type=float, default=0.5,
                    help="confidence threshold used for the saved overlay.")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: auto).")
    args = ap.parse_args()

    device = (args.device or
              ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(args.model)).to(device).eval()
    # Trainer checkpoint-<step> dirs don't contain the processor (it's only
    # written to checkpoint-best by the callback, or to <out> after a run that
    # actually finishes). Fall back to the base model's processor and force
    # do_resize=False to match training when the checkpoint has none.
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(str(args.model))
    except (OSError, EnvironmentError, ValueError):
        print(f"  no processor in {args.model}; falling back to "
              f"{args.base_model} (do_resize=False)")
        processor = Mask2FormerImageProcessor.from_pretrained(args.base_model)
        processor.do_resize = False

    num_queries = int(model.config.num_queries)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    no_object_idx = len(id2label)          # classes are 0..L-1, no-object is L
    do_resize = getattr(processor, "do_resize", None)
    print(f"num_queries={num_queries} | classes={id2label} | "
          f"processor.do_resize={do_resize} | device={device}")
    if do_resize:
        print("  NOTE: processor.do_resize=True -- if you trained with "
              "do_resize=False, this inference is NOT matching training scale.")

    images = collect_images(args.image)
    if not images:
        raise SystemExit(f"No images found at {args.image}")

    peak_active = 0
    for path in images:
        img = load_image(path)
        outputs, n_active, scores, sweep, hw = analyse(
            model, processor, img, device, no_object_idx)
        report(path.name, num_queries, n_active, scores, sweep, id2label)
        peak_active = max(peak_active, n_active)

        # overlay at the chosen viz threshold
        res, binary = post_process(processor, outputs, hw, args.viz_threshold)
        dets = masks_from_result(res, hw, binary)
        if dets:
            ov = overlay(img, [m for _, _, m in dets])
            out_png = args.out / f"{path.stem}_overlay.png"
            Image.fromarray(ov).save(out_png)
            print(f"saved overlay ({len(dets)} inst @thr"
                  f"{args.viz_threshold}) -> {out_png}")
        else:
            print(f"no instances above thr={args.viz_threshold}; no overlay.")

    if len(images) > 1:
        print(f"\n{'#' * 70}")
        print(f"PEAK active queries across {len(images)} images: "
              f"{peak_active} / {num_queries}")
        if peak_active >= 0.95 * num_queries:
            print("Across the set you ARE hitting the ceiling -> size "
                  "num_queries to ~1.5-2x your true peak instance count.")
        else:
            print("Across the set you are NOT hitting the ceiling -> queries "
                  "are not your limiting factor; look at conf/IoU instead.")
        print('#' * 70)


if __name__ == "__main__":
    main()
