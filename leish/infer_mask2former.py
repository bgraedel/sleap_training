"""
Run a trained Mask2Former (from ``train_mask2former.py``) on images, an AVI/MP4
video, or a (multi-page) TIFF stack, and save an instance-segmentation overlay
as PNG(s) or an MP4 — optionally also dumping 16-bit instance-ID masks.

Highlights
----------
* **Any input resolution.** Frames are brought to the model's training size by
  *letterbox* (aspect-preserving pad, default) or *stretch*, run, then the
  predicted masks are mapped back to each frame's native resolution. So a 512,
  1200x900, or 2048 frame all work regardless of what the model trained at.
* **16-bit microscopy data.** 16-bit / float TIFFs are contrast-stretched to
  8-bit (percentile by default) so they match the 8-bit training distribution.
  8-bit inputs pass through untouched.
* **Inputs:** a single image, a folder of images, an ``.avi``/``.mp4`` video,
  or a ``.tif``/``.tiff`` stack.
* **Outputs:** ``--save png`` (one overlay PNG per frame) and/or ``--save mp4``
  (an overlay video). ``--save-masks`` additionally writes a 16-bit instance-ID
  PNG + a per-frame JSON (id -> class, score) for downstream association.

Examples
--------
Video -> overlay MP4::

    python infer_mask2former.py \\
        --model runs/m2f_swin_t_640_v1/checkpoint-best \\
        --input movie.avi --out preds/ --save mp4

16-bit TIFF stack -> MP4 + instance-ID masks::

    python infer_mask2former.py --model <ckpt> \\
        --input stack.tif --out preds/ --save mp4 --save-masks

Folder of images -> overlay PNGs at higher conf::

    python infer_mask2former.py --model <ckpt> \\
        --input frames/ --out preds/ --save png --conf 0.6

Notes
-----
* ``--model`` should point at a checkpoint dir that contains model weights
  (e.g. ``checkpoint-best/`` or ``checkpoint-1200/``). If you pass the run root
  it will auto-use ``checkpoint-best/`` when present.
* The eval-time post-processing knobs mirror training defaults
  (``--conf 0.5 --mask-threshold 0.4 --overlap-threshold 0.5``); lower
  ``--mask-threshold`` if thin flagella come out fragmented.

Install (in addition to the training deps)::

    pip install opencv-python tifffile   # cv2 for video, tifffile for stacks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor)

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

try:
    import tifffile
    _HAVE_TIFF = True
except Exception:
    _HAVE_TIFF = False

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
TIFF_EXTS  = {".tif", ".tiff"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Checkpoint / model loading
# ---------------------------------------------------------------------------

def resolve_checkpoint(path: Path) -> Path:
    """Return a dir that actually holds model weights. If ``path`` is a run
    root without weights but with a ``checkpoint-best/``, use that."""
    path = Path(path)
    def _has_weights(p: Path) -> bool:
        return ((p / "model.safetensors").exists()
                or (p / "pytorch_model.bin").exists())
    if _has_weights(path):
        return path
    best = path / "checkpoint-best"
    if _has_weights(best):
        print(f"--model is a run root; using {best}")
        return best
    raise FileNotFoundError(
        f"No model weights (model.safetensors / pytorch_model.bin) found in "
        f"{path} or {path/'checkpoint-best'}. Point --model at a checkpoint dir.")


def load_model_and_processor(ckpt: Path, device: torch.device,
                             infer_size: Optional[int]):
    ckpt = resolve_checkpoint(ckpt)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(ckpt))
    model.to(device).eval()
    # do_resize stays False: we bring frames to the model size ourselves so the
    # masks can be letterboxed back exactly.
    processor = Mask2FormerImageProcessor.from_pretrained(str(ckpt),
                                                          do_resize=False)
    # Resolve the operating square size: CLI override > processor.size > 640.
    size = infer_size
    if size is None:
        sz = getattr(processor, "size", None) or {}
        size = sz.get("height") or sz.get("shortest_edge") or 640
    size = int(size)
    if size % 32 != 0:
        size = max(32, round(size / 32) * 32)
        print(f"WARNING: infer size rounded to nearest /32 -> {size}")
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return model, processor, size, id2label


# ---------------------------------------------------------------------------
# Frame IO
# ---------------------------------------------------------------------------

def to_uint8(frame: np.ndarray, norm: str,
             p_low: float, p_high: float) -> np.ndarray:
    """Bring a frame to uint8. 8-bit input passes through (unless norm forced);
    16-bit / float input is contrast-stretched so it matches the 8-bit
    training distribution."""
    if frame.dtype == np.uint8 and norm != "force":
        return frame
    f = frame.astype(np.float32)
    if norm in ("percentile", "force"):
        lo = np.percentile(f, p_low)
        hi = np.percentile(f, p_high)
    elif norm == "minmax":
        lo, hi = float(f.min()), float(f.max())
    else:  # "none" but non-uint8 -> just clip/scale by dtype max
        lo, hi = 0.0, float(np.iinfo(frame.dtype).max
                            if np.issubdtype(frame.dtype, np.integer) else f.max())
    if hi <= lo:
        hi = lo + 1.0
    f = np.clip((f - lo) / (hi - lo), 0, 1) * 255.0
    return f.astype(np.uint8)


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array from gray / RGB / RGBA input."""
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=2)
    if frame.shape[-1] == 4:
        return frame[..., :3]
    return frame[..., :3]


def iter_frames(input_path: Path, norm: str, p_low: float, p_high: float,
                every: int, max_frames: Optional[int]):
    """Yield ``(index, rgb_uint8, fps)``. ``fps`` is None for stills/folders.

    Handles: single image, folder of images, AVI/MP4 video, TIFF stack."""
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    def _emit(frames_iter, fps):
        kept = 0
        for i, raw in enumerate(frames_iter):
            if every > 1 and (i % every != 0):
                continue
            rgb = ensure_rgb(to_uint8(np.asarray(raw), norm, p_low, p_high))
            yield i, rgb, fps
            kept += 1
            if max_frames and kept >= max_frames:
                break

    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS | TIFF_EXTS)
        if not files:
            raise FileNotFoundError(f"No images found in {input_path}")
        def _gen():
            for p in files:
                yield np.array(Image.open(p))
        yield from _emit(_gen(), None)
        return

    if ext in VIDEO_EXTS:
        if not _HAVE_CV2:
            raise RuntimeError("Reading video needs OpenCV: pip install opencv-python")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        def _gen():
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield bgr[..., ::-1]  # BGR -> RGB
        try:
            yield from _emit(_gen(), fps)
        finally:
            cap.release()
        return

    if ext in TIFF_EXTS:
        frames = load_tiff(input_path)
        yield from _emit(iter(frames), None)
        return

    if ext in IMAGE_EXTS:
        yield from _emit(iter([np.array(Image.open(input_path))]), None)
        return

    raise ValueError(f"Unsupported input type: {input_path} (ext '{ext}')")


def load_tiff(path: Path) -> List[np.ndarray]:
    """Load a (possibly multi-page / multi-dim) TIFF into a list of 2D/3D
    frames. Disambiguates (T,H,W) stacks from a single (H,W,C) RGB image."""
    if _HAVE_TIFF:
        arr = tifffile.imread(str(path))
    else:
        img = Image.open(str(path))
        pages = []
        try:
            i = 0
            while True:
                img.seek(i)
                pages.append(np.array(img))
                i += 1
        except EOFError:
            pass
        arr = np.stack(pages) if len(pages) > 1 else pages[0]
    arr = np.asarray(arr)
    if arr.ndim == 2:                     # single grayscale frame
        return [arr]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):       # single RGB(A) frame
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]   # (T,H,W) stack
    if arr.ndim == 4:                     # (T,H,W,C) stack
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported TIFF shape {arr.shape}")


# ---------------------------------------------------------------------------
# Letterbox / stretch (resolution handling) + inverse
# ---------------------------------------------------------------------------

class Restore:
    """Maps a canvas-resolution (S,S) mask back to the frame's native (H,W).
    Stores the geometry needed to invert the chosen resize."""
    def __init__(self, fit: str, S: int, H: int, W: int,
                 scale: float, top: int, left: int, nH: int, nW: int):
        self.fit, self.S = fit, S
        self.H, self.W = H, W
        self.scale, self.top, self.left, self.nH, self.nW = (
            scale, top, left, nH, nW)

    def __call__(self, mask_canvas: np.ndarray) -> np.ndarray:
        if self.fit == "stretch":
            m = Image.fromarray(mask_canvas.astype(np.uint8) * 255)
            m = m.resize((self.W, self.H), Image.NEAREST)
            return np.asarray(m) > 127
        # letterbox: crop the padded region, then resize back to native
        crop = mask_canvas[self.top:self.top + self.nH,
                           self.left:self.left + self.nW]
        m = Image.fromarray(crop.astype(np.uint8) * 255)
        m = m.resize((self.W, self.H), Image.NEAREST)
        return np.asarray(m) > 127


def preprocess(frame_rgb: np.ndarray, S: int, fit: str
               ) -> Tuple[np.ndarray, Restore]:
    """Return an (S,S,3) uint8 canvas for the model + a Restore mapper."""
    H, W = frame_rgb.shape[:2]
    if fit == "stretch":
        canvas = np.array(Image.fromarray(frame_rgb).resize((S, S),
                                                            Image.BILINEAR))
        return canvas, Restore("stretch", S, H, W, S / max(H, W), 0, 0, S, S)
    # letterbox: scale longest side to S, pad to SxS with the frame's mean gray
    scale = S / max(H, W)
    nH, nW = max(1, round(H * scale)), max(1, round(W * scale))
    resized = np.asarray(Image.fromarray(frame_rgb).resize((nW, nH),
                                                           Image.BILINEAR))
    pad_val = int(frame_rgb.mean())
    canvas = np.full((S, S, 3), pad_val, dtype=np.uint8)
    top, left = (S - nH) // 2, (S - nW) // 2
    canvas[top:top + nH, left:left + nW] = resized
    return canvas, Restore("letterbox", S, H, W, scale, top, left, nH, nW)


# ---------------------------------------------------------------------------
# Model -> per-instance masks
# ---------------------------------------------------------------------------

def extract_instances(result: Dict, binary_maps: bool
                      ) -> List[Tuple[np.ndarray, int, float]]:
    """From one post_process result, return [(mask_SxS bool, label_id, score)]."""
    seg = result.get("segmentation", None)
    info = result.get("segments_info", [])
    out: List[Tuple[np.ndarray, int, float]] = []
    if seg is None or len(info) == 0:
        return out
    seg_t = seg if isinstance(seg, torch.Tensor) else torch.as_tensor(seg)
    if binary_maps and seg_t.dim() == 3:
        for k, d in enumerate(info):
            out.append((seg_t[k].bool().cpu().numpy(),
                        int(d["label_id"]), float(d["score"])))
    else:
        for d in info:
            out.append(((seg_t == d["id"]).bool().cpu().numpy(),
                        int(d["label_id"]), float(d["score"])))
    return out


@torch.no_grad()
def run_batch(model, processor, canvases: List[np.ndarray], S: int,
              device, conf: float, mask_thr: float, overlap_thr: float,
              use_bf16: bool) -> List[List[Tuple[np.ndarray, int, float]]]:
    enc = processor(images=canvases, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=pixel_values)
    else:
        outputs = model(pixel_values=pixel_values)
    B = pixel_values.shape[0]
    pp = dict(target_sizes=[(S, S)] * B, threshold=conf,
              mask_threshold=mask_thr, overlap_mask_area_threshold=overlap_thr)
    try:
        results = processor.post_process_instance_segmentation(
            outputs, return_binary_maps=True, **pp)
        binary = True
    except TypeError:
        binary = False
        try:
            results = processor.post_process_instance_segmentation(outputs, **pp)
        except TypeError:
            results = processor.post_process_instance_segmentation(
                outputs, target_sizes=[(S, S)] * B, threshold=conf)
    return [extract_instances(r, binary) for r in results]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def class_palette(id2label: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    """Fixed, readable colours per class (RGB). Flagellum gets a hot colour so
    thin structures pop against grayscale phase contrast."""
    named = {"body": (60, 200, 60), "flagellum": (255, 40, 40),
             "animal": (40, 140, 255)}
    fallback = [(255, 40, 40), (60, 200, 60), (40, 140, 255),
                (255, 200, 0), (200, 60, 255)]
    pal = {}
    for cid, name in id2label.items():
        pal[cid] = named.get(str(name).lower(), fallback[cid % len(fallback)])
    return pal


def instance_color(idx: int) -> Tuple[int, int, int]:
    """Distinct colour per instance via golden-ratio hue spacing."""
    h = (idx * 0.61803398875) % 1.0
    i = int(h * 6)
    f = h * 6 - i
    q, t, v = int(255 * (1 - f)), int(255 * f), 255
    return [(v, t, 0), (q, v, 0), (0, v, t), (0, q, v),
            (t, 0, v), (v, 0, q)][i % 6]


def _boundary(mask: np.ndarray) -> np.ndarray:
    """1-px boundary of a boolean mask (pure numpy, no cv2/scipy)."""
    b = np.zeros_like(mask)
    d = mask[:-1, :] ^ mask[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = mask[:, :-1] ^ mask[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b & mask


def render_overlay(frame_rgb: np.ndarray,
                   instances: List[Tuple[np.ndarray, int, float]],
                   id2label: Dict[int, str], palette, color_by: str,
                   alpha: float, show_scores: bool) -> np.ndarray:
    """Blend filled masks + crisp boundaries onto the frame. Returns RGB."""
    base = frame_rgb.astype(np.float32)
    fill = base.copy()
    boundary_rgb = np.zeros_like(base)
    boundary_hit = np.zeros(frame_rgb.shape[:2], dtype=bool)
    label_anchors = []
    for idx, (mask, label, score) in enumerate(instances):
        if not mask.any():
            continue
        color = (instance_color(idx) if color_by == "instance"
                 else palette.get(label, (255, 255, 255)))
        fill[mask] = color
        bnd = _boundary(mask)
        boundary_rgb[bnd] = color
        boundary_hit |= bnd
        if show_scores:
            ys, xs = np.where(mask)
            label_anchors.append((int(xs.mean()), int(ys.mean()),
                                  f"{id2label.get(label, label)} {score:.2f}",
                                  color))
    out = (1 - alpha) * base + alpha * fill
    out[boundary_hit] = boundary_rgb[boundary_hit]   # opaque crisp edges
    out = out.clip(0, 255).astype(np.uint8)
    if label_anchors:
        pim = Image.fromarray(out)
        dr = ImageDraw.Draw(pim)
        for x, y, txt, color in label_anchors:
            dr.text((x, y), txt, fill=tuple(int(c) for c in color))
        out = np.asarray(pim)
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class Mp4Writer:
    def __init__(self, path: Path, fps: float):
        if not _HAVE_CV2:
            raise RuntimeError("Writing MP4 needs OpenCV: pip install opencv-python")
        self.path, self.fps = Path(path), float(fps or 10.0)
        self.w = None
        self.size = None

    def write(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        if self.w is None:
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.w = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self.w.isOpened():
                raise RuntimeError(f"Could not open MP4 writer for {self.path}")
        if (w, h) != self.size:                      # frames must be constant size
            rgb = np.asarray(Image.fromarray(rgb).resize(self.size, Image.BILINEAR))
        self.w.write(rgb[..., ::-1])                 # RGB -> BGR

    def close(self):
        if self.w is not None:
            self.w.release()


def save_instance_id_png(path: Path, instances, H: int, W: int) -> List[Dict]:
    """Write a 16-bit instance-ID map (0=bg, 1..N) and return id->meta list."""
    id_map = np.zeros((H, W), dtype=np.uint16)
    meta = []
    for i, (mask, label, score) in enumerate(instances, start=1):
        id_map[mask] = i
        meta.append({"id": i, "label_id": int(label), "score": round(float(score), 4)})
    Image.fromarray(id_map, mode="I;16").save(str(path))
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="checkpoint dir with weights (e.g. .../checkpoint-best). "
                         "A run root is accepted if it contains checkpoint-best/.")
    ap.add_argument("--input", type=Path, required=True,
                    help="image, folder of images, .avi/.mp4 video, or .tif stack")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory")
    ap.add_argument("--save", choices=["png", "mp4", "both"], default="mp4",
                    help="overlay output format (default mp4). 'png' = one "
                         "overlay image per frame; 'both' = PNGs + MP4.")
    ap.add_argument("--save-masks", action="store_true",
                    help="also write a 16-bit instance-ID PNG + JSON per frame "
                         "(for downstream association).")
    ap.add_argument("--infer-size", type=int, default=None,
                    help="square size fed to the model (multiple of 32). Default: "
                         "the training size read from the checkpoint processor.")
    ap.add_argument("--fit", choices=["letterbox", "stretch"], default="letterbox",
                    help="letterbox (default) preserves aspect ratio with padding; "
                         "stretch squashes the frame to a square.")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="object-confidence threshold (default 0.5).")
    ap.add_argument("--mask-threshold", type=float, default=0.4,
                    help="per-pixel mask binarisation threshold (default 0.4). "
                         "Lower if thin flagella come out fragmented.")
    ap.add_argument("--overlap-threshold", type=float, default=0.5,
                    help="overlap_mask_area_threshold for post-processing (0.5).")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="frames per forward pass.")
    ap.add_argument("--color-by", choices=["class", "instance"], default="class",
                    help="colour masks by class (default) or per-instance.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="mask fill opacity 0..1 (default 0.5).")
    ap.add_argument("--show-scores", action="store_true",
                    help="draw class + score text per instance (clutters dense "
                         "frames; off by default).")
    ap.add_argument("--fps", type=float, default=None,
                    help="output MP4 fps. Default: source fps, or 10 for "
                         "stacks/folders.")
    ap.add_argument("--norm", choices=["auto", "percentile", "minmax", "none"],
                    default="auto",
                    help="contrast normalisation to 8-bit for 16-bit/float input "
                         "('auto' = percentile for non-8-bit, passthrough for 8-bit).")
    ap.add_argument("--p-low", type=float, default=1.0,
                    help="low percentile for --norm percentile (default 1).")
    ap.add_argument("--p-high", type=float, default=99.0,
                    help="high percentile for --norm percentile (default 99).")
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (default 1 = all).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many processed frames.")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: cuda if available).")
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    norm = "percentile" if args.norm == "auto" else args.norm

    model, processor, S, id2label = load_model_and_processor(
        args.model, device, args.infer_size)
    palette = class_palette(id2label)
    print(f"Loaded model: classes={id2label}  infer_size={S}  device={device}  "
          f"bf16={use_bf16}")

    args.out.mkdir(parents=True, exist_ok=True)
    save_png = args.save in ("png", "both")
    save_mp4 = args.save in ("mp4", "both")
    mask_dir = args.out / "masks"
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input.stem if not args.input.is_dir() else args.input.name
    mp4_writer: Optional[Mp4Writer] = None

    # Batch frames through the model.
    batch_frames: List[np.ndarray] = []
    batch_restores: List[Restore] = []
    batch_indices: List[int] = []
    src_fps_holder = {"fps": None}

    def flush():
        if not batch_frames:
            return
        canvases = []
        for fr in batch_frames:
            cv, rs = preprocess(fr, S, args.fit)
            canvases.append(cv)
            batch_restores.append(rs)
        dets = run_batch(model, processor, canvases, S, device,
                         args.conf, args.mask_threshold, args.overlap_threshold,
                         use_bf16)
        nonlocal mp4_writer
        for fr, rs, fidx, inst_canvas in zip(batch_frames, batch_restores,
                                             batch_indices, dets):
            # Map every instance mask back to this frame's native resolution.
            native = [(rs(m), lbl, sc) for (m, lbl, sc) in inst_canvas]
            overlay = render_overlay(fr, native, id2label, palette,
                                     args.color_by, args.alpha, args.show_scores)
            if save_png:
                Image.fromarray(overlay).save(
                    str(args.out / f"{stem}_{fidx:05d}.png"))
            if save_mp4:
                if mp4_writer is None:
                    fps = args.fps or src_fps_holder["fps"] or 10.0
                    mp4_writer = Mp4Writer(args.out / f"{stem}_pred.mp4", fps)
                mp4_writer.write(overlay)
            if args.save_masks:
                H, W = fr.shape[:2]
                meta = save_instance_id_png(
                    mask_dir / f"{stem}_{fidx:05d}_ids.png", native, H, W)
                (mask_dir / f"{stem}_{fidx:05d}.json").write_text(json.dumps(
                    {"frame": fidx, "instances": meta,
                     "id2label": id2label}, indent=2))
            print(f"  frame {fidx:05d}: {len(native)} instances")
        batch_frames.clear()
        batch_restores.clear()
        batch_indices.clear()

    n = 0
    for idx, rgb, fps in iter_frames(args.input, norm, args.p_low, args.p_high,
                                     args.every, args.max_frames):
        src_fps_holder["fps"] = fps
        batch_frames.append(rgb)
        batch_indices.append(idx)
        if len(batch_frames) >= args.batch_size:
            flush()
        n += 1
    flush()

    if mp4_writer is not None:
        mp4_writer.close()
        print(f"Wrote {args.out / f'{stem}_pred.mp4'}")
    if save_png:
        print(f"Wrote {n} overlay PNG(s) to {args.out}")
    if args.save_masks:
        print(f"Wrote {n} instance-ID mask(s) to {mask_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
