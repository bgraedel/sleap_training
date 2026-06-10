"""
Predict a multi-page TIFF stack with a trained nnU-Net v2 checkpoint.

nnU-Net keeps the inference file format identical to training. We trained on
uint8 grayscale PNGs (one channel), so this script:

    1. Reads the input TIFF stack (T frames, 2D each).
    2. Normalizes each frame to uint8 (per-frame percentile by default) and
       writes <stem>_FFFF_0000.png into a temp staging dir.
    3. Runs `nnUNetv2_predict` on that folder.
    4. Reassembles the predicted label PNGs (values in {0,1,2}) back into a
       single uint8 multi-page TIFF stack.

Use this with the `checkpoint_best.pth` from an interrupted training run:

    python nnunet_predict_tiff.py \\
        --base /scratch/bgraedel/nnunet \\
        --dataset-id 501 --resenc M \\
        --fold all --chk checkpoint_best.pth \\
        --tiff input.tif --out predictions.tif

Default normalization (percentile 1-99) usually matches synthetic-trained
contrast well. For 8-bit input that is already in the training range pass
`--norm none`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import imageio.v2 as iio


def to_uint8(arr: np.ndarray, mode: str, lo: float, hi: float) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {arr.shape}")
    if mode == "none":
        if arr.dtype != np.uint8:
            raise ValueError(f"--norm none requires uint8 input (got {arr.dtype})")
        return arr
    a = arr.astype(np.float32)
    if mode == "percentile":
        p_lo, p_hi = np.percentile(a, [lo, hi])
    elif mode == "minmax":
        p_lo, p_hi = float(a.min()), float(a.max())
    else:
        raise ValueError(f"unknown --norm mode {mode!r}")
    if p_hi <= p_lo:
        return np.zeros_like(a, dtype=np.uint8)
    out = np.clip((a - p_lo) / (p_hi - p_lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def run(cmd, env):
    print(">>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiff", required=True, type=Path,
                    help="input TIFF stack (2D frames; multi-page or single)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output multi-page TIFF stack (uint8, values 0/1/2)")
    # nnU-Net env (mirror nnunet_train.py)
    ap.add_argument("--base", type=Path, default=None,
                    help="base dir; uses <base>/nnUNet_raw, _preprocessed, _results")
    ap.add_argument("--raw", type=Path, default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--preprocessed", type=Path,
                    default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--results", type=Path,
                    default=os.environ.get("nnUNet_results"))
    # which model
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--config", default="2d")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--trainer", default=None,
                    help="-tr trainer variant (only needed if you trained with one)")
    ap.add_argument("--resenc", choices=["M", "L", "XL"], default=None,
                    help="ResEnc preset (sets --plans to the matching name)")
    ap.add_argument("--plans", default=None,
                    help="-p plans id, e.g. nnUNetResEncUNetMPlans")
    ap.add_argument("--chk", default="checkpoint_best.pth",
                    help="checkpoint filename inside the fold dir "
                         "(default checkpoint_best.pth; nnU-Net's own default "
                         "is checkpoint_final.pth which only exists if training "
                         "ran to completion)")
    ap.add_argument("--device", default=None,
                    help="CUDA_VISIBLE_DEVICES, e.g. '0'")
    # normalization
    ap.add_argument("--norm", choices=["percentile", "minmax", "none"],
                    default="percentile",
                    help="how to map each frame to uint8")
    ap.add_argument("--lo", type=float, default=1.0,
                    help="low percentile when --norm percentile (default 1.0)")
    ap.add_argument("--hi", type=float, default=99.0,
                    help="high percentile when --norm percentile (default 99.0)")
    ap.add_argument("--invert", action="store_true",
                    help="invert intensities after normalization (use if real "
                         "frames have opposite contrast polarity to training)")
    # misc
    ap.add_argument("--workdir", type=Path, default=None,
                    help="staging dir for input/output PNGs (default: tempdir, "
                         "deleted on exit unless --keep)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the PNG staging dir after predict (debugging)")
    ap.add_argument("--stem", default="frame",
                    help="case-name prefix (default 'frame' -> "
                         "frame_0000_0000.png)")
    args = ap.parse_args()

    # Resolve plans from --resenc.
    plans = args.plans
    if args.resenc:
        plans = plans or f"nnUNetResEncUNet{args.resenc}Plans"

    # Resolve nnU-Net paths.
    raw, pre, res = args.raw, args.preprocessed, args.results
    if args.base is not None:
        raw = raw or args.base / "nnUNet_raw"
        pre = pre or args.base / "nnUNet_preprocessed"
        res = res or args.base / "nnUNet_results"
    missing = [n for n, v in (("raw", raw), ("preprocessed", pre),
                              ("results", res)) if v is None]
    if missing:
        sys.exit(f"error: nnUNet paths not set: {missing}. Use --base or "
                 f"--raw/--preprocessed/--results or the env vars.")
    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(pre)
    env["nnUNet_results"] = str(res)
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if not args.tiff.exists():
        sys.exit(f"error: input TIFF not found: {args.tiff}")

    # Read TIFF stack.
    print(f"Reading {args.tiff} ...")
    stack = tifffile.imread(str(args.tiff))
    if stack.ndim == 2:
        stack = stack[None]
    elif stack.ndim == 3 and stack.shape[-1] in (3, 4):
        sys.exit(f"error: RGB(A) input not supported (shape {stack.shape}); "
                 "convert to grayscale first.")
    elif stack.ndim != 3:
        sys.exit(f"error: expected 2D frame or 3D stack, got shape {stack.shape}")
    n, h, w = stack.shape
    print(f"  {n} frames, {h}x{w}, dtype={stack.dtype}")

    # Staging dirs.
    if args.workdir is not None:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workdir = Path(tempfile.mkdtemp(prefix="nnunet_pred_"))
        cleanup = not args.keep
    in_dir = workdir / "in"
    out_dir = workdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    width = max(4, len(str(max(n - 1, 1))))
    case_names = []
    print(f"Writing {n} PNG frames to {in_dir} "
          f"(norm={args.norm}, invert={args.invert}) ...")
    for i in range(n):
        u8 = to_uint8(stack[i], args.norm, args.lo, args.hi)
        if args.invert:
            u8 = (255 - u8).astype(np.uint8)
        case = f"{args.stem}_{i:0{width}d}"
        case_names.append(case)
        iio.imwrite(str(in_dir / f"{case}_0000.png"), u8)

    # Run nnUNetv2_predict.
    cmd = ["nnUNetv2_predict",
           "-i", str(in_dir), "-o", str(out_dir),
           "-d", str(args.dataset_id),
           "-c", args.config,
           "-f", str(args.fold),
           "-chk", args.chk]
    if args.trainer:
        cmd += ["-tr", args.trainer]
    if plans:
        cmd += ["-p", plans]
    print()
    run(cmd, env)

    # Reassemble label PNGs into a stack.
    print(f"\nReading predictions from {out_dir} ...")
    pred = np.zeros((n, h, w), dtype=np.uint8)
    missing_preds: list[str] = []
    for i, case in enumerate(case_names):
        p = out_dir / f"{case}.png"
        if not p.exists():
            missing_preds.append(p.name)
            continue
        arr = iio.imread(str(p))
        if arr.ndim != 2:
            sys.exit(f"error: prediction {p} not 2D (shape {arr.shape})")
        if arr.shape != (h, w):
            sys.exit(f"error: prediction {p} shape {arr.shape} != input {(h, w)}")
        pred[i] = arr.astype(np.uint8)
    if missing_preds:
        sys.exit(f"error: {len(missing_preds)} predictions missing, "
                 f"first: {missing_preds[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(args.out), pred, compression="zlib")
    classes = np.unique(pred).tolist()
    print(f"Wrote {args.out}  shape={pred.shape}  classes={classes}")

    if cleanup:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"(staging kept at {workdir})")


if __name__ == "__main__":
    main()
