"""
Convert a `segmentation_dataset_creator` output into an nnU-Net v2 raw dataset
for 2D semantic segmentation of background / body / flagellum.

The segmentation builder already writes per-pixel label PNGs with exactly the
integer classes nnU-Net wants (0=background, 1=body, 2=flagellum), so this is
essentially a rename + a `dataset.json`:

    <src>/images/<stem>.png   ->  imagesTr/<stem>_0000.png   (1 grayscale channel)
    <src>/masks/<stem>.png    ->  labelsTr/<stem>.png        (values {0,1,2})

nnU-Net v2 does its OWN 5-fold cross-validation split internally, so every frame
(including the all-background negative frames) goes into imagesTr/labelsTr — do
NOT pre-split.

Result:
    $nnUNet_raw/Dataset<ID>_<NAME>/
      imagesTr/   labelsTr/   dataset.json

Usage:
    # env var (or pass --nnunet-raw)
    export nnUNet_raw=/path/nnUNet_raw
    python nnunet_convert.py --src data/leishmania_seg_ds_640 \\
        --dataset-id 501 --dataset-name LeishParts --verify

    # space-saving alternatives on Linux:
    python nnunet_convert.py --src ... --mode symlink     # or --mode hardlink
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as iio


# nnU-Net label map: must be consecutive integers from 0, background = 0.
LABELS = {"background": 0, "body": 1, "flagellum": 2}
# One grayscale input channel (the rendered phase-contrast image).
CHANNEL_NAMES = {"0": "intensity"}
VALID_LABEL_VALUES = set(LABELS.values())


def _place(src: Path, dst: Path, mode: str) -> None:
    """Put `src` at `dst` via copy / symlink / hardlink (idempotent)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        import shutil
        shutil.copy(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"unknown mode {mode!r}")


def convert(src: Path, nnunet_raw: Path, dataset_id: int, dataset_name: str,
            mode: str = "copy", limit: int | None = None,
            verify: bool = False) -> Path:
    images = src / "images"
    masks = src / "masks"
    if not images.is_dir() or not masks.is_dir():
        sys.exit(f"error: {src} must contain images/ and masks/ subfolders")

    ds_dir = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in images.glob("*.png"))
    if limit is not None:
        stems = stems[:limit]
    if not stems:
        sys.exit(f"error: no PNGs found in {images}")

    n = 0
    missing_mask = 0
    bad_label = 0
    for s in stems:
        img = images / f"{s}.png"
        msk = masks / f"{s}.png"
        if not msk.exists():
            missing_mask += 1
            continue
        if verify:
            arr = iio.imread(msk)
            if arr.ndim != 2:
                bad_label += 1
                print(f"  WARN {s}: label is not single-channel (shape {arr.shape})")
                continue
            vals = set(np.unique(arr).tolist())
            if not vals <= VALID_LABEL_VALUES:
                bad_label += 1
                print(f"  WARN {s}: label has unexpected values {sorted(vals)} "
                      f"(allowed {sorted(VALID_LABEL_VALUES)})")
                continue
        _place(img, images_tr / f"{s}_0000.png", mode)
        _place(msk, labels_tr / f"{s}.png", mode)
        n += 1
        if n % 250 == 0:
            print(f"  converted {n} cases...")

    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": LABELS,
        "numTraining": n,
        "file_ending": ".png",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"\nWrote {n} cases to {ds_dir} (mode={mode}).")
    if missing_mask:
        print(f"  skipped {missing_mask} image(s) with no matching mask")
    if bad_label:
        print(f"  skipped {bad_label} image(s) with invalid label values")
    print("\nNext steps:")
    print(f"  export nnUNet_raw={nnunet_raw}")
    print("  export nnUNet_preprocessed=/path/nnUNet_preprocessed")
    print("  export nnUNet_results=/path/nnUNet_results")
    print(f"  nnUNetv2_plan_and_preprocess -d {dataset_id} -c 2d --verify_dataset_integrity")
    print(f"  nnUNetv2_train {dataset_id} 2d all")
    print(f"  (or: python nnunet_train.py --dataset-id {dataset_id} --fold all ...)")
    return ds_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="segmentation dataset dir (containing images/ and masks/)")
    ap.add_argument("--nnunet-raw", type=Path,
                    default=os.environ.get("nnUNet_raw"),
                    help="nnUNet_raw root (default: $nnUNet_raw)")
    ap.add_argument("--dataset-id", type=int, default=501,
                    help="3-digit nnU-Net dataset id (default: 501)")
    ap.add_argument("--dataset-name", default="LeishParts",
                    help="dataset name suffix (default: LeishParts)")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"],
                    default="copy",
                    help="how to place files (default: copy; symlink/hardlink "
                         "save space on Linux/same-filesystem)")
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N cases (debugging)")
    ap.add_argument("--verify", action="store_true",
                    help="check every label is single-channel with values in {0,1,2}")
    args = ap.parse_args()

    if args.nnunet_raw is None:
        sys.exit("error: set $nnUNet_raw or pass --nnunet-raw")
    if not (1 <= args.dataset_id <= 999):
        sys.exit("error: --dataset-id must be in [1, 999]")

    convert(args.src, Path(args.nnunet_raw), args.dataset_id, args.dataset_name,
            mode=args.mode, limit=args.limit, verify=args.verify)


if __name__ == "__main__":
    main()
