#!/usr/bin/env python3
"""Convert each video.tif to video.mp4 and update labels.slp paths in place.

Background:
    sleap-nn startup is hours-slow on this dataset because each video.tif is
    parsed as a pyramidal TIFF and tifffile.pyramidize_series dominates the
    profile. MP4 opens are O(ms), so swapping the format kills the bottleneck
    while keeping frame indices (and therefore existing labels) valid 1:1.

What this does, per <root>/*/labels.slp + video.tif pair:
    1. Read every frame from video.tif (taking the base level if pyramidal).
    2. Normalize to uint8 if needed (H.264 is an 8-bit codec).
    3. Encode to video.mp4 with x264 (CRF 15, yuv444p by default).
    4. Update labels.slp to reference the new MP4. Loaded with open_videos=False
       so we don't error on the old TIFF path during the rewrite.

Frame indices are preserved 1:1, so existing labeled_frames remain valid.

Usage:
    python convert_tiffs_to_mp4.py data/leishmania_pretrain
    python convert_tiffs_to_mp4.py data/leishmania_pretrain --dry-run
    python convert_tiffs_to_mp4.py data/leishmania_pretrain --crf 12
    python convert_tiffs_to_mp4.py data/leishmania_pretrain --skip-existing
    python convert_tiffs_to_mp4.py data/leishmania_pretrain --delete-tiff

After running, your existing prepare_for_sleap_nn.py list mode should be fast:
    python prepare_for_sleap_nn.py list data/leishmania_pretrain
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def read_tiff(tiff_path: Path) -> np.ndarray:
    """Read every page from a multi-page TIFF as (T, H, W) or (T, H, W, C).

    NOTE on series vs pages:
        TIFFs written by calling tifffile.TiffWriter.write() once per frame
        (without contiguous=True or passing a 3D array in one shot) end up
        with one tifffile-"series" per page rather than one series of N pages.
        tif.series[0].asarray() would therefore return only the FIRST frame.
        We iterate tif.pages directly, which always sees every page regardless
        of how series are grouped. This also handles pyramidal TIFFs by simply
        reading whatever pages exist; if a base+downsampled pyramid is present
        you'd want to filter by shape, but the synthetic Leishmania writer
        produces flat multi-page grayscale, so all pages have the same shape.
    """
    import tifffile
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_pages = len(tif.pages)
        n_series = len(tif.series)
        print(f"    TIFF inventory: {n_pages} pages across {n_series} series")
        if n_pages == 0:
            raise ValueError(f"No pages in {tiff_path}")
        first = tif.pages[0].asarray()
        if n_pages == 1:
            return first[None] if first.ndim == 2 else first
        # Guard against mixed-shape pages (would indicate a pyramid or thumbnails).
        shapes = {tif.pages[i].shape for i in range(n_pages)}
        if len(shapes) > 1:
            # Keep only pages matching the most common shape (= main video).
            from collections import Counter
            target_shape = Counter(
                tif.pages[i].shape for i in range(n_pages)
            ).most_common(1)[0][0]
            keep = [i for i in range(n_pages) if tif.pages[i].shape == target_shape]
            print(f"    mixed page shapes {shapes}; keeping {len(keep)} pages "
                  f"of shape {target_shape}")
        else:
            keep = list(range(n_pages))
        out_shape = (len(keep),) + first.shape
        arr = np.empty(out_shape, dtype=first.dtype)
        for out_i, page_i in enumerate(keep):
            arr[out_i] = tif.pages[page_i].asarray()
        return arr


def count_mp4_frames(mp4_path: Path) -> int:
    """Count actual frames in an MP4 — defensive check against silent truncation."""
    import imageio.v2 as iio2
    reader = iio2.get_reader(str(mp4_path))
    try:
        try:
            n = reader.count_frames()
            if n is None or n < 0:
                raise ValueError
            return int(n)
        except Exception:
            # Fallback: iterate. Slow but reliable.
            return sum(1 for _ in reader)
    finally:
        reader.close()


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-video min-max normalize to uint8, unless already uint8.

    If your synthetic intensity range is consistent across videos, consider
    replacing this with a fixed (lo, hi) to keep brightness comparable.
    """
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    return (scaled * 255.0).round().clip(0, 255).astype(np.uint8)


def write_mp4(frames: np.ndarray, out_path: Path,
              crf: int, pix_fmt: str, fps: int) -> None:
    """Encode (T, H, W) or (T, H, W, C) uint8 frames to MP4 via imageio-ffmpeg.

    macro_block_size=1 prevents auto-padding of dimensions, which would shift
    pixel coordinates and break existing labels.
    """
    import imageio
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        pixelformat=pix_fmt,
        ffmpeg_params=["-crf", str(crf), "-preset", "slow"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def update_slp_paths(slp_path: Path, tiff_name: str, mp4_abs_path: Path) -> bool:
    """Rewrite any video.filename whose basename matches tiff_name to the new MP4 path."""
    import sleap_io as sio
    labels = sio.load_file(str(slp_path), open_videos=False)
    new_path = str(mp4_abs_path.resolve())
    changed = False
    for video in labels.videos:
        if Path(video.filename).name != tiff_name:
            continue
        # Prefer the canonical sleap-io method if available; fall back to direct assign.
        if hasattr(video, "replace_filename"):
            try:
                video.replace_filename(new_path, open=False)
            except TypeError:
                video.replace_filename(new_path)
        else:
            video.filename = new_path
            # Clear cached backend so MP4 backend is re-detected on next open.
            for attr in ("backend", "backend_metadata"):
                if hasattr(video, attr):
                    try:
                        setattr(video, attr, None if attr == "backend" else {})
                    except Exception:
                        pass
        changed = True
    if changed:
        labels.save(str(slp_path), embed=False)
    return changed


def convert_one(setup_dir: Path, tiff_name: str, mp4_name: str,
                crf: int, pix_fmt: str, fps: int,
                skip_existing: bool, dry_run: bool) -> tuple[bool, str]:
    tiff_path = setup_dir / tiff_name
    mp4_path = setup_dir / mp4_name
    slp_path = setup_dir / "labels.slp"

    if not slp_path.exists():
        return False, "no labels.slp"
    if not tiff_path.exists() and not (mp4_path.exists() and skip_existing):
        return False, f"no {tiff_name}"

    if dry_run:
        print(f"    DRY RUN: would write {mp4_path} and rewrite {slp_path}")
        return True, "dry-run"

    if mp4_path.exists() and skip_existing:
        print(f"    MP4 exists, skipping encode")
    else:
        t = time.time()
        frames = read_tiff(tiff_path)
        print(f"    read TIFF: shape={frames.shape} dtype={frames.dtype} "
              f"({time.time()-t:.1f}s)")
        if frames.dtype != np.uint8:
            t = time.time()
            frames = to_uint8(frames)
            print(f"    normalized to uint8 ({time.time()-t:.1f}s)")
        # Sanity check on dimensions for yuv420p (requires even H and W).
        if pix_fmt == "yuv420p" and (frames.shape[1] % 2 or frames.shape[2] % 2):
            return False, (f"yuv420p needs even H,W but got {frames.shape[1:3]}; "
                           "use --pix-fmt yuv444p")
        t = time.time()
        write_mp4(frames, mp4_path, crf=crf, pix_fmt=pix_fmt, fps=fps)
        size_mb = mp4_path.stat().st_size / 1e6
        print(f"    wrote MP4: {size_mb:.1f} MB ({time.time()-t:.1f}s)")

        # Verify the MP4 actually contains every frame we wrote. A mismatch
        # here means existing labeled_frames with frame_idx >= n_actual would
        # silently fail to load at training time.
        n_in = len(frames)
        n_out = count_mp4_frames(mp4_path)
        if n_out != n_in:
            mp4_path.unlink(missing_ok=True)
            return False, (f"MP4 frame count mismatch: input had {n_in} frames "
                           f"but encoded MP4 has {n_out}; refusing to update .slp")
        print(f"    verified MP4 frame count: {n_out}")

    t = time.time()
    changed = update_slp_paths(slp_path, tiff_name, mp4_path)
    if changed:
        print(f"    updated paths in labels.slp ({time.time()-t:.1f}s)")
    else:
        print(f"    no matching video paths in labels.slp")
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root", type=Path,
                    help="Root directory containing setup subfolders with labels.slp + video.tif")
    ap.add_argument("--tiff-name", default="video.tif",
                    help="TIFF filename inside each setup dir (default: video.tif)")
    ap.add_argument("--mp4-name", default="video.mp4",
                    help="MP4 filename to write (default: video.mp4)")
    ap.add_argument("--crf", type=int, default=0,
                    help="x264 CRF. Lower=better; 0=lossless. Default 15 ≈ near-lossless.")
    ap.add_argument("--pix-fmt", default="yuv444p",
                    help="ffmpeg pixel format. yuv444p (default) avoids chroma "
                         "subsampling and works with odd-sized frames. "
                         "Use yuv420p for max player compatibility (requires even H,W).")
    ap.add_argument("--fps", type=int, default=30,
                    help="FPS metadata for the MP4 (default: 30). Does not affect "
                         "frame ordering or training.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't re-encode if video.mp4 already exists "
                         "(still updates the .slp paths)")
    ap.add_argument("--delete-tiff", action="store_true",
                    help="Delete the original TIFF after a successful conversion + .slp rewrite")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be done without doing it")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"Not a directory: {args.root}")

    slps = sorted(args.root.rglob("labels.slp"))
    if not slps:
        sys.exit(f"No labels.slp files found under {args.root}")
    print(f"Found {len(slps)} labels.slp files under {args.root}")

    failures: list[tuple[Path, str]] = []
    t_overall = time.time()
    for i, slp in enumerate(slps, 1):
        setup = slp.parent
        print(f"\n[{i}/{len(slps)}] {setup}")
        try:
            ok, msg = convert_one(
                setup, args.tiff_name, args.mp4_name,
                crf=args.crf, pix_fmt=args.pix_fmt, fps=args.fps,
                skip_existing=args.skip_existing, dry_run=args.dry_run,
            )
            if not ok:
                print(f"    SKIP/FAIL: {msg}")
                failures.append((setup, msg))
                continue
            if args.delete_tiff and not args.dry_run:
                tiff_path = setup / args.tiff_name
                mp4_path = setup / args.mp4_name
                if tiff_path.exists() and mp4_path.exists():
                    tiff_path.unlink()
                    print(f"    deleted {tiff_path.name}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            failures.append((setup, f"{type(e).__name__}: {e}"))

    print()
    print(f"Done in {time.time()-t_overall:.1f}s. "
          f"{len(slps) - len(failures)}/{len(slps)} succeeded.")
    if failures:
        print("\nFailures:")
        for path, reason in failures:
            print(f"  {path}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
