#!/usr/bin/env python3
"""Diagnose video-reference collisions across multiple labels.slp files.

When sleap-nn loads many labels.slp files, the resulting combined Labels
object may deduplicate Video objects by filename. If two source .slp files
reference videos with the same filename, their LabeledFrames will all
point at one merged Video — and any sharing the same frame_idx will then
appear as duplicates.

This script loads every .slp listed (one per line in a file, or directly
on the command line) and prints:
  - The video filenames each .slp references
  - Any collisions between source files
  - Max frame_idx per video (to spot out-of-range references)

Usage:
  python diagnose_multi_slp.py file1.slp file2.slp ...
  OR
  python diagnose_multi_slp.py --paths-file paths.txt
  OR (extract from a sleap-nn YAML config)
  python diagnose_multi_slp.py --from-yaml config.yaml
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import sleap_io as sio


def load_paths(args):
    paths = list(args.slps)
    if args.paths_file:
        for line in Path(args.paths_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    if args.from_yaml:
        import yaml
        cfg = yaml.safe_load(Path(args.from_yaml).read_text())
        # try common locations
        loc = cfg.get("data_config", cfg).get("train_labels_path", [])
        paths.extend(loc)
    return [Path(p) for p in paths]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slps", nargs="*", help="Paths to labels.slp files")
    ap.add_argument("--paths-file", help="Text file with one .slp path per line")
    ap.add_argument("--from-yaml", help="Pull data_config.train_labels_path from a YAML")
    args = ap.parse_args()

    slp_paths = load_paths(args)
    if not slp_paths:
        sys.exit("No .slp paths provided. Use --help for options.")

    print(f"Inspecting {len(slp_paths)} source files...")
    print()

    # video_filename -> list of (source_slp, n_frames_in_slp, max_frame_idx_in_slp, video_shape)
    by_video_filename = defaultdict(list)
    bad_max_idx = []  # (source, video_filename, max_idx, vid_len)

    for slp_path in slp_paths:
        if not slp_path.exists():
            print(f"  MISSING: {slp_path}")
            continue
        labels = sio.load_file(str(slp_path))
        # Map video objects to their frame_idx ranges in THIS labels object
        per_video: dict = defaultdict(list)
        for lf in labels.labeled_frames:
            per_video[id(lf.video)].append(lf.frame_idx)
        for vid in labels.videos:
            idxs = per_video.get(id(vid), [])
            vid_len = vid.shape[0] if vid.shape is not None else None
            max_idx = max(idxs) if idxs else None
            by_video_filename[vid.filename].append((
                str(slp_path), len(idxs), max_idx, vid.shape,
            ))
            if max_idx is not None and vid_len is not None and max_idx >= vid_len:
                bad_max_idx.append((str(slp_path), vid.filename, max_idx, vid_len))

    # Report video-filename collisions
    print("=" * 80)
    print("Video filename collisions across source files:")
    print("=" * 80)
    any_collisions = False
    for vid_fn, entries in by_video_filename.items():
        if len(entries) > 1:
            any_collisions = True
            print(f"\n  Video filename: {vid_fn}")
            print(f"  Referenced by {len(entries)} source .slp files:")
            for src, n_lf, max_idx, shape in entries:
                print(f"    {src}: {n_lf} frames, max_idx={max_idx}, shape={shape}")
    if not any_collisions:
        print("  No collisions. Each video filename referenced by exactly one source.")

    # Report out-of-range frame indices found within individual source files
    print()
    print("=" * 80)
    print("Out-of-range frame indices within individual source files:")
    print("=" * 80)
    if not bad_max_idx:
        print("  None.")
    else:
        for src, vid_fn, max_idx, vid_len in bad_max_idx:
            print(f"  {src} -> {vid_fn}: max_idx={max_idx} but vid_len={vid_len}")


if __name__ == "__main__":
    main()
