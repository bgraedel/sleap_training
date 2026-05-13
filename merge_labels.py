#!/usr/bin/env python3
"""Prepare the multi-setup synthetic Leishmania output for sleap-nn training.

Two modes:

  1. list   — print a YAML-formatted list of all labels.slp files. Paste into
              your sleap-nn config under data_config.train_labels_path.

  2. merge  — write a single combined labels.slp containing all 64 sub-datasets.
              Resolves each video to an absolute path before merging so the
              identical 'video.tif' basenames don't collapse into one video.

Usage:
  python prepare_for_sleap_nn.py list  data/leishmania_pretrain
  python prepare_for_sleap_nn.py merge data/leishmania_pretrain -o merged.slp
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def cmd_list(root: Path) -> None:
    slps = sorted(root.rglob("labels.slp"))
    if not slps:
        sys.exit(f"No labels.slp files found under {root}")
    print(f"# {len(slps)} labels.slp files found under {root}")
    print("# Paste under data_config.train_labels_path in your sleap-nn config:")
    print()
    print("train_labels_path:")
    for p in slps:
        print(f"  - {p.resolve()}")


def cmd_merge(root: Path, output: Path) -> None:
    import sleap_io as sio

    slps = sorted(root.rglob("labels.slp"))
    if not slps:
        sys.exit(f"No labels.slp files found under {root}")

    print(f"Merging {len(slps)} labels.slp files...")
    merged = None
    for i, slp_path in enumerate(slps, 1):
        labels = sio.load_file(str(slp_path))

        # Disambiguate videos: each sub-dataset has its own video.tif.
        # Resolve to absolute paths so the merge keeps them as 64 separate
        # videos instead of collapsing them by basename.
        for video in labels.videos:
            vp = Path(video.filename)
            if not vp.is_absolute():
                video.filename = str((slp_path.parent / vp).resolve())

        if merged is None:
            merged = labels
        else:
            # Same merge call sio unsplit uses internally:
            # video='auto' lets sleap-io match by provenance metadata where
            # available (and falls back to path/content), and frame='keep_both'
            # prevents frames at the same index from one video from being
            # collapsed against another's.
            merged.merge(labels, video="auto", frame="keep_both")

        if i % 10 == 0 or i == len(slps):
            print(f"  ...merged {i}/{len(slps)}")

    print(f"Saving merged labels: {output}")
    merged.save(str(output))

    # Quick summary
    print()
    print(f"Merged result:")
    print(f"  videos:          {len(merged.videos)}")
    print(f"  labeled frames:  {len(merged.labeled_frames)}")
    print(f"  total instances: {sum(len(lf.instances) for lf in merged.labeled_frames)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Print YAML-ready list of labels.slp paths")
    p_list.add_argument("root", type=Path,
                        help="Root directory containing setup subfolders")

    p_merge = sub.add_parser("merge",
                             help="Merge all labels.slp files into one")
    p_merge.add_argument("root", type=Path,
                         help="Root directory containing setup subfolders")
    p_merge.add_argument("-o", "--output", type=Path, required=True,
                         help="Output path for the merged .slp")

    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list(args.root)
    elif args.cmd == "merge":
        cmd_merge(args.root, args.output)


if __name__ == "__main__":
    main()
