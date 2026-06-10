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
    slps = sorted(root.rglob("*.slp"))
    if not slps:
        sys.exit(f"No labels.slp files found under {root}")
    print(f"# {len(slps)} labels.slp files found under {root}")
    print("# Paste under data_config.train_labels_path in your sleap-nn config:")
    print()
    print("train_labels_path:")
    for p in slps:
        print(f"  - {p.resolve()}")


def cmd_merge(root: Path, output: Path) -> None:
    """Concatenate 64+ independent labels.slp files into one.

    These sub-datasets share no videos, no frames, and identical skeletons,
    so smart-merge with conflict resolution (Labels.merge) is wasted work
    and grows quadratically with the accumulator size. We just concatenate
    the underlying lists directly, which is O(N) and ~100x faster.
    """
    import sleap_io as sio

    slps = sorted(root.rglob("*.slp"))
    if not slps:
        sys.exit(f"No labels.slp files found under {root}")

    print(f"Concatenating {len(slps)} labels.slp files...")
    all_videos = []
    all_frames = []
    all_skeletons = []

    for i, slp_path in enumerate(slps, 1):
        labels = sio.load_file(str(slp_path))

        # Disambiguate videos: each sub-dataset has its own video.tif.
        # Resolve to absolute paths so the files stay distinct identifiers.
        for video in labels.videos:
            vp = Path(video.filename)
            if not vp.is_absolute():
                video.filename = str((slp_path.parent / vp).resolve())

        all_videos.extend(labels.videos)
        all_frames.extend(labels.labeled_frames)
        all_skeletons.extend(labels.skeletons)

        if i % 10 == 0 or i == len(slps):
            print(f"  ...loaded {i}/{len(slps)} ({len(all_frames)} frames so far)")

    # Deduplicate skeletons by structure (nodes + edges). sleap-io's
    # Skeleton.__eq__ defaults to identity, so two skeletons loaded from
    # separate files won't match even when they have identical nodes/edges.
    # Compare on node names + edge indices instead, and rebind each instance
    # to the canonical skeleton so the final file has exactly one.
    def _skel_key(sk):
        return (tuple(n.name for n in sk.nodes), tuple(sk.edge_inds))

    canonical: dict = {}
    for sk in all_skeletons:
        canonical.setdefault(_skel_key(sk), sk)
    unique_skeletons = list(canonical.values())

    # Rebind every instance to the canonical skeleton with matching structure.
    for lf in all_frames:
        for inst in lf.instances:
            key = _skel_key(inst.skeleton)
            if key in canonical:
                inst.skeleton = canonical[key]

    print(f"Building concatenated Labels...")
    merged = sio.Labels(
        videos=all_videos,
        skeletons=unique_skeletons,
        labeled_frames=all_frames,
    )

    print(f"Saving: {output}")
    merged.save(str(output))

    print()
    print(f"Concatenated result:")
    print(f"  videos:          {len(merged.videos)}")
    print(f"  skeletons:       {len(merged.skeletons)}")
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
