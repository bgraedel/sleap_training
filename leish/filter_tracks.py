#!/usr/bin/env python3
"""Filter SLEAP tracks by minimum length.

Removes any track shorter than `--min-length` frames from a `.slp` file,
along with all instances belonging to those tracks. Useful for cleaning up
noisy tracking output where short fragments are usually false positives or
broken associations from a tracking failure.

Usage:
  python filter_tracks.py predictions.slp -o filtered.slp -n 30
  python filter_tracks.py predictions.slp -o filtered.slp -n 30 --keep-untracked

By default, instances with no track assignment (track=None, e.g. raw
predictions before tracking was run) are dropped along with the short
tracks. Pass --keep-untracked to preserve them.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import sleap_io as sio


def filter_tracks(labels_in: Path, labels_out: Path,
                  min_length: int,
                  keep_untracked: bool = False,
                  top_n: int | None = None) -> None:
    """Filter tracks by minimum length, write filtered Labels to disk.

    Args:
        labels_in: input .slp path.
        labels_out: output .slp path.
        min_length: minimum number of frames a track must have to be kept.
        keep_untracked: if True, also keep instances with `track=None`.
                        Default is to drop them.
        top_n: if given, after the min-length filter, also restrict to the
               top-N longest tracks.
    """
    print(f"Loading {labels_in}...")
    labels = sio.load_file(str(labels_in))
    print(f"  videos:         {len(labels.videos)}")
    print(f"  labeled_frames: {len(labels.labeled_frames)}")
    print(f"  total tracks:   {len(labels.tracks)}")

    # Count number of frames each track appears in
    track_counts: dict = defaultdict(int)
    n_untracked = 0
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            if inst.track is None:
                n_untracked += 1
            else:
                track_counts[inst.track] += 1

    print(f"  total instances: "
          f"{sum(track_counts.values()) + n_untracked}  "
          f"(tracked: {sum(track_counts.values())}, "
          f"untracked: {n_untracked})")

    # Filter
    keep_tracks = {t for t, n in track_counts.items() if n >= min_length}
    if top_n is not None:
        # Sort by length descending and take top N
        sorted_tracks = sorted(track_counts.items(), key=lambda kv: -kv[1])
        keep_tracks &= {t for t, _ in sorted_tracks[:top_n]}

    dropped = set(track_counts) - keep_tracks
    print(f"\nFilter: min_length={min_length}"
          + (f", top_n={top_n}" if top_n is not None else ""))
    print(f"  tracks kept:    {len(keep_tracks)}")
    print(f"  tracks dropped: {len(dropped)}")

    if keep_tracks:
        kept_lengths = sorted(
            (track_counts[t] for t in keep_tracks), reverse=True)
        print(f"  kept track lengths: min={kept_lengths[-1]} "
              f"max={kept_lengths[0]} median={kept_lengths[len(kept_lengths) // 2]}")

    # Walk frames and drop unwanted instances
    n_before = sum(len(lf.instances) for lf in labels.labeled_frames)
    for lf in labels.labeled_frames:
        new_instances = []
        for inst in lf.instances:
            if inst.track is None:
                if keep_untracked:
                    new_instances.append(inst)
            elif inst.track in keep_tracks:
                new_instances.append(inst)
        lf.instances = new_instances
    n_after = sum(len(lf.instances) for lf in labels.labeled_frames)
    print(f"\ninstances:        {n_before} -> {n_after}")

    # Drop empty LabeledFrames (no instances remaining)
    n_lf_before = len(labels.labeled_frames)
    labels.labeled_frames = [lf for lf in labels.labeled_frames if lf.instances]
    n_lf_after = len(labels.labeled_frames)
    print(f"labeled_frames:   {n_lf_before} -> {n_lf_after}")

    # Update the tracks list (sleap-io stores it as a list of Track objects)
    labels.tracks = [t for t in labels.tracks if t in keep_tracks]
    print(f"tracks:           {len(track_counts)} -> {len(labels.tracks)}")

    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels.save(str(labels_out))
    print(f"\nWrote {labels_out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="input .slp with tracks")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="output .slp path")
    ap.add_argument("-n", "--min-length", type=int, default=30,
                    help="minimum track length in frames (default: 30)")
    ap.add_argument("--top-n", type=int, default=None,
                    help="after min-length filter, keep only the N longest "
                         "tracks (default: keep all qualifying tracks)")
    ap.add_argument("--keep-untracked", action="store_true",
                    help="keep instances with no track assignment "
                         "(default: drop them too)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    filter_tracks(args.input, args.output,
                  min_length=args.min_length,
                  keep_untracked=args.keep_untracked,
                  top_n=args.top_n)


if __name__ == "__main__":
    main()
