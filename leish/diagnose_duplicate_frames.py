#!/usr/bin/env python3
"""Diagnose duplicate LabeledFrame entries in a merged labels.slp.

Reports, per video:
  - total LabeledFrame count
  - number of unique frame indices
  - max frame_idx vs the video's actual frame count
  - example duplicate frame_idx values

If the labels file claims frame indices beyond the video's length, that's
either a generation bug, a double-concatenation bug in the merge script,
or the same .slp being included multiple times.

Usage:
  python diagnose_duplicate_frames.py /path/to/merged.slp
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sleap_io as sio


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    slp_path = Path(sys.argv[1])

    print(f"Loading {slp_path}...")
    labels = sio.load_file(str(slp_path))
    print(f"  videos:          {len(labels.videos)}")
    print(f"  labeled frames:  {len(labels.labeled_frames)}")
    print(f"  skeletons:       {len(labels.skeletons)}")
    print()

    # Group LabeledFrames by their backing video (by id, which is what sleap-io uses)
    by_video = defaultdict(list)
    for lf in labels.labeled_frames:
        by_video[id(lf.video)].append(lf)

    # For each video, examine the frame_idx distribution
    print(f"{'video name':45s} {'n_lf':>6s} {'uniq':>6s} {'max':>6s} {'vid_len':>8s} {'dups':>6s}")
    print('-' * 90)

    any_problems = False
    for vid_id, lfs in by_video.items():
        video = lfs[0].video
        video_name = Path(video.filename).parent.name + '/' + Path(video.filename).name
        if len(video_name) > 45:
            video_name = '...' + video_name[-42:]
        vid_len = video.shape[0] if video.shape is not None else None

        idxs = [lf.frame_idx for lf in lfs]
        idx_counts = Counter(idxs)
        n_dups = sum(c - 1 for c in idx_counts.values() if c > 1)
        max_idx = max(idxs) if idxs else None

        marker = ''
        if vid_len is not None and max_idx is not None and max_idx >= vid_len:
            marker = '  ← max_idx OUT OF RANGE'
            any_problems = True
        if n_dups > 0:
            marker += '  ← has duplicates'
            any_problems = True

        vid_len_str = str(vid_len) if vid_len is not None else '?'
        print(f"{video_name:45s} {len(lfs):>6d} {len(idx_counts):>6d} "
              f"{max_idx:>6d} {vid_len_str:>8s} {n_dups:>6d}{marker}")

        # Show worst-offender frame_idx for problematic videos
        if marker:
            worst = sorted(idx_counts.items(), key=lambda x: -x[1])[:5]
            for fi, cnt in worst:
                in_range = "OK" if (vid_len is None or fi < vid_len) else "OUT OF RANGE"
                print(f"    frame_idx={fi:>4d}: {cnt} copies  ({in_range})")
            print()

    print()
    if not any_problems:
        print("No duplicate or out-of-range frame indices found.")
        return

    # If there are problems, summarize
    print("Summary of the problem:")
    print("  - 'has duplicates'    → same (video, frame_idx) appears multiple times")
    print("  - 'max_idx OUT OF RANGE' → label references a frame beyond video length")
    print()
    print("Most likely cause: the source labels.slp for a setup was included more")
    print("than once during merge, OR the dataset generator wrote duplicate frames.")
    print()
    print("If only specific setups (like 100x_detail) are affected, the bug is")
    print("probably in the source .slp files for those setups, not in the merge.")


if __name__ == "__main__":
    main()
