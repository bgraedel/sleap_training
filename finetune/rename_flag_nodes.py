#!/usr/bin/env python3
"""Rename Flag_N nodes to FlagN in a SLEAP .slp (or .pkg.slp) file.

Loads the file, walks the skeleton(s), renames any node matching "Flag_<digit>"
to "Flag<digit>", and writes the result. Embedded frames in .pkg.slp files
are preserved.

Usage:
  python rename_flag_nodes.py /path/to/labels.v003.pkg.slp

Output is written to <input>.renamed.slp (does not overwrite the original).
Pass --inplace to overwrite the original instead.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import sleap_io as sio


def rename_nodes(path: Path, inplace: bool = False) -> Path:
    print(f"Loading {path}...")
    labels = sio.load_file(str(path))

    print(f"  skeletons: {len(labels.skeletons)}")
    for i, skel in enumerate(labels.skeletons):
        print(f"  skeleton {i} nodes (before): {[n.name for n in skel.nodes]}")

    n_renamed = 0
    for skel in labels.skeletons:
        for node in skel.nodes:
            # Match exactly "Flag_<digit(s)>" — preserves any other node name
            if node.name.startswith("Flag_") and node.name[5:].isdigit():
                old = node.name
                node.name = "Flag" + node.name[5:]
                print(f"  renamed: {old}  ->  {node.name}")
                n_renamed += 1

    print(f"Total renamed: {n_renamed}")

    for i, skel in enumerate(labels.skeletons):
        print(f"  skeleton {i} nodes (after): {[n.name for n in skel.nodes]}")

    if inplace:
        out_path = path
    else:
        # Insert ".renamed" right before the final .slp:
        #   labels.v003.pkg.slp  ->  labels.v003.pkg.renamed.slp
        # More intuitive than putting .renamed before all suffixes.
        out_path = path.parent / f"{path.stem}.renamed{path.suffix}"

    # Pass `embed="user"` to copy only the user-labeled frames' embedded
    # data into the new file. NOT "all" or "source":
    #   - "source" tries to re-read from the original video file path stored
    #     in the .pkg.slp (e.g. a Windows path on a different machine) -> fails.
    #   - "all" tries to embed every frame the Video object reports having
    #     (often the full source video's frame count, e.g. 560), but only the
    #     subset actually saved by SLEAP-GUI is present in the embedded data,
    #     so it crashes with IndexError at the first non-embedded index.
    #   - "user" matches exactly what SLEAP-GUI's default save behavior does:
    #     embeds only frames with user instances, which is what the source
    #     .pkg.slp already had.
    print(f"Saving {out_path} (re-embedding user-labeled frames)...")
    labels.save(str(out_path), embed="user")
    print("Done.")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="input .slp or .pkg.slp")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the input file instead of writing "
                         "<input>.renamed.slp (default: write a new file)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    rename_nodes(args.input, inplace=args.inplace)


if __name__ == "__main__":
    main()