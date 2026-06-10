#!/usr/bin/env python3
"""Dump per-node confidence map heatmaps from a trained sleap-nn bottomup model.

Fixes from previous version:
  - Handles batched returns (results may contain multiple frames stacked).
  - Uses imshow `extent` so confmaps align to image coordinates regardless
    of confmap output stride.
  - Prints actual tensor shapes for sanity checking.
  - `frames` arg is ignored for .slp inputs in sleap-nn; we process the
    full file and pick the requested frame from the result.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio


def find_confmaps(entry):
    if not isinstance(entry, dict):
        return None, None
    confmap_keys = [k for k in entry.keys()
                    if "conf" in k.lower() and "paf" not in k.lower()]
    for k in confmap_keys:
        return k, entry[k]
    for k, v in entry.items():
        if hasattr(v, "shape") and len(v.shape) >= 3 and "paf" not in k.lower():
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--slp", required=True, type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from sleap_nn.predict import run_inference

    print(f"Note: `frames` is ignored for .slp inputs; will pick frame "
          f"{args.frame} from the result.")
    results = run_inference(
        data_path=str(args.slp),
        model_paths=[str(args.model_dir)],
        make_labels=False,
        return_confmaps=True,
        batch_size=1,
    )
    print(f"results type: {type(results).__name__}, "
          f"len={len(results) if hasattr(results, '__len__') else 'N/A'}")

    entries = results if isinstance(results, list) else [results]

    skeleton_n = len(sio.load_file(str(args.slp)).skeletons[0].nodes)

    def to_nodes_HW(t):
        """Normalize to (n_nodes, H, W) or (batch, n_nodes, H, W)."""
        if t.ndim == 4:
            if t.shape[1] == skeleton_n:
                return t
            if t.shape[-1] == skeleton_n:
                return np.moveaxis(t, -1, 1)
        elif t.ndim == 3:
            if t.shape[0] == skeleton_n:
                return t
            if t.shape[-1] == skeleton_n:
                return np.moveaxis(t, -1, 0)
        return t

    per_frame = []
    key = None
    for entry in entries:
        k, raw = find_confmaps(entry)
        if raw is None:
            continue
        if key is None:
            key = k
        if hasattr(raw, "cpu"):
            raw = raw.cpu().numpy()
        raw = np.asarray(raw)
        t = to_nodes_HW(raw)
        if t.ndim == 4:
            for b in range(t.shape[0]):
                per_frame.append(t[b])
        elif t.ndim == 3:
            per_frame.append(t)

    print(f"Confmaps under key '{key}', {len(per_frame)} frames extracted")
    if not per_frame:
        print("No confmaps found.")
        return
    if args.frame >= len(per_frame):
        print(f"Requested frame {args.frame} > available {len(per_frame)-1}; "
              f"clamping.")
        args.frame = len(per_frame) - 1
    confmaps = per_frame[args.frame]
    print(f"Selected frame {args.frame}, confmap shape: {confmaps.shape}")

    labels = sio.load_file(str(args.slp))
    lf = labels.labeled_frames[args.frame]
    img = lf.image
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    H_img, W_img = img.shape[:2]
    H_cm, W_cm = confmaps.shape[-2:]
    print(f"Image: {H_img}x{W_img}, Confmap: {H_cm}x{W_cm}, "
          f"stride ratio: {H_img / H_cm:.2f}")

    skeleton = labels.skeletons[0]
    node_names = [n.name for n in skeleton.nodes]
    gt_per_node = {name: [] for name in node_names}
    for inst in lf.instances:
        for node, pt in zip(skeleton.nodes, inst.numpy()):
            if not np.isnan(pt).any():
                gt_per_node[node.name].append(pt)

    n = confmaps.shape[0]
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8), constrained_layout=True)
    axes = axes.flatten()
    for i in range(n):
        ax = axes[i]
        # CRITICAL: extent makes the confmap align with image coordinates
        # regardless of stride mismatch
        ax.imshow(img, cmap="gray", extent=(0, W_img, H_img, 0))
        ax.imshow(confmaps[i], cmap="hot", alpha=0.55,
                  vmin=0, vmax=max(confmaps[i].max(), 0.05),
                  extent=(0, W_img, H_img, 0))
        name = node_names[i] if i < len(node_names) else f"node_{i}"
        gt_pts = gt_per_node.get(name, [])
        if gt_pts:
            gt_arr = np.array(gt_pts)
            ax.scatter(gt_arr[:, 0], gt_arr[:, 1], s=30,
                       facecolors='none', edgecolors='cyan', linewidths=1.5)
        ax.set_xlim(0, W_img); ax.set_ylim(H_img, 0)
        ax.set_title(f"{name}  (max={confmaps[i].max():.3f}, n_gt={len(gt_pts)})")
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(n, len(axes)):
        axes[i].axis("off")

    out_path = args.out / f"confmaps_frame{args.frame}.png"
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()