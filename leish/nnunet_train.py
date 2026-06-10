"""
Thin orchestrator for nnU-Net v2 2D training on the converted Leishmania
body/flagellum dataset.

It sets the three required nnU-Net environment variables and runs
`nnUNetv2_plan_and_preprocess` -> `nnUNetv2_train` (and optionally
`nnUNetv2_predict`) for you, printing each underlying command so the raw CLI is
always visible. nnU-Net itself must already be installed (`pip install nnunetv2`)
and its console scripts on PATH.

Directory convention: pass `--base DIR` and the three nnU-Net folders are placed
under it (DIR/nnUNet_raw, nnUNet_preprocessed, nnUNet_results), or set each
explicitly with --raw/--preprocessed/--results (or the matching env vars).

Examples
--------
    # one model, no held-out CV fold ('all'), full preprocess + train:
    python nnunet_train.py --base /scratch/nnunet --dataset-id 501 --fold all

    # shorter schedule + a specific GPU, skip re-preprocessing:
    python nnunet_train.py --base /scratch/nnunet --dataset-id 501 --fold 0 \\
        --trainer nnUNetTrainer_250epochs --device 0 --skip-preprocess

    # train then tile-free inference on a folder of <case>_0000.png images:
    python nnunet_train.py --base /scratch/nnunet --dataset-id 501 --fold all \\
        --predict /data/test_in /data/test_out

Note: the dataset must already be converted into $nnUNet_raw with
`nnunet_convert.py`. For large real frames, predict per 640 tile and stitch
(average logits in the overlaps) rather than feeding the whole frame.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=None,
                    help="base dir; uses <base>/nnUNet_raw, _preprocessed, _results")
    ap.add_argument("--raw", type=Path, default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--preprocessed", type=Path,
                    default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--results", type=Path,
                    default=os.environ.get("nnUNet_results"))
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--config", default="2d",
                    help="nnU-Net configuration (default: 2d)")
    ap.add_argument("--fold", default="all",
                    help="'all' (single model, no held-out fold) or 0-4")
    ap.add_argument("--trainer", default=None,
                    help="trainer variant, e.g. nnUNetTrainer_250epochs "
                         "(default: nnU-Net default = 1000 epochs)")
    ap.add_argument("--resenc", choices=["M", "L", "XL"], default=None,
                    help="use a Residual Encoder preset (sets --planner and "
                         "--plans to the matching pair). M is plenty for 2D and "
                         "fits small GPUs; L is nnU-Net's new recommended default.")
    ap.add_argument("--planner", default=None,
                    help="experiment planner for plan_and_preprocess (-pl), "
                         "e.g. nnUNetPlannerResEncM")
    ap.add_argument("--plans", default=None,
                    help="plans identifier for train/predict (-p), "
                         "e.g. nnUNetResEncUNetMPlans")
    ap.add_argument("--device", default=None,
                    help="value for CUDA_VISIBLE_DEVICES, e.g. '0'")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="skip nnUNetv2_plan_and_preprocess (already done)")
    ap.add_argument("--npz", action="store_true",
                    help="pass --npz to training (needed for later ensembling)")
    ap.add_argument("--predict", nargs=2, metavar=("IN", "OUT"), default=None,
                    help="after training, run nnUNetv2_predict IN -> OUT")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to nnUNetv2_train")
    args = ap.parse_args()

    raw, pre, res = args.raw, args.preprocessed, args.results
    if args.base is not None:
        raw = raw or args.base / "nnUNet_raw"
        pre = pre or args.base / "nnUNet_preprocessed"
        res = res or args.base / "nnUNet_results"
    missing = [n for n, v in (("raw", raw), ("preprocessed", pre), ("results", res))
               if v is None]
    if missing:
        sys.exit(f"error: nnUNet paths not set: {missing}. Use --base or "
                 f"--raw/--preprocessed/--results or the env vars.")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(pre)
    env["nnUNet_results"] = str(res)
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    Path(pre).mkdir(parents=True, exist_ok=True)
    Path(res).mkdir(parents=True, exist_ok=True)

    print(f"nnUNet_raw          = {raw}")
    print(f"nnUNet_preprocessed = {pre}")
    print(f"nnUNet_results      = {res}\n")

    # Residual Encoder preset: --resenc M/L/XL fills in the matching planner +
    # plans names unless they were given explicitly.
    planner, plans = args.planner, args.plans
    if args.resenc:
        planner = planner or f"nnUNetPlannerResEnc{args.resenc}"
        plans = plans or f"nnUNetResEncUNet{args.resenc}Plans"

    if not args.skip_preprocess:
        plan_cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id),
                    "-c", args.config, "--verify_dataset_integrity"]
        if planner:
            plan_cmd += ["-pl", planner]
        run(plan_cmd, env)

    train_cmd = ["nnUNetv2_train", str(args.dataset_id), args.config, str(args.fold)]
    if args.trainer:
        train_cmd += ["-tr", args.trainer]
    if plans:
        train_cmd += ["-p", plans]
    if args.npz:
        train_cmd += ["--npz"]
    train_cmd += list(args.extra)
    run(train_cmd, env)

    if args.predict:
        in_dir, out_dir = args.predict
        predict_cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir,
                       "-d", str(args.dataset_id), "-c", args.config,
                       "-f", str(args.fold)]
        if plans:
            predict_cmd += ["-p", plans]
        run(predict_cmd, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
