#!/usr/bin/env python3
"""Stage 1 of BUCTD: train the bottom-up (BU) multi-animal pose model.

The BU model produces initial keypoint proposals that the CTD model
(Stage 2, train_dlc_ctd.py) refines. Run this script first.

Workflow performed by this script:
    1. (Optional) sanity-check labels via deeplabcut.check_labels
    2. Create multi-animal training dataset (shuffle N) via DLC
    3. (Optional) patch the generated pytorch_config.yaml to:
         - set the PAF target width (default DLC width is 20px, often too
           wide for small animals — adjacent edges overlap, neighboring
           animals' PAFs interfere in dense scenes)
         - force a chain or chain+skip1 PAF graph (NOT recommended by
           default — DLC's data-driven graph selection during evaluation
           usually beats a hand-specified topology)
         - set `num_animals` from observed post-crop instance density
         - reduce the training crop size
         - bump dataloader_workers
    4. Train the network for the configured number of epochs
    5. (Optional) evaluate on the held-out test split

Usage:
    # Recommended for dense, multi-animal scenes with small subjects:
    # let DLC keep the complete PAF graph (it prunes data-driven during eval),
    # but shrink the PAF target width if your animals are smaller than DLC's
    # 20-pixel default (true for most lab animals at typical crop sizes).
    python train_dlc_bu.py /path/to/config.yaml \\
        --gpus 0,1 --epochs 100 --batch-size 32 \\
        --prune-paf-graph none --paf-width 8 \\
        --num-animals 100 --train-crop-size 384

    # Skip sanity check and evaluation (e.g. for re-runs)
    python train_dlc_bu.py /path/to/config.yaml \\
        --skip-check-labels --skip-evaluate

    # Re-launch on an already-created shuffle (skip dataset creation but
    # still apply the patch, e.g. to change crop size mid-experiment)
    python train_dlc_bu.py /path/to/config.yaml --shuffle 2 \\
        --skip-create-dataset --train-crop-size 320

After this completes, run train_dlc_ctd.py to train the CTD stage.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path


SUGGESTED_BU_NETS = (
    "hrnet_w32",            # default — balanced
    "hrnet_w48",            # higher capacity, slower
    "dlcrnet_stride32_ms5", # DLC's multi-animal classic — faster than HRNet
    "resnet_50",            # baseline
    "resnet_101",
)


# ---- Chain graph helpers ------------------------------------------------- #

def chain_graph(n_nodes: int) -> list[list[int]]:
    """Simple sequential chain: [[0,1],[1,2],...,[n-2,n-1]]. n-1 edges."""
    return [[i, i + 1] for i in range(n_nodes - 1)]


def chain_skip1_graph(n_nodes: int) -> list[list[int]]:
    """Chain plus skip-1 (nearest non-adjacent) edges. ~2n-3 edges."""
    edges = [[i, i + 1] for i in range(n_nodes - 1)]
    edges += [[i, i + 2] for i in range(n_nodes - 2)]
    return edges


GRAPH_BUILDERS = {
    "chain": chain_graph,
    "chain_skip1": chain_skip1_graph,
    # "none" means: don't touch the graph DLC created
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", type=Path,
                    help="path to DLC project config.yaml")
    ap.add_argument("--net", default="hrnet_w32",
                    help=f"BU backbone. Suggested: {', '.join(SUGGESTED_BU_NETS)}. "
                         f"Default: hrnet_w32")
    ap.add_argument("--shuffle", type=int, default=1,
                    help="shuffle index for this BU model. Default: 1")
    ap.add_argument("--epochs", type=int, default=60,
                    help="number of training epochs. Default: 60")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="training batch size. In DDP, DLC typically interprets "
                         "this as the TOTAL batch across all GPUs (split evenly), "
                         "not per-GPU. Verify by checking iters/epoch after epoch 1. "
                         "Default: 32")
    ap.add_argument("--save-epochs", type=int, default=10,
                    help="save a snapshot every N epochs. Default: 10")
    ap.add_argument("--device", default="cuda:0",
                    help="single-GPU device. Ignored if --gpus is given.")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU IDs for DDP, e.g. '0,1'. "
                         "Default: None (single GPU, use --device).")
    ap.add_argument("--display-iters", type=int, default=100,
                    help="log loss every N iterations. Default: 100")
    ap.add_argument("--max-snapshots", type=int, default=5,
                    help="cap on snapshots kept on disk. Default: 5")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override dataloader_workers in pytorch_config.yaml. "
                         "Note: this is per-rank in DDP mode (so 2 GPUs * 8 = 16 "
                         "actual workers). If unset, leaves DLC's default.")
    ap.add_argument("--train-crop-size", type=int, default=None,
                    help="override data.train.crop_sampling width/height in "
                         "pytorch_config.yaml. Smaller = faster (~quadratic in "
                         "model compute). Default: leaves DLC's setting (usually 448).")
    ap.add_argument("--num-animals", type=int, default=None,
                    help="override num_animals in the PAF predictor. Should match "
                         "the post-crop max from the converter's histogram, with a "
                         "small headroom (e.g. p99*1.1). Default: leaves DLC's setting.")
    ap.add_argument("--prune-paf-graph", choices=["none", "chain", "chain_skip1"],
                    default="none",
                    help="Force the PAF graph to a specific structure before "
                         "training. 'none' (DEFAULT, RECOMMENDED) keeps DLC's "
                         "complete graph and lets DLC's data-driven pruning "
                         "select the optimal skeleton at evaluation time — "
                         "this is the entire point of DLC's multi-animal "
                         "design. 'chain' forces a brittle linear chain that "
                         "fails on any missed keypoint in crowded scenes — "
                         "use only for debugging or if you're SURE your animal "
                         "is unambiguously chain-structured AND your scenes "
                         "are never crowded. 'chain_skip1' adds second-nearest "
                         "edges for partial redundancy.")
    ap.add_argument("--paf-width", type=int, default=None,
                    help="Override the PAF target line width in pixels (see "
                         "`target_generator.PartAffinityFieldGenerator.width` "
                         "in pytorch_config.yaml). DLC's default is 20, sensible "
                         "for human/mouse-scale subjects at standard crops, but "
                         "too wide for small animals: adjacent PAFs along the "
                         "skeleton overlap each other, and PAFs from neighboring "
                         "animals in dense scenes interfere. Rule of thumb: set "
                         "to roughly half the typical distance between adjacent "
                         "skeleton nodes in your image. For 30-60 px animals in "
                         "a 512 crop, try 6-10. Default: leave DLC's setting.")
    ap.add_argument("--try-amp", action="store_true",
                    help="attempt to enable mixed-precision training via "
                         "pytorch_cfg_updates. NOTE: as of DLC 3.x docs, "
                         "autocast is exposed only for inference, not training, "
                         "so this is a no-op on current versions. Left in as a "
                         "forward-compatibility hook for future DLC releases.")
    ap.add_argument("--skip-check-labels", action="store_true",
                    help="skip the deeplabcut.check_labels visualization step")
    ap.add_argument("--skip-create-dataset", action="store_true",
                    help="assume the training dataset for this shuffle already exists")
    ap.add_argument("--skip-patch-config", action="store_true",
                    help="don't patch pytorch_config.yaml even if patch flags are set")
    ap.add_argument("--skip-evaluate", action="store_true",
                    help="skip evaluation after training")
    return ap.parse_args()


def print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def find_pytorch_config(project_path: Path, shuffle: int) -> Path:
    """Locate the pytorch_config.yaml for a given shuffle in this project."""
    search_dir = project_path / "dlc-models-pytorch" / "iteration-0"
    matches = list(search_dir.glob(f"*shuffle{shuffle}/train/pytorch_config.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"could not find pytorch_config.yaml for shuffle {shuffle} under "
            f"{search_dir}. Did the dataset creation step succeed?")
    if len(matches) > 1:
        print(f"  warn: multiple matches found, using first:\n    "
              + "\n    ".join(str(m) for m in matches))
    return matches[0]


def patch_pytorch_config(cfg_path: Path,
                          prune_paf_graph: str,
                          paf_width: int | None,
                          num_animals: int | None,
                          num_workers: int | None,
                          train_crop_size: int | None) -> None:
    """Modify the model architecture and training settings in pytorch_config.yaml.

    These can't all be passed via pytorch_cfg_updates reliably across DLC
    versions (especially the graph, which is referenced via a YAML anchor),
    so we patch the file directly. The patch is idempotent: re-running with
    the same args yields the same file.
    """
    import yaml
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    head = cfg["model"]["heads"]["bodypart"]
    n_bodyparts = head["predictor"]["num_multibodyparts"]
    changes: list[str] = []

    # Look up bodypart names for human-readable graph printout. Path varies
    # across DLC versions; best-effort lookup, fall back to indices.
    bodypart_names: list[str] | None = None
    for path in (
        ("metadata", "bodyparts"),
        ("metadata", "multianimalbodyparts"),
    ):
        node = cfg
        for k in path:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, list) and len(node) == n_bodyparts:
            bodypart_names = list(node)
            break

    def _edges_as_names(edges: list[list[int]]) -> str:
        if bodypart_names is None:
            return ", ".join(f"{a}->{b}" for a, b in edges)
        return ", ".join(f"{bodypart_names[a]}->{bodypart_names[b]}"
                         for a, b in edges)

    if prune_paf_graph != "none":
        new_graph = GRAPH_BUILDERS[prune_paf_graph](n_bodyparts)
        n_edges = len(new_graph)
        head["predictor"]["graph"] = new_graph
        head["predictor"]["edges_to_keep"] = list(range(n_edges))
        # The PAF generator may share the graph via YAML anchor; set explicitly
        # to be safe across loaders.
        for gen in head["target_generator"]["generators"]:
            if gen.get("type") == "PartAffinityFieldGenerator":
                gen["graph"] = new_graph
        changes.append(f"PAF graph: {prune_paf_graph} ({n_edges} edges)")
        # PAF channel count is derived from the graph at model-construction
        # time in DLC 3.0 — no separate `paf_config.channels` field to update.
        # Note: edges connect bodyparts by INDEX in `metadata.bodyparts`.
        # Print the resulting chain in node names so the user can sanity-check
        # the topology before training kicks off.
        print(f"  PAF graph edges ({prune_paf_graph}): "
              f"{_edges_as_names(new_graph)}")
        if bodypart_names is None:
            print("  (could not resolve bodypart names from config; "
                  "verify graph indices match your skeleton manually)")

    if paf_width is not None:
        found = False
        for gen in head["target_generator"]["generators"]:
            if gen.get("type") == "PartAffinityFieldGenerator":
                gen["width"] = paf_width
                found = True
        if found:
            changes.append(f"PAF target width: {paf_width} px")
        else:
            print("  warn: --paf-width set but no PartAffinityFieldGenerator "
                  "found in target_generator; check head structure")

    if num_animals is not None:
        head["predictor"]["num_animals"] = num_animals
        changes.append(f"num_animals: {num_animals}")

    if train_crop_size is not None:
        cfg.setdefault("data", {}).setdefault("train", {}).setdefault("crop_sampling", {})
        cfg["data"]["train"]["crop_sampling"]["width"] = train_crop_size
        cfg["data"]["train"]["crop_sampling"]["height"] = train_crop_size
        changes.append(f"train crop: {train_crop_size}x{train_crop_size}")

    if num_workers is not None:
        cfg.setdefault("train_settings", {})["dataloader_workers"] = num_workers
        changes.append(f"dataloader_workers: {num_workers}")

    if not changes:
        print("  (no patches requested)")
        return

    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"  patched {cfg_path}")
    for c in changes:
        print(f"    - {c}")


def expected_iters(n_train: int, batch_size: int, n_gpus: int) -> tuple[int, int]:
    """Return (iters_if_per_rank, iters_if_total) under the two common DDP
    interpretations of batch_size."""
    per_rank = max(1, n_train // (batch_size * n_gpus))
    as_total = max(1, n_train // batch_size)
    return per_rank, as_total


def main() -> int:
    args = parse_args()

    if not args.config.exists():
        sys.exit(f"config not found: {args.config}")
    config = str(args.config.resolve())
    project_path = args.config.resolve().parent

    try:
        import deeplabcut
    except ImportError:
        sys.exit("deeplabcut not installed. Run: pip install --pre deeplabcut")
    print(f"DeepLabCut version: {deeplabcut.__version__}")
    if not deeplabcut.__version__.startswith("3"):
        print("WARNING: BUCTD requires DLC 3.0+. Older versions don't have it.")

    try:
        from deeplabcut.core.engine import Engine
    except ImportError:
        try:
            from deeplabcut import Engine
        except ImportError:
            sys.exit("Cannot import Engine from deeplabcut. Upgrade with: "
                     "pip install --pre deeplabcut")
    print(f"Engine:             {Engine.PYTORCH}")

    gpus_list: list[int] | None = None
    if args.gpus:
        gpus_list = [int(x) for x in args.gpus.split(",") if x.strip()]
        if len(gpus_list) < 2:
            print(f"NOTE: --gpus has only {len(gpus_list)} GPU(s). "
                  f"For single GPU, --device is sufficient.")

    n_gpus = len(gpus_list) if gpus_list else 1

    print(f"Project config: {config}")
    print(f"BU net type:    {args.net}")
    print(f"Shuffle:        {args.shuffle}")
    if gpus_list is not None:
        print(f"GPUs (DDP):     {gpus_list}")
        print(f"Batch size:     {args.batch_size}  "
              f"(DLC's interpretation in DDP varies — see iteration count "
              f"diagnostic after epoch 1)")
    else:
        print(f"Device:         {args.device}")
        print(f"Batch size:     {args.batch_size}")

    # ---------------- 1. Optional sanity check ----------------
    if not args.skip_check_labels:
        print_section("Sanity check: deeplabcut.check_labels")
        try:
            deeplabcut.check_labels(config, draw_skeleton=True)
            print("OK — review the generated labeled-images dirs before training.")
        except Exception as e:
            print(f"WARNING: check_labels failed: {e}")
            print("Continuing — manually inspect labels if you can.")

    # ---------------- 2. Create the training dataset ----------------
    if not args.skip_create_dataset:
        print_section(f"Create multi-animal training dataset "
                      f"(shuffle={args.shuffle}, net={args.net})")
        deeplabcut.create_multianimaltraining_dataset(
            config,
            Shuffles=[args.shuffle],
            net_type=args.net,
            userfeedback=False,
            engine=Engine.PYTORCH,
        )
        print("Dataset created.")
    else:
        print("Skipping dataset creation (--skip-create-dataset).")

    # ---------------- 3. Patch pytorch_config.yaml ----------------
    if not args.skip_patch_config:
        print_section("Patch pytorch_config.yaml")
        try:
            cfg_path = find_pytorch_config(project_path, args.shuffle)
            patch_pytorch_config(
                cfg_path,
                prune_paf_graph=args.prune_paf_graph,
                paf_width=args.paf_width,
                num_animals=args.num_animals,
                num_workers=args.num_workers,
                train_crop_size=args.train_crop_size,
            )
        except FileNotFoundError as e:
            print(f"WARNING: skipping patch ({e})")
        except Exception as e:
            print(f"WARNING: patch failed: {e}")
            print("Training will proceed with the unpatched config. Verify the "
                  "config manually if speed is the issue.")

    # ---------------- 4. Train ----------------
    print_section(f"Train BU model (shuffle={args.shuffle}, {args.epochs} epochs, "
                  f"batch={args.batch_size})")

    # Diagnostic for expected iters/epoch under both DDP interpretations.
    # We can't easily know N_train here without parsing the documentation
    # path, but the user will see the actual number after epoch 1. Print
    # the formulas so they can verify.
    if gpus_list is not None:
        print("DDP iteration diagnostic:")
        print(f"  If batch_size={args.batch_size} is per-GPU (true DDP convention),")
        print(f"    iters/epoch ≈ N_train / ({args.batch_size} * {n_gpus}) "
              f"= N_train / {args.batch_size * n_gpus}")
        print(f"  If batch_size={args.batch_size} is the TOTAL batch (DLC's actual "
              f"behavior in some versions),")
        print(f"    iters/epoch ≈ N_train / {args.batch_size}  "
              f"(each rank sees half the data, batches {args.batch_size // n_gpus})")
        print(f"  Compare to the value printed by DLC after epoch 1 to confirm.")
        print(f"  If neither matches and iters/epoch ≈ N_train / {args.batch_size}, ")
        print(f"  DDP may not be sharding properly — check with nvidia-smi that "
              f"both GPUs are at ~100% utilization.")

    t0 = time.time()
    train_kwargs = dict(
        shuffle=args.shuffle,
        trainingsetindex=0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_epochs=args.save_epochs,
        displayiters=args.display_iters,
        max_snapshots_to_keep=args.max_snapshots,
        engine=Engine.PYTORCH,
    )

    pytorch_cfg_updates: dict = {}
    if gpus_list is not None:
        pytorch_cfg_updates["runner.gpus"] = gpus_list
    if args.try_amp:
        # Best-effort: set common keys. Newer DLC versions may use one of these.
        # Older versions will silently ignore unknown keys; DLC's pytorch_cfg
        # apply step does a deep-merge, so this should be safe.
        pytorch_cfg_updates["runner.use_amp"] = True
        pytorch_cfg_updates["runner.autocast.enabled"] = True
        pytorch_cfg_updates["runner.autocast.dtype"] = "bfloat16"
        print("Attempting AMP via pytorch_cfg_updates. Per DLC 3.x docs, "
              "training-side autocast isn't documented (only inference), so "
              "this likely no-ops; left in for forward compatibility.")

    if pytorch_cfg_updates:
        train_kwargs["pytorch_cfg_updates"] = pytorch_cfg_updates

    if gpus_list is None:
        train_kwargs["device"] = args.device

    deeplabcut.train_network(config, **train_kwargs)
    elapsed_h = (time.time() - t0) / 3600
    print(f"Training done in {elapsed_h:.2f} h.")

    # ---------------- 5. Evaluate ----------------
    if not args.skip_evaluate:
        print_section(f"Evaluate (shuffle={args.shuffle})")
        try:
            deeplabcut.evaluate_network(
                config,
                Shuffles=[args.shuffle],
                plotting=True,
                show_errors=True,
                comparisonbodyparts="all",
                engine=Engine.PYTORCH,
            )
            print("Evaluation done. See evaluation-results-pytorch/ for plots and CSVs.")
        except Exception as e:
            print(f"WARNING: evaluation failed: {e}")
            print("You can run deeplabcut.evaluate_network(config, Shuffles=[N]) manually.")

    print_section("Stage 1 (BU) complete")
    print(f"Next step: train the CTD stage on top of this BU model:")
    print(f"  python train_dlc_ctd.py {args.config} \\")
    print(f"      --bu-shuffle {args.shuffle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
