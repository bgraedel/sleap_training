"""
Train a Mask2Former model (HuggingFace transformers) on a Leishmania
segmentation dataset produced by ``segmentation_dataset_creator.py``.

Reads the existing layout directly (no conversion needed):

    <data_root>/
      images/<stem>.png
      instances/<stem>.png        16-bit per-pixel animal_id (0=bg)
      masks/<stem>.png            semantic part map {0=bg, 1=body, 2=flag}
      flag_instances/<stem>.png   16-bit per-flagellum id (0=bg, 1..F) [optional]
      train.txt  val.txt          ./images/<stem>.png lists

Per-instance binary masks are derived as
``(instances == animal_id) & (masks == sem_class)`` — exact ground truth,
matches how the writer documents the association. For the flagellum class,
if a ``flag_instances/`` map is present (default), each flagellum of a
dividing cell becomes its OWN instance via ``(instances == animal_id) &
(flag_instances == flag_id)``; pass ``--merge-flagella`` to collapse them
into one flagellum instance (the legacy behaviour, also used automatically
when the map is absent).

Defaults: Swin-T backbone, 2-class instance seg (body + flagellum) — the
sweet spot for thin-flagellum performance vs YOLO-seg's prototype masks.
Swap ``--model facebook/mask2former-swin-base-coco-instance`` for more
accuracy at ~3-4x the cost.

Examples
--------
Smoke test (verifies data path + model wiring in ~30 s)::

    python train_mask2former.py \\
        --data-root data/leishmania_seg_ds_640_division \\
        --out runs/m2f_smoke --smoke

Full training run::

    python train_mask2former.py \\
        --data-root data/leishmania_seg_ds_640_division \\
        --out runs/m2f_swin_t_parts_v1 \\
        --epochs 40 --batch-size 4

Use 1-class animal-only seg (closest to ``seg_yolo_classes: animal``)::

    python train_mask2former.py ... --classes animal

Use 3-class overlapping (animal + body + flagellum)::

    python train_mask2former.py ... --classes all

Train at 1024x1024 instead of the default 640x640::

    python train_mask2former.py ... --image-size 1024

Install
-------
    pip install "transformers>=4.41" accelerate torchvision pillow \\
                "torchmetrics[detection]" pycocotools albumentations

(``torchmetrics[detection]`` + ``pycocotools`` are only needed for the
periodic mAP / P / R eval. ``albumentations`` is only needed for training
augmentation. Training itself runs without both.)

A CUDA GPU with >= 12 GB VRAM is comfortable for Swin-T @ 640 with
batch=4. For Swin-B drop batch to 2 and bump ``--grad-accum 2`` to keep
the effective batch the same.

Metrics output
--------------
Every ``--eval-every`` epochs the callback runs full instance-seg eval on
val and writes three files under ``<out>/metrics/``:

  - ``metrics.csv``           epoch, P, R, mAP50, mAP50-95, mAR100 (YOLO-style)
  - ``metrics.jsonl``         one JSON record per eval (incl. per-class)
  - ``metrics_history.json``  consolidated history (rewritten each eval)

Metric definitions (YOLO-style, for monitoring — note P/R are reported at a
single fixed ``--eval-conf-threshold`` operating point, NOT the F1-optimal
point Ultralytics results.csv uses, so they are not directly comparable):
  - P, R         greedy IoU-0.5 match at ``--eval-conf-threshold``
  - mAP50        COCO mAP at IoU=0.5            (torchmetrics map_50)
  - mAP50-95     COCO mAP at IoU=0.5:0.05:0.95  (torchmetrics map)

Checkpoints
-----------
Two kinds, mirroring YOLO's ``last.pt`` / ``best.pt``:

  - ``<out>/checkpoint-<step>/``  iterative — saved every epoch, the most
                                  recent ``--save-total-limit`` (default 3)
                                  are kept (Trainer rotates older ones out).
  - ``<out>/checkpoint-best/``    best — overwritten whenever the metric
                                  named by ``--save-best-metric`` (default
                                  mAP50-95) improves on val. Includes the
                                  processor + a ``best.json`` summary.

Load either with::

    from transformers import (Mask2FormerForUniversalSegmentation,
                              Mask2FormerImageProcessor)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "runs/m2f.../checkpoint-best")
    processor = Mask2FormerImageProcessor.from_pretrained(
        "runs/m2f.../checkpoint-best")

Changes vs original
-------------------
  - Differential LRs: backbone uses ``--lr-backbone`` (default 1e-5),
    head uses ``--lr`` (default 1e-4).
  - Data augmentation: random flips + 90° rotations + colour jitter +
    Gaussian noise/blur applied jointly to image and all masks via
    albumentations (train split only). Disable with ``--no-augment``.
  - Gradient clipping: ``max_grad_norm=1.0`` by default (``--max-grad-norm``).
  - ``build_processor`` now passes ``do_resize=False`` as a constructor
    arg so the setting is always serialised into preprocessor_config.json.
  - ``--image-size`` (default 640): input resolution recorded in
    train_meta.json and processor config; supports arbitrary multiples of 32.
  - ``evaluate_full`` properly restores ``model.train()`` after eval via
    a try/finally guard, and processes the val set in batches instead of
    one sample at a time.
  - ``on_epoch_end`` tracks the last-evaluated epoch explicitly to avoid
    double-trigger / miss at float epoch boundaries.
  - ``__getitem__`` wraps IO in try/except and re-raises with the stem
    name for fast diagnosis of broken files.
  - ``min_area`` default raised from 6 → 64 pixels.
  - ``dataloader_pin_memory=True`` for free throughput with workers > 0.
  - Processor saved at run start (before training) so a checkpoint always
    exists even if training is interrupted.
  - ``torch.backends.cudnn.benchmark = True`` for fixed-resolution runs.
  - ``--resume`` CLI arg for checkpoint-resumption.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

try:
    from torchmetrics.detection import MeanAveragePrecision
    _HAVE_TM = True
except ImportError:
    _HAVE_TM = False

try:
    import albumentations as A
    _HAVE_ALB = True
except ImportError:
    _HAVE_ALB = False


# ---------------------------------------------------------------------------
# Class set
# ---------------------------------------------------------------------------
# Three modes. "parts" is the default because the body/flag disjoint instances
# are exactly what Mask2Former's mask head excels at (thin objects, full-res
# attention). "animal" gives a single class, closest to the YOLO animal
# detector. "all" emits overlapping (animal, body, flag) instances per cell —
# Mask2Former handles overlaps natively but the head has to learn the
# redundancy.

CLASS_SETS: Dict[str, Dict] = {
    "parts":  {"id2label": {0: "body", 1: "flagellum"},
               "sem_to_cls": {1: 0, 2: 1},
               "merge_per_cell": False, "include_animal": False},
    "animal": {"id2label": {0: "animal"},
               "sem_to_cls": {1: 0, 2: 0},
               "merge_per_cell": True,  "include_animal": False},
    "all":    {"id2label": {0: "animal", 1: "body", 2: "flagellum"},
               "sem_to_cls": {1: 1, 2: 2},
               "merge_per_cell": False, "include_animal": True},
}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_augmentation_pipeline() -> Optional[object]:
    """Return an albumentations Compose pipeline for joint image+mask aug,
    or None if albumentations is not installed.

    Augmentations chosen for phase-contrast microscopy:
      - Geometric: flips + 90° rotations are free wins — cells appear in all
        orientations. No large-scale spatial warping to avoid breaking thin
        flagella topology.
      - Colour: brightness/contrast jitter matches illumination variance
        between acquisitions. Hue/saturation skipped (phase contrast is
        effectively greyscale stacked to 3 channels).
      - Noise/blur: GaussNoise + GaussianBlur simulate camera noise and
        slight focus variation common in time-lapse datasets.
    """
    if not _HAVE_ALB:
        return None
    # GaussNoise's strength kwarg was renamed across albumentations versions
    # (var_limit -> std_range/var_range in 2.x). Try the explicit-strength form
    # first for reproducibility, then fall back to whatever the installed
    # version defaults to so the pipeline never crashes on construction.
    try:
        gauss_noise = A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
    except (TypeError, ValueError):
        gauss_noise = A.GaussNoise(p=0.2)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0,
                      hue=0.0, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        gauss_noise,
    ])
    # NOTE: albumentations applies the same spatial transform to every mask
    # in the `masks` kwarg list, so image/mask spatial consistency is
    # guaranteed without any extra bookkeeping.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LeishInstanceDataset(Dataset):
    """Reads ``segmentation_dataset_creator.py`` output and yields per-frame
    instance lists ready for the collator. Returns numpy arrays (no resizing
    here — the collator hands them to the HF processor).

    Parameters
    ----------
    augment:
        If True and albumentations is available, apply random spatial +
        colour augmentation to each sample. Should be True for train, False
        for val.
    """

    SEM_FLAGELLUM = 2   # semantic value the writer uses for flagellum pixels

    def __init__(self, root: Path, split: str, class_set: str = "parts",
                 min_area: int = 64, min_area_flag: int = 12,
                 split_flagella: bool = True, augment: bool = False,
                 image_size: Optional[int] = None):
        self.root = Path(root)
        self.class_cfg = CLASS_SETS[class_set]
        self.min_area = int(min_area)
        # Target training resolution. When set, each frame (image + all label
        # maps) is resized to (image_size, image_size) on load so image and
        # masks stay consistent — the HF processor's do_resize cannot do this
        # because it never sees our per-instance masks. None = no resize.
        self.target_size = int(image_size) if image_size else None
        self._warned_downscale = False
        # Thin flagella (3-5 px wide) can be far smaller than a cell body; a
        # global min_area tuned for bodies silently deletes short/clipped
        # flagella. Use a much lower floor for the flagellum class only.
        self.min_area_flag = int(min_area_flag)
        # The class id used for flagellum instances in the current class set
        # (None when flagella are merged into the whole-cell instance, e.g.
        # "animal" mode). Drives the per-class min-area selection.
        self.flag_cls_id = (
            None if self.class_cfg["merge_per_cell"]
            else self.class_cfg["sem_to_cls"].get(self.SEM_FLAGELLUM))
        # When True (default) and a flag_instances/ map is present, each
        # flagellum of a dividing cell becomes its OWN flagellum instance.
        # When False (or the map is absent) all of a cell's flagella collapse
        # into one flagellum instance (the legacy behaviour).
        self.split_flagella = bool(split_flagella)
        self.has_flag_instances = (self.root / "flag_instances").is_dir()
        self.augment = augment
        self._aug = build_augmentation_pipeline() if augment else None
        if augment and self._aug is None:
            print("WARNING: --augment requested but albumentations is not "
                  "installed — training without augmentation. "
                  "Install with: pip install albumentations")
        list_file = self.root / f"{split}.txt"
        if not list_file.exists():
            raise FileNotFoundError(
                f"{list_file} not found — is --data-root pointing at the "
                f"seg dataset (the dir with images/ instances/ masks/)?")
        rel = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
        self.stems = [Path(rp).stem for rp in rel]

    def __len__(self) -> int:
        return len(self.stems)

    def _load_image(self, stem: str) -> np.ndarray:
        img = np.array(Image.open(self.root / "images" / f"{stem}.png"))
        # Phase-contrast frames are grayscale; stack to 3 channels so the
        # ImageNet-normalised processor is happy. (Mismatched stats are
        # absorbed by fine-tuning; not worth re-computing for a synthetic set.)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        return img.astype(np.uint8)

    def _load_label_maps(self, stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inst = np.array(Image.open(self.root / "instances" / f"{stem}.png"))
        sem = np.array(Image.open(self.root / "masks" / f"{stem}.png"))
        flag_inst = None
        if self.has_flag_instances:
            fp = self.root / "flag_instances" / f"{stem}.png"
            if fp.exists():
                flag_inst = np.array(Image.open(fp)).astype(np.int32)
        return inst.astype(np.int32), sem.astype(np.int32), flag_inst

    def _resize_sample(self, image: np.ndarray, inst: np.ndarray,
                       sem: np.ndarray, flag_inst: Optional[np.ndarray]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Resize image + all label maps to ``self.target_size`` together.

        Image uses bilinear interpolation (smooth); label maps use
        nearest-neighbour via pure-numpy index sampling so integer instance /
        semantic / flagellum IDs are preserved exactly (no blending of IDs)."""
        T = self.target_size
        if T is None:
            return image, inst, sem, flag_inst
        H, W = image.shape[:2]
        if H == T and W == T:
            return image, inst, sem, flag_inst
        if T < H and not self._warned_downscale:
            print(f"WARNING: downscaling {H}x{W} -> {T}x{T} with nearest "
                  f"interpolation can erase thin flagella (3-5 px). Prefer "
                  f"building the dataset at <= the training resolution.")
            self._warned_downscale = True
        # np.array (not np.asarray) returns a writable copy — np.asarray of a
        # PIL image is read-only and triggers a torch.from_numpy warning later.
        image = np.array(Image.fromarray(image).resize((T, T), Image.BILINEAR))
        ys = (np.arange(T) * (H / T)).astype(np.int64)
        xs = (np.arange(T) * (W / T)).astype(np.int64)
        inst = inst[ys][:, xs]
        sem = sem[ys][:, xs]
        if flag_inst is not None:
            flag_inst = flag_inst[ys][:, xs]
        return image, inst, sem, flag_inst

    def _min_area_for(self, cls_id: int) -> int:
        """Per-class minimum mask area. The flagellum class uses the lower
        ``min_area_flag`` floor so short/thin flagella are not deleted."""
        return (self.min_area_flag if cls_id == self.flag_cls_id
                else self.min_area)

    def _build_instances(self, inst: np.ndarray, sem: np.ndarray,
                         flag_inst: np.ndarray = None
                         ) -> Tuple[List[np.ndarray], List[int]]:
        masks: List[np.ndarray] = []
        classes: List[int] = []
        cfg = self.class_cfg
        use_flag_split = (self.split_flagella and flag_inst is not None
                          and not cfg["merge_per_cell"])
        H, W = inst.shape
        aids = np.unique(inst)
        aids = aids[aids > 0]  # drop background
        for aid in aids:
            # Locate the instance once, then do all the boolean intersections
            # on its bounding-box sub-window rather than the full frame. On
            # dense frames (up to ~150 cells) this is the dominant CPU cost.
            ys, xs = np.where(inst == aid)
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            cell = (inst[y0:y1, x0:x1] == aid)
            sem_c = sem[y0:y1, x0:x1]
            flag_c = flag_inst[y0:y1, x0:x1] if use_flag_split else None

            def _place(sub: np.ndarray) -> np.ndarray:
                full = np.zeros((H, W), dtype=bool)
                full[y0:y1, x0:x1] = sub
                return full

            # Whole-cell (animal) instance — for "all" and "animal" modes.
            if cfg["include_animal"] or cfg["merge_per_cell"]:
                if int(cell.sum()) >= self.min_area:
                    masks.append(_place(cell))
                    # In "all" mode animal_id is class 0; in "animal" mode
                    # it's the only class (also 0).
                    classes.append(0)
            # Per-part (body, flag) instances — for "parts" and "all" modes.
            if not cfg["merge_per_cell"]:
                for sem_val, cls_id in cfg["sem_to_cls"].items():
                    thr = self._min_area_for(int(cls_id))
                    if sem_val == self.SEM_FLAGELLUM and use_flag_split:
                        # One instance per flagellum of this cell.
                        fids = np.unique(flag_c[cell])
                        fids = fids[fids > 0]
                        emitted = False
                        for fid in fids:
                            m = cell & (flag_c == int(fid))
                            if int(m.sum()) >= thr:
                                masks.append(_place(m))
                                classes.append(int(cls_id))
                                emitted = True
                        # Fallback: the cell has flagellum semantic pixels but
                        # no per-flagellum id (map gap) — emit the merged
                        # semantic flagellum so the instance isn't lost.
                        if not emitted:
                            m = cell & (sem_c == sem_val)
                            if int(m.sum()) >= thr:
                                masks.append(_place(m))
                                classes.append(int(cls_id))
                    else:
                        m = cell & (sem_c == sem_val)
                        if int(m.sum()) >= thr:
                            masks.append(_place(m))
                            classes.append(int(cls_id))
        return masks, classes

    def _apply_augmentation(self, image: np.ndarray,
                            masks: List[np.ndarray],
                            classes: List[int]
                            ) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        """Apply albumentations transforms jointly to the image and all masks
        in a SINGLE call, then filter masks AND their classes together by the
        same (per-class) area check.

        Albumentations draws fresh randomness on every ``__call__``, so the
        transform must be run exactly once and masks/classes filtered from the
        same result — otherwise the surviving-mask set and surviving-class set
        come from different random transforms and silently desync (a body mask
        could end up paired with a flagellum label)."""
        if self._aug is None or len(masks) == 0:
            return image, masks, classes
        uint8_masks = [m.astype(np.uint8) for m in masks]
        result = self._aug(image=image, masks=uint8_masks)
        # Albumentations preserves order and length of the `masks` list, so the
        # positional zip below keeps each mask matched to its class. Drop any
        # mask that fell below its per-class area floor (e.g. rotated out of
        # frame, or a thin flagellum clipped at the border).
        kept_masks: List[np.ndarray] = []
        kept_classes: List[int] = []
        for m, cls in zip(result["masks"], classes):
            mb = np.asarray(m).astype(bool)
            if int(mb.sum()) >= self._min_area_for(int(cls)):
                kept_masks.append(mb)
                kept_classes.append(int(cls))
        return result["image"], kept_masks, kept_classes

    def __getitem__(self, idx: int) -> Dict:
        stem = self.stems[idx]
        try:
            image = self._load_image(stem)
            inst, sem, flag_inst = self._load_label_maps(stem)
            image, inst, sem, flag_inst = self._resize_sample(
                image, inst, sem, flag_inst)
            masks, classes = self._build_instances(inst, sem, flag_inst)
            if self.augment and self._aug is not None and len(masks) > 0:
                # Single augmentation pass; masks and classes are filtered
                # together from the same random transform (see docstring).
                image, masks, classes = self._apply_augmentation(
                    image, masks, classes)
            return {"image": image, "masks": masks, "classes": classes}
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sample '{stem}' (index {idx}): {exc}") from exc


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def make_collator(processor):
    """HF Mask2Former expects ``pixel_values`` plus parallel lists
    ``mask_labels`` (list of (N_i, H, W) float tensors) and ``class_labels``
    (list of (N_i,) long tensors). The processor handles normalisation;
    masks pass through at the input resolution."""

    def collate(batch):
        images = [b["image"] for b in batch]
        enc = processor(images=images, return_tensors="pt")
        # Derive spatial dims from the encoded pixel_values so empty-frame
        # tensors are always consistent with what the model sees, even if
        # the processor pads to a size different from the raw image.
        _, _, pH, pW = enc["pixel_values"].shape
        mask_labels: List[torch.Tensor] = []
        class_labels: List[torch.Tensor] = []
        for b in batch:
            if len(b["classes"]) == 0:
                # Empty frame (negative tile): pass 0-instance tensors.
                # The model's loss is well-defined for empty targets.
                mask_labels.append(torch.zeros((0, pH, pW), dtype=torch.float32))
                class_labels.append(torch.zeros((0,), dtype=torch.long))
            else:
                ms = np.stack(b["masks"], axis=0)
                # Masks are built at the raw image resolution and must match the
                # (possibly padded) pixel_values the model sees. With a
                # uniform-resolution dataset + do_resize=False this always holds;
                # assert it so a future mixed-resolution set / padding processor
                # fails loudly instead of silently mis-supervising.
                if ms.shape[1] != pH or ms.shape[2] != pW:
                    raise ValueError(
                        f"mask spatial size {ms.shape[1:]} != pixel_values "
                        f"{(pH, pW)}. The processor resized/padded the image but "
                        f"masks were not adjusted. Ensure all images are the same "
                        f"size as --image-size, or enable processor resizing.")
                # float32 is required downstream: the Mask2Former mask loss
                # samples target masks with grid_sample, which needs float.
                mask_labels.append(torch.from_numpy(ms.astype(np.float32)))
                class_labels.append(torch.tensor(b["classes"], dtype=torch.long))
        enc["mask_labels"] = mask_labels
        enc["class_labels"] = class_labels
        return enc

    return collate


# ---------------------------------------------------------------------------
# Model / processor
# ---------------------------------------------------------------------------

def build_processor(name: str, image_size: int = 640) -> Mask2FormerImageProcessor:
    """Build and return the image processor.

    ``do_resize=False`` is passed as a constructor argument (not mutated
    post-construction) so the setting is guaranteed to be written into
    ``preprocessor_config.json`` by ``save_pretrained``. This means loading
    the processor from a checkpoint will always restore the correct behaviour
    without any manual intervention.

    ``image_size`` is stored in the processor's ``size`` field purely for
    documentation; the processor won't resize (``do_resize=False``) because
    resizing is done upstream in the dataset (image + masks together, which
    the processor cannot do since it never sees our per-instance masks).
    Recording the size keeps the checkpoint self-describing for inference.
    """
    assert image_size % 32 == 0, (
        f"--image-size must be a multiple of 32 (got {image_size})")
    proc = Mask2FormerImageProcessor.from_pretrained(
        name,
        do_resize=False,
        # Document the intended operating resolution so any later inference
        # script can read it from preprocessor_config.json.
        size={"height": image_size, "width": image_size},
        # Phase-contrast frames are grayscale stacked to 3 identical channels.
        # The default per-channel ImageNet mean/std would apply three different
        # shifts to three identical channels, creating an artificial colour
        # cast. Use symmetric [0.5,0.5,0.5] so all channels are treated alike.
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    return proc


def build_model(name: str, id2label: Dict[int, str],
                num_queries: Optional[int] = None,
                train_num_points: Optional[int] = None,
                importance_sample_ratio: Optional[float] = None
                ) -> Mask2FormerForUniversalSegmentation:
    label2id = {v: k for k, v in id2label.items()}
    # Config overrides passed at construction so they reach the loss criterion,
    # which reads them in Mask2FormerForUniversalSegmentation.__init__ (setting
    # them on model.config afterwards would NOT update the already-built loss).
    overrides: Dict[str, object] = {}
    if num_queries is not None:
        # COCO checkpoints ship with num_queries=100, a hard cap on instances
        # per image. Dense Leishmania frames need many more (see --num-queries).
        # Changing this re-inits the query embeddings (random), kept loadable by
        # ignore_mismatched_sizes below.
        overrides["num_queries"] = int(num_queries)
    if train_num_points is not None:
        # Mask2Former computes the mask/dice loss on sampled points, not dense
        # pixels. Thin flagella (3-5 px) occupy <1% of pixels, so more points
        # are needed for them to receive gradient.
        overrides["train_num_points"] = int(train_num_points)
    if importance_sample_ratio is not None:
        # Fraction of points drawn from uncertain/boundary regions where thin
        # structures live (default 0.75).
        overrides["importance_sample_ratio"] = float(importance_sample_ratio)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        name,
        id2label=id2label,
        label2id=label2id,
        # Re-init the classifier head (and query embeddings if num_queries
        # changed) for our config; backbone + pixel decoder + transformer
        # decoder weights are kept.
        ignore_mismatched_sizes=True,
        **overrides,
    )
    return model


def build_optimizer_and_scheduler(
        model: Mask2FormerForUniversalSegmentation,
        lr_backbone: float,
        lr_head: float,
        lr_decoder: float,
        weight_decay: float,
        num_training_steps: int,
        warmup_ratio: float,
) -> Tuple[AdamW, object]:
    """Three-group differential learning rates:

      - backbone (pretrained Swin): lowest LR, preserve pretrained features.
      - pixel/transformer decoders (pretrained): a middle LR so they adapt
        without being overwritten at the full head LR.
      - truly-new params (classification head + re-initialised query
        embeddings): the full head LR for fast convergence.

    Separate param groups all decay under the same cosine schedule, keeping
    the LR ratios constant throughout training.
    """
    # Substrings identifying the randomly-(re)initialised head params. Naming
    # can drift across transformers versions; the fallback below handles a miss.
    NEW_HEAD_KEYS = ("class_predictor", "class_embed",
                     "queries_features", "queries_embedder", "query_embed")
    backbone_params, decoder_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif any(k in name for k in NEW_HEAD_KEYS):
            head_params.append(param)
        else:
            decoder_params.append(param)

    # Fallback: if the head-key heuristic matched nothing (naming drift), revert
    # to the original 2-group split so the new head still gets the full head LR.
    if not head_params:
        head_params, decoder_params = decoder_params, []

    groups = [{"params": backbone_params, "lr": lr_backbone}]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr_decoder})
    print(f"Optimizer param groups: backbone={len(backbone_params)} "
          f"head={len(head_params)} decoder={len(decoder_params)} "
          f"(lr_backbone={lr_backbone} lr_head={lr_head} lr_decoder={lr_decoder})")

    optimizer = AdamW(groups, weight_decay=weight_decay)

    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    # HuggingFace Trainer expects a per-step scheduler.
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Periodic detection metrics (mAP50, mAP50-95, P, R) — YOLO results.csv style
# ---------------------------------------------------------------------------

def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU of two (H, W) boolean masks."""
    inter = (a & b).sum().item()
    union = a.sum().item() + b.sum().item() - inter
    return inter / max(union, 1)


def _match_frame(pred_masks, pred_labels, pred_scores,
                 gt_masks, gt_labels, num_classes, iou_thresh=0.5):
    """Greedy per-class matching at the given IoU threshold. Per-class
    (TP, FP, FN) lists. Highest-score prediction takes the best available
    same-class GT it overlaps with — standard COCO/YOLO matching at one IoU."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    if len(pred_scores) > 0:
        order = torch.argsort(pred_scores, descending=True).tolist()
    else:
        order = []
    matched = [False] * len(gt_labels)
    for pi in order:
        pc = int(pred_labels[pi])
        if pc < 0 or pc >= num_classes:
            continue
        pm = pred_masks[pi]
        best_iou, best_t = 0.0, -1
        for ti in range(len(gt_labels)):
            if matched[ti] or int(gt_labels[ti]) != pc:
                continue
            iou = _mask_iou(pm, gt_masks[ti])
            if iou > best_iou:
                best_iou, best_t = iou, ti
        if best_iou >= iou_thresh and best_t >= 0:
            tp[pc] += 1
            matched[best_t] = True
        else:
            fp[pc] += 1
    for ti, tc in enumerate(gt_labels):
        if not matched[ti]:
            c = int(tc)
            if 0 <= c < num_classes:
                fn[c] += 1
    return tp, fp, fn


@torch.no_grad()
def evaluate_full(model, dataset, processor, conf_threshold: float,
                  num_classes: int, device,
                  batch_size: int = 4, num_workers: int = 2,
                  mask_threshold: float = 0.4, overlap_threshold: float = 0.5,
                  max_detections: int = 500, log_every: int = 50) -> Dict:
    """Run the whole val set through the model, compute COCO mAP via
    torchmetrics and IoU=0.5 P/R via _match_frame. Returns a metrics dict.

    The val set is processed in batches (``batch_size``) rather than one
    sample at a time for substantially faster evaluation. ``model.train()``
    is restored in a finally-block so a mid-eval exception cannot leave the
    model stuck in eval mode.
    """
    if not _HAVE_TM:
        raise RuntimeError(
            "torchmetrics not installed. Run:  pip install "
            "'torchmetrics[detection]' pycocotools")

    was_training = model.training
    model.eval()

    # COCO's default cap of 100 detections/image silently drops the
    # lowest-scoring predictions on dense frames (>100 instances), making
    # mAP/mAR pessimistic. Raise the last threshold above the densest frame's
    # instance count. (Our custom P/R below uses ALL detections regardless.)
    metric = MeanAveragePrecision(
        iou_type="segm", class_metrics=True,
        max_detection_thresholds=[1, 10, int(max_detections)])
    try:
        metric.warn_on_many_detections = False
    except Exception:
        pass
    tp_all = [0] * num_classes
    fp_all = [0] * num_classes
    fn_all = [0] * num_classes

    # Build a plain DataLoader for batched inference. No augmentation on val.
    collate = make_collator(processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=True)

    try:
        sample_idx = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            B, _, H, W = pixel_values.shape
            outputs = model(pixel_values=pixel_values)

            # Lower mask/overlap thresholds help thin flagella survive
            # binarisation. post_process_instance_segmentation gained
            # return_binary_maps in transformers 4.41; older versions also may
            # not accept mask_threshold/overlap_mask_area_threshold — fall back.
            pp_kwargs = dict(
                target_sizes=[(H, W)] * B, threshold=conf_threshold,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_threshold)
            try:
                results = processor.post_process_instance_segmentation(
                    outputs, return_binary_maps=True, **pp_kwargs)
                binary_maps = True
            except TypeError:
                binary_maps = False
                try:
                    results = processor.post_process_instance_segmentation(
                        outputs, **pp_kwargs)
                except TypeError:
                    results = processor.post_process_instance_segmentation(
                        outputs, target_sizes=[(H, W)] * B,
                        threshold=conf_threshold)

            # Retrieve GT for this batch from the collated tensors.
            gt_mask_batch  = batch.get("mask_labels",  [None] * B)
            gt_label_batch = batch.get("class_labels", [None] * B)

            for b_idx in range(len(results)):
                r        = results[b_idx]
                seg      = r.get("segmentation", None)
                seg_info = r.get("segments_info", [])

                pred_masks_list:  List[torch.Tensor] = []
                pred_labels_list: List[int]          = []
                pred_scores_list: List[float]        = []

                if seg is not None and len(seg_info) > 0:
                    seg_t = seg if isinstance(seg, torch.Tensor) else torch.as_tensor(seg)
                    if binary_maps and seg_t.dim() == 3:
                        for k, info in enumerate(seg_info):
                            pred_masks_list.append(seg_t[k].bool().cpu())
                            pred_labels_list.append(int(info["label_id"]))
                            pred_scores_list.append(float(info["score"]))
                    else:
                        for info in seg_info:
                            sid = info["id"]
                            pred_masks_list.append((seg_t == sid).bool().cpu())
                            pred_labels_list.append(int(info["label_id"]))
                            pred_scores_list.append(float(info["score"]))

                if pred_masks_list:
                    pred_masks  = torch.stack(pred_masks_list)
                    pred_labels = torch.tensor(pred_labels_list, dtype=torch.long)
                    pred_scores = torch.tensor(pred_scores_list, dtype=torch.float)
                else:
                    pred_masks  = torch.zeros((0, H, W), dtype=torch.bool)
                    pred_labels = torch.zeros((0,), dtype=torch.long)
                    pred_scores = torch.zeros((0,), dtype=torch.float)

                # GT: from collated mask_labels / class_labels tensors.
                gt_ml = gt_mask_batch[b_idx]
                gt_cl = gt_label_batch[b_idx]
                if gt_ml is not None and len(gt_ml) > 0:
                    gt_masks  = gt_ml.bool().cpu()
                    gt_labels = gt_cl.long().cpu()
                else:
                    gt_masks  = torch.zeros((0, H, W), dtype=torch.bool)
                    gt_labels = torch.zeros((0,), dtype=torch.long)

                metric.update(
                    [{"masks": pred_masks, "labels": pred_labels, "scores": pred_scores}],
                    [{"masks": gt_masks,   "labels": gt_labels}],
                )
                tp_f, fp_f, fn_f = _match_frame(
                    pred_masks, pred_labels, pred_scores,
                    gt_masks, gt_labels, num_classes, iou_thresh=0.5)
                for c in range(num_classes):
                    tp_all[c] += tp_f[c]
                    fp_all[c] += fp_f[c]
                    fn_all[c] += fn_f[c]

                sample_idx += 1
                if log_every and sample_idx % log_every == 0:
                    print(f"  [eval] {sample_idx}/{len(dataset)}")

    finally:
        # Always restore training mode, even if eval raised an exception.
        if was_training:
            model.train()

    coco   = metric.compute()
    # Release the accumulated masks (torchmetrics stores every pred+GT until
    # compute) — important on dense val splits to keep host RAM bounded.
    metric.reset()
    # torchmetrics names the top recall key after the largest detection
    # threshold (mar_100 by default, mar_500 here), so look it up by the
    # highest mar_<N> key rather than hard-coding mar_100.
    _mar_keys = [k for k in coco
                 if k.startswith("mar_") and k.rsplit("_", 1)[-1].isdigit()]
    _mar_top = (max(_mar_keys, key=lambda k: int(k.rsplit("_", 1)[-1]))
                if _mar_keys else None)
    _mar_val = coco[_mar_top] if _mar_top else torch.tensor(0.0)
    P_per  = [tp_all[c] / max(tp_all[c] + fp_all[c], 1) for c in range(num_classes)]
    R_per  = [tp_all[c] / max(tp_all[c] + fn_all[c], 1) for c in range(num_classes)]
    P      = sum(tp_all) / max(sum(tp_all) + sum(fp_all), 1)
    R      = sum(tp_all) / max(sum(tp_all) + sum(fn_all), 1)

    def _scalar(x):
        if isinstance(x, torch.Tensor):
            return float(x.item()) if x.numel() == 1 else x.tolist()
        return x

    map_per = coco.get("map_per_class", None)
    if isinstance(map_per, torch.Tensor):
        map_per = map_per.tolist()
        if not isinstance(map_per, list):
            map_per = [map_per]

    return {
        "mAP50":         _scalar(coco.get("map_50",  torch.tensor(0.0))),
        "mAP50-95":      _scalar(coco.get("map",     torch.tensor(0.0))),
        "mAR_100":       _scalar(_mar_val),   # mAR at --eval-max-detections
        "P":             P,
        "R":             R,
        "mAP_per_class": map_per,
        "P_per_class":   P_per,
        "R_per_class":   R_per,
        "tp_per_class":  tp_all,
        "fp_per_class":  fp_all,
        "fn_per_class":  fn_all,
    }


class PeriodicMetricsCallback(TrainerCallback):
    """Run :func:`evaluate_full` on val every ``every_n_epochs`` epochs and
    save results in CSV (YOLO-style) + JSON-lines (full detail).

    Epoch tracking uses an explicit ``_last_eval_epoch`` counter rather than
    relying on ``state.epoch % N == 0``, which can misfire at float
    epoch-boundary values reported by the Trainer.
    """

    def __init__(self, eval_dataset, processor, num_classes: int,
                 id2label: Dict[int, str], *,
                 every_n_epochs: int = 5, conf_threshold: float = 0.5,
                 mask_threshold: float = 0.4, overlap_threshold: float = 0.5,
                 max_detections: int = 500,
                 eval_batch_size: int = 4, eval_num_workers: int = 2,
                 total_epochs: int = None,
                 out_dir: Path = None,
                 best_metric_key: str = "mAP50-95",
                 best_dir: Path = None):
        self.eval_dataset       = eval_dataset
        self.processor          = processor
        self.num_classes        = int(num_classes)
        self.id2label           = dict(id2label)
        self.every_n_epochs     = max(1, int(every_n_epochs))
        self.conf_threshold     = float(conf_threshold)
        self.mask_threshold     = float(mask_threshold)
        self.overlap_threshold  = float(overlap_threshold)
        self.max_detections     = int(max_detections)
        self.eval_batch_size    = int(eval_batch_size)
        self.eval_num_workers   = int(eval_num_workers)
        self.total_epochs       = int(total_epochs) if total_epochs else None
        self.out_dir            = Path(out_dir) if out_dir else None
        self.history: List[Dict] = []
        self.best_metric_key    = best_metric_key
        self.best_dir           = Path(best_dir) if best_dir else None
        self.best_value         = -float("inf")
        self.best_epoch: int    = -1
        self._last_eval_epoch: int = 0   # explicit tracker avoids float jitter
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        # Resume safety: restore the previous best so a resumed run does not
        # overwrite checkpoint-best/ with a worse model on its first eval.
        if self.best_dir is not None:
            bj = self.best_dir / "best.json"
            if bj.exists():
                try:
                    prev = json.loads(bj.read_text())
                    if prev.get("metric") == self.best_metric_key:
                        self.best_value = float(prev.get("value", -float("inf")))
                        self.best_epoch = int(prev.get("epoch", -1))
                        print(f"Resuming best-checkpoint tracking: "
                              f"{self.best_metric_key}={self.best_value:.4f} "
                              f"@ epoch {self.best_epoch}")
                except Exception:
                    pass

    def on_epoch_end(self, args, state, control, **kwargs):
        # Use floor so fractional epoch values (e.g. 4.999) round correctly.
        ep = int(state.epoch or 0)
        # Always evaluate on the final epoch so a last-epoch improvement is not
        # missed when epochs is not a multiple of every_n_epochs.
        is_final = self.total_epochs is not None and ep >= self.total_epochs
        if ep <= 0 or ep == self._last_eval_epoch:
            return control
        if ep % self.every_n_epochs != 0 and not is_final:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        self._last_eval_epoch = ep
        device = next(model.parameters()).device

        print(f"\n[epoch {ep}] computing detection metrics on val "
              f"(IoU=0.5 P/R + COCO mAP)...")
        m = evaluate_full(
            model, self.eval_dataset, self.processor,
            conf_threshold=self.conf_threshold,
            num_classes=self.num_classes,
            device=device,
            batch_size=self.eval_batch_size,
            num_workers=self.eval_num_workers,
            mask_threshold=self.mask_threshold,
            overlap_threshold=self.overlap_threshold,
            max_detections=self.max_detections,
        )

        # Pretty print (YOLO-like one-line summary, then per-class)
        print(f"  P={m['P']:.3f}  R={m['R']:.3f}  "
              f"mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50-95']:.3f}  "
              f"mAR100={m['mAR_100']:.3f}")
        for c in range(self.num_classes):
            name = self.id2label.get(c, str(c))
            ap = (m["mAP_per_class"][c]
                  if m["mAP_per_class"] and c < len(m["mAP_per_class"])
                  else float("nan"))
            print(f"    {name:>12}: "
                  f"P={m['P_per_class'][c]:.3f}  "
                  f"R={m['R_per_class'][c]:.3f}  "
                  f"mAP50-95={ap:.3f}  "
                  f"TP={m['tp_per_class'][c]} "
                  f"FP={m['fp_per_class'][c]} "
                  f"FN={m['fn_per_class'][c]}")

        # ---- Best-checkpoint tracking ----
        is_best = False
        if (self.best_dir is not None
                and self.best_metric_key != "none"
                and self.best_metric_key in m):
            current = float(m[self.best_metric_key])
            if current > self.best_value:
                self.best_value = current
                self.best_epoch = ep
                is_best = True
                self.best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(self.best_dir))
                self.processor.save_pretrained(str(self.best_dir))
                (self.best_dir / "best.json").write_text(json.dumps({
                    "epoch": ep,
                    "metric": self.best_metric_key,
                    "value": current,
                    "all_metrics": {k: v for k, v in m.items()
                                    if not isinstance(v, list)},
                }, indent=2))
                print(f"  *** new best {self.best_metric_key} = "
                      f"{current:.4f} -> saved to {self.best_dir}")

        record = {"epoch": ep, "is_best": is_best, **m}
        self.history.append(record)
        if self.out_dir is not None:
            # JSON-lines: one record per eval, full detail
            with (self.out_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            # Consolidated history (rewritten each eval)
            (self.out_dir / "metrics_history.json").write_text(
                json.dumps(self.history, indent=2))
            # YOLO-style summary CSV
            csv_path = self.out_dir / "metrics.csv"
            existed = csv_path.exists()
            with csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                if not existed:
                    w.writerow(["epoch", "P", "R", "mAP50",
                                "mAP50-95", "mAR100"])
                w.writerow([ep,
                            f"{m['P']:.4f}",       f"{m['R']:.4f}",
                            f"{m['mAP50']:.4f}",   f"{m['mAP50-95']:.4f}",
                            f"{m['mAR_100']:.4f}"])
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="seg dataset root (the dir containing images/, "
                         "instances/, masks/, train.txt, val.txt)")
    ap.add_argument("--out", type=Path, required=True,
                    help="checkpoint + log output dir")
    ap.add_argument("--model",
                    default="facebook/mask2former-swin-tiny-coco-instance",
                    help="HF model id. Swap *-tiny -> *-small/*-base/*-large "
                         "for more accuracy at higher compute.")
    ap.add_argument("--num-queries", type=int, default=400,
                    help="number of object queries = HARD cap on instances "
                         "predicted per image. COCO checkpoints default to 100; "
                         "dense frames need many more. Default 400 covers up to "
                         "~200 cells in 'parts' mode (body+flag). Use ~500 for "
                         "'--classes all' (animal+body+flag). More queries cost "
                         "decoder compute + memory.")
    ap.add_argument("--train-num-points", type=int, default=25088,
                    help="points sampled for the mask/dice loss (M2F default "
                         "12544). Raised to give thin flagella (3-5 px) more "
                         "gradient — they occupy <1%% of pixels.")
    ap.add_argument("--importance-sample-ratio", type=float, default=0.9,
                    help="fraction of mask-loss points drawn from uncertain/"
                         "boundary regions (M2F default 0.75). Higher = more "
                         "focus on thin-structure boundaries.")
    ap.add_argument("--classes", choices=list(CLASS_SETS), default="parts",
                    help="parts = body+flag (default, recommended). "
                         "animal = whole cell, 1 class. "
                         "all = animal+body+flag, overlapping instances.")
    ap.add_argument("--merge-flagella", dest="split_flagella",
                    action="store_false",
                    help="collapse a dividing cell's two flagella into ONE "
                         "flagellum instance. Default: each flagellum is its "
                         "own instance (uses the flag_instances/ map; requires "
                         "a dataset built with the per-flagellum map).")
    ap.set_defaults(split_flagella=True)
    ap.add_argument("--image-size", type=int, default=640,
                    help="training resolution (multiple of 32). Each frame "
                         "(image + label maps) is resized to this size on load "
                         "if it differs, so it is safe to train at 1024 on a "
                         "640 dataset. NOTE: upscaling interpolates, it does not "
                         "add real detail — for thin 3-5 px flagella, rendering "
                         "the dataset natively at 1024 is better than upscaling. "
                         "Also recorded in train_meta.json + processor config.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="per-device. For Swin-B/L drop to 2 and use --grad-accum.")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LR for the mask/query head (randomly re-initialised). "
                         "The backbone uses --lr-backbone.")
    ap.add_argument("--lr-backbone", type=float, default=1e-5,
                    help="LR for the pretrained Swin backbone. Should be ~10x "
                         "lower than --lr to avoid overwriting pretrained features.")
    ap.add_argument("--lr-decoder", type=float, default=3e-5,
                    help="LR for the pretrained pixel + transformer decoders "
                         "(between --lr-backbone and --lr). Lets them adapt "
                         "without being overwritten at the full head LR.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="gradient clipping max norm. M2F's transformer decoder "
                         "can produce large spikes; 1.0 is the standard value.")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="disable albumentations training augmentation.")
    ap.set_defaults(augment=True)
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers per device. Dense frames do heavy "
                         "per-instance mask building on the CPU, so 8-16 helps "
                         "keep an A6000 fed.")
    ap.add_argument("--min-area", type=int, default=64,
                    help="drop instance masks smaller than this many pixels "
                         "(default 64 — filters sub-cell noise). Applies to "
                         "body/animal instances. The original default of 6 is "
                         "almost certainly too small.")
    ap.add_argument("--min-area-flag", type=int, default=12,
                    help="separate, lower min-area floor for the FLAGELLUM "
                         "class (default 12). Thin flagella (3-5 px wide) can be "
                         "well under --min-area; a global floor silently deletes "
                         "short/clipped flagella and hurts recall.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="run full mAP/P/R eval on val every N epochs "
                         "(default 5). Set to 1 for every epoch. The Trainer "
                         "still tracks val loss every epoch on top of this.")
    ap.add_argument("--eval-conf-threshold", type=float, default=0.5,
                    help="object-confidence threshold for predictions during "
                         "the mAP/P/R eval (matches YOLO's predict default range).")
    ap.add_argument("--eval-mask-threshold", type=float, default=0.4,
                    help="per-pixel mask binarisation threshold at eval "
                         "post-processing (M2F default 0.5). Lowered to 0.4 to "
                         "help thin flagella survive binarisation.")
    ap.add_argument("--eval-overlap-threshold", type=float, default=0.5,
                    help="overlap_mask_area_threshold at eval post-processing "
                         "(M2F default 0.8). Lowered so thin flagella overlapping "
                         "a body mask are not suppressed.")
    ap.add_argument("--eval-max-detections", type=int, default=500,
                    help="max detections/image for the COCO mAP/mAR eval "
                         "(torchmetrics default 100). Set ABOVE your densest "
                         "frame's instance count or dense frames are truncated "
                         "and mAP is under-reported. 500 covers ~150 cells in "
                         "'parts'/'all'. Does NOT affect the custom P/R.")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"],
                    default="bf16",
                    help="training precision. bf16 (default) is correct for "
                         "Ampere+ (A6000). fp16 can NaN in Mask2Former's matcher "
                         "/ mask loss — avoid unless you know your GPU lacks bf16.")
    ap.add_argument("--save-best-metric",
                    choices=["mAP50-95", "mAP50", "mAR_100", "P", "R", "none"],
                    default="mAP50-95",
                    help="Save <out>/checkpoint-best/ whenever this val "
                         "metric improves. 'none' disables best tracking.")
    ap.add_argument("--save-total-limit", type=int, default=3,
                    help="number of most-recent iterative epoch checkpoints "
                         "to keep (older are deleted to save disk).")
    ap.add_argument("--resume", type=str, default=None,
                    metavar="CHECKPOINT_DIR",
                    help="resume training from a checkpoint directory "
                         "(e.g. runs/m2f_v1/checkpoint-1200).")
    ap.add_argument("--smoke", action="store_true",
                    help="run 2 train steps and 1 eval pass then exit — "
                         "verifies data, model wiring, and eval pipeline "
                         "without committing to a long run.")
    args = ap.parse_args()

    # Fix CUDA non-determinism for reproducibility; benchmark=True is safe
    # for fixed-resolution training and selects faster conv kernels after
    # a one-time warmup cost.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg      = CLASS_SETS[args.classes]
    id2label = cfg["id2label"]

    processor = build_processor(args.model, image_size=args.image_size)
    model     = build_model(args.model, id2label,
                            num_queries=args.num_queries,
                            train_num_points=args.train_num_points,
                            importance_sample_ratio=args.importance_sample_ratio)
    print(f"Model: num_queries={model.config.num_queries}  "
          f"train_num_points={model.config.train_num_points}  "
          f"importance_sample_ratio={model.config.importance_sample_ratio}")

    # Save processor immediately so a checkpoint always exists, even if
    # training is interrupted before the end-of-run save.
    args.out.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.out)

    train_ds = LeishInstanceDataset(
        args.data_root, "train",
        class_set=args.classes,
        min_area=args.min_area,
        min_area_flag=args.min_area_flag,
        split_flagella=args.split_flagella,
        augment=args.augment,
        image_size=args.image_size,
    )
    val_ds = LeishInstanceDataset(
        args.data_root, "val",
        class_set=args.classes,
        min_area=args.min_area,
        min_area_flag=args.min_area_flag,
        split_flagella=args.split_flagella,
        augment=False,   # never augment val
        image_size=args.image_size,
    )
    print(f"Train: {len(train_ds)} frames | Val: {len(val_ds)} frames")
    print(f"Classes ({len(id2label)}): {id2label}")
    print(f"Augmentation: {'ON (albumentations)' if train_ds._aug is not None else 'OFF'}")

    # Quick smoke check on the first sample so failures surface early.
    sample = train_ds[0]
    print(f"Sample 0: image {sample['image'].shape}  "
          f"instances {len(sample['classes'])}  "
          f"class hist {dict(zip(*np.unique(sample['classes'], return_counts=True))) if sample['classes'] else {}}")

    collate = make_collator(processor)

    # Precision: bf16 is the safe choice for Mask2Former on Ampere+ (A6000).
    # fp16 can NaN in the Hungarian matcher / mask loss, so it's opt-in only.
    use_bf16 = use_fp16 = False
    if args.precision == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            print("WARNING: bf16 requested but this GPU does not support it — "
                  "falling back to fp32.")
    elif args.precision == "fp16":
        if torch.cuda.is_available():
            use_fp16 = True
            print("WARNING: fp16 selected — Mask2Former's matcher / mask loss "
                  "can produce NaNs in fp16. Prefer bf16 on Ampere+ GPUs.")
        else:
            print("WARNING: fp16 requested but no CUDA device — using fp32.")
    # else fp32: both flags stay False.

    # Compute total optimizer steps for the scheduler EXACTLY as the HF Trainer
    # counts them: ceil(len/batch) batches per epoch, floored by grad_accum.
    # A plain len//(batch*accum) undercounts (non-divisible N or accum>1) and
    # leaves the cosine LR at 0 for the final steps.
    steps_per_epoch = max(
        1, math.ceil(len(train_ds) / args.batch_size) // args.grad_accum)
    num_training_steps = steps_per_epoch * (1 if args.smoke else args.epochs)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr,
        lr_decoder=args.lr_decoder,
        weight_decay=args.weight_decay,
        num_training_steps=num_training_steps,
        warmup_ratio=args.warmup_ratio,
    )

    targs = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(2 if args.smoke else -1),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # learning_rate is set per-group in the custom optimizer above;
        # this value is used only by Trainer internals that need a scalar LR
        # (e.g. logging). Set it to the head LR as the representative value.
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=(1 if args.smoke else 20),
        # No Trainer-side eval: it would run a full val forward pass every epoch
        # only to produce eval_loss, which nothing consumes (no compute_metrics,
        # load_best_model_at_end off). The PeriodicMetricsCallback owns eval and
        # best-checkpoint selection. Avoids traversing val twice on metric epochs.
        eval_strategy="no",
        save_strategy=("no" if args.smoke else "epoch"),
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        # CRITICAL: keep custom keys (mask_labels, class_labels) on batches.
        # Default True would strip everything Trainer doesn't recognise.
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )

    callbacks = []
    if not args.smoke:
        if not _HAVE_TM:
            print("WARNING: torchmetrics not installed — skipping periodic "
                  "mAP/P/R eval. Install with: "
                  "pip install 'torchmetrics[detection]' pycocotools")
        else:
            callbacks.append(PeriodicMetricsCallback(
                eval_dataset=val_ds,
                processor=processor,
                num_classes=len(id2label),
                id2label=id2label,
                every_n_epochs=args.eval_every,
                conf_threshold=args.eval_conf_threshold,
                mask_threshold=args.eval_mask_threshold,
                overlap_threshold=args.eval_overlap_threshold,
                max_detections=args.eval_max_detections,
                eval_batch_size=args.batch_size,
                eval_num_workers=args.workers,
                total_epochs=args.epochs,
                out_dir=args.out / "metrics",
                best_metric_key=args.save_best_metric,
                best_dir=(args.out / "checkpoint-best"
                          if args.save_best_metric != "none" else None),
            ))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        callbacks=callbacks or None,
        optimizers=(optimizer, scheduler),
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # ---- Smoke: run one eval pass to verify the full pipeline ----
    if args.smoke and _HAVE_TM:
        print("\n[smoke] running one eval pass to verify eval pipeline...")
        device = next(model.parameters()).device
        m = evaluate_full(model, val_ds, processor,
                          conf_threshold=args.eval_conf_threshold,
                          num_classes=len(id2label), device=device,
                          batch_size=args.batch_size,
                          num_workers=args.workers,
                          mask_threshold=args.eval_mask_threshold,
                          overlap_threshold=args.eval_overlap_threshold,
                          max_detections=args.eval_max_detections)
        print(f"[smoke eval] P={m['P']:.3f}  R={m['R']:.3f}  "
              f"mAP50={m['mAP50']:.3f}")

    if not args.smoke:
        # Save processor + metadata so the checkpoint is fully self-contained.
        processor.save_pretrained(args.out)
        meta = {
            "model":                   args.model,
            "classes":                 args.classes,
            "id2label":                id2label,
            "data_root":               str(args.data_root),
            "epochs":                  args.epochs,
            "image_size":              args.image_size,
            "do_resize":               False,
            "num_queries":             args.num_queries,
            "train_num_points":        args.train_num_points,
            "importance_sample_ratio": args.importance_sample_ratio,
            "precision":               args.precision,
            "lr":                      args.lr,
            "lr_backbone":             args.lr_backbone,
            "lr_decoder":              args.lr_decoder,
            "augmentation":            args.augment and _HAVE_ALB,
            "min_area":                args.min_area,
            "min_area_flag":           args.min_area_flag,
            "eval_mask_threshold":     args.eval_mask_threshold,
            "eval_overlap_threshold":  args.eval_overlap_threshold,
            "max_grad_norm":           args.max_grad_norm,
        }
        (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved processor + train_meta to {args.out}")


if __name__ == "__main__":
    main()