"""
Build SEGMENTATION training datasets from the synthetic Leishmania renderer.

Companion to ``dataset_builder.py``: it takes the *same* config input (the
same CLI subcommands and the same multi-setup YAML) but, instead of a
SLEAP ``labels.slp``, it writes segmentation labels for two model families:

  1. Ultralytics YOLO-seg (YOLOv8/v11-seg) — normalized polygon labels.
  2. Semantic-segmentation masks (e.g. a U-Net) — single-channel label PNGs.

Three nested classes are produced, exactly as requested:

      class 0 = animal     whole parasite silhouette (body + flagellum)
      class 1 = body        cell body only
      class 2 = flagellum   flagellum only (the part outside the body)

For YOLO each parasite emits up to three polygon instances (one per class).
For the semantic masks the label PNG uses 0=background, 1=body, 2=flagellum;
the whole-animal region is simply every non-zero pixel, so it needs no
separate channel. Body and flagellum are disjoint (the proximal flagellum
embedded in the cell body is counted as body, since it is not separately
visible in phase contrast).

Occlusion is handled the same way as the keypoint labels: a cell hidden
behind a denser neighbour does not claim those pixels (see
``synthetic_leishmania.render_scene``'s phase winner-map).

Output layout (one dataset; YOLO and mask trees share ``images/``)::

    out/
      images/<tag>_000000.png      rendered frames (shared by both formats)
      labels/<tag>_000000.txt      YOLO-seg polygons (sibling of images/)
      groups/<tag>_000000.json     animal_id per YOLO label line (association)
      masks/<tag>_000000.png       semantic part map {0,1,2} (for U-Net)
      instances/<tag>_000000.png   16-bit animal-id map {0=bg, 1..N} (association)
      flag_instances/<tag>_000000.png  16-bit per-flagellum id map {0=bg, 1..F}
                                   (a dividing cell's two flagella get distinct
                                   ids; parent animal = instances[ at same px ])
      train.txt  val.txt           image-path splits used by data.yaml
      data.yaml                    ultralytics dataset descriptor
      seg_metadata.json            class names, counts, settings

Associating body/flagellum to a unique animal
---------------------------------------------
Each parasite gets an ``animal_id`` (1..N, unique per frame). The link is
written as exact ground truth in two ways:

  * Masks/U-Net: ``instances/<stem>.png`` holds the animal id at every pixel.
    With the part map, animal ``k``'s body = ``(instances==k) & (masks==1)``
    and flagellum = ``(instances==k) & (masks==2)``.
  * YOLO: ``groups/<stem>.json`` lists an ``animal_id`` per label line, in the
    same order as ``labels/<stem>.txt``, so each body/flagellum polygon is
    tied to its animal.

A trained YOLO model emits independent detections with no ids; regroup them
post-hoc with :func:`group_parts_to_animals` (assigns each predicted body /
flagellum to the animal mask that contains it).

Examples
--------
    python segmentation_dataset_creator.py random --frames 300 --out seg/synth1
    python segmentation_dataset_creator.py video  --frames 600 --out seg/vid1
    python segmentation_dataset_creator.py multi  configs/training.yaml \\
        --out seg/leish_seg
    python segmentation_dataset_creator.py template -o configs/training.yaml

The ``multi`` command reuses the YAML resolution from dataset_builder, so a
single config can drive both the SLEAP and the segmentation generators.
Segmentation-specific options can be set as top-level YAML keys
(``seg_val_fraction``, ``seg_min_polygon_area``, ``seg_image_format``,
``seg_write_yolo``, ``seg_write_masks``) or via CLI flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import imageio.v2 as iio

import synthetic_leishmania as L
import dataset_builder as DB


# ----------------------------------------------------------------------------
# Class definitions
# ----------------------------------------------------------------------------

# YOLO instance classes (nested): animal = body | flagellum.
CLASS_NAMES = ["animal", "body", "flagellum"]
ANIMAL, BODY, FLAGELLUM = 0, 1, 2

# Which instance classes to emit into the YOLO labels. Each set starts at
# class id 0 and is contiguous, so the ids double as ultralytics name indices.
#   "all"    -> animal + body + flagellum (nested, the original 3-class scheme)
#   "animal" -> single-class animal detector/segmenter (the instance stage of
#               the recommended pipeline; pair with the semantic mask UNet)
_YOLO_CLASS_SETS = {
    "all": [ANIMAL, BODY, FLAGELLUM],
    "animal": [ANIMAL],
}

# Semantic label-map values (mutually exclusive; animal = any non-zero pixel).
SEMANTIC_BODY = 1
SEMANTIC_FLAGELLUM = 2


# ----------------------------------------------------------------------------
# Mask -> polygon conversion
# ----------------------------------------------------------------------------

def _find_contours(mask_u8: np.ndarray):
    """cv2.findContours wrapper tolerant of the 2- vs 3-tuple return across
    OpenCV versions."""
    res = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]


# --- hole encoding (keyhole / bridge) ---------------------------------------
# A YOLO-seg instance is a SINGLE closed ring: the format has no native way to
# express a hole (an enclosed background region) or several disjoint rings for
# one instance. So a mask with a genuine hole — e.g. the ANIMAL silhouette of a
# dividing cell whose two flagella cross each other, enclosing a background
# pocket — would, with a plain outer-contour trace (RETR_EXTERNAL), have that
# pocket filled in. The community work-around (ultralytics/ultralytics#1106) is
# the "keyhole": trace the outer ring AND the hole rings (RETR_CCOMP), then
# stitch each hole into its parent with a pair of near-coincident bridge edges,
# so a single self-touching ring encodes the hole and rasterises with the pocket
# left empty. NOTE this only addresses true enclosed holes; an OPEN concavity
# (the wedge between two merely-diverging flagella, or between a flagellum and
# the body) is not a hole, has no child contour, and is governed by the
# approxPolyDP simplification (`epsilon_frac`) instead — bridging does nothing
# for it.

def _contour_is_clockwise(c: np.ndarray) -> bool:
    """Shoelace sign for an (N,1,2) cv2 contour (image y points down, so a
    clockwise screen path has negative signed area under this convention)."""
    pts = c.reshape(-1, 2).astype(np.float64)
    nxt = np.roll(pts, -1, axis=0)
    return float(np.sum((nxt[:, 0] - pts[:, 0]) * (nxt[:, 1] + pts[:, 1]))) < 0


def _nearest_pair_idx(a: np.ndarray, b: np.ndarray):
    """Indices (i, j) of the closest vertices between two (N,1,2) contours."""
    pa = a.reshape(-1, 2).astype(np.float64)
    pb = b.reshape(-1, 2).astype(np.float64)
    d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(i), int(j)


def _bridge_merge(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Splice a hole `child` into its `parent` ring via a keyhole bridge.

    Parent is forced clockwise and child counter-clockwise (opposite winding),
    then both are connected at their nearest pair of vertices with two
    coincident edges, yielding one ring whose interior excludes the hole.
    """
    if not _contour_is_clockwise(parent):
        parent = parent[::-1]
    if _contour_is_clockwise(child):
        child = child[::-1]
    i, j = _nearest_pair_idx(parent, child)
    merged = np.concatenate([
        parent[:i + 1],             # parent up to the bridge point
        child[j:], child[:j + 1],   # around the whole hole, back to its start
        parent[i:],                 # rest of the parent (re-enters at bridge)
    ], axis=0)
    return merged


def mask_to_polygons(mask_bool: np.ndarray, x0: int, y0: int,
                     img_w: int, img_h: int,
                     min_area: float,
                     epsilon_frac: float = 0.004,
                     largest_only: bool = False,
                     carve_holes: bool = True) -> List[np.ndarray]:
    """Convert a tile-local boolean mask into normalized YOLO-seg polygons.

    Each returned array is (N, 2) of x,y coordinates normalized to [0, 1]
    against the full image size. `(x0, y0)` is the tile's top-left offset in
    the full image. A mask that splits into several disconnected pieces (e.g.
    a body bisected by an occluding neighbour) yields one polygon per piece,
    unless `largest_only` is set, in which case only the biggest piece is
    returned (used to guarantee a single instance per cell per class).

    When `carve_holes` is set (default), enclosed background holes are kept out
    of the polygon via the keyhole/bridge method (see above); otherwise the
    legacy outer-contour-only trace is used (holes get filled).
    """
    if not mask_bool.any():
        return []
    mask_u8 = mask_bool.astype(np.uint8)

    if not carve_holes:
        contours = list(_find_contours(mask_u8))
        if largest_only and contours:
            contours = [max(contours, key=cv2.contourArea)]
        rings = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            rings.append(cv2.approxPolyDP(c, epsilon_frac * cv2.arcLength(c, True), True))
        return _rings_to_norm_polys(rings, x0, y0, img_w, img_h)

    # Hole-aware trace: outer rings plus their hole children.
    res = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    hierarchy = res[1] if len(res) == 2 else res[2]
    if not contours or hierarchy is None:
        return []
    hierarchy = hierarchy[0]   # (N, 4): [next, prev, first_child, parent]
    areas = [cv2.contourArea(c) for c in contours]
    approx = [cv2.approxPolyDP(c, epsilon_frac * cv2.arcLength(c, True), True)
              for c in contours]

    # Outer (parent) contours = those with no parent in the hierarchy.
    outer_idx = [i for i in range(len(contours))
                 if hierarchy[i][3] < 0 and areas[i] >= min_area
                 and len(approx[i]) >= 3]
    if largest_only and outer_idx:
        outer_idx = [max(outer_idx, key=lambda i: areas[i])]

    rings = []
    for oi in outer_idx:
        ring = approx[oi]
        # Merge each non-trivial hole child of this outer contour.
        for j in range(len(contours)):
            if (hierarchy[j][3] == oi and areas[j] >= min_area
                    and len(approx[j]) >= 3):
                ring = _bridge_merge(ring, approx[j])
        rings.append(ring)
    return _rings_to_norm_polys(rings, x0, y0, img_w, img_h)


def _rings_to_norm_polys(rings, x0, y0, img_w, img_h) -> List[np.ndarray]:
    """Normalize a list of (M,1,2)/(M,2) integer rings to [0,1] YOLO polygons."""
    polys: List[np.ndarray] = []
    for ring in rings:
        poly = np.asarray(ring).reshape(-1, 2).astype(np.float64)
        if len(poly) < 3:
            continue
        poly[:, 0] = (poly[:, 0] + x0) / img_w
        poly[:, 1] = (poly[:, 1] + y0) / img_h
        np.clip(poly, 0.0, 1.0, out=poly)
        polys.append(poly)
    return polys


def _yolo_line(class_id: int, poly: np.ndarray) -> str:
    coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
    return f"{class_id} {coords}"


def group_parts_to_animals(animal_masks, body_masks, flag_masks, *,
                           min_overlap: float = 0.5):
    """Associate predicted body / flagellum masks to predicted animal masks.

    A trained YOLO-seg model emits independent detections with no link between
    an animal and its parts. This regroups them by containment: each body /
    flagellum is assigned to the animal whose mask covers the largest fraction
    of it, requiring at least `min_overlap` of the part's area to lie inside
    that animal.

    Parameters
    ----------
    animal_masks, body_masks, flag_masks : sequence of 2-D boolean arrays
        Per-detection masks for classes 0/1/2 (e.g. ``results.masks.data`` rows
        split by ``results.boxes.cls``), all at the same H x W.
    min_overlap : float
        Minimum fraction of a part's pixels inside an animal to assign it.

    Returns
    -------
    list of dict
        Aligned with `animal_masks`; entry k is
        ``{"body": [indices...], "flag": [indices...]}`` — the body/flagellum
        detections owned by animal k. Parts matching no animal are dropped.
    """
    animal_masks = [np.asarray(m, dtype=bool) for m in animal_masks]

    def assign(parts):
        owners = [[] for _ in animal_masks]
        for j, pm in enumerate(parts):
            pm = np.asarray(pm, dtype=bool)
            area = int(pm.sum())
            if area == 0:
                continue
            best_k, best_frac = -1, float(min_overlap)
            for k, am in enumerate(animal_masks):
                frac = float(np.logical_and(pm, am).sum()) / area
                if frac >= best_frac:
                    best_k, best_frac = k, frac
            if best_k >= 0:
                owners[best_k].append(j)
        return owners

    body_owners = assign(body_masks)
    flag_owners = assign(flag_masks)
    return [{"body": body_owners[k], "flag": flag_owners[k]}
            for k in range(len(animal_masks))]


# ----------------------------------------------------------------------------
# Dataset writer
# ----------------------------------------------------------------------------

class SegmentationDataset:
    """Accumulates rendered frames + masks into a YOLO-seg / semantic-mask
    dataset on disk. One writer can span many setups (multi mode): just call
    ``add_frame`` with a unique stem per frame, then ``close`` once."""

    def __init__(self, out_dir, *,
                 val_fraction: float = 0.1,
                 min_polygon_area: float = 6.0,
                 write_yolo: bool = True,
                 write_masks: bool = True,
                 image_format: str = "png",
                 split_occluded: bool = False,
                 split_flagella: bool = True,
                 carve_holes: bool = True,
                 min_visible_frac: float = 0.05,
                 yolo_classes: str = "all",
                 seed: int = 0):
        self.root = Path(out_dir)
        self.val_fraction = float(val_fraction)
        self.min_area = float(min_polygon_area)
        self.write_yolo = bool(write_yolo)
        self.write_masks = bool(write_masks)
        self.ext = image_format.lower().lstrip(".")
        if yolo_classes not in _YOLO_CLASS_SETS:
            raise ValueError(f"yolo_classes must be one of "
                             f"{list(_YOLO_CLASS_SETS)}, got {yolo_classes!r}")
        self.yolo_classes = yolo_classes
        self.yolo_class_ids = _YOLO_CLASS_SETS[yolo_classes]
        # split_occluded=False (default): one YOLO instance per cell per class,
        # using the full (amodal) silhouette, so occlusion never splits an
        # animal into several detections. True: visible (modal) regions, one
        # polygon per connected piece (an occluded cell may split).
        self.split_occluded = bool(split_occluded)
        # split_flagella=True (default): each flagellum becomes its own
        # FLAGELLUM polygon, sharing the parent cell's animal_id in the group
        # json — so dividing cells with two flagella (2F stage) produce two
        # distinct instances instead of one polygon that fuses both arms (and,
        # when the two arms touch or cross, fills the gap between them). Set
        # False to merge all of a cell's flagella into ONE flagellum polygon
        # (legacy). The semantic mask is unaffected either way (flagellum is a
        # per-pixel class, not per-instance).
        self.split_flagella = bool(split_flagella)
        # carve_holes=True (default): enclosed background holes (e.g. the pocket
        # between two crossing flagella in the ANIMAL silhouette) are kept out
        # of the emitted polygon via the keyhole/bridge method, instead of being
        # filled by a plain outer-contour trace. See mask_to_polygons.
        self.carve_holes = bool(carve_holes)
        # Drop cells whose visible fraction is below this (buried behind a
        # neighbour) so we don't label essentially-invisible animals.
        self.min_visible_frac = float(min_visible_frac)
        self.split_rng = np.random.default_rng(seed)

        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.groups_dir = self.root / "groups"
        self.masks_dir = self.root / "masks"
        self.instances_dir = self.root / "instances"
        # Per-flagellum instance map: each flagellum gets a globally-unique id
        # (within a frame), so a dividing cell's two flagella can be recovered
        # as TWO instances. Its parent animal_id is read off the instances map
        # at the same pixels (a flagellum is always a subset of one animal).
        self.flag_instances_dir = self.root / "flag_instances"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.write_yolo:
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            self.groups_dir.mkdir(parents=True, exist_ok=True)
        if self.write_masks:
            self.masks_dir.mkdir(parents=True, exist_ok=True)
            self.instances_dir.mkdir(parents=True, exist_ok=True)
            self.flag_instances_dir.mkdir(parents=True, exist_ok=True)

        self.train_list: List[str] = []
        self.val_list: List[str] = []
        self.n_frames = 0
        self.n_instances = 0

    # -- internal helpers --------------------------------------------------

    def _write_image(self, image_float: np.ndarray, stem: str) -> None:
        gray = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
        iio.imwrite(str(self.images_dir / f"{stem}.{self.ext}"), gray)

    def _write_groups(self, stem: str, lines: List[str],
                      animal_ids: List[int]) -> None:
        """Sidecar mapping each YOLO label line to its animal_id, so a body /
        flagellum polygon can be traced to a unique animal. ``animal_ids[i]``
        is the id for line ``i`` of ``labels/<stem>.txt`` (and ``classes[i]``
        its class index, for convenience)."""
        classes = [int(ln.split(" ", 1)[0]) for ln in lines]
        data = {
            "stem": stem,
            "class_names": {str(i): n for i, n in enumerate(CLASS_NAMES)},
            "animal_ids": animal_ids,    # parallel to label-file lines
            "classes": classes,
        }
        (self.groups_dir / f"{stem}.json").write_text(json.dumps(data))

    # -- public API --------------------------------------------------------

    def add_frame(self, image_float: np.ndarray, instance_masks, stem: str) -> None:
        """Write one frame's image, YOLO label + group json, and mask + instance map.

        `instance_masks` is the list returned by
        ``render_scene(..., return_masks=True)``: per parasite, None or a dict
        of tile-local masks. Each kept cell is given an ``animal_id`` (1..N,
        unique within the frame) that ties its body / flagellum to it — written
        to ``instances/<stem>.png`` (per pixel) and ``groups/<stem>.json`` (per
        YOLO line). An empty / all-None list (e.g. a background-only negative
        frame) still produces an image, an empty YOLO label, and zero maps —
        useful as hard-negative training examples.
        """
        H, W = image_float.shape[:2]
        self._write_image(image_float, stem)

        # './'-prefixed so ultralytics resolves it relative to the listing
        # file's directory (the dataset root), not the current working dir.
        rel = f"./images/{stem}.{self.ext}"
        target = self.val_list if self.split_rng.random() < self.val_fraction \
            else self.train_list
        target.append(rel)

        lines: List[str] = []
        line_animal_ids: List[int] = []     # animal_id per YOLO line (parallel)
        label_map = (np.zeros((H, W), dtype=np.uint8)
                     if self.write_masks else None)
        inst_map = (np.zeros((H, W), dtype=np.uint16)
                    if self.write_masks else None)
        # Per-flagellum instance ids (1..F within the frame); 0 = not a
        # flagellum. A dividing cell contributes one id per flagellum.
        flaginst_map = (np.zeros((H, W), dtype=np.uint16)
                        if self.write_masks else None)
        flag_id = 0     # running per-flagellum id, frame-unique
        animal_id = 0   # 1-based; assigned only to cells that pass the gates

        for inst in (instance_masks or []):
            if inst is None:
                continue
            if not inst["animal_full"].any():
                continue
            if inst["visible_frac"] < self.min_visible_frac:
                continue  # cell almost entirely hidden behind a neighbour
            self.n_instances += 1
            animal_id += 1
            y0, x0 = inst["y0"], inst["x0"]
            body_v = inst["body"]   # modal (visible, disjoint) -> semantic map
            flag_v = inst["flag"]

            if self.write_masks:
                th, tw = body_v.shape
                sub = label_map[y0:y0 + th, x0:x0 + tw]
                isub = inst_map[y0:y0 + th, x0:x0 + tw]
                # Disjoint by construction; body first, flagellum second.
                sub[body_v] = SEMANTIC_BODY
                sub[flag_v] = SEMANTIC_FLAGELLUM
                isub[body_v] = animal_id
                isub[flag_v] = animal_id
                # Per-flagellum ids: paint each flagellum of this cell with its
                # own id. Restricted to flag_v (the visible flagellum-outside-
                # body region) so it stays consistent with masks==FLAGELLUM.
                # At a crossing the later flagellum wins those few shared px.
                fsub = flaginst_map[y0:y0 + th, x0:x0 + tw]
                for fm in inst["flag_per"]:
                    fm_v = fm & flag_v
                    if not fm_v.any():
                        continue
                    flag_id += 1
                    fsub[fm_v] = flag_id

            if self.write_yolo:
                if self.split_occluded:
                    # Visible (modal) regions: an occluder crossing a cell can
                    # split it into several same-class instances.
                    animal_src, body_src, flag_src = body_v | flag_v, body_v, flag_v
                    flag_per_src = inst["flag_per"]
                    largest = False
                else:
                    # One instance per cell per class: full (amodal) extent, so
                    # occlusion never splits an animal into multiple detections.
                    animal_src = inst["animal_full"]
                    body_src = inst["body_full"]
                    flag_src = inst["flag_full"]
                    flag_per_src = inst["flag_per_full"]
                    largest = True
                for cls, src in ((ANIMAL, animal_src), (BODY, body_src),
                                 (FLAGELLUM, flag_src)):
                    if cls not in self.yolo_class_ids:
                        continue
                    # When split_flagella is enabled, emit one polygon per
                    # individual flagellum (each sharing animal_id) instead
                    # of one merged polygon for all of them. For non-dividing
                    # cells (single flagellum), this is identical to the
                    # default path.
                    if cls == FLAGELLUM and self.split_flagella:
                        sources = flag_per_src
                    else:
                        sources = [src]
                    for sub_src in sources:
                        for poly in mask_to_polygons(sub_src, x0, y0, W, H,
                                                     self.min_area,
                                                     largest_only=largest,
                                                     carve_holes=self.carve_holes):
                            lines.append(_yolo_line(cls, poly))
                            line_animal_ids.append(animal_id)

        if self.write_yolo:
            (self.labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            self._write_groups(stem, lines, line_animal_ids)
        if self.write_masks:
            iio.imwrite(str(self.masks_dir / f"{stem}.png"), label_map)
            iio.imwrite(str(self.instances_dir / f"{stem}.png"), inst_map)
            iio.imwrite(str(self.flag_instances_dir / f"{stem}.png"), flaginst_map)

        self.n_frames += 1
        if self.n_frames % 25 == 0:
            print(f"  wrote {self.n_frames} frames "
                  f"({len(self.train_list)} train / {len(self.val_list)} val)")

    def close(self) -> None:
        if self.write_yolo:
            (self.root / "train.txt").write_text("\n".join(self.train_list) + "\n")
            (self.root / "val.txt").write_text("\n".join(self.val_list) + "\n")
            self._write_data_yaml()
        if self.write_masks:
            (self.masks_dir / "classes.txt").write_text(
                "0 background\n1 body\n2 flagellum\n")
        self._write_metadata()
        print(f"Done. {self.n_frames} frames, {self.n_instances} instances "
              f"({len(self.train_list)} train / {len(self.val_list)} val) "
              f"in {self.root}")

    def _write_data_yaml(self) -> None:
        # Hand-written so we never depend on PyYAML for single-setup runs and
        # never trip over backslashes/UNC paths (single-quoted = literal).
        names = "\n".join(f"  {i}: {CLASS_NAMES[i]}" for i in self.yolo_class_ids)
        # No 'path:' key on purpose: ultralytics then resolves train/val (and
        # the './'-prefixed image paths inside them) relative to THIS data.yaml's
        # own directory. That keeps the dataset portable across machines — e.g.
        # generated on Windows/UNC, trained on a Linux cluster — with no path
        # rewriting. (A hardcoded absolute 'path:' would break on another host.)
        text = (
            "# Ultralytics YOLO-seg dataset (generated)\n"
            "# Portable: paths resolve relative to this file's directory.\n"
            "train: train.txt\n"
            "val: val.txt\n"
            f"nc: {len(self.yolo_class_ids)}\n"
            "names:\n"
            f"{names}\n"
        )
        (self.root / "data.yaml").write_text(text)

    def _write_metadata(self) -> None:
        meta = {
            "format": {"yolo_seg": self.write_yolo, "semantic_masks": self.write_masks},
            "yolo_classes": self.yolo_classes,
            "yolo_class_names": {str(i): CLASS_NAMES[i] for i in self.yolo_class_ids},
            "classes": {str(i): n for i, n in enumerate(CLASS_NAMES)},
            "semantic_mask_values": {"background": 0, "body": SEMANTIC_BODY,
                                     "flagellum": SEMANTIC_FLAGELLUM,
                                     "animal": "any non-zero pixel"},
            "image_format": self.ext,
            "val_fraction": self.val_fraction,
            "min_polygon_area": self.min_area,
            "yolo_one_instance_per_cell": not self.split_occluded,
            "yolo_mask_extent": "visible (modal)" if self.split_occluded
                                else "full silhouette (amodal)",
            "yolo_split_flagella": self.split_flagella,
            "yolo_carve_holes": self.carve_holes,
            "flag_instances_map": self.write_masks,  # per-flagellum id PNG written
            "min_visible_frac": self.min_visible_frac,
            "n_frames": self.n_frames,
            "n_instances": self.n_instances,
            "n_train": len(self.train_list),
            "n_val": len(self.val_list),
        }
        (self.root / "seg_metadata.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------
# Per-frame sampling helpers (mirror dataset_builder's jitter semantics)
# ----------------------------------------------------------------------------

def _frame_noise(cfg, rng):
    if cfg.bg_intensity_range is not None:
        bg = float(rng.uniform(*cfg.bg_intensity_range))
        return dataclasses.replace(cfg.noise, bg_intensity=bg)
    return cfg.noise


def _frame_optics_noise(cfg, noise, rng):
    if cfg.per_frame_jitter:
        optics_f = DB._jitter_optics_object(cfg.optics, rng, cfg.optics_ranges)
        noise_f = DB._jitter_noise_object(noise, rng, cfg.noise_ranges)
        return optics_f, noise_f
    return cfg.optics, noise


def _frame_clutter(cfg, rng):
    cl = cfg.clutter_level
    if isinstance(cl, tuple) and len(cl) == 2:
        return float(rng.uniform(*cl))
    return float(cl)


# ----------------------------------------------------------------------------
# Generation: per-mode
# ----------------------------------------------------------------------------

def run_random(cfg: DB.DatasetConfig, ds: SegmentationDataset, tag: str) -> None:
    """Independent frames with random parasites (best diversity for seg)."""
    rng = np.random.default_rng(cfg.seed)
    save_indices = DB.select_frames(cfg.n_frames, cfg.save_frames)
    n_kp = cfg.skeleton.n_flagellum_interior

    for saved_idx, _sim_frame in enumerate(save_indices):
        n_p = int(rng.integers(cfg.parasites_per_frame[0],
                               cfg.parasites_per_frame[1] + 1))
        t = float(rng.uniform(0, 1.0))
        parasites = DB._sample_parasites_for_frame(
            rng, cfg.image_shape, n_p, t, n_kp,
            organelle_prob=cfg.organelle_prob,
            mottle_prob=cfg.cytoplasm_mottle_prob,
            dividing_fraction=cfg.dividing_fraction,
            microtexture_prob=cfg.microtexture_prob,
            microtexture_ranges=cfg.microtexture_ranges,
        )

        noise = _frame_noise(cfg, rng)
        optics_f, noise_f = _frame_optics_noise(cfg, noise, rng)
        image, _kps, masks = L.render_scene(
            parasites, t=t, image_shape=cfg.image_shape,
            optics=optics_f, noise=noise_f, rng=rng, fast=cfg.fast,
            clutter_level=_frame_clutter(cfg, rng), return_masks=True)
        ds.add_frame(image, masks, f"{tag}_{saved_idx:06d}")


def run_video(cfg: DB.VideoConfig, ds: SegmentationDataset, tag: str) -> None:
    """Animated clip with persistent parasites (one background per clip)."""
    rng = np.random.default_rng(cfg.seed)
    save_set = set(DB.select_frames(cfg.n_frames, cfg.save_frames))
    n_kp = cfg.skeleton.n_flagellum_interior
    duration = cfg.n_frames / cfg.fps

    parasites = DB._sample_parasites_for_frame(
        rng, cfg.image_shape, cfg.n_parasites, t=0.0, n_kp=n_kp,
        organelle_prob=cfg.organelle_prob,
        mottle_prob=cfg.cytoplasm_mottle_prob,
        dividing_fraction=cfg.dividing_fraction,
        microtexture_prob=cfg.microtexture_prob,
        microtexture_ranges=cfg.microtexture_ranges,
    )
    for p in parasites:
        p.mode_schedule = L.generate_mode_schedule(p, duration, rng)

    cl = cfg.clutter_level
    cl_clip = (0.5 * (cl[0] + cl[1]) if isinstance(cl, tuple) and len(cl) == 2
               else float(cl))
    bg = L.synthetic_background(cfg.image_shape, rng,
                                intensity=cfg.noise.bg_intensity,
                                clutter_level=cl_clip)

    dt = 1.0 / cfg.fps
    saved = 0
    for i in range(cfg.n_frames):
        t = i * dt
        if i > 0:
            L.advance_parasites(parasites, dt, cfg.image_shape,
                                periodic=cfg.periodic_boundary, t=t,
                                optics=cfg.optics)
        if i not in save_set:
            continue
        image, _kps, masks = L.render_scene(
            parasites, t=t, image_shape=cfg.image_shape,
            optics=cfg.optics, noise=cfg.noise, background=bg,
            rng=rng, fast=cfg.fast, return_masks=True)
        ds.add_frame(image, masks, f"{tag}_{saved:06d}")
        saved += 1


def run_negative(cfg: DB.DatasetConfig, ds: SegmentationDataset, tag: str) -> None:
    """Background-only frames: images with empty labels / all-zero masks.
    Useful as hard negatives so the model learns to suppress clutter."""
    rng = np.random.default_rng(cfg.seed)
    save_indices = DB.select_frames(cfg.n_frames, cfg.save_frames)

    for saved_idx, _sim_frame in enumerate(save_indices):
        t = float(rng.uniform(0, 1.0))
        noise = _frame_noise(cfg, rng)
        optics_f, noise_f = _frame_optics_noise(cfg, noise, rng)
        image, _kps, masks = L.render_scene(
            [], t=t, image_shape=cfg.image_shape,
            optics=optics_f, noise=noise_f, rng=rng, fast=cfg.fast,
            clutter_level=_frame_clutter(cfg, rng), return_masks=True)
        ds.add_frame(image, masks, f"{tag}_neg_{saved_idx:06d}")


def _run_one_setup(cfg, ds: SegmentationDataset, tag: str) -> None:
    if isinstance(cfg, DB.VideoConfig):
        run_video(cfg, ds, tag)
    elif getattr(cfg, "mode", "random") == "negative":
        run_negative(cfg, ds, tag)
    else:
        run_random(cfg, ds, tag)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _add_seg_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="fraction of frames assigned to the val split")
    p.add_argument("--min-polygon-area", type=float, default=6.0,
                   help="drop YOLO polygons smaller than this (px^2)")
    p.add_argument("--image-format", choices=["png", "jpg"], default="png")
    p.add_argument("--yolo-classes", choices=["all", "animal"], default="all",
                   help="which YOLO instance classes to emit. 'all' = "
                        "animal+body+flagellum (3-class nested); 'animal' = "
                        "single-class animal instances (the recommended "
                        "pipeline's detector stage; pair with the mask UNet)")
    p.add_argument("--split-occluded", action="store_true",
                   help="emit YOLO instances from visible (modal) regions, "
                        "letting an occluded cell split into several "
                        "same-class instances. Default: one instance per cell "
                        "per class (full amodal silhouette), never split.")
    sf = p.add_mutually_exclusive_group()
    sf.add_argument("--split-flagella", dest="split_flagella", action="store_true",
                    help="emit each flagellum of a dividing cell as its own "
                         "YOLO instance (sharing animal_id in the group json). "
                         "This is the DEFAULT.")
    sf.add_argument("--no-split-flagella", dest="split_flagella", action="store_false",
                    help="legacy: merge all of a cell's flagella into one "
                         "flagellum polygon (fuses/fills between the arms).")
    p.set_defaults(split_flagella=True)
    ch = p.add_mutually_exclusive_group()
    ch.add_argument("--carve-holes", dest="carve_holes", action="store_true",
                    help="keep enclosed background holes out of polygons via the "
                         "keyhole/bridge method. This is the DEFAULT.")
    ch.add_argument("--no-carve-holes", dest="carve_holes", action="store_false",
                    help="legacy: trace outer contour only, so enclosed holes "
                         "(e.g. between crossing flagella) get filled.")
    p.set_defaults(carve_holes=True)
    p.add_argument("--min-visible-frac", type=float, default=0.05,
                   help="drop cells with less than this fraction visible "
                        "(hidden behind a neighbour)")
    p.add_argument("--no-yolo", action="store_true",
                   help="skip YOLO-seg polygon labels + data.yaml")
    p.add_argument("--no-masks", action="store_true",
                   help="skip semantic-segmentation mask PNGs")


def _dataset_from_args(args) -> SegmentationDataset:
    return SegmentationDataset(
        args.out,
        val_fraction=args.val_fraction,
        min_polygon_area=args.min_polygon_area,
        write_yolo=not args.no_yolo,
        write_masks=not args.no_masks,
        image_format=args.image_format,
        split_occluded=args.split_occluded,
        split_flagella=args.split_flagella,
        carve_holes=args.carve_holes,
        min_visible_frac=args.min_visible_frac,
        yolo_classes=args.yolo_classes,
        seed=getattr(args, "seed", 0))


def cli_random(args) -> None:
    skeleton = DB.SkeletonConfig(n_flagellum_interior=args.flag_keypoints)
    cfg = DB.DatasetConfig(
        out_dir=args.out, n_frames=args.frames,
        image_shape=tuple(args.size),
        parasites_per_frame=tuple(args.parasites),
        skeleton=skeleton, seed=args.seed, fast=args.fast,
        save_frames=DB._save_frames_from_args(args))
    ds = _dataset_from_args(args)
    run_random(cfg, ds, tag="img")
    ds.close()


def cli_video(args) -> None:
    skeleton = DB.SkeletonConfig(n_flagellum_interior=args.flag_keypoints)
    cfg = DB.VideoConfig(
        out_dir=args.out, n_frames=args.frames,
        image_shape=tuple(args.size),
        skeleton=skeleton, seed=args.seed, fast=args.fast,
        fps=args.fps, n_parasites=args.n_parasites,
        periodic_boundary=not args.no_periodic,
        save_frames=DB._save_frames_from_args(args))
    ds = _dataset_from_args(args)
    run_video(cfg, ds, tag="vid")
    ds.close()


def cli_multi(args) -> None:
    import yaml
    with open(args.config) as f:
        cfg_data = yaml.safe_load(f)

    out_dir, setups = DB._resolve_multi_config(cfg_data)

    # Segmentation output dir: --out wins, else seg_output_dir in YAML, else
    # the SLEAP output_dir with a _seg suffix.
    if args.out is not None:
        seg_root = Path(args.out)
    elif "seg_output_dir" in cfg_data:
        seg_root = Path(cfg_data["seg_output_dir"])
    else:
        seg_root = Path(str(out_dir) + "_seg")

    # Seg options: CLI flag if the user passed one, else YAML, else default.
    def _opt(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return cfg_data.get(yaml_key, default)

    ds = SegmentationDataset(
        seg_root,
        val_fraction=_opt(args.val_fraction, "seg_val_fraction", 0.1),
        min_polygon_area=_opt(args.min_polygon_area, "seg_min_polygon_area", 6.0),
        write_yolo=(not args.no_yolo) and cfg_data.get("seg_write_yolo", True),
        write_masks=(not args.no_masks) and cfg_data.get("seg_write_masks", True),
        image_format=_opt(args.image_format, "seg_image_format", "png"),
        split_occluded=args.split_occluded or cfg_data.get("seg_split_occluded", False),
        split_flagella=_opt(args.split_flagella, "seg_split_flagella", True),
        carve_holes=_opt(args.carve_holes, "seg_carve_holes", True),
        min_visible_frac=_opt(args.min_visible_frac, "seg_min_visible_frac", 0.05),
        yolo_classes=_opt(args.yolo_classes, "seg_yolo_classes", "all"),
        seed=int(cfg_data.get("seed", 0)))

    total = len(setups)
    print(f"Building one segmentation dataset from {total} setup(s) -> {seg_root}")
    for i, setup in enumerate(setups):
        name, cfg = DB._build_setup_config(setup)
        tag = cfg.tag or name.replace("/", "_").replace("\\", "_")
        print(f"=== [{i + 1}/{total}] {name} ...")
        _run_one_setup(cfg, ds, tag)
    ds.close()


def cli_template(args) -> None:
    # Same starter config as dataset_builder; seg options are optional
    # top-level keys (see module docstring) with sensible defaults.
    DB.write_template(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rand = sub.add_parser("random", help="independent frames, random parasites")
    DB._add_common_args(p_rand)
    p_rand.add_argument("--parasites", type=int, nargs=2, default=(5, 15),
                        metavar=("MIN", "MAX"))
    _add_seg_args(p_rand)
    p_rand.set_defaults(func=cli_random)

    p_vid = sub.add_parser("video", help="animated clip with persistent parasites")
    DB._add_common_args(p_vid)
    p_vid.add_argument("--fps", type=float, default=60.0)
    p_vid.add_argument("--n-parasites", type=int, default=20)
    p_vid.add_argument("--no-periodic", action="store_true",
                       help="disable wrap-around at image edges")
    _add_seg_args(p_vid)
    p_vid.set_defaults(func=cli_video)

    p_multi = sub.add_parser("multi",
                             help="run all setups from a YAML config into one "
                                  "combined segmentation dataset")
    p_multi.add_argument("config", type=Path, help="multi-setup YAML config")
    p_multi.add_argument("--out", type=Path, default=None,
                         help="segmentation output dir (default: "
                              "<output_dir>_seg or seg_output_dir from YAML)")
    # CLI seg options default to None here so YAML values can take effect.
    p_multi.add_argument("--val-fraction", type=float, default=None)
    p_multi.add_argument("--min-polygon-area", type=float, default=None)
    p_multi.add_argument("--min-visible-frac", type=float, default=None)
    p_multi.add_argument("--image-format", choices=["png", "jpg"], default=None)
    p_multi.add_argument("--yolo-classes", choices=["all", "animal"], default=None)
    p_multi.add_argument("--split-occluded", action="store_true")
    sf_m = p_multi.add_mutually_exclusive_group()
    sf_m.add_argument("--split-flagella", dest="split_flagella",
                      action="store_true", default=None)
    sf_m.add_argument("--no-split-flagella", dest="split_flagella",
                      action="store_false", default=None)
    ch_m = p_multi.add_mutually_exclusive_group()
    ch_m.add_argument("--carve-holes", dest="carve_holes",
                      action="store_true", default=None)
    ch_m.add_argument("--no-carve-holes", dest="carve_holes",
                      action="store_false", default=None)
    p_multi.add_argument("--no-yolo", action="store_true")
    p_multi.add_argument("--no-masks", action="store_true")
    p_multi.set_defaults(func=cli_multi)

    p_tpl = sub.add_parser("template", help="write a starter multi-setup YAML")
    p_tpl.add_argument("-o", "--out", type=Path, required=True)
    p_tpl.set_defaults(func=cli_template)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
