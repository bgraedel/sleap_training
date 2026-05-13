"""
Build SLEAP-importable datasets from synthetic Leishmania renderings.

Generation modes:
  - random:  N independent frames with random parasites (single-frame pose
             models, augmentation-friendly).
  - video:   one video with M frames, persistent parasite identities and
             tracks (tracking / temporal models).
  - multi:   read a YAML config and produce one dataset per "setup"
             (e.g. different magnifications / lighting / noise levels).

Utility commands:
  - subset:  extract a subset of frames from an existing labels.slp.
  - splits:  make train/val/test splits from an existing labels.slp.
  - template: write a starter multi-setup YAML to disk.

Output structure for a generation run:
    out/
      video.tif (or video.mp4)
      labels.slp           SLEAP-native; open in the GUI or load with sleap-io
      ground_truth.json    waveform parameters per parasite per frame

For multi-setup runs:
    out/
      config_resolved.yaml   fully-resolved config (audit trail)
      <setup_name>/
        video.tif
        labels.slp
        ground_truth.json
      ...

Examples
--------
    python dataset_builder.py random --frames 200 --out data/synth1
    python dataset_builder.py video --frames 600 --fps 60 --out data/video1 \\
        --n-parasites 15 --format mp4
    python dataset_builder.py template -o configs/training.yaml
    python dataset_builder.py multi configs/training.yaml
    python dataset_builder.py subset data/synth1/labels.slp \\
        --indices 0,5,10,15 -o data/synth1/sub.slp
    python dataset_builder.py splits data/synth1/labels.slp \\
        --train 0.8 --val 0.1 --test 0.1 -o data/synth1/splits
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sleap_io as sio

import synthetic_leishmania as L


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# save_frames spec: 'all', or a dict like {'every_n': 5}, {'indices': [...]},
# {'count': 50}, {'range': [start, end]}
SaveFramesSpec = Union[str, Dict[str, Any], None]


@dataclass
class SkeletonConfig:
    """Skeleton: Head -> Base -> Flag1 -> ... -> FlagN -> Tip."""
    n_flagellum_interior: int = 5
    head_name: str = "Head"
    base_name: str = "Base"
    tip_name: str = "Tip"
    flagellum_prefix: str = "Flag"

    @property
    def node_names(self) -> List[str]:
        nodes = [self.head_name, self.base_name]
        nodes += [f"{self.flagellum_prefix}{i}"
                  for i in range(1, self.n_flagellum_interior + 1)]
        nodes.append(self.tip_name)
        return nodes

    @property
    def edges(self) -> List[Tuple[str, str]]:
        nodes = self.node_names
        return [(nodes[0], nodes[1])] + [
            (nodes[i], nodes[i + 1]) for i in range(1, len(nodes) - 1)
        ]

    def to_sio_skeleton(self) -> sio.Skeleton:
        return sio.Skeleton(nodes=self.node_names, edges=self.edges)


@dataclass
class DatasetConfig:
    out_dir: Path = Path("dataset")
    n_frames: int = 200
    image_shape: Tuple[int, int] = (1024, 1024)
    parasites_per_frame: Tuple[int, int] = (5, 15)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    optics: L.OpticsParams = field(default_factory=L.OpticsParams)
    noise: L.CameraNoiseParams = field(default_factory=L.CameraNoiseParams)
    bg_intensity_range: Optional[Tuple[float, float]] = (0.25, 0.95)
    seed: int = 0
    fast: bool = False
    video_format: str = "tiff"     # 'tiff' or 'mp4'
    mp4_quality: int = 10           # imageio 0-10 (10 = highest)
    mp4_fps: float = 30.0
    save_slp: bool = True
    embed_frames: bool = False
    save_frames: SaveFramesSpec = None  # None = save all
    per_frame_jitter: bool = True       # in random mode, jitter optics/noise per frame


@dataclass
class VideoConfig(DatasetConfig):
    fps: float = 60.0
    n_parasites: int = 20
    periodic_boundary: bool = True


# ----------------------------------------------------------------------------
# Streaming video writer
# ----------------------------------------------------------------------------

class _VideoWriter:
    """
    Stream 8-bit grayscale frames to a single video file.
      - 'tiff': lossless multi-page TIFF (perfect pixel fidelity)
      - 'mp4':  H.264, smaller files but lossy. Requires even H and W.
    """
    def __init__(self, path: Path, fmt: str, *,
                 fps: float = 30.0, mp4_quality: int = 9):
        self.path = Path(path)
        self.fmt = fmt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "tiff":
            import tifffile
            self._w = tifffile.TiffWriter(str(self.path), bigtiff=True)
        elif fmt == "mp4":
            import imageio.v2 as iio2
            self._w = iio2.get_writer(
                str(self.path), fps=fps, codec="libx264",
                quality=mp4_quality, pixelformat="yuv420p",
                macro_block_size=1)
        else:
            raise ValueError(f"video_format must be 'tiff' or 'mp4', got {fmt!r}")

    def write_float(self, image_float: np.ndarray) -> None:
        gray = np.clip(image_float * 255, 0, 255).astype(np.uint8)
        if self.fmt == "tiff":
            self._w.write(gray, photometric="minisblack")
        else:
            h, w = gray.shape
            if h % 2 or w % 2:
                raise ValueError(
                    f"mp4 output requires even image dimensions, got ({h},{w})")
            self._w.append_data(gray)

    def close(self) -> None:
        self._w.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _summarize_parasite(p: L.ParasiteParams) -> dict:
    """Full JSON-serializable snapshot of a parasite (every dataclass field)."""
    d = dataclasses.asdict(p)
    if d.get("mode_schedule") is not None:
        d["mode_schedule"] = [list(item) for item in d["mode_schedule"]]
    return d


def _kp_dict_to_array(kp: dict, node_order: Sequence[str]) -> np.ndarray:
    out = np.full((len(node_order), 2), np.nan, dtype=np.float32)
    for j, name in enumerate(node_order):
        if name in kp:
            out[j] = kp[name]
    return out


def _video_filename(fmt: str) -> str:
    return "video.tif" if fmt == "tiff" else "video.mp4"


def _validate_mp4_shape(shape: Tuple[int, int]) -> None:
    h, w = shape
    if h % 2 or w % 2:
        raise ValueError(f"mp4 output requires even image dimensions, got {shape}")


def select_frames(n_frames: int, spec: SaveFramesSpec) -> List[int]:
    """Resolve a save_frames spec into a sorted list of frame indices.

    Accepts:
      - None / 'all' / {}                       -> all frames
      - {'every_n': N}                          -> every N-th frame
      - {'every_n': N, 'start': S, 'end': E}    -> every N-th in [S, E)
      - {'indices': [i1, i2, ...]}              -> specific frames
      - {'count': K}                            -> K evenly-spaced frames
      - {'range': [S, E]}                       -> all in [S, E)
    """
    if spec is None or spec == "all" or spec == {}:
        return list(range(n_frames))
    if not isinstance(spec, dict):
        raise ValueError(f"unknown save_frames spec: {spec!r}")

    if "indices" in spec:
        idx = sorted({int(i) for i in spec["indices"] if 0 <= int(i) < n_frames})
        return idx
    if "count" in spec:
        k = int(spec["count"])
        if k <= 0:
            return []
        if k >= n_frames:
            return list(range(n_frames))
        return list(np.linspace(0, n_frames - 1, k, dtype=int))
    if "range" in spec:
        s, e = spec["range"]
        return list(range(max(0, int(s)), min(n_frames, int(e))))
    if "every_n" in spec:
        n = max(1, int(spec["every_n"]))
        s = int(spec.get("start", 0))
        e = int(spec.get("end", n_frames))
        return list(range(max(0, s), min(n_frames, e), n))
    raise ValueError(f"unknown save_frames spec: {spec!r}")


# ----------------------------------------------------------------------------
# SLEAP labels builder
# ----------------------------------------------------------------------------

def build_sleap_labels(
    video_path: Path,
    keypoints_per_frame: Sequence[List[dict]],
    skeleton_cfg: SkeletonConfig,
    *,
    tracks_per_frame: Optional[Sequence[List[Optional[int]]]] = None,
    fps: Optional[float] = None,
) -> sio.Labels:
    """Build a sio.Labels object from one video file + per-frame keypoints.

    `keypoints_per_frame[i]` corresponds to frame i of the SAVED video
    (which may be a subsample of the simulated frames).
    """
    skeleton = skeleton_cfg.to_sio_skeleton()
    node_order = skeleton_cfg.node_names

    video = sio.load_video(str(video_path))
    if fps is not None:
        video.fps = fps

    track_objs: dict = {}
    if tracks_per_frame is not None:
        ids = {tid for tids in tracks_per_frame for tid in tids if tid is not None}
        for tid in sorted(ids):
            track_objs[tid] = sio.Track(name=f"parasite_{tid:04d}")

    labeled_frames = []
    for frame_idx, kps in enumerate(keypoints_per_frame):
        instances = []
        for inst_idx, kp in enumerate(kps):
            pts = _kp_dict_to_array(kp, node_order)
            # Drop fully-occluded cells: if every keypoint is NaN (i.e. all
            # nodes are either missing from the dict or marked occluded by
            # the simulator's visibility check), the instance has nothing to
            # learn from and SLEAP would reject it anyway. Skip entirely so
            # the frame's instance list only contains useful targets.
            if np.all(np.isnan(pts)):
                continue
            track = None
            if tracks_per_frame is not None:
                tid = tracks_per_frame[frame_idx][inst_idx]
                if tid is not None:
                    track = track_objs[tid]
            instances.append(sio.Instance.from_numpy(
                pts, skeleton=skeleton, track=track))
        if instances:
            labeled_frames.append(sio.LabeledFrame(
                video=video, frame_idx=frame_idx, instances=instances))

    return sio.Labels(
        videos=[video], skeletons=[skeleton],
        labeled_frames=labeled_frames,
        tracks=list(track_objs.values()),
    )


def save_labels(labels: sio.Labels, path: Path, embed: bool = False) -> None:
    if embed:
        labels.save(str(path), embed="all")
    else:
        labels.save(str(path))


# ----------------------------------------------------------------------------
# Generation: random parasites per frame
# ----------------------------------------------------------------------------

def _jitter_optics_object(optics: L.OpticsParams,
                          rng: np.random.Generator) -> L.OpticsParams:
    """Per-frame multiplicative jitter on optics. Pushed wide so frames span
    very-sharp/clean to blurry/heavy-halo within one setup."""
    return dataclasses.replace(
        optics,
        psf_sigma_um=float(optics.psf_sigma_um * rng.uniform(0.45, 1.55)),
        halo_strength=float(optics.halo_strength * rng.uniform(0.30, 1.80)),
        halo_lowpass_sigma_um=float(optics.halo_lowpass_sigma_um * rng.uniform(0.50, 1.85)),
        intensity_gain=float(optics.intensity_gain * rng.uniform(0.65, 1.40)),
        shadeoff_threshold=float(optics.shadeoff_threshold * rng.uniform(0.85, 1.18)),
        body_edge_smooth_sigma_um=float(optics.body_edge_smooth_sigma_um * rng.uniform(0.50, 1.40)),
    )


def _jitter_noise_object(noise: L.CameraNoiseParams,
                         rng: np.random.Generator) -> L.CameraNoiseParams:
    """Per-frame noise jitter. Wide range covers very grainy low-light through
    clean high-light. Clipped to physically plausible camera regimes."""
    return dataclasses.replace(
        noise,
        full_well_photons=float(np.clip(
            noise.full_well_photons * rng.uniform(0.25, 3.00), 150, 8000)),
        read_noise_e=float(np.clip(
            noise.read_noise_e * rng.uniform(0.40, 3.50), 1.0, 12.0)),
    )


def _jitter_optics_dict(optics_dict: dict,
                        rng: np.random.Generator) -> dict:
    """Per-clip jitter on raw YAML dict (used during repeats expansion)."""
    out = dict(optics_dict)
    if "psf_sigma_um" in out:
        out["psf_sigma_um"] = float(out["psf_sigma_um"] * rng.uniform(0.60, 1.55))
    if "halo_strength" in out:
        out["halo_strength"] = float(out["halo_strength"] * rng.uniform(0.30, 2.20))
    if "halo_lowpass_sigma_um" in out:
        out["halo_lowpass_sigma_um"] = float(out["halo_lowpass_sigma_um"] * rng.uniform(0.50, 1.85))
    if "intensity_gain" in out:
        out["intensity_gain"] = float(out["intensity_gain"] * rng.uniform(0.65, 1.40))
    if "shadeoff_threshold" in out:
        out["shadeoff_threshold"] = float(out["shadeoff_threshold"] * rng.uniform(0.85, 1.18))
    return out


def _jitter_noise_dict(noise_dict: dict,
                       rng: np.random.Generator) -> dict:
    """Per-clip jitter on raw YAML noise dict."""
    out = dict(noise_dict)
    if "full_well_photons" in out:
        out["full_well_photons"] = float(np.clip(
            out["full_well_photons"] * rng.uniform(0.25, 3.00), 150, 8000))
    if "read_noise_e" in out:
        out["read_noise_e"] = float(np.clip(
            out["read_noise_e"] * rng.uniform(0.40, 3.50), 1.0, 12.0))
    if "bg_intensity" in out:
        out["bg_intensity"] = float(np.clip(
            out["bg_intensity"] * rng.uniform(0.60, 1.35), 0.18, 0.95))
    return out


def generate_random_dataset(cfg: DatasetConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.video_format == "mp4":
        _validate_mp4_shape(cfg.image_shape)

    # save_frames in random mode: just generate len(selected) independent frames.
    save_indices = select_frames(cfg.n_frames, cfg.save_frames)

    video_path = out / _video_filename(cfg.video_format)
    keypoints_per_frame: List[List[dict]] = []
    ground_truth: List[dict] = []
    n_kp = cfg.skeleton.n_flagellum_interior

    with _VideoWriter(video_path, cfg.video_format,
                      fps=cfg.mp4_fps, mp4_quality=cfg.mp4_quality) as vw:
        for saved_idx, sim_frame in enumerate(save_indices):
            n_p = int(rng.integers(cfg.parasites_per_frame[0],
                                   cfg.parasites_per_frame[1] + 1))
            t = float(rng.uniform(0, 1.0))

            parasites = []
            for _ in range(n_p):
                p = L.sample_random_parasite(rng, cfg.image_shape, t=t)
                p.n_flagellum_keypoints = n_kp
                parasites.append(p)

            if cfg.bg_intensity_range is not None:
                bg_int = float(rng.uniform(*cfg.bg_intensity_range))
                noise = dataclasses.replace(cfg.noise, bg_intensity=bg_int)
            else:
                noise = cfg.noise

            # Per-frame optics/noise jitter so each frame mimics a different
            # microscopy session (slightly different focus, halo, contrast,
            # noise level). Disable via per_frame_jitter=False for debugging.
            if cfg.per_frame_jitter:
                optics_frame = _jitter_optics_object(cfg.optics, rng)
                noise_frame = _jitter_noise_object(noise, rng)
            else:
                optics_frame = cfg.optics
                noise_frame = noise

            image, kps = L.render_scene(
                parasites, t=t, image_shape=cfg.image_shape,
                optics=optics_frame, noise=noise_frame, rng=rng, fast=cfg.fast,
            )
            vw.write_float(image)

            keypoints_per_frame.append(kps)
            ground_truth.append({
                "saved_frame": saved_idx,
                "sim_frame": sim_frame,
                "t_seconds": t,
                "bg_intensity": float(noise_frame.bg_intensity),
                "psf_sigma_um": float(optics_frame.psf_sigma_um),
                "halo_strength": float(optics_frame.halo_strength),
                "n_parasites": n_p,
                "instances": [_summarize_parasite(p) for p in parasites],
            })

            if (saved_idx + 1) % 25 == 0 or saved_idx == len(save_indices) - 1:
                print(f"  generated {saved_idx + 1}/{len(save_indices)} frames")

    with open(out / "ground_truth.json", "w") as f:
        json.dump({
            "mode": "random",
            "video": video_path.name,
            "image_shape": list(cfg.image_shape),
            "skeleton": {"nodes": cfg.skeleton.node_names,
                         "edges": cfg.skeleton.edges},
            "frames": ground_truth,
        }, f, indent=2)

    slp_path = out / "labels.slp"
    if cfg.save_slp:
        labels = build_sleap_labels(
            video_path, keypoints_per_frame, cfg.skeleton,
            fps=cfg.mp4_fps if cfg.video_format == "mp4" else None,
        )
        save_labels(labels, slp_path, embed=cfg.embed_frames)
        print(f"  wrote {slp_path}")

    print(f"Done. {len(save_indices)} frames in {out}")
    return slp_path


# ----------------------------------------------------------------------------
# Generation: animated video with persistent tracks
# ----------------------------------------------------------------------------

def generate_video_dataset(cfg: VideoConfig) -> Path:
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.video_format == "mp4":
        _validate_mp4_shape(cfg.image_shape)

    save_set = set(select_frames(cfg.n_frames, cfg.save_frames))
    n_to_save = len(save_set)

    video_path = out / _video_filename(cfg.video_format)
    n_kp = cfg.skeleton.n_flagellum_interior
    duration = cfg.n_frames / cfg.fps

    parasites: List[L.ParasiteParams] = []
    for _ in range(cfg.n_parasites):
        p = L.sample_random_parasite(rng, cfg.image_shape, t=0.0)
        p.n_flagellum_keypoints = n_kp
        p.mode_schedule = L.generate_mode_schedule(p, duration, rng)
        parasites.append(p)

    bg = L.synthetic_background(cfg.image_shape, rng,
                                intensity=cfg.noise.bg_intensity)

    keypoints_per_frame: List[List[dict]] = []
    tracks_per_frame: List[List[Optional[int]]] = []
    ground_truth_frames: List[dict] = []
    dt = 1.0 / cfg.fps
    saved = 0

    with _VideoWriter(video_path, cfg.video_format,
                      fps=cfg.fps, mp4_quality=cfg.mp4_quality) as vw:
        for i in range(cfg.n_frames):
            t = i * dt
            # Always advance simulation so motion stays continuous.
            if i > 0:
                L.advance_parasites(parasites, dt, cfg.image_shape,
                                    periodic=cfg.periodic_boundary, t=t,
                                    optics=cfg.optics)

            if i not in save_set:
                continue

            image, kps = L.render_scene(
                parasites, t=t, image_shape=cfg.image_shape,
                optics=cfg.optics, noise=cfg.noise,
                background=bg, rng=rng, fast=cfg.fast,
            )
            vw.write_float(image)

            keypoints_per_frame.append(kps)
            tracks_per_frame.append(list(range(len(parasites))))
            ground_truth_frames.append({
                "saved_frame": saved,
                "sim_frame": i,
                "t_seconds": t,
                "instances": [_summarize_parasite(p) for p in parasites],
            })
            saved += 1

            if saved % 50 == 0 or saved == n_to_save:
                print(f"  saved {saved}/{n_to_save} frames "
                      f"(simulated {i + 1}/{cfg.n_frames})")

    with open(out / "ground_truth.json", "w") as f:
        json.dump({
            "mode": "video",
            "video": video_path.name,
            "fps": cfg.fps,
            "n_parasites": cfg.n_parasites,
            "n_frames_simulated": cfg.n_frames,
            "n_frames_saved": n_to_save,
            "image_shape": list(cfg.image_shape),
            "skeleton": {"nodes": cfg.skeleton.node_names,
                         "edges": cfg.skeleton.edges},
            "frames": ground_truth_frames,
        }, f, indent=2)

    slp_path = out / "labels.slp"
    if cfg.save_slp:
        # FPS in the labels reflects the *saved* video's effective rate
        # (e.g. 60 fps source saved every 5th frame -> 12 fps output).
        every_n_factor = cfg.n_frames / max(n_to_save, 1)
        effective_fps = cfg.fps / every_n_factor if every_n_factor > 0 else cfg.fps
        labels = build_sleap_labels(
            video_path, keypoints_per_frame, cfg.skeleton,
            tracks_per_frame=tracks_per_frame, fps=effective_fps,
        )
        save_labels(labels, slp_path, embed=cfg.embed_frames)
        print(f"  wrote {slp_path}")

    print(f"Done. {n_to_save} frames in {out}")
    return slp_path


# ----------------------------------------------------------------------------
# Subset / splits utilities
# ----------------------------------------------------------------------------

def subset_labels(slp_path: Path, indices: Sequence[int],
                  out_path: Path, *, embed: bool = False) -> Path:
    """Extract a frame subset from an existing .slp into a new .slp."""
    labels = sio.load_file(str(slp_path))
    subset = labels.extract(list(indices), copy=True)
    save_labels(subset, out_path, embed=embed)
    print(f"  wrote {out_path} ({len(indices)} frames)")
    return out_path


def make_splits(slp_path: Path, out_dir: Path, *,
                n_train: float = 0.8, n_val: float = 0.1,
                n_test: Optional[float] = 0.1,
                seed: int = 42, embed: bool = True) -> Path:
    """Train/val/test splits via sleap-io."""
    labels = sio.load_file(str(slp_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.make_training_splits(
        n_train=n_train, n_val=n_val, n_test=n_test,
        save_dir=str(out_dir), seed=seed, embed=embed,
    )
    print(f"  wrote splits to {out_dir}")
    return out_dir


# ----------------------------------------------------------------------------
# Multi-setup: YAML-driven generation
# ----------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override onto base. Lists are replaced, dicts are merged."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _as_tuple(v):
    return tuple(v) if isinstance(v, list) else v


def _build_optics(d: Optional[dict]) -> L.OpticsParams:
    if not d:
        return L.OpticsParams()
    valid = {f.name for f in dataclasses.fields(L.OpticsParams)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"unknown optics fields: {sorted(unknown)}")
    return L.OpticsParams(**d)


def _build_noise(d: Optional[dict]) -> L.CameraNoiseParams:
    if not d:
        return L.CameraNoiseParams()
    valid = {f.name for f in dataclasses.fields(L.CameraNoiseParams)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"unknown noise fields: {sorted(unknown)}")
    return L.CameraNoiseParams(**d)


def _build_skeleton(d: Optional[dict], flag_keypoints: int) -> SkeletonConfig:
    if d:
        return SkeletonConfig(**d)
    return SkeletonConfig(n_flagellum_interior=flag_keypoints)


def _build_setup_config(setup: dict) -> Tuple[str, DatasetConfig]:
    """Build a DatasetConfig or VideoConfig from a fully-resolved setup dict."""
    name = setup["name"]
    mode = setup.get("mode", "video")
    skeleton = _build_skeleton(setup.get("skeleton"),
                               setup.get("flag_keypoints", 6))
    optics = _build_optics(setup.get("optics"))
    noise = _build_noise(setup.get("noise"))

    common_kwargs = dict(
        out_dir=Path(setup["out_dir"]) / name,
        n_frames=int(setup.get("n_frames", 200)),
        image_shape=_as_tuple(setup.get("image_shape", (768, 768))),
        skeleton=skeleton,
        optics=optics,
        noise=noise,
        bg_intensity_range=_as_tuple(setup.get("bg_intensity_range", (0.25, 0.95))),
        seed=int(setup.get("seed", 0)),
        fast=bool(setup.get("fast", False)),
        video_format=setup.get("video_format", "tiff"),
        mp4_quality=int(setup.get("mp4_quality", 9)),
        mp4_fps=float(setup.get("mp4_fps", 30.0)),
        save_slp=bool(setup.get("save_slp", True)),
        embed_frames=bool(setup.get("embed_frames", False)),
        save_frames=setup.get("save_frames"),
        per_frame_jitter=bool(setup.get("per_frame_jitter", True)),
    )

    if mode == "video":
        return name, VideoConfig(
            **common_kwargs,
            fps=float(setup.get("fps", 60.0)),
            n_parasites=int(setup.get("n_parasites", 20)),
            periodic_boundary=bool(setup.get("periodic_boundary", True)),
        )
    elif mode == "random":
        return name, DatasetConfig(
            **common_kwargs,
            parasites_per_frame=_as_tuple(
                setup.get("parasites_per_frame", (5, 15))),
        )
    else:
        raise ValueError(f"setup '{name}': unknown mode {mode!r} "
                         f"(expected 'video' or 'random')")


def _resolve_multi_config(yaml_data: dict) -> Tuple[Path, List[dict]]:
    """Resolve top-level defaults onto each setup. Returns (out_dir, setups).

    Setups with `repeats: N` are expanded into N independent clips. Each clip
    gets its own seed and its own per-clip optics+noise jitter, so the same
    base setup spawns many short "sessions" with realistic between-session
    variation rather than one long block of identically-imaged frames."""
    if "setups" not in yaml_data:
        raise ValueError("config must contain a 'setups' list")
    if "output_dir" not in yaml_data:
        raise ValueError("config must specify 'output_dir'")

    out_dir = Path(yaml_data["output_dir"])
    base = {k: v for k, v in yaml_data.items()
            if k not in ("setups", "output_dir")}

    base_seed = int(base.get("seed", 0))
    resolved: List[dict] = []
    for i, setup in enumerate(yaml_data["setups"]):
        if "name" not in setup:
            raise ValueError(f"setup #{i} is missing a 'name'")
        merged = _deep_merge(base, setup)
        if "seed" not in setup:
            merged["seed"] = base_seed + i
        merged["out_dir"] = str(out_dir)

        repeats = int(merged.pop("repeats", 1))
        if repeats <= 1:
            resolved.append(merged)
            continue

        # Expand into N independent clips with per-clip optics jitter
        base_name = merged["name"]
        base_setup_seed = int(merged["seed"])
        # RNG seeded from base_setup_seed → reproducible jitter pattern
        jitter_rng = np.random.default_rng(base_setup_seed ^ 0xCAFEBABE)
        for k in range(repeats):
            clip = copy.deepcopy(merged)
            clip["name"] = f"{base_name}/clip_{k:03d}"
            clip["seed"] = base_setup_seed + 10000 + k
            # Apply realistic per-clip drift to optics & noise
            clip["optics"] = _jitter_optics_dict(clip.get("optics", {}), jitter_rng)
            clip["noise"] = _jitter_noise_dict(clip.get("noise", {}), jitter_rng)
            resolved.append(clip)

    return out_dir, resolved


def generate_multi(yaml_path: Path) -> List[Path]:
    """Run all setups in a multi-setup YAML config. Returns list of slp paths."""
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is required for the 'multi' command. "
            "Install with: pip install pyyaml")

    with open(yaml_path) as f:
        cfg_data = yaml.safe_load(f)

    out_dir, setups = _resolve_multi_config(cfg_data)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save fully-resolved config alongside outputs for reproducibility
    resolved_dump = {"output_dir": str(out_dir), "setups": setups}
    with open(out_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(resolved_dump, f, sort_keys=False, default_flow_style=False)

    slp_paths = []
    for i, setup in enumerate(setups):
        print(f"\n=== Setup {i + 1}/{len(setups)}: {setup['name']} ===")
        name, cfg = _build_setup_config(setup)
        if isinstance(cfg, VideoConfig):
            slp = generate_video_dataset(cfg)
        else:
            slp = generate_random_dataset(cfg)
        slp_paths.append(slp)

    print(f"\nAll {len(setups)} setups complete. Output in {out_dir}")
    return slp_paths


# ----------------------------------------------------------------------------
# Template YAML
# ----------------------------------------------------------------------------

_TEMPLATE_YAML = """\
# Multi-setup dataset configuration.
# Top-level keys are defaults; each entry under `setups:` deep-merges over them.
# Run with:  python dataset_builder.py multi this_file.yaml

output_dir: data/leishmania_train
mode: video           # 'video' or 'random'; overridable per setup
seed: 42              # base seed; per-setup default is base + setup_index

# Imaging / geometry
n_frames: 600
image_shape: [768, 768]
flag_keypoints: 8
fast: true
video_format: tiff    # 'tiff' (lossless) or 'mp4' (smaller, lossy)
embed_frames: false   # embed video bytes inside labels.slp (portable)

# Video-mode defaults
fps: 60.0
n_parasites: 20
periodic_boundary: true

# Random-mode defaults (only used when mode == 'random')
parasites_per_frame: [5, 15]
bg_intensity_range: [0.15, 0.30]

# Frame selection (which frames to write to disk):
#   all                                   - every frame
#   {every_n: 5}                          - every 5th frame
#   {every_n: 5, start: 100, end: 500}    - every 5th in window
#   {indices: [0, 50, 100, 200]}          - specific frames
#   {count: 60}                           - 60 evenly-spaced frames
#   {range: [100, 400]}                   - all frames in window
save_frames: all

# Optics (micrometres throughout). Defaults are tuned for ~60x.
optics:
  pixel_size_um: 0.108            # 100x ~ 0.065, 60x ~ 0.108, 40x ~ 0.163, 20x = 0.325
  psf_sigma_um: 0.13
  halo_strength: 0.7
  halo_lowpass_sigma_um: 3.9
  intensity_gain: 0.5
  shadeoff_threshold: 0.77
  shadeoff_strength: 0.15
  body_edge_smooth_sigma_um: 0.3

# Camera noise
noise:
  full_well_photons: 800.0
  read_noise_e: 5
  bg_intensity: 0.2
  dark_offset: -0.25

# One subdirectory per setup. Only specify what differs from defaults.
setups:
  - name: 60x_clean
    # uses everything from defaults (60x, low-light, dark bg)

  - name: 40x_clean
    optics:
      pixel_size_um: 0.163
    n_parasites: 25                # smaller parasites in image -> can fit more

  - name: 60x_high_noise
    noise:
      read_noise_e: 15
      bg_intensity: 0.12

  - name: 100x_few_cells
    optics:
      pixel_size_um: 0.065
      psf_sigma_um: 0.10
    n_parasites: 4
    save_frames: {every_n: 10}     # high mag, save sparsely

  - name: 20x_wide_field
    optics:
      pixel_size_um: 0.325
    n_parasites: 60                # huge field of view, lots of cells
    noise:
      bg_intensity: 0.4            # 20x usually has more light per pixel

  - name: 60x_random_diverse
    mode: random
    n_frames: 200
    parasites_per_frame: [3, 12]
"""


def write_template(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE_YAML)
    print(f"wrote starter config to {path}")
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_indices(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--size", type=int, nargs=2, default=(768, 768),
                   metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true",
                   help="use the fast rendering path")
    p.add_argument("--format", choices=["tiff", "mp4"], default="tiff",
                   dest="video_format",
                   help="video container; tiff is lossless, mp4 is smaller")
    p.add_argument("--mp4-quality", type=int, default=9,
                   help="mp4 encoder quality, 0-10 (default: 9)")
    p.add_argument("--embed", action="store_true",
                   help="embed video data inside labels.slp (portable but large)")
    p.add_argument("--flag-keypoints", type=int, default=6,
                   help="number of interior flagellum nodes (total = 3 + this)")
    p.add_argument("--save-every", type=int, default=1,
                   help="save every N-th frame (default: 1 = all frames)")


def _save_frames_from_args(args) -> SaveFramesSpec:
    if getattr(args, "save_every", 1) and args.save_every > 1:
        return {"every_n": args.save_every}
    return None


def cli_random(args: argparse.Namespace) -> None:
    skeleton = SkeletonConfig(n_flagellum_interior=args.flag_keypoints)
    cfg = DatasetConfig(
        out_dir=args.out, n_frames=args.frames,
        image_shape=tuple(args.size),
        parasites_per_frame=tuple(args.parasites),
        skeleton=skeleton, seed=args.seed, fast=args.fast,
        video_format=args.video_format, mp4_quality=args.mp4_quality,
        embed_frames=args.embed,
        save_frames=_save_frames_from_args(args),
    )
    generate_random_dataset(cfg)


def cli_video(args: argparse.Namespace) -> None:
    skeleton = SkeletonConfig(n_flagellum_interior=args.flag_keypoints)
    cfg = VideoConfig(
        out_dir=args.out, n_frames=args.frames,
        image_shape=tuple(args.size),
        skeleton=skeleton, seed=args.seed, fast=args.fast,
        video_format=args.video_format, mp4_quality=args.mp4_quality,
        embed_frames=args.embed, fps=args.fps,
        n_parasites=args.n_parasites,
        periodic_boundary=not args.no_periodic,
        save_frames=_save_frames_from_args(args),
    )
    generate_video_dataset(cfg)


def cli_subset(args: argparse.Namespace) -> None:
    indices = _parse_indices(args.indices)
    subset_labels(args.slp, indices, args.out, embed=args.embed)


def cli_splits(args: argparse.Namespace) -> None:
    make_splits(args.slp, args.out, n_train=args.train, n_val=args.val,
                n_test=args.test, seed=args.seed, embed=args.embed)


def cli_multi(args: argparse.Namespace) -> None:
    generate_multi(args.config)


def cli_template(args: argparse.Namespace) -> None:
    write_template(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rand = sub.add_parser("random", help="random parasites per frame")
    _add_common_args(p_rand)
    p_rand.add_argument("--parasites", type=int, nargs=2, default=(5, 15),
                        metavar=("MIN", "MAX"))
    p_rand.set_defaults(func=cli_random)

    p_vid = sub.add_parser("video", help="animated video with persistent tracks")
    _add_common_args(p_vid)
    p_vid.add_argument("--fps", type=float, default=60.0)
    p_vid.add_argument("--n-parasites", type=int, default=20)
    p_vid.add_argument("--no-periodic", action="store_true",
                       help="disable wrap-around at image edges")
    p_vid.set_defaults(func=cli_video)

    p_multi = sub.add_parser("multi",
                             help="run multiple setups from a YAML config")
    p_multi.add_argument("config", type=Path,
                         help="path to multi-setup YAML config")
    p_multi.set_defaults(func=cli_multi)

    p_tpl = sub.add_parser("template",
                           help="write a starter multi-setup YAML to disk")
    p_tpl.add_argument("-o", "--out", type=Path, required=True)
    p_tpl.set_defaults(func=cli_template)

    p_sub = sub.add_parser("subset", help="extract a frame subset to a new .slp")
    p_sub.add_argument("slp", type=Path, help="input labels.slp")
    p_sub.add_argument("--indices", type=str, required=True,
                       help="frame indices: '0,5,10' or '0-99' or '0-9,15,20-25'")
    p_sub.add_argument("-o", "--out", type=Path, required=True)
    p_sub.add_argument("--embed", action="store_true")
    p_sub.set_defaults(func=cli_subset)

    p_spl = sub.add_parser("splits", help="train/val/test split via sleap-io")
    p_spl.add_argument("slp", type=Path, help="input labels.slp")
    p_spl.add_argument("-o", "--out", type=Path, required=True)
    p_spl.add_argument("--train", type=float, default=0.8)
    p_spl.add_argument("--val", type=float, default=0.1)
    p_spl.add_argument("--test", type=float, default=0.1)
    p_spl.add_argument("--seed", type=int, default=42)
    p_spl.add_argument("--embed", action="store_true",
                       help="embed video data in split files (default: False)")
    p_spl.set_defaults(func=cli_splits)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
