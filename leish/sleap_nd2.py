"""ND2 video backend for sleap-io / sleap-nn.

Adds support for Nikon .nd2 files in SLEAP inference pipelines without
needing to convert to TIFF first. Includes optional center-cropping and
per-frame min-max → uint8 normalization to match common training setups.

Basic usage
-----------
    import sleap_nd2
    from sleap_nn.predict import run_inference

    # One-time registration (so sleap-nn's reopen-by-filename works).
    sleap_nd2.register(default_crop_size=(1024, 1024))

    with sleap_nd2.open_nd2("file.nd2", crop_size=(1024, 1024)) as video:
        labels = run_inference(
            input_video=video,
            model_paths=[model_path],
            output_path="predictions.slp",
            device="cuda", batch_size=8, peak_threshold=0.5,
            filter_min_visible_nodes=5,
        )

Coordinates note
----------------
When cropping, predicted (x, y) are relative to the crop's top-left corner.
Use `Nd2Video.crop_offset` (returns `(x0, y0)`) to remap back to absolute
ND2 coordinates if you need that downstream.

Multi-position / z-stack ND2s
-----------------------------
This module assumes a (T, Y, X) or (T, C, Y, X) layout. For ND2s with P
(positions) or Z dimensions you'll want to subclass `Nd2Video` and override
`_ensure_open` to flatten or slice the extra axis.
"""

from __future__ import annotations

import attrs
import numpy as np
import nd2
import sleap_io as sio
from contextlib import contextmanager
from sleap_io.io.video_reading import VideoBackend


# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────


@attrs.define
class Nd2Video(VideoBackend):
    """sleap-io VideoBackend backed by a Nikon ND2 file.

    Keeps the ND2 file handle open across reads for performance. Call
    `.close()` explicitly when done, or use `open_nd2()` for scoped cleanup.

    Attributes:
        normalize_uint8: If True, convert each frame to uint8.
        norm_percentiles: (lo, hi) percentiles for normalization. `(0, 100)`
            is mathematically equivalent to per-frame min-max. `(0.1, 99.9)`
            is a hot-pixel-robust alternative.
        crop_size: Optional (h, w) for a centered crop.
        crop_box: Optional (y0, x0, y1, x1) for an explicit crop. Takes
            precedence over `crop_size` if both are set.
        channel: Channel index to read for multichannel (T, C, Y, X) ND2s.
    """

    EXTS = ("nd2",)

    normalize_uint8: bool = True
    norm_percentiles: tuple = (0.0, 100.0)
    crop_size: tuple | None = None
    crop_box: tuple | None = None
    channel: int = 0

    _file: object = attrs.field(init=False, default=None)
    _lazy: object = attrs.field(init=False, default=None)
    _box: tuple | None = attrs.field(init=False, default=None)

    # ── opening / resource management ────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        self._file = nd2.ND2File(self.filename)
        self._lazy = self._file.to_dask()

        s = self._lazy.shape
        if len(s) == 3:
            full_h, full_w = s[1], s[2]
        elif len(s) == 4:
            full_h, full_w = s[2], s[3]
        else:
            self.close()
            raise ValueError(
                f"Unsupported ND2 shape {s}; subclass Nd2Video to handle this layout."
            )

        if self.crop_box is not None:
            y0, x0, y1, x1 = self.crop_box
            if not (0 <= y0 < y1 <= full_h and 0 <= x0 < x1 <= full_w):
                raise ValueError(
                    f"crop_box {self.crop_box} out of bounds for frame {(full_h, full_w)}"
                )
            self._box = (y0, x0, y1, x1)
        elif self.crop_size is not None:
            h, w = self.crop_size
            if h > full_h or w > full_w:
                raise ValueError(
                    f"crop_size {(h, w)} larger than frame {(full_h, full_w)}"
                )
            y0 = (full_h - h) // 2
            x0 = (full_w - w) // 2
            self._box = (y0, x0, y0 + h, x0 + w)
        else:
            self._box = (0, 0, full_h, full_w)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._lazy = None
                self._box = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── required VideoBackend interface ──────────────────────────────────────

    @property
    def num_frames(self) -> int:
        self._ensure_open()
        return self._lazy.shape[0]

    @property
    def img_shape(self) -> tuple:
        self._ensure_open()
        y0, x0, y1, x1 = self._box
        return (y1 - y0, x1 - x0, 1)

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        self._ensure_open()
        frame = self._slice_frame(frame_idx)
        if frame.ndim == 2:
            frame = frame[..., None]
        return self._normalize(frame)

    def _read_frames(self, frame_inds) -> np.ndarray:
        self._ensure_open()
        raw = self._slice_frame(list(frame_inds))
        return np.stack(
            [self._normalize(f[..., None] if f.ndim == 2 else f) for f in raw]
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def crop_offset(self) -> tuple:
        """`(x0, y0)` of the crop within the original frame.

        Add these to predicted coordinates to recover absolute ND2 coords.
        """
        self._ensure_open()
        return (self._box[1], self._box[0])

    def _slice_frame(self, idx):
        """Lazy slice — only the cropped region is read from disk."""
        y0, x0, y1, x1 = self._box
        if self._lazy.ndim == 3:
            return np.asarray(self._lazy[idx, y0:y1, x0:x1])
        return np.asarray(self._lazy[idx, self.channel, y0:y1, x0:x1])

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        if not self.normalize_uint8 or frame.dtype == np.uint8:
            return frame
        lo, hi = np.percentile(frame, self.norm_percentiles)
        out = np.clip(
            (frame.astype(np.float32) - lo) / max(hi - lo, 1e-9) * 255, 0, 255
        )
        return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_video(
    path,
    *,
    crop_size=None,
    crop_box=None,
    channel: int = 0,
    normalize_uint8: bool = True,
    norm_percentiles=(0.0, 100.0),
) -> sio.Video:
    """Build an `sio.Video` wrapping an ND2 file. Caller is responsible for
    closing via `video.backend.close()` (or use `open_nd2()` for auto-cleanup)."""
    backend = Nd2Video(
        filename=str(path),
        grayscale=True,
        crop_size=crop_size,
        crop_box=crop_box,
        channel=channel,
        normalize_uint8=normalize_uint8,
        norm_percentiles=norm_percentiles,
    )
    return sio.Video(filename=str(path), backend=backend)


@contextmanager
def open_nd2(path, **kwargs):
    """Context-managed version of `make_video()`; closes the file on exit.

    Yields:
        `sio.Video` pointing at the ND2.
    """
    video = make_video(path, **kwargs)
    try:
        yield video
    finally:
        video.backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# Filename-based registration
# ─────────────────────────────────────────────────────────────────────────────
#
# sleap-nn's predictor re-opens the video by filename in the post-prediction
# stage, going through `VideoBackend.from_filename`. That router maps file
# extensions to backend classes and doesn't know about .nd2, so it raises
# `ValueError: Unknown video file type: ...`. The monkey-patch below makes
# .nd2 route to `Nd2Video` with whatever defaults were registered.
#
# Important: the defaults supplied to `register()` MUST match the crop /
# channel you used when constructing your primary Video, or the reopened
# backend will read a different region and predicted coordinates won't align.

_registered: bool = False
_original_from_filename = None
_original_make_video = None
_default_crop_size = None
_default_crop_box = None
_default_channel = 0


def register(
    *,
    default_crop_size=None,
    default_crop_box=None,
    default_channel: int = 0,
) -> None:
    """Register .nd2 routing for both fresh video opens and .slp deserialization.

    The defaults are used both when sleap-nn auto-reopens an .nd2 during
    inference AND when sleap-io loads a previously-saved .slp file whose
    video metadata is missing the 'backend' key (which happens because
    Nd2Video isn't a sleap-io-native backend). They MUST match the crop /
    channel you used during inference, otherwise loaded predictions won't
    align with the source frames.
    """
    global _registered, _original_from_filename, _original_make_video
    global _default_crop_size, _default_crop_box, _default_channel

    _default_crop_size = default_crop_size
    _default_crop_box = default_crop_box
    _default_channel = default_channel

    if _registered:
        return

    # Patch 1: VideoBackend.from_filename — routes .nd2 to Nd2Video on open.
    _original_from_filename = VideoBackend.from_filename.__func__

    def patched_from_filename(cls, filename, dataset=None, grayscale=None,
                              keep_open=True, **kwargs):
        if str(filename).lower().endswith(".nd2"):
            return Nd2Video(
                filename=str(filename),
                grayscale=True if grayscale is None else grayscale,
                keep_open=keep_open,
                crop_size=_default_crop_size,
                crop_box=_default_crop_box,
                channel=_default_channel,
            )
        return _original_from_filename(
            cls, filename, dataset=dataset, grayscale=grayscale,
            keep_open=keep_open, **kwargs,
        )

    VideoBackend.from_filename = classmethod(patched_from_filename)

    # Patch 2: sleap_io.io.slp.make_video — fall through to from_filename when
    # backend metadata is missing (our case, since Nd2Video doesn't serialize).
    import sleap_io.io.slp as _slp_mod
    _original_make_video = _slp_mod.make_video

    def patched_make_video(*args, **kwargs):
            # Find the video_json arg (the dict, not the labels_path string)
            video_json = kwargs.get("video_json")
            if video_json is None:
                for arg in args:
                    if isinstance(arg, dict):
                        video_json = arg
                        break

            if isinstance(video_json, dict) and "backend" not in video_json:
                filename = video_json.get("filename", "")
                if str(filename).lower().endswith(".nd2"):
                    backend = VideoBackend.from_filename(filename)
                    return sio.Video(filename=str(filename), backend=backend)

            return _original_make_video(*args, **kwargs)

    _slp_mod.make_video = patched_make_video

    _registered = True


def unregister() -> None:
    global _registered, _original_from_filename, _original_make_video
    if not _registered:
        return
    VideoBackend.from_filename = classmethod(_original_from_filename)
    import sleap_io.io.slp as _slp_mod
    _slp_mod.make_video = _original_make_video
    _registered = False
    _original_from_filename = None
    _original_make_video = None


# ─────────────────────────────────────────────────────────────────────────────
# Post-inference coordinate remapping
# ─────────────────────────────────────────────────────────────────────────────


def remap_to_absolute(labels_path_in, labels_path_out, offset):
    """Translate every predicted point by `offset = (x0, y0)` and re-save.

    Use this when you ran inference on a cropped ND2 and need predictions
    in original-frame coordinates downstream.
    """
    x0, y0 = offset
    labels = sio.load_file(str(labels_path_in))
    for lf in labels:
        for inst in lf.instances:
            for pt in inst.points.values():
                pt.x += x0
                pt.y += y0
    labels.save(str(labels_path_out))
