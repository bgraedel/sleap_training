"""Convert a directory of .nd2 files to uint8 TIFF stacks matching training preprocessing."""

import nd2
import numpy as np
import tifffile
from pathlib import Path


def nd2_to_tiff(
    nd2_path,
    tiff_path,
    crop_size=None,         # (h, w) for center-crop, or None for full frame
    channel=0,              # for multichannel ND2s
    norm_percentiles=(0, 100),  # (0, 100) = per-frame min-max
    overwrite=False,
):
    nd2_path = Path(nd2_path)
    tiff_path = Path(tiff_path)
    if tiff_path.exists() and not overwrite:
        print(f"  exists, skipping: {tiff_path.name}")
        return tiff_path

    with nd2.ND2File(str(nd2_path)) as f:
        lazy = f.to_dask()  # avoids loading full stack into RAM

        # Resolve dimensions
        s = lazy.shape
        if len(s) == 3:                   # (T, Y, X)
            n_frames, full_h, full_w = s
            multichannel = False
        elif len(s) == 4:                 # (T, C, Y, X)
            n_frames, _, full_h, full_w = s
            multichannel = True
        else:
            raise ValueError(f"Unsupported ND2 shape {s}")

        # Crop box
        if crop_size is not None:
            h, w = crop_size
            if h > full_h or w > full_w:
                raise ValueError(f"crop_size {crop_size} > frame {(full_h, full_w)}")
            y0 = (full_h - h) // 2
            x0 = (full_w - w) // 2
            y1, x1 = y0 + h, x0 + w
        else:
            y0, x0, y1, x1 = 0, 0, full_h, full_w

        # Stream-write frame-by-frame so memory stays bounded
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tw:
            for i in range(n_frames):
                if multichannel:
                    raw = np.asarray(lazy[i, channel, y0:y1, x0:x1])
                else:
                    raw = np.asarray(lazy[i, y0:y1, x0:x1])

                # Per-frame normalization → uint8
                raw = raw.astype(np.float32)
                lo, hi = np.percentile(raw, norm_percentiles)
                out = np.clip((raw - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
                tw.write(out, contiguous=True)

    return tiff_path


# Batch convert a directory
def batch_convert(nd2_dir, tiff_dir, **kwargs):
    nd2_dir = Path(nd2_dir)
    tiff_dir = Path(tiff_dir)
    tiff_dir.mkdir(parents=True, exist_ok=True)

    nd2_files = sorted(nd2_dir.glob("*.nd2"))
    print(f"Found {len(nd2_files)} ND2 files")

    for i, nd2_path in enumerate(nd2_files, 1):
        tiff_path = tiff_dir / f"{nd2_path.stem}.tif"
        print(f"[{i}/{len(nd2_files)}] {nd2_path.name}")
        try:
            nd2_to_tiff(nd2_path, tiff_path, **kwargs)
        except Exception as e:
            print(f"   !! error: {e}")


if __name__ == "__main__":
    batch_convert(
        nd2_dir=Path("D:/Waveform Dataset 2026/C9T7 parental/POWA/CpdB"),
        tiff_dir=Path("D:/Waveform Dataset 2026/C9T7 parental/POWA/CpdB/tiffs"),
        crop_size=(1024, 1024),
        channel=0,
        norm_percentiles=(0, 100),
    )