"""
trace_flagellum.py — SpermQ-inspired flagellum tracing for SLEAP labels.

v3: initializes from the full set of inferred nodes (Base + Flag1..Flag5 + Tip)
    rather than a linear interpolation between Base and Tip. Image-based
    perpendicular refinement is kept — it's what gives sub-pixel accuracy.

Workflow:
  1. Test on one instance with visualization (`MODE = "test"`)
  2. Once happy, run on full dataset (`MODE = "full"`)
"""

import sleap_io as sio
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.interpolate import splprep, splev


# ============================================================
# CONFIG
# ============================================================
INPUT_LABELS = "labels.v002.slp"   # labels with Base + Flag1..Flag5 + Tip
OUTPUT_LABELS = "labels.v003.slp"

# Input node names IN ORDER from base to tip. The first and last are kept fixed
# as endpoints; everything in between is used as a control point for the spline
# initialization and then refined.
INPUT_NODE_ORDER = ["Base", "Flag1", "Flag2", "Flag3", "Flag4", "Flag5", "Tip"]

N_OUT_POINTS = 5                   # interior output nodes
OUT_NODE_PREFIX = "Flag"

# Tracing parameters
TRACE_PARAMS = dict(
    n_points=N_OUT_POINTS,
    n_iterations=20,
    normal_length=12,           # how far to search perpendicular (px)
    padding=20,                 # crop padding around control points (px)
    smoothing_sigma=0.4,        # pre-smoothing of input (px)
    max_drift=6.0,              # tighter than before — init is much better now
    min_amp_ratio=0.4,          # min Gaussian fit quality
    spline_smoothing=0.5,       # output spline smoothing
    init_spline_smoothing=0.0,  # 0 = interpolate through SLEAP points; raise to
                                # smooth out SLEAP jitter before refinement
    n_refine_points=None,       # None = 3x n_points
)

MODE = "test"                   # "test" or "full"
TEST_FRAME_IDX = 1
TEST_INSTANCE_IDX = 13


# ============================================================
# CORE FUNCTIONS
# ============================================================
def to_grayscale(image):
    image = np.asarray(image)
    if image.ndim == 3:
        gray = image[..., 0] if image.shape[-1] == 1 else image.mean(axis=-1)
    else:
        gray = image
    gray = gray.astype(np.float32)
    if gray.max() > 1:
        gray = gray / gray.max()
    return gray


def sample_along_line(image, center, direction, length, n_samples=None):
    if n_samples is None:
        n_samples = 2 * length + 1
    offsets = np.linspace(-length, length, n_samples)
    coords_x = center[0] + offsets * direction[0]
    coords_y = center[1] + offsets * direction[1]
    samples = map_coordinates(image, [coords_y, coords_x], order=1, mode="nearest")
    return offsets, samples


def gaussian_profile(x, amp, mean, sigma, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset


def refine_point_perpendicular(inv_image, point, tangent,
                                normal_length=8, expected_sigma=1.5,
                                max_shift_frac=0.7, min_amp_ratio=0.4):
    """Sample perpendicular to tangent, fit Gaussian, return refined position."""
    normal = np.array([-tangent[1], tangent[0]])
    offsets, samples = sample_along_line(inv_image, point, normal, normal_length)

    peak_idx = int(np.argmax(samples))
    sample_range = samples.max() - samples.min()

    if sample_range < 0.05:
        return point, False

    p0 = [sample_range, offsets[peak_idx], expected_sigma, samples.min()]
    bounds = (
        [0, -normal_length, 0.5, 0],
        [np.inf, normal_length, normal_length, np.inf],
    )

    try:
        popt, _ = curve_fit(gaussian_profile, offsets, samples, p0=p0,
                             bounds=bounds, maxfev=200)
        amp, fit_mean, fit_sigma = popt[0], popt[1], popt[2]
        if fit_sigma > normal_length * 0.4 or amp < sample_range * min_amp_ratio:
            return point, False
        shift = fit_mean
    except Exception:
        return point, False

    shift = float(np.clip(shift, -normal_length * max_shift_frac,
                          normal_length * max_shift_frac))
    return point + shift * normal, True


def _piecewise_linear_resample(points, n_samples):
    """Arc-length resample of a polyline. Used as a spline fallback."""
    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate(([0], np.cumsum(seg_lens)))
    total = cum[-1]
    if total < 1e-9:
        return np.tile(points[0], (n_samples, 1))
    targets = np.linspace(0, total, n_samples)
    out = np.zeros((n_samples, 2))
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum, t)
        if idx <= 0:
            out[i] = points[0]
        elif idx >= len(cum):
            out[i] = points[-1]
        else:
            t0, t1 = cum[idx - 1], cum[idx]
            alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-9 else 0
            out[i] = points[idx - 1] * (1 - alpha) + points[idx] * alpha
    return out


def build_initial_curve(control_points, n_samples, smoothing=0.0):
    """
    Build a dense initial curve from control points.

    With >=3 control points, fits a cubic (or lower-order) spline. With
    smoothing=0 the spline passes through every control point. With
    smoothing>0 it is allowed to deviate, which is useful if the SLEAP
    inference is noisy. Endpoints are always preserved exactly.
    """
    control_points = np.asarray(control_points, dtype=np.float64)
    if len(control_points) < 2:
        raise ValueError("Need at least 2 control points")
    if len(control_points) == 2:
        fractions = np.linspace(0, 1, n_samples)
        return np.array([control_points[0] * (1 - f) + control_points[-1] * f
                         for f in fractions])
    try:
        k = min(3, len(control_points) - 1)
        tck, _ = splprep(control_points.T, s=smoothing, k=k)
        u_dense = np.linspace(0, 1, n_samples)
        dense = np.array(splev(u_dense, tck)).T
        dense[0] = control_points[0]
        dense[-1] = control_points[-1]
        return dense
    except Exception:
        return _piecewise_linear_resample(control_points, n_samples)


def smooth_and_resample(points, n_target, smoothing=1.5, dense_n=500):
    """Spline-smooth and resample at equal arc-length intervals."""
    if len(points) < 4:
        return points
    try:
        tck, _ = splprep(points.T, s=smoothing, k=min(3, len(points) - 1))

        u_dense = np.linspace(0, 1, dense_n)
        dense = np.array(splev(u_dense, tck)).T

        diffs = np.diff(dense, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate(([0], np.cumsum(seg_lens)))
        total_len = cum_len[-1]
        if total_len < 1e-9:
            return points

        target_lens = np.linspace(0, total_len, n_target)
        result = np.zeros((n_target, 2))
        for i, t in enumerate(target_lens):
            idx = np.searchsorted(cum_len, t)
            if idx <= 0:
                result[i] = dense[0]
            elif idx >= len(cum_len):
                result[i] = dense[-1]
            else:
                t0, t1 = cum_len[idx - 1], cum_len[idx]
                alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-9 else 0
                result[i] = dense[idx - 1] * (1 - alpha) + dense[idx] * alpha

        result[0] = points[0]
        result[-1] = points[-1]
        return result
    except Exception:
        return points


def trace_flagellum(image, control_xy,
                     n_points=5, n_iterations=20, normal_length=12,
                     padding=20, smoothing_sigma=0.8,
                     max_drift=6.0, min_amp_ratio=0.4,
                     spline_smoothing=0.5,
                     init_spline_smoothing=0.0,
                     n_refine_points=None):
    """
    Trace a flagellum centerline using image-based perpendicular refinement.

    Parameters
    ----------
    image : ndarray
        Frame to trace.
    control_xy : ndarray, shape (M, 2)
        Control points (x, y) ordered from base to tip. Used to build the
        initial curve via spline interpolation. With M==2 this falls back
        to a linear initialization (original behavior).
    n_points : int
        Number of interior output points (excluding endpoints).
    """
    gray = to_grayscale(image)
    H, W = gray.shape

    control_xy = np.asarray(control_xy, dtype=np.float64)

    # Bounding box around ALL control points (not just endpoints) so the crop
    # always contains the whole flagellum even when it bows out sideways.
    x_min = max(0, int(control_xy[:, 0].min() - padding))
    x_max = min(W, int(control_xy[:, 0].max() + padding))
    y_min = max(0, int(control_xy[:, 1].min() - padding))
    y_max = min(H, int(control_xy[:, 1].max() + padding))
    crop = gray[y_min:y_max, x_min:x_max]

    smooth = gaussian(crop, sigma=smoothing_sigma)
    inverted = 1.0 - smooth

    if n_refine_points is None:
        n_refine_points = max(n_points * 3, 12)

    control_local = control_xy - np.array([x_min, y_min])
    initial_points = build_initial_curve(
        control_local, n_refine_points + 2, smoothing=init_spline_smoothing
    )
    points = initial_points.copy()

    moves_per_iter = []
    for _ in range(n_iterations):
        new_points = points.copy()
        moved_count = 0
        for i in range(1, len(points) - 1):
            tangent = points[i + 1] - points[i - 1]
            n = np.linalg.norm(tangent)
            if n < 1e-6:
                continue
            tangent /= n
            candidate, fit_ok = refine_point_perpendicular(
                inverted, points[i], tangent,
                normal_length=normal_length,
                min_amp_ratio=min_amp_ratio,
            )
            if not fit_ok:
                continue
            # Leash relative to the SLEAP-derived initial curve
            drift = np.linalg.norm(candidate - initial_points[i])
            if drift > max_drift:
                direction = (candidate - initial_points[i]) / drift
                candidate = initial_points[i] + direction * max_drift
            new_points[i] = candidate
            moved_count += 1
        moves_per_iter.append(moved_count)
        points = new_points

    points = smooth_and_resample(points, n_target=n_points + 2,
                                  smoothing=spline_smoothing)

    interior_global = points[1:-1] + np.array([x_min, y_min])

    diagnostics = {
        "crop": crop,
        "inverted": inverted,
        "all_points_local": points,
        "initial_points_local": initial_points,
        "control_local": control_local,
        "moves_per_iter": moves_per_iter,
        "x_min": x_min,
        "y_min": y_min,
    }
    return interior_global, diagnostics


# ============================================================
# SLEAP I/O HELPERS
# ============================================================
def get_control_points(instance, node_names):
    """Extract (x, y) for the given node names in order. Returns None if any
    node is missing or has NaN coords."""
    pts = []
    for name in node_names:
        try:
            pt = instance[name].numpy()
        except (KeyError, IndexError, AttributeError):
            return None
        if pt is None or np.any(np.isnan(pt)):
            return None
        pts.append(pt)
    return np.array(pts, dtype=np.float64)


# ============================================================
# VISUALIZATION
# ============================================================
def visualize_test(image, control_xy, refined_xy, diag):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].plot(control_xy[:, 0], control_xy[:, 1], "o--", color="orange",
                 label="SLEAP control points", markersize=6)
    full_curve = np.vstack([control_xy[[0]], refined_xy, control_xy[[-1]]])
    axes[0].plot(full_curve[:, 0], full_curve[:, 1], "x-", color="cyan",
                 label="Refined output", markersize=8)
    axes[0].legend(); axes[0].set_title("Full frame")

    axes[1].imshow(diag["crop"], cmap="gray")
    init = diag["initial_points_local"]
    final = diag["all_points_local"]
    ctrl = diag["control_local"]
    axes[1].plot(init[:, 0], init[:, 1], ".", color="yellow", alpha=0.5,
                 label=f"Initial dense ({len(init)})")
    axes[1].plot(final[:, 0], final[:, 1], "+", color="red", markersize=10,
                 label=f"Refined ({len(final)})")
    axes[1].plot(ctrl[:, 0], ctrl[:, 1], "o", color="orange",
                 label="SLEAP control")
    axes[1].legend()
    axes[1].set_title(f"Crop · moves/iter: {diag['moves_per_iter']}")
    plt.tight_layout(); plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    labels = sio.load_slp(INPUT_LABELS)

    if MODE == "test":
        lf = labels[TEST_FRAME_IDX]
        inst = lf.instances[TEST_INSTANCE_IDX]
        control = get_control_points(inst, INPUT_NODE_ORDER)
        if control is None:
            raise RuntimeError("Test instance has missing nodes")
        image = lf.image
        refined, diag = trace_flagellum(image, control, **TRACE_PARAMS)
        print(f"Refined to {len(refined)} interior points")
        visualize_test(image, control, refined, diag)

    elif MODE == "full":
        out_node_names = (
            [INPUT_NODE_ORDER[0]]
            + [f"{OUT_NODE_PREFIX}{i + 1}" for i in range(N_OUT_POINTS)]
            + [INPUT_NODE_ORDER[-1]]
        )
        new_skeleton = sio.Skeleton(nodes=out_node_names)

        new_lfs = []
        for lf in labels:
            image = lf.image
            new_instances = []
            for inst in lf.instances:
                control = get_control_points(inst, INPUT_NODE_ORDER)
                if control is None:
                    continue
                try:
                    refined, _ = trace_flagellum(image, control, **TRACE_PARAMS)
                except Exception as e:
                    print(f"Skipping instance at frame {lf.frame_idx}: {e}")
                    continue
                all_pts = np.vstack([control[[0]], refined, control[[-1]]])
                new_inst = sio.Instance.from_numpy(
                    points_data=all_pts, skeleton=new_skeleton
                )
                new_instances.append(new_inst)
            new_lfs.append(sio.LabeledFrame(
                video=lf.video, frame_idx=lf.frame_idx, instances=new_instances
            ))

        new_labels = sio.Labels(
            videos=labels.videos,
            skeletons=[new_skeleton],
            labeled_frames=new_lfs,
        )
        new_labels.save(OUTPUT_LABELS)
        print(f"Saved → {OUTPUT_LABELS}")