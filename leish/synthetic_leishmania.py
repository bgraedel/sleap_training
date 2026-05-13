"""
Synthetic Leishmania promastigote phase-contrast image generator.

Pipeline: body geometry -> flagellum waveform -> phase map -> PhC optics
-> camera noise -> compositing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, List, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class OpticsParams:
    """Phase-contrast imaging defaults tuned for ~60x on a Kinetix-class
    camera. All sigmas are in micrometres; they are converted to pixels at
    render time using pixel_size_um."""
    pixel_size_um: float = 0.108               # 60x objective; was 0.325 (20x)
    psf_sigma_um: float = 0.13                 # was 0.4225
    halo_strength: float = 0.7
    halo_lowpass_sigma_um: float = 3.9
    intensity_gain: float = 0.5                # was 0.9
    shadeoff_threshold: float = 0.77           # was 0.6
    shadeoff_strength: float = 0.15
    body_edge_smooth_sigma_um: float = 0.3     # was 0.39


@dataclass
class CameraNoiseParams:
    """Camera noise defaults tuned for realistic Ph1 acquisition."""
    full_well_photons: float = 2000.0   # moderate-light regime, realistic for sCMOS
    read_noise_e: float = 3             # modern sCMOS (Kinetix etc) is 1.5-3
    dark_offset: float = -0.05          # near-zero for well-calibrated cameras
    bg_intensity: float = 0.7           # bright field with contrast headroom for dark cells


@dataclass
class ParasiteParams:
    """Single Leishmania promastigote at a single time point.

    Spatial fields (lengths, widths, wavelengths, amplitudes, speeds) are
    in micrometres / micrometres-per-second. They get converted to output
    pixels at render time using OpticsParams.pixel_size_um.

    center_x/center_y are an exception — they are image-pixel coordinates,
    not micrometres."""
    center_x: float = 0.0           # image px
    center_y: float = 0.0           # image px
    body_orientation: float = 0.0   # rad

    # Body shape (um)
    body_length: float = 14.0       # um (Leishmania promastigote ~10-18)
    body_width: float = 2.2         # um (typical 1.5-3)
    body_peak_position: float = 0.5 # widest point along axis, [-1, 1]
    body_curvature: float = 0.0     # banana-bend; +/- 0.10 noticeable
    body_phase_shift: float = 4.3   # unitless phase (was 1.8; tuned for darker bodies)

    # Flagellum (um)
    flagellum_length: float = 16.0      # um (~10-25, often longer than body)
    flagellum_width: float = 0.25       # um (~0.2-0.3)
    flagellum_phase_shift: float = 0.5  # unitless phase

    # Active beat parameters
    beat_mode: str = "tip_to_base"      # 'tip_to_base' | 'base_to_tip' | 'static'
    beat_frequency: float = 22.0        # Hz
    beat_wavelength: float = 7.0        # um
    beat_amplitude_max: float = 0.6     # um
    beat_phase: float = 0.0             # rad

    # Per-mode beat parameters (override active values after mode switches)
    tip_to_base_frequency: Optional[float] = None    # Hz
    tip_to_base_wavelength: Optional[float] = None   # um
    tip_to_base_amplitude: Optional[float] = None    # um
    base_to_tip_frequency: Optional[float] = None    # Hz
    base_to_tip_wavelength: Optional[float] = None   # um
    base_to_tip_amplitude: Optional[float] = None    # um

    # Base-to-tip beat shape (Wheeler 2020 Fig 1B; defaults tuned visually
    # against real recordings)
    base_to_tip_static_curl: float = 0.95            # tip orientation in recovery, in units of pi
    base_to_tip_pulse_sharpness: float = 1.2         # 1.0 = pure sine; >1 sharpens peaks; <1 rounds them
    base_to_tip_distal_concentration: float = 0.5    # s_norm envelope exponent; <1 -> mid-flagellum-weighted

    # Tip-to-base envelope shape: env(s) = sin(pi*s_norm) ** exponent.
    # The sin envelope keeps the tip roughly fixed in tangent angle while
    # the middle of the flagellum sweeps -- which is what you actually see
    # in Leishmania videos (the tip "points" in roughly the same direction
    # throughout the beat). The exponent reshapes the central peak:
    # 1.0 = bare sin envelope (default; broad, smooth peak);
    # 2.0 = sin^2 (sharper, more concentrated mid-peak);
    # 0.5 = broader / nearer-rectangular (large amplitude over a wider
    # fraction of the flagellum -- closer to a saturated wave).
    tip_to_base_envelope_exponent: float = 1.0

    # Base-to-tip static (mean) curvature shape: power-law applied to s_norm.
    # 1.0 = linear ramp (default, gives a smooth circular-arc-ish baseline);
    # 2.0 = bend pushed distally (cane / hook silhouette);
    # 3.0 = even stronger distal hinge.
    base_to_tip_static_curl_shape: float = 1.0

    # Base-to-tip temporal asymmetry: anharmonic phase distortion that yields
    # a fast power stroke and slow recovery (real ciliary beats are not
    # strictly time-reversible). 0.0 = pure sinusoid; 0.10 = mild asymmetry
    # (default; matches typical recordings); 0.35-0.50 = aggressive flick.
    base_to_tip_temporal_asymmetry: float = 0.10

    # Base-to-tip wave propagation extent: how far along the flagellum the
    # dynamic wave keeps near-full amplitude. Specifically, the s_norm at
    # which the wave drops to ~88% of its peak; past that it falls off
    # smoothly. The Wang/Wheeler 2020 kymograph (Fig 2G) clearly shows the
    # base-to-tip wave reaching the tip — this is what produces the visible
    # tip sweep during the power stroke.
    # 1.0 = wave reaches tip near-fully (DEFAULT, matches Fig 2G);
    # 0.7 = wave dies before tip (the partially-propagating phenotype
    # Wheeler 2020 mentions in passing);
    # >1.2 = wave amplitude does not decay at all on the flagellum.
    base_to_tip_propagation_extent: float = 1.0

    # Stochastic mode switching (Wheeler 2020)
    mode_switch_rate: float = 0.0
    mode_schedule: Optional[List[Tuple[float, str]]] = None
    mode_transition_duration: float = 0.20

    # Motion: tip_to_base = forward swim; base_to_tip = turn in place
    swim_speed: float = 0.0           # um/s
    angular_velocity: float = 0.0     # rad/s
    tip_to_base_swim_speed: Optional[float] = None       # um/s
    tip_to_base_angular_velocity: Optional[float] = None # rad/s
    base_to_tip_swim_speed: Optional[float] = None       # um/s
    base_to_tip_angular_velocity: Optional[float] = None # rad/s

    # Pulse envelope for base_to_tip rotation (fraction of beat cycle)
    rotation_rise_tau_cycles: float = 0.0
    rotation_decay_tau_cycles: float = 0.0
    accumulated_rotation_velocity: float = 0.0  # rad/s, integrator state

    n_flagellum_keypoints: int = 6  # interior; total = Head + Base + N + Tip


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _coalesce(*values):
    """Return the first value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


# ----------------------------------------------------------------------------
# Micrometre -> pixel conversion
# ----------------------------------------------------------------------------
# All spatial fields on ParasiteParams and the *_um sigmas on OpticsParams are
# in micrometres. They get multiplied by `pixels_per_um(optics)` at render
# time to produce output-image pixel values. center_x / center_y are an
# exception — they are image-pixel coordinates, not µm.

def pixels_per_um(optics: "OpticsParams") -> float:
    """Output-image pixels per micrometre. Higher mag -> larger value."""
    return 1.0 / max(optics.pixel_size_um, 1e-6)


def _parasite_lengths_to_px(p: "ParasiteParams",
                            optics: "OpticsParams") -> "ParasiteParams":
    """Return a copy of p with all µm-spatial fields converted to pixels.

    Does NOT touch center_x/center_y (already in image px), nor frequencies/
    angular velocities (per-second, no length factor)."""
    s = pixels_per_um(optics)
    if s == 1.0:
        return p

    def _maybe(v):
        return None if v is None else v * s

    return replace(
        p,
        body_length=p.body_length * s,
        body_width=p.body_width * s,
        flagellum_length=p.flagellum_length * s,
        flagellum_width=p.flagellum_width * s,
        beat_wavelength=p.beat_wavelength * s,
        beat_amplitude_max=p.beat_amplitude_max * s,
        tip_to_base_wavelength=_maybe(p.tip_to_base_wavelength),
        tip_to_base_amplitude=_maybe(p.tip_to_base_amplitude),
        base_to_tip_wavelength=_maybe(p.base_to_tip_wavelength),
        base_to_tip_amplitude=_maybe(p.base_to_tip_amplitude),
    )


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def body_polygon(p: ParasiteParams, n_points: int = 120) -> np.ndarray:
    """Asymmetric tear-drop silhouette: rounded posterior, smaller anterior tip."""
    s = np.linspace(-1, 1, n_points)
    peak_s = p.body_peak_position
    extension = 0.05  # virtual extension past endpoints -> rounded tips
    r_post = (peak_s + 1.0) + extension
    r_ant = (1.0 - peak_s) + extension

    rel_post = (s - peak_s) / r_post
    rel_ant = (s - peak_s) / r_ant
    prof_post = np.sqrt(np.clip(1.0 - rel_post ** 2, 0, 1))
    prof_ant = np.sqrt(np.clip(1.0 - rel_ant ** 2, 0, 1))
    profile = np.where(s < peak_s, prof_post, prof_ant)

    half_width = (p.body_width / 2) * profile
    along_axis = s * (p.body_length / 2)
    centerline_offset = p.body_curvature * (p.body_length / 2) * (1.0 - s ** 2)

    upper = np.column_stack([along_axis, half_width + centerline_offset])
    lower = np.column_stack([along_axis[::-1],
                             -half_width[::-1] + centerline_offset[::-1]])
    poly_local = np.vstack([upper, lower])

    c, sn = np.cos(p.body_orientation), np.sin(p.body_orientation)
    R = np.array([[c, -sn], [sn, c]])
    return poly_local @ R.T + np.array([p.center_x, p.center_y])


def _theta_for_mode(p: ParasiteParams, t: float, mode: str,
                    n_points: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Tangent angle theta(s, t) along the flagellum for a given beat mode.

    Implements the Wang/Wheeler 2020 static + dynamic decomposition
    (J Cell Sci 133:jcs246637), with refinements for the four issues in
    the original implementation:

      1. Tip-to-base envelope was sin(pi*s_norm), which pinned the FREE
         tip and killed amplitude there. Replaced with a monotonically
         growing envelope: env(s) = s_norm ** tip_to_base_envelope_exponent
         (zero at the clamped base, max at the free tip).

      2. Base-to-tip static curvature was linear in s_norm, giving a
         circular arc. Replaced with a power law s_norm**cs (cs ~ 2)
         that concentrates the bend distally, matching the cane/hook
         silhouette of the recovery-stroke shape.

      3. Base-to-tip dynamic part was time-symmetric (pulse_sharpness
         only sharpens peaks symmetrically). Real ciliary beats have a
         fast power stroke and slow recovery -- broken time-reversal.
         Implemented via anharmonic time:
             phase_t = omega*t + ta * sin(omega*t)
         (same trick as the eccentric anomaly in Kepler orbits).

      4. Base-to-tip wave was forced to grow all the way to the tip
         (s_norm**dc envelope only). Wheeler 2020 explicitly notes the
         base-to-tip wave often dies before reaching the tip. Added a
         smooth tanh falloff past base_to_tip_propagation_extent.
    """
    s_norm = np.linspace(0.0, 1.0, n_points)
    L = p.flagellum_length
    s_real = s_norm * L

    if mode == "static":
        return s_norm, np.zeros_like(s_norm)

    if mode == "tip_to_base":
        freq = _coalesce(p.tip_to_base_frequency, p.beat_frequency)
        wavelength = _coalesce(p.tip_to_base_wavelength, p.beat_wavelength)
        amplitude = _coalesce(p.tip_to_base_amplitude, p.beat_amplitude_max)
    elif mode == "base_to_tip":
        freq = _coalesce(p.base_to_tip_frequency, p.beat_frequency)
        wavelength = _coalesce(p.base_to_tip_wavelength, p.beat_wavelength)
        amplitude = _coalesce(p.base_to_tip_amplitude, p.beat_amplitude_max)
    else:
        raise ValueError(f"Unknown beat_mode: {mode}")

    omega = 2 * np.pi * freq
    k = 2 * np.pi / wavelength

    if mode == "tip_to_base":
        # Envelope: sin(pi*s_norm) ** exponent.
        # The bare sin envelope (exponent=1) pins both ends -- base clamped
        # to the cell body, and the tip kept roughly stationary in tangent
        # angle. This matches what you actually see in Leishmania videos:
        # the tip "points" in roughly the same direction throughout the
        # beat while the middle of the flagellum sweeps. The exponent
        # reshapes the central peak: 1.0 = original sin (default),
        # 2.0 = sin^2 (sharper, more concentrated mid-peak),
        # 0.5 = broader / more rectangular (waves stay near peak amplitude
        # over a longer fraction of the flagellum).
        theta_amp = amplitude * k
        env = np.sin(np.pi * s_norm) ** p.tip_to_base_envelope_exponent
        theta = theta_amp * env * np.sin(k * (s_real - L) + omega * t + p.beat_phase)
        return s_norm, theta

    # ---- base_to_tip beat: static + dynamic decomposition ----
    sc = p.base_to_tip_static_curl
    sh = p.base_to_tip_pulse_sharpness
    dc = p.base_to_tip_distal_concentration
    cs = p.base_to_tip_static_curl_shape
    ta = p.base_to_tip_temporal_asymmetry
    pe = p.base_to_tip_propagation_extent

    # FIX 2: distally-concentrated static curl (cane / hook silhouette).
    # cs = 1.0 -> linear (circular arc, old behaviour).
    # cs = 2.0 -> bend pushed toward the tip (default, matches Wheeler Fig 1B).
    theta_static = (np.pi * sc) * (s_norm ** cs)

    # FIX 3: anharmonic time -> fast power stroke, slow recovery.
    # ta = 0 -> pure sinusoid in t (old behaviour).
    phase_t = omega * t + ta * np.sin(omega * t)
    phase = k * s_real - phase_t + p.beat_phase

    s_phase = np.sin(phase)
    if sh == 1.0:
        osc = s_phase
    else:
        osc = np.sign(s_phase) * (np.abs(s_phase) ** sh)

    # FIX 4: dynamic envelope rises from base (s^dc), then smoothly decays
    # only PAST the propagation extent. With pe=1.0 (default) the decay
    # onset is at the tip, so the wave reaches the tip with ~88% amplitude —
    # this is what makes the power-stroke tip-sweep visible. With pe<1.0
    # the wave dies before the tip (the partially-propagating phenotype).
    rise = s_norm ** dc
    falloff_width = 0.10
    decay = 0.5 * (1.0 - np.tanh((s_norm - pe - falloff_width) / falloff_width))
    env = rise * decay

    theta_amp_dyn = amplitude * k
    theta_dynamic = theta_amp_dyn * env * osc
    return s_norm, theta_static + theta_dynamic


def _get_active_mode_blend(p: ParasiteParams, t: float):
    """Return (mode_a, mode_b, blend) at time t. Schedule must be sorted."""
    if not p.mode_schedule:
        return p.beat_mode, p.beat_mode, 0.0

    td = p.mode_transition_duration
    half = td / 2.0
    intervals = [(0.0, p.beat_mode)] + p.mode_schedule

    current_idx = 0
    for i, (ts, _) in enumerate(intervals):
        if ts <= t:
            current_idx = i
        else:
            break
    current_mode = intervals[current_idx][1]

    # Approaching the next switch?
    if current_idx + 1 < len(intervals):
        next_ts, next_mode = intervals[current_idx + 1]
        if next_ts - t < half:
            raw = ((t - next_ts) + half) / td  # in [0, 0.5]
            blend = 0.5 - 0.5 * np.cos(np.pi * raw)
            return current_mode, next_mode, blend

    # Just past the previous switch?
    if current_idx > 0:
        prev_ts = intervals[current_idx][0]
        prev_mode = intervals[current_idx - 1][1]
        if t - prev_ts < half:
            raw = ((t - prev_ts) + half) / td  # in [0.5, 1]
            blend = 0.5 - 0.5 * np.cos(np.pi * raw)
            return prev_mode, current_mode, blend

    return current_mode, current_mode, 0.0


def beat_tangent_angle(p: ParasiteParams, t: float, n_points: int = 80):
    """Tangent angle along flagellum, blended across mode transitions."""
    mode_a, mode_b, blend = _get_active_mode_blend(p, t)
    s_norm, theta_a = _theta_for_mode(p, t, mode_a, n_points)
    if blend == 0.0 or mode_a == mode_b:
        return s_norm, theta_a
    _, theta_b = _theta_for_mode(p, t, mode_b, n_points)
    return s_norm, (1 - blend) * theta_a + blend * theta_b


beat_lateral_displacement = beat_tangent_angle  # legacy alias


def generate_mode_schedule(p: ParasiteParams, duration: float,
                           rng: np.random.Generator) -> list:
    """Stochastic mode-switch schedule. Returns sorted list of (t, new_mode)."""
    if p.mode_switch_rate <= 0:
        return []
    schedule = []
    current_mode = p.beat_mode
    t = 0.0
    max_switches = int(duration * p.mode_switch_rate * 5) + 5
    for _ in range(max_switches):
        t += rng.exponential(1.0 / p.mode_switch_rate)
        if t >= duration:
            break
        new_mode = "base_to_tip" if current_mode == "tip_to_base" else "tip_to_base"
        schedule.append((float(t), new_mode))
        current_mode = new_mode
    return schedule


def flagellum_curve(p: ParasiteParams, t: float,
                    n_points: int = 80) -> np.ndarray:
    """Flagellum centerline in image coords, base -> tip."""
    s_norm, theta = beat_tangent_angle(p, t, n_points=n_points)
    L = p.flagellum_length
    ds = L / (n_points - 1) if n_points > 1 else L

    dx = np.cos(theta) * ds
    dy = np.sin(theta) * ds
    x_local = np.concatenate([[0.0], np.cumsum(dx[:-1])])
    y_local = np.concatenate([[0.0], np.cumsum(dy[:-1])])
    local = np.column_stack([x_local, y_local]).astype(np.float32)

    c, sn = np.cos(p.body_orientation), np.sin(p.body_orientation)
    R = np.array([[c, -sn], [sn, c]])
    world = local @ R.T

    anterior_offset = (p.body_length / 2) * np.array([c, sn])
    return world + np.array([p.center_x, p.center_y]) + anterior_offset


# ----------------------------------------------------------------------------
# Rendering (per-parasite, in a local bounding-box tile)
# ----------------------------------------------------------------------------

def _make_keypoints(p: ParasiteParams, flag: np.ndarray) -> dict:
    posterior = np.array([p.center_x, p.center_y]) - (p.body_length / 2) * np.array([
        np.cos(p.body_orientation), np.sin(p.body_orientation),
    ])
    interior_idx = np.linspace(0, len(flag) - 1, p.n_flagellum_keypoints + 2)[1:-1]
    interior = [flag[int(round(i))] for i in interior_idx]

    kp = {"Head": tuple(posterior), "Base": tuple(flag[0])}
    for k, pt in enumerate(interior, start=1):
        kp[f"Flag{k}"] = tuple(pt)
    kp["Tip"] = tuple(flag[-1])
    return kp


def render_parasite_phase(p: ParasiteParams, t: float, image_shape: tuple,
                          *,
                          optics: Optional[OpticsParams] = None,
                          halo_margin: int = 20,
                          edge_smooth_sigma_px: Optional[float] = None):
    """
    Render one parasite into a local bbox tile.

    Returns (tile, (y0, x0), keypoints). `tile` is None if the parasite is
    fully outside the image.

    If `optics` is provided, parasite spatial dimensions and edge smoothing
    are scaled by ``magnification_scale(optics)`` so the same parasite spec
    renders at different pixel sizes correctly. If `edge_smooth_sigma_px` is
    given explicitly, it overrides the optics-derived value.

    Body phase comes from a 3D-thickness model (rotational symmetry around
    the long axis), modulated by a density envelope along the body axis.
    Flagellum is rasterized as an antialiased thick polyline.
    """
    if optics is not None:
        p = _parasite_lengths_to_px(p, optics)
        if edge_smooth_sigma_px is None:
            edge_smooth_sigma_px = optics.body_edge_smooth_sigma_um * pixels_per_um(optics)
    elif edge_smooth_sigma_px is None:
        edge_smooth_sigma_px = 2.0

    H, W = image_shape

    body = body_polygon(p)
    flag = flagellum_curve(p, t, n_points=80)
    keypoints = _make_keypoints(p, flag)

    pts = np.vstack([body, flag])
    x0 = max(0, int(np.floor(pts[:, 0].min())) - halo_margin)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - halo_margin)
    x1 = min(W, int(np.ceil(pts[:, 0].max())) + halo_margin)
    y1 = min(H, int(np.ceil(pts[:, 1].max())) + halo_margin)
    th, tw = y1 - y0, x1 - x0
    if th <= 0 or tw <= 0:
        return None, (0, 0), keypoints

    offset = np.array([x0, y0], dtype=np.float32)
    body_local = np.round(body - offset).astype(np.int32)
    flag_local = (flag - offset).astype(np.float32)

    # Body mask
    body_mask = np.zeros((th, tw), dtype=np.uint8)
    cv2.fillPoly(body_mask, [body_local], 1)

    # Ellipsoidal thickness from interior distance transform:
    # thickness(d_norm) = sqrt(1 - (1 - d_norm)^2), peaks at the centerline.
    dist = distance_transform_edt(body_mask).astype(np.float32)
    dmax = dist.max()
    dist_n = dist / dmax if dmax > 0 else dist
    thickness = np.sqrt(np.clip(2 * dist_n - dist_n ** 2, 0, 1))

    # Density envelope along body axis (denser near widest point).
    yy, xx = np.indices((th, tw)).astype(np.float32)
    cx_l = p.center_x - x0
    cy_l = p.center_y - y0
    ax, ay = np.cos(p.body_orientation), np.sin(p.body_orientation)
    along = ((xx - cx_l) * ax + (yy - cy_l) * ay) / (p.body_length / 2)
    density_env = 0.55 + 0.45 * np.exp(
        -((along - p.body_peak_position) ** 2) / (2 * 0.40 ** 2))

    body_phase = (thickness * p.body_phase_shift * density_env).astype(np.float32)
    body_phase = cv2.GaussianBlur(body_phase, (0, 0),
                                  sigmaX=max(edge_smooth_sigma_px, 1e-3))

    # Flagellum: rasterize into its own buffer, then max-compose.
    flag_buf = np.zeros((th, tw), dtype=np.float32)
    flag_int = np.round(flag_local).astype(np.int32)
    cv2.polylines(flag_buf, [flag_int], isClosed=False,
                  color=float(p.flagellum_phase_shift),
                  thickness=max(1, int(round(p.flagellum_width))),
                  lineType=cv2.LINE_AA)

    np.maximum(body_phase, flag_buf, out=body_phase)
    return body_phase, (y0, x0), keypoints


# ----------------------------------------------------------------------------
# Phase contrast optics + camera noise
# ----------------------------------------------------------------------------

def simulate_phase_contrast(phase_map: np.ndarray, optics: OpticsParams) -> np.ndarray:
    """Hybrid PhC: exp darkening + DoG halo + softplus shading-off.

    PSF and halo lowpass are in micrometres; converted to pixels using
    optics.pixel_size_um.
    """
    s = pixels_per_um(optics)
    psf_sigma = optics.psf_sigma_um * s
    halo_lowpass = optics.halo_lowpass_sigma_um * s

    phi = cv2.GaussianBlur(phase_map, (0, 0), sigmaX=psf_sigma)

    # Exp darkening: bodies tend to dark gray, edges/background near 1.0.
    intensity = np.exp(-optics.intensity_gain * phi)

    # Halo from DoG:  -min(inner - outer, 0)  ==  max(outer - inner, 0)
    inner = cv2.GaussianBlur(phi, (0, 0), sigmaX=psf_sigma * 1.2)
    outer = cv2.GaussianBlur(phi, (0, 0), sigmaX=halo_lowpass)
    halo_signal = np.maximum(outer - inner, 0)
    intensity += optics.halo_strength * halo_signal

    # Shading-off via numerically-stable softplus.
    sharpness = 4.0
    sx = sharpness * (phi - optics.shadeoff_threshold)
    softplus_excess = np.where(
        sx > 30,
        phi - optics.shadeoff_threshold,
        np.log1p(np.exp(np.clip(sx, -50, 30))) / sharpness,
    )
    intensity += optics.shadeoff_strength * softplus_excess

    return intensity


def add_camera_noise(intensity: np.ndarray, noise: CameraNoiseParams,
                     rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise + Gaussian read noise + dark offset."""
    photons = np.clip(intensity, 0, None) * noise.full_well_photons
    out = rng.poisson(photons).astype(np.float32) / noise.full_well_photons
    out += rng.normal(0, noise.read_noise_e / noise.full_well_photons,
                      intensity.shape).astype(np.float32)
    out += noise.dark_offset
    return out

# ----------------------------------------------------------------------------
# Fast variants (for real-time / preview use; visually ~indistinguishable)
# ----------------------------------------------------------------------------

def simulate_phase_contrast_fast(phase_map: np.ndarray, optics: OpticsParams,
                                 halo_downsample: int = 4) -> np.ndarray:
    """
    Same model as simulate_phase_contrast, faster:
      - Drops the redundant 'inner' blur (uses phi directly).
      - Computes the large-sigma halo blur at downsampled resolution.

    For typical defaults (halo_lowpass_sigma_um ~= 3.9, ds=4) the halo
    reconstruction error is negligible because we only need the low
    frequencies; the band the upsample throws away is below the
    Nyquist of the downsampled image anyway.

    PSF/halo kernels are in micrometres; converted to pixels using
    optics.pixel_size_um.
    """
    s = pixels_per_um(optics)
    psf_sigma = optics.psf_sigma_um * s
    halo_lowpass = optics.halo_lowpass_sigma_um * s

    phi = cv2.GaussianBlur(phase_map, (0, 0), sigmaX=psf_sigma)
    intensity = np.exp(-optics.intensity_gain * phi)

    # Halo lowpass at coarse resolution
    H, W = phase_map.shape
    ds = halo_downsample
    Hs, Ws = max(H // ds, 4), max(W // ds, 4)
    small = cv2.resize(phase_map, (Ws, Hs), interpolation=cv2.INTER_AREA)
    small_blurred = cv2.GaussianBlur(
        small, (0, 0), sigmaX=halo_lowpass / ds)
    outer = cv2.resize(small_blurred, (W, H), interpolation=cv2.INTER_LINEAR)

    # Use phi (already PSF-blurred) as the inner term — saves one full blur.
    halo_signal = np.maximum(outer - phi, 0)
    intensity += optics.halo_strength * halo_signal

    sharpness = 4.0
    sx = sharpness * (phi - optics.shadeoff_threshold)
    softplus_excess = np.where(
        sx > 30, phi - optics.shadeoff_threshold,
        np.log1p(np.exp(np.clip(sx, -50, 30))) / sharpness,
    )
    intensity += optics.shadeoff_strength * softplus_excess
    return intensity


def add_camera_noise_fast(intensity: np.ndarray, noise: CameraNoiseParams,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Gaussian approximation to Poisson noise.

    For mean photon count N >> 1, Poisson(N) ~ N(N, sqrt(N)). With the
    default full_well=5000 and bg_intensity=0.75 we're at ~3750 photons
    per pixel — the approximation is exact to 3+ decimal places, and
    one rng.normal call is much cheaper than rng.poisson per-pixel.
    """
    photons = np.clip(intensity, 0, None) * noise.full_well_photons
    z = rng.standard_normal(intensity.shape, dtype=np.float32)
    out = (photons + np.sqrt(photons) * z) / noise.full_well_photons
    out += rng.standard_normal(intensity.shape, dtype=np.float32) * (
        noise.read_noise_e / noise.full_well_photons)
    out += noise.dark_offset
    return out



# ----------------------------------------------------------------------------
# Background
# ----------------------------------------------------------------------------

def synthetic_background(shape: tuple, rng: np.random.Generator,
                         intensity: float = 0.75) -> np.ndarray:
    """
    PhC background:
      - Vignetting (radial darkening)
      - Slow illumination tilt
      - Multi-scale texture (low-freq blotches, mid grime, fine grain)
      - Small dust particles
      - Out-of-focus debris rings
      - Out-of-focus blob "ghost" cells
    """
    H, W = shape
    bg = np.full((H, W), intensity, dtype=np.float32)

    yy, xx = np.meshgrid(np.linspace(-1, 1, H),
                         np.linspace(-1, 1, W), indexing="ij")

    # Vignetting (radial falloff toward corners)
    r2 = (xx ** 2 + yy ** 2).astype(np.float32)
    bg *= 1.0 - rng.uniform(0.06, 0.14) * r2

    # Slow illumination tilt
    bg += (rng.uniform(0.015, 0.04) *
           (rng.uniform(-1, 1) * xx + rng.uniform(-1, 1) * yy)).astype(np.float32)

    # Multi-scale texture
    for scale, amp in [(60, rng.uniform(0.012, 0.022)),
                       (18, rng.uniform(0.006, 0.012)),
                       (6,  rng.uniform(0.004, 0.008))]:
        n_h = max(H // scale, 4)
        n_w = max(W // scale, 4)
        coarse = rng.normal(0, 1, (n_h, n_w)).astype(np.float32)
        texture = zoom(coarse, (H / n_h, W / n_w), order=3)[:H, :W]
        s = texture.std()
        if s > 0:
            texture /= s
        bg += amp * 0.15 * texture.astype(np.float32)

    # Small dust particles — rate, size, and strength all vary widely per frame
    # so different "sessions" have visibly different field-of-view cleanliness.
    n_dust = int(rng.poisson(rng.uniform(8, 30)))
    for _ in range(n_dust):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        radius = rng.uniform(0.8, 6.0)        # was (1.5, 4.0)
        strength = rng.uniform(0.05, 0.85)    # was (0.05, 0.6)
        rr = int(np.ceil(4 * radius))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        r2_d = (dx_ ** 2 + dy_ ** 2) / (radius ** 2)
        blob = (-strength * np.exp(-r2_d)
                + 0.25 * strength * np.exp(-r2_d / 5))
        bg[y_lo:y_hi, x_lo:x_hi] += blob.astype(np.float32)

    # Out-of-focus debris rings (concentric Airy-like patterns)
    n_debris = int(rng.poisson(rng.uniform(2, 12)))   # was poisson(6)
    for _ in range(n_debris):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        radius = rng.uniform(8, 22)
        strength = rng.uniform(0.001, 0.05)
        ring_phase = rng.uniform(0, 2 * np.pi)
        rr = int(np.ceil(2.5 * radius))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        r = np.sqrt(dx_ ** 2 + dy_ ** 2) / radius
        ring = strength * np.cos(2 * np.pi * r + ring_phase) * np.exp(-r * 0.6)
        bg[y_lo:y_hi, x_lo:x_hi] += np.where(r < 2.5, ring, 0).astype(np.float32)

    # Out-of-focus "ghost" cells: faint, blurry, off the focal plane
    n_blobs = int(rng.poisson(rng.uniform(0.3, 3.0)))   # was poisson(1.0)
    for _ in range(n_blobs):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        radius = rng.uniform(20, 45)
        strength = rng.uniform(0.02, 0.05)
        rr = int(np.ceil(2.5 * radius))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        r2_b = (dx_ ** 2 + dy_ ** 2) / (radius ** 2)
        blob = (-strength * np.exp(-r2_b * 1.5)
                + 0.4 * strength * np.exp(-r2_b * 0.5))
        bg[y_lo:y_hi, x_lo:x_hi] += blob.astype(np.float32)

    return bg


# ----------------------------------------------------------------------------
# Scene compositing
# ----------------------------------------------------------------------------

def render_scene(parasites, t: float, image_shape: tuple,
                 optics: OpticsParams, noise: CameraNoiseParams,
                 background: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None,
                 fast: bool = False,
                 occlusion_aware_labels: bool = True,
                 occlusion_patch_radius: int = 1,
                 occlusion_majority_threshold: float = 0.5):
    """
    Render multiple parasites at time t. Returns (image, [keypoints]).

    Set ``fast=True`` for the real-time variant: ~1.5–2x faster, visually
    indistinguishable. Uses :func:`simulate_phase_contrast_fast` and
    :func:`add_camera_noise_fast` (Gaussian-approximated Poisson noise).

    Occlusion handling
    ------------------
    When ``occlusion_aware_labels=True`` (default), keypoints belonging to
    cells whose phase contribution is dominated by other cells around the
    keypoint are removed from the returned keypoint dict (i.e. marked as
    occluded — the downstream label-builder will write NaN coords for them,
    and SLEAP will treat the keypoint as not-visible).

    The check uses a winner-map: at every pixel, which cell index had the
    largest phase shift contribution? A keypoint is then judged by:

      1. **Pixel fast path** — if THIS cell wins at the exact keypoint
         pixel, the keypoint is visible regardless of neighbours. This
         protects thin-flagellum keypoints near a larger cell's body: the
         flagellum still wins at its own thin pixels.
      2. **Patch vote** — if the cell loses at the keypoint pixel, fall
         back to a ``(2r+1)x(2r+1)`` patch check. Visible only if the cell
         owns at least ``occlusion_majority_threshold`` of the patch's
         non-background pixels. Background pixels (no cell contributed any
         phase) are neutral, not occluders.

    Set ``occlusion_aware_labels=False`` to recover the previous (incorrect
    in dense scenes) behaviour where every cell's keypoints are labelled
    regardless of whether the cell is actually visible.
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = image_shape

    phase_total = np.zeros((H, W), dtype=np.float32)
    # Per-pixel index of the cell currently winning the max-composite.
    # -1 = background (no cell has contributed any phase here yet).
    winner_map = np.full((H, W), -1, dtype=np.int32) if occlusion_aware_labels else None

    all_keypoints = []
    for i, p in enumerate(parasites):
        tile, (y0, x0), kp = render_parasite_phase(
            p, t, image_shape, optics=optics,
        )
        all_keypoints.append(kp)
        if tile is None:
            continue
        th, tw = tile.shape
        region = phase_total[y0:y0 + th, x0:x0 + tw]
        if occlusion_aware_labels:
            # Where does THIS cell strictly beat the running max? Those are
            # the pixels we'll claim as ours in the winner_map.
            beats = tile > region
            region[beats] = tile[beats]
            winner_map[y0:y0 + th, x0:x0 + tw][beats] = i
        else:
            np.maximum(region, tile, out=region)

    if occlusion_aware_labels:
        # Per-cell, per-keypoint visibility check.
        #
        # Two-step rule, ordered most-lenient first:
        #   (1) Fast path: if THIS cell is the winner at the exact keypoint
        #       pixel, the cell is visibly present at that pixel in the
        #       rendered image. Mark visible regardless of neighbours.
        #       This is what protects thin-flagellum keypoints near a
        #       larger neighbour's body: the flagellum wins at its own
        #       pixels even though the surrounding 3x3 is mostly other.
        #
        #   (2) Patch vote: if the cell loses at the keypoint pixel, check
        #       a (2r+1)x(2r+1) patch — visible only if the cell owns at
        #       least ``occlusion_majority_threshold`` of the patch's
        #       non-background pixels. Background pixels (no cell drew
        #       anything there) are neutral, not counted as occluders.
        r = max(0, int(occlusion_patch_radius))
        thr = float(occlusion_majority_threshold)
        for i, kp in enumerate(all_keypoints):
            occluded = []
            for name, (x, y) in kp.items():
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < W and 0 <= iy < H):
                    occluded.append(name)
                    continue
                # (1) Pixel-level fast path
                if winner_map[iy, ix] == i:
                    continue
                # (2) Patch vote
                ya, yb = max(0, iy - r), min(H, iy + r + 1)
                xa, xb = max(0, ix - r), min(W, ix + r + 1)
                patch = winner_map[ya:yb, xa:xb]
                own = int(np.sum(patch == i))
                others = int(np.sum((patch != i) & (patch != -1)))
                non_bg = own + others
                if non_bg > 0 and own < non_bg * thr:
                    occluded.append(name)
            for name in occluded:
                kp.pop(name, None)

    if fast:
        intensity = simulate_phase_contrast_fast(phase_total, optics)
    else:
        intensity = simulate_phase_contrast(phase_total, optics)

    if background is None:
        background = synthetic_background((H, W), rng, intensity=noise.bg_intensity)
    composite = background * intensity

    if fast:
        image = add_camera_noise_fast(composite, noise, rng)
    else:
        image = add_camera_noise(composite, noise, rng)
    return np.clip(image, 0, 1), all_keypoints


# ----------------------------------------------------------------------------
# Motion
# ----------------------------------------------------------------------------

def _motion_for_mode(p: ParasiteParams, mode: str) -> Tuple[float, float]:
    if mode == "tip_to_base":
        return (
            _coalesce(p.tip_to_base_swim_speed, p.swim_speed),
            _coalesce(p.tip_to_base_angular_velocity, p.angular_velocity),
        )
    if mode == "base_to_tip":
        return (
            _coalesce(p.base_to_tip_swim_speed, p.swim_speed),
            _coalesce(p.base_to_tip_angular_velocity, p.angular_velocity),
        )
    return 0.0, 0.0


def _motion_pulse_factor(p: ParasiteParams, t: float, mode: str) -> float:
    """
    Pulsed force-generation factor over a beat cycle.

    tip_to_base: continuous (1.0). base_to_tip: bursts during the power
    stroke, ~zero during recovery; cycle-average is normalized to 1.0 so
    configured swim/angular means are preserved on average. Optional
    rise/decay envelope smooths acceleration.
    """
    if mode != "base_to_tip":
        return 1.0
    freq = _coalesce(p.base_to_tip_frequency, p.beat_frequency)
    if freq <= 0:
        return 1.0

    cycle = ((freq * t) + p.beat_phase / (2 * np.pi)) % 1.0

    peak_phase = 0.15
    half_width = 0.07
    d = cycle - peak_phase
    if d > 0.5:
        d -= 1.0
    elif d < -0.5:
        d += 1.0
    if abs(d) > half_width:
        return 0.0

    base_pulse = 0.5 * (1.0 + np.cos(np.pi * d / half_width)) / half_width

    rise_tau = p.rotation_rise_tau_cycles
    decay_tau = p.rotation_decay_tau_cycles
    if rise_tau <= 0 and decay_tau <= 0:
        return base_pulse

    pulse_local = (d + half_width) / (2 * half_width)  # in [0, 1]
    if pulse_local < 0.5 and rise_tau > 0:
        env = 1.0 - np.exp(-(pulse_local / 0.5) / rise_tau)
    elif pulse_local >= 0.5 and decay_tau > 0:
        env = np.exp(-((pulse_local - 0.5) / 0.5) / decay_tau)
    else:
        env = 1.0
    return base_pulse * env


def get_active_motion(p: ParasiteParams, t: float, dt: float) -> Tuple[float, float]:
    """
    (swim_speed, angular_velocity) at time t, with mode-blend handling
    and pulsed/accumulated rotation for base_to_tip.

    Mutates `p.accumulated_rotation_velocity`.
    """
    mode_a, mode_b, blend = _get_active_mode_blend(p, t)

    def motion_for(mode):
        speed_mean, omega_mean = _motion_for_mode(p, mode)
        if mode != "base_to_tip":
            return speed_mean, omega_mean

        # Impulse-and-decay rotation
        pulse = _motion_pulse_factor(p, t, mode)
        impulse = omega_mean * pulse * 0.5
        decay_rate = 2.0  # /s

        p.accumulated_rotation_velocity *= max(0.0, 1.0 - decay_rate * dt)
        p.accumulated_rotation_velocity += impulse * dt
        max_rot = abs(omega_mean) * 3.0
        p.accumulated_rotation_velocity = float(np.clip(
            p.accumulated_rotation_velocity, -max_rot, max_rot))
        return speed_mean, p.accumulated_rotation_velocity

    sa, oa = motion_for(mode_a)
    if blend == 0.0 or mode_a == mode_b:
        return sa, oa
    sb, ob = motion_for(mode_b)
    return (1 - blend) * sa + blend * sb, (1 - blend) * oa + blend * ob


def advance_parasites(parasites, dt: float, image_shape: tuple,
                      periodic: bool = True, t: float = 0.0,
                      optics: Optional[OpticsParams] = None) -> None:
    """Advance positions/orientations by dt (in-place).

    swim_speed is in µm/s. If `optics` is provided, it gets converted to
    px/s via ``pixels_per_um(optics)`` so center_x/center_y (image px)
    advance correctly. Without optics, swim_speed is treated as if already
    in px/s (legacy behavior)."""
    H, W = image_shape
    s = pixels_per_um(optics) if optics is not None else 1.0
    for p in parasites:
        speed, omega = get_active_motion(p, t, dt)
        speed *= s
        p.center_x += speed * np.cos(p.body_orientation) * dt
        p.center_y += speed * np.sin(p.body_orientation) * dt
        p.body_orientation += omega * dt
        if periodic:
            p.center_x %= W
            p.center_y %= H


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------

def sample_random_parasite(rng: np.random.Generator,
                           image_shape: tuple,
                           t: float = 0.0,
                           force_beat_mode: Optional[str] = None,
                           swim_speed_range: tuple = (0.0, 50.0),
                           ) -> ParasiteParams:
    """Draw realistic random parameters for one parasite. Spatial values
    are in micrometres; positions are in image pixels.

    Sampling distribution covers wild-type Leishmania promastigotes plus
    common mutant / non-WT phenotypes the model is likely to encounter:
      - aggressive curls (high base_to_tip_static_curl)
      - altered beat frequencies (slower or faster than WT)
      - paralysed / immotile cells (~5%, beat_mode='static')
    Body and flagellum sizes span procyclic (~10-12 um body, ~equal flag)
    through metacyclic (~8-10 um body, ~20 um flag) and intermediate forms.
    """
    H, W = image_shape
    margin = 60
    cx = rng.uniform(margin, W - margin)
    cy = rng.uniform(margin, H - margin)
    theta = rng.uniform(0, 2 * np.pi)

    # Per-mode beat parameters. Frequencies widened to cover slower/faster
    # mutants in addition to the WT 20-25 Hz / ~5 Hz means.
    ttb_freq = rng.uniform(15, 28)
    ttb_wavelen = rng.uniform(5.0, 10.0)     # um
    ttb_amp = rng.uniform(0.4, 1.0)          # um
    btt_freq = rng.uniform(2, 8)
    btt_wavelen = rng.uniform(6.0, 10.0)     # um
    btt_amp = rng.uniform(0.8, 2.0)          # um

    # Base-to-tip shape: ranges centred on visually-tuned defaults, widened
    # to cover aggressive-curl mutants and varied phenotypes.
    btt_static_curl = rng.uniform(0.65, 1.45)
    btt_pulse_sharpness = rng.uniform(1.0, 2.0)
    btt_distal_concentration = rng.uniform(0.35, 1.30)

    # Refined beat shape (Wang/Wheeler 2020 framework):
    #   - tip-to-base sin envelope exponent: 1.0 = bare sin; <1 = broader
    #   - base-to-tip static curl shape: 1.0 = circular arc; 2+ = distal hook
    #   - base-to-tip temporal asymmetry: 0 = sinusoidal; ~0.4 = ciliary
    #   - base-to-tip propagation extent: how far the wave actually reaches
    ttb_envelope_exp = rng.uniform(0.5, 2.0)
    btt_static_curl_shape = rng.uniform(1.0, 2.5)
    btt_temporal_asym = rng.uniform(0.0, 0.30)
    btt_propagation_extent = (
        rng.uniform(0.95, 1.20) if rng.random() < 0.85
        else rng.uniform(0.55, 0.85)  # 15% partial-propagation phenotype
    )

    # ~5% of cells: paralysed / immotile (dead, recently divided, motility mutant)
    is_paralysed = rng.random() < 0.05
    if is_paralysed:
        beat_mode = "static"
    elif force_beat_mode is not None:
        beat_mode = force_beat_mode
    else:
        beat_mode = rng.choice(["tip_to_base", "base_to_tip"], p=[0.7, 0.3])

    if beat_mode == "tip_to_base":
        freq, wavelength, amp = ttb_freq, ttb_wavelen, ttb_amp
    elif beat_mode == "base_to_tip":
        freq, wavelength, amp = btt_freq, btt_wavelen, btt_amp
    else:  # static
        freq, wavelength, amp = 0.0, 7.0, 0.0

    # Mode-switching: only motile cells switch
    mode_switch_rate = 0.0 if is_paralysed else (
        rng.uniform(0.25, 0.7) if rng.random() < 0.5 else 0.0)

    # Motion: zero for paralysed cells; otherwise mode-dependent (um/s)
    if is_paralysed:
        ttb_swim, ttb_angvel = 0.0, 0.0
        btt_swim, btt_angvel = 0.0, 0.0
    else:
        ttb_swim = rng.uniform(5.0, 18.0) if rng.random() < 0.85 else 0.0
        ttb_angvel = rng.uniform(-0.3, 0.3)
        btt_swim = rng.uniform(1.5, 5.0)
        btt_angvel = rng.choice([-1, 1]) * rng.uniform(2.5, 5.5)

    # Body phase shift varies per cell — wide range gives visibly diverse opacity.
    # 1.2 produces faint, partially-transparent cells (out-of-focus, thin
    # metacyclics); 7.0 produces very dark dense mature cells (~23x contrast range).
    body_phase = rng.uniform(1.2, 7.0)

    return ParasiteParams(
        center_x=cx, center_y=cy, body_orientation=theta,
        body_length=rng.uniform(8.0, 18.0),         # um (procyclic to nectomonad)
        body_width=rng.uniform(1.2, 3.5),           # um — wider range covers thin metacyclics
        body_peak_position=rng.uniform(0.40, 0.65),
        body_curvature=rng.uniform(-0.15, 0.15),
        body_phase_shift=body_phase,
        flagellum_length=rng.uniform(5.0, 30.0),    # um — covers just-divided
        flagellum_width=rng.uniform(0.2, 0.35),     # um
        beat_mode=beat_mode,
        beat_frequency=freq,
        beat_wavelength=wavelength,
        beat_amplitude_max=amp,
        beat_phase=rng.uniform(0, 2 * np.pi),
        tip_to_base_frequency=ttb_freq,
        tip_to_base_wavelength=ttb_wavelen,
        tip_to_base_amplitude=ttb_amp,
        base_to_tip_frequency=btt_freq,
        base_to_tip_wavelength=btt_wavelen,
        base_to_tip_amplitude=btt_amp,
        base_to_tip_static_curl=btt_static_curl,
        base_to_tip_pulse_sharpness=btt_pulse_sharpness,
        base_to_tip_distal_concentration=btt_distal_concentration,
        tip_to_base_envelope_exponent=ttb_envelope_exp,
        base_to_tip_static_curl_shape=btt_static_curl_shape,
        base_to_tip_temporal_asymmetry=btt_temporal_asym,
        base_to_tip_propagation_extent=btt_propagation_extent,
        mode_switch_rate=mode_switch_rate,
        tip_to_base_swim_speed=ttb_swim,
        tip_to_base_angular_velocity=ttb_angvel,
        base_to_tip_swim_speed=btt_swim,
        base_to_tip_angular_velocity=btt_angvel,
        rotation_rise_tau_cycles=rng.uniform(0.01, 0.05),
        rotation_decay_tau_cycles=rng.uniform(0.01, 0.05),
        swim_speed=ttb_swim if beat_mode == "tip_to_base" else btt_swim,
        angular_velocity=ttb_angvel if beat_mode == "tip_to_base" else btt_angvel,
    )
