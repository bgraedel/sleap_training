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

    # Base-to-tip beat shape (Wheeler 2020 Fig 1B; calibrated against Fig 2H,
    # which shows the mean static tip tangent angle ~ -2 rad / ~115 deg)
    base_to_tip_static_curl: float = 0.6             # tip tangent angle (units of pi); was 0.95 (~171 deg, too tight)
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
    # a slightly skewed dynamic wave. Note: Wheeler 2020 Fig 2C kymograph
    # shows roughly evenly-spaced wavefronts, so temporal asymmetry is not
    # well-constrained for Leishmania — default kept at 0 (pure sinusoid).
    # Older default of 0.10 with the +ta*sin(omega*t) form actually made the
    # power stroke LONGER than recovery, opposite to canonical ciliary beats.
    base_to_tip_temporal_asymmetry: float = 0.0

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

    # Static (paralysed) mode shape. 0.0 = straight (dead/recently divided).
    # Non-zero values produce the curled-paralysed phenotype of axoneme
    # protein deletion mutants like dIC140 and dHydin (Wheeler 2020 Fig 6),
    # which curl with the OPPOSITE polarisation to the normal base-to-tip
    # static curvature (negative sign applied inside _theta_for_mode).
    # Values around 1.5-2.8 give the >360 deg spiral curl the paper measures
    # (radius ~2.6 um).
    static_mode_curl: float = 0.0   # tip tangent angle magnitude in units of pi

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

    # --- Body internal density structure (intensity unevenness) ----------
    # Real Leishmania bodies are not uniformly dark: discrete organelles
    # (nucleus, kinetoplast) create local density maxima, and the cytoplasm
    # has low-frequency mottling. Positions are in normalized body-axis units
    # where 0 = body center, +1 = anterior (flagellar) tip, -1 = posterior
    # (Head) tip. Strengths are additive multipliers on the density envelope.
    nucleus_position: float = 0.65      # broad dense region, ~central
    nucleus_strength: float = 0.0       # 0 = off; 0.3-0.5 = visible
    nucleus_width: float = 0.18         # axial sigma (normalized units)
    kinetoplast_position: float = 0.75  # small intense spot toward flagellar pocket
    kinetoplast_strength: float = 0.0   # 0 = off; 0.4-0.7 = strong dark dot
    kinetoplast_width: float = 0.07     # axial sigma (normalized units)
    cytoplasm_mottle_strength: float = 0.0   # 0 = smooth; 0.1-0.3 = patchy
    cytoplasm_mottle_scale: float = 2.5      # spatial frequency of mottling
    # Fine, high-frequency cytoplasm grain on top of the smooth mottle above —
    # the speckly "not perfectly even" intracellular texture seen at high mag.
    # Same intrinsic-coordinate (pose-invariant) sinusoid model as the mottle,
    # but more components at a higher spatial frequency. 0 = off.
    cytoplasm_grain_strength: float = 0.0    # 0 = off; 0.1-0.4 = grainy
    cytoplasm_grain_scale: float = 8.0       # spatial frequency (higher = finer)
    body_texture_seed: int = 0          # per-cell RNG seed for reproducible mottle

    # --- High-magnification micro-texture (visible ~>=60-100x) -----------
    # The smooth ellipsoidal body + low-frequency mottle above captures the
    # 20x/40x look, but at 100x phase contrast real promastigotes show
    # discrete sub-organelle detail it misses:
    #   - dark granules: small intense black dots (lipid droplets, glycosomes,
    #     acidocalcisomes); HIGHER optical path -> add phase -> darker.
    #   - vacuoles / clear (white) spots: lower-refractive-index inclusions;
    #     in POSITIVE phase contrast they read BRIGHTER than the cytoplasm
    #     (a clear hole inside the dark body), typically with a thin dark rim
    #     produced automatically by the halo/shade-off optics. Modeled as a
    #     NEGATIVE local phase delta.
    #   - irregular outline: the silhouette is lumpy and asymmetric, not the
    #     analytic tear-drop.
    #   - pronounced tips: tips taper to a clearer, finer point.
    # Sizes are in µm, so a feature only resolves when the pixel size is small
    # enough (a 0.3 µm granule is ~1 px at 20x but ~5 px at 100x). All defaults
    # are 0 / off, so low-magnification output and every existing dataset are
    # bit-for-bit unchanged until these are explicitly enabled.
    granule_density: float = 0.0        # mean dark granules per cell (Poisson count)
    granule_strength: float = 0.6       # absorption depth per granule, 0..1
                                        # (fraction of light blocked; higher = blacker)
    granule_size_um: float = 0.28       # granule radius sigma (um)
    vacuole_density: float = 0.0        # mean bright vacuoles/clear spots per cell
    vacuole_strength: float = 0.6       # subtracted density per vacuole (brighter)
    vacuole_size_um: float = 0.6        # vacuole radius sigma (um)
    # Small bright "white dots": tiny refractile specks, distinct from the
    # larger clear vacuoles. Modeled as a multiplicative BRIGHTENING (amplitude
    # object, the bright counterpart of the absorbing granules) so they read
    # crisply white regardless of the underlying body phase.
    whitedot_density: float = 0.0       # mean white dots per cell (Poisson count)
    whitedot_strength: float = 0.8      # brightness boost per dot (transmission += )
    whitedot_size_um: float = 0.12      # white-dot radius sigma (um); keep small
    microtexture_seed: int = 0          # RNG seed for granule/vacuole/dot placement
                                        # (0 -> fall back to body_texture_seed)
    body_edge_irregularity: float = 0.0 # 0 = smooth outline; 0.05-0.15 = lumpy
    tip_sharpness: float = 0.0          # 0 = legacy rounded tips; >0 = pointier

    # --- Division state --------------------------------------------------
    # A dividing cell is modeled as a SINGLE ParasiteParams with a fatter,
    # rounder, slightly bent body and a second flagellum enabled (Wheeler
    # 2011, Wheeler/Gluenz/Gull 2013). This flag is informational only.
    is_dividing_daughter: bool = False

    # --- Second flagellum (dividing-cell phenotype) ----------------------
    # During Leishmania promastigote cytokinesis the kinetoplast and basal
    # body duplicate, and a SECOND flagellum grows from the new probasal
    # body in a NEW flagellar pocket immediately adjacent to the old one
    # (Wheeler/Gluenz/Gull 2011, 2013). The two pockets are only resolvable
    # by EM (~0.3-0.6 um apart), so in PhC the two flagella appear to exit
    # the anterior pole side-by-side, roughly parallel with only a few
    # degrees of divergence. The new flagellum reaches ~half to two-thirds
    # the length of the old by the time mitosis completes.
    #
    # When enabled, render_parasite_phase draws a second flagellum starting
    # at the same anterior pole as the primary, offset laterally and
    # slightly tilted; length = ``flagellum_length * second_flagellum_length_scale``.
    # Keypoints follow the PRIMARY flagellum only — the second flagellum is
    # rendered context but not labeled, mirroring how SLEAP would just track
    # the dominant flagellum on a dividing cell.
    second_flagellum_enabled: bool = False
    second_flagellum_length_scale: float = 1.0    # 0..1; fraction of main length
    second_flagellum_lateral_offset: float = 0.0  # um, perpendicular at base
    second_flagellum_angle_offset: float = 0.0    # rad, divergence at base
    second_flagellum_phase_offset: float = 0.0    # rad, beat phase offset

    # --- Flagellar-beat-driven body wobble -------------------------------
    # Real Leishmania bodies counter-oscillate in response to the flagellar
    # beat: in low-Reynolds-number swimming the lateral fluid-drag reaction
    # to each beat stroke shows up as a small lateral wiggle plus a slight
    # yaw of the cell body, at the beat frequency. Reported amplitudes for
    # promastigotes are ~0.3-0.8 um lateral and ~3-12 deg yaw (Walker 2019,
    # Wheeler 2017 supplementary tracking traces). Applied at render time;
    # the underlying center_x/center_y/body_orientation (the swim state) are
    # unchanged in advance_parasites, so the wobble does not drift the cell.
    body_lateral_wobble_amplitude: float = 0.0  # um, perpendicular to body axis
    body_yaw_wobble_amplitude: float = 0.0      # rad, rotation about body center
    body_wobble_phase_lag: float = 0.0          # rad, offset from beat phase

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
        second_flagellum_lateral_offset=p.second_flagellum_lateral_offset * s,
        body_lateral_wobble_amplitude=p.body_lateral_wobble_amplitude * s,
        granule_size_um=p.granule_size_um * s,
        vacuole_size_um=p.vacuole_size_um * s,
        whitedot_size_um=p.whitedot_size_um * s,
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

    # Tip sharpening: the bare sqrt profile gives a blunt, semicircular cap.
    # Raising the profile to a power that grows toward the ends (|s|->1)
    # pinches the shoulders in so the tips taper to a finer, clearer point,
    # while leaving the widest part of the body unchanged (exponent->1 at the
    # centre). 0 = legacy rounded tips.
    if p.tip_sharpness > 0:
        tip_w = s ** 2  # 0 at centre, 1 at both tips
        profile = profile ** (1.0 + 3.0 * p.tip_sharpness * tip_w)

    half_width = (p.body_width / 2) * profile
    along_axis = s * (p.body_length / 2)
    centerline_offset = p.body_curvature * (p.body_length / 2) * (1.0 - s ** 2)

    # Outline irregularity: real high-mag bodies are lumpy and asymmetric, not
    # the smooth analytic tear-drop. Perturb the upper and lower edges with
    # INDEPENDENT seeded low-frequency noise so the silhouette is asymmetric.
    # The (1 - s**2) taper anchors both tips (zero perturbation at the ends) so
    # irregularity never detaches or balloons a tip. Because the body mask is
    # rasterised from this same polygon, segmentation labels follow the lumps.
    if p.body_edge_irregularity > 0:
        def _edge_bump(seed_offset):
            erng = np.random.default_rng((p.body_texture_seed + seed_offset)
                                         & 0x7FFFFFFF)
            b = np.zeros_like(s)
            for _ in range(5):
                f = erng.uniform(1.5, 6.0)
                ph = erng.uniform(0, 2 * np.pi)
                amp = erng.uniform(0.4, 1.0)
                b += amp * np.sin(np.pi * f * (s + 1.0) + ph)
            b /= (np.max(np.abs(b)) + 1e-6)
            return b * (1.0 - s ** 2)
        upper_hw = half_width * (1.0 + p.body_edge_irregularity * _edge_bump(101))
        lower_hw = half_width * (1.0 + p.body_edge_irregularity * _edge_bump(202))
    else:
        upper_hw = lower_hw = half_width

    upper = np.column_stack([along_axis, upper_hw + centerline_offset])
    lower = np.column_stack([along_axis[::-1],
                             -lower_hw[::-1] + centerline_offset[::-1]])
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
        if p.static_mode_curl == 0.0:
            return s_norm, np.zeros_like(s_norm)
        # Curled-paralysed phenotype (Wheeler 2020 Fig 6). Negative sign:
        # the paper shows these mutants curl OPPOSITE to the base-to-tip
        # static curvature, with the PFR ending up on the outside of the
        # coil. Use the same distal-concentration shape as the base-to-tip
        # static so the bend is pushed toward the tip.
        theta = -np.pi * p.static_mode_curl * (s_norm ** p.base_to_tip_static_curl_shape)
        return s_norm, theta

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


def second_flagellum_curve(p: ParasiteParams, t: float,
                           n_points: int = 80) -> Optional[np.ndarray]:
    """Centerline of the second (new) flagellum in a dividing cell.

    Starts at the same anterior pole as the primary flagellum, offset
    laterally (perpendicular to the body axis) by
    ``second_flagellum_lateral_offset`` and tilted by
    ``second_flagellum_angle_offset``. Length is the primary length scaled
    by ``second_flagellum_length_scale``. The beat shares the primary's
    waveform but with ``second_flagellum_phase_offset`` added to the beat
    phase so the two flagella are not perfectly synchronised.

    Returns None when the second flagellum is disabled or has zero length.
    """
    if not p.second_flagellum_enabled or p.second_flagellum_length_scale <= 0:
        return None
    L2 = p.flagellum_length * p.second_flagellum_length_scale
    if L2 <= 1e-6:
        return None

    p_beat = replace(
        p,
        flagellum_length=L2,
        beat_phase=p.beat_phase + p.second_flagellum_phase_offset,
    )
    _, theta = beat_tangent_angle(p_beat, t, n_points=n_points)
    theta = theta + p.second_flagellum_angle_offset

    ds = L2 / (n_points - 1) if n_points > 1 else L2
    dx = np.cos(theta) * ds
    dy = np.sin(theta) * ds
    x_local = np.concatenate([[0.0], np.cumsum(dx[:-1])])
    y_local = np.concatenate([[0.0], np.cumsum(dy[:-1])])
    # Lateral offset in body-local frame (perpendicular to body axis).
    y_local = y_local + p.second_flagellum_lateral_offset

    local = np.column_stack([x_local, y_local]).astype(np.float32)
    c, sn = np.cos(p.body_orientation), np.sin(p.body_orientation)
    R = np.array([[c, -sn], [sn, c]])
    world = local @ R.T
    anterior_offset = (p.body_length / 2) * np.array([c, sn])
    return world + np.array([p.center_x, p.center_y]) + anterior_offset


def _apply_body_wobble(p: ParasiteParams, t: float) -> ParasiteParams:
    """Apply flagellar-beat-driven body wobble to (center, orientation).

    Low-Reynolds-number swimmers like Leishmania promastigotes cannot
    "absorb" the flagellar beat into a heavy body the way large swimmers
    do: each stroke generates a lateral drag reaction that the cell body
    must take up as a small lateral translation plus a slight yaw. The
    response is at the beat frequency, with amplitude set by the ratio of
    body to flagellar drag (typically ~0.3-0.8 um lateral and a few deg
    of yaw for promastigotes).

    Modelled here as a pure sinusoid driven by the active beat mode's
    frequency. Returns the input unchanged when both amplitudes are zero
    or the active mode has no beat (static / paralysed cell).
    """
    if (p.body_lateral_wobble_amplitude == 0.0
            and p.body_yaw_wobble_amplitude == 0.0):
        return p

    mode_a, mode_b, blend = _get_active_mode_blend(p, t)

    def _freq(m):
        if m == "tip_to_base":
            return _coalesce(p.tip_to_base_frequency, p.beat_frequency) or 0.0
        if m == "base_to_tip":
            return _coalesce(p.base_to_tip_frequency, p.beat_frequency) or 0.0
        return 0.0  # static: no beat -> no wobble

    freq = (1.0 - blend) * _freq(mode_a) + blend * _freq(mode_b)
    if freq <= 1e-3:
        return p

    omega = 2.0 * np.pi * freq
    phase = omega * t + p.beat_phase + p.body_wobble_phase_lag
    s_phase = float(np.sin(phase))

    # Lateral offset in body-local +y direction (perpendicular to body axis),
    # rotated into world frame by body_orientation.
    lat = p.body_lateral_wobble_amplitude * s_phase
    c, sn = np.cos(p.body_orientation), np.sin(p.body_orientation)
    dx = -sn * lat
    dy = c * lat

    yaw = p.body_yaw_wobble_amplitude * s_phase

    return replace(
        p,
        center_x=p.center_x + dx,
        center_y=p.center_y + dy,
        body_orientation=p.body_orientation + yaw,
    )


# ----------------------------------------------------------------------------
# Rendering (per-parasite, in a local bounding-box tile)
# ----------------------------------------------------------------------------

# Minimum flagellum MASK width in output pixels — segmentation label ONLY; the
# rendered image keeps the physically-correct (possibly thin) flagellum. A real
# ~0.25 um flagellum is <=1 px at low magnification (e.g. 20x), too thin to be a
# usable segmentation target, so the label mask is floored here while the image
# is left untouched.
MIN_FLAGELLUM_MASK_PX = 3


def _polyline_thickness_for_width(target_w: int) -> int:
    """Translate desired rendered line width (output px) to the cv2.polylines
    ``thickness`` arg. cv2 fattens lines with thickness>=2 by 1-2 px (the
    rasterised width is N+2 for odd N>=3 and N+1 for even N>=2). Passing the
    target width directly produces masks that are ~2 px wider than requested
    — e.g. ``MIN_FLAGELLUM_MASK_PX=3`` rendered as a 5-px-wide line. This
    inverts the mapping so odd targets >=3 hit exactly and even targets at
    most overshoot by 1 px (acceptable for masks).
    """
    w = int(target_w)
    if w <= 1:
        return 1
    return max(1, w - 1)


def _flag_chain_points(flag: np.ndarray, n_interior: int) -> list:
    """Evenly-spaced interior samples along a flagellum curve (excludes the
    base and tip endpoints)."""
    interior_idx = np.linspace(0, len(flag) - 1, n_interior + 2)[1:-1]
    return [flag[int(round(i))] for i in interior_idx]


def _make_keypoints(p: ParasiteParams, flag: np.ndarray,
                    flag2: Optional[np.ndarray] = None) -> dict:
    posterior = np.array([p.center_x, p.center_y]) - (p.body_length / 2) * np.array([
        np.cos(p.body_orientation), np.sin(p.body_orientation),
    ])

    kp = {"Head": tuple(posterior), "Base": tuple(flag[0])}
    for k, pt in enumerate(_flag_chain_points(flag, p.n_flagellum_keypoints), start=1):
        kp[f"Flag{k}"] = tuple(pt)
    kp["Tip"] = tuple(flag[-1])

    # Dividing cells have a second flagellum from the same anterior pole
    # (shared Base). Emit it as its own keypoint chain (Flag2_1..Flag2_N, Tip2)
    # using the SAME interior count, so a fixed two-flagellum skeleton applies
    # to every cell: single-flagellum cells simply omit these nodes (they
    # become NaN / not-visible in SLEAP, which is the correct label).
    if flag2 is not None and len(flag2) >= 2:
        for k, pt in enumerate(_flag_chain_points(flag2, p.n_flagellum_keypoints), start=1):
            kp[f"Flag2_{k}"] = tuple(pt)
        kp["Tip2"] = tuple(flag2[-1])
    return kp


def render_parasite_phase(p: ParasiteParams, t: float, image_shape: tuple,
                          *,
                          optics: Optional[OpticsParams] = None,
                          halo_margin: int = 20,
                          edge_smooth_sigma_px: Optional[float] = None,
                          return_masks: bool = False):
    """
    Render one parasite into a local bbox tile.

    Returns (tile, (y0, x0), keypoints, transmission). `tile` is None if the
    parasite is fully outside the image. ``transmission`` is a float32 tile
    (same shape as `tile`, values in [0.02, 4.0], 1.0 = no effect) carrying the
    amplitude micro-texture — absorbing dark granules (<1) and bright white
    dots (>1) — applied multiplicatively to the composited image in
    :func:`render_scene` AFTER the phase optics (these are amplitude, not
    phase, objects — see the granule/white-dot comment in the body). It is None
    when the cell has no granules or white dots.

    If ``return_masks=True``, returns
    (tile, (y0, x0), keypoints, body_mask, flag_mask, flag_masks_per,
    transmission) instead.
    ``body_mask`` and ``flag_mask`` are crisp binary ``uint8`` tiles (same
    shape as `tile`) marking the body silhouette and the union of all
    flagella, for segmentation-label generation. ``flag_masks_per`` is a list
    of ``uint8`` tiles, one per flagellum (length 1 for normal cells, length
    2 for dividing cells with ``second_flagellum_enabled``), so the writer
    can emit each flagellum as its own YOLO instance while still sharing the
    parent cell's animal_id. ``body_mask`` / ``flag_mask`` are None,
    ``flag_masks_per`` is the empty list, and ``transmission`` is None when
    `tile` is None.

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

    # Beat-driven body wobble (post-conversion so amplitude is in px). Acts
    # on (center_x, center_y, body_orientation); body, both flagella, and
    # keypoints are all derived from the wobbled pose so they move together.
    p = _apply_body_wobble(p, t)

    H, W = image_shape

    body = body_polygon(p)
    flag = flagellum_curve(p, t, n_points=80)
    flag2 = second_flagellum_curve(p, t, n_points=80)
    keypoints = _make_keypoints(p, flag, flag2)

    pts_list = [body, flag]
    if flag2 is not None:
        pts_list.append(flag2)
    pts = np.vstack(pts_list)
    x0 = max(0, int(np.floor(pts[:, 0].min())) - halo_margin)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - halo_margin)
    x1 = min(W, int(np.ceil(pts[:, 0].max())) + halo_margin)
    y1 = min(H, int(np.ceil(pts[:, 1].max())) + halo_margin)
    th, tw = y1 - y0, x1 - x0
    if th <= 0 or tw <= 0:
        if return_masks:
            return None, (0, 0), keypoints, None, None, [], None
        return None, (0, 0), keypoints, None

    offset = np.array([x0, y0], dtype=np.float32)
    body_local = np.round(body - offset).astype(np.int32)
    flag_local = (flag - offset).astype(np.float32)
    flag2_local = ((flag2 - offset).astype(np.float32)
                   if flag2 is not None else None)

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

    # Organelle density maxima: nucleus (broad) + kinetoplast (small, intense)
    # create the characteristic uneven darkness of real Leishmania bodies.
    if p.nucleus_strength > 0:
        density_env += p.nucleus_strength * np.exp(
            -((along - p.nucleus_position) ** 2) / (2 * p.nucleus_width ** 2))
    if p.kinetoplast_strength > 0:
        density_env += p.kinetoplast_strength * np.exp(
            -((along - p.kinetoplast_position) ** 2) / (2 * p.kinetoplast_width ** 2))

    # Low-frequency cytoplasm mottling: per-cell random sinusoids along the
    # body axis (plus a mild cross-axis term) give patchy density variation.
    if p.cytoplasm_mottle_strength > 0:
        across = (-(xx - cx_l) * ay + (yy - cy_l) * ax) / max(p.body_width / 2, 1e-3)
        mrng = np.random.default_rng(p.body_texture_seed)
        mottle = np.zeros((th, tw), dtype=np.float32)
        n_comp = 4
        for _ in range(n_comp):
            f = p.cytoplasm_mottle_scale * mrng.uniform(0.5, 1.5)
            phase = mrng.uniform(0, 2 * np.pi)
            amp = mrng.uniform(0.5, 1.0) / n_comp
            cross_w = mrng.uniform(0.0, 0.4)
            mottle += amp * np.sin(2 * np.pi * f * (along + cross_w * across) + phase)
        density_env = density_env * (1.0 + p.cytoplasm_mottle_strength * mottle)

    # Fine high-frequency cytoplasm grain: same intrinsic-coordinate sinusoid
    # model as the mottle (so it is pose-invariant and does not flicker), but
    # more components at a higher spatial frequency — the speckly, uneven look
    # of real cytoplasm at high magnification.
    if p.cytoplasm_grain_strength > 0:
        across_g = (-(xx - cx_l) * ay + (yy - cy_l) * ax) / max(p.body_width / 2, 1e-3)
        grng2 = np.random.default_rng((p.body_texture_seed + 777) & 0x7FFFFFFF)
        grain = np.zeros((th, tw), dtype=np.float32)
        n_comp = 8
        for _ in range(n_comp):
            f = p.cytoplasm_grain_scale * grng2.uniform(0.6, 1.6)
            phase = grng2.uniform(0, 2 * np.pi)
            amp = grng2.uniform(0.5, 1.0) / n_comp
            cross_w = grng2.uniform(0.3, 1.0)
            grain += amp * np.sin(2 * np.pi * f * (along + cross_w * across_g) + phase)
        density_env = density_env * (1.0 + p.cytoplasm_grain_strength * grain)

    density_env = np.clip(density_env, 0.0, None)
    body_phase = (thickness * p.body_phase_shift * density_env).astype(np.float32)
    body_phase = cv2.GaussianBlur(body_phase, (0, 0),
                                  sigmaX=max(edge_smooth_sigma_px, 1e-3))

    # High-magnification micro-texture. Two DIFFERENT physical mechanisms, on
    # purpose:
    #
    #   * Vacuoles / clear spots are PHASE objects (lower refractive index than
    #     the cytoplasm) -> a NEGATIVE phase delta. In this optics model that
    #     reads BRIGHTER than the body (a clear hole), with the halo/shade-off
    #     terms giving the thin dark rim for free. Clipped at 0 so it bottoms
    #     out at background brightness, never a super-bright negative-phase pit.
    #
    #   * Granules are ABSORBING (amplitude) objects, NOT phase objects. This is
    #     deliberate: the body already sits well past the shade-off crossover
    #     (~phi 2.4 for the default gain/shadeoff), where ADDING phase makes a
    #     pixel brighter, not darker — so a phase granule on a dense body comes
    #     out white. A real dark granule (lipid droplet, glycosome) instead
    #     attenuates transmitted light, so we model it as a multiplicative
    #     darkening of the final image (`granule_absorption`, applied in
    #     render_scene AFTER the phase optics), which is guaranteed dark
    #     regardless of the underlying body phase.
    #
    # Spots are placed in BODY-INTRINSIC coordinates (normalized along/across
    # the cell axis), NOT in raster pixel order, so they stay locked to the
    # same material point as the cell swims and rotates — exactly how the
    # nucleus/kinetoplast/mottle above are pose-invariant. (An earlier version
    # picked the j-th interior pixel each frame, which made the spots appear to
    # wander inside the body because that pixel list changes with pose.) Local
    # `thickness` fades them toward the rim. Sizes are already in pixels
    # (converted by _parasite_lengths_to_px). Placement is seeded for
    # reproducibility across renders of the same cell.
    #
    #   * Granules (dark) and small white dots (bright) are both AMPLITUDE
    #     objects: they multiply a per-cell `transmission` map (<1 darkens,
    #     >1 brightens) applied to the image AFTER the phase optics, so they
    #     read crisply dark / white regardless of the body phase.
    transmission = None
    if (p.granule_density > 0 or p.vacuole_density > 0
            or p.whitedot_density > 0) and dmax > 0:
        seed = int(p.microtexture_seed) or int(p.body_texture_seed) or 1
        grng = np.random.default_rng(seed ^ 0x9E3779B9)
        L_half = p.body_length / 2.0
        W_half = max(p.body_width / 2.0, 1e-3)

        def _spot_field(n_mean, sigma_px, strength, out):
            """Accumulate Gaussian blobs (peak ~strength) into `out`, placed in
            body-intrinsic (along, across) coords mapped to the current pose."""
            n = int(grng.poisson(max(0.0, float(n_mean))))
            for _ in range(n):
                # along in [-0.9, 0.9]; across kept inside the tapered
                # silhouette via the ~sqrt(1-along^2) width profile, with a
                # 0.85 margin so spots don't sit on the rim. Fully analytic ->
                # identical every frame (no rasterisation-dependent flicker).
                a = grng.uniform(-0.9, 0.9)
                prof = np.sqrt(max(0.0, 1.0 - a * a))
                c = grng.uniform(-0.85, 0.85) * prof
                c_off = p.body_curvature * L_half * (1.0 - a * a)
                lat = c * W_half + c_off
                gx = cx_l + a * L_half * ax + lat * (-ay)
                gy = cy_l + a * L_half * ay + lat * ax
                sig = max(float(sigma_px) * grng.uniform(0.6, 1.4), 0.5)
                amp = float(strength) * grng.uniform(0.6, 1.0)
                rr = int(np.ceil(3.0 * sig))
                gxi, gyi = int(round(gx)), int(round(gy))
                xa, xb = max(0, gxi - rr), min(tw, gxi + rr + 1)
                ya, yb = max(0, gyi - rr), min(th, gyi + rr + 1)
                if xb <= xa or yb <= ya:
                    continue
                dxg = (np.arange(xa, xb)[None, :] - gx).astype(np.float32)
                dyg = (np.arange(ya, yb)[:, None] - gy).astype(np.float32)
                out[ya:yb, xa:xb] += amp * np.exp(
                    -(dxg ** 2 + dyg ** 2) / (2.0 * sig ** 2))

        # Vacuoles -> negative phase (bright clear spots).
        if p.vacuole_density > 0:
            vac = np.zeros((th, tw), dtype=np.float32)
            _spot_field(p.vacuole_density, p.vacuole_size_um,
                        p.vacuole_strength, vac)
            body_phase = body_phase - thickness * p.body_phase_shift * vac
            np.clip(body_phase, 0.0, None, out=body_phase)

        # Granules (dark) + white dots (bright) -> one multiplicative
        # transmission map: 1 - depth*thickness for absorbing granules, times
        # 1 + boost*thickness for brightening white dots. thickness fades both
        # toward the rim. Clipped to [0.02, 4.0]: never fully black, capped
        # bright (the final image is clipped to white downstream anyway).
        # Returned to render_scene to apply AFTER the phase optics.
        if p.granule_density > 0 or p.whitedot_density > 0:
            transmission = np.ones((th, tw), dtype=np.float32)
            if p.granule_density > 0:
                depth = np.zeros((th, tw), dtype=np.float32)
                _spot_field(p.granule_density, p.granule_size_um,
                            p.granule_strength, depth)
                np.clip(depth, 0.0, 1.0, out=depth)
                transmission *= (1.0 - depth * thickness)
            if p.whitedot_density > 0:
                boost = np.zeros((th, tw), dtype=np.float32)
                _spot_field(p.whitedot_density, p.whitedot_size_um,
                            p.whitedot_strength, boost)
                transmission *= (1.0 + boost * thickness)
            np.clip(transmission, 0.02, 4.0, out=transmission)
            transmission = transmission.astype(np.float32)

    # Flagellum: rasterize into its own buffer at its true physical width, then
    # max-compose. The rendered image keeps the (possibly thin) flagellum as-is.
    flag_buf = np.zeros((th, tw), dtype=np.float32)
    flag_int = np.round(flag_local).astype(np.int32)
    flag_thickness = max(1, int(round(p.flagellum_width)))
    cv2.polylines(flag_buf, [flag_int], isClosed=False,
                  color=float(p.flagellum_phase_shift),
                  thickness=flag_thickness,
                  lineType=cv2.LINE_AA)

    # Second flagellum (dividing-cell phenotype): rasterised into the same
    # flag_buf with the same phase shift and thickness, so a dividing cell
    # shows two flagella exiting the anterior pole side-by-side.
    flag2_int = (np.round(flag2_local).astype(np.int32)
                 if flag2_local is not None else None)
    if flag2_int is not None:
        cv2.polylines(flag_buf, [flag2_int], isClosed=False,
                      color=float(p.flagellum_phase_shift),
                      thickness=flag_thickness,
                      lineType=cv2.LINE_AA)

    np.maximum(body_phase, flag_buf, out=body_phase)

    if return_masks:
        # Crisp binary mask for segmentation labels. The MASK width is floored
        # at MIN_FLAGELLUM_MASK_PX so a 1-2 px rendered flagellum still yields a
        # usable (>=3 px) training target. This widens ONLY the label, never the
        # rendered image. The cv2 thickness arg is translated via
        # _polyline_thickness_for_width so the RASTERISED width matches the
        # request (cv2 otherwise fattens thick lines by 1-2 px on each side).
        #
        # Each flagellum is also rasterised into its own buffer in
        # `flag_masks_per`, so a dividing cell with two flagella can be
        # emitted as TWO separate YOLO instances (sharing one animal_id)
        # rather than collapsed into one polygon by the contour finder.
        mask_target_w = max(MIN_FLAGELLUM_MASK_PX, flag_thickness)
        mask_thickness = _polyline_thickness_for_width(mask_target_w)

        flag_masks_per: List[np.ndarray] = []
        fm1 = np.zeros((th, tw), dtype=np.uint8)
        cv2.polylines(fm1, [flag_int], isClosed=False, color=1,
                      thickness=mask_thickness, lineType=cv2.LINE_8)
        flag_masks_per.append(fm1)
        if flag2_int is not None:
            fm2 = np.zeros((th, tw), dtype=np.uint8)
            cv2.polylines(fm2, [flag2_int], isClosed=False, color=1,
                          thickness=mask_thickness, lineType=cv2.LINE_8)
            flag_masks_per.append(fm2)

        # Union mask: pixel-wise OR of all per-flagellum masks. Cheaper than
        # rebuilding from scratch and guarantees consistency with the parts.
        flag_mask = flag_masks_per[0].copy()
        for fm in flag_masks_per[1:]:
            np.maximum(flag_mask, fm, out=flag_mask)
        return (body_phase, (y0, x0), keypoints,
                body_mask, flag_mask, flag_masks_per, transmission)

    return body_phase, (y0, x0), keypoints, transmission


# ----------------------------------------------------------------------------
# Dividing cells
# ----------------------------------------------------------------------------

def make_dividing_pair(base: ParasiteParams,
                       division_stage: float,
                       max_splay_angle: float = 0.0,   # unused; kept for API compat
                       posterior_separation: float = 0.0,  # unused; kept for API compat
                       width_factor: float = 1.0,      # unused; kept for API compat
                       asymmetry: float = 0.0,         # unused; kept for API compat
                       rng: Optional[np.random.Generator] = None
                       ) -> List[ParasiteParams]:
    """Build a Leishmania promastigote at a pre-cytokinesis division stage.

    Biology (Wheeler/Gluenz/Gull 2011, 2013; Wang/Wheeler/Carrington 2020):
    promastigote cell-cycle progression duplicates the kinetoplast and basal
    body, then assembles a SECOND flagellum from the new probasal body inside
    a NEW flagellar pocket immediately adjacent to the old one. The two
    pockets are ~0.3-0.6 um apart and unresolvable in PhC — both flagella
    appear to exit the same anterior point, roughly parallel with only a few
    degrees of divergence. The new flagellum reaches ~½-⅔ the length of the
    old by the time mitosis completes.

    Through this stage the cell is ONE body: it widens (~1.5 um -> ~2.5 um),
    becomes proportionally rounder (aspect ~8:1 -> ~4:1), shows a gentle
    sigmoid bend along its long axis as the two basal-body units segregate,
    and may resolve two slightly darker organelle patches. This "one body,
    two flagella" stage occupies ~30-40% of the cell cycle and is the
    dominant "dividing-looking" phenotype in PhC of a log-phase culture
    (cytokinesis itself is only ~10-15%).

    Returns a list of ONE ParasiteParams (kept as a list for backwards
    compatibility with the previous two-daughter signature; callers should
    use ``list.extend(make_dividing_pair(...))``).

    Parameters
    ----------
    base : ParasiteParams
        Template (orientation, center, lengths, beat...).
    division_stage : float in [0, 1]
        Pre-cytokinesis progression. 0 ~ just-duplicated kinetoplast, nF
        very short. 1 ~ late G2: body fully widened, nF nearly equal to
        the old flagellum, organelle patches at near-final positions.
    rng : np.random.Generator, optional
        Per-cell jitter (beat phase, lateral side, texture seed).

    Notes
    -----
    Late cytokinesis (wishbone / two daughters joined at the posterior) is
    a brief stage (~10-15% of the cycle) and not modelled here.
    `max_splay_angle`, `posterior_separation`, `width_factor`, `asymmetry`
    are accepted but ignored — they apply to the now-removed two-daughter
    morphology.
    """
    if rng is None:
        rng = np.random.default_rng()
    stage = float(np.clip(division_stage, 0.0, 1.0))

    # Body widens substantially and stays at similar length, dropping the
    # aspect ratio from ~8:1 to ~4:1 across the 2F stage (Wheeler 2011
    # cytometry). Even the *earliest* "dividing-looking" cells (just after
    # K-duplication, when the nF is barely visible) are already noticeably
    # fatter than G1 — a slim tear-drop with a tiny second flagellum stub
    # doesn't look "dividing" in PhC. So we apply a baseline +40% widening
    # at stage=0 on top of an additional +60% across the stage range.
    width_boost = 1.4 + 0.6 * stage

    # Gentle banana bend (Wheeler 2011 Fig 1: cells show a subtle sigmoid
    # along the long axis as the two units segregate). A single-direction
    # parabolic bend is biologically not far off and is what body_polygon
    # supports natively.
    side = -1.0 if rng.random() < 0.5 else 1.0
    bend = side * (0.06 + 0.10 * stage)

    # Two density centres along the body axis: the duplicated kinetoplast +
    # the nucleus, both visibly darker in PhC. The kinetoplast is always
    # near the anterior (flagellar) end; the nucleus is more central. Their
    # combined dark patches create the "two cells along bent axis" look.
    nuc_pos = 0.05 - 0.30 * stage           # nucleus drifts toward posterior daughter
    kpl_pos = 0.45 + 0.15 * stage           # kinetoplast drifts toward anterior daughter

    # Second flagellum: starts very short, grows to ~⅔ of the old by late G2.
    # Length scale 0.25 -> 0.70 across the stage.
    second_length_scale = 0.25 + 0.45 * stage

    # Lateral offset of the second flagellum at its base: matches the
    # ~0.3-0.6 um pocket spacing (Wheeler/Gluenz/Gull 2013).
    side2 = -1.0 if rng.random() < 0.5 else 1.0
    lateral_offset = side2 * (0.30 + 0.30 * stage)   # um, in original units

    # Few degrees of divergence — the new flagellum is FREE in its pocket
    # (no FAZ tether in Leishmania, unlike T. brucei), so it diverges
    # slightly from the old as it exits the pocket.
    angle_offset = side2 * (0.04 + 0.06 * stage)     # rad (~2-6 degrees)

    phase_offset = float(rng.uniform(0.4, np.pi))    # not in sync with primary

    return [replace(
        base,
        body_width=float(base.body_width * width_boost),
        # Push peak-width to the center quickly so the silhouette reads as
        # an oval/almond, not a tear-drop. Centered by stage ~0.4.
        body_peak_position=float(base.body_peak_position
                                 * max(0.0, 1.0 - 2.5 * stage)),
        body_curvature=float(base.body_curvature + bend),
        # Duplicated organelles: two dark patches along the body axis.
        nucleus_position=float(nuc_pos),
        nucleus_strength=float(0.45 + 0.15 * stage),
        nucleus_width=0.14,
        kinetoplast_position=float(kpl_pos),
        kinetoplast_strength=float(0.55 + 0.20 * stage),
        kinetoplast_width=0.09,
        # Second flagellum from the same anterior pole.
        second_flagellum_enabled=True,
        second_flagellum_length_scale=float(second_length_scale),
        second_flagellum_lateral_offset=float(lateral_offset),
        second_flagellum_angle_offset=float(angle_offset),
        second_flagellum_phase_offset=phase_offset,
        # Per-cell jitter so re-rolls aren't identical.
        body_texture_seed=int(rng.integers(0, 2**31 - 1)),
        beat_phase=float(base.beat_phase + rng.uniform(-0.4, 0.4)),
        is_dividing_daughter=True,
    )]


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
                         intensity: float = 0.75,
                         clutter_level: float = 1.0) -> np.ndarray:
    """
    PhC background:
      - Vignetting (radial darkening)
      - Slow illumination tilt
      - Multi-scale texture (low-freq blotches, mid grime, fine grain)
      - Small dust particles
      - Out-of-focus debris rings (Airy/Newton ring patterns)
      - Out-of-focus blob "ghost" cells (circular)
      - Out-of-focus ELONGATED "ghost" cells (rod-shaped — mimics defocused
        Leishmania or other elongated debris, the classic false-positive trap)

    `clutter_level` (default 1.0):
      - 1.0  : normal density and intensity of artifacts (default)
      - 0.5  : sparse, clean backgrounds
      - 2.0+ : heavy clutter, intended for "dirty" negative-frame setups
               where the model needs to learn to suppress on busy backgrounds.
      Counts of every artifact type scale linearly; their max intensities
      scale sub-linearly (sqrt) to avoid producing physically unrealistic
      gigantic structures.

    The negative-frame setups in YAML can pass `clutter_level: 2.0` or higher
    via the optics dict to produce trap-rich training backgrounds.
    """
    cl = float(clutter_level)
    cl_sq = float(np.sqrt(cl))    # used for intensity scaling

    H, W = shape
    bg = np.full((H, W), intensity, dtype=np.float32)

    yy, xx = np.meshgrid(np.linspace(-1, 1, H),
                         np.linspace(-1, 1, W), indexing="ij")

    # Vignetting (radial falloff toward corners)
    bg *= 1.0 - rng.uniform(0.06, 0.14) * (xx ** 2 + yy ** 2).astype(np.float32)

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
    n_dust = int(rng.poisson(rng.uniform(8, 30) * cl))
    for _ in range(n_dust):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        radius = rng.uniform(0.8, 6.0)
        strength = rng.uniform(0.05, 0.85) * cl_sq
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

    # Out-of-focus debris rings (Newton/Airy-like patterns).
    # Stronger and larger than before; some have multiple visible oscillations.
    n_debris = int(rng.poisson(rng.uniform(2, 12) * cl))
    for _ in range(n_debris):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        # Bimodal sizes: small tight rings and large diffuse rings
        if rng.random() < 0.7:
            radius = rng.uniform(8, 22)
        else:
            radius = rng.uniform(22, 45)        # large rings (was max 22)
        # Wider strength range; occasionally very strong
        strength = rng.uniform(0.005, 0.10) * cl_sq
        if rng.random() < 0.15:                 # 15% are high-contrast
            strength *= rng.uniform(2.0, 4.0)
        # Slower decay -> more visible oscillations (more cell-mimicking)
        decay = rng.uniform(0.25, 0.7)
        ring_phase = rng.uniform(0, 2 * np.pi)
        rr = int(np.ceil(3.0 * radius))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        r = np.sqrt(dx_ ** 2 + dy_ ** 2) / radius
        ring = strength * np.cos(2 * np.pi * r + ring_phase) * np.exp(-r * decay)
        bg[y_lo:y_hi, x_lo:x_hi] += np.where(r < 3.0, ring, 0).astype(np.float32)

    # Out-of-focus circular "ghost" cells: faint, blurry, defocused.
    n_blobs = int(rng.poisson(rng.uniform(0.5, 5.0) * cl))
    for _ in range(n_blobs):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        radius = rng.uniform(20, 50)
        strength = rng.uniform(0.02, 0.10) * cl_sq
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

    # ELONGATED out-of-focus ring artifacts — what slightly elongated particles
    # look like when defocused. NOT solid rod-shaped blobs (a previous version
    # had it wrong). The real PhC signature is:
    #   - Concentric ring pattern (like the standard debris rings above)
    #   - Slightly elliptical (aspect ~1.2-2.0:1)
    #   - Ring oscillations clearly visible on the LONG sides (perpendicular
    #     to elongation axis)
    #   - Faded/invisible at the TIPS (along elongation axis), because the
    #     ring curvature there is too tight to resolve
    # Effectively: a Newton ring pattern with two opposite "missing" arcs.
    # This is the classic false-positive trap — looks vaguely cell-shaped at
    # a glance but isn't a cell.
    n_anisotropic_rings = int(rng.poisson(rng.uniform(0.3, 3.0) * cl))
    for _ in range(n_anisotropic_rings):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        # Mean radius (geometric mean of major/minor axes)
        radius = rng.uniform(15, 40)
        # Aspect ratio of the ring pattern (semi-major / semi-minor)
        aspect = rng.uniform(1.2, 2.2)
        a = radius * np.sqrt(aspect)             # semi-major
        b = radius / np.sqrt(aspect)             # semi-minor
        theta = rng.uniform(0, 2 * np.pi)
        ring_phase = rng.uniform(0, 2 * np.pi)

        strength = rng.uniform(0.01, 0.08) * cl_sq
        if rng.random() < 0.15:                  # 15% are high-contrast
            strength *= rng.uniform(2.0, 3.5)
        decay = rng.uniform(0.25, 0.6)

        # How sharp the "tip fading" is. Higher = more abrupt invisibility
        # at the long-axis tips; lower = more uniform ring.
        tip_fade_sharpness = rng.uniform(1.5, 4.0)

        rr = int(np.ceil(3.0 * a))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        # Rotate into ellipse's local frame
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u = (cos_t * dx_ + sin_t * dy_) / a       # along major
        v = (-sin_t * dx_ + cos_t * dy_) / b      # along minor
        # Anisotropic radial coordinate (1 = on the ellipse)
        r = np.sqrt(u ** 2 + v ** 2)
        # Angle in elliptical coords (0 = +major axis direction, pi/2 = +minor)
        phi = np.arctan2(v, u)
        # Tip fade: sin^N(phi). =1 on the sides (phi = ±pi/2), =0 at tips
        # (phi = 0, pi). So ring is visible perpendicular to long axis,
        # invisible parallel.
        tip_mask = np.abs(np.sin(phi)) ** tip_fade_sharpness
        ring = (strength * np.cos(2 * np.pi * r + ring_phase)
                * np.exp(-r * decay) * tip_mask)
        # Only contribute within the visible region
        bg[y_lo:y_hi, x_lo:x_hi] += np.where(r < 3.0, ring, 0).astype(np.float32)

    # HAIRS / FILAMENTS — long thin curving structures (real hairs, fibers,
    # cotton-fluff strands, dead-cell filaments). Stretch across most of the
    # frame, can extend off-edge. Dark line with PhC bright halo on both sides.
    # Highly characteristic real-microscopy contamination that the model has
    # to learn is NOT a flagellum.
    n_hairs = int(rng.poisson(rng.uniform(0.1, 1.5) * cl))
    for _ in range(n_hairs):
        # Endpoints can lie outside the frame so hairs go edge-to-edge
        x0 = rng.uniform(-0.2 * W, 1.2 * W)
        y0 = rng.uniform(-0.2 * H, 1.2 * H)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(80, max(120, 1.4 * max(H, W)))
        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)

        # Smooth curvature: perpendicular sinusoidal deflection
        perp_x, perp_y = -np.sin(angle), np.cos(angle)
        curve_amp = rng.normal(0, 1) * rng.uniform(5, 40)
        # Optional second-harmonic for a more natural bend
        bend_phase = rng.uniform(0, np.pi)
        n_freq = rng.choice([1, 1, 2])  # mostly single-bend, sometimes S-curve

        thickness = rng.uniform(0.8, 2.2)
        line_strength = rng.uniform(0.40, 0.95) * cl_sq
        # Halo should be visibly weaker than the line — real PhC hairs are
        # predominantly dark with only a *subtle* halo glow. Previous version
        # had bright accumulated halos that made the hair look like a bright
        # fiber instead of a dark one.
        halo_strength = rng.uniform(0.04, 0.14) * cl_sq
        halo_sigma_factor = rng.uniform(1.8, 3.0)

        # Sample densely along the curve so successive Gaussian stamps overlap
        # smoothly and produce a continuous line (no beading).
        n_seg = max(40, int(length / 1.2))
        ts = np.linspace(0, 1, n_seg)
        bend = curve_amp * np.sin(np.pi * n_freq * ts + bend_phase)
        xs = x0 + (x1 - x0) * ts + perp_x * bend
        ys = y0 + (y1 - y0) * ts + perp_y * bend

        # Per-stamp amplitude: scaled down since we stamp densely. The
        # constant 0.18 was chosen so dense overlap gives continuous-looking
        # line intensities comparable to dust at similar `strength` values.
        per_stamp = 0.18

        rr = int(np.ceil(thickness * halo_sigma_factor * 2.5))
        sig2 = thickness ** 2
        sig2_halo = sig2 * halo_sigma_factor ** 2

        for px, py in zip(xs, ys):
            if (px < -rr or px >= W + rr
                    or py < -rr or py >= H + rr):
                continue
            x_lo = max(0, int(px) - rr); x_hi = min(W, int(px) + rr + 1)
            y_lo = max(0, int(py) - rr); y_hi = min(H, int(py) + rr + 1)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue
            dx_ = np.arange(x_lo, x_hi) - px
            dy_ = (np.arange(y_lo, y_hi) - py)[:, None]
            r2 = dx_ ** 2 + dy_ ** 2
            stamp = (-line_strength * np.exp(-r2 / sig2)
                     + halo_strength * np.exp(-r2 / sig2_halo))
            bg[y_lo:y_hi, x_lo:x_hi] += (per_stamp * stamp).astype(np.float32)

    # FINE SPECKS — high density of very small dust (sub-pixel to ~1.5 px).
    # These give the background a "real microscopy grit" texture that the
    # current larger-dust setting doesn't quite capture. Independent of the
    # main `n_dust` so the two distributions can coexist.
    n_specks = int(rng.poisson(rng.uniform(20, 120) * cl))
    if n_specks > 0:
        speck_x = rng.uniform(0, W - 1, n_specks).astype(np.int32)
        speck_y = rng.uniform(0, H - 1, n_specks).astype(np.int32)
        speck_strength = (rng.uniform(0.10, 0.55, n_specks) * cl_sq).astype(np.float32)
        # Each speck is just one or a few pixels — too small for a kernel
        # loop to make sense. Add directly to the background array.
        # Some specks darker, some specks brighter (sign mix)
        signs = rng.choice([-1, 1], n_specks, p=[0.85, 0.15]).astype(np.float32)
        np.add.at(bg, (speck_y, speck_x), -speck_strength * signs * 0.06)

    # SCRATCHES — straight short dark streaks from coverslip damage,
    # sometimes axis-aligned (parallel to camera readout) and sometimes at
    # random angles. Distinct from hairs in that they're straight and sharper.
    # Rare: most frames have none, a small fraction have one.
    n_scratches = int(rng.poisson(rng.uniform(0.02, 0.4) * cl))
    for _ in range(n_scratches):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        if rng.random() < 0.5:
            # Axis-aligned (more common from manufacturing/handling)
            angle = float(rng.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
                          + rng.normal(0, 0.05))
        else:
            angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(10, 80)

        thickness = rng.uniform(0.6, 1.6)
        line_strength = rng.uniform(0.45, 0.95) * cl_sq
        halo_strength = rng.uniform(0.03, 0.10) * cl_sq

        n_seg = max(20, int(length / 1.0))
        ts = np.linspace(-0.5, 0.5, n_seg)
        xs = cx + length * np.cos(angle) * ts
        ys = cy + length * np.sin(angle) * ts

        per_stamp = 0.15
        sig2 = thickness ** 2
        sig2_halo = sig2 * 6.0
        rr = int(np.ceil(thickness * 6))
        for px, py in zip(xs, ys):
            if px < -rr or px >= W + rr or py < -rr or py >= H + rr:
                continue
            x_lo = max(0, int(px) - rr); x_hi = min(W, int(px) + rr + 1)
            y_lo = max(0, int(py) - rr); y_hi = min(H, int(py) + rr + 1)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue
            dx_ = np.arange(x_lo, x_hi) - px
            dy_ = (np.arange(y_lo, y_hi) - py)[:, None]
            r2 = dx_ ** 2 + dy_ ** 2
            stamp = (-line_strength * np.exp(-r2 / sig2)
                     + halo_strength * np.exp(-r2 / sig2_halo))
            bg[y_lo:y_hi, x_lo:x_hi] += (per_stamp * stamp).astype(np.float32)

    # BUBBLES — air bubbles in the mounting medium. In PhC these have the
    # OPPOSITE signature of dust: bright interior with a sharp dark rim. Very
    # characteristic look. Sizes vary widely (small to substantial fraction
    # of the FOV).
    n_bubbles = int(rng.poisson(rng.uniform(0.02, 0.30) * cl))
    for _ in range(n_bubbles):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        radius = rng.uniform(15, 80)
        edge_width = radius * rng.uniform(0.05, 0.15)
        bright_amp = rng.uniform(0.10, 0.30) * cl_sq
        dark_amp = rng.uniform(0.25, 0.55) * cl_sq

        rr = int(np.ceil(radius * 1.4))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        rdist = np.sqrt(dx_ ** 2 + dy_ ** 2)
        # Soft bright interior (Gaussian inside the rim)
        interior_falloff = np.maximum(0.0, 1.0 - (rdist / (radius - edge_width)) ** 2)
        interior_falloff = np.where(rdist < radius - edge_width, interior_falloff, 0.0)
        # Sharp dark rim at radius = radius
        rim_gauss = np.exp(-((rdist - radius) ** 2) / (edge_width ** 2))
        bubble = bright_amp * interior_falloff - dark_amp * rim_gauss
        bg[y_lo:y_hi, x_lo:x_hi] += bubble.astype(np.float32)

    # CRYSTALS — sharp-edged angular structures (salt crystals from media,
    # protein crystals). Distinct from blobs by their sharp edges and from
    # rings by their angular geometry. Rare: most frames have none.
    n_crystals = int(rng.poisson(rng.uniform(0.02, 0.25) * cl))
    for _ in range(n_crystals):
        cx = rng.uniform(0, W); cy = rng.uniform(0, H)
        length = rng.uniform(12, 50)               # along major axis
        aspect = rng.uniform(1.2, 3.0)
        width = length / aspect
        theta = rng.uniform(0, 2 * np.pi)
        strength = rng.uniform(0.10, 0.40) * cl_sq

        # Sharp logistic edge transition (~1 px wide)
        edge_sharpness = rng.uniform(2.5, 5.0)

        # 30% of crystals have an internal "fracture" line (cross/streak)
        has_fracture = rng.random() < 0.3

        rr = int(np.ceil(length * 0.8))
        x_lo, x_hi = max(0, int(cx) - rr), min(W, int(cx) + rr)
        y_lo, y_hi = max(0, int(cy) - rr), min(H, int(cy) + rr)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        dy_ = np.arange(y_lo, y_hi)[:, None] - cy
        dx_ = np.arange(x_lo, x_hi)[None, :] - cx
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u = (cos_t * dx_ + sin_t * dy_) / (length / 2)
        v = (-sin_t * dx_ + cos_t * dy_) / (width / 2)
        # Chebyshev-norm: 1 on the rectangle's edge, < 1 inside, > 1 outside.
        # Sharp logistic gives sharp-edged crystal shape.
        norm = np.maximum(np.abs(u), np.abs(v))
        mask = 1.0 / (1.0 + np.exp(edge_sharpness * (norm - 1.0)))
        crystal = -strength * mask

        if has_fracture:
            # Internal dark line along the long axis
            frac_strength = strength * rng.uniform(0.4, 0.9)
            frac_thickness = rng.uniform(0.05, 0.15)
            frac = -frac_strength * np.exp(-(v ** 2) / (frac_thickness ** 2)) * mask
            crystal += frac

        bg[y_lo:y_hi, x_lo:x_hi] += crystal.astype(np.float32)

    return bg


# ----------------------------------------------------------------------------
# Scene compositing
# ----------------------------------------------------------------------------

def _resolve_instance_masks(mask_tiles, winner_map):
    """Resolve per-cell body/flagellum masks against the phase `winner_map`.

    `mask_tiles[i]` is None (cell i off-frame) or a tuple
    ``(y0, x0, body_mask, flag_mask)`` of tile-local binary masks from
    :func:`render_parasite_phase`. Occlusion is applied via `winner_map`
    (the per-pixel index of the cell that won the phase max-composite): a mask
    pixel is dropped only where a DIFFERENT cell wins (true occlusion).
    Background pixels (winner -1) are kept — otherwise clipping against the
    phase footprint would shrink the widened flagellum label
    (drawn >= MIN_FLAGELLUM_MASK_PX) back to the cell's ~1 px rendered line.

    Returns a list aligned with `mask_tiles`. Each entry is None or a dict::

        {"y0", "x0", "body", "flag",
         "body_full", "flag_full", "animal_full", "visible_frac"}

    The ``*_full`` masks are the cell's full (amodal) true extent, ignoring
    inter-cell occlusion: ``body_full`` is the whole body, ``flag_full`` the
    whole flagellum, ``animal_full = body_full | flag_full`` (always a single
    connected region — the flagellum grows out of the body). These give one
    instance per cell that never splits under occlusion.

    ``body`` / ``flag`` are the occlusion-resolved *visible* (modal) masks:
    ``body`` is the visible body silhouette and ``flag`` the visible flagellum
    *outside* the body (the proximal flagellum embedded in the body counts as
    body, as it is not separately visible in phase contrast). These two are
    disjoint and suited to per-pixel semantic targets.

    ``visible_frac`` is the fraction of the cell's full extent that it actually
    wins in the composite (1.0 = fully visible, ~0 = almost entirely hidden),
    so callers can drop cells buried behind a neighbour.
    """
    out = []
    for idx, rec in enumerate(mask_tiles):
        if rec is None:
            out.append(None)
            continue
        # Accept legacy 4-tuples (no per-flagellum list) for safety, though
        # render_scene now always emits the 5-tuple.
        if len(rec) == 4:
            y0, x0, body_m, flag_m = rec
            flag_per = [flag_m] if flag_m is not None else []
        else:
            y0, x0, body_m, flag_m, flag_per = rec
        th, tw = body_m.shape
        win = winner_map[y0:y0 + th, x0:x0 + tw]
        owns = win == idx
        # Drop only pixels a DIFFERENT cell wins (real occlusion). Background
        # (-1) is kept so the widened flagellum mask is not clipped back to the
        # thin rendered line.
        occluded = (win != idx) & (win != -1)
        body_b = body_m.astype(bool)
        flag_b = flag_m.astype(bool)
        animal_b = body_b | flag_b
        n_full = int(animal_b.sum())
        # Per-flagellum: same occlusion resolution as the union, just kept as
        # distinct buffers so the writer can emit them as separate instances.
        flag_per_full = [m.astype(bool) for m in flag_per]
        flag_per_visible = [m & ~occluded & ~body_b for m in flag_per_full]
        out.append({
            "y0": int(y0), "x0": int(x0),
            "body": body_b & ~occluded,
            "flag": flag_b & ~occluded & ~body_b,
            "body_full": body_b,
            "flag_full": flag_b,
            "animal_full": animal_b,
            "flag_per_full": flag_per_full,
            "flag_per": flag_per_visible,
            "visible_frac": float((animal_b & owns).sum()) / max(n_full, 1),
        })
    return out


def render_scene(parasites, t: float, image_shape: tuple,
                 optics: OpticsParams, noise: CameraNoiseParams,
                 background: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None,
                 fast: bool = False,
                 occlusion_aware_labels: bool = True,
                 occlusion_patch_radius: int = 1,
                 occlusion_majority_threshold: float = 0.5,
                 clutter_level: float = 1.0,
                 return_masks: bool = False):
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
    # Multiplicative transmission map for the amplitude micro-texture: dark
    # granules (<1) and bright white dots (>1). 1.0 = no effect. These bypass
    # the phase optics and scale the final image directly (see the granule /
    # white-dot comment in render_parasite_phase). Accumulated as a product.
    transmission_total = np.ones((H, W), dtype=np.float32)
    # Per-pixel index of the cell currently winning the max-composite.
    # -1 = background (no cell has contributed any phase here yet). Needed
    # for occlusion-aware keypoints AND for occlusion-resolved masks.
    need_winner = occlusion_aware_labels or return_masks
    winner_map = np.full((H, W), -1, dtype=np.int32) if need_winner else None

    # Per-parasite (y0, x0, body_mask, flag_mask) tiles, collected only when
    # masks are requested and resolved against winner_map after compositing.
    mask_tiles = [] if return_masks else None

    all_keypoints = []
    for i, p in enumerate(parasites):
        if return_masks:
            tile, (y0, x0), kp, body_m, flag_m, flag_per, transm = \
                render_parasite_phase(
                    p, t, image_shape, optics=optics, return_masks=True)
        else:
            tile, (y0, x0), kp, transm = render_parasite_phase(
                p, t, image_shape, optics=optics)
        all_keypoints.append(kp)
        if tile is None:
            if return_masks:
                mask_tiles.append(None)
            continue
        th, tw = tile.shape
        region = phase_total[y0:y0 + th, x0:x0 + tw]
        if need_winner:
            # Where does THIS cell strictly beat the running max? Those are
            # the pixels we'll claim as ours in the winner_map.
            beats = tile > region
            region[beats] = tile[beats]
            winner_map[y0:y0 + th, x0:x0 + tw][beats] = i
        else:
            np.maximum(region, tile, out=region)
        if transm is not None:
            ar = transmission_total[y0:y0 + th, x0:x0 + tw]
            np.multiply(ar, transm, out=ar)
        if return_masks:
            mask_tiles.append((y0, x0, body_m, flag_m, flag_per))

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

    instance_masks = (_resolve_instance_masks(mask_tiles, winner_map)
                      if return_masks else None)

    if fast:
        intensity = simulate_phase_contrast_fast(phase_total, optics)
    else:
        intensity = simulate_phase_contrast(phase_total, optics)

    if background is None:
        background = synthetic_background((H, W), rng, intensity=noise.bg_intensity,
                                          clutter_level=clutter_level)
    # Amplitude micro-texture (dark granules, bright white dots) scales the
    # noise-free image multiplicatively, before camera noise so shot noise
    # scales with the changed photon count. No-op (all ones) when no cell has
    # granules or white dots.
    composite = background * intensity * transmission_total

    if fast:
        image = add_camera_noise_fast(composite, noise, rng)
    else:
        image = add_camera_noise(composite, noise, rng)
    image = np.clip(image, 0, 1)

    if return_masks:
        return image, all_keypoints, instance_masks
    return image, all_keypoints


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
    ttb_wavelen = rng.uniform(5.0, 30.0)     # um
    ttb_amp = rng.uniform(0.4, 4.0)          # um
    btt_freq = rng.uniform(2, 10)
    btt_wavelen = rng.uniform(1.0, 20.0)     # um
    btt_amp = rng.uniform(0.8, 6.0)          # um

    # Base-to-tip shape: ranges calibrated against Wheeler 2020 Fig 2H
    # (mean static tip tangent angle ~115 deg, with individual cells reaching
    # ~170 deg). Old range (0.65, 1.45) was shifted high.
    btt_static_curl = rng.uniform(0.65, 1.45)
    btt_pulse_sharpness = rng.uniform(1.0, 2.0)
    btt_distal_concentration = rng.uniform(0.35, 1.30)

    # Refined beat shape (Wang/Wheeler 2020 framework):
    #   - tip-to-base sin envelope exponent: 1.0 = bare sin; <1 = broader
    #   - base-to-tip static curl shape: 1.0 = circular arc; 2+ = distal hook
    #   - base-to-tip temporal asymmetry: kept small; sign + magnitude not
    #     well-constrained by the paper for Leishmania (see Fig 2C kymograph)
    #   - base-to-tip propagation extent: how far the wave actually reaches
    ttb_envelope_exp = rng.uniform(0.1, 2.0)
    btt_static_curl_shape = rng.uniform(1.0, 3)
    btt_temporal_asym = rng.uniform(0.0, 0.05)
    btt_propagation_extent = (
        rng.uniform(0.95, 1.20) if rng.random() < 0.85
        else rng.uniform(0.55, 0.85)  # 15% partial-propagation phenotype
    )

    # ~5% of cells: paralysed / immotile. Split per Wheeler 2020 Fig 6:
    # ~40% truly straight (dead, recently divided, generic motility mutant),
    # ~60% curled-paralysed (dIC140/dHydin-like, with the >360 deg spiral
    # curl that the paper specifically documents).
    is_paralysed = rng.random() < 0.05
    if is_paralysed:
        static_mode_curl = 0.0 if rng.random() < 0.4 else rng.uniform(1.5, 2.8)
    else:
        static_mode_curl = 0.0
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

    # Motion: zero for paralysed cells; otherwise mode-dependent (um/s).
    # Note: base-to-tip angular velocity has a FIXED sign — the flagellum
    # bend is always +y in the local frame (Wheeler 2020 fixed polarisation
    # finding), so the rotation it induces is also fixed in the cell frame.
    # Visual diversity in the lab frame comes from random body_orientation.
    if is_paralysed:
        ttb_swim, ttb_angvel = 0.0, 0.0
        btt_swim, btt_angvel = 0.0, 0.0
    else:
        ttb_swim = rng.uniform(5.0, 18.0) if rng.random() < 0.85 else 0.0
        ttb_angvel = rng.uniform(-0.3, 0.3)
        btt_swim = rng.uniform(1.5, 5.0)
        btt_angvel = rng.uniform(2.5, 5.5)

    # Body phase shift varies per cell — wide range gives visibly diverse opacity.
    # 1.2 produces faint, partially-transparent cells (out-of-focus, thin
    # metacyclics); 7.0 produces very dark dense mature cells (~23x contrast range).
    body_phase = rng.uniform(1.2, 7.0)

    # Beat-driven body wobble: paralysed cells don't wobble; motile cells get
    # ~0.2-0.7 um lateral and ~3-10 deg yaw, with a random phase relative to
    # the beat. Lateral and yaw are kept somewhat correlated (single sin) but
    # the user can flip yaw sign via negative amplitude in the GUI.
    if is_paralysed:
        lat_wobble, yaw_wobble, wobble_lag = 0.0, 0.0, 0.0
    else:
        lat_wobble  = rng.uniform(0.0, 0.001)
        yaw_wobble  = rng.uniform(0.001, 0.07) * (1 if rng.random() < 0.5 else -1)
        wobble_lag  = rng.uniform(0.0, 0.5 * np.pi)

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
        static_mode_curl=static_mode_curl,
        mode_switch_rate=mode_switch_rate,
        tip_to_base_swim_speed=ttb_swim,
        tip_to_base_angular_velocity=ttb_angvel,
        base_to_tip_swim_speed=btt_swim,
        base_to_tip_angular_velocity=btt_angvel,
        rotation_rise_tau_cycles=rng.uniform(0.01, 0.05),
        rotation_decay_tau_cycles=rng.uniform(0.01, 0.05),
        swim_speed=ttb_swim if beat_mode == "tip_to_base" else btt_swim,
        angular_velocity=ttb_angvel if beat_mode == "tip_to_base" else btt_angvel,
        body_lateral_wobble_amplitude=lat_wobble,
        body_yaw_wobble_amplitude=yaw_wobble,
        body_wobble_phase_lag=wobble_lag,
    )
