"""
Real-time live preview with full parameter editing (Dear PyGui).

Adds, on top of the basic version:
  - per-parasite sliders for body, flagellum, beat (per-mode), motion, and
    mode-switching parameters
  - parasite selector + apply-to-all toggle
  - beat mode combobox + buttons to regenerate / clear the stochastic
    mode schedule
  - JSON export/import of optics + noise + all parasites (incl. mode_schedule)
  - red ring overlay on the selected parasite

UI niceties (improved):
  - autoscaling layout: the preview fills the left side, controls live on
    the right; resizing the OS window resizes both
  - the rendered image rescales to fit the preview pane (aspect preserved)
  - controls split into three tabs (Scene / Optics & Noise / Parasite)
  - keyboard shortcuts: Space = pause, R = reseed all, B = rebuild bg
  - live skeleton overlay (toggle) so you can verify SLEAP keypoint labels
  - playback speed decoupled from the frame-rate cap: "max fps" only controls
    smoothness; "playback speed" controls how fast the biology evolves

Install:
    pip install dearpygui

Usage:
    python live_demo_gui.py [--size 512] [--n 8] [--fps 60]
"""
import argparse
import dataclasses as dc
import json
import time

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

import synthetic_leishmania as L
from synthetic_leishmania import ParasiteParams


# ---------------------------------------------------------------------------
# Slider definitions: (field_name, type, (min, max))
# Adjust ranges here as you learn what's useful in practice.
# ---------------------------------------------------------------------------
OPTICS_RANGES = [
    ("pixel_size_um",             float, (0.05, 1.5)),  # 100x ~ 0.065, 60x ~ 0.108, 40x ~ 0.163, 20x = 0.325, 10x ~ 0.65
    ("psf_sigma_um",              float, (0.05, 2.0)),
    ("halo_strength",             float, (0.0, 2.0)),
    ("halo_lowpass_sigma_um",     float, (0.5, 15.0)),
    ("intensity_gain",            float, (0.0, 3.0)),
    ("shadeoff_threshold",        float, (0.0, 2.0)),
    ("shadeoff_strength",         float, (0.0, 1.0)),
    ("body_edge_smooth_sigma_um", float, (0.05, 2.0)),
]
NOISE_RANGES = [
    ("full_well_photons", float, (100.0, 30000.0)),
    ("read_noise_e",      float, (0.0, 30.0)),
    ("dark_offset",       float, (-0.5, 0.5)),
    ("bg_intensity",      float, (0.0, 1.5)),
]
PARASITE_GROUPS = {
    "Body": [
        ("body_length",        float, (5.0, 30.0)),    # um
        ("body_width",         float, (0.5, 6.0)),     # um
        ("body_peak_position", float, (-1.0, 1.0)),
        ("body_curvature",     float, (-0.3, 0.3)),
        ("body_phase_shift",   float, (0.5, 5.0)),
    ],
    "Flagellum": [
        ("flagellum_length",      float, (5.0, 30.0)),  # um
        ("flagellum_width",       float, (0.1, 1.5)),   # um
        ("flagellum_phase_shift", float, (0.0, 3.0)),
    ],
    "Beat: tip-to-base": [
        ("tip_to_base_frequency",  float, (0.0, 50.0)),
        ("tip_to_base_wavelength", float, (2.0, 20.0)),  # um
        ("tip_to_base_amplitude",  float, (0.0, 3.0)),   # um
    ],
    "Beat: base-to-tip": [
        ("base_to_tip_frequency",  float, (0.0, 30.0)),
        ("base_to_tip_wavelength", float, (3.0, 25.0)),  # um
        ("base_to_tip_amplitude",  float, (0.0, 5.0)),   # um
    ],
    "Beat shape: tip-to-base": [
        ("tip_to_base_envelope_exponent", float, (0.5, 3.0)),
    ],
    "Beat shape: base-to-tip": [
        ("base_to_tip_static_curl",          float, (0.3, 2.0)),
        ("base_to_tip_static_curl_shape",    float, (1.0, 3.5)),
        ("base_to_tip_pulse_sharpness",      float, (0.4, 3.0)),
        ("base_to_tip_distal_concentration", float, (0.3, 2.5)),
        ("base_to_tip_temporal_asymmetry",   float, (0.0, 0.7)),
        ("base_to_tip_propagation_extent",   float, (0.4, 1.5)),
    ],
    "Motion: tip-to-base": [
        ("tip_to_base_swim_speed",       float, (0.0, 50.0)),  # um/s
        ("tip_to_base_angular_velocity", float, (-3.0, 3.0)),
    ],
    "Motion: base-to-tip": [
        ("base_to_tip_swim_speed",       float, (0.0, 20.0)),  # um/s
        ("base_to_tip_angular_velocity", float, (-8.0, 8.0)),
    ],
    "Mode switching": [
        ("mode_switch_rate",         float, (0.0, 5.0)),
        ("mode_transition_duration", float, (0.0, 1.0)),
    ],
    "Body density: organelles": [
        ("nucleus_position",     float, (-1.0, 1.0)),
        ("nucleus_strength",     float, (0.0, 1.5)),
        ("nucleus_width",        float, (0.02, 0.5)),
        ("kinetoplast_position", float, (-1.0, 1.0)),
        ("kinetoplast_strength", float, (0.0, 1.5)),
        ("kinetoplast_width",    float, (0.02, 0.3)),
    ],
    "Body density: cytoplasm": [
        ("cytoplasm_mottle_strength", float, (0.0, 1.0)),
        ("cytoplasm_mottle_scale",    float, (0.5, 8.0)),
        ("cytoplasm_grain_strength",  float, (0.0, 0.6)),   # fine high-freq grain
        ("cytoplasm_grain_scale",     float, (3.0, 25.0)),
        ("body_texture_seed",         int,   (0, 1_000_000)),
    ],
    # High-magnification detail. Granules = dark dots; vacuoles = bright/clear
    # spots; edge irregularity = lumpy outline; tip sharpness = pointier tips.
    # Sizes are in µm, so they only become visible at small pixel_size_um
    # (set ~0.065 for 100x in the Optics tab). density = mean count per cell.
    "Body micro-texture (high-mag)": [
        ("granule_density",        float, (0.0, 30.0)),
        ("granule_strength",       float, (0.0, 1.0)),     # absorption depth 0..1
        ("granule_size_um",        float, (0.05, 1.0)),    # um
        ("vacuole_density",        float, (0.0, 15.0)),
        ("vacuole_strength",       float, (0.0, 1.5)),
        ("vacuole_size_um",        float, (0.1, 1.5)),     # um
        ("whitedot_density",       float, (0.0, 40.0)),
        ("whitedot_strength",      float, (0.0, 3.0)),     # brightness boost
        ("whitedot_size_um",       float, (0.04, 0.4)),    # um (small)
        ("body_edge_irregularity", float, (0.0, 0.3)),
        ("tip_sharpness",          float, (0.0, 2.5)),
        ("microtexture_seed",      int,   (0, 1_000_000)),
    ],
    "Second flagellum (dividing cells)": [
        ("second_flagellum_length_scale",   float, (0.0, 1.0)),
        ("second_flagellum_lateral_offset", float, (-2.0, 2.0)),   # um
        ("second_flagellum_angle_offset",   float, (-0.4, 0.4)),   # rad
        ("second_flagellum_phase_offset",   float, (-np.pi, np.pi)),
    ],
    "Body wobble (beat-driven)": [
        ("body_lateral_wobble_amplitude", float, (0.0, 2.0)),         # um
        ("body_yaw_wobble_amplitude",     float, (-0.5, 0.5)),        # rad
        ("body_wobble_phase_lag",         float, (-np.pi, np.pi)),    # rad
    ],
    "Pose": [
        ("center_x",         float, (0.0, 4096.0)),  # image px
        ("center_y",         float, (0.0, 4096.0)),  # image px
        ("body_orientation", float, (-2 * np.pi, 2 * np.pi)),
    ],
}

BEAT_MODES   = ["tip_to_base", "base_to_tip", "static"]
RENDER_MODES = ["fast", "accurate", "skip_noise"]

# Kymograph (Wheeler 2020 Fig 2A,C). Tangent angle along the flagellum
# (horizontal axis, base on the left, tip on the right) plotted over time
# (vertical axis, oldest at the top, newest at the bottom -- as in the paper).
# Yellow = positive tangent angle, cyan = negative, black = zero.
KYMO_HISTORY_FRAMES = 200            # ~3 s of history at 60 fps
KYMO_FLAG_SAMPLES   = 80             # samples along the flagellum
KYMO_HEIGHT_PX      = 170            # display strip height in the GUI
KYMO_THETA_MAX      = np.deg2rad(144)  # ±144° matches Fig 2's colour bar


def _kymo_to_rgba(buf, vmax=KYMO_THETA_MAX, gamma=1.0):
    """Map a tangent-angle buffer (radians) to a yellow/black/cyan RGBA image.

    Parameters
    ----------
    vmax : float
        Tangent angle (radians) that maps to fully saturated yellow/cyan.
        Smaller -> higher contrast for low-amplitude beats.
    gamma : float
        Gamma applied to the normalised magnitude. >1 brightens the dim
        parts of the kymograph (useful for the small angles of the
        symmetric tip-to-base beat); <1 suppresses them.
    """
    if vmax <= 0:
        vmax = 1e-6
    norm = np.clip(buf / vmax, -1.0, 1.0)
    if gamma != 1.0 and gamma > 0:
        norm = np.sign(norm) * (np.abs(norm) ** (1.0 / gamma))
    pos = np.maximum(norm, 0.0)
    neg = np.maximum(-norm, 0.0)
    rgba = np.empty((*buf.shape, 4), dtype=np.float32)
    rgba[..., 0] = pos          # R: positive only -> yellow
    rgba[..., 1] = pos + neg    # G: both
    rgba[..., 2] = neg          # B: negative only -> cyan
    rgba[..., 3] = 1.0
    return rgba


def _update_kymo_buffer(buf, theta_new):
    """Scroll buffer up by one row, write the new tangent-angle profile at
    the bottom. Newest data ends up at row -1, so when rendered as an image
    time increases downward — matching the Wheeler 2020 Fig 2 convention."""
    buf[:-1] = buf[1:]
    n = min(theta_new.shape[0], buf.shape[1])
    buf[-1, :n] = theta_new[:n]
    if n < buf.shape[1]:
        buf[-1, n:] = 0.0


def _slider_tag(field): return f"par_slider__{field}"


# ---------------------------------------------------------------------------
# Parasite construction / serialization helpers
# ---------------------------------------------------------------------------
def _make_parasite(rng, shape):
    p = L.sample_random_parasite(rng, shape)
    p.mode_schedule = L.generate_mode_schedule(p, duration=600.0, rng=rng)
    return p


_PARAMS_VERSION = 2
_REFERENCE_PIXEL_SIZE_UM = 0.325  # for migrating v1 (reference-px) exports

# Spatial fields whose units changed from reference-px to µm in v2.
_V1_TO_V2_PARASITE_LENGTH_FIELDS = (
    "body_length", "body_width", "flagellum_length", "flagellum_width",
    "beat_wavelength", "beat_amplitude_max",
    "tip_to_base_wavelength", "tip_to_base_amplitude",
    "base_to_tip_wavelength", "base_to_tip_amplitude",
    "swim_speed",
    "tip_to_base_swim_speed", "base_to_tip_swim_speed",
)
_V1_TO_V2_OPTICS_RENAMES = {
    "psf_sigma_px":              "psf_sigma_um",
    "halo_lowpass_sigma_px":     "halo_lowpass_sigma_um",
    "body_edge_smooth_sigma_px": "body_edge_smooth_sigma_um",
}


def _migrate_v1_to_v2(data):
    """Convert reference-px (v1) JSON to µm (v2) in-place."""
    px = _REFERENCE_PIXEL_SIZE_UM
    opt = data.get("optics", {})
    for old_name, new_name in _V1_TO_V2_OPTICS_RENAMES.items():
        if old_name in opt and new_name not in opt:
            opt[new_name] = opt.pop(old_name) * px
    for pd in data.get("parasites", []):
        for f in _V1_TO_V2_PARASITE_LENGTH_FIELDS:
            if pd.get(f) is not None:
                pd[f] = pd[f] * px
    data["version"] = _PARAMS_VERSION
    return data


def _serialize(optics, noise, parasites):
    out = {
        "version": _PARAMS_VERSION,
        "optics": dc.asdict(optics),
        "noise": dc.asdict(noise),
        "parasites": [],
    }
    for p in parasites:
        pd = dc.asdict(p)
        if pd.get("mode_schedule") is not None:
            pd["mode_schedule"] = [list(item) for item in pd["mode_schedule"]]
        out["parasites"].append(pd)
    return out


def _coerce_parasite(pd):
    pd = dict(pd)
    if pd.get("mode_schedule"):
        pd["mode_schedule"] = [(float(t), str(m)) for t, m in pd["mode_schedule"]]
    valid = {f.name for f in dc.fields(ParasiteParams)}
    pd = {k: v for k, v in pd.items() if k in valid}
    return ParasiteParams(**pd)


def _deserialize(data, optics, noise, parasites_list):
    if data.get("version", 1) < _PARAMS_VERSION:
        data = _migrate_v1_to_v2(data)
    for k, v in data.get("optics", {}).items():
        if hasattr(optics, k):
            setattr(optics, k, v)
    for k, v in data.get("noise", {}).items():
        if hasattr(noise, k):
            setattr(noise, k, v)
    parasites_list[:] = [_coerce_parasite(pd) for pd in data.get("parasites", [])]


# ---------------------------------------------------------------------------
# Selection-ring overlay
# ---------------------------------------------------------------------------
def _draw_selection_overlay(rgba, parasite, pixel_size_um, color=(1.0, 0.25, 0.25)):
    H, W = rgba.shape[:2]
    cx, cy = int(parasite.center_x), int(parasite.center_y)
    if not (0 <= cx < W and 0 <= cy < H):
        return
    # body_length / flagellum_length are in µm; convert to output pixels so the
    # ring actually encircles the rendered cell (before the v2 µm migration
    # these were reference-px and the ring silently shrank to the cell centre).
    s = 1.0 / max(pixel_size_um, 1e-6)
    radius = int(max(parasite.body_length, parasite.flagellum_length) * s * 0.7) + 4
    # Operate only on a bbox around the ring instead of allocating full-frame
    # index grids and a full-frame distance map every frame.
    r_out = radius + 1
    x0 = max(0, cx - r_out); x1 = min(W, cx + r_out + 1)
    y0 = max(0, cy - r_out); y1 = min(H, cy + r_out + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.indices((y1 - y0, x1 - x0))
    d2 = (xx + x0 - cx) ** 2 + (yy + y0 - cy) ** 2
    ring = (d2 >= (radius - 1) ** 2) & (d2 <= (radius + 1) ** 2)
    sub = rgba[y0:y1, x0:x1]
    sub[ring, 0] = color[0]
    sub[ring, 1] = color[1]
    sub[ring, 2] = color[2]


def _draw_keypoints_overlay(rgba, all_keypoints,
                            node_color=(0.10, 1.0, 0.30),
                            head_color=(1.0, 0.30, 0.30),
                            tip_color=(0.30, 0.55, 1.0),
                            edge_color=(1.0, 0.80, 0.10)):
    """Overlay the SLEAP skeleton for every cell.

    Chains: Head -> Base -> Flag1..FlagN -> Tip, plus the dividing-cell second
    flagellum Base -> Flag2_1..Flag2_N -> Tip2. Occluded / off-image nodes are
    already absent from each dict (``render_scene`` drops them), so the overlay
    shows exactly what SLEAP would treat as visible. ``cv2`` draws non-AA lines
    on the float32 RGBA buffer (AA is 8-bit only)."""
    for kp in all_keypoints:
        if not kp:
            continue
        flag_idx = sorted(int(n[4:]) for n in kp
                          if n.startswith("Flag") and n[4:].isdigit())
        flag2_idx = sorted(int(n[6:]) for n in kp
                           if n.startswith("Flag2_") and n[6:].isdigit())
        chains = (
            ["Head", "Base"] + [f"Flag{k}" for k in flag_idx] + ["Tip"],
            ["Base"] + [f"Flag2_{k}" for k in flag2_idx] + ["Tip2"],
        )
        for names in chains:
            pts = [(int(round(kp[n][0])), int(round(kp[n][1])))
                   for n in names if n in kp]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(rgba, a, b, (*edge_color, 1.0), 1, lineType=cv2.LINE_8)
        for name, (x, y) in kp.items():
            col = (head_color if name == "Head"
                   else tip_color if name in ("Tip", "Tip2")
                   else node_color)
            cv2.circle(rgba, (int(round(x)), int(round(y))), 2,
                       (*col, 1.0), -1, lineType=cv2.LINE_8)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def _build_theme():
    """Subtle theme: tighter padding, rounded corners, slightly darker bg."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     6, 3)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding,      4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,   0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (24, 26, 30))
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(size=512, n_parasites=4, target_fps=60.0, seed=42):
    rng = np.random.default_rng(seed)
    shape = (size, size)
    optics = L.OpticsParams()
    noise  = L.CameraNoiseParams()
    background = L.synthetic_background(shape, rng, intensity=noise.bg_intensity)
    parasites = [_make_parasite(rng, shape) for _ in range(n_parasites)]

    state = {
        "t": 0.0, "paused": False, "mode": "fast",
        "n_parasites": n_parasites, "target_fps": target_fps,
        "selected": 0, "apply_to_all": False, "highlight": True,
        "show_keypoints": True,
        # Playback speed (sim-seconds per wall-second) is decoupled from
        # target_fps; target_fps is now ONLY the render-rate cap. last_frame_time
        # drives the wall-clock dt; frozen_noise_seed stabilises the image while
        # paused so it doesn't shimmer.
        "playback_speed": 0.25, "last_frame_time": None,
        "frozen_noise_seed": 1234,
        "rebuild_bg": False, "reseed_all": False,
        "regen_schedule": False, "clear_schedule": False,
        "division_stage": 0.5,
        # Per-parasite-index snapshot of the cell BEFORE any division-stage
        # modifications. Captured on "split into pair"; division_stage
        # re-derives the cell from this base so it updates in real time.
        "division_bases": {},
        "import_path": None, "export_path": None,
        "flash_msg": "", "flash_until": 0.0,
        "img_display_size": size,
        # Rolling tangent-angle history for the selected parasite, fed to
        # the kymograph widget. Time runs down the rows; distance along
        # the flagellum runs across the columns (base on the left).
        "kymo_buffer": np.zeros((KYMO_HISTORY_FRAMES, KYMO_FLAG_SAMPLES),
                                dtype=np.float32),
        # Kymograph colour mapping (user-adjustable via sliders).
        "kymo_vmax_deg": float(np.rad2deg(KYMO_THETA_MAX)),  # ±deg at full saturation
        "kymo_gamma":    1.0,                                # >1 brightens dim features
        # Resizable layout state. split_x_frac is the fraction of the
        # viewport width given to the left column (preview + kymograph);
        # split_y_frac is the fraction of that column's height given to
        # the main image. The user drags the 4-px gutters to resize.
        "split_x_frac": 0.58,
        "split_y_frac": 0.62,
        "dragging":         None,    # None | "h" | "v"
        "mouse_was_down":   False,
        "splitter_x":       0,       # cached splitter pixel positions (set in _layout)
        "splitter_y":       0,
        "splitter_y_extent": 0,      # how far across the X axis the V-splitter runs
    }

    # Preallocated RGBA display buffer (alpha fixed at 1.0); reused every frame
    # instead of allocating a fresh (size, size, 4) array per render.
    state["tex_buf"] = np.zeros((size, size, 4), dtype=np.float32)
    state["tex_buf"][..., 3] = 1.0

    def flash(msg, dur=3.0):
        state["flash_msg"] = msg
        state["flash_until"] = time.perf_counter() + dur
        print(msg)

    def refresh_parasite_sliders(parasite):
        """Push selected parasite's values into the per-parasite widgets."""
        for fields in PARASITE_GROUPS.values():
            for field, _, _ in fields:
                tag = _slider_tag(field)
                if not dpg.does_item_exist(tag):
                    continue
                v = getattr(parasite, field, None)
                if v is None:
                    continue
                dpg.set_value(tag, v)
        if dpg.does_item_exist("par_beat_mode"):
            dpg.set_value("par_beat_mode", parasite.beat_mode)

    def _apply_division(idx, save_base=False):
        """(Re)compute parasites[idx] as a dividing cell from a stored base.

        save_base=True (used by "split selected into pair"): snapshot the
        current cell as the base, stripping any prior dividing fields so
        re-applying division doesn't compound. The base is the cell as it
        was *before* any division-stage transformation.

        save_base=False (used by the live division sliders): re-derive from
        the existing base. If no base exists yet (the cell isn't a divider),
        nothing happens — the slider value is just stored for the next
        button press.
        """
        if not (0 <= idx < len(parasites)):
            return
        bases = state["division_bases"]
        # Only snapshot a base if none exists yet — otherwise pressing the
        # button repeatedly would rebase on the already-widened cell and
        # compound the division (body_width 2.0 -> 3.4 -> 5.78 -> ...).
        if save_base and idx not in bases:
            bases[idx] = dc.replace(
                parasites[idx],
                is_dividing_daughter=False,
                second_flagellum_enabled=False,
                second_flagellum_length_scale=1.0,
                second_flagellum_lateral_offset=0.0,
                second_flagellum_angle_offset=0.0,
                second_flagellum_phase_offset=0.0,
            )
        if idx not in bases:
            return
        pair = L.make_dividing_pair(
            bases[idx],
            division_stage=state["division_stage"],
            rng=rng,
        )
        parasites[idx] = pair[0]
        # Mirror the new body_width / second_flagellum_* / organelle values
        # into the per-field sliders so the user sees them update live.
        if state["selected"] == idx:
            refresh_parasite_sliders(parasites[idx])

    def resync_selector():
        items = [f"Parasite {i}" for i in range(max(len(parasites), 1))]
        if not parasites:
            items = ["(none)"]
        dpg.configure_item("par_selector", items=items)
        if state["selected"] >= len(parasites):
            state["selected"] = max(0, len(parasites) - 1)
        if parasites:
            dpg.set_value("par_selector", items[state["selected"]])
            refresh_parasite_sliders(parasites[state["selected"]])
        dpg.set_value("n_parasites_slider", state["n_parasites"])

    # --- DPG setup ---
    dpg.create_context()

    init = np.zeros(size * size * 4, dtype=np.float32)
    init[3::4] = 1.0
    with dpg.texture_registry():
        dpg.add_raw_texture(width=size, height=size, default_value=init,
                            tag="frame_tex", format=dpg.mvFormat_Float_rgba)

        # Kymograph texture (small native size; scaled up at display time).
        kymo_init = np.zeros(KYMO_HISTORY_FRAMES * KYMO_FLAG_SAMPLES * 4, dtype=np.float32)
        kymo_init[3::4] = 1.0
        dpg.add_raw_texture(width=KYMO_FLAG_SAMPLES, height=KYMO_HISTORY_FRAMES,
                            default_value=kymo_init,
                            tag="kymo_tex",
                            format=dpg.mvFormat_Float_rgba)

    with dpg.file_dialog(directory_selector=False, show=False, tag="dlg_export",
                         callback=lambda s, a: state.update(export_path=a["file_path_name"]),
                         width=700, height=400):
        dpg.add_file_extension(".json")
        dpg.add_file_extension(".*")
    with dpg.file_dialog(directory_selector=False, show=False, tag="dlg_import",
                         callback=lambda s, a: state.update(import_path=a["file_path_name"]),
                         width=700, height=400):
        dpg.add_file_extension(".json")
        dpg.add_file_extension(".*")

    # =========================================================================
    # Top-level windows. Three resizable panes:
    #   preview_win  - main rendered image (top-left)
    #   kymo_win     - tangent-angle kymograph (bottom-left)
    #   ctrl_win     - tabbed controls (right column)
    # plus a thin status bar at the bottom. The 4-px gutters between them
    # are draggable splitters; see the mouse-handling block in the main
    # render loop.
    # =========================================================================
    win_flags = dict(no_close=True, no_collapse=True, no_move=True,
                     no_resize=True, no_title_bar=True)
    with dpg.window(tag="preview_win", pos=(0, 0), **win_flags):
        with dpg.child_window(tag="preview_pane", border=False,
                              autosize_x=True, autosize_y=True):
            dpg.add_image("frame_tex", tag="frame_image",
                          width=size, height=size)

    with dpg.window(tag="kymo_win", pos=(0, size), **win_flags):
        with dpg.group(horizontal=True):
            dpg.add_text("Tangent-angle kymograph (selected parasite)",
                         color=(180, 180, 180))
            dpg.add_text("yellow = +,  cyan = -,  black = 0",
                         color=(140, 140, 140))
        dpg.add_image("kymo_tex", tag="kymo_image",
                      width=size, height=KYMO_HEIGHT_PX)

    with dpg.window(tag="status_win", pos=(0, 0), **win_flags):
        dpg.add_text("", tag="status")

    # =========================================================================
    # Controls window (tabbed)
    # =========================================================================
    with dpg.window(tag="ctrl_win", pos=(size + 30, 0), **win_flags):
        with dpg.tab_bar(tag="main_tabs"):

            # ----------------------------- Scene -----------------------------
            with dpg.tab(label="Scene"):
                dpg.add_combo(RENDER_MODES, label="render mode",
                              default_value="fast",
                              callback=lambda s, a: state.update(mode=a))
                dpg.add_slider_int(label="parasites", tag="n_parasites_slider",
                                   default_value=n_parasites,
                                   min_value=0, max_value=64,
                                   callback=lambda s, a: state.update(n_parasites=a))
                dpg.add_slider_float(label="max fps (smoothness)",
                                     default_value=target_fps,
                                     min_value=1.0, max_value=240.0,
                                     format="%.0f",
                                     callback=lambda s, a: state.update(target_fps=a))
                dpg.add_slider_float(label="playback speed",
                                     tag="playback_speed_slider",
                                     default_value=state["playback_speed"],
                                     min_value=0.02, max_value=2.0,
                                     format="%.2fx",
                                     callback=lambda s, a: state.update(playback_speed=a))
                dpg.add_text("  max fps caps the render rate only; playback speed\n"
                             "  sets how fast the biology evolves (1.0 = real time)",
                             color=(140, 140, 140))
                dpg.add_checkbox(label="paused", tag="cb_paused",
                                 callback=lambda s, a: state.update(paused=a))
                dpg.add_checkbox(label="highlight selected",
                                 default_value=True,
                                 callback=lambda s, a: state.update(highlight=a))
                dpg.add_checkbox(label="show keypoints", tag="cb_show_kp",
                                 default_value=True,
                                 callback=lambda s, a: state.update(show_keypoints=a))

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="reseed all",
                                   callback=lambda: state.update(reseed_all=True))
                    dpg.add_button(label="rebuild bg",
                                   callback=lambda: state.update(rebuild_bg=True))

                dpg.add_separator()
                dpg.add_text("Export / Import (JSON)")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Export...",
                                   callback=lambda: dpg.show_item("dlg_export"))
                    dpg.add_button(label="Import...",
                                   callback=lambda: dpg.show_item("dlg_import"))

                dpg.add_separator()
                dpg.add_text("Kymograph display", color=(180, 180, 180))
                dpg.add_slider_float(
                    label="vmax (deg)", tag="kymo_vmax_slider",
                    default_value=state["kymo_vmax_deg"],
                    min_value=10.0, max_value=360.0, format="%.0f",
                    callback=lambda s, a: state.update(kymo_vmax_deg=a))
                dpg.add_slider_float(
                    label="gamma", tag="kymo_gamma_slider",
                    default_value=state["kymo_gamma"],
                    min_value=0.3, max_value=3.0, format="%.2f",
                    callback=lambda s, a: state.update(kymo_gamma=a))
                dpg.add_text("  vmax: tangent angle (deg) at full saturation",
                             color=(140, 140, 140))
                dpg.add_text("  gamma: >1 brightens dim features",
                             color=(140, 140, 140))

                def _reset_kymo_display():
                    state["kymo_vmax_deg"] = 144.0
                    state["kymo_gamma"]    = 1.0
                    if dpg.does_item_exist("kymo_vmax_slider"):
                        dpg.set_value("kymo_vmax_slider", 144.0)
                    if dpg.does_item_exist("kymo_gamma_slider"):
                        dpg.set_value("kymo_gamma_slider", 1.0)

                dpg.add_button(label="reset (144 deg, gamma 1.0)",
                               callback=_reset_kymo_display)

                dpg.add_separator()
                dpg.add_text("Keyboard shortcuts", color=(160, 160, 160))
                dpg.add_text("  Space  - pause / resume", color=(140, 140, 140))
                dpg.add_text("  R      - reseed all parasites", color=(140, 140, 140))
                dpg.add_text("  B      - rebuild background", color=(140, 140, 140))
                dpg.add_text("Drag the 4-px gutters between panes to resize.",
                             color=(140, 140, 140))

            # -------------------------- Optics & Noise ------------------------
            with dpg.tab(label="Optics & Noise"):
                with dpg.collapsing_header(label="Optics", default_open=True):
                    for field, _, (lo, hi) in OPTICS_RANGES:
                        dpg.add_slider_float(
                            label=field,
                            default_value=float(getattr(optics, field)),
                            min_value=float(lo), max_value=float(hi),
                            format="%.3g",
                            user_data=(optics, field),
                            callback=lambda s, a, u: setattr(u[0], u[1], a))

                with dpg.collapsing_header(label="Camera noise", default_open=True):
                    for field, _, (lo, hi) in NOISE_RANGES:
                        dpg.add_slider_float(
                            label=field,
                            default_value=float(getattr(noise, field)),
                            min_value=float(lo), max_value=float(hi),
                            format="%.3g",
                            user_data=(noise, field),
                            callback=lambda s, a, u: setattr(u[0], u[1], a))

            # ----------------------------- Parasite ---------------------------
            with dpg.tab(label="Parasite"):

                def _on_select(s, a):
                    try:
                        idx = int(a.split()[1])
                    except (ValueError, IndexError):
                        return
                    if idx != state["selected"]:
                        state["kymo_buffer"][:] = 0.0  # discard previous cell's history
                    state["selected"] = idx
                    if 0 <= idx < len(parasites):
                        refresh_parasite_sliders(parasites[idx])

                sel_items = [f"Parasite {i}" for i in range(max(n_parasites, 1))]
                dpg.add_combo(sel_items, label="selected", tag="par_selector",
                              default_value=sel_items[0] if sel_items else "",
                              callback=_on_select)
                dpg.add_checkbox(label="apply changes to ALL parasites",
                                 callback=lambda s, a: state.update(apply_to_all=a))

                dpg.add_separator()

                def par_cb(s, a, u):
                    field = u
                    if state["apply_to_all"]:
                        for p in parasites:
                            if hasattr(p, field):
                                setattr(p, field, a)
                    else:
                        idx = state["selected"]
                        if 0 <= idx < len(parasites) and hasattr(parasites[idx], field):
                            setattr(parasites[idx], field, a)

                with dpg.collapsing_header(label="Beat mode", default_open=True):
                    def _on_beat_mode(s, a):
                        if state["apply_to_all"]:
                            for p in parasites:
                                p.beat_mode = a
                        else:
                            idx = state["selected"]
                            if 0 <= idx < len(parasites):
                                parasites[idx].beat_mode = a
                    init_mode = parasites[0].beat_mode if parasites else "tip_to_base"
                    dpg.add_combo(BEAT_MODES, label="beat_mode",
                                  tag="par_beat_mode",
                                  default_value=init_mode, callback=_on_beat_mode)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="regen schedule",
                                       callback=lambda: state.update(regen_schedule=True))
                        dpg.add_button(label="clear schedule",
                                       callback=lambda: state.update(clear_schedule=True))
                    dpg.add_text("(regen uses current mode_switch_rate)",
                                 color=(160, 160, 160))

                for group_name, fields in PARASITE_GROUPS.items():
                    with dpg.collapsing_header(label=group_name):
                        for field, typ, (lo, hi) in fields:
                            init_val = (getattr(parasites[0], field, 0.0)
                                        if parasites else 0.0)
                            if init_val is None:
                                init_val = 0.0
                            if typ is int:
                                dpg.add_slider_int(
                                    label=field, tag=_slider_tag(field),
                                    default_value=int(init_val),
                                    min_value=int(lo), max_value=int(hi),
                                    user_data=field, callback=par_cb)
                            else:
                                dpg.add_slider_float(
                                    label=field, tag=_slider_tag(field),
                                    default_value=float(init_val),
                                    min_value=float(lo), max_value=float(hi),
                                    format="%.3g",
                                    user_data=field, callback=par_cb)

                # Dividing cells: model the dominant pre-cytokinesis stage
                # (one rounder body, two flagella from the same anterior end;
                # Wheeler 2011, 2013). Press "split selected into pair" to
                # snapshot the current cell as a base, then the division_stage
                # slider live-modifies the cell from that base. Fine control
                # of body width / bend / nF length etc. lives in the per-field
                # sliders above (Body / Second flagellum / Body density).
                with dpg.collapsing_header(label="Dividing cells"):
                    def _on_stage(sender, app_data):
                        state["division_stage"] = app_data
                        idx = state["selected"]
                        if idx in state["division_bases"]:
                            _apply_division(idx, save_base=False)
                    dpg.add_slider_float(
                        label="division_stage",
                        default_value=state["division_stage"],
                        min_value=0.0, max_value=1.0, format="%.2f",
                        callback=_on_stage)
                    dpg.add_button(
                        label="split selected into pair",
                        callback=lambda: _apply_division(state["selected"],
                                                        save_base=True))
                    dpg.add_text("(button snapshots current params as base;"
                                 " division_stage then live-modifies from it.",
                                 color=(160, 160, 160))
                    dpg.add_text(" Fine control: use Body / Second flagellum"
                                 " / Body density sliders above.)",
                                 color=(160, 160, 160))

    # --- viewport, theme, layout, keys ---
    dpg.create_viewport(title="Leishmania live preview",
                        width=size + 700, height=size + 140,
                        min_width=720, min_height=440)

    theme = _build_theme()
    dpg.bind_theme(theme)

    GUTTER    = 4       # px between panes (the draggable splitter zone)
    HIT_TOL   = 6       # px tolerance around a splitter line that counts as a grab
    STATUS_H  = 26      # px

    def _layout():
        """Re-layout the three panes + status bar from state-driven fractions.
        Also caches the splitter line positions so the mouse handler in the
        render loop can hit-test them."""
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()

        # Horizontal split: left column (preview + kymo) vs right column (controls).
        preview_col_w = int(vw * state["split_x_frac"])
        preview_col_w = max(260, min(preview_col_w, vw - 300))
        ctrl_w        = max(280, vw - preview_col_w - GUTTER)

        # Vertical split inside the left column: main image vs kymograph.
        usable_h = max(200, vh - STATUS_H - GUTTER)
        main_h   = int(usable_h * state["split_y_frac"])
        main_h   = max(140, min(main_h, usable_h - 120))
        kymo_h   = max(80, usable_h - main_h - GUTTER)

        dpg.configure_item("preview_win", pos=(0, 0),
                           width=preview_col_w, height=main_h)
        dpg.configure_item("kymo_win",    pos=(0, main_h + GUTTER),
                           width=preview_col_w, height=kymo_h)
        dpg.configure_item("ctrl_win",    pos=(preview_col_w + GUTTER, 0),
                           width=ctrl_w, height=vh - STATUS_H)
        dpg.configure_item("status_win",  pos=(0, vh - STATUS_H),
                           width=vw, height=STATUS_H)

        # Resize the image widgets inside the panes.
        main_avail_w = preview_col_w - 24
        main_avail_h = main_h - 24
        main_size    = max(64, min(main_avail_w, main_avail_h))
        if dpg.does_item_exist("frame_image"):
            dpg.configure_item("frame_image", width=main_size, height=main_size)

        kymo_disp_w = max(64, preview_col_w - 24)
        kymo_disp_h = max(48, kymo_h - 48)   # leave room for the title row
        if dpg.does_item_exist("kymo_image"):
            dpg.configure_item("kymo_image", width=kymo_disp_w, height=kymo_disp_h)

        # Cache splitter positions for hit testing.
        state["img_display_size"]   = main_size
        state["splitter_x"]         = preview_col_w + GUTTER // 2     # H-splitter (full height)
        state["splitter_y"]         = main_h + GUTTER // 2            # V-splitter (left column only)
        state["splitter_y_extent"]  = preview_col_w                   # H-extent of V-splitter

    dpg.set_viewport_resize_callback(_layout)

    # Keyboard shortcuts (use getattr for version portability)
    KEY_SPACE = getattr(dpg, "mvKey_Spacebar", 32)
    KEY_R     = getattr(dpg, "mvKey_R", 82)
    KEY_B     = getattr(dpg, "mvKey_B", 66)

    def _toggle_pause():
        state["paused"] = not state["paused"]
        if dpg.does_item_exist("cb_paused"):
            dpg.set_value("cb_paused", state["paused"])

    with dpg.handler_registry():
        dpg.add_key_press_handler(KEY_SPACE, callback=_toggle_pause)
        dpg.add_key_press_handler(KEY_R,
                                  callback=lambda: state.update(reseed_all=True))
        dpg.add_key_press_handler(KEY_B,
                                  callback=lambda: state.update(rebuild_bg=True))

    dpg.setup_dearpygui()
    dpg.show_viewport()
    _layout()  # initial layout pass

    def _check_splitter_hit(mx, my):
        """Return 'h', 'v', or None if (mx, my) lies on a splitter line."""
        sx = state["splitter_x"]
        sy = state["splitter_y"]
        sy_extent = state["splitter_y_extent"]
        vh = dpg.get_viewport_client_height()
        # Horizontal splitter (vertical line at x = sx; full height)
        if abs(mx - sx) <= HIT_TOL and 0 <= my <= vh - STATUS_H:
            return "h"
        # Vertical splitter (horizontal line at y = sy; only across the left column)
        if abs(my - sy) <= HIT_TOL and 0 <= mx <= sy_extent:
            return "v"
        return None

    # --- Render loop ---
    frame_times = []
    while dpg.is_dearpygui_running():
        loop_start = time.perf_counter()

        # ---- Splitter drag: poll mouse state each frame (no widget conflict
        # because the splitter zones are 4-px gaps with no widgets in them).
        mouse_down = dpg.is_mouse_button_down(0)
        mx, my     = dpg.get_mouse_pos(local=False)

        if mouse_down and not state["mouse_was_down"]:
            # Mouse just went down: see if the press landed on a splitter.
            state["dragging"] = _check_splitter_hit(mx, my)

        if mouse_down and state["dragging"]:
            vw_now = dpg.get_viewport_client_width()
            vh_now = dpg.get_viewport_client_height()
            if state["dragging"] == "h":
                state["split_x_frac"] = max(0.18, min(0.85, mx / max(vw_now, 1)))
                _layout()
            elif state["dragging"] == "v":
                usable_h = max(1, vh_now - STATUS_H - GUTTER)
                state["split_y_frac"] = max(0.15, min(0.85, my / usable_h))
                _layout()

        if not mouse_down:
            state["dragging"] = None
        state["mouse_was_down"] = mouse_down

        # Structural changes
        if state["reseed_all"]:
            parasites[:] = [_make_parasite(rng, shape)
                            for _ in range(state["n_parasites"])]
            state["reseed_all"] = False
            state["kymo_buffer"][:] = 0.0
            state["division_bases"].clear()
            resync_selector()
        else:
            changed = False
            while len(parasites) < state["n_parasites"]:
                parasites.append(_make_parasite(rng, shape))
                changed = True
            if len(parasites) > state["n_parasites"]:
                del parasites[state["n_parasites"]:]
                # Drop stale division-base snapshots for removed indices.
                for k in [k for k in state["division_bases"]
                          if k >= state["n_parasites"]]:
                    del state["division_bases"][k]
                changed = True
            if changed:
                state["kymo_buffer"][:] = 0.0
                resync_selector()

        if state["rebuild_bg"]:
            background = L.synthetic_background(shape, rng,
                                                intensity=noise.bg_intensity)
            state["rebuild_bg"] = False

        if state["regen_schedule"]:
            idx = state["selected"]
            targets = parasites if state["apply_to_all"] else (
                [parasites[idx]] if 0 <= idx < len(parasites) else [])
            for p in targets:
                p.mode_schedule = L.generate_mode_schedule(p, 600.0, rng)
            state["regen_schedule"] = False

        if state["clear_schedule"]:
            idx = state["selected"]
            targets = parasites if state["apply_to_all"] else (
                [parasites[idx]] if 0 <= idx < len(parasites) else [])
            for p in targets:
                p.mode_schedule = []
            state["clear_schedule"] = False

        if state["export_path"]:
            try:
                with open(state["export_path"], "w") as f:
                    json.dump(_serialize(optics, noise, parasites), f, indent=2)
                flash(f"Exported to {state['export_path']}")
            except Exception as e:
                flash(f"Export failed: {e}")
            state["export_path"] = None

        if state["import_path"]:
            try:
                with open(state["import_path"]) as f:
                    data = json.load(f)
                _deserialize(data, optics, noise, parasites)
                state["n_parasites"] = len(parasites)
                background = L.synthetic_background(shape, rng,
                                                    intensity=noise.bg_intensity)
                state["kymo_buffer"][:] = 0.0
                resync_selector()
                flash(f"Imported {state['import_path']}")
            except Exception as e:
                flash(f"Import failed: {e}")
            state["import_path"] = None

        # --- Simulation timestep -------------------------------------------
        # Decoupled from the render-rate cap: dt is the REAL wall-clock time
        # since the last frame, times the playback-speed multiplier. So
        # "max fps" only controls smoothness (how many frames we draw), while
        # "playback speed" controls how fast the biology evolves (1.0 = real
        # time). wall_dt is clamped so a stall or a long pause doesn't make the
        # next step jump.
        now0 = time.perf_counter()
        if state["last_frame_time"] is None:
            wall_dt = 1.0 / max(state["target_fps"], 1.0)
        else:
            wall_dt = now0 - state["last_frame_time"]
        state["last_frame_time"] = now0
        wall_dt = min(max(wall_dt, 0.0), 0.1)
        sim_dt = wall_dt * max(state["playback_speed"], 0.0)

        paused = state["paused"]

        # Advance motion + beat only while running.
        if not paused:
            L.advance_parasites(parasites, sim_dt, shape, periodic=True,
                                t=state["t"], optics=optics)
            state["t"] += sim_dt

        # Render EVERY frame (even when paused) so parameter edits show
        # immediately. While paused, use a fixed-seed rng so the camera noise
        # is frozen instead of shimmering between otherwise-identical frames.
        render_rng = (rng if not paused
                      else np.random.default_rng(state["frozen_noise_seed"]))
        frame_keypoints = []
        try:
            mode = state["mode"]
            if mode == "skip_noise":
                phase = np.zeros(shape, dtype=np.float32)
                transmission = np.ones(shape, dtype=np.float32)
                for p in parasites:
                    tile, (y0, x0), kp, transm = L.render_parasite_phase(
                        p, state["t"], shape, optics=optics)
                    frame_keypoints.append(kp)
                    if tile is None:
                        continue
                    th, tw = tile.shape
                    np.maximum(phase[y0:y0+th, x0:x0+tw], tile,
                               out=phase[y0:y0+th, x0:x0+tw])
                    if transm is not None:
                        ar = transmission[y0:y0+th, x0:x0+tw]
                        np.multiply(ar, transm, out=ar)
                intensity = L.simulate_phase_contrast_fast(phase, optics)
                img = np.clip(background * intensity * transmission, 0, 1)
            else:
                img, frame_keypoints = L.render_scene(
                    parasites, state["t"], shape, optics, noise,
                    background=background, rng=render_rng,
                    fast=(mode == "fast"))
        except Exception as e:
            flash(f"Render error: {e}")
            img = np.zeros(shape, dtype=np.float32)
            frame_keypoints = []

        # --- Kymograph: sample the selected cell's tangent angle at the SAME
        # t the frame was rendered (positions are advanced BEFORE rendering
        # above, so this is now aligned — the old code sampled one dt late),
        # and scroll the history only while running.
        if not paused and parasites:
            sel = min(state["selected"], len(parasites) - 1)
            try:
                _, theta = L.beat_tangent_angle(parasites[sel], state["t"],
                                               n_points=KYMO_FLAG_SAMPLES)
                _update_kymo_buffer(state["kymo_buffer"], theta.astype(np.float32))
            except Exception:
                pass
        kymo_rgba = _kymo_to_rgba(
            state["kymo_buffer"],
            vmax=np.deg2rad(state["kymo_vmax_deg"]),
            gamma=state["kymo_gamma"],
        )
        dpg.set_value("kymo_tex", kymo_rgba.ravel())

        # --- Compose the display texture (grayscale -> RGBA) in a reused
        # buffer; alpha stays 1.0 from initialisation.
        tex = state["tex_buf"]
        tex[..., 0] = img
        tex[..., 1] = img
        tex[..., 2] = img
        if state["show_keypoints"]:
            _draw_keypoints_overlay(tex, frame_keypoints)
        if state["highlight"] and parasites:
            idx = min(state["selected"], len(parasites) - 1)
            _draw_selection_overlay(tex, parasites[idx], optics.pixel_size_um)
        dpg.set_value("frame_tex", tex.ravel())

        # Status line
        now = time.perf_counter()
        frame_times.append(now)
        frame_times = [ft for ft in frame_times if now - ft < 1.0]
        fps = len(frame_times)
        msg = (f"{fps} fps   t={state['t']:.1f}s   mode={state['mode']}"
               f"   sel=#{state['selected']}"
               f"{'   PAUSED' if state['paused'] else ''}"
               f"{'   APPLY-ALL' if state['apply_to_all'] else ''}")
        if now < state["flash_until"]:
            msg += "   |   " + state["flash_msg"]
        dpg.set_value("status", msg)

        # Throttle
        target_dt = 1.0 / max(state["target_fps"], 1.0)
        elapsed = time.perf_counter() - loop_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--fps", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(size=args.size, n_parasites=args.n, target_fps=args.fps, seed=args.seed)