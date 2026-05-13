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
    "Pose": [
        ("center_x",         float, (0.0, 4096.0)),  # image px
        ("center_y",         float, (0.0, 4096.0)),  # image px
        ("body_orientation", float, (-2 * np.pi, 2 * np.pi)),
    ],
}

BEAT_MODES   = ["tip_to_base", "base_to_tip", "static"]
RENDER_MODES = ["fast", "accurate", "skip_noise"]


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
def _draw_selection_overlay(rgba, parasite, color=(1.0, 0.25, 0.25)):
    H, W = rgba.shape[:2]
    cx, cy = int(parasite.center_x), int(parasite.center_y)
    if not (0 <= cx < W and 0 <= cy < H):
        return
    radius = int(max(parasite.body_length, parasite.flagellum_length) * 0.7) + 4
    yy, xx = np.indices((H, W))
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (d2 >= (radius - 1) ** 2) & (d2 <= (radius + 1) ** 2)
    rgba[ring, 0] = color[0]
    rgba[ring, 1] = color[1]
    rgba[ring, 2] = color[2]


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
        "rebuild_bg": False, "reseed_all": False,
        "regen_schedule": False, "clear_schedule": False,
        "import_path": None, "export_path": None,
        "flash_msg": "", "flash_until": 0.0,
        "img_display_size": size,
    }

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
    # Preview window (image + status bar)
    # =========================================================================
    win_flags = dict(no_close=True, no_collapse=True, no_move=True,
                     no_resize=True, no_title_bar=True)
    with dpg.window(tag="preview_win", pos=(0, 0), **win_flags):
        # Image sits in a child window so it can be centered/scrolled if needed
        with dpg.child_window(tag="preview_pane", border=False,
                              height=-30, autosize_x=True):
            dpg.add_image("frame_tex", tag="frame_image",
                          width=size, height=size)
        dpg.add_separator()
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
                dpg.add_slider_float(label="target fps",
                                     default_value=target_fps,
                                     min_value=1.0, max_value=240.0,
                                     format="%.0f",
                                     callback=lambda s, a: state.update(target_fps=a))
                dpg.add_checkbox(label="paused", tag="cb_paused",
                                 callback=lambda s, a: state.update(paused=a))
                dpg.add_checkbox(label="highlight selected",
                                 default_value=True,
                                 callback=lambda s, a: state.update(highlight=a))

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
                dpg.add_text("Keyboard shortcuts", color=(160, 160, 160))
                dpg.add_text("  Space  - pause / resume", color=(140, 140, 140))
                dpg.add_text("  R      - reseed all parasites", color=(140, 140, 140))
                dpg.add_text("  B      - rebuild background", color=(140, 140, 140))

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

    # --- viewport, theme, layout, keys ---
    dpg.create_viewport(title="Leishmania live preview",
                        width=size + 500, height=size + 100,
                        min_width=600, min_height=400)

    theme = _build_theme()
    dpg.bind_theme(theme)

    def _layout():
        """Re-layout both windows + rescale the rendered image to fit."""
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()

        ctrl_w   = max(360, min(500, int(vw * 0.40)))
        preview_w = max(220, vw - ctrl_w)

        dpg.configure_item("preview_win", pos=(0, 0),
                           width=preview_w, height=vh)
        dpg.configure_item("ctrl_win", pos=(preview_w, 0),
                           width=ctrl_w, height=vh)

        # Fit image into the preview pane (square, aspect preserved).
        # Reserve a small margin for window padding + the status row.
        avail_w = preview_w - 24
        avail_h = vh - 70
        avail = max(64, min(avail_w, avail_h))
        if dpg.does_item_exist("frame_image"):
            dpg.configure_item("frame_image", width=avail, height=avail)
        state["img_display_size"] = avail

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

    # --- Render loop ---
    frame_times = []
    while dpg.is_dearpygui_running():
        loop_start = time.perf_counter()

        # Structural changes
        if state["reseed_all"]:
            parasites[:] = [_make_parasite(rng, shape)
                            for _ in range(state["n_parasites"])]
            state["reseed_all"] = False
            resync_selector()
        else:
            changed = False
            while len(parasites) < state["n_parasites"]:
                parasites.append(_make_parasite(rng, shape))
                changed = True
            if len(parasites) > state["n_parasites"]:
                del parasites[state["n_parasites"]:]
                changed = True
            if changed:
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
                resync_selector()
                flash(f"Imported {state['import_path']}")
            except Exception as e:
                flash(f"Import failed: {e}")
            state["import_path"] = None

        dt = 1.0 / max(state["target_fps"], 1.0)

        # Render
        if not state["paused"]:
            mode = state["mode"]
            if mode == "skip_noise":
                phase = np.zeros(shape, dtype=np.float32)
                for p in parasites:
                    tile, (y0, x0), _ = L.render_parasite_phase(
                        p, state["t"], shape, optics=optics)
                    if tile is None:
                        continue
                    th, tw = tile.shape
                    np.maximum(phase[y0:y0+th, x0:x0+tw], tile,
                               out=phase[y0:y0+th, x0:x0+tw])
                intensity = L.simulate_phase_contrast_fast(phase, optics)
                img = np.clip(background * intensity, 0, 1)
            else:
                img, _ = L.render_scene(
                    parasites, state["t"], shape, optics, noise,
                    background=background, rng=rng,
                    fast=(mode == "fast"))
            L.advance_parasites(parasites, dt, shape, periodic=True,
                                t=state["t"], optics=optics)
            state["t"] += dt

            tex = np.empty((size, size, 4), dtype=np.float32)
            tex[..., 0] = img
            tex[..., 1] = img
            tex[..., 2] = img
            tex[..., 3] = 1.0
            if state["highlight"] and parasites:
                idx = min(state["selected"], len(parasites) - 1)
                _draw_selection_overlay(tex, parasites[idx])
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
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(size=args.size, n_parasites=args.n, target_fps=args.fps, seed=args.seed)