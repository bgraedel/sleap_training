# Synthetic Leishmania Dataset for SLEAP

Generate a large synthetic phase-contrast dataset of Leishmania promastigotes
for pretraining a SLEAP pose model, then finetune on real data.

## Files

| File                          | Purpose                                                                       |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `synthetic_leishmania.py`     | Core simulator: parasite morphology, beat patterns, phase-contrast optics, camera noise. Imported by everything else. |
| `dataset_builder.py`          | CLI for generating SLEAP-format datasets in random and video modes; multi-setup YAML support. |
| `live_demo_gui.py`            | Dear PyGui live preview — sliders for every parameter + per-parasite editor. Use this to tune the simulator against your real recordings before generating a big dataset. |
| `sleap_pretrain_config.yaml`  | Multi-setup config for the full pretraining dataset (~14k frames, 64 datasets).                |
| `example_config.yaml`         | Minimal multi-setup template, useful as a starting point for custom configs.   |

## Install

```bash
pip install numpy opencv-python scipy pillow imageio pyyaml sleap-io
pip install dearpygui   # for the GUI only
```

## Quick start

**1. Tune the simulator against your real data (optional but recommended):**

```bash
python live_demo_gui.py
```

Open one of your real frames in your image viewer next to the GUI. Adjust optics
(`pixel_size_um`, `psf_sigma_um`, `halo_strength`, `intensity_gain`,
`shadeoff_threshold`) and noise (`bg_intensity`, `full_well_photons`,
`read_noise_e`) until the synthetic image looks like your real images. Export as
JSON if you want to save the tuned settings, then feed those values into
`sleap_pretrain_config.yaml` defaults.

**2. Generate the full pretraining dataset:**

```bash
python dataset_builder.py multi sleap_pretrain_config.yaml
```

Output: `data/leishmania_pretrain/` with 64 sub-datasets:

```
data/leishmania_pretrain/
├── config_resolved.yaml          # full audit trail
├── 20x_main/{video.tif, labels.slp, ground_truth.json}
├── 60x_main/...
├── 20x_dense/...
├── 60x_dense/...
├── 20x_very_dense/...
├── 60x_very_dense/...
├── 20x_dimmer/...
├── 60x_dimmer/...
├── 20x_low_snr/...
├── 60x_low_snr/...
├── 20x_defocused/...
├── 60x_defocused/...
├── 40x_intermediate/...
├── 100x_detail/...
├── 20x_short_clips/clip_000/...   # 25 short tracking clips
├── 20x_short_clips/clip_001/...
├── ... (clip_024)
└── 60x_short_clips/clip_000/...   # 25 more
```

**3. Train on synthetic:**

```bash
# Use SLEAP's training CLI with all labels.slp files merged
sleap-train baseline_medium_rf.json \
    data/leishmania_pretrain/**/labels.slp
```

**4. Finetune on your real data:**

Either continue from the synthetic-pretrained checkpoint with your 600 real
instances mixed with ~10–20% synthetic frames sampled from the random pools, or
just point SLEAP at the real labels with the synthetic checkpoint as init.

## What's in the synthetic distribution

| Axis                      | Range                                                  |
| ------------------------- | ------------------------------------------------------ |
| Magnification             | 20× (45%), 60× (49%), 40× (6%), 100× (4%)              |
| Cells per frame           | 1 – 200 (very_dense setups have 100+ cells)             |
| Background intensity      | 0.18 – 0.95 (within physically realistic envelope)      |
| Photon budget (full_well) | 150 – 8000 photons                                      |
| Read noise                | 1.0 – 12 e⁻                                             |
| PSF (focus quality)       | 0.06 – 0.50 µm sigma                                    |
| Halo brightness           | 0.21 – 2.2× base (subtle to dramatic glow)              |
| Halo sharpness            | tight 2 µm sigma to diffuse 11 µm sigma                  |
| Body length               | 8 – 18 µm (procyclic to nectomonad)                     |
| Body width                | 1.2 – 3.5 µm                                            |
| Flagellum length          | 5 – 30 µm                                               |
| Body opacity (per-cell)   | 1.2 – 7.0 phase shift (~23× contrast range)             |
| Beat frequency            | 15 – 28 Hz tip-to-base, 2 – 8 Hz base-to-tip           |
| Tip-to-base envelope      | sin(πs)^p, exponent 0.5 – 2.0 (peak in the middle, tip held nearly static in tangent angle) |
| Static curl (recovery)    | base_to_tip_static_curl 0.65 – 1.45 (covers WT + aggressive-curl mutants) |
| Static curl shape         | linear arc (1.0) → distal hook (2.5)                    |
| Temporal asymmetry        | 0.0 – 0.30 (0 = symmetric sinusoid; >0 = faster power stroke) |
| Wave propagation          | 85% reach tip (extent 0.95 – 1.20); 15% partial (extent 0.55 – 0.85) |
| Paralysed cells           | ~5% of population (zero motility)                       |

Short tracking clips: 50 separate 1.5 s sequences (25 at 20×, 25 at 60×) with
persistent identities and per-clip optics drift.

## Skeleton

Default skeleton: `Head → Base → Flag1 → Flag2 → Flag3 → Flag4 → Flag5 → Tip`
(8 nodes, 5 of them named "Flag*"). Set in YAML via `flag_keypoints: 5`.

To change: edit `flag_keypoints: N` in `sleap_pretrain_config.yaml` defaults.
The skeleton will become `Head, Base, Flag1...FlagN, Tip` (N+3 nodes).

## Compute

Generating the full 14k frames takes several hours on a single CPU; the
very_dense setups dominate (200 cells per 768×768 frame). To speed things up:

- Reduce `parasites_per_frame` upper bounds in the dense/very_dense setups
- Drop `n_frames` proportionally
- Run setups in parallel by splitting the YAML into pieces

For training, a default-medium SLEAP UNet on a single GPU finetuned on this
data should converge in a few hours.

## Tweaking

The simulator has many knobs. Most useful to revisit:

- **`body_phase_shift`** range in `sample_random_parasite` — controls how dark cells appear (1.2–7.0)
- **`flagellum_length`** range — currently 5–30 µm
- **Beat shape** params in `sample_random_parasite` (per Wang/Wheeler 2020 framework):
  - `tip_to_base_envelope_exponent` — width of the central sin-envelope peak
  - `base_to_tip_static_curl_shape` — circular arc (1.0) vs distal hook (~2.5)
  - `base_to_tip_temporal_asymmetry` — fast-power-stroke vs symmetric sinusoid
  - `base_to_tip_propagation_extent` — whether the wave reaches the tip (set <0.85 for partial-propagation phenotype)
- **Jitter ranges** in `dataset_builder.py` (`_jitter_optics_object`, `_jitter_noise_object`) — multiplicative spread per frame
- **`bg_intensity_range`** in YAML defaults — per-frame brightness range

If your real data looks different from the synthetic, the GUI is the fastest
way to find what to change. You can save the tuned params with the GUI's JSON
export, then either feed them to a single-cell `ParasiteParams(**params)` for
debugging, or use them to inform the sampler ranges in `sample_random_parasite`.
