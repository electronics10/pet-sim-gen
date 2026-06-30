# Mass Data Generation for Monte Carlo PET Simulation

![GitHub release](https://img.shields.io/github/v/release/electronics10/pet-sim-gen?include_prereleases)

Generate large, diverse Monte Carlo PET datasets — via
[`mcgpu-pet-wrapper`](https://github.com/electronics10/mcgpu-pet-wrapper) — for
downstream machine learning. Each sample provides the **unscattered (Trues)** and
**scattered (Scatter)** sinograms *separately*, the supervision signal no real
scanner can give. The tool is **task-agnostic**: it produces the simulations and
the bookkeeping; what you treat as input vs. label (scatter estimation,
attenuation-map prediction, reconstruction, denoising, …) is your choice
downstream.

---

## Table of contents

1. [What it produces](#1-what-it-produces)
2. [Install](#2-install)
3. [Quick start](#3-quick-start)
4. [The design in one picture](#4-the-design-in-one-picture)
5. [`bounds.json`: what objects can exist](#5-boundsjson-what-objects-can-exist)
6. [Stratification: flattening coverage](#6-stratification-flattening-coverage)
7. [Generating bounds automatically](#7-generating-bounds-automatically)
8. [Robustness: crashes, resume, atomicity](#8-robustness-crashes-resume-atomicity)
9. [Extending it](#9-extending-it)
10. [API reference](#10-api-reference)

---

## 1. What it produces

For each phantom, a self-contained run directory:

```
data/runs/run_00042/
  config.json             # exact geometry/physics used (reproducibility)
  MCGPU-PET.in            # generated simulator input
  voxel_space.vox.gz      # the object: per-voxel material, density, activity
  sinogram_Trues.raw.gz   # unscattered coincidences  (clean reference)
  sinogram_Scatter.raw.gz # scattered coincidences    (the scatter ground truth)
  image_Trues.raw.gz      # per-voxel emitted-true counts (activity ground truth)
  image_Scatter.raw.gz
  recipe.json             # the exact paint instructions that made this phantom
  MCGPU-PET.out           # simulator log
  DONE                    # sentinel written last (marks the dir complete)
```

Plus, at the dataset root:

```
data/manifest.jsonl       # one line per completed sample (task-agnostic facts)
data/failures.jsonl       # one line per failed sample (seed + error + traceback)
data/run_config.json      # record generate info, bounds, and config so the dataset is self-contained
```

The manifest records only facts true of *any* phantom — total activity, total
mass, mean density, object count, wall time, and the full recipe.

---

## 2. Install

You need Python 3.10 or newer. This package drives [`mcgpu-pet-wrapper`](https://github.com/electronics10/mcgpu-pet-wrapper) (the GPU Monte Carlo simulator interface); it is pulled in automatically, so you don't need to clone anything.

Install into an isolated environment (Python's built-in `venv`, conda/mamba, uv, or pixi). For example with conda:

```bash
conda create -n petgen python=3.10
conda activate petgen

pip install git+https://github.com/electronics10/pet-sim-gen.git
```

This installs `pet_sim_gen` and the `mcgpu-pet-wrapper` dependency in one step.

> **Already using uv or pixi?** Same URL: `uv add git+https://github.com/electronics10/pet-sim-gen.git` or `pixi add --pypi "pet-sim-gen @ git+https://github.com/electronics10/pet-sim-gen.git"`.

For developers
> Simply git clone the repository and try uv sync.

**Heads up**
> **GPU note.** Running simulations needs Linux + an NVIDIA GPU with CUDA (a constraint inherited from the wrapper's bundled binary). The pure-logic parts of this package — recipe sampling, stratification, bounds generators — run anywhere without a GPU, useful for inspecting recipes before committing GPU time.

---

## 3. Quick start

The interface is the `generate_dataset` function; it captures the whole experiment definition (n, config, bounds, seed, stratification) as version-controlled code.

**Plain generation (10 samples into ./data):**

```python
# run_example.py
import mcgpu_pet_wrapper as mpw
import pet_sim_gen as psg


cfg = mpw.default_config()
bounds = psg.suggest_bounds(cfg)

psg.generate_dataset(
    n=10, 
    config = cfg,
    bounds = bounds,
    out_dir="data"
)
```

```bash
python run_example.py
```

You could/should modify the configuration (image and sinogram domain dimension, scanner geometry, process configuration, etc.) and the bounds (phantoms' sizes, density and activity range, etc.). The easiest way may be modifying json files directly and load them back afterward. One may try:

```python
cfg = mpw.default_config()
mpw.save_config(cfg, "tmp/config.json")
bounds = psg.suggest_bounds(cfg)
psg.save_bounds(bounds, "tmp/bounds.json")
```

Change the configuration of both files as you want. Normally, you might want to change the scanner geometry or tighten the bounds for different materials. After you finish the change, you may load them back and actually run the `generate_dataset` function.

```python
cfg = mpw.load_config("tmp/config.json")
bounds = psg.load_bounds("tmp/bounds.json")
psg.generate_dataset(10, cfg, bounds, "data")
```

**Interrupt any time (Ctrl-C) and rerun the same script** — completed samples are
skipped; it resumes where it stopped.

---

## 4. The design in one picture

```
config + bounds ──► sample_recipe ──► Recipe ──► build_voxel_grid ──► VoxelGrid ──► Runner ──► outputs
 (geometry +        (draws objects    (plain     (wrapper's builder)  (wrapper)    (wrapper,
  what can exist)    from bounds)       data)                                       subprocess)
        └────────────────────────── generate_dataset (frozen, robust loop) ────────────────────────┘
                              ▲
                StratifiedSampler (optional, steers which recipes are kept)
```

Two layers, separated on purpose:

- **Upstream (changes):** *what to sample* — the `config`, the `bounds`
  dict.
- **Downstream (frozen, robust):** *how to run a batch reliably* —
  `generate_dataset`. It does not care what a phantom is or what task it serves.

A **Recipe** is plain serializable data (a list of objects to paint, plus the
seed). It is GPU-free, instantly inspectable, and is logged verbatim to each
`run_XXXXX/recipe.json` — so every sample is reproducible from its recipe, and
the recipe round-trips (`Recipe.from_dict`) for resume and analysis.

---

## 5. `bounds`: what objects can exist

`bounds` is the **object-distribution policy**: the main thing you customize. It
is a plain dict (build it with `suggest_bounds`, edit, and pass directly; or load
your own JSON with `load_bounds`) defining, per material, the density and activity
ranges to sample, plus object counts and sizes. It deliberately contains **no
scatter, no task, and no stratification** — those are decided elsewhere.

You rarely write it from scratch: call `suggest_bounds(config)` for a
config-consistent starting dict, edit, then pass it to
`generate_dataset(..., bounds=...)`. Every field carries an inline `_doc` string
in the generated dict, so the dict itself is self-documenting. The shape:

```jsonc
{
  "n_objects":  { "min": 1, "max": 6 },
  "size_mm":    { "min": 2.0, "max": 21.0 },     // base half-size; FOV-derived cap
  "aspect_ratio": { "min": 0.5, "max": 2.0 },    // per-axis -> elongated shapes
  "cylinder_height_mm": { "min": 8.0, "max": 150.0 },

  "background": { "material_id": 1, "density_g_cm3": 0.0012, "activity_Bq_per_mL": 0.0 },

  "insert_materials": {
    "entries": [
      { "material_id": 2, "relative_frequency": 1.0,
        "density_g_cm3": [0.3, 1.9], "activity_Bq_per_mL": [0, 20000] },
      { "material_id": 3, "relative_frequency": 1.0,
        "density_g_cm3": [0.3, 1.9], "activity_Bq_per_mL": [0, 20000] },
      { "material_id": 4, "relative_frequency": 1.0,
        "density_g_cm3": [0.3, 1.9], "activity_Bq_per_mL": [0, 20000] }
    ]
  },

  "fov_margin_mm": { "value": 0.0 }
}
```

Key ideas:

- **Two primitive shapes only:** ellipsoid and elliptic cylinder. Spheres and
  circular cylinders are their degenerate cases (equal semi-axes), so they appear
  naturally without a separate type. `aspect_ratio` multiplies the base
  `size_mm` per axis, so the same base size explores spheres through elongated
  bodies.
- **`material_id` indexes the config's material list**, 1-based: id 1 is the first
  listed (assume air, reserved for the background), id 2 the second, etc. At 511 keV
  Compton dominates and its rate tracks electron density, so *water at a chosen
  density* is a valid surrogate for soft tissues (e.g. water at 0.3 g/cm³ ≈ lung).
- **`relative_frequency`** sets how often each material is chosen, normalized
  internally over the materials you list — so it is robust to adding/removing
  materials. Default 1.0 each ⇒ uniform.
- **"Coupling" is implicit in range width.** Want a material always cold? Give it
  a tight activity range near 0. Want brightness decoupled from density? Give it a
  wide, independent activity range.
- **FOV guard:** objects are kept inside the transaxial radius (minus
  `fov_margin_mm`) and the axial extent. Objects that cannot fit are rejected and
  re-drawn; `VoxelGrid.validate()` and the wrapper's scanner-radius warning are
  the final backstop.

---

## 6. Generating bounds

Writing `bounds` from scratch is tedious. Two generators emit a valid starting
dict from a config (a *family* — add your own with the same `config -> dict`
signature):

```python
import mcgpu_pet_wrapper as mpw
import pet_sim_gen as psg

cfg = mpw.default_config()

bounds = psg.suggest_bounds(cfg)            # broad, FOV-consistent ranges (safe scaffold)
psg.save_bounds(bounds, "tmp/bounds.json")  # write, edit by hand, then load_bounds() back

bounds = psg.suggest_bounds_realistic(cfg)  # physiological per-material ranges (partial)
```

- **`suggest_bounds`** — broad ranges; object base size capped at 0.8× the
  transaxial FOV radius (aspect ratios apply on top), so large (high-scatter)
  objects are reachable. The safe default; tighten to taste. Every field includes
  an inline `_doc` string.
- **`suggest_bounds_realistic`** — matches each material to plausible 511-keV
  density/activity ranges by recognizing its name (water, adipose, spongiosa,
  lung, bone). **Partial by design**: unrecognized materials fall back to the
  broad ranges with no error. Fully realistic phantom generation is hard and may
  not suit every task (scatter work prefers physical validity over anatomical
  realism), so it is a hook for users who want it, not a promise.

`load_bounds(path)` and `save_bounds(bounds, path)` round-trip the dict through
JSON so you can edit geometry, scanner, materials, and ranges by hand between
runs.

---

## 7. Stratification: flattening coverage

If you sample objects uniformly, some *derived* quantity (e.g. scatter fraction)
piles up in the middle and starves the extremes — exactly where a model is
weakest. **Stratification** steers generation so a chosen scalar is *uniformly
covered*.

This **cannot** be done by editing `bounds` (flattening a nonlinear function of
the parameters needs accept/reject at generation time), so it is passed at the
call site, not stored in bounds. It is also fully **generic**: you supply a
`key_fn(recipe) -> float` and a `(min, max, n_bins)` target; the tool flattens
coverage over that key. The key can measure anything.

A scatter-fraction proxy ships as **one example key** (not core; see the
proxy's own doc in the following subsection):

```python
import mcgpu_pet_wrapper as mpw
import pet_sim_gen as psg
from pet_sim_gen.examples import sf_proxy      # NOTE: flat module, not a submodule

cfg = mpw.default_config()
bounds = psg.suggest_bounds(cfg)

psg.generate_dataset(
    n=1000, config=cfg, bounds=bounds, out_dir="data",
    stratify_key=sf_proxy,             # any recipe -> float works
    stratify_target=(0.05, 0.3, 5),  # flatten over the key in [0.05, 0.3], 5 bins
)
```

A custom key is just a callable. Write your own `key_fn(recipe) -> float` and pass it.

**`(min, max, n_bins)` means:** cover the key's range from `min` to `max`, split
into `n_bins` equal bands, aiming for ~equal counts per band.

> **Reachability.** The target must lie within what your bounds can actually
> produce. If you ask for a `max` your objects can't reach, the top bins never
> fill and the run reports it (logging the failure and continuing). The fix is to
> **widen the bounds** (bigger/denser objects) or **lower the target** — not to
> wait longer. Check your achievable range first, GPU-free:
>
> ```python
> import numpy as np
> import mcgpu_pet_wrapper as mpw
> import pet_sim_gen as psg
> from pet_sim_gen.examples import sf_proxy
>
> cfg = mpw.default_config(); bounds = psg.suggest_bounds(cfg)
> v = np.array([sf_proxy(psg.sample_recipe(seed=i, bounds=bounds, config=cfg))
>               for i in range(1000)])
> print(v.min(), np.percentile(v, 99), v.max())   # set the target max below the 99th pct
> ```

> **Proxy vs. truth.** A proxy key flattens the *proxy*, which only approximates
> the true post-simulation quantity. Set the target in **proxy units**, not
> true units (run the proxy on a reference phantom to convert). Verify realized
> coverage downstream from the simulation files. The proxy steers sampling; it is never a actual label.

### 7.1 The scatter-fraction proxy

The shipped `sf_proxy` is a redesigned, validated key — not the naive
"size × density" sketch. It estimates the **activity-weighted mean escape optical
depth**: it paints the recipe on a coarse grid, computes how much attenuating
material each emitting voxel's photons must cross to leave the phantom, and
weights by the *activity fraction* (because SF is a ratio, so activity magnitude
cancels). On a 20-sample Monte-Carlo battery it ranks at Spearman ≈ 0.84, versus
≈ 0.21 for a naive sum-of-per-insert optical depths. Two decisions carried that
gain: weighting by activity fraction (not total), and capturing the dominant path
(not summing inserts). A cheaper closed-form fallback, `max_tau_proxy`
(Spearman ≈ 0.81, ~100× faster), is provided for when voxelization bottlenecks.
See the proxy's full write-up for the physics, limits (it ranks, it does not
calibrate; SF saturates in the optically-thick regime), and how to replace it.

<details>

## The scatter-fraction proxy (`sf_proxy`)

### What it is

`sf_proxy(recipe) -> float` is a cheap, pre-simulation estimate used **only to
steer dataset coverage** toward a flat (or shaped) distribution of scatter
fraction. It is a *stratification key*, never a label: the ground-truth scatter
fraction always comes from the Monte-Carlo sinograms downstream. Because it only
feeds a stratifier, what matters is that it **rank-orders** phantoms by their true
scatter fraction — its absolute value is irrelevant.

### Definition we are approximating

The scatter fraction of a phantom is

    SF = scattered_coincidences / (scattered + true) coincidences

counted over *detected* events (both 511 keV photons reach the ring inside the
energy window). This is a post-simulation quantity; the proxy estimates how a
phantom will rank on it before paying for the simulation.

### Physical basis

Two facts about this simulator fix the form of the proxy:

1. **The energy window and resolution dominate SF, and they are held constant**
   across every run (they match a fixed real acquisition). So the *acquisition*
   contribution to SF is a constant; the only thing that varies sample-to-sample
   is the **phantom geometry, density, and activity layout**.

2. **At 511 keV scatter is Compton scatter, whose rate is proportional to
   electron density**, which — because most material in the body is water-equivalent
   cross-sections scaled by mass density — is simply proportional to `ρ`. So
   "amount of scattering material" is just mass, with no per-material correction.

Given a fixed window, the residual SF spread is therefore driven by **how much
attenuating material a coincidence must traverse**, weighted by **where the
activity actually sits**. A decay deep inside a large dense body produces a
photon pair that crosses more material — and is more likely to scatter — than the
same decay near a surface or in a thin body.

### The estimate

For each emitting voxel `v` we estimate the mean optical depth its photons cross
to escape the phantom:

    τ(v) = mean over escape directions of  Σ μ·ρ·dℓ   along the ray from v

where `μ = 0.096 cm⁻¹` is water's linear attenuation at 511 keV and the path
integral runs through the voxelized phantom. The proxy is the
**activity-weighted mean** of this depth, wrapped into `[0, 1)`:

    sf_proxy = 1 − exp( − Σ_v  w(v)·τ(v) ),    w(v) = activity(v) / Σ activity

Implementation: the recipe is painted onto a coarse grid (2 mm; ranking-only, so
resolution is cheap), `τ(v)` is approximated by averaging escape depth along the
six axis directions, and the result is weighted by the normalized activity map.

### Two design decisions that matter

These are the choices that make the proxy work; a naive version gets them wrong.

- **Weight by activity *fraction*, not total activity.** SF is a *ratio*; the
  activity magnitude cancels (both numerator and denominator scale with it). What
  survives is *where* the activity is relative to the mass. Using total activity
  instead of the normalized fraction injects a confound that does **not** track
  SF (verified: near-zero rank correlation).

- **Aggregate the dominant path, not a sum over inserts.** SF is governed by the
  deepest material a coincidence crosses, not the count of separate objects.
  Summing per-insert optical depths conflates "one deep body" with "many shallow
  ones" and ranks poorly. The activity-weighted depth above captures the dominant
  path automatically; a closed-form `max`-over-inserts variant
  (`max_tau_proxy`) is a cheaper stand-in with the same intent.

### Validation

Rank correlation against Monte-Carlo SF (Spearman ρ):

| key                              | ρ      | note                          |
|----------------------------------|--------|-------------------------------|
| `sf_proxy` (activity-weighted)   | ~0.90  | the key in use                |
| `max_tau_proxy` (closed-form)    | ~0.81  | ~100× cheaper fallback        |
| total mass `Σ ρV`                | ~0.67  | geometry-blind                |
| sum of per-insert optical depths | ~0.21–0.63 | the naive version; unusable |
| total activity                   | ~0.18–0.38 | confound; correctly near-zero |

Measured on 20- and 50-sample batches. Stratifying reshapes the phantom mix, so a key validated on plain
sampling should be re-checked on the set it steers into.

### Limits (read before trusting it)

- **It ranks, it does not calibrate.** `sf_proxy` systematically over- or
  under-shoots the absolute SF value; only the ordering is meaningful. Do not read
  its number as a predicted scatter fraction.
- **Set the stratifier target in proxy units, not true-SF units.** A true SF of,
  say, 0.27 from a hand-built phantom corresponds to some *different* `sf_proxy`
  value; run `sf_proxy` on that phantom to get the real ceiling before setting
  `stratify_target`.
- **Coverage is bounded by what the sampler can build, not by the target.** If the
  target range exceeds the reachable regime, the top bins fill with the highest
  *available* phantoms — which may be lower-SF than intended. Reaching a regime is
  a *bounds* problem; stratification only balances *within* the reachable set.
- **SF saturates.** Once a phantom is optically thick, more material stops raising
  SF (multiple-scatter leaves the window; trues also attenuate). The "more
  material → more scatter" intuition holds in the optically-thin regime and bends
  over near the top, so the proxy's discriminating power is weakest exactly where
  SF is highest.
- **Approximations in the escape estimate.** Six-direction averaging (not full 4π)
  and treating the two back-to-back photons as independent escapes are deliberate
  ranking-only simplifications. If rank correlation slips on a new distribution,
  replacing the six axes with a real spherical direction sample is the first
  refinement.

### Replacing it

`sf_proxy` is an *example* key in `examples.py`, not core. The stratifier accepts
any `key_fn(recipe) -> float`. A different task (attenuation-path coverage, mean
density, object-size balance, …) supplies its own key and validates it against whatever downstream label that task cares about.

</details>

---

## 8. Robustness: crashes, resume, atomicity

`generate_dataset` is built to be left running and to survive failures:

- **Crash isolation.** Each sample is wrapped in try/except; a failure is logged
  to `failures.jsonl` and the batch continues. MCGPU runs as a **subprocess**, so
  a GPU crash kills the subprocess, not the loop. A per-sample `timeout_s` kills
  hangs.
- **Resume.** A completed sample gets a `DONE` sentinel; the loop skips any sample
  whose `DONE` exists. **Rerun the same script to resume** after any interruption.
  Stratified runs additionally rebuild their per-bin counts from the completed
  samples on startup (`observe`), so resumed coverage stays flat. Per-index RNGs
  (`seed_for(base_seed, i)`) keep each sample's candidate stream stable across an
  interruption, so the preserved guarantee is dataset-level flat coverage.
- **Atomicity.** Each sample is staged and simulated in `run_XXXXX.tmp`, then
  atomically renamed to `run_XXXXX`. A directory at the final name is therefore
  *guaranteed complete* — a crash mid-write leaves a `.tmp` that is wiped and
  redone, never mistaken for done.
- **Crash-safe logs.** Manifest and failures are append-only JSONL, flushed per
  line; a truncated last line is trivially discardable.
- **Provenance.** `run_config.json` records the exact call-site arguments (n,
  base_seed, bounds, config, stratify target, and a hash of the stratify key's
  source file) so the dataset is self-contained and you can detect if a key's
  definition drifted since generation. Re-running into the same folder appends to
  a `runs` list rather than clobbering, keeping full lineage.
- **Fail-fast.** Stratification mis-wiring (key without target, or a
  non-callable key) is rejected before any work, and each phantom is validated
  *before* the expensive simulation, so mistakes fail in milliseconds.

A good habit: run a small batch, interrupt it mid-way, rerun, and confirm it
resumes cleanly before launching a long run.

---

## 9. Extending it

The package separates *what churns* from *what is frozen*, so extensions are
local:

- **New object distribution?** Edit the `bounds` dict. Nothing else changes.
- **New stratification target?** Write a `key_fn(recipe) -> float` and pass it to
  `generate_dataset`. The example `sf_proxy` in `examples.py` is a template.
- **New bounds generator?** Add a `config -> dict` function to `bounds_tools.py`.
- **New shape primitive?** Add a `PaintInstruction` kind + a builder call in
  `sampler.py`. The recipe interface and orchestrator do not move.

The orchestrator and the recipe interface are intended to stay frozen as the
upstream sampling evolves.

---

## 10. API reference

```python
import pet_sim_gen as psg

psg.sample_recipe            # (seed, bounds, config) -> Recipe   (pure, GPU-free)
psg.build_voxel_grid         # (recipe, bounds, config) -> VoxelGrid (validated)
psg.generate_dataset         # the robust batch loop; returns a summary dict
psg.StratifiedSampler        # generic coverage-flattening wrapper
psg.suggest_bounds           # config -> broad bounds dict
psg.suggest_bounds_realistic # config -> physiological bounds dict (partial)
psg.load_bounds, psg.save_bounds   # JSON round-trip for the bounds dict
psg.Recipe, psg.PaintInstruction   # the plain-data recipe types

from pet_sim_gen.examples import sf_proxy   # example stratification key (flat module)
```

`generate_dataset` signature (note: `config` and `bounds` are required, in that
order after `n`):

```python
generate_dataset(
    n,                       # number of samples
    config,                  # wrapper config dict (e.g. mpw.default_config())
    bounds,                  # object-distribution dict (e.g. psg.suggest_bounds(config))
    out_dir="data",          # output root
    base_seed=0,             # reproducible per-sample seeds
    timeout_s=3600.0,        # per-sample hang guard
    stratify_key=None,       # optional callable recipe -> float
    stratify_target=None,    # optional (min, max, n_bins); required with stratify_key
    verbose=True,
) -> dict                    # {"completed", "skipped", "failed", "manifest"}
```

---

*This project produces data; it stays out of the modeling. The simulator's
fidelity (and the sim-to-real gap) is inherited from MCGPU-PET — validate against
a reference (e.g. GATE/PeneloPET) and a physical phantom before trusting absolute
numbers. Scatter labels here are randoms-free by construction; real pipelines
remove randoms upstream before scatter correction.*