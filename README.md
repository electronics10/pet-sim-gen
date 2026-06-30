# pet-sim-gen

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

You can/should modify the configuration (image and sinogram domain dimension, scanner geometry, process configuration, etc.) and the bounds (phantoms' sizes, density and activity range, etc.). The easiest way may be modifying json files directly and load them back afterward. One may try:

```python
import mcgpu_pet_wrapper as mpw
import pet_sim_gen as psg


cfg = mpw.default_config()
mpw.save_config(cfg, "tmp/manual_config.json")
bounds = psg.suggest_bounds(cfg)
psg.save_bounds(bounds, "tmp/manual_config.json")
```

**Interrupt any time (Ctrl-C) and rerun the same script** — completed samples are
skipped; it resumes where it stopped.

Inspect recipes *without* a GPU first (catches sampling mistakes for free):

```python
from pet_sim_gen import sample_recipe, suggest_bounds_maximal
from mcgpu_pet_wrapper import default_config
cfg = default_config(); bounds = suggest_bounds_maximal(cfg)
for i in range(5):
    r = sample_recipe(seed=i, bounds=bounds, config=cfg)
    print(len(r.instructions), "objects")
```

---

## 4. The design in one picture

```
bounds.json  ──►  sample_recipe  ──►  Recipe  ──►  build_voxel_grid  ──►  VoxelGrid  ──►  Runner  ──►  outputs
 (what objects     (draws objects     (plain      (wrapper's builder)   (wrapper)       (wrapper,
  can exist)        from bounds)        data)                                            subprocess)
       └────────────────────────── generate_dataset (frozen, robust loop) ───────────────────────────┘
                              ▲
                StratifiedSampler (optional, steers which recipes are kept)
```

Two layers, separated on purpose:

- **Upstream (changes often):** *what to sample* — `sample_recipe`, `bounds.json`.
- **Downstream (frozen, robust):** *how to run a batch reliably* —
  `generate_dataset`. It does not care what a phantom is or what task it serves.

A **Recipe** is plain serializable data (a list of objects to paint). It is
GPU-free, instantly inspectable, and is logged verbatim — so every sample is
reproducible from its recipe.

---

## 5. `bounds.json`: what objects can exist

`bounds` is the **object-distribution policy**: the one thing you normally
customize. It is a plain dict (pass it directly, or load it from your own JSON
file) that defines, per material, the density and activity ranges to sample, plus
object counts and sizes. It deliberately contains **no scatter, no task, and no
stratification** — those are decided elsewhere.

You rarely write it from scratch: call `suggest_bounds_maximal(config)` to get a
config-consistent starting dict (§7), edit the fields below, then pass it to
`generate_dataset(bounds=...)`. An annotated reference with every field documented
ships at `pet_sim_gen/examples/bounds.annotated.json`. The shape:

```jsonc
{
  "n_objects":  { "min": 1, "max": 5 },
  "size_mm":    { "min": 4.0, "max": 18.0 },     // base half-size; FOV-capped
  "aspect_ratio": { "min": 0.5, "max": 2.0 },    // per-axis -> elongated shapes
  "cylinder_height_mm": { "min": 8.0, "max": 60.0 },

  "background": { "material_id": 1, "density_g_cm3": 0.0012, "activity_Bq_per_mL": 0.0 },

  "insert_materials": {
    "entries": [
      { "material_id": 2, "relative_frequency": 1.0,
        "density_g_cm3": [0.3, 1.1],  "activity_Bq_per_mL": [0, 20000] },
      { "material_id": 3, "relative_frequency": 1.0,
        "density_g_cm3": [0.9, 0.97], "activity_Bq_per_mL": [0, 10000] },
      { "material_id": 4, "relative_frequency": 1.0,
        "density_g_cm3": [1.0, 1.2],  "activity_Bq_per_mL": [0, 5000] }
    ]
  },

  "fov_margin_mm": { "value": 3.0 }
}
```

Key ideas:

- **Two primitive shapes only:** ellipsoid and elliptic cylinder. Spheres and
  circular cylinders are their degenerate cases (equal semi-axes), so they appear
  naturally without a separate type.
- **`material_id` indexes the config's material list**, 1-based: id 1 is the first
  listed (air, reserved for the background), id 2 the second, etc. At 511 keV
  Compton dominates and its rate tracks electron density, so *water at a chosen
  density* is a valid surrogate for soft tissues (e.g. water at 0.3 g/cm³ ≈ lung).
- **`relative_frequency`** sets how often each material is chosen, normalized
  internally over the materials you list — so it is robust to adding/removing
  materials. Default 1.0 each ⇒ uniform.
- **"Coupling" is implicit in range width.** Want a material to be always cold?
  Give it a tight activity range near 0. Want brightness decoupled from density?
  Give it a wide, independent activity range. No separate coupling knob is needed.
- **FOV guard:** objects are kept inside the transaxial radius (minus
  `fov_margin_mm`). Objects that cannot fit are rejected and re-drawn; the
  simulator's own validator is the final backstop.

---

## 6. Stratification: flattening coverage

If you sample objects uniformly, some *derived* quantity (e.g. scatter fraction)
piles up in the middle and starves the extremes — exactly where a model is
weakest. **Stratification** steers generation so a chosen scalar is *uniformly
covered*.

This **cannot** be done by editing `bounds.json` (flattening a nonlinear function
of the parameters needs accept/reject at generation time), so it is passed at the
call site, not stored in bounds. It is also fully **generic**: you supply a
`key_fn(recipe) -> float` and a `(min, max, n_bins)` target; the tool flattens
coverage over that key. The key can measure anything.

A scatter-fraction proxy is provided as **one example key** (not core):

```python
from pet_sim_gen import generate_dataset
from pet_sim_gen.examples.sf_proxy import sf_proxy

generate_dataset(
    n=2000, out_dir="data",
    stratify_key=sf_proxy,            # any recipe -> float works
    stratify_target=(0.05, 0.46, 12) # flatten over SF-proxy in [0.05, 0.46], 12 bins
)
```

A custom key is just a callable — there is no CLI flag for it, by design: a shell
argument can't carry a Python function, and baking one key in would betray the
tool's task-agnostic core. Write `key_fn(recipe) -> float` and pass it.

**`(min, max, n_bins)` means:** cover the key's range from `min` to `max`, split
into `n_bins` equal bands, aiming for ~equal counts per band.

> **Reachability.** The target must lie within what your bounds can actually
> produce. If you ask for a `max` your objects can't reach, the top bins never
> fill and the stratifier reports it (and the run logs the failure and continues).
> The fix is to **widen the bounds** (bigger/denser objects) or **lower the
> target** — not to wait longer. Check your achievable range first, GPU-free:
>
> ```python
> import numpy as np
> from pet_sim_gen import sample_recipe, suggest_bounds_maximal
> from pet_sim_gen.examples.sf_proxy import sf_proxy
> from mcgpu_pet_wrapper import default_config
> cfg = default_config(); bounds = suggest_bounds_maximal(cfg)
> v = np.array([sf_proxy(sample_recipe(seed=i, bounds=bounds, config=cfg))
>               for i in range(2000)])
> print(v.min(), np.percentile(v, 99), v.max())   # set sf-max below the 99th pct
> ```

> **Proxy vs. truth.** A proxy key flattens the *proxy*, which only approximates
> the true post-simulation quantity. Verify realized coverage downstream from the
> stored `sinogram_Trues`/`sinogram_Scatter` files. The proxy is for steering
> only — never a label.

---

## 7. Generating bounds automatically

Editing `bounds.json` from scratch is tedious. Two generators emit a valid
starting bounds dict from a config (a *family* — add your own with the same
`config -> dict` signature):

```python
from pet_sim_gen import suggest_bounds_maximal, suggest_bounds_realistic
from mcgpu_pet_wrapper import default_config
import json

cfg = default_config()

b = suggest_bounds_maximal(cfg)      # broad, FOV-consistent ranges (safe scaffold)
json.dump(b, open("my_bounds.json", "w"), indent=2)   # edit, then pass to generate_dataset

b = suggest_bounds_realistic(cfg)    # physiological per-material ranges (partial)
```

- **`suggest_bounds_maximal`** — broad ranges; object size capped at 0.7× the
  transaxial FOV radius so large (high-scatter) objects are reachable. The safe
  default; tighten to taste.
- **`suggest_bounds_realistic`** — matches each material to plausible 511-keV
  density/activity ranges by recognizing its name (water, adipose, spongiosa,
  lung, bone). **Partial by design**: unrecognized materials fall back to broad
  ranges. Fully realistic phantom generation is hard and may not suit every task
  (scatter work prefers physical validity over anatomical realism), so it is a
  hook for users who want it, not a promise.

---

## 8. Robustness: crashes, resume, atomicity

`generate_dataset` is built to be left running and to survive failures:

- **Crash isolation.** Each sample is wrapped in try/except; a failure is logged
  to `failures.jsonl` and the batch continues. MCGPU runs as a **subprocess**, so
  a GPU crash kills the subprocess, not the loop. A per-sample timeout kills hangs.
- **Resume.** A completed sample gets a `DONE` sentinel; the loop skips any sample
  whose `DONE` exists. **Rerun the same script to resume** after any interruption.
  Stratified runs additionally rebuild their per-bin counts from the completed
  samples on startup, so resumed coverage stays flat (an interrupted-and-resumed
  stratified run reproduces the same final coverage as an uninterrupted one).
- **Atomicity.** Each sample is staged and simulated in `run_XXXXX.tmp`, then
  atomically renamed to `run_XXXXX`. A directory at the final name is therefore
  *guaranteed complete* — a crash mid-write leaves a `.tmp` that is wiped and
  redone, never mistaken for done.
- **Crash-safe logs.** Manifest and failures are append-only JSONL, flushed per
  line; a truncated last line is trivially discardable.
- **Fail-fast.** The phantom is validated *before* the expensive simulation, so
  bad geometry fails in milliseconds.

A good habit: run a small batch, interrupt it mid-way, rerun, and confirm it
resumes cleanly before launching a long run.

---

## 9. Extending it

The package separates *what churns* from *what is frozen*, so extensions are local:

- **New object distribution?** Edit `bounds.json`. Nothing else changes.
- **New stratification target?** Write a `key_fn(recipe) -> float` and pass it to
  `generate_dataset`. The example `sf_proxy` is a template (`pet_sim_gen/examples/`).
- **New bounds generator?** Add a `config -> dict` function to `bounds_tools.py`.
- **New shape primitive?** Add a `PaintInstruction` kind + a builder call in
  `sampler.py`. The recipe interface and orchestrator do not move.

The orchestrator and the recipe interface are intended to stay frozen as the
upstream sampling evolves.

---

## 10. API reference

```python
from pet_sim_gen import (
    sample_recipe,            # (seed, bounds, config) -> Recipe   (pure, GPU-free)
    build_voxel_grid,         # (recipe, bounds, config) -> VoxelGrid (validated)
    generate_dataset,         # the robust batch loop; returns a summary dict
    StratifiedSampler,        # generic coverage-flattening wrapper
    suggest_bounds_maximal,   # config -> broad bounds dict
    suggest_bounds_realistic, # config -> physiological bounds dict (partial)
    Recipe, PaintInstruction, # the plain-data recipe types
)
from pet_sim_gen.examples.sf_proxy import sf_proxy   # example stratification key
```

`generate_dataset` signature:

```python
generate_dataset(
    n,                       # number of samples
    out_dir="data",          # output root (working directory by default)
    bounds=None,             # dict or path; None -> suggest_bounds_maximal(config)
    config=None,             # path/dict; None -> wrapper default_config()
    base_seed=0,             # reproducible per-sample seeds
    timeout_s=3600.0,        # per-sample hang guard
    stratify_key=None,       # optional callable recipe -> float
    stratify_target=None,    # optional (min, max, n_bins)
    verbose=True,
) -> dict                    # {"completed", "skipped", "failed", "manifest"}
```

---

*This project produces data; it stays out of the modeling. The simulator's
fidelity (and the sim-to-real gap) is inherited from MCGPU-PET — validate against
a reference (e.g. GATE/PeneloPET) and a physical phantom before trusting absolute
numbers. Scatter labels here are randoms-free by construction; real pipelines
remove randoms upstream before scatter correction.*