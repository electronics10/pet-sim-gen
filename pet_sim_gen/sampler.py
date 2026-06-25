"""
sampler.py -- the UPSTREAM, churning part of pet-sim-data.

Pure function: (seed, bounds, wrapper_config) -> Recipe.

A Recipe is plain serializable data: a list of PaintInstruction (one per object,
largest-to-smallest) plus the seed. It knows nothing about voxels, the GPU, or
the simulator -- it is just "what to paint". Downstream consumes it; this file is
the only place that changes when sampling design evolves. Stratification is a
separate runtime wrapper (bounds_tools.StratifiedSampler), not part of this file.

Primitives (two general kinds; special cases are degenerate):
  - "ellipsoid"          : add_ellipsoid; equal semi-axes => sphere.
  - "elliptic_cylinder"  : add_elliptic_cylinder; equal rx,ry => cylinder.

Material/density/activity: each object picks a MATERIAL from the configured
insert_materials (by normalized relative_frequency), then draws density and
activity from THAT material's ranges. 'Coupling' is implicit in range width.

FOV guard: objects kept inside the transaxial radius limit (minus margin) and
the axial extent. VoxelGrid.validate() and the wrapper's scanner-radius warning
are the backstop; the robust orchestrator tolerates the occasional reject.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Literal, Any

import numpy as np

from mcgpu_pet_wrapper import voxel_space_extent_mm, radial_fov_mm


# ----------------------------------------------------------------------------
# Recipe data types -- the interface contract with the downstream builder
# ----------------------------------------------------------------------------

@dataclass
class PaintInstruction:
    """One object to paint. Mirrors the builder's add_* signatures."""
    kind: Literal["ellipsoid", "elliptic_cylinder"]
    center_mm: tuple[float, float, float]
    material_id: int
    density_g_cm3: float
    activity_Bq_per_mL: float

    semi_axes_mm: tuple[float, float, float] | None = None   # ellipsoid
    rx_mm: float | None = None                                # elliptic_cylinder
    ry_mm: float | None = None
    height_mm: float | None = None
    axis: Literal["x", "y", "z"] | None = None
    theta_deg: float = 0.0

    approx_volume_mm3: float = 0.0   # for large-to-small ordering only


@dataclass
class Recipe:
    seed: int
    instructions: list[PaintInstruction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed,
                "instructions": [asdict(i) for i in self.instructions]}


# ----------------------------------------------------------------------------
# The sampler
# ----------------------------------------------------------------------------

def sample_recipe(seed: int, bounds: dict, config: dict) -> Recipe:
    """Draw one phantom recipe. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)

    # --- FOV geometry. radial FOV = R*cos(...) <= R always, so r_fov binds. ---
    ext_x, ext_y, ext_z = voxel_space_extent_mm(config)
    r_fov = radial_fov_mm(config) / 2.0
    margin = float(bounds["fov_margin_mm"]["value"])
    r_limit = max(0.0, r_fov - margin)
    cx0, cy0, cz0 = ext_x / 2.0, ext_y / 2.0, ext_z / 2.0
    z_half = ext_z / 2.0 - margin

    # --- material picker (normalized relative_frequency) ---------------------
    mat_entries = bounds["insert_materials"]["entries"]
    freqs = np.array([float(e.get("relative_frequency", 1.0)) for e in mat_entries],
                     dtype=float)
    if freqs.sum() <= 0:
        raise ValueError("insert_materials relative_frequency must sum to > 0")
    mat_probs = freqs / freqs.sum()

    # --- how many objects ----------------------------------------------------
    n_lo = int(bounds["n_objects"]["min"])
    n_hi = int(bounds["n_objects"]["max"])
    n_objects = int(rng.integers(n_lo, n_hi + 1))

    instructions: list[PaintInstruction] = []
    attempts = 0
    max_attempts = n_objects * 20

    while len(instructions) < n_objects and attempts < max_attempts:
        attempts += 1
        inst = _sample_one_object(rng, bounds, mat_entries, mat_probs,
                                  r_limit, z_half, (cx0, cy0, cz0))
        if inst is not None:
            instructions.append(inst)

    instructions.sort(key=lambda i: i.approx_volume_mm3, reverse=True)
    return Recipe(seed=seed, instructions=instructions)


def _sample_one_object(rng, bounds, mat_entries, mat_probs,
                       r_limit, z_half, center0) -> "PaintInstruction | None":
    cx0, cy0, cz0 = center0

    # material first, then density/activity from THAT material's ranges
    mi = int(rng.choice(len(mat_entries), p=mat_probs))
    entry = mat_entries[mi]
    material_id = int(entry["material_id"])
    density = _uniform(rng, entry["density_g_cm3"])
    activity = _uniform(rng, entry["activity_Bq_per_mL"])

    base = _uniform(rng, _pair(bounds["size_mm"]))
    arx = _uniform(rng, _pair(bounds["aspect_ratio"]))
    ary = _uniform(rng, _pair(bounds["aspect_ratio"]))
    arz = _uniform(rng, _pair(bounds["aspect_ratio"]))

    kind = rng.choice(["ellipsoid", "elliptic_cylinder"])

    if kind == "ellipsoid":
        ax, ay, az = base * arx, base * ary, base * arz
        r_xy = math.hypot(ax, ay)
        if r_xy > r_limit or az > z_half:
            return None
        center = _fit_center(rng, cx0, cy0, cz0, r_limit, r_xy, z_half, az)
        if center is None:
            return None
        vol = (4.0 / 3.0) * math.pi * ax * ay * az
        return PaintInstruction(
            kind="ellipsoid", center_mm=center, material_id=material_id,
            density_g_cm3=density, activity_Bq_per_mL=activity,
            semi_axes_mm=(ax, ay, az), approx_volume_mm3=vol,
        )

    else:  # elliptic_cylinder
        rx, ry = base * arx, base * ary
        height = _uniform(rng, _pair(bounds["cylinder_height_mm"]))
        axis = str(rng.choice(["x", "y", "z"]))
        theta = float(rng.uniform(0.0, 180.0))
        if axis == "z":
            r_xy = math.hypot(rx, ry)
            half_axial = height / 2.0
        else:
            r_xy = math.hypot(max(rx, ry), height / 2.0)
            half_axial = max(rx, ry)
        if r_xy > r_limit or half_axial > z_half:
            return None
        center = _fit_center(rng, cx0, cy0, cz0, r_limit, r_xy, z_half, half_axial)
        if center is None:
            return None
        vol = math.pi * rx * ry * height
        return PaintInstruction(
            kind="elliptic_cylinder", center_mm=center, material_id=material_id,
            density_g_cm3=density, activity_Bq_per_mL=activity,
            rx_mm=rx, ry_mm=ry, height_mm=height, axis=axis, theta_deg=theta,
            approx_volume_mm3=vol,
        )


def _fit_center(rng, cx0, cy0, cz0, r_limit, r_obj_xy, z_half, half_axial):
    r_off = r_limit - r_obj_xy
    z_off = z_half - half_axial
    if r_off < 0 or z_off < 0:
        return None
    rho = r_off * math.sqrt(rng.uniform(0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return (cx0 + rho * math.cos(phi), cy0 + rho * math.sin(phi),
            cz0 + rng.uniform(-z_off, z_off))


def _pair(spec: dict) -> tuple[float, float]:
    return (float(spec["min"]), float(spec["max"]))


def _uniform(rng, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(float(lo), float(hi)))


# ----------------------------------------------------------------------------
# Applying a recipe to the wrapper's builder (the build step)
# ----------------------------------------------------------------------------

def build_voxel_grid(recipe: Recipe, bounds: dict, config: dict):
    """Apply a Recipe through the wrapper's VoxelSpaceBuilder -> validated VoxelGrid.
    Background (air) first, then inserts largest-to-smallest (already sorted)."""
    from mcgpu_pet_wrapper import (
        VoxelSpaceBuilder, voxel_space_shape_xyz, grid_size_mm,
    )
    shape_xyz = voxel_space_shape_xyz(config)
    gsize = grid_size_mm(config)
    n_materials = len(config["mcgpu"]["materials"])
    builder = VoxelSpaceBuilder(shape_xyz, gsize,
                               material_names=[f"mat{i+1}" for i in range(n_materials)])

    bg = bounds["background"]
    builder.fill_background(
        material_id=int(bg["material_id"]),
        density=float(bg["density_g_cm3"]),
        activity_Bq_per_mL=float(bg["activity_Bq_per_mL"]),
    )

    for inst in recipe.instructions:
        if inst.kind == "ellipsoid":
            builder.add_ellipsoid(
                center_mm=inst.center_mm, semi_axes_mm=inst.semi_axes_mm,
                material_id=inst.material_id, density=inst.density_g_cm3,
                activity_Bq_per_mL=inst.activity_Bq_per_mL,
            )
        elif inst.kind == "elliptic_cylinder":
            builder.add_elliptic_cylinder(
                center_mm=inst.center_mm, rx_mm=inst.rx_mm, ry_mm=inst.ry_mm,
                height_mm=inst.height_mm, axis=inst.axis, theta_deg=inst.theta_deg,
                material_id=inst.material_id, density=inst.density_g_cm3,
                activity_Bq_per_mL=inst.activity_Bq_per_mL,
            )
        else:
            raise ValueError(f"unknown kind {inst.kind!r}")

    return builder.build()   # validates


# ----------------------------------------------------------------------------
# GPU-free self-inspection
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    from mcgpu_pet_wrapper import default_config

    bounds_path = sys.argv[1] if len(sys.argv) > 1 else "bounds.json"
    with open(bounds_path) as f:
        bounds = json.load(f)
    config = default_config()

    print("Generating 50 recipes (no GPU)...\n")
    counts = []
    for i in range(50):
        rec = sample_recipe(seed=i, bounds=bounds, config=config)
        counts.append(len(rec.instructions))
        if i < 3:
            print(f"--- seed={i}: {len(rec.instructions)} objects ---")
            for inst in rec.instructions:
                print(f"  {inst.kind:17s} mat={inst.material_id} "
                      f"rho={inst.density_g_cm3:.2f} act={inst.activity_Bq_per_mL:.0f} "
                      f"c=({inst.center_mm[0]:.0f},{inst.center_mm[1]:.0f},{inst.center_mm[2]:.0f})")
            print()
    print(f"object-count over 50: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/len(counts):.1f}")
    print("OK: recipes generate. Next: paint one and view a slice, then run generate.py.")