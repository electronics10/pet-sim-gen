from __future__ import annotations
from pathlib import Path
import json

from mcgpu_pet_wrapper import radial_fov_mm, voxel_space_extent_mm


def load_bounds(path: str | Path) -> dict:
    with open(Path(path), 'r') as file:
        bounds = json.load(file)
    return bounds

def save_bounds(bounds: dict, path: str | Path = "bounds.json") -> None:
    with open(Path(path), 'w') as file:
        json.dump(bounds, file, indent=2)
    print("Bounds save to", path)

# Fraction of the radial FOV radius a single object's base size may reach.
def suggest_bounds(config: dict) -> dict:
    """Broad, FOV-consistent bounds derived from `config`. A STARTING POINT.
    - one insert_materials entry per material id >= 2 (id 1 = air = background),
      each with a broad density/activity span and relative_frequency 1.0.
    The user is expected to tighten ranges/frequencies to their task.
    """
    fov_radius = 0.8 * radial_fov_mm(config) / 2.0 # multiplied by aspect ratio downstream
    size_max = fov_radius
    size_min = min(2.0, 0.25 * size_max)
    _, _, ext_z = voxel_space_extent_mm(config)
    cyl_h_max = ext_z

    materials = config["mcgpu"]["materials"]   # index 0 -> material_id 1 (air)
    entries = []
    for idx in range(1, len(materials)):       # skip id 1 (air)
        entries.append({
            "material_id": idx + 1,
            "relative_frequency": 1.0,
            "density_g_cm3": [0.3, 1.9],
            "activity_Bq_per_mL": [0.0, 20000.0],
        })
    if not entries:
        raise ValueError(
            "config has no insert materials (need >= 2 materials; id 1 is air)."
        )

    return {
        "_doc": "Object-distribution spec. Auto-suggested by suggest_bounds(config). Broad ranges; tighten to taste.",
        "n_objects": {
            "_doc": "Number of inserts per phantom, uniform integer in [min, max] inclusive.",
            "min": 1, "max": 6},
        "size_mm": {
            "_doc": "Base half-size (sphere-equivalent radius) of an object, uniform in [min, max] mm.",
            "min": round(size_min, 2), "max": round(size_max, 2)},
        "aspect_ratio": {
            "_doc": "Per-axis multiplier on base size, uniform in [min, max], drawn independently per axis. Equal multipliers -> sphere/cylinder (degenerate); spread -> elongated.",
            "min": 0.5, "max": 2.0},
        "cylinder_height_mm": {
            "_doc": "Height of elliptic cylinders, uniform in [min, max] mm.",
            "min": 8.0, "max": round(cyl_h_max, 2)},
        "background": {
            "_doc": "Whole volume filled with this first (air). Painted before all inserts so no voxel is left as vacuum (material_id 0 is rejected by VoxelGrid.validate). material_id 1 = first material in the config list.",
            "material_id": 1, 
            "density_g_cm3": 0.0012,
            "activity_Bq_per_mL": 0.0},
        "insert_materials": {
            "_doc": "One entry per material an insert may use. For each: the density range and activity range sampled when that material is chosen, and relative_frequency (how often it is chosen, normalized internally over the materials listed here). 'Coupling' is implicit: tighten a material's activity range to make it effectively cold/hot; widen for decoupled brightness.",
            "entries": entries},
        "fov_margin_mm": {
            "_doc": "Inserts kept at least this far inside the transaxial radius limit, so objects do not spill past the scanner's sensitive region.",
            "value": 0.0},
    }


# Optional per-material physiological defaults, keyed by material *name substring*
# found in the config's materials filenames. Extend as needed. These are rough
# 511-keV-relevant values; a fuller implementation would read real tables.
_PHYSIOLOGICAL = {
    # name_substring: (density_range, activity_range)
    "water":     ((0.95, 1.05), (1000.0, 8000.0)),
    "adipose":   ((0.90, 0.97), (0.0, 3000.0)),
    "spongiosa": ((1.05, 1.20), (0.0, 1500.0)),
    "lung":      ((0.25, 0.50), (0.0, 2000.0)),
    "bone":      ((1.40, 1.92), (0.0, 800.0)),
}


def suggest_bounds_realistic(config: dict) -> dict:
    """Physiological bounds: per-material density/activity matched to tissue-like
    values, inferred from each material's filename.

    STATUS: partial / best-effort. This maps known material-name substrings (see
    _PHYSIOLOGICAL) to plausible 511-keV density and activity ranges. Materials
    whose names are not recognized fall back to the broad maximal span and emit
    no error (so it always returns a usable dict). A fuller implementation would
    read actual ICRP/PENELOPE tables and set realistic shapes; that is non-
    trivial and may not match every task (scatter work, e.g., prefers physical
    validity over anatomical realism), so it is intentionally left as a hook.

    Raises nothing; unknown materials just get broad defaults.
    """
    base = suggest_bounds(config)
    materials = config["mcgpu"]["materials"]
    for entry in base["insert_materials"]["entries"]:
        mid = entry["material_id"]
        name = materials[mid - 1].lower()   # 1-based id -> 0-based list
        for key, (drange, arange) in _PHYSIOLOGICAL.items():
            if key in name:
                entry["density_g_cm3"] = list(drange)
                entry["activity_Bq_per_mL"] = list(arange)
                break
        # unknown material name -> keep the broad maximal span (no error)
    base["_doc"] = ("Object-distribution spec. Auto-suggested by suggest_bounds_realistic(config).")
    return base