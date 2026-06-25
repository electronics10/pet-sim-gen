"""
pet_sim_gen -- generate large, diverse Monte Carlo PET datasets (via
mcgpu-pet-wrapper) for downstream ML (scatter estimation, attenuation
prediction, reconstruction, ...).

The package is task-agnostic. It produces simulations with separated Trues and
Scatter sinograms, emission images, and a per-sample manifest; what counts as
input vs. label is the downstream consumer's choice.

Public API:
  Recipe, PaintInstruction, sample_recipe, build_voxel_grid   -- the sampler
  generate_dataset                                            -- the orchestrator
  StratifiedSampler                                           -- generic steering
  suggest_bounds_maximal, suggest_bounds_realistic            -- bounds generators
  (suggest_bounds is an alias for suggest_bounds_maximal)

Example stratification keys live in pet_sim_gen.examples (e.g. sf_proxy); they
are templates, not core.
"""

from .sampler import (
    Recipe, PaintInstruction, sample_recipe, build_voxel_grid,
)
from .generate import generate_dataset, seed_for
from .stratification import StratifiedSampler
from .bounds_tools import (
    suggest_bounds, suggest_bounds_maximal, suggest_bounds_realistic,
)

__all__ = [
    "Recipe", "PaintInstruction", "sample_recipe", "build_voxel_grid",
    "generate_dataset", "seed_for",
    "StratifiedSampler",
    "suggest_bounds", "suggest_bounds_maximal", "suggest_bounds_realistic",
]

__version__ = "0.1.0"