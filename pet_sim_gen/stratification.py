"""
stratification.py -- generic runtime coverage-flattening for the sampler.

StratifiedSampler wraps the plain sampler so that a generated dataset is roughly
FLAT over a chosen scalar `key_fn(recipe) -> float`. It is fully domain-agnostic:
the key can measure scatter, mean density, attenuation path length, object size,
or anything else the caller supplies. This module knows nothing about any task.

Why this is a runtime component and not a bounds setting: flattening a nonlinear
function of the sampling parameters cannot be achieved by editing static ranges.
It requires accept/reject during generation. Hence it lives here, separate from
bounds_tools (which only shapes the static distribution).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .sampler import sample_recipe, Recipe


class StratifiedSampler:
    """Flatten dataset coverage over `key_fn(recipe)` across target=(min,max,n_bins).

    Mechanism (rejection sampling): draw a candidate from sample_recipe, compute
    its key, find its bin; accept while that bin is at or below the current
    minimum bin count (fills emptiest bins first), else reject and redraw.
    Result: ~equal counts per bin => flat coverage over the key's target range.

    Honesty / limits:
      - Flatness is over the KEY value. If the key is a pre-simulation proxy for
        some post-simulation quantity, the realized coverage of that quantity is
        only approximately flat; verify it downstream from the stored outputs.
      - If `target` max exceeds what the bounds can actually produce, the top
        bins never fill; next_recipe returns None after max_attempts_per_accept
        so the caller can decide what to do (the orchestrator logs and continues).
        The fix for a persistently unreachable target is to WIDEN THE BOUNDS or
        LOWER THE TARGET -- not to raise max_attempts (that only spins longer).

    Resume semantics:
      Acceptance is histogram-coupled: each decision depends on the counts of
      everything accepted so far. That state must be rebuilt on resume, else a
      restarted run double-fills bins it already filled. The orchestrator calls
      `observe(recipe)` for every completed sample on startup to reconstruct
      `bin_counts` before drawing new ones. (Because acceptance is order- and
      history-dependent, an *individual* index is not guaranteed to map to the
      identical recipe across a resume with gaps; the preserved guarantee is
      dataset-level flat coverage, which is the actual purpose.)
    """

    def __init__(self, bounds: dict, config: dict,
                 key_fn: Callable[[Recipe], float],
                 target: tuple[float, float, int],
                 max_attempts_per_accept: int = 200):
        self.bounds = bounds
        self.config = config
        self.key_fn = key_fn
        self.kmin, self.kmax, self.n_bins = target
        if self.kmax <= self.kmin or self.n_bins < 1:
            raise ValueError(f"bad target {target!r}")
        self.max_attempts = max_attempts_per_accept
        self._counts = [0] * self.n_bins

    def _bin_of(self, value: float) -> int | None:
        if value < self.kmin or value >= self.kmax:
            return None
        frac = (value - self.kmin) / (self.kmax - self.kmin)
        return min(int(frac * self.n_bins), self.n_bins - 1)

    def observe(self, recipe: Recipe) -> None:
        """Count an already-realized recipe toward its bin WITHOUT drawing/accepting.

        Used on resume to rebuild bin_counts from completed samples so the run
        continues toward flat coverage instead of re-filling bins. A recipe whose
        key falls outside [kmin, kmax) is ignored (it was never a target sample).
        """
        b = self._bin_of(self.key_fn(recipe))
        if b is not None:
            self._counts[b] += 1

    def next_recipe(self, rng: np.random.Generator) -> Recipe | None:
        """Draw one accepted recipe (flat-coverage steering). Returns None if no
        acceptable recipe was found within max_attempts (e.g. an unreachable
        target bin when the others are already full)."""
        for _ in range(self.max_attempts):
            seed = int(rng.integers(0, 2**63 - 1))
            recipe = sample_recipe(seed=seed, bounds=self.bounds, config=self.config)
            b = self._bin_of(self.key_fn(recipe))
            if b is None:
                continue
            if self._counts[b] <= min(self._counts):
                self._counts[b] += 1
                return recipe
        return None

    @property
    def bin_counts(self) -> list[int]:
        return list(self._counts)