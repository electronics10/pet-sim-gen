"""
generate.py -- the frozen, robust, TASK-AGNOSTIC orchestrator.

Runs a resumable, crash-tolerant batch of phantom simulations. Knows nothing
about scatter, attenuation, or any downstream task: it logs only facts true of
any phantom (activity, mass, density, geometry, timing). Task-specific
quantities (e.g. scatter fraction) are computed DOWNSTREAM by the consumer from
the stored sinograms -- never here.

Stratification is optional and passed at the call site (a key_fn + target); the
orchestrator just asks the StratifiedSampler for the next recipe. It does not
know what the key measures.

Robustness invariants:
  - Crash isolation : try/except-continue; MCGPU runs as a subprocess (Runner);
                      Runner's timeout_s kills hangs.
  - Resumability    : a DONE sentinel marks a complete sample; skip if present.
                      Stratified runs also rebuild bin counts from completed
                      samples on startup so resume stays flat (see
                      _rebuild_strat_state); per-index RNGs keep candidate draws
                      stable across interruption.
  - Atomicity       : stage + simulate in run_dir.tmp, then os.rename to run_dir.
  - Crash-safe log  : append-only JSONL, flushed per line.
  - Fail-fast       : VoxelGrid.validate() runs before the expensive simulation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mcgpu_pet_wrapper import default_config, load_config, build_run, Runner

from .sampler import sample_recipe, build_voxel_grid, Recipe
from .stratification import StratifiedSampler


def seed_for(base_seed: int, index: int) -> int:
    ss = np.random.SeedSequence([base_seed, index])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _describe_key_fn(key_fn: Optional[Callable]) -> Optional[dict]:
    """Provenance for a stratify_key callable.

    A function cannot be serialized to JSON; we record what lets a reader (a)
    locate it and (b) DETECT DRIFT. We capture the fully-qualified name plus a
    sha256 of the defining source file, because the function's behavior depends
    on constants in that file (e.g. sf_proxy's 0.6 factor, _MU_LIN_PER_CM_WATER).
    The reference resolving later does NOT guarantee identical behavior; the hash
    is what tells you whether the source changed since this dataset was made.
    """
    if key_fn is None:
        return None
    info: dict = {
        "qualname": f"{getattr(key_fn, '__module__', '?')}."
                    f"{getattr(key_fn, '__qualname__', repr(key_fn))}",
    }
    try:
        src_file = inspect.getsourcefile(key_fn) or inspect.getfile(key_fn)
        info["source_file"] = src_file
        if src_file and Path(src_file).exists():
            data = Path(src_file).read_bytes()
            info["source_sha256"] = hashlib.sha256(data).hexdigest()
            info["source_n_bytes"] = len(data)
    except (TypeError, OSError):
        # builtins, C-extensions, lambdas defined in REPL, etc.
        info["source_file"] = None
        info["note"] = "source unavailable; only qualname recorded (may be a lambda/builtin)"
    return info


def _write_run_config(out_dir: Path, *, n: int, base_seed: int, timeout_s: float,
                      config: dict, bounds: dict, stratified: bool,
                      stratify_key: Optional[Callable],
                      stratify_target: Optional[tuple]) -> None:
    """Write ./data/run_config.json: the call-site args that produced this dataset.

    Atomic (tmp + rename) and append-history-aware: if a file already exists
    (resume, or a second batch into the same dir) the new record is appended to
    a 'runs' list rather than clobbering, so the folder keeps the full lineage.
    Per-sample recipes already live in run_*/recipe.json; this fills the ONE gap
    those don't cover -- the configuration the orchestrator never otherwise saves.
    """
    record = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_requested": n,
        "base_seed": base_seed,
        "timeout_s": timeout_s,
        "stratified": stratified,
        "stratify_target": list(stratify_target) if stratify_target else None,
        "stratify_key": _describe_key_fn(stratify_key),
        "bounds": bounds,
        "config": config,
        "python": sys.version,
        "numpy": np.__version__,
    }
    path = out_dir / "run_config.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            runs = existing.get("runs", [existing]) if isinstance(existing, dict) else [existing]
        except (json.JSONDecodeError, OSError):
            runs = []
        runs.append(record)
        payload = {"runs": runs}
    else:
        payload = {"runs": [record]}
    tmp = out_dir / "run_config.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.rename(path)


def _rebuild_strat_state(strat: StratifiedSampler, out_root: Path,
                         verbose: bool) -> None:
    """Rebuild the stratifier's bin counts from completed samples on disk.

    Disk (a run_XXXXX dir with both DONE and recipe.json) is the source of truth
    for what completed -- more reliable than the manifest, which can lag a DONE by
    one line after a crash. One-time, cheap relative to the simulations.
    """
    n = 0
    for run_dir in sorted(out_root.glob("run_*")):
        if not (run_dir / "DONE").exists():
            continue
        recipe_file = run_dir / "recipe.json"
        if not recipe_file.exists():
            continue
        strat.observe(Recipe.from_dict(json.loads(recipe_file.read_text())))
        n += 1
    if verbose and n:
        print(f"Stratified resume: rebuilt bin counts from {n} completed "
              f"sample(s) -> bin_counts={strat.bin_counts}")


def _commit_sample(index, recipe, vg, result, tmp_dir, run_dir) -> dict:
    """Task-agnostic realized facts + atomic commit. No task-specific metrics."""
    record = {
        "index": index,
        "seed": recipe.seed,
        "n_objects": len(recipe.instructions),
        "realized": {
            "total_activity_Bq": vg.total_activity_Bq,
            "total_mass_g": vg.total_mass_g,
            "mean_density_g_cm3": float(vg.density.mean()),
        },
        "wall_time_s": result.wall_time_s,
        "returncode": result.returncode,
        "recipe": recipe.to_dict(),
    }
    (tmp_dir / "recipe.json").write_text(json.dumps(recipe.to_dict(), indent=2))
    (tmp_dir / "DONE").write_text("ok\n")
    tmp_dir.rename(run_dir)
    return record


def _simulate_recipe(recipe, index, out_root, bounds, config, timeout_s):
    """build -> stage -> simulate -> commit. Raises on any failure."""
    run_dir = out_root / f"run_{index:05d}"
    tmp_dir = out_root / f"run_{index:05d}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    vg = build_voxel_grid(recipe, bounds, config)        # validate() fails fast
    build_run(tmp_dir, config, vg)
    result = Runner()(tmp_dir, on_existing="error", verbose=False, timeout_s=timeout_s)
    return _commit_sample(index, recipe, vg, result, tmp_dir, run_dir)


def generate_dataset(
    n: int,
    config: dict,
    bounds: dict,
    out_dir: str | Path = "data",
    base_seed: int = 0,
    timeout_s: float = 3600.0,
    stratify_key: Optional[Callable[[Recipe], float]] = None,
    stratify_target: Optional[tuple[float, float, int]] = None,
    verbose: bool = True,
) -> dict:
    """Generate `n` simulated samples into `out_dir`. Resumable: rerun to continue.

    bounds: dict, path, or None. None -> suggest_bounds_maximal(config), a broad,
    FOV-consistent scaffold (printed as a notice; tighten it for production runs).
    Pass a dict (e.g. an edited suggest_bounds_maximal output) or a path to a JSON
    file to use your own object-distribution policy.

    stratify_key + stratify_target: optional. If both given, sampling is steered
    to flat coverage over key(recipe) across (min, max, n_bins). Else plain.
    The orchestrator is agnostic to what `key` measures.
    Returns a summary dict.
    """
    # Fail fast on stratification mis-wiring, before any expensive work, so a
    # mistake surfaces once with a fix -- not as N identical per-sample failures.
    if (stratify_key is None) != (stratify_target is None):
        raise ValueError(
            "stratify_key and stratify_target must be given together "
            "(got one without the other); pass both, or neither for plain sampling."
        )
    if stratify_key is not None and not callable(stratify_key):
        raise TypeError(
            f"stratify_key must be callable (recipe -> float), got "
            f"{type(stratify_key).__name__}. A common cause is importing the example "
            "MODULE instead of the function -- use "
            "`from pet_sim_gen.examples import sf_proxy`."
        )

    out_dir = Path(out_dir)
    out_root = out_dir / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    failures_path = out_dir / "failures.jsonl"

    stratified = stratify_key is not None and stratify_target is not None
    strat = (StratifiedSampler(bounds, config, stratify_key, stratify_target)
             if stratified else None)
    # Stratification is histogram-coupled, so its bin counts must be rebuilt from
    # already-completed samples on resume (else a restart double-fills bins).
    if stratified:
        _rebuild_strat_state(strat, out_root, verbose)

    # Persist the call-site configuration that the data folder otherwise loses.
    # Per-sample recipe.json files capture WHAT was drawn; this captures the
    # bounds/config/seed/key that decided why. Written once, before the loop, so
    # an immediate crash still leaves the instructions on disk.
    _write_run_config(out_dir, n=n, base_seed=base_seed, timeout_s=timeout_s,
                      config=config, bounds=bounds, stratified=stratified,
                      stratify_key=stratify_key, stratify_target=stratify_target)
    if verbose:
        print(f"Wrote run_config.json (generation instructions) -> "
              f"{out_dir / 'run_config.json'}")

    n_done = n_failed = n_skipped = 0
    if verbose:
        print(f"Batch: {n} samples ({'stratified' if stratified else 'plain'}) "
              f"-> {out_root}")
        print("Resume-safe: rerun to continue after interruption.\n")

    for i in range(n):
        run_dir = out_root / f"run_{i:05d}"
        if (run_dir / "DONE").exists():
            n_skipped += 1
            continue

        recipe = None
        try:
            if stratified:
                # Per-index RNG (mirrors the plain path's seed_for): a sample's
                # candidate stream no longer depends on how many draws earlier
                # samples rejected, so it is stable across interruption/resume.
                rng_i = np.random.default_rng(seed_for(base_seed, i))
                recipe = strat.next_recipe(rng_i)
                if recipe is None:
                    raise RuntimeError(
                        "stratifier could not place a sample (target bin "
                        f"unreachable given bounds?). bin_counts={strat.bin_counts}. "
                        "Widen bounds or lower the target."
                    )
            else:
                recipe = sample_recipe(seed=seed_for(base_seed, i),
                                       bounds=bounds, config=config)

            record = _simulate_recipe(recipe, i, out_root, bounds, config, timeout_s)
            with open(manifest_path, "a") as mf:
                mf.write(json.dumps(record) + "\n"); mf.flush()
            n_done += 1
            if verbose:
                print(f"[{i:05d}] OK  objs={record['n_objects']} "
                      f"t={record['wall_time_s']:.1f}s")
        except KeyboardInterrupt:
            if verbose:
                print("\nInterrupted. Rerun the same script to resume.")
            break
        except Exception as e:
            n_failed += 1
            with open(failures_path, "a") as ff:
                ff.write(json.dumps({
                    # The ACTUAL seed of the drawn recipe (matches manifest); for
                    # stratified runs this differs from seed_for(base_seed, i).
                    # None if the failure happened before a recipe was drawn.
                    "index": i, "seed": (recipe.seed if recipe is not None else None),
                    "error": repr(e), "traceback": traceback.format_exc(),
                }) + "\n"); ff.flush()
            tmp_dir = out_root / f"run_{i:05d}.tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if verbose:
                print(f"[{i:05d}] FAIL {type(e).__name__}: {e} (logged, continuing)")
            continue

    if verbose:
        print(f"\nDone. completed={n_done} skipped={n_skipped} failed={n_failed}")
        print(f"manifest: {manifest_path}")
        if n_failed:
            print(f"failures: {failures_path}")
    return {"completed": n_done, "skipped": n_skipped, "failed": n_failed,
            "manifest": str(manifest_path)}