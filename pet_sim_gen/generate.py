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
  - Atomicity       : stage + simulate in run_dir.tmp, then os.rename to run_dir.
  - Crash-safe log  : append-only JSONL, flushed per line.
  - Fail-fast       : VoxelGrid.validate() runs before the expensive simulation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mcgpu_pet_wrapper import default_config, load_config, build_run, Runner

from .sampler import sample_recipe, build_voxel_grid, Recipe
from .stratification import StratifiedSampler


def seed_for(base_seed: int, index: int) -> int:
    ss = np.random.SeedSequence([base_seed, index])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


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
    out_dir: str | Path = "data",
    bounds: dict | str | Path = "bounds.json",
    config: dict | str | Path | None = None,
    base_seed: int = 0,
    timeout_s: float = 3600.0,
    stratify_key: Optional[Callable[[Recipe], float]] = None,
    stratify_target: Optional[tuple[float, float, int]] = None,
    verbose: bool = True,
) -> dict:
    """Generate `n` simulated samples into `out_dir`. Resumable: rerun to continue.

    stratify_key + stratify_target: optional. If both given, sampling is steered
    to flat coverage over key(recipe) across (min, max, n_bins). Else plain.
    The orchestrator is agnostic to what `key` measures.
    Returns a summary dict.
    """
    out_dir = Path(out_dir)
    out_root = out_dir / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    failures_path = out_dir / "failures.jsonl"

    if not isinstance(bounds, dict):
        bounds = json.loads(Path(bounds).read_text())
    if config is None:
        config = default_config()
    elif not isinstance(config, dict):
        config = load_config(config)

    stratified = stratify_key is not None and stratify_target is not None
    strat = (StratifiedSampler(bounds, config, stratify_key, stratify_target)
             if stratified else None)
    strat_rng = np.random.default_rng(base_seed) if stratified else None

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

        try:
            if stratified:
                recipe = strat.next_recipe(strat_rng)
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
                rho = record["realized"]["mean_density_g_cm3"]
                print(f"[{i:05d}] OK  objs={record['n_objects']} "
                      f"rho_bar={rho:.3f} t={record['wall_time_s']:.1f}s")
        except KeyboardInterrupt:
            if verbose:
                print("\nInterrupted. Rerun the same command to resume.")
            break
        except Exception as e:
            n_failed += 1
            with open(failures_path, "a") as ff:
                ff.write(json.dumps({
                    "index": i, "seed": seed_for(base_seed, i),
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


def main():
    ap = argparse.ArgumentParser(description="Robust PET sim data generation.")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--bounds", type=str, default="bounds.json")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    # Example stratification by the scatter-fraction proxy (examples/sf_proxy.py).
    ap.add_argument("--stratify-sf", action="store_true",
                    help="stratify by the example sf_proxy key")
    ap.add_argument("--sf-min", type=float, default=0.05)
    ap.add_argument("--sf-max", type=float, default=0.46)
    ap.add_argument("--sf-bins", type=int, default=12)
    args = ap.parse_args()

    key = target = None
    if args.stratify_sf:
        from pet_sim_gen.examples.sf_proxy import sf_proxy
        key = sf_proxy
        target = (args.sf_min, args.sf_max, args.sf_bins)

    generate_dataset(
        n=args.n, out_dir=args.out, bounds=args.bounds, config=args.config,
        base_seed=args.base_seed, timeout_s=args.timeout_s,
        stratify_key=key, stratify_target=target,
    )


if __name__ == "__main__":
    main()