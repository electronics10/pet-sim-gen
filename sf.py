import mcgpu_pet_wrapper as mpw
from pathlib import Path


def scatter_fraction(run_dir):
    cfg = mpw.load_config(run_dir / "config.json")
    true = mpw.read_sinogram(run_dir, cfg)
    scatter = mpw.read_sinogram(run_dir, cfg, scatter=True)
    total = true + scatter

    print(run_dir, "-SF: ", float(scatter.sum()/total.sum()))


root = Path("data/runs")
for run_dir in root.glob("*"):
    if run_dir.is_dir(): scatter_fraction(run_dir)