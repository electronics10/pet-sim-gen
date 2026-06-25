"""
examples/sf_proxy.py -- ONE example stratification key (scatter-fraction proxy).

This is an EXAMPLE, not core. pet_sim_gen's StratifiedSampler is domain-agnostic
and flattens coverage over any `key_fn(recipe) -> float`. A scatter-correction
user supplies this key; a different task supplies a different one (mean density,
attenuation path length, ...). Copy this as a template for your own key.

The proxy itself is scatter physics, so it deliberately lives here in examples/
rather than in the general package -- keeping the tool free of any task bias.
"""

from __future__ import annotations

import math

import numpy as np

from pet_sim_gen.sampler import Recipe


# Water linear attenuation at 511 keV, cm^-1.
_MU_WATER_511_PER_CM = 0.096


def sf_proxy(recipe: Recipe) -> float:
    """Cheap pre-simulation scatter-fraction proxy:
        s = 1 - exp(-mu_water * rho_bar * d_eff)

    rho_bar : mean insert density (g/cm^3).
    d_eff   : mean effective chord (cm), 2*sqrt(rx*ry) averaged over inserts.

    Heuristic for STEERING coverage only -- never a label. Blind to insert
    position and multiple-scatter saturation, which is fine because it only
    steers sampling. Verify realized scatter coverage downstream from the stored
    sinogram_Trues / sinogram_Scatter files.
    """
    if not recipe.instructions:
        return 0.0
    densities, chords_cm = [], []
    for inst in recipe.instructions:
        densities.append(inst.density_g_cm3)
        if inst.kind == "ellipsoid":
            ax, ay, _ = inst.semi_axes_mm
            d_mm = 2.0 * math.sqrt(ax * ay)
        else:  # elliptic_cylinder
            d_mm = 2.0 * math.sqrt(inst.rx_mm * inst.ry_mm)
        chords_cm.append(d_mm / 10.0)
    rho_bar = float(np.mean(densities))
    d_eff_cm = float(np.mean(chords_cm))
    return 1.0 - math.exp(-_MU_WATER_511_PER_CM * rho_bar * d_eff_cm)