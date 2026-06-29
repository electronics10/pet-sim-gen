"""
examples.py -- ONE example stratification key (scatter-fraction proxy).

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


# _MU_PER_CM = 0.096 # Water linear attenuation at 511 keV, cm^-1. (many materials are around the same)

# def sf_proxy(recipe: Recipe) -> float:
#     """Cheap pre-simulation scatter-fraction proxy:
#         s = 1 - exp(-mu * rho * d)

#     rho : density (g/cm^3).
#     d   : effective chord (cm).

#     Heuristic for STEERING coverage only -- never a label. Blind to insert
#     position and multiple-scatter saturation, which is fine because it only
#     steers sampling. Verify realized scatter coverage downstream from the stored
#     sinogram_Trues / sinogram_Scatter files.
#     """
#     if not recipe.instructions:
#         return 0.0
    
#     # inst = recipe.instructions[0] # first one largest
#     # rho = inst.density_g_cm3
#     # if inst.kind == "ellipsoid":
#     #     ax, ay, _ = inst.semi_axes_mm
#     #     d_cm = math.sqrt(ax * ay) / 10
#     # else:  # elliptic_cylinder
#     #     d_cm = math.sqrt(inst.rx_mm * inst.ry_mm) / 10

#     densities, chords_cm = [], []
#     for inst in recipe.instructions:
#         densities.append(inst.density_g_cm3)
#         if inst.kind == "ellipsoid":
#             ax, ay, _ = inst.semi_axes_mm
#             d_mm = math.sqrt(ax * ay) # times 2 is heuristically wrong
#         else:  # elliptic_cylinder
#             d_mm = math.sqrt(inst.rx_mm * inst.ry_mm) # times 2 is heuristically wrong
#         chords_cm.append(d_mm / 10.0)
#     rho = float(np.mean(densities))
#     d_cm = float(np.mean(chords_cm))
#     return 1.0 - math.exp(-_MU_PER_CM * rho * d_cm)

_MU_MASS_CM2_G = 0.096  # water mass atten. at 511 keV, cm^2/g

def _mean_chord_cm_ellipsoid(a, b, c):  # mm in, cm out
    V = (4/3) * math.pi * a * b * c
    p = 1.6075
    S = 4*math.pi*(((a*b)**p + (a*c)**p + (b*c)**p)/3)**(1/p)
    return (4*V/S) / 10.0

def sf_proxy(recipe: Recipe) -> float:
    if not recipe.instructions:
        return 0.0
    tau = 0.0
    for inst in recipe.instructions:
        if inst.kind == "ellipsoid":
            ax, ay, az = inst.semi_axes_mm
            L = _mean_chord_cm_ellipsoid(ax, ay, az)
        else:
            rx, ry = inst.rx_mm, inst.ry_mm
            h = getattr(inst, "height_mm", 2*max(rx, ry))
            V = math.pi * rx * ry * h
            P = math.pi*(3*(rx+ry) - math.sqrt((3*rx+ry)*(rx+3*ry)))  # Ramanujan
            S = 2*math.pi*rx*ry + P*h
            L = (4*V/S) / 10.0
        tau += _MU_MASS_CM2_G * inst.density_g_cm3 * L
    return 1.0 - math.exp(-tau)