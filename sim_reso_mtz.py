#!/usr/bin/env python3
"""
sim_reso_mtz.py — Fast empirical nanoBragg2 / XDS / Aimless / XDSCONV simulator for MTZ files (v2.2).

Given a ground-truth structure-factor MTZ (F0) and a desired resolution limit d_min (Å),
this tool:
  1. Maps d_min to the calibrated Wilson B-factor B_Wilson(d_min), detector edge/corner geometry
     (d_edge, d_corner, d_mtz_min), 18-knot deterministic XDS spot-profile transfer curve T_xds(u),
     high-u within-shell profile/summation capture non-linearity delta_nl_u1, Voronoi detector/beamstop
     completeness p_obs(u), outer-corner background offset beta_bg(u) + per-seed radial-ring background
     fluctuation sigma_bg_cor, and 3-knot u-dependent merged intensity & anomalous error parameters
     calibrated strictly on training seeds .01-.05 (575 datasets; seeds .06-.10 held out) in 1aho_Bfac/nanoBragg2.
  2. Estimates the input F0's intrinsic Wilson B-factor and overall intensity scale so the
     simulation generalizes automatically to arbitrary overall scales or pre-blurred inputs.
  3. Keeps physical ground truth F_true = F0 * exp(-Delta_B * s^2) pure (for --keep-true), while
     applying XDS's deterministic spot-profile capture I_xds = T_xds(u) * f_nl(u, I/I_Wil) * I_true
     (disableable via --ideal-transfer) and realistic SIGIMEAN values calibrated in the same frame.
  4. Draws Friedel-mate intensities (I+, I-) using orthogonal sum/difference random scatter matching
     BOTH the shell-scaled random error budget (a_stat(u) * SIGIMEAN, c_sysI(u) * I_xds, low-angle
     shell bleed floor c_wbg) and anomalous mate differences (a_ind * SIGDANO_I + sqrt(2) * c_sys_ind * I_xds),
     then converts (I+, SIGI+, I-, SIGI-) -> (F+, SIGF+, F-, SIGF-, F, SIGF, DANO, SIGDANO) via
     Wolfgang Kabsch's exact XDSCONV (WILSON_STATISTICS=TRUE, CCP4_I) clamped sliding-window French-Wilson & merging.

Dependencies: numpy, gemmi (runs under /programs/ccp4-9/bin/ccp4-python or standard Python).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np


CRITERION_KEY: Dict[str, str] = {
    "cc12": "d_cc12",       # Aimless CC1/2 = 0.3 (col 3 of reso_summary_10x.txt)
    "isig1.5": "d_isig15",  # Aimless Mn(I/sd) = 1.5
    "isig2.0": "d_isig20",  # Aimless Mn(I/sd) = 2.0
    "ccright": "d_ccright", # Sliding-window CC(I_obs, I_true) = 0.3
    "maprms": "d_maprms",   # Hard cutoff minimizing real-space map RMS error
}

CRITERION_COL_V1: Dict[str, int] = {
    "cc12": 1,
    "isig1.5": 2,
    "isig2.0": 3,
    "ccright": 4,
    "maprms": 5,
}


def f_prot_sq(s2: np.ndarray) -> np.ndarray:
    """Average protein atomic scattering factor squared <f^2(s^2)> (C:0.63, N:0.17, O:0.20),
    where s2 = (sin(theta)/lambda)^2 = 1 / (4 * d^2)."""
    fc = (
        2.3100 * np.exp(-20.8439 * s2)
        + 1.0200 * np.exp(-10.2075 * s2)
        + 1.5886 * np.exp(-0.5687 * s2)
        + 0.8650 * np.exp(-51.6512 * s2)
        + 0.2156
    )
    fn = (
        12.2126 * np.exp(-0.0057 * s2)
        + 3.1322 * np.exp(-9.8933 * s2)
        + 2.0125 * np.exp(-28.9975 * s2)
        + 1.1663 * np.exp(-0.5826 * s2)
        - 11.5290
    )
    fo = (
        3.0485 * np.exp(-13.2771 * s2)
        + 2.2868 * np.exp(-5.7011 * s2)
        + 1.5463 * np.exp(-0.3239 * s2)
        + 0.8670 * np.exp(-32.9089 * s2)
        + 0.2508
    )
    return 0.63 * fc * fc + 0.17 * fn * fn + 0.20 * fo * fo


def pava_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm (PAVA) for isotonic non-decreasing smoothing."""
    n = len(y)
    vals = y.astype(np.float64).tolist()
    wts = [1.0] * n
    counts = [1] * n
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            tot_w = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot_w
            wts[i] = tot_w
            counts[i] += counts[i + 1]
            del vals[i + 1]
            del wts[i + 1]
            del counts[i + 1]
            while i > 0 and vals[i - 1] > vals[i]:
                tot_w = wts[i - 1] + wts[i]
                vals[i - 1] = (vals[i - 1] * wts[i - 1] + vals[i] * wts[i]) / tot_w
                wts[i - 1] = tot_w
                counts[i - 1] += counts[i]
                del vals[i]
                del wts[i]
                del counts[i]
                i -= 1
        else:
            i += 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for v, c in zip(vals, counts):
        out[pos : pos + c] = v
        pos += c
    return out


def analytic_b_wilson(d_min: float, criterion: str) -> float:
    """Smooth power-law fit B_Wilson(d_min) used for out-of-table extrapolation."""
    if criterion == "cc12":
        b = 19.90 + 5.642 * (d_min ** 3.065) - 0.0155 * (d_min ** 6.129)
    elif criterion == "isig1.5":
        b = 14.52 + 6.858 * (d_min ** 2.823) - 0.0172 * (d_min ** 5.646)
    elif criterion == "isig2.0":
        b = 15.29 + 6.041 * (d_min ** 2.902) - 0.0142 * (d_min ** 5.804)
    elif criterion == "ccright":
        b = 15.19 + 9.520 * (d_min ** 2.666) - 0.0300 * (d_min ** 5.332)
    else:  # maprms
        b = 15.50 + 13.50 * (d_min ** 2.550)
    return max(2.0, float(b))


def hat_weights_3(u_arr: np.ndarray, knots: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-linear 3-knot hat weights summing to 1.0 across u = d_edge / d."""
    u0, u1, u2 = float(knots[0]), float(knots[1]), float(knots[2])
    w0 = np.where(u_arr <= u0, 1.0, np.where(u_arr < u1, (u1 - u_arr) / (u1 - u0), 0.0))
    w2 = np.where(u_arr >= u2, 1.0, np.where(u_arr > u1, (u_arr - u1) / (u2 - u1), 0.0))
    w1 = np.maximum(0.0, 1.0 - w0 - w2)
    return w0, w1, w2


def interpolate_calibration(
    cal: dict,
    d_min: float,
    criterion: str = "cc12",
    b_wilson_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Map desired d_min (or explicit B_Wilson) to calibrated nanoBragg2 v2.2 parameters."""
    raw_rows = cal["rows"]
    is_v2 = isinstance(raw_rows[0], dict)

    if is_v2:
        bw_grid = np.array([float(r["B_Wilson"]) for r in raw_rows], dtype=np.float64)
        ckey = CRITERION_KEY[criterion]
        d_raw = np.array([float(r[ckey]) for r in raw_rows], dtype=np.float64)
    else:
        arr_v1 = np.asarray(raw_rows, dtype=np.float64)
        bw_grid = arr_v1[:, 0]
        d_raw = arr_v1[:, CRITERION_COL_V1[criterion]]

    d_mono = pava_nondecreasing(d_raw) + 1e-6 * (bw_grid - bw_grid[0])

    if b_wilson_override is not None:
        bw = float(b_wilson_override)
    else:
        if d_min <= d_mono[0]:
            b_hi = analytic_b_wilson(d_min, criterion)
            b_hi0 = analytic_b_wilson(float(d_mono[0]), criterion)
            bw = max(2.0, float(bw_grid[0]) + (b_hi - b_hi0))
        elif d_min >= d_mono[-1]:
            b_lo = analytic_b_wilson(d_min, criterion)
            b_lo0 = analytic_b_wilson(float(d_mono[-1]), criterion)
            bw = float(bw_grid[-1]) + max(0.0, b_lo - b_lo0)
        else:
            bw = float(np.interp(d_min, d_mono, bw_grid))

    if not is_v2:
        # Fallback support for legacy v1 JSON
        sc = cal.get("scatter_params", {})
        interp_row = np.array([np.interp(bw, bw_grid, arr_v1[:, j]) for j in range(arr_v1.shape[1])])
        return {
            "B_Wilson": bw,
            "d_crit": float(np.interp(bw, bw_grid, d_mono)),
            "d_edge": float(interp_row[6]),
            "d_corner": float(interp_row[7]),
            "d_mtz_min": float(interp_row[7]),
            "p_miss_low12": 0.0,
            "sig0": float(interp_row[9]),
            "b1_u2": float(interp_row[10]),
            "w_wat": float(interp_row[11]),
            "g_shot": float(interp_row[12]),
            "c_sig": float(interp_row[13]),
            "b2_cor": 0.0,
            "rms_ln_si": float(interp_row[14]),
            "rms_ln_si_cor": float(interp_row[14]),
            "a_stat": float(sc.get("a_stat", 1.00)),
            "a_stat_u1": float(sc.get("a_stat", 1.00)),
            "a_stat_u2": float(sc.get("a_stat", 1.00)),
            "c_sysI": float(sc.get("c_sysI", 0.032)),
            "c_sysI_u1": float(sc.get("c_sysI", 0.032)),
            "c_sysI_u2": float(sc.get("c_sysI", 0.032)),
            "c_wbg": 0.0,
            "a_ind": 0.94,
            "c_sys_ind": 0.012,
            "delta_nl_u1": 0.0,
            "sigma_bg_cor": 0.0,
            "u_t_knots": [0.0, 1.3],
            "t_xds": [1.0, 1.0],
            "u_p_knots": [0.0, 1.35],
            "p_obs": [1.0, 1.0],
            "u_bg_knots": [0.0, 1.3],
            "beta_bg": [0.0, 0.0],
            "u_sc_knots": [0.30, 0.68, 0.94],
        }

    scalar_defaults = {
        "c_wbg": 0.0,
        "delta_nl_u1": 0.0,
        "sigma_bg_cor": 0.0,
    }

    def _row_val(r: dict, k: str) -> float:
        if k in r:
            return float(r[k])
        if k == "rms_ln_si_cor":
            return float(r.get("rms_ln_si", 0.14))
        if k == "a_stat_u2":
            return float(r.get("a_stat_u1", r.get("a_stat", 1.0)))
        if k == "c_sysI_u2":
            return float(r.get("c_sysI_u1", r.get("c_sysI", 0.032)))
        return float(scalar_defaults.get(k, 0.0))

    scalar_keys = (
        "d_edge",
        "d_corner",
        "d_mtz_min",
        "p_miss_low12",
        "sig0",
        "b1_u2",
        "w_wat",
        "g_shot",
        "c_sig",
        "b2_cor",
        "rms_ln_si",
        "rms_ln_si_cor",
        "a_stat",
        "a_stat_u1",
        "a_stat_u2",
        "c_sysI",
        "c_sysI_u1",
        "c_sysI_u2",
        "c_wbg",
        "a_ind",
        "c_sys_ind",
        "delta_nl_u1",
        "sigma_bg_cor",
    )
    vec_keys = ("t_xds", "p_obs", "beta_bg")

    out: Dict[str, Any] = {
        "B_Wilson": bw,
        "u_t_knots": [float(x) for x in cal["u_t_knots"]],
        "u_p_knots": [float(x) for x in cal["u_p_knots"]],
        "u_bg_knots": [float(x) for x in cal["u_bg_knots"]],
        "u_sc_knots": [float(x) for x in cal.get("u_sc_knots", [0.30, 0.68, 0.94])],
    }

    if bw <= bw_grid[0]:
        r0 = raw_rows[0]
        d_ref = float(d_mono[0])
        scale_d = d_min / d_ref if b_wilson_override is None else math.sqrt(max(bw, 2.0) / bw_grid[0])
        for k in scalar_keys:
            out[k] = _row_val(r0, k)
        out["d_edge"] *= min(scale_d, 1.0)
        out["d_corner"] *= min(scale_d, 1.0)
        out["d_mtz_min"] *= min(scale_d, 1.0)
        out["d_crit"] = d_min if b_wilson_override is None else d_ref * scale_d
        for vk in vec_keys:
            out[vk] = np.asarray(r0[vk], dtype=np.float64).copy()
    elif bw >= bw_grid[-1]:
        rl = raw_rows[-1]
        d_ref = float(d_mono[-1])
        d_crit = d_min if b_wilson_override is None else d_ref * ((bw / bw_grid[-1]) ** 0.35)
        scale_d = max(d_crit / d_ref, 1.0)
        for k in scalar_keys:
            out[k] = _row_val(rl, k)
        out["d_edge"] *= scale_d
        out["d_corner"] *= scale_d
        out["d_mtz_min"] *= scale_d
        out["sig0"] /= scale_d ** 1.5
        out["g_shot"] /= scale_d ** 1.0
        out["d_crit"] = d_crit
        for vk in vec_keys:
            out[vk] = np.asarray(rl[vk], dtype=np.float64).copy()
    else:
        out["d_crit"] = float(np.interp(bw, bw_grid, d_mono))
        for k in scalar_keys:
            vals = np.array([_row_val(r, k) for r in raw_rows], dtype=np.float64)
            out[k] = float(np.interp(bw, bw_grid, vals))
        for vk in vec_keys:
            mat = np.asarray([r[vk] for r in raw_rows], dtype=np.float64)
            out[vk] = np.array([np.interp(bw, bw_grid, mat[:, j]) for j in range(mat.shape[1])], dtype=np.float64)

    return out


# ==============================================================================
# Physical nanoBragg2 / nonBragg Reference Constants & Experimental Scalings
# ==============================================================================
PILATUS_6M_WIDTH_MM = 423.636   # 2463 px * 0.172 mm
PILATUS_6M_DIAG_MM = 606.946    # corner diagonal in nanoBragg/run_the_simulation.com
PILATUS_6M_PIXEL_MM = 0.172

NB2_REF: Dict[str, float] = {
    "xtal_vol_um3": 17549.0,       # 21^3 + 17^3 + 15^3 um^3 (3 composite sub-crystals)
    "incoh_vol_um3": 9261.0,       # 21^3 um^3 (-samplesize in render_par.com)
    "beam_size_um": 30.0,          # 30e-3 mm square top-hat beam (900 um^2)
    "flux_phs": 1.0e12,            # photons/s in beam (flux density = 1e12 / 900 ph/um^2/s)
    "exposure_s": 36.0,            # 360 images * 0.1 s/image
    "wedge_deg": 360.0,            # 360 deg total rotation
    "osc_deg": 1.0,                # 1.0 deg/image
    "wavelength_A": 0.977794,      # 12,680 eV
    "water_um": 25.0,              # 25e-3 mm liquid water background
    "air_mm": 25.0,                # 25 mm air path background
    "mosaicity_deg": 0.20,         # 0.20 deg mosaic spread
    "divergence_mrad": 0.10,       # 0.10 mrad beam divergence
    "dispersion_pct": 0.20,        # 0.20 % dlambda/lambda spectral dispersion
    "calib_err": 0.03,             # 3 % spatially correlated pixel-gain error
    "dose_mgy": 0.0,               # zero radiation damage in baseline nanoBragg2
    "rad_b_per_mgy": 0.70,         # ~0.7 A^2 / MGy average wedge global B growth
    "cell_vol_A3": 56230.683,      # 1AHO: 45.9 * 40.7 * 30.1 A^3
}

# Exact molecular structure-factor curves F(s) vs s = sin(theta)/lambda (A^-1)
# sampled directly from nanoBragg/water.stol and nanoBragg/air.stol:
_STOL_GRID = np.array(
    [
        0.00, 0.04, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.20, 0.22, 0.25, 0.28, 0.31, 0.35, 0.40, 0.45, 0.50, 0.58, 0.70,
    ],
    dtype=np.float64,
)
_F_WATER_GRID = np.array(
    [
        2.570, 2.595, 3.047, 3.743, 5.093, 7.049, 7.755, 8.009, 7.821, 7.388,
        6.753, 6.743, 5.844, 4.549, 4.307, 4.356, 4.117, 3.634, 3.150, 2.550, 2.050,
    ],
    dtype=np.float64,
)
_F_AIR_GRID = np.array(
    [
        14.095, 13.560, 12.360, 11.545, 10.650, 9.755, 9.315, 8.885, 8.470, 8.075,
        7.355, 6.745, 5.990, 5.455, 5.090, 4.775, 4.530, 4.365, 4.210, 3.950, 3.600,
    ],
    dtype=np.float64,
)


def dist_from_dedge(d_edge: float, wavelength: float) -> float:
    """Convert inscribed detector edge resolution d_edge (A) to Pilatus 6M distance (mm)."""
    sin_th = min(0.999, max(1e-6, wavelength / (2.0 * max(d_edge, 0.25))))
    two_th = 2.0 * math.asin(sin_th)
    tan_2th = math.tan(two_th)
    if tan_2th <= 0.0:
        return 2000.0
    return (0.5 * PILATUS_6M_WIDTH_MM) / tan_2th


def dedge_dcorner_from_dist(dist_mm: float, wavelength: float) -> Tuple[float, float]:
    """Compute Pilatus 6M (d_edge, d_corner) in Angstroms from distance (mm) and wavelength (A)."""
    d_mm = max(float(dist_mm), 15.0)
    two_th_e = math.atan2(0.5 * PILATUS_6M_WIDTH_MM, d_mm)
    two_th_c = math.atan2(0.5 * PILATUS_6M_DIAG_MM, d_mm)
    d_edg = wavelength / (2.0 * math.sin(0.5 * two_th_e))
    d_cor = wavelength / (2.0 * math.sin(0.5 * two_th_c))
    return float(d_edg), float(d_cor)


def interpolate_calibration_by_dedge(cal: dict, d_edge_ref_equiv: float, criterion: str = "cc12") -> Dict[str, Any]:
    """Look up baseline detector/XDS/background calibration parameters at a physical detector
    distance corresponding to reference-wavelength edge resolution d_edge_ref_equiv."""
    raw_rows = cal["rows"]
    if not isinstance(raw_rows[0], dict):
        return interpolate_calibration(cal, d_edge_ref_equiv, criterion=criterion)

    bw_grid = np.array([float(r["B_Wilson"]) for r in raw_rows], dtype=np.float64)
    de_grid = pava_nondecreasing(np.array([float(r["d_edge"]) for r in raw_rows], dtype=np.float64))
    de_grid = de_grid + 1e-7 * (bw_grid - bw_grid[0])

    if d_edge_ref_equiv <= de_grid[0]:
        p = interpolate_calibration(cal, float(raw_rows[0][CRITERION_KEY[criterion]]), criterion=criterion, b_wilson_override=float(bw_grid[0]))
        sc = max(0.25, d_edge_ref_equiv / float(de_grid[0]))
        p["d_edge"] *= sc
        p["d_corner"] *= sc
        p["d_mtz_min"] *= sc
        p["d_crit"] *= sc
        p["sig0"] /= sc ** 1.5
        p["g_shot"] /= sc ** 1.0
        return p
    if d_edge_ref_equiv >= de_grid[-1]:
        p = interpolate_calibration(cal, float(raw_rows[-1][CRITERION_KEY[criterion]]), criterion=criterion, b_wilson_override=float(bw_grid[-1]))
        sc = max(1.0, d_edge_ref_equiv / float(de_grid[-1]))
        p["d_edge"] *= sc
        p["d_corner"] *= sc
        p["d_mtz_min"] *= sc
        p["d_crit"] *= sc
        p["sig0"] /= sc ** 1.5
        p["g_shot"] /= sc ** 1.0
        return p

    bw_equiv = float(np.interp(d_edge_ref_equiv, de_grid, bw_grid))
    return interpolate_calibration(cal, d_edge_ref_equiv, criterion=criterion, b_wilson_override=bw_equiv)


def nonbragg_incoherent_f2(s_mag: np.ndarray, b_wilson: float, wavelength: float) -> np.ndarray:
    """Exact crystal B-diffuse + Compton scattering factor squared from render_par.com (incoherent.stol)."""
    s2 = s_mag * s_mag
    fo = (
        3.0485 * np.exp(-13.2771 * s2)
        + 2.2868 * np.exp(-5.7011 * s2)
        + 1.5463 * np.exp(-0.3239 * s2)
        + 0.8670 * np.exp(-32.908897 * s2)
        + 0.2508
    )
    expo = np.clip(2.0 * max(b_wilson, 0.0) * s2, 0.0, 100.0)
    diffuse = (fo * fo) * (1.0 - np.exp(-expo))
    sin_th = np.clip(s_mag * wavelength, 0.0, 0.9999)
    cos_th = np.sqrt(1.0 - sin_th * sin_th)
    energy_ev = 12398.4245 / max(wavelength, 0.05)
    kappa = energy_ev / 511000.0
    compton = np.maximum(0.0, (8.0 - fo) / ((1.0 + kappa * (1.0 - cos_th)) ** 2))
    return diffuse + compton


def has_experimental_overrides(args: argparse.Namespace) -> bool:
    """Return True if any non-default physical/experimental data-collection flag was supplied."""
    return any(
        [
            getattr(args, "pdb_typical", False),
            getattr(args, "match_beam", False),
            getattr(args, "scale_by_cell", False),
            getattr(args, "auto_dose", False),
            getattr(args, "xtal_vol_um3", None) is not None,
            getattr(args, "xtal_size_um", None) is not None,
            getattr(args, "beam_size_um", None) is not None,
            getattr(args, "flux", None) is not None,
            getattr(args, "exposure_s", None) is not None,
            getattr(args, "wedge_deg", None) is not None,
            getattr(args, "osc_deg", None) is not None,
            getattr(args, "wavelength", None) is not None,
            getattr(args, "distance_mm", None) is not None,
            getattr(args, "water_um", None) is not None,
            getattr(args, "air_mm", None) is not None,
            getattr(args, "mosaicity_deg", None) is not None,
            getattr(args, "divergence_mrad", None) is not None,
            getattr(args, "dispersion_pct", None) is not None,
            getattr(args, "calib_err", None) is not None,
            getattr(args, "noncal_err", None) is not None,
            getattr(args, "beam_flicker", None) is not None,
            getattr(args, "shutter_jitter_ms", None) is not None,
            getattr(args, "sample_vib_um", None) is not None,
            getattr(args, "spindle_wobble_um", None) is not None,
            getattr(args, "spindle_vel_err", None) is not None,
            int(getattr(args, "n_det_pos", 1) or 1) != 1,
            getattr(args, "dose_mgy", None) is not None,
            float(getattr(args, "d_min", 2.0)) <= 0.0,
        ]
    )


def resolve_experimental_config(
    args: argparse.Namespace, cell_vol_A3: float, scale_i: float = 1.0
) -> Dict[str, Any]:
    """Resolve physical beam, crystal, background, detector, mechanical/flicker, and dose parameters."""
    active = has_experimental_overrides(args)
    if not active:
        return {"active": False}

    if getattr(args, "xtal_vol_um3", None) is not None and getattr(args, "xtal_size_um", None) is not None:
        raise ValueError("Specify either --xtal-vol-um3 or --xtal-size-um, not both.")

    # Optional --pdb-typical preset (matches 2013-2023 PDB <B_atom> vs d_min median curve;
    # any explicit CLI flag still overrides individual preset values)
    preset_xtal_um = 13.5 if getattr(args, "pdb_typical", False) else None
    preset_water_um = 60.0 if getattr(args, "pdb_typical", False) else None
    preset_mos_deg = 0.30 if getattr(args, "pdb_typical", False) else None
    preset_noncal = 0.02 if getattr(args, "pdb_typical", False) else None

    if getattr(args, "xtal_vol_um3", None) is not None:
        v_xtal = max(1e-3, float(args.xtal_vol_um3))
        l_xtal = v_xtal ** (1.0 / 3.0)
    elif getattr(args, "xtal_size_um", None) is not None:
        l_xtal = max(0.05, float(args.xtal_size_um))
        v_xtal = l_xtal ** 3
    elif preset_xtal_um is not None:
        l_xtal = preset_xtal_um
        v_xtal = l_xtal ** 3
    else:
        v_xtal = NB2_REF["xtal_vol_um3"]
        l_xtal = v_xtal ** (1.0 / 3.0)

    l_xtal_ref = NB2_REF["xtal_vol_um3"] ** (1.0 / 3.0)
    a_beam_ref = NB2_REF["beam_size_um"] ** 2
    phi_inc_ref = NB2_REF["flux_phs"] / a_beam_ref  # ph / um^2 / s

    if getattr(args, "beam_size_um", None) is not None:
        l_beam = max(0.1, float(args.beam_size_um))
        beam_auto_scaled = False
    elif getattr(args, "match_beam", False):
        l_beam = max(0.1, l_xtal)
        beam_auto_scaled = True
    elif l_xtal > NB2_REF["beam_size_um"]:
        # Open guard slits to match crystals larger than 30 um at constant beam brightness
        l_beam = l_xtal
        beam_auto_scaled = True
    else:
        l_beam = NB2_REF["beam_size_um"]
        beam_auto_scaled = False

    a_beam = l_beam * l_beam

    # Illuminated crystal volume (clipped to beam footprint if crystal exceeds explicit beam width)
    if l_xtal > l_beam:
        v_illum = a_beam * l_xtal
        beam_clipped = True
    else:
        v_illum = v_xtal
        beam_clipped = False

    if getattr(args, "flux", None) is not None:
        flux_phs = max(1.0, float(args.flux))
    elif beam_auto_scaled:
        flux_phs = phi_inc_ref * a_beam
    else:
        flux_phs = NB2_REF["flux_phs"]

    phi_inc = flux_phs / a_beam  # ph / um^2 / s

    exposure_s = max(1e-6, float(args.exposure_s)) if getattr(args, "exposure_s", None) is not None else NB2_REF["exposure_s"]
    wedge_deg = max(0.05, float(args.wedge_deg)) if getattr(args, "wedge_deg", None) is not None else NB2_REF["wedge_deg"]
    osc_deg = max(0.005, float(args.osc_deg)) if getattr(args, "osc_deg", None) is not None else NB2_REF["osc_deg"]
    wavelength = max(0.1, float(args.wavelength)) if getattr(args, "wavelength", None) is not None else NB2_REF["wavelength_A"]

    if getattr(args, "water_um", None) is not None:
        water_um = max(0.0, float(args.water_um))
    elif getattr(args, "match_beam", False) and preset_water_um is None:
        water_um = NB2_REF["water_um"] * (l_xtal / l_xtal_ref)
    elif preset_water_um is not None:
        water_um = preset_water_um
    else:
        water_um = NB2_REF["water_um"]

    air_mm = max(0.0, float(args.air_mm)) if getattr(args, "air_mm", None) is not None else NB2_REF["air_mm"]
    mos_deg = (
        max(0.005, float(args.mosaicity_deg))
        if getattr(args, "mosaicity_deg", None) is not None
        else (preset_mos_deg if preset_mos_deg is not None else NB2_REF["mosaicity_deg"])
    )
    div_mrad = max(0.0, float(args.divergence_mrad)) if getattr(args, "divergence_mrad", None) is not None else NB2_REF["divergence_mrad"]
    disp_pct = max(0.0, float(args.dispersion_pct)) if getattr(args, "dispersion_pct", None) is not None else NB2_REF["dispersion_pct"]
    calib_err = max(0.0, float(args.calib_err)) if getattr(args, "calib_err", None) is not None else NB2_REF["calib_err"]

    # Non-calibration multiplicative intensity errors per spot passage
    # (beam flicker, shutter timing jitter, sample cryo-loop vibration, spindle wobble):
    # Unlike pixel-anchored calib_err, these do NOT repeat when revisiting the same detector pixels.
    n_det_pos = max(1, int(getattr(args, "n_det_pos", 1) or 1))
    s_lumped = (
        max(0.0, float(args.noncal_err))
        if getattr(args, "noncal_err", None) is not None
        else (preset_noncal if preset_noncal is not None else 0.0)
    )
    s_flicker = max(0.0, float(args.beam_flicker)) if getattr(args, "beam_flicker", None) is not None else 0.0

    n_frames_tot = max(1.0, wedge_deg / osc_deg)
    t_frame_ms = 1000.0 * exposure_s / n_frames_tot
    dphi_rock_deg = math.sqrt(mos_deg ** 2 + (0.05729578 * div_mrad) ** 2)
    n_spot_frames = max(1.0, dphi_rock_deg / osc_deg)
    t_spot_ms = t_frame_ms * n_spot_frames

    jitter_ms = max(0.0, float(args.shutter_jitter_ms)) if getattr(args, "shutter_jitter_ms", None) is not None else 0.0
    s_jitter = jitter_ms / (max(t_frame_ms, 1e-6) * math.sqrt(n_spot_frames))

    w_eff_um = math.sqrt(l_beam * l_beam + l_xtal * l_xtal)
    vib_um = max(0.0, float(args.sample_vib_um)) if getattr(args, "sample_vib_um", None) is not None else 0.0
    s_vib_spatial = math.sqrt(
        (0.70 * vib_um / w_eff_um) ** 2
        + (3.5 * (vib_um ** 2) / (w_eff_um ** 2 + 3.5 * (vib_um ** 2))) ** 2
    )
    s_vib = s_vib_spatial / math.sqrt(1.0 + t_spot_ms / 50.0)

    wob_um = max(0.0, float(args.spindle_wobble_um)) if getattr(args, "spindle_wobble_um", None) is not None else 0.0
    s_wob_soc = math.sqrt(
        (0.70 * wob_um / w_eff_um) ** 2
        + (3.5 * (wob_um ** 2) / (w_eff_um ** 2 + 3.5 * (wob_um ** 2))) ** 2
    )
    vel_err = max(0.0, float(args.spindle_vel_err)) if getattr(args, "spindle_vel_err", None) is not None else 0.0
    s_wob_vel = vel_err / math.sqrt(n_spot_frames)
    s_wobble = math.sqrt(s_wob_soc ** 2 + s_wob_vel ** 2)

    sigma_noncal = math.sqrt(
        s_lumped ** 2 + s_flicker ** 2 + s_jitter ** 2 + s_vib ** 2 + s_wobble ** 2
    )

    rad_b_per_mgy = float(getattr(args, "rad_b_per_mgy", NB2_REF["rad_b_per_mgy"]))
    if getattr(args, "dose_mgy", None) is not None:
        dose_mgy = max(0.0, float(args.dose_mgy))
    elif getattr(args, "auto_dose", False):
        # ~2.0e10 ph/um^2 per MGy at 1.0 A for protein crystal (~lambda^2 photoabsorption scaling)
        fluence_um2 = phi_inc * exposure_s
        dose_mgy = (fluence_um2 / 2.0e10) * ((wavelength / 1.0) ** 2)
    else:
        dose_mgy = 0.0

    delta_b_rad = max(0.0, rad_b_per_mgy * dose_mgy)

    # Dimensionless physical scaling ratios relative to nanoBragg2 baseline
    gamma_wedge = wedge_deg / NB2_REF["wedge_deg"]
    gamma_pix = min(gamma_wedge, float(n_det_pos))
    gamma_rate = ((phi_inc / phi_inc_ref) * (exposure_s / NB2_REF["exposure_s"])) / gamma_wedge
    gamma_vol = v_illum / NB2_REF["xtal_vol_um3"]
    gamma_wave_bragg = (wavelength / NB2_REF["wavelength_A"]) ** 3
    gamma_wave_bg = (wavelength / NB2_REF["wavelength_A"]) ** 2
    # Exact Darwin unit-cell scaling relative to sim_reso_mtz's k0-normalized units:
    # N_photons / N_photons_v22 = (V_cell_0 / V_cell)^2 * (k0_in / k0_ref)
    gamma_cell = (
        ((NB2_REF["cell_vol_A3"] / max(cell_vol_A3, 100.0)) ** 2) * max(scale_i, 1e-12)
        if getattr(args, "scale_by_cell", False)
        else 1.0
    )

    # Per-observation Bragg photon gain ratio G / G_0
    gamma_g_obs = max(1e-18, gamma_rate * gamma_wave_bragg * gamma_vol * gamma_cell)

    # Pixel-anchored calibration error factor (leaves small discrete-pixel profile floor unless --ideal-transfer)
    f_prof_floor = 0.0 if getattr(args, "ideal_transfer", False) else 0.12
    f_cal_sq = f_prof_floor + (1.0 - f_prof_floor) * ((calib_err / NB2_REF["calib_err"]) ** 2)
    f_cal = math.sqrt(f_cal_sq)

    return {
        "active": True,
        "v_xtal_um3": v_xtal,
        "l_xtal_um": l_xtal,
        "v_illum_um3": v_illum,
        "beam_clipped": beam_clipped,
        "beam_size_um": l_beam,
        "a_beam_um2": a_beam,
        "flux_phs": flux_phs,
        "phi_inc_um2_s": phi_inc,
        "exposure_s": exposure_s,
        "t_frame_ms": t_frame_ms,
        "wedge_deg": wedge_deg,
        "osc_deg": osc_deg,
        "dphi_rock_deg": dphi_rock_deg,
        "wavelength_A": wavelength,
        "distance_mm_override": float(args.distance_mm) if getattr(args, "distance_mm", None) is not None else None,
        "water_um": water_um,
        "air_mm": air_mm,
        "mosaicity_deg": mos_deg,
        "divergence_mrad": div_mrad,
        "dispersion_pct": disp_pct,
        "calib_err": calib_err,
        "n_det_pos": n_det_pos,
        "sigma_noncal": sigma_noncal,
        "sigma_lumped": s_lumped,
        "sigma_flicker": s_flicker,
        "sigma_jitter": s_jitter,
        "sigma_vib": s_vib,
        "sigma_wobble": s_wobble,
        "dose_mgy": dose_mgy,
        "delta_b_rad": delta_b_rad,
        "scale_by_cell": bool(getattr(args, "scale_by_cell", False)),
        "cell_vol_A3": float(cell_vol_A3),
        "gamma_wedge": gamma_wedge,
        "gamma_pix": gamma_pix,
        "gamma_rate": gamma_rate,
        "gamma_vol": gamma_vol,
        "gamma_wave_bg": gamma_wave_bg,
        "gamma_cell": gamma_cell,
        "gamma_g_obs": gamma_g_obs,
        "f_cal": f_cal,
    }


def compute_exp_background_and_spot_factors(
    s_mag: np.ndarray,
    b_wilson: float,
    b_geom_ref: float,
    dist_mm: float,
    dist_ref_mm: float,
    exp_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bg_var_mult, spot_width_ratio) arrays across reflection s = 1/(2d) for experimental config.
    bg_var_mult multiplies per-observation sig_bg_0^2; equals 1.0 identically at NB2_REF."""
    f_wat2 = np.interp(s_mag, _STOL_GRID, _F_WATER_GRID) ** 2
    f_air2 = np.interp(s_mag, _STOL_GRID, _F_AIR_GRID) ** 2

    a_beam0 = NB2_REF["beam_size_um"] ** 2
    s_air0 = (a_beam0 * (1000.0 * NB2_REF["air_mm"])) * (1.2e-3 / 28.0) * f_air2
    s_wat0 = (a_beam0 * NB2_REF["water_um"]) * (1.0 / 18.0) * f_wat2
    s_xtl0 = NB2_REF["incoh_vol_um3"] * (1.2 / 16.0) * nonbragg_incoherent_f2(
        s_mag, b_geom_ref, NB2_REF["wavelength_A"]
    )
    tot0 = np.maximum(s_air0 + s_wat0 + s_xtl0, 1e-12)

    a_beam = exp_cfg["a_beam_um2"]
    s_air = (a_beam * (1000.0 * exp_cfg["air_mm"])) * (1.2e-3 / 28.0) * f_air2
    s_wat = (a_beam * exp_cfg["water_um"]) * (1.0 / 18.0) * f_wat2
    s_xtl = (NB2_REF["incoh_vol_um3"] * exp_cfg["gamma_vol"]) * (1.2 / 16.0) * nonbragg_incoherent_f2(
        s_mag, b_wilson, exp_cfg["wavelength_A"]
    )
    scatter_ratio = (s_air + s_wat + s_xtl) / tot0

    # 3D spot integration shoebox ratio (transverse solid angle * rotation-slice width)
    deg2rad = math.pi / 180.0
    mos0_rad = 0.5 * NB2_REF["mosaicity_deg"] * deg2rad
    div0_rad = 1e-3 * NB2_REF["divergence_mrad"]
    disp0 = 0.01 * NB2_REF["dispersion_pct"]
    px0_rad = PILATUS_6M_PIXEL_MM / max(dist_ref_mm, 15.0)

    mos_rad = 0.5 * exp_cfg["mosaicity_deg"] * deg2rad
    div_rad = 1e-3 * exp_cfg["divergence_mrad"]
    disp = 0.01 * exp_cfg["dispersion_pct"]
    px_rad = PILATUS_6M_PIXEL_MM / max(dist_mm, 15.0)

    sin_th0 = np.clip(s_mag * NB2_REF["wavelength_A"], 0.0, 0.98)
    tan_th0 = sin_th0 / np.sqrt(1.0 - sin_th0 * sin_th0)
    sin_th = np.clip(s_mag * exp_cfg["wavelength_A"], 0.0, 0.98)
    tan_th = sin_th / np.sqrt(1.0 - sin_th * sin_th)

    om0 = np.sqrt(
        (mos0_rad ** 2 + div0_rad ** 2 + px0_rad ** 2)
        * (mos0_rad ** 2 + div0_rad ** 2 + (2.0 * disp0 * tan_th0) ** 2 + px0_rad ** 2)
    )
    om = np.sqrt(
        (mos_rad ** 2 + div_rad ** 2 + px_rad ** 2)
        * (mos_rad ** 2 + div_rad ** 2 + (2.0 * disp * tan_th) ** 2 + px_rad ** 2)
    )

    dphi0 = math.sqrt(NB2_REF["mosaicity_deg"] ** 2 + (div0_rad / deg2rad) ** 2 + (0.5 * NB2_REF["osc_deg"]) ** 2)
    dphi = math.sqrt(exp_cfg["mosaicity_deg"] ** 2 + (div_rad / deg2rad) ** 2 + (0.5 * exp_cfg["osc_deg"]) ** 2)
    shoebox_ratio = (om / np.maximum(om0, 1e-12)) * (dphi / max(dphi0, 1e-6))

    # Ratio of detected background photons per observation, divided by G^2 to bring to MTZ scale
    n_bg_obs_ratio = (
        exp_cfg["gamma_rate"]
        * exp_cfg["gamma_wave_bg"]
        * scatter_ratio
        * shoebox_ratio
    )
    bg_var_mult = n_bg_obs_ratio / (exp_cfg["gamma_g_obs"] ** 2)

    spot_w0 = np.sqrt(mos0_rad ** 2 + div0_rad ** 2 + (2.0 * disp0 * tan_th0) ** 2)
    spot_w = np.sqrt(mos_rad ** 2 + div_rad ** 2 + (2.0 * disp * tan_th) ** 2)
    spot_width_ratio = spot_w / np.maximum(spot_w0, 1e-8)
    return bg_var_mult, spot_width_ratio


def eval_shell_criterion_metric(
    d_eval: float,
    bw_cand: float,
    cal_geom: Dict[str, Any],
    b_geom_ref: float,
    d_edge: float,
    dist_mm: float,
    dist_ref_mm: float,
    k0_in: float,
    scale_i: float,
    b0_nom_ref: float,
    b0_form_ref: float,
    m_max: int,
    mean_m_pg: float,
    criterion: str,
    exp_cfg: Optional[Dict[str, Any]],
) -> float:
    """Analytical shell criterion value (CC1/2, <I/sigI>, or CC_true) at resolution d_eval (A)
    for crystal B_Wilson = bw_cand (nanoBragg2 gauge)."""
    s_val = 1.0 / (2.0 * max(d_eval, 0.25))
    s2_val = s_val * s_val
    u_val = d_edge / max(d_eval, 0.25)
    f_ring_v = 1.0 if u_val <= 1.0 else max(1.0 - (4.0 / math.pi) * math.acos(max(-1.0, min(1.0, 1.0 / u_val))), 0.22)

    t_v = float(np.interp(u_val, cal_geom["u_t_knots"], cal_geom["t_xds"]))
    if exp_cfg is not None and exp_cfg.get("active", False):
        bg_mult_arr, sp_rat_arr = compute_exp_background_and_spot_factors(
            np.array([s_val], dtype=np.float64), bw_cand, b_geom_ref, dist_mm, dist_ref_mm, exp_cfg
        )
        bg_var_mult = float(bg_mult_arr[0])
        sp_rat = float(sp_rat_arr[0])
        if t_v < 1.0 and abs(sp_rat - 1.0) > 1e-4:
            t_v = max(0.02, t_v ** (sp_rat ** 0.5))
        g_mult = 1.0 / exp_cfg["gamma_g_obs"]
        gw = exp_cfg["gamma_wedge"]
        gp = exp_cfg["gamma_pix"]
        eff_obs = max(1.0, gw * mean_m_pg * f_ring_v) if gw < 1.0 else (gw * mean_m_pg * f_ring_v)
        eff_pix = max(1.0, gp * mean_m_pg * f_ring_v) if gp < 1.0 else (gp * mean_m_pg * f_ring_v)
        mult_fac = math.sqrt(float(m_max) / max(eff_obs, 0.22))
        mult_fac_pix = math.sqrt(float(m_max) / max(eff_pix, 0.22))
        f_cal = exp_cfg["f_cal"]
        sig_nc = exp_cfg["sigma_noncal"]
        rad_sys = 0.0025 * exp_cfg["dose_mgy"]
    else:
        bg_var_mult = 1.0
        g_mult = 1.0
        mult_fac = math.sqrt(float(m_max) / max(mean_m_pg * f_ring_v, 0.22))
        mult_fac_pix = mult_fac
        f_cal = 1.0
        gp = 1.0
        sig_nc = 0.0
        rad_sys = 0.0

    b_form_cand = bw_cand - (b0_nom_ref - b0_form_ref)
    fp2_r = float(f_prot_sq(np.array(s2_val))) / float(f_prot_sq(np.array(0.0)))
    i_sig = max(1e-30, k0_in * fp2_r * math.exp(-2.0 * b_form_cand * s2_val) * max(t_v, 1e-4))

    wat_b = math.exp(-(((s_val - 0.160) / 0.045) ** 2))
    cor_b = max(0.0, u_val - 0.95) ** 2
    sig_bg0 = (float(cal_geom["sig0"]) * scale_i) * math.exp(
        float(cal_geom["b1_u2"]) * (u_val * u_val)
        + float(cal_geom["w_wat"]) * wat_b
        + float(cal_geom.get("b2_cor", 0.0)) * cor_b
    )
    sig_bg2 = (sig_bg0 * sig_bg0) * bg_var_mult
    g_sh = (float(cal_geom["g_shot"]) * scale_i) * g_mult
    c_sg_cal = float(cal_geom["c_sig"]) * f_cal
    c_sg_nc = sig_nc / math.sqrt(max(float(m_max), 1.0))
    c_sg = math.sqrt(c_sg_cal * c_sg_cal + c_sg_nc * c_sg_nc)
    # Include median log-normal dispersion factor exp(0) = 1
    sig_i = mult_fac * math.sqrt(max(1e-30, sig_bg2 + g_sh * i_sig + (c_sg * i_sig) ** 2))

    wk0, wk1, wk2 = hat_weights_3(np.array([u_val]), cal_geom["u_sc_knots"])
    a_st = math.sqrt(
        float(wk0[0]) * (float(cal_geom["a_stat"]) ** 2)
        + float(wk1[0]) * (float(cal_geom.get("a_stat_u1", cal_geom["a_stat"])) ** 2)
        + float(wk2[0]) * (float(cal_geom.get("a_stat_u2", cal_geom["a_stat"])) ** 2)
    )
    c_sy0 = math.sqrt(
        float(wk0[0]) * (float(cal_geom["c_sysI"]) ** 2)
        + float(wk1[0]) * (float(cal_geom.get("c_sysI_u1", cal_geom["c_sysI"])) ** 2)
        + float(wk2[0]) * (float(cal_geom.get("c_sysI_u2", cal_geom["c_sysI"])) ** 2)
    )
    c_cal_rem_sq = max(0.0, (a_st * c_sg_cal) ** 2 * (mult_fac_pix ** 2 - mult_fac ** 2))
    c_nc_rem_sq = max(0.0, 1.0 - a_st * a_st) * ((mult_fac * c_sg_nc) ** 2)
    c_sy = math.sqrt(
        (c_sy0 * f_cal) ** 2 / max(gp, 0.02) + c_cal_rem_sq + c_nc_rem_sq + rad_sys * rad_sys
    )
    var_err = (a_st * sig_i) ** 2 + (c_sy * i_sig) ** 2

    if criterion in ("isig1.5", "isig2.0"):
        return i_sig / max(sig_i, 1e-30)
    if criterion in ("ccright", "maprms"):
        return i_sig / math.sqrt(i_sig * i_sig + var_err)
    # Default 'cc12': analytical Karplus-Diederichs acentric shell CC1/2
    return (i_sig * i_sig) / (i_sig * i_sig + 2.0 * var_err)


def solve_b_wilson_for_dmin(
    cal: dict,
    d_min: float,
    criterion: str,
    cal_geom: Dict[str, Any],
    b_geom_ref: float,
    d_edge: float,
    dist_mm: float,
    dist_ref_mm: float,
    k0_in: float,
    scale_i: float,
    b0_nom_ref: float,
    b0_form_ref: float,
    m_max: int,
    mean_m_pg: float,
    exp_cfg: Dict[str, Any],
) -> float:
    """Solve 1D bisection for crystal B_Wilson that achieves target resolution d_min under exp_cfg,
    anchored so exp_cfg == NB2_REF reproduces the calibrated B_Wilson(d_min) identically."""
    cal_base_d = interpolate_calibration(cal, d_min, criterion=criterion)
    bw_base_d = float(cal_base_d["B_Wilson"])
    d_edg_base = float(cal_base_d["d_edge"])
    dist_base = dist_from_dedge(d_edg_base, NB2_REF["wavelength_A"])

    target_m = eval_shell_criterion_metric(
        d_min,
        bw_base_d,
        cal_base_d,
        bw_base_d,
        d_edg_base,
        dist_base,
        dist_base,
        k0_in,
        scale_i,
        b0_nom_ref,
        b0_form_ref,
        8,
        float(np.clip(mean_m_pg * (8.0 / max(m_max, 1)), 1.0, 8.0)),
        criterion,
        None,
    )

    lo_b, hi_b = -40.0, 2500.0
    m_lo = eval_shell_criterion_metric(
        d_min, lo_b, cal_geom, b_geom_ref, d_edge, dist_mm, dist_ref_mm,
        k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg, criterion, exp_cfg,
    )
    if m_lo <= target_m:
        return lo_b
    for _ in range(52):
        mid = 0.5 * (lo_b + hi_b)
        m_mid = eval_shell_criterion_metric(
            d_min, mid, cal_geom, b_geom_ref, d_edge, dist_mm, dist_ref_mm,
            k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg, criterion, exp_cfg,
        )
        if m_mid >= target_m:
            lo_b = mid
        else:
            hi_b = mid
    return 0.5 * (lo_b + hi_b)


def solve_emergent_dmin(
    cal: dict,
    bw_eff: float,
    criterion: str,
    exp_cfg: Dict[str, Any],
    k0_in: float,
    scale_i: float,
    b0_nom_ref: float,
    b0_form_ref: float,
    m_max: int,
    mean_m_pg: float,
    d_min_floor: float = 0.55,
) -> float:
    """Solve for the emergent resolution limit d_min (A) given fixed crystal B_Wilson = bw_eff and exp_cfg."""
    wl = exp_cfg["wavelength_A"]
    dist_ovr = exp_cfg.get("distance_mm_override")
    m_pg_8 = float(np.clip(mean_m_pg * (8.0 / max(m_max, 1)), 1.0, 8.0))

    def gap_at_d(d_cand: float) -> float:
        d_thr = float(np.clip(d_cand, 0.86, 4.20))
        cal_thr = interpolate_calibration(cal, d_thr, criterion=criterion)
        bw_thr = float(cal_thr["B_Wilson"])
        d_edg_thr = float(cal_thr["d_edge"])
        dist_thr = dist_from_dedge(d_edg_thr, NB2_REF["wavelength_A"])
        m_tgt = eval_shell_criterion_metric(
            d_thr, bw_thr, cal_thr, bw_thr, d_edg_thr, dist_thr, dist_thr,
            k0_in, scale_i, b0_nom_ref, b0_form_ref,
            8, m_pg_8, criterion, None,
        )
        if dist_ovr is not None:
            d_mm = dist_ovr
            d_edg_c, _ = dedge_dcorner_from_dist(d_mm, wl)
            d_edg_eq = d_edg_c * (NB2_REF["wavelength_A"] / wl)
            cal_g = interpolate_calibration_by_dedge(cal, d_edg_eq, criterion=criterion)
            dist_g_ref = d_mm
        else:
            d_geom_lkp = min(d_cand, 4.50)
            cal_ref_d = interpolate_calibration(cal, d_geom_lkp, criterion=criterion)
            sc_hi = max(1.0, d_cand / 4.50)
            d_edg_ref = float(cal_ref_d["d_edge"]) * sc_hi
            dist_ref_d = dist_from_dedge(d_edg_ref, NB2_REF["wavelength_A"])
            d_mm = dist_ref_d
            d_edg_c, _ = dedge_dcorner_from_dist(d_mm, wl)
            cal_g = dict(cal_ref_d)
            if sc_hi > 1.0:
                cal_g["sig0"] = float(cal_ref_d["sig0"]) / (sc_hi ** 1.5)
                cal_g["g_shot"] = float(cal_ref_d["g_shot"]) / sc_hi
            dist_g_ref = dist_ref_d

        m_cur = eval_shell_criterion_metric(
            d_cand, bw_eff, cal_g, float(cal_g["B_Wilson"]), d_edg_c, d_mm, dist_g_ref,
            k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg, criterion, exp_cfg,
        )
        return m_cur - m_tgt

    d_lo = max(d_min_floor, 0.52 * wl)
    if dist_ovr is not None:
        _, d_cor_ovr = dedge_dcorner_from_dist(dist_ovr, wl)
        d_lo = max(d_lo, d_cor_ovr)
    d_hi = 12.0
    if d_lo >= d_hi:
        return float(d_lo)

    grid = np.geomspace(d_lo, d_hi, 40)
    gaps = [gap_at_d(float(dg)) for dg in grid]
    if gaps[0] >= 0.0:
        return float(d_lo)
    brk = None
    for k in range(len(gaps) - 1):
        if gaps[k] < 0.0 and gaps[k + 1] >= 0.0:
            brk = (float(grid[k]), float(grid[k + 1]))
            break
    if brk is None:
        return float(d_hi)
    a_d, b_d = brk
    for _ in range(42):
        mid_d = 0.5 * (a_d + b_d)
        if gap_at_d(mid_d) >= 0.0:
            b_d = mid_d
        else:
            a_d = mid_d
    return 0.5 * (a_d + b_d)


def compute_symmetry_factors(
    spacegroup: gemmi.SpaceGroup, hkls: np.ndarray
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Vectorized calculation of point-group/Laue multiplicity M_pg(hkl), M_max,
    is_centric(hkl), and epsilon(hkl) for any crystallographic space group."""
    ops = spacegroup.operations()

    sym_rots = []
    for op in ops.sym_ops:
        mat = np.array(
            [
                op.apply_to_hkl([1, 0, 0]),
                op.apply_to_hkl([0, 1, 0]),
                op.apply_to_hkl([0, 0, 1]),
            ],
            dtype=np.int32,
        )
        sym_rots.append(mat)
    sym_rots_arr = np.stack(sym_rots, axis=0)

    laue_set = {}
    pg_set = {}
    for mat in sym_rots_arr:
        key_p = tuple(mat.ravel().tolist())
        key_m = tuple((-mat).ravel().tolist())
        pg_set[key_p] = mat
        laue_set[key_p] = mat
        laue_set[key_m] = -mat

    pg_mats = np.stack(list(pg_set.values()), axis=0)
    laue_mats = np.stack(list(laue_set.values()), axis=0)
    m_max = int(laue_mats.shape[0])

    rot_laue = np.einsum("ij,kjl->kil", hkls, laue_mats)
    stab_laue = np.all(rot_laue == hkls[None, :, :], axis=2).sum(axis=0)
    m_pg = (m_max // np.maximum(stab_laue, 1)).astype(np.int32)

    rot_pg = np.einsum("ij,kjl->kil", hkls, pg_mats)
    is_centric = np.any(np.all(rot_pg == -hkls[None, :, :], axis=2), axis=0)

    rot_sym = np.einsum("ij,kjl->kil", hkls, sym_rots_arr)
    stab_sym = np.all(rot_sym == hkls[None, :, :], axis=2).sum(axis=0)
    epsilon = (len(ops.cen_ops) * stab_sym).astype(np.float64)

    return m_pg, m_max, is_centric, epsilon


def estimate_f0_wilson_scale(
    s2: np.ndarray,
    i0: np.ndarray,
    epsilon: np.ndarray,
    b0_form_1aho: float,
    b0_nominal_1aho: float,
) -> Tuple[float, float, float]:
    """Estimate input F0's extrapolated s=0 intensity scale K0_in, true form-factor-normalized
    Wilson B0_form, and equivalent nanoBragg2-gauge B0_nominal."""
    valid = np.isfinite(i0) & (i0 > 0.0) & np.isfinite(s2) & (s2 > 0.0)
    s2_v = s2[valid]
    ie_v = i0[valid] / epsilon[valid]

    s2_lo, s2_hi = 0.03, 0.35
    mask_w = (s2_v >= s2_lo) & (s2_v <= s2_hi)
    if mask_w.sum() < 200:
        s2_lo = float(np.percentile(s2_v, 10))
        s2_hi = float(np.percentile(s2_v, 98))
        mask_w = (s2_v >= s2_lo) & (s2_v <= s2_hi)

    n_bins = min(32, max(6, int(mask_w.sum() // 80)))
    edges = np.linspace(s2_lo, s2_hi, n_bins + 1)
    bc, by = [], []
    for j in range(n_bins):
        m = (s2_v >= edges[j]) & (s2_v < edges[j + 1] if j < n_bins - 1 else s2_v <= edges[j + 1])
        if m.sum() >= 5:
            sc_m = 0.5 * (edges[j] + edges[j + 1])
            mean_ie = float(np.mean(ie_v[m]))
            if mean_ie > 0.0:
                bc.append(sc_m)
                by.append(mean_ie / float(f_prot_sq(np.array(sc_m))))

    if len(bc) >= 2:
        slope, icp = np.polyfit(np.asarray(bc), np.log(np.asarray(by)), 1)
        b0_form = -0.5 * float(slope)
        k0_in = math.exp(float(icp)) * float(f_prot_sq(np.array(0.0)))
    else:
        b0_form = b0_form_1aho
        k0_in = float(np.mean(ie_v / f_prot_sq(s2_v))) * float(f_prot_sq(np.array(0.0)))

    b0_nominal = b0_nominal_1aho + (b0_form - b0_form_1aho)
    return k0_in, b0_form, b0_nominal


def xdsconv_sliding_mean(
    s2: np.ndarray,
    ip: np.ndarray,
    im: np.ndarray,
    is_centric: np.ndarray,
    half_win: int = 100,
    clamped: bool = True,
) -> np.ndarray:
    """Wolfgang Kabsch's XDSCONV clamped sliding-window <Intensity> over unmerged records
    (1 record per centric, 2 records I+/I- per acentric) sorted by s^2."""
    cen_m = is_centric & np.isfinite(ip)
    acp_m = (~is_centric) & np.isfinite(ip)
    acm_m = (~is_centric) & np.isfinite(im)
    recs_s2 = np.concatenate([s2[cen_m], s2[acp_m], s2[acm_m]])
    recs_i = np.concatenate([ip[cen_m], ip[acp_m], im[acm_m]])
    if len(recs_i) == 0:
        return np.full(len(s2), np.nan, dtype=np.float64)

    order = np.argsort(recs_s2, kind="mergesort")
    rs2 = recs_s2[order]
    ri = recs_i[order]
    n = len(ri)
    csum = np.concatenate([[0.0], np.cumsum(ri)])
    idx = np.arange(n)
    win = 2 * half_win + 1
    if clamped and n > win:
        lo = np.clip(idx - half_win, 0, n - win)
        hi = lo + win
    else:
        lo = np.maximum(0, idx - half_win)
        hi = np.minimum(n, idx + half_win + 1)
    mean_sorted = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    pos = np.clip(np.searchsorted(rs2, s2), 0, n - 1)
    return mean_sorted[pos]


def estimate_shell_sigma_prior(
    s2: np.ndarray,
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    epsilon: np.ndarray,
    k0_scaled: float,
    b_form_eff: float,
    mode: str = "ccp4",
) -> np.ndarray:
    """Alternative shell prior Sigma(hkl) = eps * <I/eps>(s^2) for non-XDSCONV truncate modes
    ('ccp4' 60 equal 1/d^2 bins, or 'smooth' strictly positive prior)."""
    valid = np.isfinite(i_obs) & np.isfinite(sig_i) & (sig_i > 0.0)
    wilson_curve = k0_scaled * (f_prot_sq(s2) / float(f_prot_sq(np.array(0.0)))) * np.exp(-2.0 * b_form_eff * s2)
    wilson_curve = np.maximum(wilson_curve, 1e-6)

    if valid.sum() < 60:
        return epsilon * wilson_curve

    if mode == "ccp4":
        s4 = 4.0 * s2
        s4_v = s4[valid]
        n_bins = 60
        edges = np.linspace(float(np.min(s4_v)), float(np.max(s4_v)) + 1e-9, n_bins + 1)
        bin_idx = np.clip(np.digitize(s4, edges) - 1, 0, n_bins - 1)
        bin_idx_v = bin_idx[valid]
        ie_v = i_obs[valid] / epsilon[valid]
        bin_means = np.full(n_bins, np.nan, dtype=np.float64)
        for b in range(n_bins):
            m = bin_idx_v == b
            if np.any(m):
                mn = float(np.mean(ie_v[m]))
                if mn > 0.0:
                    bin_means[b] = mn
        return epsilon * bin_means[bin_idx]

    s2_v = s2[valid]
    ie_v = i_obs[valid] / epsilon[valid]
    se_v = sig_i[valid] / epsilon[valid]
    n_shells = min(40, max(8, int(valid.sum() // 150)))
    quantiles = np.quantile(s2_v, np.linspace(0.0, 1.0, n_shells + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9

    sh_s2 = np.empty(n_shells, dtype=np.float64)
    sh_ie = np.empty(n_shells, dtype=np.float64)
    for j in range(n_shells):
        m = (s2_v >= quantiles[j]) & (s2_v <= quantiles[j + 1])
        sh_s2[j] = float(np.mean(s2_v[m]))
        w_prior = k0_scaled * (float(f_prot_sq(np.array(sh_s2[j]))) / float(f_prot_sq(np.array(0.0)))) * math.exp(
            -2.0 * b_form_eff * sh_s2[j]
        )
        raw_mean = float(np.mean(ie_v[m]))
        noise_floor = 0.25 * float(np.mean(se_v[m])) / math.sqrt(max(int(m.sum()), 1))
        sh_ie[j] = max(raw_mean, max(w_prior, noise_floor, 1e-6))

    if n_shells >= 3:
        log_sh = np.log(sh_ie)
        sm = log_sh.copy()
        sm[1:-1] = 0.25 * log_sh[:-2] + 0.50 * log_sh[1:-1] + 0.25 * log_sh[2:]
        sh_ie = np.exp(sm)

    interp_ie = np.exp(np.interp(s2, sh_s2, np.log(sh_ie)))
    return epsilon * np.maximum(interp_ie, 1e-6)


# Precompute 64-point Gauss-Legendre nodes/weights on [0, 1] once
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_U = 0.5 * (_GL_X + 1.0)
_GL_WU = 0.5 * _GL_W


def french_wilson_vectorized(
    i_obs: np.ndarray,
    sig_i: np.ndarray,
    sigma_prior: np.ndarray,
    is_centric: np.ndarray,
    min_isigi: float = -3.7,
    min_h_score: float = -1e9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact overflow-safe vectorized French-Wilson Bayesian posterior integration:
        (I_obs, SIGI) -> (F_obs, SIGF)
    Integrates in y = F / sqrt(SIGI) >= 0 using 64-point Gauss-Legendre quadrature over the
    exact conditional 8.5-sigma support, rejecting I/SIGI < min_isigi or h < min_h_score."""
    n = len(i_obs)
    f_out = np.full(n, np.nan, dtype=np.float64)
    sigf_out = np.full(n, np.nan, dtype=np.float64)

    valid = np.isfinite(i_obs) & np.isfinite(sig_i) & (sig_i > 0.0) & np.isfinite(sigma_prior) & (sigma_prior > 0.0)
    if not np.any(valid):
        return f_out, sigf_out

    iv = i_obs[valid]
    sv = sig_i[valid]
    sp = sigma_prior[valid]
    ic = is_centric[valid]

    isigi = iv / sv
    # French & Wilson (1978) Eq. 9 / XDSCONV Score = I/Sigma - Sigma / (Type * <Intensity>)
    h = np.where(ic, isigi - 0.5 * (sv / sp), isigi - (sv / sp))
    keep = (isigi >= min_isigi) & (h >= min_h_score)
    if not np.any(keep):
        return f_out, sigf_out

    valid_idx = np.flatnonzero(valid)[keep]
    sv_k = sv[keep]
    ic_k = ic[keep]
    h_k = h[keep]

    chunk_size = 25000
    f_res = np.empty(len(h_k), dtype=np.float64)
    sf_res = np.empty(len(h_k), dtype=np.float64)
    r_cut = 8.5

    for start in range(0, len(h_k), chunk_size):
        end = min(start + chunk_size, len(h_k))
        hc = h_k[start:end]
        icc = ic_k[start:end]
        svc = sv_k[start:end]

        y2_lo = np.where(hc > r_cut, hc - r_cut, 0.0)
        y2_hi = np.where(
            hc >= 0.0,
            hc + r_cut,
            (r_cut * r_cut) / (np.sqrt(hc * hc + r_cut * r_cut) - hc),
        )
        y_lo = np.sqrt(y2_lo)
        y_hi = np.sqrt(np.maximum(y2_hi, 1e-12))
        dy = y_hi - y_lo

        y = y_lo[:, None] + dy[:, None] * _GL_U[None, :]
        y2 = y * y
        neg_h = hc < 0.0
        log_w = np.empty_like(y)
        if np.any(~neg_h):
            log_w[~neg_h] = -0.5 * ((y2[~neg_h] - hc[~neg_h, None]) ** 2)
        if np.any(neg_h):
            log_w[neg_h] = -0.5 * (y2[neg_h] * y2[neg_h]) + hc[neg_h, None] * y2[neg_h]

        ac_mask = ~icc
        if np.any(ac_mask):
            log_w[ac_mask] += np.log(np.maximum(y[ac_mask], 1e-300))

        log_w -= np.max(log_w, axis=1, keepdims=True)
        w = np.exp(log_w) * _GL_WU[None, :]

        z0 = np.sum(w, axis=1)
        y_mean = np.sum(w * y, axis=1) / z0
        y2_mean = np.sum(w * y2, axis=1) / z0
        y_var = np.maximum(y2_mean - y_mean * y_mean, 1e-12)

        sqrt_s = np.sqrt(svc)
        f_res[start:end] = sqrt_s * y_mean
        sf_res[start:end] = sqrt_s * np.sqrt(y_var)

    f_out[valid_idx] = f_res
    sigf_out[valid_idx] = sf_res
    return f_out, sigf_out


def xdsconv_truncate_and_merge(
    s2: np.ndarray,
    ip_obs: np.ndarray,
    sig_ip: np.ndarray,
    im_obs: np.ndarray,
    sig_im: np.ndarray,
    is_centric: np.ndarray,
    half_win: int = 100,
    clamped: bool = True,
    min_isigi: float = -3.0,
    min_score: float = -35.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replicate Wolfgang Kabsch's XDSCONV (OUTPUT_FILE=temp.hkl CCP4_I, WILSON_STATISTICS=TRUE,
    FRIEDEL'S_LAW=FALSE) + f2mtz + cad amplitude conversion and Friedel-mate merging."""
    mn_i = xdsconv_sliding_mean(s2, ip_obs, im_obs, is_centric, half_win=half_win, clamped=clamped)

    fp_obs, sig_fp = french_wilson_vectorized(
        ip_obs, sig_ip, mn_i, is_centric, min_isigi=min_isigi, min_h_score=min_score
    )
    im_ac_in = np.where(is_centric, np.nan, im_obs)
    fm_ac, sig_fm_ac = french_wilson_vectorized(
        im_ac_in, sig_im, mn_i, is_centric, min_isigi=min_isigi, min_h_score=min_score
    )

    fm_obs = np.where(is_centric, fp_obs, fm_ac)
    sig_fm = np.where(is_centric, sig_fp, sig_fm_ac)

    n = len(s2)
    f_obs = np.full(n, np.nan, dtype=np.float64)
    sig_f = np.full(n, np.nan, dtype=np.float64)
    dano = np.full(n, np.nan, dtype=np.float64)
    sig_dano = np.full(n, np.nan, dtype=np.float64)
    isym = np.full(n, np.nan, dtype=np.float64)

    ok_p = np.isfinite(fp_obs) & np.isfinite(sig_fp) & (sig_fp > 0.0)
    ok_m = np.isfinite(fm_obs) & np.isfinite(sig_fm) & (sig_fm > 0.0)

    # Centrics
    cen_ok = is_centric & ok_p
    f_obs[cen_ok] = fp_obs[cen_ok]
    sig_f[cen_ok] = sig_fp[cen_ok]
    dano[cen_ok] = 0.0
    sig_dano[cen_ok] = 0.0
    isym[cen_ok] = 0.0

    # Acentrics with both F+ and F- present: exact XDSCONV CCP4_I arithmetic mean & SE
    both_ac = (~is_centric) & ok_p & ok_m
    if np.any(both_ac):
        f_obs[both_ac] = 0.5 * (fp_obs[both_ac] + fm_obs[both_ac])
        sig_f[both_ac] = 0.5 * np.sqrt(sig_fp[both_ac] ** 2 + sig_fm[both_ac] ** 2)
        dano[both_ac] = fp_obs[both_ac] - fm_obs[both_ac]
        sig_dano[both_ac] = np.sqrt(sig_fp[both_ac] ** 2 + sig_fm[both_ac] ** 2)
        isym[both_ac] = 0.0

    # Acentrics with only F+ or only F- surviving Score cutoff
    only_p = (~is_centric) & ok_p & (~ok_m)
    if np.any(only_p):
        f_obs[only_p] = fp_obs[only_p]
        sig_f[only_p] = sig_fp[only_p]
        isym[only_p] = 1.0

    only_m = (~is_centric) & (~ok_p) & ok_m
    if np.any(only_m):
        f_obs[only_m] = fm_obs[only_m]
        sig_f[only_m] = sig_fm[only_m]
        isym[only_m] = 2.0

    return f_obs, sig_f, fp_obs, sig_fp, fm_obs, sig_fm, dano, sig_dano, isym


def detect_input_columns(
    mtz: gemmi.Mtz,
    f_col: Optional[str],
    f_minus_col: Optional[str],
    phi_col: Optional[str],
) -> Tuple[str, Optional[str], bool, Optional[str]]:
    """Return (col_plus_or_mean, col_minus_or_None, is_intensity, phi_col_or_None)."""
    labels = {c.label: c for c in mtz.columns}

    det_phi: Optional[str] = None
    if phi_col:
        if phi_col not in labels:
            raise ValueError(f"Requested --phi-col '{phi_col}' not found in MTZ columns: {list(labels)}")
        det_phi = phi_col
    else:
        for cand in ("PHIright", "PHIC", "PHIC_ALL", "PHI", "PHWT", "PHFOFCWT"):
            if cand in labels:
                det_phi = cand
                break
        if det_phi is None:
            for c in mtz.columns:
                if c.type == "P":
                    det_phi = c.label
                    break

    if f_col:
        if "," in f_col and not f_minus_col:
            p_lbl, m_lbl = [x.strip() for x in f_col.split(",", 1)]
            f_col, f_minus_col = p_lbl, m_lbl
        if f_col not in labels:
            raise ValueError(f"Requested --f-col '{f_col}' not found in MTZ columns: {list(labels)}")
        if f_minus_col and f_minus_col not in labels:
            raise ValueError(f"Requested --f-minus-col '{f_minus_col}' not found in MTZ columns: {list(labels)}")
        is_int = labels[f_col].type in ("J", "K")
        return f_col, f_minus_col, is_int, det_phi

    for p_cand, m_cand in (("Fplus", "Fminus"), ("F(+)", "F(-)"), ("FP(+)", "FP(-)")):
        if p_cand in labels and m_cand in labels:
            return p_cand, m_cand, False, det_phi

    for cand in ("F0", "Fright", "F", "FP", "FC", "FC_ALL", "FMODEL", "FWT", "Fcalc"):
        if cand in labels and labels[cand].type == "F":
            return cand, None, False, det_phi
    for c in mtz.columns:
        if c.type == "F":
            return c.label, None, False, det_phi

    for p_cand, m_cand in (("Iplus", "Iminus"), ("I(+)", "I(-)")):
        if p_cand in labels and m_cand in labels:
            return p_cand, m_cand, True, det_phi
    for cand in ("I0", "Iright", "IMEAN", "I", "IOBS"):
        if cand in labels and labels[cand].type == "J":
            return cand, None, True, det_phi
    for c in mtz.columns:
        if c.type == "J":
            return c.label, None, True, det_phi

    raise ValueError(f"Could not auto-detect structure-factor or intensity column in: {list(labels)}")


def compute_cc12_from_sigi(i_obs: np.ndarray, sig_i: np.ndarray) -> float:
    """Analytical Aimless-like CC1/2 estimate from merged (I, SIGI)."""
    m = np.isfinite(i_obs) & np.isfinite(sig_i) & (sig_i > 0.0)
    if m.sum() < 10:
        return float("nan")
    iv = i_obs[m]
    sv2 = float(np.mean(sig_i[m] ** 2))
    var_tot = float(np.var(iv))
    var_sig = max(0.0, var_tot - sv2)
    denom = var_sig + 2.0 * sv2
    return var_sig / denom if denom > 0.0 else 0.0


def simulate_dataset(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()

    cal_path = Path(args.cal_file) if args.cal_file else Path(__file__).with_name("nanobragg2_calibration.json")
    if not cal_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))

    mtz_in = gemmi.read_mtz_file(str(args.input_mtz))
    col_p, col_m, is_int_in, col_phi = detect_input_columns(
        mtz_in, args.f_col, args.f_minus_col, args.phi_col
    )

    arr_in = np.array(mtz_in, copy=False)
    h_idx = [mtz_in.column_with_label(lbl).idx for lbl in ("H", "K", "L")]
    hkls_all = np.rint(arr_in[:, h_idx]).astype(np.int32)
    d_all = np.asarray(mtz_in.make_d_array(), dtype=np.float64)

    vp = arr_in[:, mtz_in.column_with_label(col_p).idx].astype(np.float64)
    if col_m is not None:
        vm = arr_in[:, mtz_in.column_with_label(col_m).idx].astype(np.float64)
        valid_in = np.isfinite(vp) & np.isfinite(vm) & (d_all > 0.0)
        fp0 = np.sqrt(np.maximum(vp[valid_in], 0.0)) if is_int_in else np.maximum(vp[valid_in], 0.0)
        fm0 = np.sqrt(np.maximum(vm[valid_in], 0.0)) if is_int_in else np.maximum(vm[valid_in], 0.0)
        f0_mean = np.sqrt(0.5 * (fp0 * fp0 + fm0 * fm0))
        has_anom_in = True
    else:
        valid_in = np.isfinite(vp) & (d_all > 0.0)
        f0_mean = np.sqrt(np.maximum(vp[valid_in], 0.0)) if is_int_in else np.maximum(vp[valid_in], 0.0)
        fp0 = f0_mean
        fm0 = f0_mean
        has_anom_in = False

    hkls = hkls_all[valid_in]
    d = d_all[valid_in]
    s2 = 1.0 / (4.0 * d * d)
    i0_mean = f0_mean * f0_mean

    phi_true = None
    if col_phi is not None:
        phi_true = arr_in[valid_in, mtz_in.column_with_label(col_phi).idx].astype(np.float64)

    existing_freer = None
    if not args.regen_freer:
        for fr_lbl in ("FreeR_flag", "FREE", "R-free-flags"):
            col_obj = mtz_in.column_with_label(fr_lbl)
            if col_obj is not None:
                existing_freer = arr_in[valid_in, col_obj.idx].astype(np.float64)
                break

    # 1. Symmetry factors across ASU reflections
    m_pg, m_max, is_centric, epsilon = compute_symmetry_factors(mtz_in.spacegroup, hkls)

    # 2. Estimate intrinsic Wilson B0 and intensity scale of input F0
    k0_ref = float(cal["K0_ref_1aho"])
    b0_form_ref = float(cal["B0_form_1aho"])
    b0_nom_ref = float(cal.get("B0_nominal_1aho", 18.2))
    k0_in, b0_form_in, b0_nom_est = estimate_f0_wilson_scale(
        s2, i0_mean, epsilon, b0_form_ref, b0_nom_ref
    )
    if args.B0 is not None:
        b0_nom_used = float(args.B0)
        b0_form_used = b0_nom_used - (b0_nom_ref - b0_form_ref)
    else:
        b0_nom_used = b0_nom_est
        b0_form_used = b0_form_in

    scale_i = k0_in / k0_ref
    exp_cfg = resolve_experimental_config(args, float(mtz_in.cell.volume), scale_i)

    # 3. Map desired d_min -> calibrated B_Wilson, geometry, transfer T_xds(u), and SIGIMEAN model
    if not exp_cfg["active"]:
        d_min_eff = float(args.d_min)
        d_emergent = d_min_eff
        cal_p = interpolate_calibration(
            cal, d_min_eff, criterion=args.criterion, b_wilson_override=args.B_wilson
        )
        b_geom_ref = float(cal_p["B_Wilson"])
        b_wilson = b_geom_ref
        if args.delta_B is not None:
            delta_b = float(args.delta_B)
            b_wilson = b0_nom_used + delta_b
        else:
            delta_b = b_wilson - b0_nom_used
            if delta_b < 0.0 and not args.allow_sharpen:
                if not args.quiet:
                    print(
                        f"WARNING: Input F0 already has estimated B_Wilson={b0_nom_used:.2f} A^2 > target "
                        f"{b_wilson:.2f} A^2 for d_min={args.d_min} A; clamping Delta_B=0.0 "
                        f"(pass --allow-sharpen to apply negative Delta_B={delta_b:.2f}).",
                        file=sys.stderr,
                    )
                delta_b = 0.0
        d_edg = float(cal_p["d_edge"])
        d_cor = float(cal_p["d_corner"])
        d_mtz_min = float(cal_p["d_mtz_min"])
        dist_mm = dist_from_dedge(d_edg, NB2_REF["wavelength_A"])
        dist_ref_mm = dist_mm
    else:
        mean_m_pg = float(np.mean(m_pg))
        wl = exp_cfg["wavelength_A"]
        dist_ovr = exp_cfg["distance_mm_override"]
        d_in_min_all = float(np.min(d))

        def _geom_for_d(d_target: float) -> Tuple[Dict[str, Any], float, float, float, float, float]:
            if dist_ovr is not None:
                d_mm_loc = dist_ovr
                d_edg_loc, d_cor_loc = dedge_dcorner_from_dist(d_mm_loc, wl)
                d_edg_eq = d_edg_loc * (NB2_REF["wavelength_A"] / wl)
                cg = dict(interpolate_calibration_by_dedge(cal, d_edg_eq, criterion=args.criterion))
                d_mtz_loc = float(cg["d_mtz_min"]) * (d_cor_loc / max(float(cg["d_corner"]), 1e-6))
                return cg, d_edg_loc, d_cor_loc, d_mtz_loc, d_mm_loc, d_mm_loc
            cg = dict(interpolate_calibration(cal, d_target, criterion=args.criterion))
            d_edg_ref_loc = float(cg["d_edge"])
            d_cor_ref_loc = float(cg["d_corner"])
            d_ref_mm_loc = dist_from_dedge(d_edg_ref_loc, NB2_REF["wavelength_A"])
            d_edg_loc, d_cor_loc = dedge_dcorner_from_dist(d_ref_mm_loc, wl)
            d_mtz_loc = float(cg["d_mtz_min"]) * (d_cor_loc / max(d_cor_ref_loc, 1e-6))
            return cg, d_edg_loc, d_cor_loc, d_mtz_loc, d_ref_mm_loc, d_ref_mm_loc

        if float(args.d_min) <= 0.0:
            # Forward auto-d_min mode: crystal B_Wilson (+ radiation dose) determines emergent d_min
            if args.delta_B is not None:
                delta_b_init = float(args.delta_B)
            elif args.B_wilson is not None:
                delta_b_init = float(args.B_wilson) - b0_nom_used
            else:
                delta_b_init = 0.0
            if delta_b_init < 0.0 and not args.allow_sharpen:
                delta_b_init = 0.0
            delta_b = delta_b_init + exp_cfg["delta_b_rad"]
            b_wilson = b0_nom_used + delta_b
            d_emergent = solve_emergent_dmin(
                cal, b_wilson, args.criterion, exp_cfg,
                k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg,
                d_min_floor=max(0.55, d_in_min_all),
            )
            d_min_eff = d_emergent
            cal_p, d_edg, d_cor, d_mtz_min, dist_mm, dist_ref_mm = _geom_for_d(d_min_eff)
        elif args.B_wilson is not None or args.delta_B is not None:
            # Forward explicit-B mode with user-specified nominal d_min
            d_min_eff = float(args.d_min)
            if args.delta_B is not None:
                delta_b_init = float(args.delta_B)
            else:
                delta_b_init = float(args.B_wilson) - b0_nom_used
            if delta_b_init < 0.0 and not args.allow_sharpen:
                if not args.quiet:
                    print(
                        f"WARNING: Input F0 estimated B_Wilson={b0_nom_used:.2f} A^2 > requested B_Wilson; "
                        f"clamping initial Delta_B=0.0 (pass --allow-sharpen to sharpen).",
                        file=sys.stderr,
                    )
                delta_b_init = 0.0
            delta_b = delta_b_init + exp_cfg["delta_b_rad"]
            b_wilson = b0_nom_used + delta_b
            cal_p, d_edg, d_cor, d_mtz_min, dist_mm, dist_ref_mm = _geom_for_d(d_min_eff)
            d_emergent = solve_emergent_dmin(
                cal, b_wilson, args.criterion, exp_cfg,
                k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg,
                d_min_floor=max(0.55, d_in_min_all),
            )
        else:
            # Inverse target-d_min mode: solve for crystal B_Wilson that achieves d_min under exp_cfg
            d_min_eff = float(args.d_min)
            cal_p, d_edg, d_cor, d_mtz_min, dist_mm, dist_ref_mm = _geom_for_d(d_min_eff)
            b_geom_tmp = float(cal_p["B_Wilson"])
            bw_solved = solve_b_wilson_for_dmin(
                cal, d_min_eff, args.criterion, cal_p, b_geom_tmp, d_edg, dist_mm, dist_ref_mm,
                k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg, exp_cfg,
            )
            delta_b_init = bw_solved - exp_cfg["delta_b_rad"] - b0_nom_used
            if delta_b_init < 0.0 and not args.allow_sharpen:
                delta_b_init = 0.0
                delta_b = exp_cfg["delta_b_rad"]
                b_wilson = b0_nom_used + delta_b
                d_emergent = solve_emergent_dmin(
                    cal, b_wilson, args.criterion, exp_cfg,
                    k0_in, scale_i, b0_nom_ref, b0_form_ref, m_max, mean_m_pg,
                    d_min_floor=max(0.55, d_in_min_all),
                )
                if not args.quiet:
                    print(
                        f"WARNING: Experimental setup is photon/background/dose-limited and cannot reach "
                        f"d_min={d_min_eff:.3f} A without sharpening F0 (solved B_Wilson={bw_solved:.2f} A^2 < "
                        f"B0+Delta_B_rad={b_wilson:.2f} A^2); clamping initial Delta_B=0.0 "
                        f"(emergent {args.criterion} limit = {d_emergent:.3f} A).",
                        file=sys.stderr,
                    )
            else:
                b_wilson = bw_solved
                delta_b = b_wilson - b0_nom_used
                d_emergent = d_min_eff

        b_geom_ref = float(cal_p["B_Wilson"])
        cal_p["d_crit"] = d_emergent

    b_form_final = b0_form_used + delta_b

    # 4. Apply resolution cutoff (default 'corner' uses calibrated FreeRed.mtz d_mtz_min)
    if args.cutoff == "corner":
        d_cut = d_mtz_min
    elif args.cutoff == "edge":
        d_cut = d_edg
    elif args.cutoff == "dmin":
        d_cut = d_min_eff
    else:  # "input"
        d_cut = float(np.min(d))

    d_in_min = float(np.min(d))
    if d_in_min > d_cut + 1e-3 and not args.quiet:
        print(
            f"NOTE: Input MTZ stops at d={d_in_min:.3f} A, coarser than {args.cutoff} cutoff d={d_cut:.3f} A.",
            file=sys.stderr,
        )

    keep_cut = d >= d_cut if args.cutoff == "corner" else d >= (d_cut - 1e-6)
    hkls = hkls[keep_cut]
    d = d[keep_cut]
    s2 = s2[keep_cut]
    fp0 = fp0[keep_cut]
    fm0 = fm0[keep_cut]
    f0_mean = f0_mean[keep_cut]
    m_pg = m_pg[keep_cut]
    is_centric = is_centric[keep_cut]
    epsilon = epsilon[keep_cut]
    if phi_true is not None:
        phi_true = phi_true[keep_cut]
    if existing_freer is not None:
        existing_freer = existing_freer[keep_cut]

    n_ref = len(d)
    if n_ref == 0:
        raise ValueError(f"No reflections remain after applying d >= {d_cut:.3f} A cutoff.")

    rng = np.random.default_rng(args.seed)

    # 5. Pure physical ground truth F_true, I_true at target B_Wilson (uncorrupted by XDS spot-profile roll-off)
    atten_f = np.exp(-delta_b * s2)
    fp_true = fp0 * atten_f
    fm_true = fm0 * atten_f
    if has_anom_in:
        f_cen = np.sqrt(0.5 * (fp_true[is_centric] ** 2 + fm_true[is_centric] ** 2))
        fp_true[is_centric] = f_cen
        fm_true[is_centric] = f_cen
    ip_true = fp_true * fp_true
    im_true = fm_true * fm_true
    i_true = 0.5 * (ip_true + im_true)
    f_true = np.sqrt(i_true)

    # 6. Detector coordinate u = d_edge / d, XDS transfer T_xds(u), background offset, & completeness mask
    u_det = d_edg / d
    f_ring = np.where(
        u_det <= 1.0,
        1.0,
        np.maximum(1.0 - (4.0 / math.pi) * np.arccos(np.clip(1.0 / u_det, -1.0, 1.0)), 0.22),
    )
    f_p2_rel = f_prot_sq(s2) / float(f_prot_sq(np.array(0.0)))
    i_wil = np.maximum(epsilon * k0_in * f_p2_rel * np.exp(-2.0 * b_form_final * s2), 1e-12)

    s_mag = np.sqrt(s2)
    if exp_cfg["active"]:
        bg_var_mult, spot_w_ratio = compute_exp_background_and_spot_factors(
            s_mag, b_wilson, b_geom_ref, dist_mm, dist_ref_mm, exp_cfg
        )
    else:
        bg_var_mult = None
        spot_w_ratio = None

    if args.ideal_transfer:
        t_u = np.ones(n_ref, dtype=np.float64)
        f_nl = np.ones(n_ref, dtype=np.float64)
        bg_u = np.zeros(n_ref, dtype=np.float64)
    else:
        t_u = np.interp(u_det, cal_p["u_t_knots"], cal_p["t_xds"])
        if spot_w_ratio is not None and np.any(np.abs(spot_w_ratio - 1.0) > 1e-4):
            t_u = np.where(
                t_u < 1.0,
                np.maximum(0.02, np.power(np.maximum(t_u, 1e-6), np.sqrt(spot_w_ratio))),
                t_u,
            )
        bg_u = np.interp(u_det, cal_p["u_bg_knots"], cal_p["beta_bg"])

        # High-u within-shell XDS profile-fitting vs direct-summation intensity capture non-linearity
        # (mean-preserving per resolution shell; ramps in only where T_xds(u) rolls off above u=0.56)
        delta_nl_u1 = float(cal_p.get("delta_nl_u1", 0.0))
        if abs(delta_nl_u1) > 1e-6:
            r_iw = i_true / i_wil
            br = r_iw / (r_iw + 1.5)
            al_nl = float(np.clip((b_wilson - 24.0) / 5.0, 0.0, 1.0))
            del_vec = np.interp(
                u_det,
                [(1.0 - al_nl) * 0.44 + al_nl * 0.56, (1.0 - al_nl) * 0.64 + al_nl * 0.72],
                [0.0, delta_nl_u1],
            )
            n_sh_nl = min(24, max(2, n_ref // 120))
            ord_s2 = np.argsort(s2)
            f_nl = np.ones(n_ref, dtype=np.float64)
            for ch_nl in np.array_split(ord_s2, n_sh_nl):
                bbar_c = float(np.sum(i_true[ch_nl] * br[ch_nl]) / max(float(np.sum(i_true[ch_nl])), 1e-12))
                raw_g = np.maximum(0.25, 1.0 + del_vec[ch_nl] * (br[ch_nl] - bbar_c))
                f_nl[ch_nl] = raw_g * (float(np.sum(i_true[ch_nl])) / max(float(np.sum(i_true[ch_nl] * raw_g)), 1e-12))
        else:
            f_nl = np.ones(n_ref, dtype=np.float64)

        # Per-dataset radial-ring background fluctuation strictly outside d_crit (u > u_lim_sim)
        sig_bg_cor = float(cal_p.get("sigma_bg_cor", 0.0))
        if sig_bg_cor > 1e-5:
            u_lim_sim = d_edg / max(float(cal_p["d_crit"]), 0.5)
            u_rings = np.arange(u_lim_sim, 1.50, 0.065)
            if len(u_rings) >= 2:
                amp_rings = sig_bg_cor * np.clip((u_rings - u_lim_sim) / 0.11, 0.0, 1.0)
                z_rings = rng.normal(0.0, 1.0, size=len(u_rings)) * amp_rings
                z_rings[0] = 0.0
                bg_u = bg_u + np.where(u_det > u_lim_sim, np.interp(u_det, u_rings, z_rings), 0.0)

    t_eff = t_u * f_nl
    ip_xds = ip_true * t_eff
    im_xds = im_true * t_eff
    i_xds = i_true * t_eff

    obs_mask = np.ones(n_ref, dtype=bool)
    obs_p_wedge: Optional[np.ndarray] = None
    obs_m_wedge: Optional[np.ndarray] = None
    if not args.no_corner_drop:
        p_base = np.clip(np.interp(u_det, cal_p["u_p_knots"], cal_p["p_obs"]), 0.0, 1.0)
        if m_max != 8:
            p_base = 1.0 - np.power(np.maximum(1.0 - p_base, 0.0), float(m_max) / 8.0)
        p_obs = np.where(d > 12.0, max(0.0, 1.0 - float(cal_p["p_miss_low12"])), p_base)
        obs_mask = rng.random(n_ref) < p_obs
        if exp_cfg["active"] and exp_cfg["wedge_deg"] < 180.0:
            w_arc = min(1.0, (exp_cfg["wedge_deg"] + exp_cfg["dphi_rock_deg"]) / 180.0)
            m_half = np.maximum(1, m_pg // 2)
            p_mate_ac = 1.0 - np.power(np.maximum(1.0 - w_arc * f_ring, 0.0), m_half)
            p_cen_w = 1.0 - np.power(np.maximum(1.0 - w_arc * f_ring, 0.0), m_pg)
            obs_p_wedge = obs_mask & (rng.random(n_ref) < np.where(is_centric, p_cen_w, p_mate_ac))
            obs_m_wedge = np.where(is_centric, obs_p_wedge, obs_mask & (rng.random(n_ref) < p_mate_ac))
            obs_mask = obs_p_wedge | obs_m_wedge

    # 7. Simulate realistic SIGIMEAN calibrated in the XDS-attenuated intensity frame
    wat_bump = np.exp(-(((s_mag - 0.160) / 0.045) ** 2))
    cor_bump = np.maximum(0.0, u_det - 0.95) ** 2
    sig_bg = (float(cal_p["sig0"]) * scale_i) * np.exp(
        float(cal_p["b1_u2"]) * (u_det * u_det)
        + float(cal_p["w_wat"]) * wat_bump
        + float(cal_p.get("b2_cor", 0.0)) * cor_bump
    )
    g_shot = float(cal_p["g_shot"]) * scale_i
    c_sig = float(cal_p["c_sig"])

    if not exp_cfg["active"]:
        mult_factor = np.sqrt(float(m_max) / (m_pg.astype(np.float64) * f_ring))
        mult_factor_pix = mult_factor
        c_sig_cal = c_sig
        c_sig_nc = 0.0
    else:
        sig_bg = sig_bg * np.sqrt(bg_var_mult)
        g_shot = g_shot / exp_cfg["gamma_g_obs"]
        c_sig_cal = c_sig * exp_cfg["f_cal"]
        c_sig_nc = exp_cfg["sigma_noncal"] / math.sqrt(max(float(m_max), 1.0))
        c_sig = math.sqrt(c_sig_cal * c_sig_cal + c_sig_nc * c_sig_nc)
        gw = exp_cfg["gamma_wedge"]
        gp = exp_cfg["gamma_pix"]
        raw_obs = gw * m_pg.astype(np.float64) * f_ring
        eff_obs = np.maximum(1.0, raw_obs) if gw < 1.0 else raw_obs
        raw_pix = gp * m_pg.astype(np.float64) * f_ring
        eff_pix = np.maximum(1.0, raw_pix) if gp < 1.0 else raw_pix
        mult_factor = np.sqrt(float(m_max) / np.maximum(eff_obs, 0.22))
        mult_factor_pix = np.sqrt(float(m_max) / np.maximum(eff_pix, 0.22))

    rms_si = float(cal_p["rms_ln_si"])
    rms_si_cor = float(cal_p.get("rms_ln_si_cor", rms_si))
    w_cor_si = np.clip((u_det - 0.98) / 0.08, 0.0, 1.0)
    rms_si_vec = (1.0 - w_cor_si) * rms_si + w_cor_si * rms_si_cor
    eta_sig = np.clip(rng.normal(0.0, 1.0, size=n_ref) * rms_si_vec, -2.5 * rms_si_vec, 2.5 * rms_si_vec)
    sig_i_base = mult_factor * np.sqrt(sig_bg * sig_bg + g_shot * i_xds + (c_sig * i_xds) ** 2) * np.exp(eta_sig)

    # 8. Draw noisy intensities and convert to amplitudes
    a_u0 = float(args.a_stat if args.a_stat is not None else cal_p["a_stat"])
    a_u1 = float(args.a_stat if args.a_stat is not None else cal_p.get("a_stat_u1", a_u0))
    a_u2 = float(args.a_stat if args.a_stat is not None else cal_p.get("a_stat_u2", a_u1))
    c_u0 = float(args.c_sys if args.c_sys is not None else cal_p["c_sysI"])
    c_u1 = float(args.c_sys if args.c_sys is not None else cal_p.get("c_sysI_u1", c_u0))
    c_u2 = float(args.c_sys if args.c_sys is not None else cal_p.get("c_sysI_u2", c_u1))
    c_wbg = float(0.0 if args.c_sys is not None else cal_p.get("c_wbg", 0.0))
    a_ind = float(cal_p["a_ind"])
    c_ind = float(cal_p["c_sys_ind"])

    write_anom = bool(args.anomalous or (has_anom_in and not args.no_anomalous))

    if args.noise_model == "gaussian-i":
        a0_eff, a1_eff, a2_eff = 1.0, 1.0, 1.0
        c0_eff, c1_eff, c2_eff, c_wbg_eff = 0.0, 0.0, 0.0, 0.0
        a_ind_eff, c_ind_eff = 1.0, 0.0
    else:
        a0_eff, a1_eff, a2_eff = a_u0, a_u1, a_u2
        c0_eff, c1_eff, c2_eff, c_wbg_eff = c_u0, c_u1, c_u2, c_wbg
        a_ind_eff, c_ind_eff = a_ind, c_ind
        if args.a_stat is not None:
            a_ind_eff = a0_eff
        if args.c_sys is not None:
            c_ind_eff = c0_eff * 0.35

    wk0, wk1, wk2 = hat_weights_3(u_det, cal_p["u_sc_knots"])
    a_stat_vec = np.sqrt(wk0 * (a0_eff ** 2) + wk1 * (a1_eff ** 2) + wk2 * (a2_eff ** 2))
    c_sys_vec = np.sqrt(wk0 * (c0_eff ** 2) + wk1 * (c1_eff ** 2) + wk2 * (c2_eff ** 2))
    if not exp_cfg["active"] or args.noise_model == "gaussian-i":
        c_ind_vec: Any = c_ind_eff
    elif args.c_sys is not None:
        c_ind_vec = c_ind_eff
    else:
        f_cal = exp_cfg["f_cal"]
        gp = max(exp_cfg["gamma_pix"], 0.02)
        rad_sys_mean = 0.0025 * exp_cfg["dose_mgy"]
        rad_sys_diff = 0.0040 * exp_cfg["dose_mgy"]
        # Pixel-anchored calibration error floors at distinct pixel count (gamma_pix),
        # whereas non-calibration errors (flicker, jitter, vibration, wobble) keep averaging with eff_obs!
        c_cal_rem_mean_sq = np.maximum(0.0, ((a_stat_vec * c_sig_cal) ** 2) * (mult_factor_pix ** 2 - mult_factor ** 2))
        c_nc_rem_mean_sq = np.maximum(0.0, 1.0 - a_stat_vec ** 2) * ((mult_factor * c_sig_nc) ** 2)
        c_sys_vec = np.sqrt(
            ((c_sys_vec * f_cal) ** 2) / gp + c_cal_rem_mean_sq + c_nc_rem_mean_sq + rad_sys_mean ** 2
        )
        c_cal_rem_ind_sq = np.maximum(0.0, 2.0 * ((a_ind_eff * c_sig_cal) ** 2) * (mult_factor_pix ** 2 - mult_factor ** 2))
        c_nc_rem_ind_sq = np.maximum(0.0, 2.0 * (1.0 - a_ind_eff ** 2)) * ((mult_factor * c_sig_nc) ** 2)
        c_ind_vec = np.sqrt(
            ((c_ind_eff * f_cal) ** 2) / gp + c_cal_rem_ind_sq + c_nc_rem_ind_sq + rad_sys_diff ** 2
        )
    sig_wbg = c_wbg_eff * (i_wil * t_u) * np.clip((0.85 - u_det) / 0.15, 0.0, 1.0)

    if args.noise_model == "gaussian-f":
        sig_i_rep = np.where(obs_mask, sig_i_base, np.nan)
        f_center = np.sqrt(i_xds)
        sigma_prior0 = estimate_shell_sigma_prior(
            s2, np.where(obs_mask, i_xds, np.nan), sig_i_rep, epsilon, k0_in, b_form_final, mode="smooth"
        )
        min_isigi_gf = float(args.fw_min_isigi if args.fw_min_isigi is not None else -3.7)
        _, sig_f = french_wilson_vectorized(
            np.where(obs_mask, i_xds, np.nan), sig_i_rep, sigma_prior0, is_centric, min_isigi=min_isigi_gf
        )
        z_f = rng.normal(0.0, 1.0, size=n_ref)
        f_obs = np.where(obs_mask & np.isfinite(sig_f), np.abs(f_center + sig_f * z_f), np.nan)
        i_obs = f_obs * f_obs
        sig_i_out = sig_i_rep
        write_anom = False
    else:
        # Generate Friedel-mate intensities (I+, I-) via exact orthogonal sum/difference coordinates
        # so merged Var(IMEAN) == (a_stat*SIGI)^2 + (c_sys*I)^2 + sig_wbg^2 AND
        # Var(I+ - I-) == (a_ind*SIGD)^2 + 2*(c_ind*I)^2.
        # Multiplicative systematic error smoothly blends Gaussian (at low B <= 25) into unit-variance
        # clipped Student-t (df=5, [-5.5, 5.5] / 1.26681 => Var=1 at B >= 35) matching subcrystal spot splitting.
        z_sum = rng.normal(0.0, 1.0, size=n_ref)
        z_diff = rng.normal(0.0, 1.0, size=n_ref)
        w_t5 = 0.0 if args.noise_model == "gaussian-i" else float(np.clip((b_wilson - 25.0) / 10.0, 0.0, 1.0))
        z_sys_sum = math.sqrt(1.0 - w_t5) * rng.normal(0.0, 1.0, size=n_ref) + math.sqrt(w_t5) * (
            np.clip(rng.standard_t(5.0, size=n_ref), -5.5, 5.5) / 1.26681
        )
        z_sys_diff = math.sqrt(1.0 - w_t5) * rng.normal(0.0, 1.0, size=n_ref) + math.sqrt(w_t5) * (
            np.clip(rng.standard_t(5.0, size=n_ref), -5.5, 5.5) / 1.26681
        )
        e_wbg = sig_wbg * rng.normal(0.0, 1.0, size=n_ref)

        sig_ip = np.sqrt(2.0) * sig_i_base * np.exp(0.005 * z_diff)
        sig_im = np.sqrt(2.0) * sig_i_base * np.exp(-0.005 * z_diff)
        sig_diff = np.sqrt(sig_ip * sig_ip + sig_im * sig_im)

        e_stat_p = a_stat_vec * sig_i_base * z_sum + e_wbg + 0.5 * a_ind_eff * sig_diff * z_diff
        e_stat_m = a_stat_vec * sig_i_base * z_sum + e_wbg - 0.5 * a_ind_eff * sig_diff * z_diff
        inv_rt2 = 1.0 / math.sqrt(2.0)
        s_p = c_sys_vec * z_sys_sum + inv_rt2 * c_ind_vec * z_sys_diff
        s_m = c_sys_vec * z_sys_sum - inv_rt2 * c_ind_vec * z_sys_diff
        var_sp = c_sys_vec * c_sys_vec + 0.5 * (c_ind_vec * c_ind_vec)
        g_sys_p = np.exp(np.clip(s_p, -0.6, 0.6) - 0.5 * var_sp)
        g_sys_m = np.exp(np.clip(s_m, -0.6, 0.6) - 0.5 * var_sp)

        bg_shift = bg_u * sig_i_base
        ip_ac = ip_xds * g_sys_p + bg_shift + e_stat_p
        im_ac = im_xds * g_sys_m + bg_shift + e_stat_m

        sig_ic = sig_i_base
        g_sys_c = np.exp(np.clip(c_sys_vec * z_sys_sum, -0.6, 0.6) - 0.5 * (c_sys_vec * c_sys_vec))
        ic_obs = i_xds * g_sys_c + bg_u * sig_ic + a_stat_vec * sig_ic * z_sum + e_wbg

        wp = 1.0 / (sig_ip * sig_ip)
        wm = 1.0 / (sig_im * sig_im)
        i_obs_ac = (wp * ip_ac + wm * im_ac) / (wp + wm)
        sig_i_ac = 1.0 / np.sqrt(wp + wm)

        i_obs = np.where(is_centric, ic_obs, i_obs_ac)
        sig_i_out = np.where(is_centric, sig_ic, sig_i_ac)
        ip_obs = np.where(is_centric, ic_obs, ip_ac)
        im_obs = np.where(is_centric, ic_obs, im_ac)
        sig_ip = np.where(is_centric, sig_ic, sig_ip)
        sig_im = np.where(is_centric, sig_ic, sig_im)

        if obs_p_wedge is not None and obs_m_wedge is not None:
            only_pw = (~is_centric) & obs_p_wedge & (~obs_m_wedge)
            only_mw = (~is_centric) & (~obs_p_wedge) & obs_m_wedge
            i_obs = np.where(only_pw, ip_obs, np.where(only_mw, im_obs, i_obs))
            sig_i_out = np.where(only_pw, sig_ip, np.where(only_mw, sig_im, sig_i_out))
            i_obs = np.where(obs_mask, i_obs, np.nan)
            sig_i_out = np.where(obs_mask, sig_i_out, np.nan)
            ip_obs = np.where(obs_p_wedge, ip_obs, np.nan)
            sig_ip = np.where(obs_p_wedge, sig_ip, np.nan)
            im_obs = np.where(obs_m_wedge, im_obs, np.nan)
            sig_im = np.where(obs_m_wedge, sig_im, np.nan)
        else:
            i_obs = np.where(obs_mask, i_obs, np.nan)
            sig_i_out = np.where(obs_mask, sig_i_out, np.nan)
            ip_obs = np.where(obs_mask, ip_obs, np.nan)
            im_obs = np.where(obs_mask, im_obs, np.nan)
            sig_ip = np.where(obs_mask, sig_ip, np.nan)
            sig_im = np.where(obs_mask, sig_im, np.nan)

        if args.truncate_mode == "xdsconv":
            xc_p = cal.get("xdsconv_params", {})
            hw = int(xc_p.get("half_win_records", 100))
            clamped = bool(xc_p.get("clamped", True))
            min_isigi = float(args.fw_min_isigi if args.fw_min_isigi is not None else xc_p.get("min_isigi", -3.0))
            min_score = float(xc_p.get("min_score", -35.0))
            f_obs, sig_f, fp_obs, sig_fp, fm_obs, sig_fm, dano, sig_dano, isym = xdsconv_truncate_and_merge(
                s2,
                ip_obs,
                sig_ip,
                im_obs,
                sig_im,
                is_centric,
                half_win=hw,
                clamped=clamped,
                min_isigi=min_isigi,
                min_score=min_score,
            )
        else:
            min_isigi = float(args.fw_min_isigi if args.fw_min_isigi is not None else -3.7)
            sigma_prior = estimate_shell_sigma_prior(
                s2, i_obs, sig_i_out, epsilon, k0_in, b_form_final, mode=args.truncate_mode
            )
            f_obs, sig_f = french_wilson_vectorized(i_obs, sig_i_out, sigma_prior, is_centric, min_isigi=min_isigi)
            if write_anom:
                fp_obs, sig_fp = french_wilson_vectorized(ip_obs, sig_ip, sigma_prior, is_centric, min_isigi=min_isigi)
                fm_obs, sig_fm = french_wilson_vectorized(im_obs, sig_im, sigma_prior, is_centric, min_isigi=min_isigi)
                if obs_p_wedge is not None:
                    ok_pa = np.isfinite(fp_obs) & np.isfinite(sig_fp)
                    ok_ma = np.isfinite(fm_obs) & np.isfinite(sig_fm)
                    dano = np.where(is_centric, 0.0, np.where(ok_pa & ok_ma, fp_obs - fm_obs, np.nan))
                    sig_dano = np.where(is_centric, 0.0, np.where(ok_pa & ok_ma, np.sqrt(sig_fp * sig_fp + sig_fm * sig_fm), np.nan))
                    dano = np.where(np.isfinite(f_obs), dano, np.nan)
                    sig_dano = np.where(np.isfinite(f_obs), sig_dano, np.nan)
                    isym = np.where(
                        ~np.isfinite(f_obs),
                        np.nan,
                        np.where(is_centric | (ok_pa & ok_ma), 0.0, np.where(ok_pa, 1.0, 2.0)),
                    )
                else:
                    dano = np.where(is_centric, 0.0, fp_obs - fm_obs)
                    sig_dano = np.where(is_centric, 0.0, np.sqrt(sig_fp * sig_fp + sig_fm * sig_fm))
                    dano = np.where(np.isfinite(f_obs), dano, np.nan)
                    sig_dano = np.where(np.isfinite(f_obs), sig_dano, np.nan)
                    isym = np.where(np.isfinite(f_obs), 0.0, np.nan)

    # 9. FreeR_flag (0 .. N_bins-1 CCP4 convention, default 0..19)
    if existing_freer is not None and np.all(np.isfinite(existing_freer)):
        freer = existing_freer
    else:
        n_bins_free = max(2, int(round(1.0 / max(args.free_fraction, 0.005))))
        freer = rng.integers(0, n_bins_free, size=n_ref).astype(np.float64)

    # 10. Assemble output Gemmi MTZ
    mtz_out = gemmi.Mtz(with_base=True)
    mtz_out.spacegroup = mtz_in.spacegroup
    mtz_out.set_cell_for_all(mtz_in.cell)
    if not exp_cfg["active"]:
        mtz_out.history = [
            f"Simulated by sim_reso_mtz.py v2: d_min={args.d_min:.3f}A ({args.criterion}), "
            f"B_Wilson={b_wilson:.2f}A^2, Delta_B={delta_b:.2f}A^2, seed={args.seed}"
        ]
    else:
        mtz_out.history = [
            f"sim_reso_mtz v2: d={d_min_eff:.3f}A({args.criterion},em={d_emergent:.3f}A) BW={b_wilson:.2f} dB={delta_b:.2f} s={args.seed}",
            f"Exp: Vi={exp_cfg['v_illum_um3']:.1f}um3 bm={exp_cfg['beam_size_um']:.1f}um flx={exp_cfg['flux_phs']:.1e} t={exp_cfg['exposure_s']:.1f}s w={exp_cfg['wedge_deg']:.0f}d",
            f"Err: wl={exp_cfg['wavelength_A']:.4f}A wat={exp_cfg['water_um']:.0f}um cal={exp_cfg['calib_err']:.4f} ncal={exp_cfg['sigma_noncal']:.4f} dose={exp_cfg['dose_mgy']:.1f}MGy",
        ]

    ds = mtz_out.add_dataset("sim_nanoBragg2")
    ds.crystal_name = "XTAL"
    ds.project_name = "SIM_NANOBRAGG2"
    if not exp_cfg["active"]:
        ds.wavelength = (
            mtz_in.datasets[1].wavelength
            if len(mtz_in.datasets) > 1 and mtz_in.datasets[1].wavelength > 0
            else 0.977794
        )
    else:
        ds.wavelength = float(exp_cfg["wavelength_A"])

    mtz_out.add_column("FreeR_flag", "I", dataset_id=0)
    cols_data: List[np.ndarray] = [
        hkls[:, 0].astype(np.float32),
        hkls[:, 1].astype(np.float32),
        hkls[:, 2].astype(np.float32),
        freer.astype(np.float32),
    ]

    mtz_out.add_column("F", "F", dataset_id=ds.id)
    mtz_out.add_column("SIGF", "Q", dataset_id=ds.id)
    cols_data.extend([f_obs.astype(np.float32), sig_f.astype(np.float32)])

    if write_anom:
        mtz_out.add_column("DANO", "D", dataset_id=ds.id)
        mtz_out.add_column("SIGDANO", "Q", dataset_id=ds.id)
        mtz_out.add_column("ISYM", "Y", dataset_id=ds.id)
        mtz_out.add_column("Fplus", "G", dataset_id=ds.id)
        mtz_out.add_column("SIGFplus", "L", dataset_id=ds.id)
        mtz_out.add_column("Fminus", "G", dataset_id=ds.id)
        mtz_out.add_column("SIGFminus", "L", dataset_id=ds.id)
        cols_data.extend(
            [
                dano.astype(np.float32),
                sig_dano.astype(np.float32),
                isym.astype(np.float32),
                fp_obs.astype(np.float32),
                sig_fp.astype(np.float32),
                fm_obs.astype(np.float32),
                sig_fm.astype(np.float32),
            ]
        )

    mtz_out.add_column("IMEAN", "J", dataset_id=ds.id)
    mtz_out.add_column("SIGIMEAN", "Q", dataset_id=ds.id)
    cols_data.extend([i_obs.astype(np.float32), sig_i_out.astype(np.float32)])

    if write_anom:
        mtz_out.add_column("Iplus", "K", dataset_id=ds.id)
        mtz_out.add_column("SIGIplus", "M", dataset_id=ds.id)
        mtz_out.add_column("Iminus", "K", dataset_id=ds.id)
        mtz_out.add_column("SIGIminus", "M", dataset_id=ds.id)
        cols_data.extend(
            [
                ip_obs.astype(np.float32),
                sig_ip.astype(np.float32),
                im_obs.astype(np.float32),
                sig_im.astype(np.float32),
            ]
        )

    if args.keep_true:
        mtz_out.add_column("F_true", "F", dataset_id=ds.id)
        mtz_out.add_column("I_true", "J", dataset_id=ds.id)
        cols_data.extend([f_true.astype(np.float32), i_true.astype(np.float32)])
        if phi_true is not None:
            mtz_out.add_column("PHI_true", "P", dataset_id=ds.id)
            cols_data.append(phi_true.astype(np.float32))

    data_mat = np.column_stack(cols_data)
    mtz_out.set_data(data_mat)

    out_path = Path(args.output) if args.output else Path(f"sim_{d_min_eff:.2f}A.mtz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mtz_out.write_to_file(str(out_path))

    elapsed = time.perf_counter() - t0

    # 11. Diagnostic summary
    if not args.quiet:
        v_f = np.isfinite(f_obs) & np.isfinite(sig_f) & (sig_f > 0.0)
        in_res = v_f & (d >= d_min_eff)
        # Evaluate both k,B-scaled R_F vs F_true and 20-shell-scaled R_F vs F_true
        r_f_kb = float("nan")
        r_f_sh = float("nan")
        if int(in_res.sum()) > 20 and np.sum(f_true[in_res]) > 0:
            idx_in = np.flatnonzero(in_res)
            ord_in = idx_in[np.argsort(s2[idx_in])]
            n_sh = min(20, max(2, len(ord_in) // 15))
            xs, ys, ws = [], [], []
            loc_i = np.ones(n_ref, dtype=np.float64)
            for ch in np.array_split(ord_in, n_sh):
                so, st = float(np.sum(i_obs[ch])), float(np.sum(i_true[ch]))
                if so > 0 and st > 0:
                    xs.append(float(np.mean(s2[ch])))
                    ys.append(math.log(st / so))
                    ws.append(len(ch))
                    loc_i[ch] = st / so
            if len(xs) >= 2:
                slp, icp = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
                sc_kb = np.sqrt(np.exp(icp + slp * s2[in_res]))
                r_f_kb = float(np.sum(np.abs(f_obs[in_res] * sc_kb - f_true[in_res])) / np.sum(f_true[in_res]))
            r_f_sh = float(
                np.sum(np.abs(f_obs[in_res] * np.sqrt(loc_i[in_res]) - f_true[in_res])) / np.sum(f_true[in_res])
            )

        sh = np.isfinite(i_obs) & np.isfinite(sig_i_out) & (np.abs(d - d_min_eff) <= 0.05 * d_min_eff)
        if sh.sum() < 25:
            idx_near = np.argsort(np.abs(np.where(np.isfinite(i_obs), d, 1e9) - d_min_eff))[: min(80, n_ref)]
            sh = np.zeros(n_ref, dtype=bool)
            sh[idx_near] = np.isfinite(i_obs[idx_near])
        cc_true_sh = (
            float(np.corrcoef(i_obs[sh], i_true[sh])[0, 1])
            if sh.sum() >= 5 and np.std(i_true[sh]) > 0
            else float("nan")
        )
        cc12_sh = compute_cc12_from_sigi(i_obs[sh], sig_i_out[sh])
        isig_sh = float(np.mean(i_obs[sh] / sig_i_out[sh])) if sh.sum() >= 5 else float("nan")

        b_true_cryst = b_wilson - (b0_nom_ref - b0_form_ref)
        print(f"=== sim_reso_mtz.py v2 Summary ({elapsed*1000:.1f} ms) ===")
        print(f"  Input MTZ         : {args.input_mtz} (cols: {col_p}{',' + col_m if col_m else ''})")
        print(f"  Space group / Cell: {mtz_in.spacegroup.xhm()} | {[round(x, 2) for x in mtz_in.cell.parameters]}")
        if not exp_cfg["active"]:
            print(f"  Target d_min      : {d_min_eff:.3f} A ({args.criterion})")
        else:
            em_note = f" | Emergent {args.criterion} limit: {d_emergent:.3f} A" if abs(d_emergent - d_min_eff) > 1e-3 or float(args.d_min) <= 0.0 else ""
            print(f"  Target d_min      : {d_min_eff:.3f} A ({args.criterion}){em_note}")
            clip_note = " [beam-clipped]" if exp_cfg["beam_clipped"] else ""
            print(
                f"  Experiment setup  : V_illum = {exp_cfg['v_illum_um3']:.1f} um^3 (L={exp_cfg['l_xtal_um']:.2f} um{clip_note}), "
                f"beam = {exp_cfg['beam_size_um']:.1f} um, flux = {exp_cfg['flux_phs']:.2e} ph/s, exp = {exp_cfg['exposure_s']:.2f} s"
            )
            print(
                f"  Optics / Bg / Err : wedge = {exp_cfg['wedge_deg']:.1f} deg (osc={exp_cfg['osc_deg']:.2f} deg, n_det_pos={exp_cfg['n_det_pos']}), "
                f"wl = {exp_cfg['wavelength_A']:.4f} A, dist = {dist_mm:.1f} mm, water = {exp_cfg['water_um']:.1f} um, air = {exp_cfg['air_mm']:.1f} mm"
            )
            nc_parts = []
            if exp_cfg["sigma_lumped"] > 0:
                nc_parts.append(f"lumped={100*exp_cfg['sigma_lumped']:.2f}%")
            if exp_cfg["sigma_flicker"] > 0:
                nc_parts.append(f"flicker={100*exp_cfg['sigma_flicker']:.2f}%")
            if exp_cfg["sigma_jitter"] > 0:
                nc_parts.append(f"jitter={100*exp_cfg['sigma_jitter']:.2f}%")
            if exp_cfg["sigma_vib"] > 0:
                nc_parts.append(f"vib={100*exp_cfg['sigma_vib']:.2f}%")
            if exp_cfg["sigma_wobble"] > 0:
                nc_parts.append(f"wobble={100*exp_cfg['sigma_wobble']:.2f}%")
            nc_str = f"{100*exp_cfg['sigma_noncal']:.2f}% ({', '.join(nc_parts)})" if nc_parts else "0.00%"
            print(
                f"  Fractional errors : calib_err (pixel-locked) = {100*exp_cfg['calib_err']:.2f}%, "
                f"noncal_err (non-repeating) = {nc_str}, dose = {exp_cfg['dose_mgy']:.2f} MGy (+{exp_cfg['delta_b_rad']:.2f} A^2)"
            )
        print(f"  Input B0          : {b0_nom_used:.2f} A^2 (nanoBragg2 gauge) | {b0_form_used:.2f} A^2 (form-factor Wilson B)")
        print(f"  Target B_Wilson   : {b_wilson:.2f} A^2 (nanoBragg2 gauge) | {b_true_cryst:.2f} A^2 (form-factor Wilson B)")
        c_rep_mid = float(np.median(c_sys_vec))
        print(
            f"  Applied Delta_B   : {delta_b:.2f} A^2 | XDS transfer: {'ideal (1.0)' if args.ideal_transfer else 'calibrated T_xds(u)'} "
            f"| Scatter: a=({a0_eff:.2f},{a1_eff:.2f},{a2_eff:.2f}), c_sys=({c0_eff:.4f},{c1_eff:.4f},{c2_eff:.4f})"
            + (f" [eff med={c_rep_mid:.4f}]" if exp_cfg["active"] else "")
        )
        print(
            f"  Detector geometry : d_edge = {d_edg:.3f} A, d_corner = {d_cor:.3f} A, d_mtz_min = {d_mtz_min:.3f} A "
            f"(cutoff='{args.cutoff}' -> {d_cut:.3f} A)"
        )
        print(f"  Reflections       : {int(v_f.sum())} valid F / {n_ref} rows (d >= d_min: {int(in_res.sum())})")
        if np.isfinite(r_f_sh):
            print(
                f"  R_F (d >= d_min)  : k,B-scaled = {100*r_f_kb:.2f}% | 20-shell-scaled = {100*r_f_sh:.2f}%"
            )
        print(
            f"  Shell @ d_min     : Mn(I/sd) = {isig_sh:.2f}, est CC1/2 = {cc12_sh:.3f}, "
            f"CC(Iobs,Itrue) = {cc_true_sh:.3f}"
        )
        print(f"  Output MTZ        : {out_path} ({len(mtz_out.columns)} columns)")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast empirical v2 simulator of nanoBragg2/XDS/Aimless/XDSCONV MTZ datasets from F0 and d_min."
    )
    p.add_argument("input_mtz", help="Ground-truth MTZ file for simulation, or observed MTZ/SF-mmCIF file when --deduce is used.")
    p.add_argument(
        "d_min",
        nargs="?",
        type=float,
        default=None,
        help="Desired resolution limit d_min in Angstroms (e.g. 2.0), <=0 in forward physical mode to solve emergent d_min, or omit with --deduce.",
    )
    p.add_argument("-o", "--output", default=None, help="Output MTZ path (default: sim_<d_min>A.mtz).")
    p.add_argument("--f-col", default=None, help="Amplitude/intensity column label (or 'Fplus,Fminus'). Auto-detected if omitted.")
    p.add_argument("--f-minus-col", default=None, help="Optional Friedel-minus column label if not comma-separated in --f-col.")
    p.add_argument("--phi-col", default=None, help="Optional ground-truth phase column for --keep-true.")
    p.add_argument(
        "--criterion",
        choices=list(CRITERION_KEY.keys()),
        default="cc12",
        help="Resolution limit definition matching d_min (default: cc12 = Aimless CC1/2=0.3).",
    )
    p.add_argument(
        "--cutoff",
        choices=["corner", "edge", "dmin", "input"],
        default="corner",
        help="High-resolution reflection cutoff in output MTZ (default: 'corner' matching FreeRed.mtz d_mtz_min).",
    )
    p.add_argument("--no-corner-drop", action="store_true", help="Keep 100%% completeness (disable beamstop & detector/corner drop-off).")
    p.add_argument("--B-wilson", type=float, default=None, help="Override target B_Wilson (A^2, nanoBragg2 gauge).")
    p.add_argument("--B0", type=float, default=None, help="Override estimated intrinsic B0 of input_mtz (A^2, nanoBragg2 gauge).")
    p.add_argument("--delta-B", type=float, default=None, help="Override Delta_B applied to F0 directly (A^2).")
    p.add_argument("--allow-sharpen", action="store_true", help="Allow negative Delta_B if input F0 already has higher B than target.")
    p.add_argument(
        "--ideal-transfer",
        action="store_true",
        help="Disable XDS spot-profile resolution transfer curve T_xds(u) and outer-corner background offset (pure Wilson falloff).",
    )
    p.add_argument("--apply-xds-b", action="store_true", help="Deprecated v1 compatibility flag (v2 includes curved T_xds(u) by default).")
    p.add_argument(
        "--noise-model",
        choices=["nanobragg2", "gaussian-i", "gaussian-f"],
        default="nanobragg2",
        help="Scatter model: 'nanobragg2' (default, calibrated random scatter + XDSCONV), 'gaussian-i' (pure SIGI + XDSCONV), or 'gaussian-f'.",
    )
    p.add_argument("--a-stat", type=float, default=None, help="Override statistical SIGIMEAN scatter multiplier.")
    p.add_argument("--c-sys", type=float, default=None, help="Override within-shell systematic intensity error fraction c_sysI.")
    p.add_argument("--anomalous", action="store_true", help="Force full 19-column anomalous FreeRed.mtz output even for non-anomalous F0.")
    p.add_argument("--no-anomalous", action="store_true", help="Output merged F/SIGF/IMEAN/SIGIMEAN only, even if input has F+/F-.")
    p.add_argument("--keep-true", action="store_true", help="Append pure ground-truth F_true, I_true (and PHI_true if present) columns to output MTZ.")
    p.add_argument("--free-fraction", type=float, default=0.05, help="Fraction for FreeR_flag=0 if input lacks FreeR_flag (default: 0.05).")
    p.add_argument("--regen-freer", action="store_true", help="Regenerate FreeR_flag even if input_mtz already has one.")
    p.add_argument(
        "--truncate-mode",
        choices=["xdsconv", "ccp4", "smooth"],
        default="xdsconv",
        help="Amplitude conversion: 'xdsconv' (default, exact XDSCONV CCP4_I sliding-window FW & F+/F- merge), 'ccp4', or 'smooth'.",
    )
    p.add_argument("--fw-min-isigi", type=float, default=None, help="Minimum I/SIGI before rejecting F as NaN (default: -3.0 for xdsconv, -3.7 otherwise).")
    p.add_argument("--seed", type=int, default=1, help="Random number generator seed (default: 1).")
    p.add_argument("--cal-file", default=None, help="Path to custom calibration JSON.")
    p.add_argument("--quiet", action="store_true", help="Suppress summary printout.")

    eg = p.add_argument_group(
        "Experimental & mechanical data-collection parameters (optional; defaults = nanoBragg2 baseline)"
    )
    eg.add_argument("--pdb-typical", action="store_true", help="Preset matching 2013-2023 PDB median <B_atom> vs d_min (13.5 um crystal in 30 um beam, 60 um water, 0.30 deg mosaicity, 2%% noncal error).")
    eg.add_argument("--xtal-vol-um3", type=float, default=None, help="Crystal volume in um^3 (nanoBragg2 ref: 17549 um^3 = 21^3+17^3+15^3).")
    eg.add_argument("--xtal-size-um", type=float, default=None, help="Cubic crystal edge length in um (nanoBragg2 ref: 25.99 um).")
    eg.add_argument("--beam-size-um", type=float, default=None, help="Square top-hat X-ray beam width in um (nanoBragg2 ref: 30.0 um).")
    eg.add_argument("--match-beam", action="store_true", help="Scale beam width (and water thickness unless --water-um set) to crystal size at constant beam brightness.")
    eg.add_argument("--flux", type=float, default=None, help="Incident beam flux in photons/s (nanoBragg2 ref: 1e12 ph/s in 30x30 um^2).")
    eg.add_argument("--exposure-s", type=float, default=None, help="Total dataset exposure time in seconds (nanoBragg2 ref: 36.0 s).")
    eg.add_argument("--wedge-deg", type=float, default=None, help="Total rotation wedge in degrees (nanoBragg2 ref: 360.0 deg; <180 reduces completeness, >360 multi-pass).")
    eg.add_argument("--osc-deg", type=float, default=None, help="Oscillation width per image in degrees (nanoBragg2 ref: 1.0 deg; e.g. 0.1 for fine-slicing).")
    eg.add_argument("--wavelength", type=float, default=None, help="X-ray wavelength in Angstroms (nanoBragg2 ref: 0.977794 A).")
    eg.add_argument("--distance-mm", type=float, default=None, help="Override Pilatus 6M detector distance in mm.")
    eg.add_argument("--water-um", type=float, default=None, help="Illuminated liquid/vitreous water thickness in um (nanoBragg2 ref: 25.0 um).")
    eg.add_argument("--air-mm", type=float, default=None, help="Illuminated air path length in mm (nanoBragg2 ref: 25.0 mm; 0 for helium/in-vacuum).")
    eg.add_argument("--mosaicity-deg", type=float, default=None, help="Crystal mosaic spread in degrees (nanoBragg2 ref: 0.20 deg).")
    eg.add_argument("--divergence-mrad", type=float, default=None, help="Beam divergence in mrad (nanoBragg2 ref: 0.10 mrad).")
    eg.add_argument("--dispersion-pct", type=float, default=None, help="Spectral dispersion dlambda/lambda in percent (nanoBragg2 ref: 0.20%%).")
    eg.add_argument("--calib-err", type=float, default=None, help="Pixel-anchored detector flat-field/gain fractional RMS error (nanoBragg2 ref: 0.03; repeats on same pixels across 360 deg turns).")
    eg.add_argument("--n-det-pos", type=int, default=1, help="Number of distinct detector/kappa positions across multi-turn (>360 deg) sweeps so pixel calibration error averages down (default: 1).")
    eg.add_argument("--noncal-err", type=float, default=None, help="Lumped non-calibration fractional intensity error per spot passage (does NOT repeat on same pixels; default: 0.0).")
    eg.add_argument("--beam-flicker", type=float, default=None, help="Uncompensated incident beam intensity fractional flicker per spot passage (default: 0.0).")
    eg.add_argument("--shutter-jitter-ms", type=float, default=None, help="Mechanical shutter / frame-gate timing RMS jitter in ms (couples to exposure_s, wedge_deg, osc_deg; default: 0.0).")
    eg.add_argument("--sample-vib-um", type=float, default=None, help="Cryo-stream sample/loop transverse RMS vibration in um (couples to beam_size_um, xtal_size_um, frame time; default: 0.0).")
    eg.add_argument("--spindle-wobble-um", type=float, default=None, help="Goniometer spindle sphere-of-confusion RMS runout in um (couples to beam_size_um, xtal_size_um; default: 0.0).")
    eg.add_argument("--spindle-vel-err", type=float, default=None, help="Spindle angular velocity fractional ripple d(omega)/omega (default: 0.0).")
    eg.add_argument("--dose-mgy", type=float, default=None, help="Absorbed X-ray dose in MGy (adds global B growth and non-isomorphism; default: 0.0).")
    eg.add_argument("--auto-dose", action="store_true", help="Estimate absorbed dose (MGy) automatically from incident photon fluence and wavelength.")
    eg.add_argument("--rad-b-per-mgy", type=float, default=0.70, help="Wedge-averaged global Wilson B increase per MGy of dose (default: 0.70 A^2/MGy).")
    eg.add_argument("--scale-by-cell", action="store_true", help="Apply physical Darwin unit-cell volume & scattering density scaling relative to 1AHO (56,231 A^3).")

    ig = p.add_argument_group(
        "Inverse forensic deduction mode (deduce experimental parameters from observed F,SIGF / I,SIGI)"
    )
    ig.add_argument("--deduce", action="store_true", help="Run inverse parameter deduction on input_mtz (MTZ or SF-mmCIF) instead of forward simulation.")
    ig.add_argument("--sigf-col", default=None, help="Amplitude sigma column label for --deduce (auto-detected if omitted).")
    ig.add_argument("--i-col", default=None, help="Intensity column label for --deduce (auto-detected if omitted).")
    ig.add_argument("--sigi-col", default=None, help="Intensity sigma column label for --deduce (auto-detected if omitted).")
    ig.add_argument("--f-only", action="store_true", help="In --deduce mode, ignore IMEAN/SIGIMEAN columns and invert strictly from (F, SIGF).")
    ig.add_argument("--n-residues", type=int, default=None, help="Residues per ASU for exact absolute-electron Wilson scaling in --deduce mode.")
    ig.add_argument("--solvent-content", type=float, default=None, help="Solvent fraction (0..1) if --n-residues omitted in --deduce mode (default: 0.47).")
    ig.add_argument("--fc-col", default=None, help="Optional reference/model amplitude column (e.g. FC or F_true) to audit GAIN/SDfac in --deduce mode.")
    ig.add_argument("--model", default=None, help="Optional PDB/mmCIF coordinate model to audit GAIN/SDfac in --deduce mode.")
    ig.add_argument("--deduce-json", default=None, help="Write structured --deduce output dictionary to JSON file.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.deduce:
        from deduce_mtz_params import deduce_dataset_parameters, format_deduction_report

        res = deduce_dataset_parameters(
            input_path=args.input_mtz,
            cal_file=args.cal_file,
            f_col=args.f_col,
            sigf_col=args.sigf_col,
            i_col=args.i_col,
            sigi_col=args.sigi_col,
            fc_col=args.fc_col,
            model_path=args.model,
            n_residues=args.n_residues,
            solvent_content=args.solvent_content,
            f_only=args.f_only,
        )
        if args.deduce_json:
            out_p = Path(args.deduce_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        if not args.quiet:
            print(format_deduction_report(res))
        return 0

    if args.d_min is None:
        parser.error("the following arguments are required: d_min (unless --deduce is specified)")
    return simulate_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())

