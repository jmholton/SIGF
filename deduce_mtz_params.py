#!/usr/bin/env python3
"""
deduce_mtz_params.py — Inverse forensic & physical parameter deduction from diffraction MTZ / SF-mmCIF files.

Given a merged crystallographic reflection file containing (H, K, L, F, SIGF) [and optionally
IMEAN, SIGIMEAN, F+/F-, I+/I-, ISYM, or a reference FC / PDB model], this module inverts the
forward error-generation physics of sim_reso_mtz.py WITHOUT reading experimental headers/papers:

  1. Amplitude conversion forensic fingerprint:
       Distinguishes CCP4 TRUNCATE / ctruncate, Kabsch XDSCONV (Friedel-pair French-Wilson +
       0.5*(F+ + F-) merge at I/SIGI >= -3.0), hard sigma cuts (F >= k * SIGF), and non-Bayesian
       sqrt(I) conversions from centric vs acentric F/SIGF lower envelopes, then performs exact
       Bayesian French-Wilson de-truncation (F, SIGF) -> (I_hat, SIGI_hat, I_post) when
       intensities are absent.
  2. Detector cutoff & rank-adaptive multi-pass / multi-crystal changepoint scanner:
       Detects square-detector corner roll-off (fitting inscribed d_edge with French-Wilson
       survival correction) vs spherical cutoff, and scans local rank-adaptive piecewise-linear
       intensity-matched log(SIGI) jump discontinuities across d to find multi-pass / multi-crystal
       merge boundaries (e.g. 3.50, 2.80, 1.11 A in 1AHO) with zero false positives on smooth
       single-pass datasets.
  3. Absolute electron scaling & Debye-Babinet / XDS-transfer-corrected Wilson B-factor:
       Places (F, SIGF, I_hat, SIGI_hat) onto absolute electron (e, e^2) scale via Wilson
       statistics normalized by the universal folded-protein + solvent Debye-Babinet radial
       envelope D_prot(s) (and T_xds(u) when detector corners are present), extracting unbiased
       K0_file, crystallographic Wilson B_form, and nanoBragg2-gauge B_Wilson.
  4. 3D reciprocal-space forensics:
       - Goniometer spindle / capillary axis from the Lorentz missing-reflection blind cone.
       - Symmetry-multiplicity scaling exponent alpha_M (SIGI ~ M_pg^-alpha_M) and 3x3
         directional SIGF anisotropy tensor (incomplete rotation wedge & beam polarization).
       - Low-angle dynamic-range overload / saturation detector (16-bit scanner ceiling).
  5. Joint 2D (shell x intensity-octile) variance decomposition:
       Fits SIGI_ref^2(I, d) = sigma_bg^2(d) + g_shot * I + (c_merged * I)^2 using bias-free
       posterior conditional expectation I_post = E[I_true | data], separating shell-by-shell
       background sigma_bg(d) from within-shell shot noise g_shot and asymptotic proportional
       error floor c_merged (plus within-shell log-SIGI dispersion rms_ln_si).
  6. Background ring & mount classification:
       Decomposes sigma_bg^2(s) across air, silica glass capillary (~3.9 A local peak),
       liquid/vitreous water (~3.15 A), crystal Compton/diffuse scatter, and ice rings.
  7. Darwin photon-budget invariant, physical wavelength & crystal-size bounds, and replay CLI:
       Computes the exact wedge-independent Darwin invariant (Phi_tot * V_illum * lambda^3 * DQE)
       with proper Lorentz-polarization factor LP(d, lambda), diffraction-geometry upper bound on
       wavelength, radiation-damage dose lower bound on illuminated crystal volume (RT vs 100 K
       cryo), nanoBragg2-equivalent experimental parameters, and an executable sim_reso_mtz.py
       forward-replay command.
  8. Optional model cross-check (--fc-col or --model):
       Evaluates weak-Fcalc background z-residuals vs bright-Fcalc residuals to detect
       uncalibrated detector GAIN / uninflated SDfac (invisible from F, SIGF alone).

Dependencies: numpy, gemmi.
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

from sim_reso_mtz import (
    NB2_REF,
    _F_AIR_GRID,
    _F_WATER_GRID,
    _STOL_GRID,
    compute_cc12_from_sigi,
    compute_symmetry_factors,
    dist_from_dedge,
    f_prot_sq,
    french_wilson_vectorized,
    interpolate_calibration,
    interpolate_calibration_by_dedge,
    nonbragg_incoherent_f2,
)


# Universal folded-protein + explicit-solvent radial Debye-Babinet modulation D_prot(s)
# vs s = sin(theta)/lambda = 1/(2d) (A^-1): captures Babinet solvent dip (~7.5 A),
# secondary-structure packing peak (~3.8-4.2 A), and covalent 1.2 A bond peak.
_DEBYE_S_KNOTS = np.array(
    [
        0.020, 0.035, 0.050, 0.065, 0.080, 0.095, 0.110, 0.125, 0.140, 0.155,
        0.170, 0.190, 0.210, 0.235, 0.260, 0.290, 0.330, 0.380, 0.440, 0.520, 0.580,
    ],
    dtype=np.float64,
)
_DEBYE_D_PROT = np.array(
    [
        1.5293, 0.9644, 0.6022, 0.4575, 0.4956, 0.9489, 1.4674, 1.7847, 1.7847, 1.5507,
        1.2607, 1.1975, 1.0943, 1.1503, 0.9435, 0.8363, 0.8724, 0.9056, 1.1834, 1.0079, 0.9316,
    ],
    dtype=np.float64,
)


# ==============================================================================
# 1. Exact French-Wilson Lookup & Bayesian De-Truncation Tables
# ==============================================================================
_FW_TABLE_CACHE: Optional[Dict[str, np.ndarray]] = None


def _get_fw_tables() -> Dict[str, np.ndarray]:
    """Build exact 1D French-Wilson posterior moment tables over z = I/sigI - gamma*sigI/Sigma
    using the single shared french_wilson_vectorized integrator from sim_reso_mtz."""
    global _FW_TABLE_CACHE
    if _FW_TABLE_CACHE is not None:
        return _FW_TABLE_CACHE

    z_grid = np.linspace(-20.0, 28.0, 4801, dtype=np.float64)
    ones = np.ones_like(z_grid)
    huge_prior = np.full_like(z_grid, 1e18)

    mu_ac, sd_ac = french_wilson_vectorized(
        z_grid, ones, huge_prior, np.zeros(len(z_grid), dtype=bool), min_isigi=-1e9
    )
    mu_cen, sd_cen = french_wilson_vectorized(
        z_grid, ones, huge_prior, np.ones(len(z_grid), dtype=bool), min_isigi=-1e9
    )
    r_ac = mu_ac / np.maximum(sd_ac, 1e-12)
    r_cen = mu_cen / np.maximum(sd_cen, 1e-12)
    _FW_TABLE_CACHE = {
        "z": z_grid,
        "mu_ac": mu_ac,
        "sd_ac": sd_ac,
        "prod_ac": mu_ac * sd_ac,
        "r_ac": r_ac,
        "mu_cen": mu_cen,
        "sd_cen": sd_cen,
        "prod_cen": mu_cen * sd_cen,
        "r_cen": r_cen,
    }
    return _FW_TABLE_CACHE


def exact_nnls(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Exact active-set non-negative least squares for small column counts (p <= 10)
    or projected Gauss-Seidel on normal equations for larger sparse shell systems."""
    n_cols = a_mat.shape[1]
    if n_cols <= 10:
        best_res = float("inf")
        best_x = np.zeros(n_cols, dtype=np.float64)
        for mask_bits in range(1, 1 << n_cols):
            cols = [j for j in range(n_cols) if (mask_bits >> j) & 1]
            sub = a_mat[:, cols]
            c_sub, _, _, _ = np.linalg.lstsq(sub, b_vec, rcond=None)
            if np.all(c_sub >= -1e-11):
                c_sub = np.maximum(0.0, c_sub)
                cand = np.zeros(n_cols, dtype=np.float64)
                for k_idx, col_j in enumerate(cols):
                    cand[col_j] = c_sub[k_idx]
                diff = a_mat @ cand - b_vec
                res = float(np.dot(diff, diff))
                if res < best_res:
                    best_res = res
                    best_x = cand
        return best_x

    ata = a_mat.T @ a_mat
    atb = a_mat.T @ b_vec
    diag = np.maximum(np.diag(ata), 1e-30)
    x = np.maximum(0.0, np.linalg.lstsq(a_mat, b_vec, rcond=None)[0])
    for _ in range(400):
        x_prev = x.copy()
        for j in range(n_cols):
            r_j = atb[j] - float(np.dot(ata[j], x)) + ata[j, j] * x[j]
            x[j] = max(0.0, r_j / diag[j])
        if float(np.max(np.abs(x - x_prev))) <= 1e-10 * (1.0 + float(np.max(x))):
            break
    return x


def _merge_friedel_pair(
    vp: np.ndarray,
    sp: np.ndarray,
    vm: np.ndarray,
    sm: np.ndarray,
    spacegroup: gemmi.SpaceGroup,
    hkls: np.ndarray,
    is_intensity: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge Friedel mates (F+/F- or I+/I-) into mean amplitude or intensity when merged columns are absent."""
    ops = spacegroup.operations()
    is_cen = np.fromiter((ops.is_reflection_centric(h) for h in hkls.tolist()), dtype=bool, count=len(hkls))
    ok_p = np.isfinite(vp) & np.isfinite(sp) & (sp > 0.0)
    ok_m = np.isfinite(vm) & np.isfinite(sm) & (sm > 0.0)
    both = ok_p & ok_m
    out_v = np.full(len(vp), np.nan, dtype=np.float64)
    out_s = np.full(len(vp), np.nan, dtype=np.float64)

    only_p = ok_p & (~ok_m)
    only_m = ok_m & (~ok_p)
    out_v[only_p] = vp[only_p]
    out_s[only_p] = sp[only_p]
    out_v[only_m] = vm[only_m]
    out_s[only_m] = sm[only_m]

    if np.any(both):
        # On centrics where F+ == F- is duplicated in both columns, keep single-observation sigma!
        dup_cen = both & is_cen & (np.abs(vp - vm) <= 1e-4 * (np.abs(vp) + 1.0))
        out_v[dup_cen] = vp[dup_cen]
        out_s[dup_cen] = sp[dup_cen]
        ind_both = both & (~dup_cen)
        if np.any(ind_both):
            if is_intensity:
                wp = 1.0 / np.maximum(sp[ind_both] * sp[ind_both], 1e-30)
                wm = 1.0 / np.maximum(sm[ind_both] * sm[ind_both], 1e-30)
                out_v[ind_both] = (wp * vp[ind_both] + wm * vm[ind_both]) / (wp + wm)
                out_s[ind_both] = 1.0 / np.sqrt(wp + wm)
            else:
                out_v[ind_both] = 0.5 * (vp[ind_both] + vm[ind_both])
                out_s[ind_both] = 0.5 * np.sqrt(sp[ind_both] * sp[ind_both] + sm[ind_both] * sm[ind_both])

    # Also preserve zero-sigma legacy values if both sigmas are 0/NaN
    if not np.any(np.isfinite(out_v)):
        ok_vp0 = np.isfinite(vp)
        ok_vm0 = np.isfinite(vm)
        out_v = np.where(ok_vp0 & ok_vm0, 0.5 * (vp + vm), np.where(ok_vp0, vp, np.where(ok_vm0, vm, np.nan)))
        out_s = np.zeros_like(out_v)
    return out_v, out_s


# ==============================================================================
# 2. Universal Reflection File Reader (MTZ & SF-mmCIF)
# ==============================================================================
def load_reflection_file(
    path_str: str,
    f_col: Optional[str] = None,
    sigf_col: Optional[str] = None,
    i_col: Optional[str] = None,
    sigi_col: Optional[str] = None,
    fc_col: Optional[str] = None,
    f_only: bool = False,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load (H, K, L) and available amplitude/intensity/anomalous columns from MTZ or SF-mmCIF.
    Never auto-loads F0/F_true/FC unless fc_col is explicitly requested (strictly blind by default)."""
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"Reflection file not found: {p}")

    suffix = p.suffix.lower()
    is_cif = suffix in (".cif", ".ent", ".sf", ".mmcif")
    if not is_cif:
        with p.open("rb") as fh:
            magic = fh.read(4)
        if magic != b"MTZ ":
            is_cif = True

    if not is_cif:
        mtz = gemmi.read_mtz_file(str(p))
        arr = np.array(mtz, copy=False)
        labels = {c.label: c for c in mtz.columns}

        def _get_col(lbl: Optional[str]) -> Optional[np.ndarray]:
            if lbl is None or lbl not in labels:
                return None
            return arr[:, labels[lbl].idx].astype(np.float64)

        det_f, det_sf = f_col, sigf_col
        if det_f is None:
            for fc, sc in (
                ("F", "SIGF"),
                ("FP", "SIGFP"),
                ("FOBS", "SIGFOBS"),
                ("F-obs", "F-obs-sigma"),
                ("F_meas_au", "F_meas_sigma_au"),
            ):
                if fc in labels and sc in labels:
                    det_f, det_sf = fc, sc
                    break
            if det_f is None:
                for c in mtz.columns:
                    if c.type == "F" and c.label not in ("F0", "F_true", "FC", "FC_ALL", "FMODEL", "FWT", "DELFWT"):
                        q_cand = f"SIG{c.label}"
                        if q_cand in labels:
                            det_f, det_sf = c.label, q_cand
                            break
        elif det_sf is None:
            cand_s = f"SIG{det_f}"
            if cand_s in labels:
                det_sf = cand_s

        det_i, det_si = (None, None) if f_only else (i_col, sigi_col)
        if not f_only and det_i is None:
            for ic, sc in (
                ("IMEAN", "SIGIMEAN"),
                ("I", "SIGI"),
                ("IOBS", "SIGIOBS"),
                ("I-obs", "I-obs-sigma"),
            ):
                if ic in labels and sc in labels:
                    det_i, det_si = ic, sc
                    break
            if det_i is None:
                for c in mtz.columns:
                    if c.type == "J" and c.label not in ("I0", "I_true"):
                        q_cand = f"SIG{c.label}"
                        if q_cand in labels:
                            det_i, det_si = c.label, q_cand
                            break
        elif not f_only and det_i is not None and det_si is None:
            cand_si = f"SIG{det_i}"
            if cand_si in labels:
                det_si = cand_si

        det_isym = "ISYM" if "ISYM" in labels else None
        det_dano, det_sdano = ("DANO", "SIGDANO") if ("DANO" in labels and "SIGDANO" in labels) else (None, None)
        det_fp, det_sfp, det_fm, det_sfm = None, None, None, None
        for fp_l, sfp_l, fm_l, sfm_l in (
            ("Fplus", "SIGFplus", "Fminus", "SIGFminus"),
            ("F(+)", "SIGF(+)", "F(-)", "SIGF(-)"),
            ("FP(+)", "SIGFP(+)", "FP(-)", "SIGFP(-)"),
        ):
            if all(k in labels for k in (fp_l, sfp_l, fm_l, sfm_l)):
                det_fp, det_sfp, det_fm, det_sfm = fp_l, sfp_l, fm_l, sfm_l
                break

        if fc_col is not None and fc_col not in labels:
            raise ValueError(f"Requested --fc-col '{fc_col}' not found in MTZ columns: {list(labels)}")

        h_idx = [labels[k].idx for k in ("H", "K", "L")]
        hkls = np.rint(arr[:, h_idx]).astype(np.int32)
        d_arr = np.asarray(mtz.make_d_array(), dtype=np.float64)
        wls = [float(ds.wavelength) for ds in mtz.datasets if float(ds.wavelength) > 0.0]
        header_wl = wls[-1] if wls else None

        f_mtz = _get_col(det_f)
        sf_mtz = _get_col(det_sf)
        i_mtz = _get_col(det_i)
        si_mtz = _get_col(det_si)

        if f_mtz is None and det_fp is not None:
            f_mtz, sf_mtz = _merge_friedel_pair(
                _get_col(det_fp), _get_col(det_sfp), _get_col(det_fm), _get_col(det_sfm),
                mtz.spacegroup, hkls, is_intensity=False,
            )
            det_f, det_sf = f"{det_fp}/{det_fm}", f"{det_sfp}/{det_sfm}"

        if not f_only and i_mtz is None:
            for ip_l, sip_l, im_l, sim_l in (
                ("Iplus", "SIGIplus", "Iminus", "SIGIminus"),
                ("I(+)", "SIGI(+)", "I(-)", "SIGI(-)"),
            ):
                if all(k in labels for k in (ip_l, sip_l, im_l, sim_l)):
                    i_mtz, si_mtz = _merge_friedel_pair(
                        _get_col(ip_l), _get_col(sip_l), _get_col(im_l), _get_col(sim_l),
                        mtz.spacegroup, hkls, is_intensity=True,
                    )
                    det_i, det_si = f"{ip_l}/{im_l}", f"{sip_l}/{sim_l}"
                    break

        return {
            "path": str(p),
            "format": "MTZ",
            "spacegroup": mtz.spacegroup,
            "cell": mtz.cell,
            "header_wavelength_A": header_wl,
            "history": [h.strip() for h in mtz.history],
            "hkls": hkls,
            "d": d_arr,
            "f_col": det_f,
            "sigf_col": det_sf,
            "i_col": det_i,
            "sigi_col": det_si,
            "F": f_mtz,
            "SIGF": sf_mtz,
            "I": i_mtz,
            "SIGI": si_mtz,
            "ISYM": _get_col(det_isym),
            "DANO": _get_col(det_dano),
            "SIGDANO": _get_col(det_sdano),
            "Fplus": _get_col(det_fp),
            "SIGFplus": _get_col(det_sfp),
            "Fminus": _get_col(det_fm),
            "SIGFminus": _get_col(det_sfm),
            "FC": _get_col(fc_col) if fc_col else None,
            "fc_col": fc_col,
        }

    doc = gemmi.cif.read(str(p))
    rblocks = gemmi.as_refln_blocks(doc)
    if not rblocks:
        raise ValueError(f"No reflection loop found in CIF file: {p}")
    rb = rblocks[0]
    cif_sg = rb.spacegroup
    cif_cell = rb.cell
    hkls = np.asarray(rb.make_miller_array(), dtype=np.int32)
    if cif_sg is None or cif_cell.volume <= 1.01:
        # Legacy PDB SF-mmCIF files (e.g. 1998-2005 r1rb5sf.ent) omit _cell/_symmetry from the SF file;
        # fall back to --model or a sibling <stem>.pdb coordinate file if present.
        cand_models: List[Path] = []
        if model_path:
            cand_models.append(Path(model_path))
        stem_clean = p.stem[:-3] if p.stem.lower().endswith("-sf") else (p.stem[1:-2] if (p.stem.lower().startswith("r") and p.stem.lower().endswith("sf")) else p.stem)
        cand_models.extend([p.with_suffix(".pdb"), p.parent / f"{stem_clean}.pdb"])
        for cm in cand_models:
            if cm.is_file():
                try:
                    st_fb = gemmi.read_structure(str(cm))
                    fb_sg = st_fb.find_spacegroup()
                    if fb_sg is not None and st_fb.cell.volume > 1.01:
                        cif_sg = fb_sg
                        cif_cell = st_fb.cell
                        break
                except Exception:
                    pass
    if cif_sg is None or cif_cell.volume <= 1.01:
        raise ValueError(
            f"Missing cell or space group in CIF reflection block: {p} (supply --model <coords.pdb> or convert to MTZ)"
        )

    if rb.cell.volume > 1.01 and rb.spacegroup is not None:
        d_arr = np.asarray(rb.make_d_array(), dtype=np.float64)
    else:
        orth_b = reciprocal_cartesian_matrix(cif_cell)
        sc_c = hkls @ orth_b.T
        inv_d = np.sqrt(np.maximum(np.sum(sc_c * sc_c, axis=1), 1e-24))
        d_arr = np.where(np.any(hkls != 0, axis=1), 1.0 / inv_d, np.nan)
    col_labels = list(rb.column_labels())

    def _cif_arr(candidates: Tuple[str, ...]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        for c_lbl in candidates:
            if c_lbl in col_labels:
                vals = np.asarray(rb.make_float_array(c_lbl), dtype=np.float64)
                if np.any(np.isfinite(vals)):
                    return c_lbl, vals
        return None, None

    det_f, f_vals = (
        (f_col, np.asarray(rb.make_float_array(f_col), dtype=np.float64))
        if f_col
        else _cif_arr(("F_meas_au", "F_meas", "FP"))
    )
    det_sf, sf_vals = (
        (sigf_col, np.asarray(rb.make_float_array(sigf_col), dtype=np.float64))
        if sigf_col
        else _cif_arr(("F_meas_sigma_au", "F_meas_sigma", "SIGFP"))
    )
    if f_only:
        det_i, i_vals, det_si, si_vals = None, None, None, None
    else:
        det_i, i_vals = (
            (i_col, np.asarray(rb.make_float_array(i_col), dtype=np.float64))
            if i_col
            else _cif_arr(("intensity_meas", "I_meas", "IMEAN"))
        )
        det_si, si_vals = (
            (sigi_col, np.asarray(rb.make_float_array(sigi_col), dtype=np.float64))
            if sigi_col
            else _cif_arr(("intensity_sigma", "I_meas_sigma", "SIGIMEAN"))
        )
    det_fp_c, fp_c = _cif_arr(("pdbx_F_plus", "F_plus"))
    det_sfp_c, sfp_c = _cif_arr(("pdbx_F_plus_sigma", "F_plus_sigma"))
    det_fm_c, fm_c = _cif_arr(("pdbx_F_minus", "F_minus"))
    det_sfm_c, sfm_c = _cif_arr(("pdbx_F_minus_sigma", "F_minus_sigma"))

    if f_vals is None and fp_c is not None and fm_c is not None and sfp_c is not None and sfm_c is not None:
        f_vals, sf_vals = _merge_friedel_pair(
            fp_c, sfp_c, fm_c, sfm_c, cif_sg, hkls, is_intensity=False
        )
        det_f = f"{det_fp_c}/{det_fm_c}"
        det_sf = f"{det_sfp_c}/{det_sfm_c}"

    if not f_only and i_vals is None:
        det_ip_c, ip_c = _cif_arr(("pdbx_I_plus", "I_plus"))
        det_sip_c, sip_c = _cif_arr(("pdbx_I_plus_sigma", "I_plus_sigma"))
        det_im_c, im_c = _cif_arr(("pdbx_I_minus", "I_minus"))
        det_sim_c, sim_c = _cif_arr(("pdbx_I_minus_sigma", "I_minus_sigma"))
        if ip_c is not None and im_c is not None and sip_c is not None and sim_c is not None:
            i_vals, si_vals = _merge_friedel_pair(
                ip_c, sip_c, im_c, sim_c, cif_sg, hkls, is_intensity=True
            )
            det_i = f"{det_ip_c}/{det_im_c}"
            det_si = f"{det_sip_c}/{det_sim_c}"

    fc_vals = np.asarray(rb.make_float_array(fc_col), dtype=np.float64) if fc_col else None
    header_wl = float(rb.wavelength) if rb.wavelength > 0.0 else None

    return {
        "path": str(p),
        "format": "mmCIF",
        "spacegroup": cif_sg,
        "cell": cif_cell,
        "header_wavelength_A": header_wl,
        "history": [],
        "hkls": hkls,
        "d": d_arr,
        "f_col": det_f,
        "sigf_col": det_sf,
        "i_col": det_i,
        "sigi_col": det_si,
        "F": f_vals,
        "SIGF": sf_vals,
        "I": i_vals,
        "SIGI": si_vals,
        "ISYM": None,
        "DANO": None,
        "SIGDANO": None,
        "Fplus": fp_c,
        "SIGFplus": sfp_c,
        "Fminus": fm_c,
        "SIGFminus": sfm_c,
        "FC": fc_vals,
        "fc_col": fc_col,
    }


# ==============================================================================
# 3. Theoretical ASU Reflection Lattice (Completeness, Corners, Blind Cone)
# ==============================================================================
def reciprocal_cartesian_matrix(cell: gemmi.UnitCell) -> np.ndarray:
    """Return 3x3 matrix B such that s_cart = B @ hkl (and row-wise hkls @ B.T) has |s_cart| = 1/d (A^-1)."""
    ca = math.cos(math.radians(float(cell.alpha)))
    cb = math.cos(math.radians(float(cell.beta)))
    cg = math.cos(math.radians(float(cell.gamma)))
    sg = math.sin(math.radians(float(cell.gamma)))
    o_mat = np.array(
        [
            [float(cell.a), float(cell.b) * cg, float(cell.c) * cb],
            [0.0, float(cell.b) * sg, float(cell.c) * (ca - cb * cg) / max(sg, 1e-12)],
            [0.0, 0.0, float(cell.volume) / max(float(cell.a) * float(cell.b) * sg, 1e-12)],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(o_mat).T


def build_theoretical_asu_lattice(
    cell: gemmi.UnitCell, spacegroup: gemmi.SpaceGroup, d_min_limit: float, d_max_limit: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Generate all symmetry-unique non-absent ASU reflections (hkls, d) down to d_min_limit."""
    d_lim = max(0.35, float(d_min_limit) * 0.999999)
    max_h = int(math.ceil(cell.a / d_lim)) + 1
    max_k = int(math.ceil(cell.b / d_lim)) + 1
    max_l = int(math.ceil(cell.c / d_lim)) + 1

    try:
        asu = gemmi.ReciprocalAsu(spacegroup)
        ops = spacegroup.operations()
        orth_frac = reciprocal_cartesian_matrix(cell)
        inv_d_max = 1.0 / d_lim
        inv_d_min = 1.0 / max(d_max_limit, d_lim + 1e-3)

        pg = spacegroup.point_group_hm()
        if pg in ("222", "mmm", "422", "4/mmm", "23", "m-3", "432", "m-3m"):
            h_rng = np.arange(0, max_h + 1, dtype=np.int32)
            k_rng = np.arange(0, max_k + 1, dtype=np.int32)
            l_rng = np.arange(0, max_l + 1, dtype=np.int32)
        elif spacegroup.crystal_system_str() == "monoclinic":
            h_rng = np.arange(-max_h, max_h + 1, dtype=np.int32)
            k_rng = np.arange(0, max_k + 1, dtype=np.int32)
            l_rng = np.arange(0, max_l + 1, dtype=np.int32)
        else:
            h_rng = np.arange(-max_h, max_h + 1, dtype=np.int32)
            k_rng = np.arange(-max_k, max_k + 1, dtype=np.int32)
            l_rng = np.arange(0, max_l + 1, dtype=np.int32)

        asu_hkls_list: List[np.ndarray] = []
        hh, kk = np.meshgrid(h_rng, k_rng, indexing="ij")
        hh_f = hh.ravel()
        kk_f = kk.ravel()
        for l_val in l_rng:
            ll_f = np.full_like(hh_f, l_val)
            cand = np.column_stack([hh_f, kk_f, ll_f])
            if l_val == 0:
                cand = cand[(cand[:, 0] != 0) | (cand[:, 1] != 0)]
            sc = cand @ orth_frac.T
            inv_d2 = np.sum(sc * sc, axis=1)
            m_sp = (inv_d2 <= inv_d_max * inv_d_max) & (inv_d2 >= inv_d_min * inv_d_min)
            if not np.any(m_sp):
                continue
            sub = cand[m_sp]
            kept = [row for row in sub.tolist() if asu.is_in(row) and not ops.is_systematically_absent(row)]
            if kept:
                asu_hkls_list.append(np.asarray(kept, dtype=np.int32))

        if not asu_hkls_list:
            return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.float64), "empty_asu_lattice"
        asu_hkls = np.vstack(asu_hkls_list)
        sc_all = asu_hkls @ orth_frac.T
        asu_d = 1.0 / np.sqrt(np.sum(sc_all * sc_all, axis=1))
        return asu_hkls, asu_d, None
    except Exception as exc:
        print(
            f"WARNING: Failed to build theoretical ASU lattice ({exc}); completeness and blind-cone checks disabled.",
            file=sys.stderr,
        )
        return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.float64), str(exc)


def canonicalize_hkls_to_asu(spacegroup: gemmi.SpaceGroup, hkls: np.ndarray) -> np.ndarray:
    """Map arbitrary (H, K, L) indices into gemmi's canonical ReciprocalAsu for exact set matching."""
    asu = gemmi.ReciprocalAsu(spacegroup)
    ops = spacegroup.operations()
    out = np.empty_like(hkls)
    for idx, row in enumerate(hkls.tolist()):
        if asu.is_in(row):
            out[idx] = row
        else:
            hkl_asu, _ = asu.to_asu(row, ops)
            out[idx] = hkl_asu
    return out


# ==============================================================================
# 4. Module 2: Amplitude Conversion Forensic Classifier & De-Truncation
# ==============================================================================
def analyze_amplitude_conversion_and_detruncate(
    s2: np.ndarray,
    f_obs: Optional[np.ndarray],
    sig_f: Optional[np.ndarray],
    i_obs: Optional[np.ndarray],
    sig_i: Optional[np.ndarray],
    isym: Optional[np.ndarray],
    is_centric: np.ndarray,
    epsilon: np.ndarray,
) -> Dict[str, Any]:
    """Identify I->F conversion software/cutoffs and produce bias-free (i_hat, sig_i_hat, i_post)."""
    fw = _get_fw_tables()
    n = len(s2)

    has_f = f_obs is not None and sig_f is not None and np.any(np.isfinite(f_obs) & (sig_f > 0))
    has_i = i_obs is not None and sig_i is not None and np.any(np.isfinite(i_obs) & (sig_i > 0))
    if not has_f and not has_i:
        # Support legacy 1970s-1990s sigma-less PDB depositions (e.g. 101M: F_meas_sigma_au = 0.000 everywhere):
        # still deduce completeness, cutoff geometry, blind cone, overloads, Wilson B_form, and wavelength bounds!
        vf_nosig = (f_obs is not None) and np.any(np.isfinite(f_obs) & (f_obs > 0.0))
        vi_nosig = (i_obs is not None) and np.any(np.isfinite(i_obs))
        if vf_nosig or vi_nosig:
            if vf_nosig:
                assert f_obs is not None
                v_ns = np.isfinite(f_obs) & (f_obs > 0.0)
                ip_ns = np.where(v_ns, f_obs * f_obs, np.nan)
            else:
                assert i_obs is not None
                v_ns = np.isfinite(i_obs)
                ip_ns = np.where(v_ns, np.maximum(i_obs, 0.0), np.nan)
            return {
                "info": {
                    "has_F_columns": False,
                    "has_I_columns": False,
                    "mislabeled_amplitudes_in_intensity_column": False,
                    "sigma_missing_or_zero": True,
                    "software": "Legacy sigma-less deposition (all SIGF/SIGI = 0 or missing; error-variance forensics unavailable)",
                    "truncate_mode": "none",
                    "inferred_min_isigi": None,
                    "cutoff_reached": False,
                },
                "valid": v_ns,
                "i_hat": ip_ns.copy(),
                "sig_i_hat": np.full(n, np.nan, dtype=np.float64),
                "i_post": ip_ns,
                "sigma_source": "missing_or_zero_sigma",
            }
        raise ValueError("No valid (F, SIGF) or (I, SIGI) observations found in file.")

    info: Dict[str, Any] = {
        "has_F_columns": bool(has_f),
        "has_I_columns": bool(has_i),
        "mislabeled_amplitudes_in_intensity_column": False,
    }

    # Detect mislabeled structure-factor amplitudes (F, SIGF) deposited inside intensity columns (I, SIGI),
    # e.g. 2V6F-sf.cif: strictly non-negative values + Rayleigh acentric second-moment <X^2>/<X>^2 ~ 4/pi = 1.273
    # (vs >= 1.50 even for perfectly twinned intensities, and 2.00 for untwinned Wilson intensities).
    mislabeled_f_note: Optional[str] = None
    if has_i and not has_f:
        assert i_obs is not None and sig_i is not None
        vi_chk = np.isfinite(i_obs) & np.isfinite(sig_i) & (sig_i > 0.0)
        if int(np.sum(vi_chk)) >= 150 and bool(np.all(i_obs[vi_chk] >= 0.0)):
            m_ac_m = vi_chk & (~is_centric) & (s2 >= 0.004) & (i_obs > 0.0)
            if int(np.sum(m_ac_m)) < 100:
                m_ac_m = vi_chk & (~is_centric) & (i_obs > 0.0)
            mom_med = float("nan")
            if int(np.sum(m_ac_m)) >= 80:
                ord_m = np.argsort(s2[m_ac_m])
                xm = (i_obs[m_ac_m] / np.maximum(epsilon[m_ac_m], 1.0))[ord_m]
                n_sh_m = min(12, max(3, len(xm) // 80))
                sh_moms = [
                    float(np.mean(ch * ch) / max(float(np.mean(ch)) ** 2, 1e-30))
                    for ch in np.array_split(xm, n_sh_m)
                    if len(ch) >= 15
                ]
                if sh_moms:
                    mom_med = float(np.median(sh_moms))
            snr_chk = i_obs[vi_chk] / sig_i[vi_chk]
            ac_chk = snr_chk[~is_centric[vi_chk]]
            cen_chk = snr_chk[is_centric[vi_chk]]
            min_ac_i = float(np.min(ac_chk)) if len(ac_chk) > 0 else float("nan")
            min_cen_i = float(np.min(cen_chk)) if len(cen_chk) >= 15 else float("nan")
            fw_floor_split = (
                np.isfinite(min_ac_i)
                and np.isfinite(min_cen_i)
                and 1.75 <= min_ac_i <= 2.45
                and 1.20 <= min_cen_i <= 1.70
                and (min_ac_i - min_cen_i) >= 0.35
            )
            if (np.isfinite(mom_med) and mom_med < 1.42) or fw_floor_split:
                info["mislabeled_amplitudes_in_intensity_column"] = True
                if np.isfinite(mom_med):
                    info["acentric_second_moment_ratio"] = round(mom_med, 4)
                mislabeled_f_note = (
                    f"MISLABELED AMPLITUDES IN INTENSITY COLUMN "
                    f"(0 negative values, acentric <X^2>/<X>^2 = {mom_med:.3f} ~ 4/pi) -> "
                    if np.isfinite(mom_med)
                    else "MISLABELED AMPLITUDES IN INTENSITY COLUMN (0 negative values, French-Wilson F/SIGF floor) -> "
                )
                f_obs = i_obs
                sig_f = sig_i
                has_f = True
                has_i = False
                info["has_F_columns"] = True
                info["has_I_columns"] = False

    if has_f:
        assert f_obs is not None and sig_f is not None
        vf = np.isfinite(f_obs) & np.isfinite(sig_f) & (sig_f > 0.0) & (f_obs >= 0.0)
        snr_f = np.full(n, np.nan, dtype=np.float64)
        snr_f[vf] = f_obs[vf] / sig_f[vf]

        ac_vals = np.sort(snr_f[vf & (~is_centric)])
        cen_vals = np.sort(snr_f[vf & is_centric])

        min_ac = float(ac_vals[0]) if len(ac_vals) > 0 else float("nan")
        q005_ac = float(np.percentile(ac_vals, 0.5)) if len(ac_vals) >= 20 else min_ac
        q02_ac = float(np.percentile(ac_vals, 2.0)) if len(ac_vals) >= 20 else min_ac
        min_cen = float(cen_vals[0]) if len(cen_vals) > 0 else float("nan")
        q02_cen = float(np.percentile(cen_vals, 2.0)) if len(cen_vals) >= 10 else min_cen
        n_zero_f = int(np.sum(vf & (f_obs <= 1e-8)))

        info.update(
            {
                "n_valid_F": int(np.sum(vf)),
                "n_acentric_F": int(len(ac_vals)),
                "n_centric_F": int(len(cen_vals)),
                "min_fsigf_acentric": min_ac,
                "p05_fsigf_acentric": q005_ac,
                "p20_fsigf_acentric": q02_ac,
                "min_fsigf_centric": min_cen,
                "p20_fsigf_centric": q02_cen,
            }
        )

        inferred_mode = "ccp4"
        inferred_min_isigi: Optional[float] = None
        software_label = "Unknown"
        cutoff_reached = bool(min_ac < 3.4 or (np.isfinite(min_cen) and min_cen < 2.6))

        if not cutoff_reached:
            software_label = "High-SNR dataset (weak-reflection French-Wilson floor not reached)"
            inferred_mode = "ccp4"
        elif n_zero_f > 0 or min_ac < 1.15:
            software_label = "Non-Bayesian sqrt(max(0, I)) or Gaussian amplitude draw (e.g. synthetic |F0 + N(0,SIGF)|)"
            inferred_mode = "naive_sqrt"
        elif np.isfinite(min_cen) and len(cen_vals) >= 15 and abs(min_ac - min_cen) < 0.12 and abs(round(min_ac) - min_ac) < 0.06:
            k_cut = round(min_ac, 1)
            software_label = f"Hard amplitude sigma cutoff (F >= {k_cut:.1f} * SIGF)"
            inferred_mode = "hard_sigma_cut"
        else:
            has_paired_pileup = False
            if isym is not None and np.any(np.isfinite(isym)):
                paired_ac = vf & (~is_centric) & (np.abs(isym) < 0.25)
                if np.sum(paired_ac) >= 20:
                    p_min_paired = float(np.min(snr_f[paired_ac]))
                    if p_min_paired > 2.55 and (not np.isfinite(min_cen) or min_cen < 1.95):
                        has_paired_pileup = True
            if not has_paired_pileup and q005_ac > 2.58 and np.isfinite(min_cen) and min_cen < 1.90:
                has_paired_pileup = True

            if has_paired_pileup:
                inferred_mode = "xdsconv"
                # XDSCONV cuts on raw I/SIGI >= -3.0 (allowing prior-shifted h = I/SIGI - SIGI/<I> down to -35)
                inferred_min_isigi = -3.0
                z_min_tbl = float(fw["z"][0])
                z_eff_low = float(np.interp(min_cen, fw["r_cen"], fw["z"])) if np.isfinite(min_cen) else -4.5
                z_note = (
                    f"min J_eff <= {z_min_tbl:+.1f} [table floor]"
                    if z_eff_low <= z_min_tbl + 0.05
                    else f"min J_eff = {z_eff_low:+.2f}"
                )
                software_label = (
                    f"Kabsch XDSCONV (WILSON_STATISTICS=TRUE, separate F+/F- French-Wilson + 0.5*(F++F-), "
                    f"I/SIGI >= -3.0 cut; {z_note})"
                )
            else:
                inferred_mode = "ccp4"
                z_min_tbl = float(fw["z"][0])
                z_ac = float(np.interp(min_ac, fw["r_ac"], fw["z"]))
                z_cen = float(np.interp(min_cen, fw["r_cen"], fw["z"])) if np.isfinite(min_cen) and len(cen_vals) >= 10 else z_ac
                z_est = 0.7 * z_ac + 0.3 * z_cen
                # Note: continuous Gauss-Legendre J_eff = -4.00 corresponds to F/SIGF_ac = 1.964, F/SIGF_cen = 1.346,
                # which is the exact output of CCP4 TRUNCATE (-3.7s / -4.0s prior-adjusted cutoff)!
                if -4.18 <= z_est <= -3.55:
                    inferred_min_isigi = -3.7
                    software_label = (
                        f"CCP4 TRUNCATE / ctruncate (merged French-Wilson, classic -3.7/-4.0 sigma prior-adjusted cut; "
                        f"min F/SIGF = {min_ac:.3f} ac, {min_cen:.3f} cen)"
                    )
                elif z_est < -4.18:
                    inferred_min_isigi = round(max(-6.0, z_est), 2)
                    j_str = f"<= {z_min_tbl:+.1f} [table floor]" if z_est <= z_min_tbl + 0.05 else f"= {z_est:+.2f}"
                    software_label = f"Merged French-Wilson with permissive negative-I tail (J_eff_min {j_str})"
                else:
                    inferred_min_isigi = round(min(2.0, z_est + 0.25), 2)
                    software_label = f"Merged French-Wilson (TRUNCATE/ctruncate-like; J_eff_min = {z_est:+.2f})"

        if mislabeled_f_note is not None:
            software_label = mislabeled_f_note + software_label
        info["software"] = software_label
        info["truncate_mode"] = inferred_mode
        info["inferred_min_isigi"] = inferred_min_isigi
        info["cutoff_reached"] = cutoff_reached
    else:
        info["software"] = "Intensity-only file (no F/SIGF conversion applied)"
        info["truncate_mode"] = "none"
        info["inferred_min_isigi"] = None
        info["cutoff_reached"] = False

    if has_i:
        assert i_obs is not None and sig_i is not None
        vi = np.isfinite(i_obs) & np.isfinite(sig_i) & (sig_i > 0.0)
        isig_vals = i_obs[vi] / sig_i[vi]
        min_isig_direct = float(np.min(isig_vals))
        info["n_valid_I"] = int(np.sum(vi))
        info["min_isigi_direct"] = min_isig_direct
        # Note: IMEAN is untruncated, so min(IMEAN/SIGIMEAN) is NOT the I->F cutoff!
        # Only refine inferred_min_isigi from IMEAN when merged ccp4 F has rejected (NaN) rows bracketing a sharp cut.
        if has_f and info.get("truncate_mode") == "ccp4":
            assert f_obs is not None and sig_f is not None
            f_on_vi = np.isfinite(f_obs[vi]) & np.isfinite(sig_f[vi]) & (sig_f[vi] > 0.0)
            if np.any(f_on_vi) and np.any(~f_on_vi):
                max_rej = float(np.max(isig_vals[~f_on_vi]))
                min_surv = float(np.min(isig_vals[f_on_vi]))
                if max_rej <= min_surv + 0.05:
                    info["inferred_min_isigi"] = round(0.5 * (max_rej + min_surv), 2)

        if has_f:
            assert f_obs is not None and sig_f is not None
            vf_i = vi & np.isfinite(f_obs) & np.isfinite(sig_f)
            # Deposited F_meas_au^2 and intensity_meas in SF-mmCIF files frequently carry independent
            # arbitrary scale factors (e.g. F^2 / I = 100..3300); place i_post onto the exact I scale!
            m_sc = vf_i & (i_obs > 3.0 * sig_i) & (f_obs > 0.0)
            if int(np.sum(m_sc)) < 20:
                m_sc = vf_i & (i_obs > 0.0) & (f_obs > 0.0)
            if np.any(m_sc):
                scale_i_over_f2 = float(np.median(i_obs[m_sc] / np.maximum(f_obs[m_sc] * f_obs[m_sc], 1e-30)))
                if abs(scale_i_over_f2 - 1.0) < 0.02:
                    scale_i_over_f2 = 1.0
            else:
                scale_i_over_f2 = 1.0
            info["f2_over_i_scale_ratio"] = round(1.0 / max(scale_i_over_f2, 1e-30), 6)
            mult_fvar = np.where((~is_centric) & (info.get("truncate_mode") == "xdsconv"), 2.0, 1.0)
            i_post = np.where(
                vf_i,
                (f_obs * f_obs + mult_fvar * (sig_f * sig_f)) * scale_i_over_f2,
                np.maximum(i_obs, 0.25 * sig_i),
            )
        else:
            i_post = np.maximum(i_obs, 0.25 * sig_i)
        return {
            "info": info,
            "valid": vi,
            "i_hat": i_obs.copy(),
            "sig_i_hat": sig_i.copy(),
            "i_post": i_post,
            "sigma_source": "direct_SIGI",
        }

    assert f_obs is not None and sig_f is not None
    vf = np.isfinite(f_obs) & np.isfinite(sig_f) & (sig_f > 0.0) & (f_obs >= 0.0)
    r = np.zeros(n, dtype=np.float64)
    r[vf] = f_obs[vf] / sig_f[vf]

    sig_i_hat = np.full(n, np.nan, dtype=np.float64)
    z_eff_arr = np.full(n, np.nan, dtype=np.float64)

    mode = info.get("truncate_mode", "ccp4")
    if mode in ("naive_sqrt", "hard_sigma_cut"):
        sig_i_hat[vf] = 2.0 * f_obs[vf] * sig_f[vf]
        i_hat = np.where(vf, f_obs * f_obs - sig_f * sig_f, np.nan)
        i_post = np.where(vf, f_obs * f_obs + sig_f * sig_f, np.nan)
        return {
            "info": info,
            "valid": vf,
            "i_hat": i_hat,
            "sig_i_hat": sig_i_hat,
            "i_post": i_post,
            "sigma_source": "naive_2FSIGF",
        }

    cen_m = vf & is_centric
    if np.any(cen_m):
        rc = r[cen_m]
        zc = np.interp(rc, fw["r_cen"], fw["z"])
        pc = np.interp(zc, fw["z"], fw["prod_cen"])
        prod_eff = np.where(rc > 12.0, 0.5, pc)
        sig_i_hat[cen_m] = (f_obs[cen_m] * sig_f[cen_m]) / np.maximum(prod_eff, 1e-6)
        z_eff_arr[cen_m] = np.where(rc > 12.0, 0.5 * rc, zc)

    ac_m = vf & (~is_centric)
    if np.any(ac_m):
        ra = r[ac_m]
        if mode == "xdsconv":
            if isym is not None and np.any(np.isfinite(isym[ac_m])):
                paired = np.abs(isym[ac_m]) < 0.25
            else:
                paired = ra >= 2.55
            r_mate = np.where(paired, ra / math.sqrt(2.0), ra)
            za = np.interp(r_mate, fw["r_ac"], fw["z"])
            pa = np.interp(za, fw["z"], fw["prod_ac"])
            prod_eff = np.where(r_mate > 12.0, 0.5, pa)
            sig_i_hat[ac_m] = (f_obs[ac_m] * sig_f[ac_m]) / np.maximum(prod_eff, 1e-6)
            z_single = np.where(r_mate > 12.0, 0.5 * r_mate, za)
            z_eff_arr[ac_m] = np.where(paired, math.sqrt(2.0) * z_single, z_single)
        else:
            za = np.interp(ra, fw["r_ac"], fw["z"])
            pa = np.interp(za, fw["z"], fw["prod_ac"])
            prod_eff = np.where(ra > 12.0, 0.5, pa)
            sig_i_hat[ac_m] = (f_obs[ac_m] * sig_f[ac_m]) / np.maximum(prod_eff, 1e-6)
            z_eff_arr[ac_m] = np.where(ra > 12.0, 0.5 * ra, za)

    mult_fvar = np.where((~is_centric) & (mode == "xdsconv"), 2.0, 1.0)
    i_post = np.where(vf, f_obs * f_obs + mult_fvar * (sig_f * sig_f), np.nan)

    s2_v = s2[vf]
    ip_v = i_post[vf] / epsilon[vf]
    n_sh = min(36, max(6, int(np.sum(vf)) // 150))
    qs = np.quantile(s2_v, np.linspace(0.0, 1.0, n_sh + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    sh_x = np.empty(n_sh, dtype=np.float64)
    sh_y = np.empty(n_sh, dtype=np.float64)
    for j in range(n_sh):
        m_j = (s2_v >= qs[j]) & (s2_v <= qs[j + 1])
        sh_x[j] = float(np.mean(s2_v[m_j]))
        sh_y[j] = max(float(np.mean(ip_v[m_j])), 1e-6)
    sigma_shell = epsilon * np.exp(np.interp(s2, sh_x, np.log(sh_y)))

    gamma_fw = np.where(is_centric, 0.5, 1.0)
    if mode == "xdsconv":
        gamma_fw = np.where(is_centric, 0.5, 2.0)

    i_hat = np.full(n, np.nan, dtype=np.float64)
    high_snr = vf & (r > 12.0)
    low_snr = vf & (~high_snr)
    i_hat[high_snr] = f_obs[high_snr] * f_obs[high_snr]
    i_hat[low_snr] = sig_i_hat[low_snr] * (
        z_eff_arr[low_snr] + gamma_fw[low_snr] * (sig_i_hat[low_snr] / np.maximum(sigma_shell[low_snr], 1e-6))
    )

    return {
        "info": info,
        "valid": vf,
        "i_hat": i_hat,
        "sig_i_hat": sig_i_hat,
        "i_post": i_post,
        "sigma_source": (
            "mislabeled_I_as_F_French_Wilson"
            if info.get("mislabeled_amplitudes_in_intensity_column")
            else "French-Wilson_inverted_SIGF"
        ),
    }


# ==============================================================================
# 5. Detector Geometry & Rank-Adaptive Multi-Pass Merge Boundary Scanner
# ==============================================================================
def detect_geometry_and_pass_boundaries(
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    hkls_all_rows: np.ndarray,
    d_all_rows: np.ndarray,
    hkls_obs: np.ndarray,
    d_obs: np.ndarray,
    i_post_raw: np.ndarray,
    sig_i_raw: np.ndarray,
    m_pg: np.ndarray,
    m_max: int,
) -> Dict[str, Any]:
    """Detect square-detector corner cutoff (d_edge) vs hard spherical cut, low-angle overloads,
    spindle blind cone, and step discontinuities from multi-pass / multi-crystal merging."""
    d_min_obs = float(np.min(d_obs))
    d_max_obs = float(np.max(d_obs))

    asu_hkls, asu_d, lattice_err = build_theoretical_asu_lattice(
        cell, spacegroup, d_min_obs, d_max_limit=max(d_max_obs * 1.05, 50.0)
    )
    obs_asu = canonicalize_hkls_to_asu(spacegroup, hkls_obs) if len(asu_d) > 0 else hkls_obs

    has_lattice = len(asu_d) > 0
    if has_lattice:
        off = 2048
        obs_keys = (
            (obs_asu[:, 0].astype(np.int64) + off) * 16777216
            + (obs_asu[:, 1].astype(np.int64) + off) * 4096
            + (obs_asu[:, 2].astype(np.int64) + off)
        )
        asu_keys = (
            (asu_hkls[:, 0].astype(np.int64) + off) * 16777216
            + (asu_hkls[:, 1].astype(np.int64) + off) * 4096
            + (asu_hkls[:, 2].astype(np.int64) + off)
        )
        obs_key_set = set(obs_keys.tolist())
        asu_present = np.fromiter((k in obs_key_set for k in asu_keys.tolist()), dtype=bool, count=len(asu_keys))

        # Also check all MTZ rows (including NaN-rejected outer corner rows when present!) to pin physical detector d_edge
        if len(hkls_all_rows) > len(hkls_obs):
            rows_asu = canonicalize_hkls_to_asu(spacegroup, hkls_all_rows)
            row_keys = set(
                (
                    (rows_asu[:, 0].astype(np.int64) + off) * 16777216
                    + (rows_asu[:, 1].astype(np.int64) + off) * 4096
                    + (rows_asu[:, 2].astype(np.int64) + off)
                ).tolist()
            )
            asu_in_rows = np.fromiter((k in row_keys for k in asu_keys.tolist()), dtype=bool, count=len(asu_keys))
        else:
            asu_in_rows = asu_present

        in_dmax = asu_d <= d_max_obs * 1.0001
        overall_compl = float(np.mean(asu_present[in_dmax])) if np.any(in_dmax) else float("nan")
    else:
        asu_present = np.zeros(0, dtype=bool)
        asu_in_rows = np.zeros(0, dtype=bool)
        overall_compl = float("nan")

    # 1. Distinguish square-detector corner roll-off, STARANISO ellipsoidal truncation, and spherical cutoff
    cutoff_type = "spherical"
    inferred_d_edge = d_min_obs
    ellipsoid_limits_a: Optional[List[float]] = None
    outer_compl = overall_compl
    if has_lattice and np.sum(asu_d <= d_max_obs) >= 200:
        s3 = 1.0 / (asu_d ** 3)
        m_in = asu_d <= d_max_obs * 1.0001
        qs = np.quantile(s3[m_in], np.linspace(0.0, 1.0, 25))
        bin_d, bin_c_obs, bin_c_row = [], [], []
        for j in range(24):
            mb = m_in & (s3 >= qs[j]) & (s3 <= qs[j + 1])
            if np.sum(mb) >= 8:
                bin_d.append(float(np.median(asu_d[mb])))
                bin_c_obs.append(float(np.mean(asu_present[mb])))
                bin_c_row.append(float(np.mean(asu_in_rows[mb])))
        bin_d_a = np.asarray(bin_d)
        bin_c_a = np.asarray(bin_c_obs)
        bin_cr_a = np.asarray(bin_c_row)
        if len(bin_c_a) >= 6:
            outer_compl = float(bin_c_a[-1])
            plateau_c = float(np.percentile(bin_c_a[1 : max(4, len(bin_c_a) - 6)], 75))
            # Square-detector corners and ellipsoidal (STARANISO) cutoffs both drop below 49% of plateau completeness
            # in the outermost 1/d^3 bin. Separate them via 3D directional resolution limits along principal axes!
            if outer_compl < 0.49 * plateau_c:
                cutoff_type = "detector_corners"
                u_3d = np.array([0.0, 0.99, 1.02, 1.05, 1.09, 1.14, 1.22, 1.30, 1.38], dtype=np.float64)
                p_3d = np.array([1.0, 1.00, 0.99, 0.92, 0.73, 0.52, 0.21, 0.10, 0.03], dtype=np.float64)
                best_err = float("inf")
                best_de = float(bin_d_a[-4])
                fit_sl = slice(2, len(bin_d_a))
                for de_cand in np.linspace(d_min_obs * 1.02, d_min_obs * 1.42, 120):
                    u_c = de_cand / bin_d_a[fit_sl]
                    pred_surv = plateau_c * np.interp(u_c, u_3d, p_3d)
                    err = float(np.mean((bin_c_a[fit_sl] - pred_surv) ** 2))
                    if err < best_err:
                        best_err = err
                        best_de = float(de_cand)
                inferred_d_edge = best_de

                # Check for ellipsoidal anisotropic cutoff (STARANISO / anisotropic truncation):
                # Test both crystal reciprocal axes (a*, b*, c*) and observed <s s^T> eigenvectors.
                # Under STARANISO, at least one direction present at low d (>=35% compl) is hard-cut to ~0% at high d!
                orth_el = reciprocal_cartesian_matrix(cell)
                s_obs_vec = hkls_obs @ orth_el.T
                if len(s_obs_vec) >= 150:
                    cov_s = (s_obs_vec.T @ s_obs_vec) / float(len(s_obs_vec))
                    _, eig_vecs = np.linalg.eigh(cov_s)
                    ax_unit = orth_el / np.maximum(np.linalg.norm(orth_el, axis=0, keepdims=True), 1e-12)
                    s_obs_u = s_obs_vec / np.maximum(np.linalg.norm(s_obs_vec, axis=1, keepdims=True), 1e-12)
                    s_asu_vec = asu_hkls[m_in] @ orth_el.T
                    s_asu_u = s_asu_vec / np.maximum(np.linalg.norm(s_asu_vec, axis=1, keepdims=True), 1e-12)
                    d_asu_in = asu_d[m_in]
                    pres_in = asu_present[m_in]

                    for basis_mat in (ax_unit, eig_vecs):
                        d_obs_caps, rel_caps, cap_vecs = [], [], []
                        for k_ax in range(3):
                            vk = basis_mat[:, k_ax]
                            in_obs_k = np.abs(s_obs_u @ vk) >= 0.819  # 35 deg half-angle cap
                            in_asu_k = np.abs(s_asu_u @ vk) >= 0.819
                            if int(np.sum(in_obs_k)) >= 12 and int(np.sum(in_asu_k)) >= 20:
                                d_ok = float(np.percentile(d_obs[in_obs_k], 1.5))
                                d_ak = float(np.percentile(d_asu_in[in_asu_k], 1.5))
                                d_obs_caps.append(d_ok)
                                rel_caps.append(d_ok / max(d_ak, 1e-6))
                                cap_vecs.append(in_asu_k)
                        if len(d_obs_caps) == 3:
                            idx_worst = int(np.argmax(rel_caps))
                            d_worst = d_obs_caps[idx_worst]
                            aniso_d_rat = max(rel_caps) / max(min(rel_caps), 1e-6)
                            low_core_m = cap_vecs[idx_worst] & (d_asu_in >= 1.08 * d_worst)
                            hi_tip_m = cap_vecs[idx_worst] & (d_asu_in <= 0.94 * d_worst)
                            low_core_c = float(np.mean(pres_in[low_core_m])) if int(np.sum(low_core_m)) >= 10 else 0.0
                            hi_tip_c = float(np.mean(pres_in[hi_tip_m])) if int(np.sum(hi_tip_m)) >= 10 else 1.0
                            if aniso_d_rat >= 1.15 and low_core_c >= 0.35 and hi_tip_c <= 0.05:
                                cutoff_type = "ellipsoidal_anisotropic"
                                inferred_d_edge = float(d_worst)
                                ellipsoid_limits_a = [round(x, 3) for x in sorted(d_obs_caps)]
                                break

    # 2. Spindle / capillary axis detection via reciprocal-space Lorentz blind cone
    spindle_info: Dict[str, Any] = {"detected": False, "axis": None, "axis_completeness": {}}
    if has_lattice:
        d_cone_cut = max(
            d_min_obs,
            inferred_d_edge if cutoff_type in ("detector_corners", "ellipsoidal_anisotropic") else d_min_obs,
        )
        m_cone = (asu_d >= d_cone_cut) & (asu_d <= min(d_max_obs, 15.0))
        if np.sum(m_cone) >= 150:
            orth_frac = reciprocal_cartesian_matrix(cell)
            sc_vec = asu_hkls[m_cone] @ orth_frac.T
            sc_norm = sc_vec / np.maximum(np.linalg.norm(sc_vec, axis=1, keepdims=True), 1e-12)
            pres_c = asu_present[m_cone]
            base_c = float(np.mean(pres_c))

            recip_axes = {}
            for idx_ax, ax_lbl in enumerate(("a*", "b*", "c*")):
                e_hkl = np.zeros(3, dtype=np.float64)
                e_hkl[idx_ax] = 1.0
                v_ax = orth_frac @ e_hkl
                v_ax /= max(float(np.linalg.norm(v_ax)), 1e-12)
                cos_ang = np.abs(sc_norm @ v_ax)
                in_ax = cos_ang > 0.906
                if np.sum(in_ax) >= 12:
                    c_ax = float(np.mean(pres_c[in_ax]))
                    n_ax = int(np.sum(in_ax))
                    se_ax = math.sqrt(max(base_c * (1.0 - base_c), 0.01) / n_ax)
                    z_drop = (base_c - c_ax) / max(se_ax, 1e-6)
                    recip_axes[ax_lbl] = {
                        "completeness": round(c_ax, 4),
                        "n_possible": n_ax,
                        "drop_vs_overall": round(base_c - c_ax, 4),
                        "z_drop": round(z_drop, 2),
                    }
            spindle_info["axis_completeness"] = recip_axes
            if len(recip_axes) == 3:
                worst_ax = min(recip_axes.items(), key=lambda kv: kv[1]["completeness"])
                other_med = float(np.median([v["completeness"] for k, v in recip_axes.items() if k != worst_ax[0]]))
                if (other_med - worst_ax[1]["completeness"]) >= 0.08 and worst_ax[1]["z_drop"] >= 4.0:
                    real_ax = {"a*": "a", "b*": "b", "c*": "c"}[worst_ax[0]]
                    spindle_info["detected"] = True
                    spindle_info["axis"] = f"{real_ax}-axis ({worst_ax[0]} blind cone: {100*worst_ax[1]['completeness']:.1f}% vs {100*other_med:.1f}% orthogonal)"

    # 3. Low-angle detector saturation / overload check (away from blind cone, beamstop, & outer cutoff!)
    overload_info: Dict[str, Any] = {"detected": False}
    if has_lattice:
        d_core_lim = max(
            d_min_obs,
            inferred_d_edge if cutoff_type in ("detector_corners", "ellipsoidal_anisotropic") else d_min_obs,
        )
        if d_core_lim <= 2.6:
            m_low_asu = (asu_d >= 3.2) & (asu_d <= min(d_max_obs * 0.92, 12.0))
            if spindle_info["detected"] and spindle_info["axis"]:
                ax_idx = {"a": 0, "b": 1, "c": 2}[spindle_info["axis"][0]]
                orth_frac = reciprocal_cartesian_matrix(cell)
                sc_l = asu_hkls[m_low_asu] @ orth_frac.T
                sc_ln = sc_l / np.maximum(np.linalg.norm(sc_l, axis=1, keepdims=True), 1e-12)
                v_sp = orth_frac[:, ax_idx] / max(float(np.linalg.norm(orth_frac[:, ax_idx])), 1e-12)
                not_cone = np.abs(sc_ln @ v_sp) <= 0.82
                n_low_tot = int(np.sum(not_cone))
                n_low_miss = int(np.sum(~asu_present[m_low_asu][not_cone]))
            else:
                n_low_tot = int(np.sum(m_low_asu))
                n_low_miss = int(np.sum(~asu_present[m_low_asu]))

            m_mid_asu = (asu_d >= 2.0) & (asu_d < 3.0)
            has_mid_ref = int(np.sum(m_mid_asu)) >= 50
            mid_miss_rate = float(np.mean(~asu_present[m_mid_asu])) if has_mid_ref else 1.0
            low_miss_rate = n_low_miss / max(n_low_tot, 1)
            overload_info = {
                "detected": bool(has_mid_ref and n_low_tot >= 60 and n_low_miss >= 6 and low_miss_rate > max(0.018, 2.5 * mid_miss_rate)),
                "n_missing_low_angle_noncone": n_low_miss,
                "n_possible_low_angle_noncone": n_low_tot,
                "low_angle_missing_pct": round(100.0 * low_miss_rate, 2),
            }

    # 4. Rank-Adaptive Blind Multi-Pass / Multi-Crystal Merge Boundary Scanner
    # Sorting by 1/d and using an adaptive local rank window [k - W, k + W] catches pass boundaries
    # BOTH at low resolution (e.g. 3.50 A & 2.80 A, top 2-5% of reflections!) AND at atomic resolution (1.11 A)!
    sig_norm = sig_i_raw * np.sqrt(m_pg.astype(np.float64) / float(m_max))
    s_inv_d = 1.0 / d_obs
    good = np.isfinite(sig_norm) & (sig_norm > 0.0) & np.isfinite(i_post_raw) & (i_post_raw > 0.0)
    if cutoff_type == "detector_corners":
        good = good & (d_obs >= 1.01 * inferred_d_edge)

    ord_s = np.argsort(s_inv_d[good])
    s_g = s_inv_d[good][ord_s]
    lsig_g = np.log(sig_norm[good][ord_s])
    l_ip_g = np.log(i_post_raw[good][ord_s])
    hkls_g = hkls_obs[good][ord_s]
    orth_cp = reciprocal_cartesian_matrix(cell)
    sc_g = hkls_g @ orth_cp.T
    sc_gn = sc_g / np.maximum(np.linalg.norm(sc_g, axis=1, keepdims=True), 1e-12)
    ca2_g = sc_gn[:, 0] ** 2
    cb2_g = sc_gn[:, 1] ** 2
    n_g = len(s_g)

    changepoints: List[Dict[str, Any]] = []
    if n_g >= 400:
        s_lo_valid = float(s_g[min(75, n_g - 1)])
        hi_margin = max(76, int(0.015 * n_g))
        s_hi_valid = float(s_g[max(0, n_g - hi_margin)])
        if s_hi_valid > s_lo_valid + 0.03:
            # Log-spaced 1/d grid gives equal fractional resolution density (dd/d) from low-res (4 A) to atomic-res (0.9 A)
            cand_s = np.geomspace(max(s_lo_valid, 0.04), s_hi_valid, 280)
        else:
            cand_s = np.array([], dtype=np.float64)

        scores_z = np.zeros(len(cand_s), dtype=np.float64)
        steps_ln = np.zeros(len(cand_s), dtype=np.float64)
        wins_arr = np.zeros(len(cand_s), dtype=np.float64)

        for idx_c, s0 in enumerate(cand_s):
            # Scale-invariant relative window (+/- 7.5% in d <=> +/- 0.075 * s0), with rank-adaptive floor of 290
            # reflections per side at low resolution so sparse low-d pass boundaries (e.g. 3.50 A, 2.80 A) have power!
            win_s = max(0.025, 0.075 * float(s0))
            rk = int(np.searchsorted(s_g, s0, side="left"))
            lo_i = min(int(np.searchsorted(s_g, s0 - win_s, side="left")), max(12, rk - 290))
            hi_i = max(int(np.searchsorted(s_g, s0 + win_s, side="right")), min(n_g - 12, rk + 290))
            n_l = rk - lo_i
            n_r = hi_i - rk
            if n_l < 75 or n_r < 75:
                continue
            eff_w = 0.5 * float(s_g[hi_i - 1] - s_g[lo_i])
            if eff_w > 0.15:
                continue
            wins_arr[idx_c] = eff_w
            n_seg = hi_i - lo_i
            ds = s_g[lo_i:hi_i] - s0
            rhs = np.zeros(n_seg, dtype=np.float64)
            rhs[n_l:] = 1.0
            li = l_ip_g[lo_i:hi_i] - float(np.mean(l_ip_g[lo_i:hi_i]))
            x_mat = np.column_stack(
                [
                    np.ones(n_seg),
                    np.minimum(ds, 0.0),
                    np.maximum(ds, 0.0),
                    rhs,
                    li,
                    li * li,
                    ca2_g[lo_i:hi_i] - float(np.mean(ca2_g[lo_i:hi_i])),
                    cb2_g[lo_i:hi_i] - float(np.mean(cb2_g[lo_i:hi_i])),
                ]
            )
            y_seg = lsig_g[lo_i:hi_i]
            beta, _, _, _ = np.linalg.lstsq(x_mat, y_seg, rcond=None)
            resid = y_seg - x_mat @ beta
            dof = max(1, n_seg - x_mat.shape[1])
            mad_s = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
            s2_err = max(mad_s * mad_s, float(np.dot(resid, resid)) / dof * 0.25, 1e-6)
            cov_diag = np.linalg.pinv(x_mat.T @ x_mat)[3, 3]
            se_step = math.sqrt(max(s2_err * cov_diag, 1e-10))
            step_val = float(beta[3])
            steps_ln[idx_c] = step_val
            scores_z[idx_c] = abs(step_val) / se_step

        order_pk = np.argsort(-scores_z)
        kept_s: List[Tuple[float, float]] = []
        for idx_p in order_pk:
            if scores_z[idx_p] < 5.2 or abs(steps_ln[idx_p]) < 0.115:
                break
            s_pk = float(cand_s[idx_p])
            w_pk = float(wins_arr[idx_p])
            if any(abs(s_pk - sk) < min(0.052, 0.80 * min(w_pk, wk)) for sk, wk in kept_s):
                continue

            # Confirm matched-intensity median jump immediately across s_pk
            hw_s = 0.55 * w_pk
            rk_m = int(np.searchsorted(s_g, s_pk, side="left"))
            lo_m = min(int(np.searchsorted(s_g, s_pk - hw_s, side="left")), max(0, rk_m - 120))
            hi_m = max(int(np.searchsorted(s_g, s_pk + hw_s, side="right")), min(n_g, rk_m + 120))
            if rk_m - lo_m < 40 or hi_m - rk_m < 40:
                continue
            sl_l = slice(lo_m, rk_m)
            sl_r = slice(rk_m, hi_m)
            li_lr = l_ip_g[lo_m:hi_m]
            qs_i = np.percentile(li_lr, [15, 42, 68, 92])
            med_rats = []
            for jq in range(len(qs_i) - 1):
                ml_q = (l_ip_g[sl_l] >= qs_i[jq]) & (l_ip_g[sl_l] < qs_i[jq + 1])
                mr_q = (l_ip_g[sl_r] >= qs_i[jq]) & (l_ip_g[sl_r] < qs_i[jq + 1])
                if np.sum(ml_q) >= 6 and np.sum(mr_q) >= 6:
                    med_rats.append(math.exp(float(np.median(lsig_g[sl_r][mr_q]) - np.median(lsig_g[sl_l][ml_q]))))
            if not med_rats:
                continue
            raw_ratio = float(np.median(med_rats))
            if (raw_ratio - 1.0) * steps_ln[idx_p] <= 0.0 or abs(math.log(raw_ratio)) < 0.095:
                continue

            kept_s.append((s_pk, w_pk))
            step_ratio_sigf = math.exp(steps_ln[idx_p])
            changepoints.append(
                {
                    "d_A": round(1.0 / s_pk, 3),
                    "sig_ratio_high_over_low_res": round(step_ratio_sigf, 3),
                    "matched_bin_ratio": round(raw_ratio, 3),
                    "z_score": round(float(scores_z[idx_p]), 1),
                    "event": (
                        "Additional pass turns ON toward higher resolution"
                        if step_ratio_sigf < 1.0
                        else "Pass turns OFF toward higher resolution (fewer passes active)"
                    ),
                }
            )

        changepoints.sort(key=lambda x: -x["d_A"])

    return {
        "has_asu_lattice": has_lattice,
        "lattice_error": lattice_err,
        "asu_d": asu_d,
        "asu_present": asu_present,
        "overall_completeness": overall_compl,
        "outer_shell_completeness": outer_compl,
        "cutoff_type": cutoff_type,
        "inferred_d_edge_A": float(inferred_d_edge),
        "ellipsoidal_principal_d_limits_A": ellipsoid_limits_a,
        "spindle_blind_cone": spindle_info,
        "low_angle_overloads": overload_info,
        "pass_boundaries": changepoints,
    }


def _crossing_d(
    sh_d_a: np.ndarray,
    metric_arr: np.ndarray,
    thresh: float,
    d_fallback: Optional[float] = None,
) -> Tuple[float, bool]:
    """Interpolate resolution d (A) where metric_arr drops below thresh (from high-d to low-d)."""
    fb = float(d_fallback) if d_fallback is not None else (float(np.min(sh_d_a)) if len(sh_d_a) > 0 else 0.0)
    if len(metric_arr) == 0 or np.all(metric_arr >= thresh):
        return fb, False
    for idx in range(len(metric_arr) - 1):
        if metric_arr[idx] >= thresh and metric_arr[idx + 1] < thresh:
            frac = (metric_arr[idx] - thresh) / max(metric_arr[idx] - metric_arr[idx + 1], 1e-9)
            return float(sh_d_a[idx] + frac * (sh_d_a[idx + 1] - sh_d_a[idx])), True
    return float(sh_d_a[0]), True


# ==============================================================================
# 6. Absolute Electron Scale (K0_e, B_form) with Debye-Babinet & T_xds Normalization
# ==============================================================================
def estimate_absolute_electron_wilson(
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    s2: np.ndarray,
    d: np.ndarray,
    i_hat: np.ndarray,
    sig_i_hat: np.ndarray,
    i_post: np.ndarray,
    epsilon: np.ndarray,
    valid: np.ndarray,
    geom_info: Dict[str, Any],
    cal: dict,
    n_residues_asu: Optional[int] = None,
    solvent_content: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate file Wilson scale (K0_file, B_form) normalized by universal folded-protein
    Debye-Babinet envelope D_prot(s) (and XDS spot-profile transfer T_xds(u) when corners present)."""
    n_sym = len(spacegroup.operations())
    v_cell = float(cell.volume)
    k0_ref_1aho = float(cal["K0_ref_1aho"])
    b0_form_1aho = float(cal["B0_form_1aho"])
    b0_nom_1aho = float(cal["B0_nominal_1aho"])
    gauge_offset = b0_nom_1aho - b0_form_1aho

    e2_per_residue = k0_ref_1aho / 256.0
    if n_residues_asu is not None and n_residues_asu > 0:
        n_res_cell = float(n_residues_asu * n_sym)
        k0_abs_e2 = e2_per_residue * n_res_cell
        vm_est = v_cell / max(n_res_cell * 110.0, 1.0)
        f_solv_est = float(np.clip(1.0 - 1.23 / max(vm_est, 1.25), 0.15, 0.85))
        scale_is_assumed = False
        scale_basis = f"n_residues_asu={n_residues_asu} ({int(n_res_cell)} res/cell)"
    else:
        scale_is_assumed = True
        f_solv_est = float(solvent_content) if solvent_content is not None else 0.47
        f_solv_est = float(np.clip(f_solv_est, 0.10, 0.88))
        v_prot = (1.0 - f_solv_est) * v_cell
        n_res_cell = v_prot / 135.3
        k0_abs_e2 = e2_per_residue * n_res_cell
        tag_s = "USER-ASSUMED" if solvent_content is not None else "ASSUMED default"
        scale_basis = (
            f"Matthews_volume [{tag_s}: solvent={100*f_solv_est:.1f}%, ~{n_res_cell/n_sym:.0f} res/ASU; "
            f"pass --n-residues for exact scale]"
        )

    s2_v = s2[valid]
    d_v = d[valid]
    ih_v = i_hat[valid]
    sh_v = sig_i_hat[valid]
    ip_v = i_post[valid]
    ep_v = epsilon[valid]

    n_sh = min(32, max(6, int(np.sum(valid)) // 110))
    edges_q = np.quantile(s2_v, np.linspace(0.0, 1.0, n_sh + 1))
    edges_q[0] -= 1e-9
    edges_q[-1] += 1e-9

    has_sig_v = bool(np.any(np.isfinite(sh_v) & (sh_v > 0.0)))
    sh_s2, sh_d, sh_ie, sh_cc12, sh_isig, sh_n = [], [], [], [], [], []
    for j in range(n_sh):
        m = (s2_v >= edges_q[j]) & (s2_v <= edges_q[j + 1])
        if np.sum(m) < 8:
            continue
        s2_m = float(np.mean(s2_v[m]))
        d_m = 0.5 / math.sqrt(max(s2_m, 1e-12))
        m_sp = m & np.isfinite(sh_v) & (sh_v > 0.0)
        if int(np.sum(m_sp)) >= 4:
            cc12_m = compute_cc12_from_sigi(ih_v[m_sp], sh_v[m_sp])
            isig_m = float(np.mean(ih_v[m_sp] / sh_v[m_sp]))
        else:
            cc12_m = float("nan")
            isig_m = 5.0
        mean_ih = float(np.mean(ih_v[m] / ep_v[m]))
        mean_ip = float(np.mean(ip_v[m] / ep_v[m]))
        ie_m = mean_ih if mean_ih > 0.20 * mean_ip else 0.5 * max(mean_ih, 0.0) + 0.5 * mean_ip
        if ie_m > 0.0:
            sh_s2.append(s2_m)
            sh_d.append(d_m)
            sh_ie.append(ie_m)
            sh_cc12.append(cc12_m if np.isfinite(cc12_m) else (1.0 if not has_sig_v else 0.0))
            sh_isig.append(isig_m if np.isfinite(isig_m) else 5.0)
            sh_n.append(int(np.sum(m)))

    sh_s2_a = np.asarray(sh_s2, dtype=np.float64)
    sh_d_a = np.asarray(sh_d, dtype=np.float64)
    sh_ie_a = np.asarray(sh_ie, dtype=np.float64)
    sh_cc_a = np.asarray(sh_cc12, dtype=np.float64)
    sh_is_a = np.asarray(sh_isig, dtype=np.float64)
    sh_n_a = np.asarray(sh_n, dtype=np.float64)

    d_min_v = float(np.min(d_v))
    if has_sig_v:
        d_cc12, cc12_crossed = _crossing_d(sh_d_a, sh_cc_a, 0.30, d_min_v)
        d_isig15, isig15_crossed = _crossing_d(sh_d_a, sh_is_a, 1.50, d_min_v)
        d_isig20, isig20_crossed = _crossing_d(sh_d_a, sh_is_a, 2.00, d_min_v)
    else:
        d_cc12, cc12_crossed = d_min_v, False
        d_isig15, isig15_crossed = d_min_v, False
        d_isig20, isig20_crossed = d_min_v, False

    # Universal Debye-Babinet radial modulation D_prot(s) allows reliable shells down to d = 9 A (s2 >= 0.003)!
    sh_s_mag = np.sqrt(sh_s2_a)
    d_prot_sh = np.interp(sh_s_mag, _DEBYE_S_KNOTS, _DEBYE_D_PROT)

    # Raw apparent Wilson B (before T_xds de-rolloff); restrict to complete isotropic core for both
    # square-detector corners and STARANISO ellipsoidal cutoffs to avoid outer directional survivorship bias!
    good_w = (sh_s2_a >= 0.004) & ((sh_cc_a >= 0.30) | (sh_is_a >= 1.10))
    if geom_info["cutoff_type"] in ("detector_corners", "ellipsoidal_anisotropic"):
        good_w = good_w & (sh_d_a >= 0.98 * geom_info["inferred_d_edge_A"])
    if np.sum(good_w) < 4:
        good_w = (sh_cc_a >= 0.18) | (sh_is_a >= 0.60)
        if geom_info["cutoff_type"] == "ellipsoidal_anisotropic" and int(np.sum(good_w & (sh_d_a >= 0.95 * geom_info["inferred_d_edge_A"]))) >= 3:
            good_w = good_w & (sh_d_a >= 0.95 * geom_info["inferred_d_edge_A"])
    if np.sum(good_w) < 3:
        good_w = sh_ie_a > 0.0

    f_p2_0 = float(f_prot_sq(np.array(0.0)))
    wts = np.sqrt(sh_n_a[good_w]) * np.clip(sh_is_a[good_w], 0.5, 10.0)
    y_app = sh_ie_a[good_w] / (f_prot_sq(sh_s2_a[good_w]) * d_prot_sh[good_w])
    if len(y_app) >= 2:
        slp_a, icp_a = np.polyfit(sh_s2_a[good_w], np.log(np.maximum(y_app, 1e-30)), 1, w=wts)
        b_form_apparent = -0.5 * float(slp_a)
        k0_file_apparent = math.exp(float(icp_a)) * f_p2_0
    else:
        b_form_apparent = 25.0
        k0_file_apparent = float(np.mean(sh_ie_a))

    # When square-detector corners are present, divide out calibrated T_xds(u) so XDS spot-profile
    # roll-off neither inflates B_form nor biases the extrapolated s=0 scale K0_file!
    if geom_info["cutoff_type"] == "detector_corners":
        d_edg = geom_info["inferred_d_edge_A"]
        cal_g = interpolate_calibration_by_dedge(cal, d_edg, criterion="cc12")
        u_sh = d_edg / np.maximum(sh_d_a, 0.25)
        t_sh = np.maximum(np.interp(u_sh, cal_g["u_t_knots"], cal_g["t_xds"]), 0.08)
        y_corr = sh_ie_a[good_w] / (f_prot_sq(sh_s2_a[good_w]) * d_prot_sh[good_w] * t_sh[good_w])
        if len(y_corr) >= 2:
            slp_c, icp_c = np.polyfit(sh_s2_a[good_w], np.log(np.maximum(y_corr, 1e-30)), 1, w=wts)
            b_form_true = -0.5 * float(slp_c)
            k0_file = math.exp(float(icp_c)) * f_p2_0
        else:
            b_form_true = b_form_apparent
            k0_file = k0_file_apparent
    else:
        b_form_true = b_form_apparent
        k0_file = k0_file_apparent

    b_nb2_apparent = b_form_apparent + gauge_offset
    b_nb2_pre_transfer = b_form_true + gauge_offset
    k_amp_file_per_e = math.sqrt(max(k0_file, 1e-30) / max(k0_abs_e2, 1e-30))
    scale_i_file_per_e2 = k0_file / max(k0_abs_e2, 1e-30)

    # Compute completeness for d >= d_cc12 now that d_cc12 is known
    if geom_info["has_asu_lattice"]:
        asu_d = geom_info["asu_d"]
        asu_p = geom_info["asu_present"]
        m_cc = (asu_d <= float(np.max(d_v)) * 1.0001) & (asu_d >= d_cc12)
        compl_cc12 = float(np.mean(asu_p[m_cc])) if np.any(m_cc) else geom_info["overall_completeness"]
    else:
        compl_cc12 = float("nan")

    return {
        "k0_file": float(k0_file),
        "k0_abs_e2": float(k0_abs_e2),
        "k_amp_file_per_e": float(k_amp_file_per_e),
        "scale_i_file_per_e2": float(scale_i_file_per_e2),
        "scale_basis": scale_basis,
        "scale_is_assumed": bool(scale_is_assumed),
        "f_solvent_est": float(f_solv_est),
        "n_res_asu_est": float(n_res_cell / n_sym),
        "B_form_apparent": float(b_form_apparent),
        "B_form_pre_transfer": float(b_form_true),
        "B_Wilson_apparent_nb2": float(b_nb2_apparent),
        "B_Wilson_pre_transfer_nb2": float(b_nb2_pre_transfer),
        "d_file_min": float(np.min(d_v)),
        "d_file_max": float(np.max(d_v)),
        "d_cc12_030": float(d_cc12),
        "d_est_cc12_from_sigi_030": float(d_cc12),
        "cc12_crossed": bool(cc12_crossed),
        "completeness_d_ge_cc12": float(compl_cc12),
        "d_isig15": float(d_isig15),
        "isig15_crossed": bool(isig15_crossed),
        "d_isig20": float(d_isig20),
        "isig20_crossed": bool(isig20_crossed),
        "outer_shell_cc12": float(sh_cc_a[-1]) if len(sh_cc_a) else float("nan"),
        "outer_shell_isig": float(sh_is_a[-1]) if len(sh_is_a) else float("nan"),
        "shells": {
            "d": sh_d_a.tolist(),
            "s2": sh_s2_a.tolist(),
            "mean_I_eps": sh_ie_a.tolist(),
            "cc12": sh_cc_a.tolist(),
            "isig": sh_is_a.tolist(),
            "n": [int(x) for x in sh_n_a.tolist()],
        },
    }


# ==============================================================================
# 7. Module 5 & 6: Joint 2D Variance Fit, Anisotropy, & Background Decomposition
# ==============================================================================
def fit_variance_and_anisotropy(
    cell: gemmi.UnitCell,
    hkls: np.ndarray,
    d: np.ndarray,
    s2: np.ndarray,
    i_post_e2: np.ndarray,
    sig_i_e2: np.ndarray,
    m_pg: np.ndarray,
    m_max: int,
    is_centric: np.ndarray,
    isym: Optional[np.ndarray],
    geom_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit SIGI_ref^2 = sigma_bg^2(d) + g_shot * I + c_merged^2 * I^2 jointly across (shell, I-octile),
    plus directional SIGF anisotropy, symmetry-multiplicity scaling exponent, and per-zone fits."""
    if not bool(np.any(np.isfinite(sig_i_e2) & (sig_i_e2 > 0.0))):
        return {
            "global": {
                "n": int(len(d)),
                "sigma_bg_med_e2": float("nan"),
                "sigma_bg_lows_e2": float("nan"),
                "sigma_bg_highs_e2": float("nan"),
                "g_shot_e2_per_ph": 0.0,
                "c_merged": 0.0,
                "rms_ln_si": float("nan"),
                "shells_s": [],
                "shells_d": [],
                "shells_bg_e2": [],
            },
            "g_shot_best_e2_per_ph": 0.0,
            "outer_pass_g_shot_e2_per_ph": 0.0,
            "zones": [],
            "alpha_multiplicity_exp": None,
            "anisotropy": {},
            "paired_friedel_fraction": None,
            "wedge_deg_from_isym": None,
            "background_profile": {
                "mount_regime": "Unavailable (sigma-less deposition: all SIGF/SIGI = 0 or missing)",
                "is_rt_capillary": False,
                "has_ice_rings": False,
                "max_ice_spike_ratio": 1.0,
            },
            "anomalous_sigma_model": {"detected": False},
        }
    d_edg = geom_info["inferred_d_edge_A"]
    if geom_info["cutoff_type"] == "detector_corners":
        u_det = d_edg / np.maximum(d, 0.25)
        f_ring = np.where(
            u_det <= 1.0,
            1.0,
            np.maximum(1.0 - (4.0 / math.pi) * np.arccos(np.clip(1.0 / u_det, -1.0, 1.0)), 0.22),
        )
    else:
        u_det = np.zeros_like(d)
        f_ring = np.ones_like(d)

    mult_to_ref = np.sqrt((m_pg.astype(np.float64) * f_ring) / float(m_max))
    sig_i_ref = sig_i_e2 * mult_to_ref

    def _fit_2d_zone(mask_z: np.ndarray) -> Dict[str, Any]:
        iz = i_post_e2[mask_z]
        sz = sig_i_ref[mask_z]
        s2z = s2[mask_z]
        n_z = len(iz)
        if n_z < 30:
            return {
                "n": n_z,
                "sigma_bg_med_e2": float("nan"),
                "g_shot_e2_per_ph": float("nan"),
                "c_merged": float("nan"),
                "shells_s": [],
                "shells_d": [],
                "shells_bg_e2": [],
            }

        n_sh = min(24, max(1, n_z // 130))
        # On atomic/sub-Angstrom datasets (d_min < 1.25 A, s2_max > 0.16), equal-count 1/d^3 quantiles put >96%
        # of shells below 2.2 A and skip the 5.5 -> 2.5 A air / 3.9 A silica / 3.15 A water ring region entirely!
        if n_z >= 3500 and float(np.max(s2z)) > 0.16 and int(np.sum(s2z < 0.060)) >= 120:
            low_s_cuts = [0.095, 0.128, 0.158, 0.195, 0.245]
            s2_fixed = [float(np.min(s2z)) - 1e-9]
            for sc_cut in low_s_cuts:
                s2_c = sc_cut * sc_cut
                if int(np.sum((s2z > s2_fixed[-1]) & (s2z <= s2_c))) >= 36:
                    s2_fixed.append(s2_c)
            rem_m = s2z > s2_fixed[-1]
            n_rem = max(4, 26 - len(s2_fixed))
            if int(np.sum(rem_m)) >= 80:
                qs_hi = np.quantile(s2z[rem_m], np.linspace(0.0, 1.0, n_rem + 1))[1:]
                qs_s2 = np.concatenate([np.asarray(s2_fixed, dtype=np.float64), qs_hi])
            else:
                qs_s2 = np.quantile(s2z, np.linspace(0.0, 1.0, n_sh + 1))
            qs_s2[0] = float(np.min(s2z)) - 1e-9
            qs_s2[-1] = float(np.max(s2z)) + 1e-9
            n_sh = len(qs_s2) - 1
        else:
            qs_s2 = np.quantile(s2z, np.linspace(0.0, 1.0, n_sh + 1))
            qs_s2[0] -= 1e-9
            qs_s2[-1] += 1e-9

        sh_s_centers: List[float] = []
        used_shells = 0
        cell_records: List[Tuple[int, float, float]] = []

        for j in range(n_sh):
            ms = (s2z >= qs_s2[j]) & (s2z <= qs_s2[j + 1])
            if np.sum(ms) < 16:
                continue
            iz_s = iz[ms]
            sz2_s = sz[ms] ** 2
            qi = np.percentile(iz_s, np.linspace(0.0, 100.0, 9))
            added = 0
            for k in range(8):
                mc = (iz_s >= qi[k]) & (iz_s <= qi[k + 1] if k == 7 else iz_s < qi[k + 1])
                if np.sum(mc) >= 3:
                    x_med = float(np.median(iz_s[mc]))
                    y_med = float(np.median(sz2_s[mc]))
                    if y_med > 0.0:
                        cell_records.append((used_shells, x_med, y_med))
                        added += 1
            if added >= 2:
                sh_s_centers.append(math.sqrt(float(np.mean(s2z[ms]))))
                used_shells += 1

        if used_shells == 0:
            s_bg0 = float(np.percentile(sz, 15))
            return {
                "n": n_z,
                "sigma_bg_med_e2": s_bg0,
                "sigma_bg_lows_e2": s_bg0,
                "sigma_bg_highs_e2": s_bg0,
                "g_shot_e2_per_ph": 0.0,
                "c_merged": 0.0,
                "rms_ln_si": 0.0,
                "shells_s": [],
                "shells_d": [],
                "shells_bg_e2": [],
            }

        n_cells = len(cell_records)
        a_mat = np.zeros((n_cells, used_shells + 2), dtype=np.float64)
        b_vec = np.zeros(n_cells, dtype=np.float64)
        for r_idx, (sh_id, x_m, y_m) in enumerate(cell_records):
            w = 1.0 / max(y_m, 1e-12)
            a_mat[r_idx, sh_id] = 1.0 * w
            a_mat[r_idx, used_shells] = x_m * w
            a_mat[r_idx, used_shells + 1] = (x_m * x_m) * w
            b_vec[r_idx] = y_m * w

        sol = exact_nnls(a_mat, b_vec)
        bg_shells = np.sqrt(np.maximum(sol[:used_shells], 0.0))
        g_shot_est = float(sol[used_shells])
        c_merged_est = math.sqrt(max(float(sol[used_shells + 1]), 0.0))

        bg_interp = np.interp(np.sqrt(s2z), np.asarray(sh_s_centers), bg_shells)
        pred_sig = np.sqrt(np.maximum(bg_interp ** 2 + g_shot_est * iz + (c_merged_est * iz) ** 2, 1e-24))
        ln_res = np.log(np.maximum(sz, 1e-24) / pred_sig)
        rms_ln_si = 1.4826 * float(np.median(np.abs(ln_res - np.median(ln_res))))

        return {
            "n": int(n_z),
            "sigma_bg_med_e2": float(np.median(bg_shells)),
            "sigma_bg_lows_e2": float(bg_shells[0]),
            "sigma_bg_highs_e2": float(bg_shells[-1]),
            "g_shot_e2_per_ph": g_shot_est,
            "c_merged": c_merged_est,
            "rms_ln_si": float(rms_ln_si),
            "shells_s": [round(float(x), 5) for x in sh_s_centers],
            "shells_d": [round(0.5 / max(float(x), 1e-6), 3) for x in sh_s_centers],
            "shells_bg_e2": [round(float(x), 3) for x in bg_shells.tolist()],
        }

    interior = (u_det <= 1.02) if geom_info["cutoff_type"] == "detector_corners" else np.ones(len(d), dtype=bool)
    global_fit = _fit_2d_zone(interior)

    zone_fits: List[Dict[str, Any]] = []
    bnds = [cp["d_A"] for cp in geom_info["pass_boundaries"]]
    if bnds:
        edges_d = [float(np.max(d)) + 0.01] + bnds + [float(np.min(d)) - 0.001]
        for idx_z in range(len(edges_d) - 1):
            d_hi, d_lo = edges_d[idx_z], edges_d[idx_z + 1]
            mz = interior & (d <= d_hi) & (d > d_lo)
            if np.sum(mz) >= 40:
                zf = _fit_2d_zone(mz)
                zf["d_range_A"] = [round(d_hi, 3), round(max(d_lo, float(np.min(d))), 3)]
                zone_fits.append(zf)

    s2_med = float(np.percentile(s2[interior], 45))
    outer_fit = _fit_2d_zone(interior & (s2 >= s2_med))
    g_shot_best = (
        outer_fit["g_shot_e2_per_ph"]
        if (np.isfinite(outer_fit["g_shot_e2_per_ph"]) and outer_fit["g_shot_e2_per_ph"] > 0.0 and global_fit["g_shot_e2_per_ph"] == 0.0)
        else global_fit["g_shot_e2_per_ph"]
    )
    if zone_fits and zone_fits[-1]["g_shot_e2_per_ph"] > 0.0 and len(bnds) >= 1:
        outer_pass_g = zone_fits[-1]["g_shot_e2_per_ph"]
    else:
        outer_pass_g = g_shot_best

    alpha_m_est: Optional[float] = None
    if len(np.unique(m_pg[interior])) >= 2 and len(global_fit["shells_s"]) >= 2:
        sh_s_arr = np.asarray(global_fit["shells_s"], dtype=np.float64)
        sh_bg_arr = np.asarray(global_fit["shells_bg_e2"], dtype=np.float64)
        bg_pred = np.interp(np.sqrt(s2[interior]), sh_s_arr, sh_bg_arr)
        sig_base_pred = np.sqrt(
            np.maximum(
                bg_pred ** 2
                + g_shot_best * i_post_e2[interior]
                + (global_fit["c_merged"] * i_post_e2[interior]) ** 2,
                1e-24,
            )
        ) * np.sqrt(float(m_max) / np.maximum(f_ring[interior], 0.22))
        y_m = np.log(sig_base_pred / np.maximum(sig_i_e2[interior], 1e-24))
        x_m = np.log(np.maximum(m_pg[interior].astype(np.float64), 1.0))
        if float(np.std(x_m)) > 0.05:
            slp_m, _ = np.polyfit(x_m, y_m, 1)
            alpha_m_est = round(float(slp_m), 3)

    aniso_info: Dict[str, Any] = {"anisotropy_ratio_max_min": 1.0, "axes_rel_sigma": {}}
    if len(global_fit["shells_s"]) >= 2:
        sh_s_arr = np.asarray(global_fit["shells_s"], dtype=np.float64)
        sh_bg_arr = np.asarray(global_fit["shells_bg_e2"], dtype=np.float64)
        bg_all = np.interp(np.sqrt(s2), sh_s_arr, sh_bg_arr)
        pred_ref = np.sqrt(
            np.maximum(bg_all ** 2 + g_shot_best * i_post_e2 + (global_fit["c_merged"] * i_post_e2) ** 2, 1e-24)
        )
        rel_sig = sig_i_ref / pred_ref

        orth_frac = reciprocal_cartesian_matrix(cell)
        sc_vec = hkls @ orth_frac.T
        sc_norm = sc_vec / np.maximum(np.linalg.norm(sc_vec, axis=1, keepdims=True), 1e-12)

        ax_meds = {}
        for idx_ax, ax_lbl in enumerate(("a*", "b*", "c*")):
            v_ax = orth_frac[:, idx_ax] / max(float(np.linalg.norm(orth_frac[:, idx_ax])), 1e-12)
            m_ax = interior & (np.abs(sc_norm @ v_ax) > 0.82)
            if np.sum(m_ax) >= 20:
                ax_meds[ax_lbl] = round(float(np.median(rel_sig[m_ax])), 3)
        if len(ax_meds) >= 2:
            vals_ax = list(ax_meds.values())
            aniso_info = {
                "anisotropy_ratio_max_min": round(max(vals_ax) / max(min(vals_ax), 1e-4), 3),
                "axes_rel_sigma": ax_meds,
            }

    wedge_from_isym: Optional[float] = None
    paired_ac_frac: Optional[float] = None
    if isym is not None and np.any(np.isfinite(isym)):
        ac_in = interior & (~is_centric) & np.isfinite(isym) & (i_post_e2 > 4.0 * sig_i_e2)
        if np.sum(ac_in) >= 50:
            paired_ac_frac = float(np.mean(np.abs(isym[ac_in]) < 0.25))
            if paired_ac_frac < 0.96:
                m_half_med = max(1.0, float(np.median(m_pg[ac_in])) / 2.0)
                p_mate = (2.0 * paired_ac_frac) / (1.0 + paired_ac_frac)
                w_arc = 1.0 - ((1.0 - min(p_mate, 0.9999)) ** (1.0 / m_half_med))
                wedge_from_isym = round(180.0 * w_arc, 1)

    bg_class: Dict[str, Any] = {"mount_regime": "Insufficient low/mid-angle shells to decompose background ring"}
    if len(global_fit["shells_s"]) >= 5:
        sh_s_bg = np.asarray(global_fit["shells_s"], dtype=np.float64)
        sh_bg2 = np.asarray(global_fit["shells_bg_e2"], dtype=np.float64) ** 2
        sh_d_arr = 0.5 / np.maximum(sh_s_bg, 1e-6)

        if bnds:
            step_corr = np.ones_like(sh_bg2)
            for cp in geom_info["pass_boundaries"]:
                # Undo step jump for shells with d < cp["d_A"] so continuous ring shape is restored
                step_corr = np.where(sh_d_arr < cp["d_A"], step_corr / (cp["matched_bin_ratio"] ** 2), step_corr)
            sh_bg2 = sh_bg2 * step_corr

        b_air = np.interp(sh_s_bg, _STOL_GRID, _F_AIR_GRID) ** 2
        b_air /= float(np.max(b_air))
        b_glass = np.exp(-(((sh_s_bg - 0.1265) / 0.024) ** 2))
        b_wat = np.exp(-(((sh_s_bg - 0.1600) / 0.036) ** 2))
        b_high = 1.0 + 2.5 * (sh_s_bg / max(float(np.max(sh_s_bg)), 0.2)) ** 2

        m_bgf = sh_s_bg <= 0.42
        if np.sum(m_bgf) >= 4:
            y_b = sh_bg2[m_bgf]
            w_b = 1.0 / np.maximum(y_b, 1e-12)
            x_b = np.column_stack([b_air[m_bgf], b_glass[m_bgf], b_wat[m_bgf], b_high[m_bgf]])
            c_bg = exact_nnls(x_b * w_b[:, None], y_b * w_b)

            tot_ring = c_bg[0] + c_bg[1] + c_bg[2] + c_bg[3] + 1e-30
            f_air = float(c_bg[0] / tot_ring)
            f_glass = float(c_bg[1] / tot_ring)
            f_wat = float(c_bg[2] / tot_ring)
            f_base = float(c_bg[3] / tot_ring)

            # Physical peak check for silica glass capillary (~3.9 A = s~0.128): must drop toward 2.0-2.7 A (s~0.19-0.25),
            # not merely rise monotonically into the outer detector edge at d_min ~ 3.4 A!
            m_cap_pk = (sh_d_arr >= 3.45) & (sh_d_arr <= 5.0)
            m_cap_fl = (sh_d_arr >= 1.90) & (sh_d_arr < 2.95)
            has_capillary_peak = False
            if np.any(m_cap_pk) and np.any(m_cap_fl):
                has_capillary_peak = float(np.max(sh_bg2[m_cap_pk])) > 1.35 * float(np.median(sh_bg2[m_cap_fl]))

            ice_d = (3.897, 3.669, 3.441, 2.249)
            ice_ratios = []
            for di in ice_d:
                m_on = np.abs(d - di) <= 0.022
                m_off = (np.abs(d - di) > 0.045) & (np.abs(d - di) <= 0.12)
                if np.sum(m_on) >= 8 and np.sum(m_off) >= 20:
                    ice_ratios.append(float(np.median(sig_i_ref[m_on]) / max(float(np.median(sig_i_ref[m_off])), 1e-12)))
            max_ice_spike = max(ice_ratios) if ice_ratios else 1.0
            has_ice = max_ice_spike > 1.38

            if has_capillary_peak and f_glass > 0.18 and f_glass > 0.45 * f_wat and not has_ice:
                mount_str = "Room-temperature glass capillary + liquid mother liquor (3.8-4.1 A silica + 3.2 A water ring, no ice)"
                is_rt_capillary = True
            elif has_ice:
                mount_str = f"Cryogenic loop with crystalline ice ring contamination (max ice spike {max_ice_spike:.2f}x)"
                is_rt_capillary = False
            elif f_wat > 0.12:
                mount_str = "Loop / thin-film solvent mount (~3.15 A water ring; 100 K cryo or capillary-free RT, no ice)"
                is_rt_capillary = False
            elif f_air < 0.06 and f_wat < 0.08:
                mount_str = "Low-background / helium-or-vacuum path or large crystal dominating solvent/air scatter"
                is_rt_capillary = False
            else:
                mount_str = "Standard air + thin solvent mount"
                is_rt_capillary = False

            bg_class = {
                "mount_regime": mount_str,
                "is_rt_capillary": is_rt_capillary,
                "has_ice_rings": has_ice,
                "max_ice_spike_ratio": round(max_ice_spike, 3),
                "variance_fractions": {
                    "air_low_s": round(f_air, 3),
                    "silica_glass_3p9A": round(f_glass, 3),
                    "water_ring_3p15A": round(f_wat, 3),
                    "compton_obliquity_base": round(f_base, 3),
                },
            }

    # Detect anomalous / non-Poisson SIGF error-model behavior ("weirdsigma"):
    # When SIGF fails to rise as sigma_bg/(2F) on faint reflections, SIGI^2 = 4*F^2*SIGF^2 collapses toward 0 as I -> 0
    # (driving NNLS sigma_bg -> 0 across >=40% of shells while dumping variance into a massive linear-in-I g_shot / SdB*I term!).
    sh_bg_list = global_fit.get("shells_bg_e2", [])
    zero_bg_frac = (
        float(np.mean(np.asarray(sh_bg_list, dtype=np.float64) <= 1e-3))
        if len(sh_bg_list) >= 4
        else 0.0
    )
    anomalous_sigma: Dict[str, Any] = {"detected": False}
    if len(sh_bg_list) >= 4 and (
        (zero_bg_frac >= 0.35 and float(g_shot_best) > 8.0)
        or (global_fit["sigma_bg_lows_e2"] <= 1e-3 and global_fit["sigma_bg_highs_e2"] <= 1e-3 and float(g_shot_best) > 5.0)
    ):
        anomalous_sigma = {
            "detected": True,
            "zero_bg_shell_fraction": round(zero_bg_frac, 3),
            "diagnosis": (
                f"ANOMALOUS / NON-POISSON SIGF MODEL ('weirdsigma'): {int(round(100*zero_bg_frac))}% of shells have "
                f"sigma_bg -> 0 (bg_photons <= 0 collapse) with huge linear-in-I variance (g_shot = {g_shot_best:.1f} e^2/ph, "
                f"c_merged = {100*global_fit['c_merged']:.2f}%). SIGF is flat/increasing with F instead of rising as "
                f"sigma_bg/(2F) on weak reflections — characteristic of heavy SCALA SdB*I / multi-wavelength MAD "
                f"(SOLVE/SCALEIT) error-model inflation or synthetic sigmas; apparent L_xtal and water/air=0 reflect the "
                f"error model rather than photon counting."
            ),
        }

    return {
        "global": global_fit,
        "g_shot_best_e2_per_ph": float(g_shot_best),
        "outer_pass_g_shot_e2_per_ph": float(outer_pass_g),
        "zones": zone_fits,
        "alpha_multiplicity_exp": alpha_m_est,
        "anisotropy": aniso_info,
        "paired_friedel_fraction": paired_ac_frac,
        "wedge_deg_from_isym": wedge_from_isym,
        "background_profile": bg_class,
        "anomalous_sigma_model": anomalous_sigma,
    }


# ==============================================================================
# 8. Module 7: Physical Darwin Invariants, Dose/Size Bounds & nanoBragg2 Inversion
# ==============================================================================
def deduce_physical_parameters_and_replay(
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    m_max: int,
    wilson_info: Dict[str, Any],
    conv_info: Dict[str, Any],
    geom_info: Dict[str, Any],
    var_info: Dict[str, Any],
    cal: dict,
) -> Dict[str, Any]:
    """Invert (g_shot, sigma_bg(s), c_merged, B_form, d_edge) into universal Darwin invariants,
    physical bounds (wavelength, dose, crystal size), nanoBragg2-equivalent experimental parameters,
    and an executable sim_reso_mtz.py forward-replay command."""
    v_cell_a3 = float(cell.volume)
    d_file_min = wilson_info["d_file_min"]
    d_cc12 = wilson_info["d_cc12_030"]
    d_edg_obs = geom_info["inferred_d_edge_A"]
    raw_g_shot = float(var_info["g_shot_best_e2_per_ph"])
    g_shot_resolved = bool(np.isfinite(raw_g_shot) and raw_g_shot > 1e-5)
    g_shot_e2 = max(raw_g_shot, 1e-6)
    g_outer_e2 = max(var_info["outer_pass_g_shot_e2_per_ph"], 1e-6)
    c_merged = var_info["global"]["c_merged"]

    # 1. Diffraction-geometry bounds on X-ray wavelength lambda
    lam_practical_max = round(2.0 * d_file_min * math.sin(math.radians(32.5)), 3)
    lam_backscatter_max = round(2.0 * d_file_min * 0.98, 3)
    cu_ka_ruled_out = bool(lam_practical_max < 1.48)
    lam_nom = min(0.98, 0.85 * lam_practical_max)

    # 2. Universal First-Principles Darwin Invariant with angle-appropriate Lorentz-polarization LP(d, lambda)
    # Note: K_Darwin = 1/g_shot = (M_max / 2pi) * Phi_tot * V_illum * (r_e^2 * lambda^3 / V_cell^2) * <LP> * DQE_eff
    # Evaluate <LP> = (1 + cos^2(2th)) / (2 * sin(2th)) at typical high-angle resolution d_eval ~ 1.15 * d_edg_obs
    d_lp = max(1.12 * d_edg_obs, 0.85)
    sin_th = min(0.92, lam_nom / (2.0 * d_lp))
    two_th = 2.0 * math.asin(sin_th)
    lp_eff = (1.0 + math.cos(two_th) ** 2) / max(2.0 * math.sin(two_th), 0.08)

    r_e_um = 2.81794e-9
    v_cell_um3 = v_cell_a3 * 1e-12
    darwin_prod_um_a3: Optional[float] = (
        (2.0 * math.pi * (v_cell_um3 ** 2))
        / (float(m_max) * (r_e_um ** 2) * 1e-12 * lp_eff * g_shot_e2)
        if g_shot_resolved
        else None
    )

    # 3. Radiation-damage lower bound on illuminated crystal volume V_illum (Holton 2009: 2e10 ph/um^2/MGy at 1 A)
    is_rt = bool(var_info["background_profile"].get("is_rt_capillary", False))
    has_ice = bool(var_info["background_profile"].get("has_ice_rings", False))
    active_temp_regime = (
        "Room-temperature capillary"
        if is_rt
        else ("Cryogenic (<273 K, ice-confirmed)" if has_ice else "Cryogenic (100 K) or capillary-free RT")
    )
    if darwin_prod_um_a3 is not None:
        v_min_cryo_20mgy: Optional[float] = darwin_prod_um_a3 / (2.0e10 * 20.0 * lam_nom)
        v_min_rt_30kgy: Optional[float] = darwin_prod_um_a3 / (2.0e10 * 0.030 * lam_nom)
        v_min_rt_2kgy: Optional[float] = darwin_prod_um_a3 / (2.0e10 * 0.002 * lam_nom)
        l_min_cryo_um: Optional[float] = v_min_cryo_20mgy ** (1.0 / 3.0)
        l_min_rt_30kgy_um: Optional[float] = v_min_rt_30kgy ** (1.0 / 3.0)
        l_min_rt_2kgy_um: Optional[float] = v_min_rt_2kgy ** (1.0 / 3.0)
    else:
        v_min_cryo_20mgy = None
        v_min_rt_30kgy = None
        v_min_rt_2kgy = None
        l_min_cryo_um = None
        l_min_rt_30kgy_um = None
        l_min_rt_2kgy_um = None

    # 4. Exact inversion against nanoBragg2 / sim_reso_mtz reference model
    if geom_info["cutoff_type"] == "detector_corners":
        cal_g = interpolate_calibration_by_dedge(cal, d_edg_obs, criterion="cc12")
    else:
        cal_g = interpolate_calibration(cal, float(np.clip(d_cc12, 0.86, 4.90)), criterion="cc12")

    d_edg_ref = float(cal_g["d_edge"])
    scale_k0_vs_1aho = wilson_info["k0_abs_e2"] / float(cal["K0_ref_1aho"])
    g_ref_abs = float(cal_g["g_shot"]) * scale_k0_vs_1aho
    gamma_cell_phys = ((NB2_REF["cell_vol_A3"] / max(v_cell_a3, 100.0)) ** 2) * scale_k0_vs_1aho

    est_water_um: Optional[float] = None
    est_air_mm: Optional[float] = None
    if g_shot_resolved:
        gamma_g_obs_inferred = g_ref_abs / g_shot_e2
        nb2_photon_product_ratio: Optional[float] = gamma_g_obs_inferred / max(gamma_cell_phys, 1e-12)
        v_equiv_nb2_um3: Optional[float] = NB2_REF["xtal_vol_um3"] * nb2_photon_product_ratio
        l_equiv_nb2_um: Optional[float] = max(0.1, v_equiv_nb2_um3) ** (1.0 / 3.0)

        sh_s = np.asarray(var_info["global"]["shells_s"], dtype=np.float64)
        sh_bg = np.asarray(var_info["global"]["shells_bg_e2"], dtype=np.float64)
        if len(sh_s) >= 4:
            u_sh = (d_edg_obs if geom_info["cutoff_type"] == "detector_corners" else d_edg_ref) * (2.0 * sh_s)
            wat_b = np.exp(-(((sh_s - 0.160) / 0.045) ** 2))
            cor_b = np.maximum(0.0, u_sh - 0.95) ** 2
            sig_bg0_sh = (float(cal_g["sig0"]) * scale_k0_vs_1aho) * np.exp(
                float(cal_g["b1_u2"]) * (u_sh * u_sh)
                + float(cal_g["w_wat"]) * wat_b
                + float(cal_g.get("b2_cor", 0.0)) * cor_b
            )
            bg_var_mult_obs = (sh_bg / np.maximum(sig_bg0_sh, 1e-12)) ** 2

            f_wat2 = np.interp(sh_s, _STOL_GRID, _F_WATER_GRID) ** 2
            f_air2 = np.interp(sh_s, _STOL_GRID, _F_AIR_GRID) ** 2
            a_beam0 = NB2_REF["beam_size_um"] ** 2
            s_air0 = (a_beam0 * (1000.0 * NB2_REF["air_mm"])) * (1.2e-3 / 28.0) * f_air2
            s_wat0 = (a_beam0 * NB2_REF["water_um"]) * (1.0 / 18.0) * f_wat2
            s_xtl0 = NB2_REF["incoh_vol_um3"] * (1.2 / 16.0) * nonbragg_incoherent_f2(
                sh_s, float(cal_g["B_Wilson"]), NB2_REF["wavelength_A"]
            )
            tot0 = s_air0 + s_wat0 + s_xtl0

            a_beam_est = max(a_beam0, l_equiv_nb2_um ** 2)
            gamma_vol_est = v_equiv_nb2_um3 / NB2_REF["xtal_vol_um3"]
            s_xtl_est = (NB2_REF["incoh_vol_um3"] * gamma_vol_est) * (1.2 / 16.0) * nonbragg_incoherent_f2(
                sh_s, wilson_info["B_Wilson_pre_transfer_nb2"], NB2_REF["wavelength_A"]
            )
            lhs = bg_var_mult_obs * (gamma_g_obs_inferred ** 2) * tot0 - s_xtl_est
            m_fit = u_sh <= 0.96
            if geom_info["pass_boundaries"]:
                s_cut_pass = 0.5 / max(float(geom_info["pass_boundaries"][0]["d_A"]), 0.25)
                if int(np.sum(m_fit & (sh_s < s_cut_pass))) >= 3:
                    m_fit = m_fit & (sh_s < s_cut_pass)
                else:
                    m_fit = np.zeros_like(m_fit, dtype=bool)
            if np.sum(m_fit) >= 3:
                col_air = (a_beam_est * 1000.0) * (1.2e-3 / 28.0) * f_air2[m_fit]
                col_wat = a_beam_est * (1.0 / 18.0) * f_wat2[m_fit]
                w_s = 1.0 / np.maximum(tot0[m_fit], 1.0)
                c_aw = exact_nnls(
                    np.column_stack([col_air * w_s, col_wat * w_s]),
                    np.maximum(lhs[m_fit], 0.0) * w_s,
                )
                est_air_mm = round(max(0.0, float(c_aw[0])), 1) + 0.0
                est_water_um = round(max(0.0, float(c_aw[1])), 1) + 0.0
    else:
        nb2_photon_product_ratio = None
        v_equiv_nb2_um3 = None
        l_equiv_nb2_um = None

    b_nb2_pre_transfer = max(2.0, wilson_info["B_Wilson_pre_transfer_nb2"])
    db_xds_rolloff = wilson_info["B_Wilson_apparent_nb2"] - wilson_info["B_Wilson_pre_transfer_nb2"]

    c_sig_ref = max(float(cal_g["c_sig"]), 1e-4)
    f_cal_obs = c_merged / c_sig_ref
    f_prof = 0.12 if geom_info["cutoff_type"] == "detector_corners" else 0.0
    if f_cal_obs * f_cal_obs > f_prof:
        est_calib_err = NB2_REF["calib_err"] * math.sqrt((f_cal_obs * f_cal_obs - f_prof) / (1.0 - f_prof))
    else:
        est_calib_err = 0.0

    # Assemble clean executable sim_reso_mtz.py replay CLI
    cli_parts = ["python sim_reso_mtz.py F0.mtz"]
    if wilson_info["cc12_crossed"]:
        cli_parts.append(f"{d_cc12:.3f}")
    else:
        cli_parts.append(f"{d_file_min:.3f}")
        cli_parts.append(f"--B-wilson {b_nb2_pre_transfer:.2f}")

    if geom_info["cutoff_type"] in ("spherical", "ellipsoidal_anisotropic"):
        cli_parts.append("--cutoff dmin --ideal-transfer")

    t_mode = conv_info.get("truncate_mode", "xdsconv")
    if t_mode in ("xdsconv", "ccp4"):
        cli_parts.append(f"--truncate-mode {t_mode}")
    min_is = conv_info.get("inferred_min_isigi")
    if min_is is not None:
        default_is = -3.0 if t_mode == "xdsconv" else -3.7
        if abs(min_is - default_is) > 0.15:
            cli_parts.append(f"--fw-min-isigi {min_is:.2f}")

    if abs(v_cell_a3 - NB2_REF["cell_vol_A3"]) / NB2_REF["cell_vol_A3"] > 0.03:
        cli_parts.append("--scale-by-cell")

    if l_equiv_nb2_um is not None and abs(l_equiv_nb2_um - 25.99) > 1.5:
        cli_parts.append(f"--xtal-size-um {l_equiv_nb2_um:.2f}")
    if est_water_um is not None and abs(est_water_um - NB2_REF["water_um"]) > 6.0:
        cli_parts.append(f"--water-um {est_water_um:.1f}")
    if est_air_mm is not None and abs(est_air_mm - NB2_REF["air_mm"]) > 5.0:
        cli_parts.append(f"--air-mm {est_air_mm:.1f}")
    if var_info["wedge_deg_from_isym"] is not None:
        cli_parts.append(f"--wedge-deg {var_info['wedge_deg_from_isym']:.1f}")
    if not conv_info.get("sigma_missing_or_zero") and abs(est_calib_err - NB2_REF["calib_err"]) > 0.004:
        cli_parts.append(f"--calib-err {est_calib_err:.4f}")

    return {
        "assumptions": {
            "scale_is_assumed": bool(wilson_info.get("scale_is_assumed", True)),
            "solvent_fraction_used": round(float(wilson_info["f_solvent_est"]), 4),
            "assumed_nominal_wavelength_A": round(float(lam_nom), 4),
            "cc12_estimated_from_sigi_not_halfdatasets": True,
        },
        "g_shot_resolved": bool(g_shot_resolved),
        "darwin_invariant_phi_V_lam3_um_A3": float(darwin_prod_um_a3) if darwin_prod_um_a3 is not None else None,
        "wavelength_bounds_A": {
            "practical_flat_detector_max_2th65": lam_practical_max,
            "physical_backscatter_max_2th180": lam_backscatter_max,
            "cu_kalpha_1p54A_ruled_out": cu_ka_ruled_out,
            "assumed_nominal_wavelength_for_LP_and_dose_A": round(float(lam_nom), 4),
        },
        "crystal_size_lower_bounds": {
            "active_regime": active_temp_regime,
            "min_edge_um_cryo_20MGy": round(l_min_cryo_um, 1) if l_min_cryo_um is not None else None,
            "min_vol_um3_cryo_20MGy": round(v_min_cryo_20mgy, 1) if v_min_cryo_20mgy is not None else None,
            "min_edge_um_RT_30kGy": round(l_min_rt_30kgy_um, 1) if l_min_rt_30kgy_um is not None else None,
            "min_edge_um_RT_2kGy_low_decay": round(l_min_rt_2kgy_um, 1) if l_min_rt_2kgy_um is not None else None,
        },
        "nanobragg2_equivalent": {
            "photon_budget_ratio_vs_nb2": round(nb2_photon_product_ratio, 4) if nb2_photon_product_ratio is not None else None,
            "equiv_xtal_vol_um3_at_nb2_fluence": round(v_equiv_nb2_um3, 1) if v_equiv_nb2_um3 is not None else None,
            "equiv_xtal_edge_um_at_nb2_fluence": round(l_equiv_nb2_um, 2) if l_equiv_nb2_um is not None else None,
            "equiv_exposure_s_at_nb2_xtal": round(NB2_REF["exposure_s"] * nb2_photon_product_ratio, 2) if nb2_photon_product_ratio is not None else None,
            "inferred_B_Wilson_pre_transfer_nb2": round(b_nb2_pre_transfer, 2),
            "estimated_xds_rolloff_dB_A2": round(db_xds_rolloff, 2),
            "inferred_pilatus_dist_mm_at_0p98A": round(dist_from_dedge(d_edg_obs, NB2_REF["wavelength_A"]), 1),
            "inferred_water_um": est_water_um,
            "inferred_air_mm": est_air_mm,
            "inferred_calib_err": round(est_calib_err, 4),
        },
        "suggested_replay_command": " ".join(cli_parts),
    }


# ==============================================================================
# 9. Module 8: Optional Model / FC Audit for Uncalibrated GAIN / SDfac
# ==============================================================================
def audit_sdfac_against_reference(
    cell: gemmi.UnitCell,
    hkls: np.ndarray,
    d: np.ndarray,
    s2: np.ndarray,
    f_obs: Optional[np.ndarray],
    sig_f: Optional[np.ndarray],
    i_hat: np.ndarray,
    sig_i_hat: np.ndarray,
    fc_raw: np.ndarray,
    pass_boundaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Cross-check claimed SIGF/SIGI against a reference amplitude column (FC / F_true / model)
    to uncover uncalibrated detector GAIN / SDfac (invisible from F, SIGF alone!)."""
    valid = np.isfinite(i_hat) & np.isfinite(sig_i_hat) & (sig_i_hat > 0.0) & np.isfinite(fc_raw) & (fc_raw >= 0.0)
    if np.sum(valid) < 80:
        return {"available": False}

    idx_v = np.flatnonzero(valid)
    ord_v = idx_v[np.argsort(s2[idx_v])]
    n_sh = min(28, max(2, len(ord_v) // 75))
    ic_raw = fc_raw * fc_raw

    # Pre-scale radially, then fit 5-parameter traceless anisotropic B_cart so crystal diffraction
    # anisotropy (e.g. 1.7x b* in 1AHO) does not masquerade as coordinate/intensity error!
    ic_pre = np.full(len(d), np.nan, dtype=np.float64)
    for ch in np.array_split(ord_v, n_sh):
        s_o = float(np.sum(np.maximum(i_hat[ch], 0.0)))
        s_c = float(np.sum(ic_raw[ch]))
        if s_o > 0.0 and s_c > 0.0:
            ic_pre[ch] = ic_raw[ch] * (s_o / s_c)

    aniso_mult = np.ones(len(d), dtype=np.float64)
    orth_b = reciprocal_cartesian_matrix(cell)
    sc_v = hkls @ orth_b.T
    sx, sy, sz_c = sc_v[:, 0], sc_v[:, 1], sc_v[:, 2]
    s2_c = sx * sx + sy * sy + sz_c * sz_c
    m_str = valid & np.isfinite(ic_pre) & (ic_pre > 0.0) & (i_hat > 4.0 * sig_i_hat)
    if np.sum(m_str) >= 120:
        a_an = np.column_stack(
            [
                np.ones(int(np.sum(m_str))),
                sx[m_str] ** 2 - s2_c[m_str] / 3.0,
                sy[m_str] ** 2 - s2_c[m_str] / 3.0,
                2.0 * sx[m_str] * sy[m_str],
                2.0 * sx[m_str] * sz_c[m_str],
                2.0 * sy[m_str] * sz_c[m_str],
            ]
        )
        y_an = np.log(np.maximum(i_hat[m_str], 1e-12) / ic_pre[m_str])
        coef_an, _, _, _ = np.linalg.lstsq(a_an, y_an, rcond=None)
        log_m = (
            coef_an[1] * (sx * sx - s2_c / 3.0)
            + coef_an[2] * (sy * sy - s2_c / 3.0)
            + coef_an[3] * (2.0 * sx * sy)
            + coef_an[4] * (2.0 * sx * sz_c)
            + coef_an[5] * (2.0 * sy * sz_c)
        )
        aniso_mult = np.exp(np.clip(log_m, -2.5, 2.5))

    ic_aniso = ic_raw * aniso_mult
    ic_scaled = np.full(len(d), np.nan, dtype=np.float64)
    fc_scaled = np.full(len(d), np.nan, dtype=np.float64)

    for ch in np.array_split(ord_v, n_sh):
        s_o = float(np.sum(np.maximum(i_hat[ch], 0.0)))
        s_c = float(np.sum(ic_aniso[ch]))
        if s_o > 0.0 and s_c > 0.0:
            sc = s_o / s_c
            ic_scaled[ch] = ic_aniso[ch] * sc
            fc_scaled[ch] = np.sqrt(ic_scaled[ch])

    ok = valid & np.isfinite(ic_scaled)

    # Estimate global fractional model/systematic intensity discrepancy c_mod_i on high-SNR shells (d <= 3.0 A)
    m_hi = ok & (d <= max(3.0, float(np.percentile(d[ok], 55))))
    if np.sum(m_hi) < 60:
        m_hi = ok
    q75_g = float(np.percentile(ic_scaled[m_hi], 75))
    m_br_g = m_hi & (ic_scaled >= q75_g) & ((i_hat / sig_i_hat) >= 5.0) & (ic_scaled > 0.0)
    c_mod_i = 0.0
    if np.sum(m_br_g) >= 20:
        rel_i = (i_hat[m_br_g] - ic_scaled[m_br_g]) / ic_scaled[m_br_g]
        tot_i = 1.4826 * float(np.median(np.abs(rel_i - np.median(rel_i))))
        stat_i = float(np.median(sig_i_hat[m_br_g] / ic_scaled[m_br_g]))
        c_mod_i = math.sqrt(max(0.0, tot_i * tot_i - stat_i * stat_i))

    def _zone_sdfac(mz: np.ndarray) -> Dict[str, Any]:
        # Restrict to non-solvent shells (d <= 2.4 A when available in zone) and select lowest (IC_scaled / SIGI_hat)
        # so unmodeled bulk-solvent / coordinate error (c_mod_i * IC / SIGI) is negligible while i_hat/sig_i_hat is uncompressed!
        mz_clean = mz & (d <= 2.4) if int(np.sum(mz & (d <= 2.4))) >= 60 else mz
        idx_z = np.flatnonzero(mz_clean)
        snr_ic = ic_scaled[idx_z] / sig_i_hat[idx_z]
        k_take = max(25, min(len(idx_z), 600, int(0.30 * len(idx_z))))
        sub = idx_z[np.argsort(snr_ic)[:k_take]]
        zi_wk = (i_hat[sub] - ic_scaled[sub]) / sig_i_hat[sub]
        sdf_raw = 1.4826 * float(np.median(np.abs(zi_wk - np.median(zi_wk))))
        mod_ratio = c_mod_i * float(np.median(ic_scaled[sub] / sig_i_hat[sub]))
        sdfac_bg = math.sqrt(max(0.05 * sdf_raw * sdf_raw, sdf_raw * sdf_raw - mod_ratio * mod_ratio))

        z_i_all = (i_hat[mz] - ic_scaled[mz]) / sig_i_hat[mz]
        rms_zi_all = 1.4826 * float(np.median(np.abs(z_i_all - np.median(z_i_all))))

        q75 = float(np.percentile(ic_scaled[mz], 75))
        m_br = mz & (ic_scaled >= q75) & (ic_scaled > 0.0)
        rel_br = (i_hat[m_br] - ic_scaled[m_br]) / np.maximum(ic_scaled[m_br], 1e-12)
        tot_br = 1.4826 * float(np.median(np.abs(rel_br - np.median(rel_br))))
        stat_br = float(np.median((sdfac_bg * sig_i_hat[m_br]) / np.maximum(ic_scaled[m_br], 1e-12)))
        c_sys_est = math.sqrt(max(0.0, tot_br * tot_br - stat_br * stat_br))

        r_f_sh = float("nan")
        if f_obs is not None and np.any(np.isfinite(f_obs[mz])):
            mf = mz & np.isfinite(f_obs) & np.isfinite(fc_scaled)
            sum_fc = float(np.sum(fc_scaled[mf]))
            if sum_fc > 1e-12:
                k_rf = float(np.sum(f_obs[mf])) / sum_fc
                r_f_sh = float(np.sum(np.abs(f_obs[mf] - k_rf * fc_scaled[mf])) / max(float(np.sum(f_obs[mf])), 1e-12))

        return {
            "n": int(np.sum(mz)),
            "true_over_claimed_SDfac_weakFC": round(sdfac_bg, 3),
            "overall_zI_mad_std": round(rms_zi_all, 3),
            "inferred_c_sysI_pct": round(100.0 * c_sys_est, 2),
            "shell_scaled_R_F_pct": round(100.0 * r_f_sh, 2) if np.isfinite(r_f_sh) else None,
        }

    overall = _zone_sdfac(ok)
    by_zone = []
    bnds = [cp["d_A"] for cp in pass_boundaries]
    edges = [float(np.max(d[ok])) + 0.01] + bnds + [float(np.min(d[ok])) - 0.001]
    if len(edges) > 2:
        for j in range(len(edges) - 1):
            mz = ok & (d <= edges[j]) & (d > edges[j + 1])
            if np.sum(mz) >= 40:
                zr = _zone_sdfac(mz)
                zr["d_range_A"] = [round(edges[j], 3), round(max(edges[j + 1], float(np.min(d[ok]))), 3)]
                by_zone.append(zr)
    else:
        d_q25 = float(np.percentile(d[ok], 25))
        m_out = ok & (d <= d_q25)
        if np.sum(m_out) >= 40:
            zr = _zone_sdfac(m_out)
            zr["d_range_A"] = [round(d_q25, 3), round(float(np.min(d[ok])), 3)]
            by_zone.append(zr)

    return {"available": True, "overall": overall, "zones": by_zone}


def compute_fc_from_model(
    model_path: str, cell: gemmi.UnitCell, spacegroup: gemmi.SpaceGroup, hkls: np.ndarray, d_min: float
) -> np.ndarray:
    """Compute |F_calc| from a coordinate model (PDB/mmCIF) for optional SDfac auditing."""
    st = gemmi.read_structure(str(model_path))
    st.cell = cell
    st.spacegroup_hm = spacegroup.xhm()
    d_lim = max(0.45, float(d_min) * 0.96)
    dc = gemmi.DensityCalculatorX()
    dc.d_min = d_lim
    dc.set_grid_cell_and_spacegroup(st)
    dc.put_model_density_on_grid(st[0])
    sf_grid = gemmi.transform_map_to_f_phi(dc.grid, half_l=True)
    hkl_arr = np.ascontiguousarray(hkls, dtype=np.int32)
    f_cpx = sf_grid.get_value_by_hkl(hkl_arr, unblur=float(dc.blur))
    return np.abs(f_cpx).astype(np.float64)


# ==============================================================================
# 10. End-to-End Deduction Orchestrator & Report Formatter
# ==============================================================================
def deduce_dataset_parameters(
    input_path: str,
    cal_file: Optional[str] = None,
    f_col: Optional[str] = None,
    sigf_col: Optional[str] = None,
    i_col: Optional[str] = None,
    sigi_col: Optional[str] = None,
    fc_col: Optional[str] = None,
    model_path: Optional[str] = None,
    n_residues: Optional[int] = None,
    solvent_content: Optional[float] = None,
    f_only: bool = False,
) -> Dict[str, Any]:
    """Run full blind inverse deduction on an MTZ or SF-mmCIF file and return structured dictionary."""
    t0 = time.perf_counter()
    cal_path = Path(cal_file) if cal_file else Path(__file__).with_name("nanobragg2_calibration.json")
    if not cal_path.is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {cal_path}")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))

    raw = load_reflection_file(
        input_path,
        f_col=f_col,
        sigf_col=sigf_col,
        i_col=i_col,
        sigi_col=sigi_col,
        fc_col=fc_col,
        f_only=f_only,
        model_path=model_path,
    )

    d_all = raw["d"]
    pos_d = np.isfinite(d_all) & (d_all > 0.0)
    hkls = raw["hkls"][pos_d]
    d = d_all[pos_d]
    s2 = 1.0 / (4.0 * d * d)

    def _sub(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return arr[pos_d] if arr is not None else None

    f_arr = _sub(raw["F"])
    sf_arr = _sub(raw["SIGF"])
    i_arr = _sub(raw["I"])
    si_arr = _sub(raw["SIGI"])
    isym_arr = _sub(raw["ISYM"])
    fc_arr = _sub(raw["FC"])

    sg = raw["spacegroup"]
    cell = raw["cell"]
    m_pg, m_max, is_centric, epsilon = compute_symmetry_factors(sg, hkls)

    conv = analyze_amplitude_conversion_and_detruncate(
        s2, f_arr, sf_arr, i_arr, si_arr, isym_arr, is_centric, epsilon
    )
    val = conv["valid"]

    # 1. Detect geometry (corners vs spherical, spindle blind cone, pass boundaries) FIRST (100% scale-invariant!)
    geom_raw = detect_geometry_and_pass_boundaries(
        cell,
        sg,
        hkls,
        d,
        hkls[val],
        d[val],
        conv["i_post"][val],
        conv["sig_i_hat"][val],
        m_pg[val],
        m_max,
    )

    # 2. Estimate Debye-Babinet & T_xds-corrected Wilson scale K0_file and B_form
    wilson = estimate_absolute_electron_wilson(
        cell,
        sg,
        s2,
        d,
        conv["i_hat"],
        conv["sig_i_hat"],
        conv["i_post"],
        epsilon,
        val,
        geom_raw,
        cal,
        n_residues_asu=n_residues,
        solvent_content=solvent_content,
    )

    # Convert intensities and uncertainties onto absolute electron^2 scale
    sc_i = wilson["scale_i_file_per_e2"]
    i_post_e2 = conv["i_post"][val] / sc_i
    sig_i_e2 = conv["sig_i_hat"][val] / sc_i

    # Populate max low-angle F in absolute electrons for overload report
    m_low_obs = d[val] >= 3.2
    geom_raw["low_angle_overloads"]["max_observed_F_low_e"] = (
        round(math.sqrt(float(np.max(i_post_e2[m_low_obs]))), 1) if np.any(m_low_obs) else 0.0
    )
    geom_clean = {k: v for k, v in geom_raw.items() if k not in ("asu_d", "asu_present")}
    geom_clean["completeness_d_ge_cc12"] = wilson["completeness_d_ge_cc12"]

    var_res = fit_variance_and_anisotropy(
        cell,
        hkls[val],
        d[val],
        s2[val],
        i_post_e2,
        sig_i_e2,
        m_pg[val],
        m_max,
        is_centric[val],
        isym_arr[val] if isym_arr is not None else None,
        geom_clean,
    )

    phys = deduce_physical_parameters_and_replay(
        cell, sg, m_max, wilson, conv["info"], geom_clean, var_res, cal
    )

    sdfac_audit: Dict[str, Any] = {"available": False}
    is_mislab = bool(conv["info"].get("mislabeled_amplitudes_in_intensity_column"))
    f_for_aud = f_arr if f_arr is not None else (i_arr if is_mislab else None)
    sf_for_aud = sf_arr if sf_arr is not None else (si_arr if is_mislab else None)
    if fc_arr is not None:
        sdfac_audit = audit_sdfac_against_reference(
            cell,
            hkls[val],
            d[val],
            s2[val],
            f_for_aud[val] if f_for_aud is not None else None,
            sf_for_aud[val] if sf_for_aud is not None else None,
            conv["i_hat"][val],
            conv["sig_i_hat"][val],
            fc_arr[val],
            geom_clean["pass_boundaries"],
        )
        sdfac_audit["source"] = f"column:{raw['fc_col']}"
    elif model_path is not None:
        fc_mod = compute_fc_from_model(model_path, cell, sg, hkls[val], float(np.min(d[val])))
        sdfac_audit = audit_sdfac_against_reference(
            cell,
            hkls[val],
            d[val],
            s2[val],
            f_for_aud[val] if f_for_aud is not None else None,
            sf_for_aud[val] if sf_for_aud is not None else None,
            conv["i_hat"][val],
            conv["sig_i_hat"][val],
            fc_mod,
            geom_clean["pass_boundaries"],
        )
        sdfac_audit["source"] = f"model:{model_path}"

    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    return {
        "elapsed_ms": round(elapsed_ms, 1),
        "file_metadata": {
            "path": raw["path"],
            "format": raw["format"],
            "spacegroup": sg.xhm(),
            "laue_max_multiplicity": m_max,
            "cell_parameters": [round(float(x), 3) for x in cell.parameters],
            "cell_volume_A3": round(float(cell.volume), 2),
            "header_wavelength_A_not_used_in_blind_fit": raw["header_wavelength_A"],
            "columns_used": {
                "F": raw["f_col"],
                "SIGF": raw["sigf_col"],
                "I": raw["i_col"],
                "SIGI": raw["sigi_col"],
                "sigma_source": conv["sigma_source"],
            },
            "n_rows_total": int(len(raw["d"])),
            "n_valid_reflections": int(np.sum(val)),
        },
        "amplitude_conversion": conv["info"],
        "wilson_and_resolution": {k: v for k, v in wilson.items() if k != "shells"},
        "geometry_and_passes": geom_clean,
        "variance_and_anisotropy": var_res,
        "physical_deduction": phys,
        "optional_sdfac_audit": sdfac_audit,
    }


def format_deduction_report(res: Dict[str, Any]) -> str:
    """Format a clear human-readable forensic & physical summary report."""
    fm = res["file_metadata"]
    ac = res["amplitude_conversion"]
    wr = res["wilson_and_resolution"]
    gp = res["geometry_and_passes"]
    va = res["variance_and_anisotropy"]
    pd = res["physical_deduction"]
    sa = res["optional_sdfac_audit"]

    compl_hdr = (
        f"{100*gp['overall_completeness']:.1f}%"
        if np.isfinite(gp["overall_completeness"])
        else f"unavailable ({gp.get('lattice_error')})"
    )
    lines = [
        f"=== Inverse Diffraction Parameter Deduction ({res['elapsed_ms']:.1f} ms) ===",
        f"  File / Format     : {fm['path']} ({fm['format']})",
        f"  Space group / Cell: {fm['spacegroup']} (M_max={fm['laue_max_multiplicity']}) | {fm['cell_parameters']} (V={fm['cell_volume_A3']:.1f} A^3)",
        f"  Columns analyzed  : F={fm['columns_used']['F']},{fm['columns_used']['SIGF']} | "
        f"I={fm['columns_used']['I']},{fm['columns_used']['SIGI']} [{fm['columns_used']['sigma_source']}]",
        f"  Reflections       : {fm['n_valid_reflections']} valid / {fm['n_rows_total']} rows "
        f"(d = {wr['d_file_max']:.2f} .. {wr['d_file_min']:.4f} A, compl = {compl_hdr})",
        "",
        "1. Intensity -> Amplitude Conversion & Resolution Cutoff:",
        f"  Software / Cutoff : {ac['software']}",
    ]
    if ac.get("has_F_columns"):
        lines.append(
            f"  F/SIGF floor      : acentric min = {ac['min_fsigf_acentric']:.4f} (p0.5% = {ac['p05_fsigf_acentric']:.3f}), "
            f"centric min = {ac['min_fsigf_centric']:.4f} (n_cen = {ac['n_centric_F']})"
        )
    if ac.get("has_I_columns") and "min_isigi_direct" in ac:
        lines.append(f"  Direct I/SIGI min : {ac['min_isigi_direct']:+.3f} sigma (untruncated IMEAN extremum)")

    cc_note = (
        f"{wr['d_cc12_030']:.3f} A"
        if wr["cc12_crossed"]
        else f"< {wr['d_file_min']:.3f} A (outer est. CC1/2 = {wr['outer_shell_cc12']:.3f}, <I/sigI> = {wr['outer_shell_isig']:.2f})"
    )
    lines.append(
        f"  Resolution limits : file d_min = {wr['d_file_min']:.3f} A | est. CC1/2(SIGI)=0.30 at {cc_note} | "
        f"<I/sigI>=2.0 at {wr['d_isig20']:.3f} A"
    )
    el_lims = gp.get("ellipsoidal_principal_d_limits_A")
    el_str = f", principal limits={el_lims} A" if el_lims else ""
    lines.append(
        f"  Outer geometry    : {gp['cutoff_type']} (inferred isotropic/inscribed d_edge = {gp['inferred_d_edge_A']:.3f} A, "
        f"outer-shell compl = {100*gp['outer_shell_completeness']:.1f}%{el_str})"
    )

    lines.extend(
        [
            "",
            "2. Absolute Electron Scaling & Wilson B-Factor:",
            f"  Scale basis       : {wr['scale_basis']} -> amplitude file/e^- scale k_file = {wr['k_amp_file_per_e']:.4f}x",
            f"  Crystal Wilson B  : B_form = {wr['B_form_pre_transfer']:.2f} A^2 (apparent = {wr['B_form_apparent']:.2f} A^2) | "
            f"nanoBragg2-gauge = {wr['B_Wilson_pre_transfer_nb2']:.2f} A^2 (apparent = {wr['B_Wilson_apparent_nb2']:.2f} A^2)",
        ]
    )

    lines.append("")
    lines.append("3. Multi-Pass Merging & 3D Reciprocal-Space Forensics:")
    if gp["pass_boundaries"]:
        lines.append(f"  Merged passes     : MULTI-PASS / MULTI-CRYSTAL MERGE DETECTED ({len(gp['pass_boundaries']) + 1} regimes):")
        for cp in gp["pass_boundaries"]:
            lines.append(
                f"    * d = {cp['d_A']:.3f} A : SIGI jumps x{cp['matched_bin_ratio']:.3f} toward high-res "
                f"(z = {cp['z_score']:.1f}) -> {cp['event']}"
            )
    else:
        lines.append("  Merged passes     : Single continuous pass (0 step discontinuities in SIGI vs d)")

    sp = gp["spindle_blind_cone"]
    if sp["detected"]:
        lines.append(f"  Spindle / mount   : Lorentz blind cone detected -> {sp['axis']}")
    else:
        ax_c = ", ".join(f"{k}:{100*v['completeness']:.1f}%" for k, v in sp.get("axis_completeness", {}).items())
        lines.append(f"  Spindle / mount   : No missing blind cone ({ax_c or 'isotropic completeness'})")

    an = va["anisotropy"]
    if "anisotropy_ratio_max_min" in an:
        ax_s = ", ".join(f"{k}:{v:.2f}x" for k, v in an.get("axes_rel_sigma", {}).items())
        m_exp = f", M_pg exp = {va['alpha_multiplicity_exp']:.2f} [ideal 360d = 0.50]" if va["alpha_multiplicity_exp"] is not None else ""
        w_isym = f", ISYM wedge ~ {va['wedge_deg_from_isym']:.0f} deg" if va["wedge_deg_from_isym"] is not None else ""
        lines.append(f"  SIGF anisotropy   : max/min = {an['anisotropy_ratio_max_min']:.2f}x ({ax_s}){m_exp}{w_isym}")
    else:
        lines.append("  SIGF anisotropy   : unavailable (insufficient directional sigma coverage or sigma-less deposition)")

    ov = gp["low_angle_overloads"]
    if ov.get("detected"):
        lines.append(
            f"  Low-d saturation  : DETECTED ({ov['n_missing_low_angle_noncone']}/{ov['n_possible_low_angle_noncone']} "
            f"[{ov['low_angle_missing_pct']:.1f}%] non-cone d>3.2A missing; max F_low = {ov['max_observed_F_low_e']:.0f} e^- -> 16-bit overload signature)"
        )
    else:
        lines.append("  Low-d saturation  : None detected (wide dynamic range / unsaturated low-angle pass)")

    gf = va["global"]
    bg_p = va["background_profile"]
    lines.extend(
        [
            "",
            "4. Intensity Variance Decomposition [SIGI_ref^2 = sigma_bg^2(d) + g_shot*I + (c_merged*I)^2, on e^2 scale]:",
            f"  Global (M={fm['laue_max_multiplicity']} ref) : med sigma_bg = {gf['sigma_bg_med_e2']:.1f} e^2 "
            f"(low-s {gf['sigma_bg_lows_e2']:.1f} -> high-s {gf['sigma_bg_highs_e2']:.1f}), "
            f"g_shot = {va['g_shot_best_e2_per_ph']:.3f} e^2/ph, c_merged = {100*gf['c_merged']:.2f}% "
            f"(SIGF/F floor = {50*gf['c_merged']:.2f}%), rms_ln_si = {gf['rms_ln_si']:.3f}",
        ]
    )
    for zf in va["zones"]:
        dr = zf["d_range_A"]
        lines.append(
            f"    Zone {dr[0]:5.2f}-{dr[1]:5.3f}A : n={zf['n']:5d} | med sigma_bg = {zf['sigma_bg_med_e2']:7.1f} e^2 | "
            f"g_shot = {zf['g_shot_e2_per_ph']:6.2f} e^2/ph | c_merged = {100*zf['c_merged']:5.2f}%"
        )
    if va.get("anomalous_sigma_model", {}).get("detected"):
        lines.append(f"  SIGF model check  : {va['anomalous_sigma_model']['diagnosis']}")
    lines.append(f"  Background mount  : {bg_p['mount_regime']}")
    if "variance_fractions" in bg_p:
        vf_d = bg_p["variance_fractions"]
        lines.append(
            f"    Ring fractions  : air={100*vf_d['air_low_s']:.0f}%, silica(3.9A)={100*vf_d['silica_glass_3p9A']:.0f}%, "
            f"water(3.15A)={100*vf_d['water_ring_3p15A']:.0f}%, Compton/base={100*vf_d['compton_obliquity_base']:.0f}%"
        )

    wb = pd["wavelength_bounds_A"]
    cs = pd["crystal_size_lower_bounds"]
    nb = pd["nanobragg2_equivalent"]
    assump = pd.get("assumptions", {})
    lam_assump = assump.get("assumed_nominal_wavelength_A", 0.98)
    sc_assump_tag = (
        f" [uses assumed solvent={100*assump.get('solvent_fraction_used', 0.47):.0f}%]"
        if assump.get("scale_is_assumed", False)
        else ""
    )
    if wb["physical_backscatter_max_2th180"] < 1.535:
        cu_str = " [1.54 A Cu K-alpha PHYSICALLY IMPOSSIBLE at any 2th -> synchrotron/Mo]"
    elif wb["cu_kalpha_1p54A_ruled_out"]:
        cu_str = " [un-swung 2th<=65d Cu K-alpha ruled out -> synchrotron/Mo or high-2th swing arm]"
    else:
        cu_str = ""
    hdr_wl = f" (file header wavelength = {fm['header_wavelength_A_not_used_in_blind_fit']:.4f} A, unused)" if fm["header_wavelength_A_not_used_in_blind_fit"] else ""
    lines.append("")
    lines.append("5. Deduceable Physical Invariants, Inequalities & nanoBragg2-Equivalent Setup:")
    if pd["darwin_invariant_phi_V_lam3_um_A3"] is not None:
        lines.append(
            f"  Darwin invariant  : Phi_tot * V_illum * lambda^3 * DQE = {pd['darwin_invariant_phi_V_lam3_um_A3']:.3e} ph * um * A^3 "
            f"({nb['photon_budget_ratio_vs_nb2']:.3f}x nanoBragg2 baseline; LP at assumed lambda~{lam_assump:.2f}A{sc_assump_tag})"
        )
    else:
        lines.append(
            "  Darwin invariant  : unresolved (shot-noise slope g_shot <= 0; errors dominated by flat background, "
            "serial hit-multiplicity, or missing sigmas)"
        )
    lines.append(
        f"  Wavelength bound  : lambda <= {wb['practical_flat_detector_max_2th65']:.3f} A (flat det 2th<=65d), "
        f"< {wb['physical_backscatter_max_2th180']:.3f} A (2th<180d){cu_str}{hdr_wl}"
    )
    if cs["min_edge_um_cryo_20MGy"] is not None:
        lines.append(
            f"  Min crystal size  : >= {cs['min_edge_um_cryo_20MGy']:.1f} um at 100K cryo (<=20 MGy) | "
            f">= {cs['min_edge_um_RT_30kGy']:.0f}-{cs['min_edge_um_RT_2kGy_low_decay']:.0f} um at RT capillary (30-2 kGy) "
            f"[active: {cs['active_regime']}; assumes lambda~{lam_assump:.2f}A{sc_assump_tag}]"
        )
        lines.append(
            f"  nanoBragg2 equiv  : L_xtal = {nb['equiv_xtal_edge_um_at_nb2_fluence']:.2f} um (V = {nb['equiv_xtal_vol_um3_at_nb2_fluence']:.0f} um^3) "
            f"at 4e10 ph/um^2 [or t_exp = {nb['equiv_exposure_s_at_nb2_xtal']:.2f} s at 26 um xtal], "
            f"water ~ {nb['inferred_water_um']} um, air ~ {nb['inferred_air_mm']} mm, calib_err ~ {100*nb['inferred_calib_err']:.2f}%{sc_assump_tag}"
        )
    else:
        lines.append("  Min crystal size  : unresolved (requires positive Poisson shot-noise slope g_shot > 0)")
        lines.append(f"  nanoBragg2 equiv  : crystal size / water / air unresolved (g_shot <= 0), calib_err ~ {100*nb['inferred_calib_err']:.2f}%")
    lines.append(f"  Replay CLI        : {pd['suggested_replay_command']}")

    if sa.get("available"):
        ov_s = sa["overall"]
        lines.extend(
            [
                "",
                f"6. Optional Reference Audit ({sa['source']}) — Uncalibrated GAIN / SDfac Check:",
                f"  Overall           : true/claimed SDfac (weak FC) = {ov_s['true_over_claimed_SDfac_weakFC']:.3f}x, "
                f"zI MAD-std = {ov_s['overall_zI_mad_std']:.3f}, c_sysI = {ov_s['inferred_c_sysI_pct']:.2f}%, "
                f"20-shell R_F = {ov_s['shell_scaled_R_F_pct']}%",
            ]
        )
        for zr in sa.get("zones", []):
            dr = zr["d_range_A"]
            lines.append(
                f"    Zone {dr[0]:5.2f}-{dr[1]:5.3f}A : n={zr['n']:5d} | true/claimed SDfac = {zr['true_over_claimed_SDfac_weakFC']:.3f}x | "
                f"c_sysI = {zr['inferred_c_sysI_pct']:5.2f}% | shell R_F = {zr['shell_scaled_R_F_pct']}%"
            )

    return "\n".join(lines)


def build_deduce_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Blindly deduce experimental data-collection & processing parameters from (H, K, L, F, SIGF)."
    )
    p.add_argument("input_file", help="Input MTZ or SF-mmCIF (.cif / .ent) reflection file.")
    p.add_argument("--f-col", default=None, help="Amplitude column label (auto-detected if omitted).")
    p.add_argument("--sigf-col", default=None, help="Amplitude sigma column label (auto-detected if omitted).")
    p.add_argument("--i-col", default=None, help="Intensity column label (auto-detected if omitted).")
    p.add_argument("--sigi-col", default=None, help="Intensity sigma column label (auto-detected if omitted).")
    p.add_argument("--f-only", action="store_true", help="Ignore IMEAN/SIGIMEAN columns even if present (test pure F/SIGF inversion).")
    p.add_argument("--n-residues", type=int, default=None, help="Residues per asymmetric unit for exact absolute electron Wilson scaling.")
    p.add_argument("--solvent-content", type=float, default=None, help="Solvent fraction (0..1) if --n-residues omitted (default: 0.47).")
    p.add_argument("--fc-col", default=None, help="Optional reference/model amplitude column (e.g. FC or F_true) to audit GAIN/SDfac.")
    p.add_argument("--model", default=None, help="Optional PDB/mmCIF coordinate model to audit GAIN/SDfac.")
    p.add_argument("--cal-file", default=None, help="Path to custom nanobragg2_calibration.json.")
    p.add_argument("--json", default=None, help="Write full structured deduction dictionary to JSON file.")
    p.add_argument("--quiet", action="store_true", help="Suppress human-readable report.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_deduce_arg_parser().parse_args(argv)
    res = deduce_dataset_parameters(
        input_path=args.input_file,
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
    if args.json:
        out_p = Path(args.json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not args.quiet:
        print(format_deduction_report(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

