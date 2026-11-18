#!/usr/bin/env python3
"""
build_calibration_v2.py — Extract complete v2.2 empirical calibration strictly from training
seeds .01-.05 (575 datasets across 115 B_Wilson values; seeds .06-.10 held out for validation)
so sim_reso_mtz.py reproduces:
  1. Exact XDSCONV (CCP4_I, WILSON_STATISTICS=TRUE) clamped sliding-window French-Wilson & F+/F- merging.
  2. Exact deterministic 18-knot XDS spot-profile transfer curve T_xds(u) vs u = d_edge / d,
      plus high-u within-shell profile/summation intensity capture non-linearity delta_nl_u1.
  3. Three-knot u-dependent random scatter (a_stat(u), c_sysI(u)), low-angle shell-intensity bleed
     floor c_wbg, and anomalous mate scatter (a_ind, c_sys_ind) fitted in the exact 20-shell u_law frame.
  4. Exact SIGIMEAN parameters calibrated in the T_xds(u)-attenuated signal frame with conditional
     corner multiplicity floor (<I/sigI> & CC @ d_min).
  5. Exact d_mtz_min, low-res beamstop completeness, Voronoi detector completeness p_obs(u), and
     outer-corner background offset beta_bg(u) + per-seed radial-ring fluctuation sigma_bg_cor.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import gemmi
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError as exc:
    raise ImportError(
        "scipy is required to rebuild nanobragg2_calibration.json; install with `pip install \"sigf[calibration]\"` or `pip install scipy`."
    ) from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_reso_mtz as sm

DEFAULT_NB2 = Path(os.environ.get("NANOBRAGG2_DIR", "/home/jamesh/projects/resolution/challenge/1aho_Bfac/nanoBragg2"))

U_T_KNOTS = [
    0.0, 0.16, 0.26, 0.35, 0.44, 0.52, 0.60, 0.67, 0.74,
    0.80, 0.86, 0.91, 0.96, 1.01, 1.07, 1.14, 1.22, 1.34,
]
U_P_KNOTS = [0.0, 0.20, 0.55, 0.82, 0.94, 0.99, 1.02, 1.05, 1.09, 1.14, 1.22, 1.35]
U_BG_KNOTS = [
    0.0, 0.92, 0.96, 0.99, 1.02, 1.05, 1.08, 1.11,
    1.14, 1.17, 1.20, 1.23, 1.26, 1.29, 1.32, 1.36,
]
U_SC_KNOTS = [0.30, 0.68, 0.94]


def hkl_key(h):
    h = np.rint(h).astype(np.int64) + 300
    return (h[:, 0] * 600 + h[:, 1]) * 600 + h[:, 2]


def hat_weights_3(u_arr, knots=U_SC_KNOTS):
    u0, u1, u2 = knots
    w0 = np.where(u_arr <= u0, 1.0, np.where(u_arr < u1, (u1 - u_arr) / (u1 - u0), 0.0))
    w2 = np.where(u_arr >= u2, 1.0, np.where(u_arr > u1, (u_arr - u1) / (u2 - u1), 0.0))
    w1 = np.maximum(0.0, 1.0 - w0 - w2)
    return w0, w1, w2


def nl_ramp(u_arr, bw, dnl):
    al = float(np.clip((bw - 24.0) / 5.0, 0.0, 1.0))
    u_lo = (1.0 - al) * 0.44 + al * 0.56
    u_hi = (1.0 - al) * 0.64 + al * 0.72
    return np.interp(u_arr, [u_lo, u_hi], [0.0, dnl])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild nanobragg2_calibration.json from training seeds .01-.05.")
    ap.add_argument("--nb2-dir", type=Path, default=DEFAULT_NB2, help="Directory containing FrightB0.mtz, reso_summary_10x.txt, and proc*/FreeRed.mtz.")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "nanobragg2_calibration.json", help="Output calibration JSON path.")
    args = ap.parse_args(argv)
    NB2 = args.nb2_dir
    if not (NB2 / "FrightB0.mtz").is_file():
        raise FileNotFoundError(f"nanoBragg2 directory missing FrightB0.mtz: {NB2} (pass --nb2-dir <path>)")

    t0 = time.perf_counter()
    b0 = gemmi.read_mtz_file(str(NB2 / "FrightB0.mtz"))
    a0_gt = np.array(b0, copy=False)
    hk0 = np.rint(a0_gt[:, :3]).astype(np.int32)
    d0 = np.asarray(b0.make_d_array(), dtype=np.float64)
    s2_0 = 1.0 / (4.0 * d0 * d0)
    fp0_all = a0_gt[:, 3].astype(np.float64)
    fm0_all = a0_gt[:, 4].astype(np.float64)
    f0_all = np.sqrt(0.5 * (fp0_all ** 2 + fm0_all ** 2))
    i0_all = f0_all ** 2
    m_pg0, m_max, is_cen0, eps0 = sm.compute_symmetry_factors(b0.spacegroup, hk0)

    k0_ref, b0_form_ref, _ = sm.estimate_f0_wilson_scale(s2_0, i0_all, eps0, 5.758475, 18.2)
    f_p2_0 = float(sm.f_prot_sq(np.array(0.0)))
    print(f"Reference FrightB0: K0_ref={k0_ref:.2f}, B0_form={b0_form_ref:.5f}")

    keys0 = hkl_key(hk0)
    ord0 = np.argsort(keys0)
    skeys0 = keys0[ord0]
    d0_sorted = np.sort(d0)[::-1]
    n_gt_all = len(d0_sorted)

    rs_by_base = {}
    n_train_rows = 0
    with open(NB2 / "reso_summary_10x.txt") as fh:
        for line in fh:
            w = line.split()
            if len(w) < 22:
                continue
            base, seed_str = w[0].split(".")
            if int(seed_str) > 5:
                continue  # Strictly hold out seeds .06-.10 for out-of-sample evaluation
            vals = (
                w[0],
                float(w[1]),
                float(w[2].rstrip("A")) if w[2] != "-" else np.nan,
                float(w[3].rstrip("A")) if w[3] != "-" else np.nan,
                float(w[4].rstrip("A")) if w[4] != "-" else np.nan,
                float(w[7]) if w[7] != "-" else np.nan,
                float(w[8]) if w[8] != "-" else np.nan,
                float(w[11]),  # d_corner
                float(w[12]),  # d_edge
            )
            rs_by_base.setdefault(base, []).append(vals)
            n_train_rows += 1

    print(
        f"\nExtracting Training-Split (.01-.05, {n_train_rows} datasets) v2.2 Calibration "
        f"Across {len(rs_by_base)} B_Wilson Groups..."
    )
    cal_rows = []
    for idx_b, (base, rlist) in enumerate(sorted(rs_by_base.items(), key=lambda kv: kv[1][0][1])):
        Bw = rlist[0][1]
        ex_b = Bw - 18.2
        d_cc12 = float(np.nanmean([x[2] for x in rlist]))
        d_is15 = float(np.nanmean([x[3] for x in rlist]))
        d_is20 = float(np.nanmean([x[4] for x in rlist]))
        d_ccr = float(np.nanmean([x[5] for x in rlist]))
        d_mrm = float(np.nanmean([x[6] for x in rlist]))
        d_cor_geom = float(np.nanmean([x[7] for x in rlist]))
        d_edg = float(np.nanmean([x[8] for x in rlist]))

        n_rows_list = []
        all_u = []
        all_s2 = []
        all_mp = []
        all_itr_nom = []
        all_im_lowscaled = []
        all_sim_lowscaled = []

        seed_vii_list = []
        ano_z_diff = []

        obs_u_all = []
        obs_flag_all = []
        beamstop_miss_d12 = []

        for proc_id, _, dcc_s, _, _, _, _, _, _ in rlist:
            mtz_p = NB2 / f"proc{proc_id}" / "FreeRed.mtz"
            if not mtz_p.exists():
                continue
            mtz = gemmi.read_mtz_file(str(mtz_p))
            arr = np.array(mtz, copy=False)
            c = {col.label: col.idx for col in mtz.columns}
            k = hkl_key(arr[:, :3])
            pos = np.clip(np.searchsorted(skeys0, k), 0, len(skeys0) - 1)
            ok = skeys0[pos] == k
            arr = arr[ok]
            gi = ord0[pos[ok]]

            d = d0[gi]
            s2 = s2_0[gi]
            mp = m_pg0[gi]
            ic = is_cen0[gi]
            n_rows_list.append(int(len(d)))

            im = arr[:, c["IMEAN"]].astype(np.float64)
            sim = arr[:, c["SIGIMEAN"]].astype(np.float64)
            ip = arr[:, c["Iplus"]].astype(np.float64)
            sip = arr[:, c["SIGIplus"]].astype(np.float64)
            im_m = arr[:, c["Iminus"]].astype(np.float64)
            sim_m = arr[:, c["SIGIminus"]].astype(np.float64)

            vi = np.isfinite(im) & np.isfinite(sim) & (sim > 0)
            if vi.sum() < 20:
                continue

            # Non-overlapping completeness split: d > 12A (beamstop) vs d <= 12A (p_obs(u))
            d_cut_s = float(d.min())
            gt_in_cut = d0 >= (d_cut_s - 1e-7)
            obs_keys = set(keys0[gi[vi]].tolist())
            gt_sub_keys = keys0[gt_in_cut]
            gt_sub_d = d0[gt_in_cut]
            gt_obs_flags = np.fromiter(
                (1.0 if kk in obs_keys else 0.0 for kk in gt_sub_keys),
                dtype=np.float64,
                count=len(gt_sub_keys),
            )
            low12 = gt_sub_d > 12.0
            if np.any(low12):
                beamstop_miss_d12.append(1.0 - float(np.mean(gt_obs_flags[low12])))
            hi12 = ~low12
            if np.any(hi12):
                obs_u_all.append(d_edg / gt_sub_d[hi12])
                obs_flag_all.append(gt_obs_flags[hi12])

            eb2 = np.exp(-2.0 * ex_b * s2)
            ip_t = (fp0_all[gi] ** 2) * eb2
            im_t = (fm0_all[gi] ** 2) * eb2
            i_t = 0.5 * (ip_t + im_t)

            d_lim = dcc_s if np.isfinite(dcc_s) else d_cc12
            vii = vi & (d >= d_lim)
            if vii.sum() < 20:
                vii = vi
            idx_vii = np.flatnonzero(vii)
            ord_vii = idx_vii[np.argsort(s2[idx_vii])]
            low25 = ord_vii[: max(12, len(ord_vii) // 4)]
            k0_low = float(np.sum(i_t[low25]) / np.sum(im[low25]))

            im_ls = im * k0_low
            sim_ls = sim * k0_low

            all_u.append(d_edg / d[vi])
            all_s2.append(s2[vi])
            all_mp.append(mp[vi])
            all_itr_nom.append(i_t[vi])
            all_im_lowscaled.append(im_ls[vi])
            all_sim_lowscaled.append(sim_ls[vi])

            n_sh = min(20, max(2, len(ord_vii) // 15))
            loc = np.full(len(d), k0_low, dtype=np.float64)
            for ch in np.array_split(ord_vii, n_sh):
                so, st = np.sum(im[ch]), np.sum(i_t[ch])
                if so > 0 and st > 0:
                    loc[ch] = st / so

            seed_vii_list.append(
                (d_edg / d[vii], s2[vii], mp[vii], eps0[gi[vii]], i_t[vii], im[vii], sim[vii], k0_low)
            )

            ac_v = vii & (~ic) & np.isfinite(ip) & np.isfinite(im_m) & (sip > 0) & (sim_m > 0)
            if np.any(ac_v):
                diff_obs = (ip[ac_v] - im_m[ac_v]) * loc[ac_v]
                diff_tru = ip_t[ac_v] - im_t[ac_v]
                si_diff = np.sqrt(sip[ac_v] ** 2 + sim_m[ac_v] ** 2) * loc[ac_v]
                ano_z_diff.append((diff_obs - diff_tru, si_diff, i_t[ac_v]))

        # Exact midpoint d_mtz_min between reflection idx_r and idx_r+1 of FrightB0.mtz (9-decimal precision)
        mean_rows = float(np.mean(n_rows_list))
        idx_r = min(max(int(round(mean_rows)) - 1, 0), n_gt_all - 1)
        if idx_r + 1 < n_gt_all:
            d_mtz_min = 0.5 * float(d0_sorted[idx_r] + d0_sorted[idx_r + 1])
        else:
            d_mtz_min = float(d0_sorted[-1] - 1e-5)
        p_miss_low12 = float(np.mean(beamstop_miss_d12)) if beamstop_miss_d12 else 0.0

        u_cat = np.concatenate(all_u)
        s2_cat = np.concatenate(all_s2)
        mp_cat = np.concatenate(all_mp)
        itr_cat = np.concatenate(all_itr_nom)
        im_cat = np.concatenate(all_im_lowscaled)
        sim_cat = np.concatenate(all_sim_lowscaled)

        # 1. Fit Voronoi-binned detector completeness curve p_obs(u) on d <= 12A
        ou_cat = np.concatenate(obs_u_all)
        of_cat = np.concatenate(obs_flag_all)
        p_knots = []
        for k_i, uk in enumerate(U_P_KNOTS):
            b_lo = 0.5 * (U_P_KNOTS[k_i - 1] + uk) if k_i > 0 else -0.01
            b_hi = 0.5 * (uk + U_P_KNOTS[k_i + 1]) if k_i + 1 < len(U_P_KNOTS) else (uk + 0.08)
            m_u = (ou_cat >= b_lo) & (ou_cat < b_hi)
            if np.sum(m_u) >= 8:
                val = float(np.mean(of_cat[m_u]))
                if uk > 1.03 and p_knots:
                    val = min(val, p_knots[-1])
                p_knots.append(round(val, 5))
            else:
                p_knots.append(p_knots[-1] if p_knots else 1.0)

        # 2. Fit 18-knot XDS transfer curve T_xds(u) preserving full low-u slope, fine 16-knot outer-corner
        #    background offset beta_bg(u), and per-seed radial-ring background SD sigma_bg_cor.
        u_lim_b = (d_edg / max(d_cc12, 0.5)) + 1e-6
        u_t_cut = u_lim_b + 0.015

        t_raw = []
        pop_knots = []
        for k_i, uk in enumerate(U_T_KNOTS):
            b_lo = 0.5 * (U_T_KNOTS[k_i - 1] + uk) if k_i > 0 else -0.01
            b_hi = 0.5 * (uk + U_T_KNOTS[k_i + 1]) if k_i + 1 < len(U_T_KNOTS) else 1.60
            m_u = (u_cat >= b_lo) & (u_cat < b_hi)
            if uk <= u_t_cut:
                if np.sum(m_u) >= 8 and np.sum(itr_cat[m_u]) > 1e-6 and np.sum(im_cat[m_u]) > 0.0:
                    val = float(np.sum(im_cat[m_u]) / np.sum(itr_cat[m_u]))
                    t_raw.append(val)
                    pop_knots.append((uk, val))
                else:
                    t_raw.append(np.nan)
            else:
                t_prev = next((x for x in reversed(t_raw) if np.isfinite(x)), 0.50)
                if np.sum(m_u) >= 8 and np.sum(itr_cat[m_u]) > 1e-6:
                    raw_rat = float(np.sum(im_cat[m_u]) / np.sum(itr_cat[m_u]))
                    val = float(np.clip(raw_rat, 0.02, t_prev))
                else:
                    val = max(0.02, 0.82 * t_prev)
                t_raw.append(val)

        for k_i in range(len(U_T_KNOTS)):
            if not np.isfinite(t_raw[k_i]):
                if k_i > 0 and np.isfinite(t_raw[k_i - 1]):
                    t_raw[k_i] = t_raw[k_i - 1]
                elif pop_knots:
                    t_raw[k_i] = pop_knots[0][1]
                else:
                    t_raw[k_i] = 1.0

        t_plat_vals = [t_raw[j] for j, uj in enumerate(U_T_KNOTS) if 0.25 <= uj <= 0.55 and np.isfinite(t_raw[j])]
        t_plat = float(np.max(t_plat_vals)) if t_plat_vals else 1.0
        for k_i, uk in enumerate(U_T_KNOTS):
            if uk < 0.36:
                t_raw[k_i] = max(t_raw[k_i], 0.93 * t_plat)
            if uk > 1.01 and k_i > 0:
                t_raw[k_i] = min(t_raw[k_i], t_raw[k_i - 1])

        t0_scale = max(float(t_raw[0]), 1e-6)
        t_knots = [round(float(v / t0_scale), 5) for v in t_raw]
        im_cat = im_cat / t0_scale
        sim_cat = sim_cat / t0_scale
        i_xds_true = itr_cat * np.interp(u_cat, U_T_KNOTS, t_knots)

        bg_knots = []
        for k_i, uk in enumerate(U_BG_KNOTS):
            if uk <= u_lim_b:
                bg_knots.append(0.0)
                continue
            b_lo = 0.5 * (U_BG_KNOTS[k_i - 1] + uk)
            b_hi = 0.5 * (uk + U_BG_KNOTS[k_i + 1]) if k_i + 1 < len(U_BG_KNOTS) else 1.60
            m_u = (u_cat >= b_lo) & (u_cat < b_hi)
            if np.sum(m_u) >= 5 and np.sum(sim_cat[m_u]) > 1e-6:
                bg_val = float(np.clip(np.sum(im_cat[m_u] - i_xds_true[m_u]) / np.sum(sim_cat[m_u]), -0.85, 0.45))
                bg_knots.append(round(bg_val, 4))
            else:
                bg_knots.append(bg_knots[-1] if bg_knots else 0.0)

        ring_edges = np.arange(max(u_lim_b + 0.06, 1.04), 1.36, 0.07)
        ring_vars = []
        for j_r in range(len(ring_edges) - 1):
            r_lo, r_hi = ring_edges[j_r], ring_edges[j_r + 1]
            seed_ms, seed_invn = [], []
            for u_s, itr_s, im_s_ls, sim_s_ls in zip(all_u, all_itr_nom, all_im_lowscaled, all_sim_lowscaled):
                mr_s = (u_s >= r_lo) & (u_s < r_hi)
                ns_r = int(np.sum(mr_s))
                if ns_r >= 8:
                    im_sc = im_s_ls[mr_s] / t0_scale
                    sim_sc = sim_s_ls[mr_s] / t0_scale
                    ix_sc = itr_s[mr_s] * np.interp(u_s[mr_s], U_T_KNOTS, t_knots)
                    if np.sum(sim_sc) > 1e-6:
                        seed_ms.append(float(np.sum(im_sc - ix_sc) / np.sum(sim_sc)))
                        seed_invn.append(1.0 / ns_r)
            if len(seed_ms) >= 3:
                v_emp = float(np.var(seed_ms, ddof=1)) - float(np.mean(seed_invn))
                ring_vars.append(max(0.0, v_emp))
        sigma_bg_cor = float(np.clip(math.sqrt(float(np.median(ring_vars))), 0.05, 0.42)) if ring_vars else 0.24

        # 3. Fit SIGIMEAN model directly in the i_xds_true frame (conditional corner f_ring floor 0.22)
        f_ring = np.where(
            u_cat <= 1.0,
            1.0,
            np.maximum(1.0 - (4.0 / math.pi) * np.arccos(np.clip(1.0 / u_cat, -1.0, 1.0)), 0.22),
        )
        si_c = sim_cat * np.sqrt((mp_cat / 8.0) * f_ring)
        u2 = u_cat * u_cat
        cor_bump = np.maximum(0.0, u_cat - 0.95) ** 2
        s_mag = np.sqrt(s2_cat)
        wat_bump = np.exp(-(((s_mag - 0.160) / 0.045) ** 2))

        if len(si_c) > 25000:
            step = len(si_c) // 25000
            sl_idx = slice(None, None, step)
        else:
            sl_idx = slice(None)
        si_sub = si_c[sl_idx]
        u2_sub = u2[sl_idx]
        cb_sub = cor_bump[sl_idx]
        wb_sub = wat_bump[sl_idx]
        ix_sub = i_xds_true[sl_idx]

        def res_sig(p):
            bg = np.exp(p[0] + p[1] * u2_sub + p[2] * wb_sub + p[5] * cb_sub)
            var = bg * bg + math.exp(p[3]) * ix_sub + (math.exp(p[4]) * ix_sub) ** 2
            return np.log(si_sub) - 0.5 * np.log(np.maximum(var, 1e-12))

        opt_s = least_squares(
            res_sig,
            [math.log(max(np.percentile(si_sub, 15), 0.1)), 0.6, 0.2, math.log(0.2), math.log(0.01), 0.0],
            bounds=([-10, -1.0, 0.0, -10, -8, -8.0], [15, 5.0, 3.5, 5, -2, 4.0]),
            loss="soft_l1",
        )
        ps = opt_s.x
        m_int_si = u2_sub <= 0.96
        r_int_si = opt_s.fun[m_int_si] if np.sum(m_int_si) >= 30 else opt_s.fun
        rms_ln_si = float(np.clip(np.median(np.abs(r_int_si - np.median(r_int_si))) / 0.67449, 0.06, 0.22))
        m_cor_si = u2_sub > 1.02
        if np.sum(m_cor_si) >= 25:
            r_cor_si = opt_s.fun[m_cor_si]
            rms_ln_si_cor = float(
                np.clip(np.median(np.abs(r_cor_si - np.median(r_cor_si))) / 0.67449, rms_ln_si, 0.42)
            )
        else:
            rms_ln_si_cor = max(rms_ln_si, 0.26)

        # 4. Extract high-u profile/summation non-linearity delta_nl_u1 and 3-knot random scatter
        #    (a_stat[0..2], c_sysI[0..2], c_wbg) in the exact 20-shell u_law frame on seeds .01-.05.
        sh_u_l, sh_it_l, sh_ish_l, sh_sish_l, sh_iw_l, sh_fdof_l, sh_dof_l, sh_tdet_l = (
            [], [], [], [], [], [], [], []
        )
        real_ulaw_seeds = []
        real_zimed_seeds = []
        real_sirel_seeds = []
        for uv, s2v, mpv, epsv, itv, imv, simv, k0s in seed_vii_list:
            ord_v = np.argsort(s2v)
            tv = np.maximum(np.interp(uv, U_T_KNOTS, t_knots), 0.04)
            n_sh_v = min(20, max(2, len(ord_v) // 15))
            loc_r = np.ones(len(s2v), dtype=np.float64)
            for ch in np.array_split(ord_v, n_sh_v):
                so, st = float(np.sum(imv[ch])), float(np.sum(itv[ch]))
                if so <= 0 or st <= 0 or len(ch) < 8:
                    continue
                loc = st / so
                loc_r[ch] = loc
                dof = math.sqrt(len(ch) / (len(ch) - 1.0)) if len(ch) > 1 else 1.0
                p_ch = itv[ch] / st
                fdof = np.maximum(0.25, 1.0 - 2.0 * p_ch + float(np.sum(p_ch * p_ch))) * (dof ** 2)
                t_ch_norm = tv[ch] * (st / max(float(np.sum(itv[ch] * tv[ch])), 1e-12))
                fp2_ch = sm.f_prot_sq(s2v[ch]) / f_p2_0
                iw_ch = np.maximum(epsv[ch] * k0_ref * fp2_ch * np.exp(-2.0 * (b0_form_ref + ex_b) * s2v[ch]), 1e-12)
                sh_u_l.append(uv[ch])
                sh_it_l.append(itv[ch])
                sh_ish_l.append(imv[ch] * loc)
                sh_sish_l.append(simv[ch] * loc)
                sh_iw_l.append(iw_ch)
                sh_fdof_l.append(fdof)
                sh_dof_l.append(np.full(len(ch), dof, dtype=np.float64))
                sh_tdet_l.append(t_ch_norm)
            den_r = np.sqrt((1.05 * simv * loc_r) ** 2 + (0.064 * itv) ** 2)
            real_ulaw_seeds.append(float(np.std((imv * loc_r - itv) / np.maximum(den_r, 1e-12))))
            real_zimed_seeds.append(float(np.median((imv * loc_r - itv) / np.maximum(simv * loc_r, 1e-12))))
            real_sirel_seeds.append(float(np.median((simv * loc_r) / np.maximum(itv, 1e-12))))

        su = np.concatenate(sh_u_l)
        sit = np.concatenate(sh_it_l)
        sish = np.concatenate(sh_ish_l)
        ssish = np.concatenate(sh_sish_l)
        siw = np.concatenate(sh_iw_l)
        sfdof = np.concatenate(sh_fdof_l)
        sdof = np.concatenate(sh_dof_l)
        stdet = np.concatenate(sh_tdet_l)
        src = sit / siw

        mu_hi_nl = su >= 0.68
        if np.sum(mu_hi_nl) >= 30:
            br_m = src[mu_hi_nl] / (src[mu_hi_nl] + 1.5)
            bbar_m = float(np.sum(sit[mu_hi_nl] * br_m) / max(float(np.sum(sit[mu_hi_nl])), 1e-12))
            db_m = br_m - bbar_m
            var_m = ssish[mu_hi_nl] ** 2 + (0.04 * sit[mu_hi_nl]) ** 2
            wg_m = 1.0 / np.maximum(var_m, 1e-12)
            num_nl = float(np.sum(wg_m * (sish[mu_hi_nl] - sit[mu_hi_nl] * stdet[mu_hi_nl]) * sit[mu_hi_nl] * db_m))
            den_nl = float(np.sum(wg_m * ((sit[mu_hi_nl] * db_m) ** 2)))
            delta_nl_u1 = float(np.clip(num_nl / max(den_nl, 1e-12), 0.0, 0.45))
        else:
            delta_nl_u1 = 0.22

        del_vec = nl_ramp(su, Bw, delta_nl_u1)
        br_all = src / (src + 1.5)
        fnl = np.ones_like(su)
        pos_cur = 0
        for uv_c in sh_u_l:
            ln_c = len(uv_c)
            sl = slice(pos_cur, pos_cur + ln_c)
            pos_cur += ln_c
            bbar_sl = float(np.sum(sit[sl] * br_all[sl]) / max(float(np.sum(sit[sl])), 1e-12))
            raw_g = np.maximum(0.25, 1.0 + del_vec[sl] * (br_all[sl] - bbar_sl))
            fnl[sl] = raw_g * (float(np.sum(sit[sl])) / max(float(np.sum(sit[sl] * raw_g)), 1e-12))

        den_u = np.sqrt((1.05 * ssish) ** 2 + (0.064 * sit) ** 2)
        u_tot = ((sish - sit) * sdof) / np.maximum(den_u, 1e-12)
        det_tot = sit * stdet * fnl
        u_det = ((det_tot - sit) * sdof) / np.maximum(den_u, 1e-12)

        w0, w1, w2 = hat_weights_3(su, U_SC_KNOTS)
        wa = (ssish / den_u) ** 2
        wc = ((sit / den_u) ** 2) * sfdof
        ramp_wbg = np.clip((0.85 - su) / 0.15, 0.0, 1.0)
        wwbg = ((siw * ramp_wbg) / den_u) ** 2

        ok_all = np.abs(u_tot) <= 10.0
        if np.sum(ok_all) < 30:
            ok_all = np.ones(len(u_tot), dtype=bool)
        cX, cvdet, cy, cw = [], [], [], []
        for ulo, uhi in [(0.0, 0.55), (0.55, 0.74), (0.74, 0.87), (0.87, 2.0)]:
            mb = ok_all & (su >= ulo) & (su < uhi)
            if int(np.sum(mb)) < 16:
                continue
            nq = min(4, max(1, int(np.sum(mb)) // 20))
            qs = np.quantile(src[mb], np.linspace(0.0, 1.0, nq + 1))
            qs[0] -= 1e-6
            qs[-1] += 1e-6
            for qj in range(nq):
                mc = mb & (src >= qs[qj]) & (src <= qs[qj + 1]) & (np.abs(u_tot) <= 8.5)
                if int(np.sum(mc)) < 8:
                    continue
                cX.append(
                    [
                        float(np.mean(w0[mc] * wa[mc])),
                        float(np.mean(w1[mc] * wa[mc])),
                        float(np.mean(w2[mc] * wa[mc])),
                        float(np.mean(w0[mc] * wc[mc])),
                        float(np.mean(w1[mc] * wc[mc])),
                        float(np.mean(w2[mc] * wc[mc])),
                        float(np.mean(wwbg[mc])),
                    ]
                )
                cvdet.append(float(np.mean(u_det[mc] ** 2)))
                cy.append(math.sqrt(float(np.mean(u_tot[mc] ** 2))))
                cw.append(math.sqrt(int(np.sum(mc)) / max(int(np.sum(ok_all)), 1)) * (1.5 if ulo == 0.0 else 1.0))

        cX.append(
            [
                float(np.mean(w0[ok_all] * wa[ok_all])),
                float(np.mean(w1[ok_all] * wa[ok_all])),
                float(np.mean(w2[ok_all] * wa[ok_all])),
                float(np.mean(w0[ok_all] * wc[ok_all])),
                float(np.mean(w1[ok_all] * wc[ok_all])),
                float(np.mean(w2[ok_all] * wc[ok_all])),
                float(np.mean(wwbg[ok_all])),
            ]
        )
        cvdet.append(float(np.mean((u_det[ok_all] - np.mean(u_det[ok_all])) ** 2)))
        cy.append(float(np.std(u_tot[ok_all])))
        cw.append(4.0)
        mX = np.asarray(cX, dtype=np.float64)
        vvdet = np.asarray(cvdet, dtype=np.float64)
        vy = np.asarray(cy, dtype=np.float64)
        vw = np.asarray(cw, dtype=np.float64)

        snrc = sit / np.maximum(ssish, 1e-9)
        zi_rnd = ((sish - det_tot) * sdof) / np.maximum(ssish, 1e-9)
        mw_out = (snrc < 2.5) & (su >= 0.80) & (np.abs(zi_rnd) < 6.0)
        a2_lb_anc = 0.96 if Bw < 24.0 else 0.82
        if int(np.sum(mw_out)) >= 20:
            a2_anc = float(
                np.clip(
                    math.sqrt(max(0.5, float(np.var(zi_rnd[mw_out])) - float(np.mean((0.06 * snrc[mw_out]) ** 2)))),
                    a2_lb_anc,
                    1.03,
                )
            )
        else:
            a2_anc = max(0.95, a2_lb_anc)

        a0_lb = 0.78 if Bw < 26.0 else 0.62
        c0_lb = 0.016 if Bw < 28.0 else 0.005
        c0_ub = 0.024 if Bw < 28.0 else 0.055
        c2_ub = 0.095 if Bw < 28.0 else 0.120

        def res_u7(p):
            pred = np.sqrt(np.maximum(vvdet + mX @ (p * p), 1e-9))
            reg = np.array(
                [
                    0.06 * (p[0] - 0.78),
                    0.08 * (p[1] - 0.96),
                    1.80 * (p[2] - a2_anc),
                    0.06 * (p[3] - (0.020 if Bw < 28.0 else 0.010)),
                    0.08 * (p[4] - 0.045),
                    0.08 * (p[5] - 0.075),
                    0.08 * (p[6] - 0.003),
                ],
                dtype=np.float64,
            )
            return np.concatenate([vw * (pred - vy), reg])

        opt_u = least_squares(
            res_u7,
            [max(0.78, a0_lb), 0.96, a2_anc, max(0.010, c0_lb), 0.045, 0.075, 0.003],
            bounds=(
                [a0_lb, 0.75, max(0.78, a2_anc - 0.05), c0_lb, 0.014, 0.025, 0.0],
                [1.03, 1.12, min(1.04, a2_anc + 0.05), c0_ub, 0.095, c2_ub, 0.020],
            ),
        )
        a0_b, a1_b, a2_b, c0_b, c1_b, c2_b, cwbg_b = [float(v) for v in opt_u.x]

        # Closed-loop intensity u_law match strictly on training seeds .01-.05
        # (eliminates small Jensen/clipping/shell-DOF bias while keeping outer a2_b intact for CC@dmin)
        tgt_ulaw = float(np.mean(real_ulaw_seeds)) if real_ulaw_seeds else 0.87
        tgt_zimed = float(np.mean(real_zimed_seeds)) if real_zimed_seeds else 0.0
        tgt_sirel = float(np.mean(real_sirel_seeds)) if real_sirel_seeds else 0.05
        w_t5_cal = float(np.clip((Bw - 25.0) / 10.0, 0.0, 1.0))
        n_cal_passes = 3 if Bw < 28.0 else 2
        for _pass in range(n_cal_passes):
            rng_cal = np.random.default_rng(777)
            sim_ulaw_5 = []
            sim_zimed_5 = []
            sim_sirel_5 = []
            for uv, s2v, mpv, epsv, itv, imv, simv, k0s in seed_vii_list:
                nv = len(uv)
                tv_c = np.interp(uv, U_T_KNOTS, t_knots)
                fp2_c = sm.f_prot_sq(s2v) / f_p2_0
                iw_c = np.maximum(epsv * k0_ref * fp2_c * np.exp(-2.0 * (b0_form_ref + ex_b) * s2v), 1e-12)
                br_c = (itv / iw_c) / ((itv / iw_c) + 1.5)
                del_c = nl_ramp(uv, Bw, delta_nl_u1)
                ord_c = np.argsort(s2v)
                fnl_c = np.ones(nv, dtype=np.float64)
                for ch_c in np.array_split(ord_c, min(20, max(2, nv // 120))):
                    bb_c = float(np.sum(itv[ch_c] * br_c[ch_c]) / max(float(np.sum(itv[ch_c])), 1e-12))
                    rg_c = np.maximum(0.25, 1.0 + del_c[ch_c] * (br_c[ch_c] - bb_c))
                    fnl_c[ch_c] = rg_c * (float(np.sum(itv[ch_c])) / max(float(np.sum(itv[ch_c] * rg_c)), 1e-12))
                ix_c = itv * tv_c * fnl_c
                fr_c = np.where(
                    uv <= 1.0,
                    1.0,
                    np.maximum(1.0 - (4.0 / math.pi) * np.arccos(np.clip(1.0 / uv, -1.0, 1.0)), 0.22),
                )
                wb_c = np.exp(-(((np.sqrt(s2v) - 0.160) / 0.045) ** 2))
                cb_c = np.maximum(0.0, uv - 0.95) ** 2
                sbg_c = math.exp(ps[0]) * np.exp(ps[1] * (uv * uv) + ps[2] * wb_c + ps[5] * cb_c)
                w_co_c = np.clip((uv - 0.98) / 0.08, 0.0, 1.0)
                rms_vec_c = (1.0 - w_co_c) * rms_ln_si + w_co_c * rms_ln_si_cor
                eta_c = np.clip(rng_cal.normal(0.0, 1.0, size=nv) * rms_vec_c, -2.5 * rms_vec_c, 2.5 * rms_vec_c)
                sib_c = (
                    np.sqrt(8.0 / (mpv.astype(np.float64) * fr_c))
                    * np.sqrt(sbg_c ** 2 + math.exp(ps[3]) * ix_c + (math.exp(ps[4]) * ix_c) ** 2)
                    * np.exp(eta_c)
                )
                wk0c, wk1c, wk2c = hat_weights_3(uv, U_SC_KNOTS)
                ast_c = np.sqrt(wk0c * (a0_b ** 2) + wk1c * (a1_b ** 2) + wk2c * (a2_b ** 2))
                csy_c = np.sqrt(wk0c * (c0_b ** 2) + wk1c * (c1_b ** 2) + wk2c * (c2_b ** 2))
                swbg_c = cwbg_b * (iw_c * tv_c) * np.clip((0.85 - uv) / 0.15, 0.0, 1.0)
                zt5 = np.clip(rng_cal.standard_t(5.0, size=nv), -5.5, 5.5) / 1.26681
                zg = rng_cal.normal(0.0, 1.0, size=nv)
                zss_c = math.sqrt(1.0 - w_t5_cal) * zg + math.sqrt(w_t5_cal) * zt5
                iob_c = (
                    ix_c * np.exp(np.clip(csy_c * zss_c, -0.6, 0.6) - 0.5 * (csy_c ** 2))
                    + ast_c * sib_c * rng_cal.normal(0.0, 1.0, size=nv)
                    + swbg_c * rng_cal.normal(0.0, 1.0, size=nv)
                )
                loc_sc = np.ones(nv, dtype=np.float64)
                for ch_c in np.array_split(ord_c, min(20, max(2, nv // 15))):
                    if np.sum(iob_c[ch_c]) > 0:
                        loc_sc[ch_c] = np.sum(itv[ch_c]) / np.sum(iob_c[ch_c])
                den_sc = np.sqrt((1.05 * sib_c * loc_sc) ** 2 + (0.064 * itv) ** 2)
                sim_ulaw_5.append(float(np.std((iob_c * loc_sc - itv) / np.maximum(den_sc, 1e-12))))
                sim_zimed_5.append(float(np.median((iob_c * loc_sc - itv) / np.maximum(sib_c * loc_sc, 1e-12))))
                sim_sirel_5.append(float(np.median((sib_c * loc_sc) / np.maximum(itv, 1e-12))))
            if Bw < 28.0:
                si_rat = float(np.clip((1.012 * tgt_sirel) / max(float(np.mean(sim_sirel_5)), 1e-9), 0.96, 1.07))
                ln_si = math.log(si_rat)
                ps[0] += ln_si
                ps[3] += 2.0 * ln_si
                ps[4] += ln_si
            ul_rat = float(np.clip(tgt_ulaw / max(float(np.mean(sim_ulaw_5)), 1e-6), 0.92, 1.12))
            k_adj = ul_rat ** 1.45
            a1_b = float(np.clip(a1_b * k_adj, 0.72, 1.18))
            c1_b = float(np.clip(c1_b * k_adj, 0.012, 0.115))
            c2_b = float(np.clip(c2_b * k_adj, 0.020, c2_ub if Bw < 28.0 else 0.135))
            cwbg_b = float(np.clip(cwbg_b * k_adj, 0.0, 0.025))
            if Bw < 30.0:
                a0_b = float(np.clip(a0_b * ul_rat, a0_lb, 1.08))
                c0_b = float(np.clip(c0_b * ul_rat, c0_lb, c0_ub))
            if Bw < 28.0:
                err_zim = float(np.mean(sim_zimed_5)) - (tgt_zimed + 0.014)
                delta_nl_u1 = float(np.clip(delta_nl_u1 + 0.95 * err_zim, 0.05, 0.24))

        # 5. Per-reflection normalized anomalous mate difference scatter (accounting for French-Wilson shrinkage)
        if ano_z_diff:
            ares = np.concatenate([x[0] for x in ano_z_diff])
            asi = np.concatenate([x[1] for x in ano_z_diff])
            ait = np.concatenate([x[2] for x in ano_z_diff])
            qsa = np.quantile(ait, np.linspace(0.0, 1.0, 7))
            ba_snr2, ba_zstd = [], []
            for j in range(6):
                mq = (ait >= qsa[j]) & (ait <= qsa[j + 1])
                if np.sum(mq) >= 15:
                    zaj = ares[mq] / asi[mq]
                    saj = ait[mq] / asi[mq]
                    mad_a = max(float(np.median(np.abs(zaj - np.median(zaj))) / 0.67449), 0.2)
                    zac = np.clip(zaj, -10.0 * mad_a, 10.0 * mad_a)
                    ba_snr2.append(float(np.mean(saj ** 2)))
                    ba_zstd.append(float(np.std(zac)))
            if len(ba_zstd) >= 2:
                ba_snr2 = np.asarray(ba_snr2)
                ba_zstd = np.asarray(ba_zstd)
                opt_an = least_squares(
                    lambda p: np.sqrt(p[0] ** 2 + 2.0 * (p[1] ** 2) * ba_snr2) - ba_zstd,
                    [0.95, 0.014],
                    bounds=([0.72, 0.002], [1.15, 0.06]),
                )
                fw_ano_comp = 1.010 + 0.005 * float(np.clip((Bw - 26.0) / 8.0, 0.0, 1.0))
                a_ind_b = float(np.clip(opt_an.x[0] * fw_ano_comp, 0.72, 1.15))
                c_ind_b = float(opt_an.x[1])
            else:
                a_ind_b, c_ind_b = 0.95, 0.014
        else:
            a_ind_b, c_ind_b = 0.95, 0.014

        cal_rows.append(
            {
                "B_Wilson": round(Bw, 3),
                "d_cc12": round(d_cc12, 5),
                "d_isig15": round(d_is15, 5),
                "d_isig20": round(d_is20, 5),
                "d_ccright": round(d_ccr, 5),
                "d_maprms": round(d_mrm, 5),
                "d_edge": round(d_edg, 5),
                "d_corner": round(d_cor_geom, 5),
                "d_mtz_min": round(d_mtz_min, 9),
                "p_miss_low12": round(p_miss_low12, 5),
                "sig0": round(math.exp(ps[0]), 5),
                "b1_u2": round(float(ps[1]), 5),
                "w_wat": round(float(ps[2]), 5),
                "g_shot": round(math.exp(ps[3]), 5),
                "c_sig": round(math.exp(ps[4]), 5),
                "b2_cor": round(float(ps[5]), 5),
                "rms_ln_si": round(rms_ln_si, 5),
                "rms_ln_si_cor": round(rms_ln_si_cor, 5),
                "a_stat": round(a0_b, 4),
                "a_stat_u1": round(a1_b, 4),
                "a_stat_u2": round(a2_b, 4),
                "c_sysI": round(c0_b, 5),
                "c_sysI_u1": round(c1_b, 5),
                "c_sysI_u2": round(c2_b, 5),
                "c_wbg": round(cwbg_b, 5),
                "a_ind": round(a_ind_b, 4),
                "c_sys_ind": round(c_ind_b, 5),
                "delta_nl_u1": round(delta_nl_u1, 5),
                "sigma_bg_cor": round(sigma_bg_cor, 4),
                "t_xds": t_knots,
                "p_obs": p_knots,
                "beta_bg": bg_knots,
            }
        )
        if idx_b % 15 == 0 or idx_b == len(rs_by_base) - 1:
            r = cal_rows[-1]
            print(
                f"  [{idx_b+1:3d}/115] B={Bw:5.1f} d_cc12={d_cc12:4.2f} | "
                f"a=({r['a_stat']:.2f},{r['a_stat_u1']:.2f},{r['a_stat_u2']:.2f}) "
                f"c=({r['c_sysI']:.4f},{r['c_sysI_u1']:.4f},{r['c_sysI_u2']:.4f}) "
                f"dnl={r['delta_nl_u1']:+.3f} sbg={r['sigma_bg_cor']:.2f}"
            )

    # Cross-B smoothing to suppress 5-seed calibration Monte Carlo variance:
    #   - B < 140: light 3-point [0.20, 0.60, 0.20] smoothing on noise scalars & outer t_xds (leave beta_bg unsmoothed)
    #   - B >= 140: 5-point [0.10, 0.22, 0.36, 0.22, 0.10] smoothing across curve & noise parameters
    n_b = len(cal_rows)
    orig_t = [np.asarray(r["t_xds"], dtype=np.float64) for r in cal_rows]
    orig_bg = [np.asarray(r["beta_bg"], dtype=np.float64) for r in cal_rows]
    sc_keys = (
        "a_stat",
        "a_stat_u1",
        "a_stat_u2",
        "c_sysI",
        "c_sysI_u1",
        "c_sysI_u2",
        "c_wbg",
        "a_ind",
        "c_sys_ind",
        "b2_cor",
        "rms_ln_si",
        "rms_ln_si_cor",
        "delta_nl_u1",
        "sigma_bg_cor",
    )
    orig_sc = {k: [float(r[k]) for r in cal_rows] for k in sc_keys}
    kw3 = np.array([0.20, 0.60, 0.20], dtype=np.float64)
    kw5 = np.array([0.10, 0.22, 0.36, 0.22, 0.10], dtype=np.float64)
    for i in range(n_b):
        bw_i = cal_rows[i]["B_Wilson"]
        if bw_i < 140.0:
            if i == 0:
                continue  # Keep B=18.2 endpoint unblurred
            idxs = [min(max(i + d_i, 0), n_b - 1) for d_i in (-1, 0, 1)]
            sm_t = orig_t[i].copy()
            sm_t_all = sum(w * orig_t[j] for w, j in zip(kw3, idxs))
            for k_i, uk in enumerate(U_T_KNOTS):
                if uk > 1.01:
                    sm_t[k_i] = min(sm_t_all[k_i], sm_t[k_i - 1])
            cal_rows[i]["t_xds"] = [round(float(v), 5) for v in sm_t]
            for k in sc_keys:
                val = sum(w * orig_sc[k][j] for w, j in zip(kw3, idxs))
                prec = 5 if ("sys" in k or "wbg" in k or "nl" in k or "rms_ln" in k or k == "b2_cor") else 4
                cal_rows[i][k] = round(float(val), prec)
        else:
            idxs = [min(max(i + d_i, 0), n_b - 1) for d_i in (-2, -1, 0, 1, 2)]
            sm_t = sum(w * orig_t[j] for w, j in zip(kw5, idxs))
            sm_t[0] = 1.0
            u_in_i = (cal_rows[i]["d_edge"] / max(cal_rows[i]["d_cc12"], 0.5)) + 0.015
            for k_i, uk in enumerate(U_T_KNOTS):
                if uk > u_in_i and k_i > 0:
                    sm_t[k_i] = min(sm_t[k_i], sm_t[k_i - 1])
            cal_rows[i]["t_xds"] = [round(float(v), 5) for v in sm_t]
            sm_bg = sum(w * orig_bg[j] for w, j in zip(kw5, idxs))
            u_lim_i = cal_rows[i]["d_edge"] / max(cal_rows[i]["d_cc12"], 0.5)
            for k_i, uk in enumerate(U_BG_KNOTS):
                if uk <= u_lim_i:
                    sm_bg[k_i] = 0.0
            cal_rows[i]["beta_bg"] = [round(float(v), 4) for v in sm_bg]
            for k in sc_keys:
                val = sum(w * orig_sc[k][j] for w, j in zip(kw5, idxs))
                prec = 5 if ("sys" in k or "wbg" in k or "nl" in k or "rms_ln" in k or k == "b2_cor") else 4
                cal_rows[i][k] = round(float(val), prec)

    out_cal = {
        "version": 2,
        "description": (
            "v2.2 empirical 115-point calibration strictly from training seeds .01-.05 (575 nanoBragg2 1AHO datasets; "
            "seeds .06-.10 held out for validation), with 18-knot XDS transfer T_xds(u), high-u within-shell "
            "profile/summation non-linearity delta_nl_u1, 3-knot (u, I/I_Wilson) random scatter + low-u shell bleed "
            "floor c_wbg, 16-knot outer-corner beta_bg(u) + per-seed radial-ring background fluctuation sigma_bg_cor, "
            "and XDSCONV sliding-window truncation."
        ),
        "train_seeds": ["01", "02", "03", "04", "05"],
        "heldout_seeds": ["06", "07", "08", "09", "10"],
        "K0_ref_1aho": k0_ref,
        "B0_form_1aho": b0_form_ref,
        "B0_nominal_1aho": 18.2,
        "u_t_knots": U_T_KNOTS,
        "u_p_knots": U_P_KNOTS,
        "u_bg_knots": U_BG_KNOTS,
        "u_sc_knots": U_SC_KNOTS,
        "xdsconv_params": {"half_win_records": 100, "clamped": True, "min_isigi": -3.0, "min_score": -35.0},
        "rows": cal_rows,
    }
    dst = args.out
    dst.write_text(json.dumps(out_cal, indent=2) + "\n")
    print(f"\nWrote {dst} ({len(cal_rows)} rows) in {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
