# `SIGF` — Bi-Directional Diffraction `(F, SIGF)` Simulator & Blind Experimental Forensics

Fast, standalone Python toolkit (`numpy` + `gemmi` only) for **both directions** of macromolecular X-ray diffraction error modeling:

1. **Forward Simulation (`sim_reso_mtz.py` / `sim-reso-mtz`):** Generates realistic merged structure-factor & intensity MTZ datasets (`FreeR_flag`, `F`, `SIGF`, `DANO`, `SIGDANO`, `ISYM`, `Fplus`, `SIGFplus`, `Fminus`, `SIGFminus`, `IMEAN`, `SIGIMEAN`, `Iplus`, `SIGIplus`, `Iminus`, `SIGIminus`) from ground-truth amplitudes `F0` and a target resolution limit `d_min` (or physical beam/crystal/background/dose parameters) in **~0.2 s**, calibrated against `575` pixel-level `nanoBragg` + `nonBragg` + `XDS` + `XDSCONV` simulations (`18.2–518.2 Å²` Wilson $B$) and validated out-of-sample on `574` held-out datasets.
2. **Blind Inverse Deduction (`--deduce` / `deduce_mtz_params.py` / `deduce-mtz-params`):** Given **only** `(H, K, L, F, SIGF)` (or `IMEAN, SIGIMEAN`) from any `.mtz` or SF-mmCIF (`.cif` / `.ent`) — without reading PDB headers or papers — reconstructs the $I \to F$ truncation software & cutoff, absolute electron scale, Debye–Babinet Wilson $B$, square-detector corner $d_{\text{edge}}$ vs spherical cutoff, multi-pass / multi-crystal merge discontinuities, spindle Lorentz blind cone, 16-bit low-angle overload saturation, diffuse background mount regime (`RT silica capillary` vs `100 K cryo water` vs `ice rings`), anomalous `weirdsigma` error models, universal Darwin photon-budget invariant ($\Phi_{\text{tot}} V_{\text{illum}} \lambda^3 \cdot \text{DQE}$), wavelength & radiation-damage crystal-size bounds, and an executable forward replay command in **~0.3–0.8 s**.

---

## 1. Installation

Requires Python $\ge 3.8$ with `numpy` and `gemmi` (included out of the box in `ccp4-python` and `phenix.python`, or installable via `pip`):

```bash
git clone https://github.com/jmholton/SIGF.git
cd SIGF
pip install .
```

You can also run `python sim_reso_mtz.py` and `python deduce_mtz_params.py` directly from a checkout with zero installation.

---

## 2. Quick Start

### A. Blind Inverse Forensics (`(H, K, L, F, SIGF) -> Experimental Parameters`)

```bash
# Deduce experimental & processing parameters blindly from any MTZ or SF-mmCIF (.cif / .ent):
python sim_reso_mtz.py 1aho.mtz --deduce --f-only --n-residues 64

# Works directly on PDB structure-factor mmCIF files (auto-detects F_meas_au / intensity_meas):
python deduce_mtz_params.py 7KR0-sf.cif --n-residues 169 --json 7kr0_deduced.json

# Optional: cross-check claimed SIGF against an atomic model or FC column to uncover uncalibrated detector GAIN / SDfac:
python sim_reso_mtz.py 1aho.mtz --deduce --f-col FP --sigf-col SIGFP --model 1aho.pdb --n-residues 64
```

### B. Forward Simulation (`F0 + Experiment / d_min -> Realistic MTZ`)

```bash
# Baseline nanoBragg2 Pilatus 6M + XDS + XDSCONV simulation at CC1/2 = 0.30 resolution d_min = 2.0 A:
python sim_reso_mtz.py FrightB0.mtz 2.0 -o sim_2.0A.mtz

# Keep un-attenuated crystal ground truth (F_true, I_true, and PHI_true if --phi-col given):
python sim_reso_mtz.py FrightB0.mtz 2.0 --keep-true -o sim_2.0A.mtz

# Spherical cutoff at d_min + ideal unit spot-profile transfer + CCP4 TRUNCATE (-3.7 sigma):
python sim_reso_mtz.py F0.mtz 1.8 --cutoff dmin --ideal-transfer --truncate-mode ccp4 -o sim_ccp4_1.8A.mtz

# Specify physical beamline, crystal, solvent/air background & mechanical errors (solves B_Wilson to hit 2.0 A):
python sim_reso_mtz.py F0.mtz 2.0 --xtal-size-um 10 --water-um 80 --exposure-s 10 --calib-err 0.015 -o sim_10um.mtz

# Forward physical mode (d_min <= 0 -> solves emergent resolution limit from crystal B_Wilson & dose):
python sim_reso_mtz.py F0.mtz 0 --B-wilson 55.0 --xtal-size-um 12 --auto-dose -o sim_fwd.mtz

# Multi-turn sweep separating pixel-locked flat-field error (--calib-err) from non-repeating mechanical jitter:
python sim_reso_mtz.py F0.mtz 1.8 --wedge-deg 3600 --exposure-s 36 --sample-vib-um 3.0 -o sim_10turn.mtz
```

---

## 3. Blind Experimental Showcase (`1AHO` vs `1RB5` vs `7KR0`)

Running `--deduce` strictly on `(H, K, L, F, SIGF)` without reading PDB headers or publications cleanly separates hardware eras, sample mounts, processing pipelines, and pathological error models:

| Deduceable Property | **`1AHO`** (1997 Mar 16-bit IP, 2 crystals, RT capillary) | **`1RB5`** (1997 SSRL ADSC CCD, 4-$\lambda$ Se-MAD, `SCALA`) | **`7KR0`** (2020 ALS 8.3.1 Pilatus3 6M, `17 keV`, `60.8×` `XDS`) |
| :--- | :--- | :--- | :--- |
| **$I \to F$ software** | `CCP4 TRUNCATE (-3.7/-4.0σ)` (`min F/SIGF = 1.963` ac, `1.347` cen) | `Merged French-Wilson` (`0σ` cut; `min = 1.949` ac, `1.308` cen) | `Merged French-Wilson` (`phenix`; `min = 1.350` cen, matches `REMARK 3`) |
| **Usable resolution** | File `0.965 Å` (`CC1/2=0.3` at `0.997 Å`, `<I/sigI>=2` at `0.965 Å`) | File `1.700 Å`, **`CC1/2=0.3` at `1.826 Å`** (`REMARK 200`: `1.80 Å`) | `spherical` at `0.770 Å` (`CC1/2 = 0.611`, `<I/sigI>=2` at `0.793 Å`) |
| **Pass / crystal merges** | **`d = 1.126 Å` step (`×1.50`, `z = 40.1`)** (Crystal 1 stops at `1.11 Å`) | Single continuous pass (`1` crystal) | Single continuous pass (`1` crystal) |
| **Spindle & low-$d$ overloads** | **`c`-axis blind cone (`56.9%` vs `92.2%`)** + **`7.9%` 16-bit low-$d$ overloads** | No blind cone, zero low-$d$ overloads (CCD) | **Zero blind cone (`>99%`)** even in `C 2` + zero overloads (20-bit PAD) |
| **Background & temperature** | **RT glass capillary + mother liquor** (`3.9 Å` silica hump, no ice) | **Cryo (`90 K`)** + **`weirdsigma` flagged** (`sigma_bg -> 0`, `g_shot = 95 e²/ph`) | **Cryo (`100 K`)**, He / thin loop (`water ~ 7 µm, air ~ 4 mm`, `c_merged = 0.53%`) |
| **Wavelength & crystal bounds** | **`1.54 Å` Cu $K\alpha$ ruled out (`λ <= 1.037 Å`)**; RT xtal **`>= 134–330 µm`** (true `292 µm`) | `λ <= 1.827 Å` (true `0.93–1.07 Å`) | **`λ <= 0.827 Å` hard X-rays** (true `0.7292 Å`); 100 K xtal `>= 25.5 µm` |
| **Model `SDfac` check** | `2.38×` (`>1.13 Å`) vs `1.03×` (`<1.13 Å`) across the 2 crystals | **`0.47×` (`0.40×` outer)** — confirms `2.1–2.5×` `SCALA`/MAD `SIGF` inflation | **`1.029×` overall (`0.941×` outer)** — calibrated within `~3–6%` of exact |

---

## 4. Physical & Statistical Formulation

### 4.1 Forward Model (`sim_reso_mtz.py`)
- **Gauge-Consistent `B_Wilson(d_min)` & 18-Knot XDS Transfer $T_{\text{XDS}}(u)$:** Interpolates a PAVA-monotonized 115-point lookup table ([nanobragg2_calibration.json](nanobragg2_calibration.json), built strictly from training seeds `.01–.05` via `build_calibration_v2.py`), modeling subcrystal spot-profile roll-off $T_{\text{XDS}}(u)$ ($u = d_{\text{edge}}/d$), within-shell profile/summation non-linearity $f_{\text{nl}}(u, I/I_{\text{Wilson}})$, outer-corner background residuals $\beta_{\text{bg}}(u)$, and 3-knot intensity scatter ($a_{\text{stat}}(u), c_{\text{sysI}}(u)$).
- **Orthogonal Friedel-Mate Noise & Conversion:** Generates $(I^+, I^-)$ matching anomalous `zDANO` and `CC_ano`, followed by Kabsch `XDSCONV` (`201`-record sliding-window `<I>`, separate $F^+/F^-$ French–Wilson at $I/\sigma_I \ge -3.0$, $F = \frac{1}{2}(F^+ + F^-)$) or merged `CCP4 TRUNCATE`.
- **First-Principles Experimental Scaling:** Scales Bragg photons ($\propto \Phi_{\text{tot}} V_{\text{illum}} \lambda^3 / V_{\text{cell}}^2$) against exact `nonBragg` diffuse scattering (`water.stol`, `air.stol`, and `incoherent.stol` Compton + $B$-diffuse profiles), separating pixel-locked flat-field error (`--calib-err`, which repeats on the same detector pixels across multi-turn sweeps unless `--n-det-pos > 1`) from non-repeating mechanical errors (`--beam-flicker`, `--shutter-jitter-ms`, `--sample-vib-um`, `--spindle-wobble-um`).

### 4.2 Inverse Solver (`deduce_mtz_params.py`)
- **Exact French–Wilson Inversion & Bias-Free Regressor:** Reconstructs pre-truncation $\hat{\sigma}_I = \frac{F \cdot \text{SIGF}}{\mu_y(z)\sigma_y(z)}$ smoothly from high SNR ($\mu_y\sigma_y \to \frac{1}{2}$) down to $z = -20$, and uses the exact Bayesian posterior expectation $I_{\text{post}} = \mathbb{E}[I_{\text{true}} \mid I_{\text{obs}}, \sigma_I, \Sigma] = F^2 + \gamma\sigma_F^2$ in a joint 2D `(shell × intensity-octile)` active-set NNLS fit $\sigma_{I,\text{ref}}^2 = \sigma_{\text{bg}}^2(d) + g_{\text{shot}} I + c_{\text{merged}}^2 I^2$.
- **Universal Wedge-Independent Darwin Invariant:** Because single-pass photon gain scales as $1/\Delta\phi$ while multiplicity scales as $\Delta\phi$, total rotation wedge $\Delta\phi$ cancels out of merged $g_{\text{shot}}$, pinning $\Phi_{\text{tot}} V_{\text{illum}} \lambda^3 \cdot \text{DQE} = \frac{2\pi V_{\text{cell}}^2}{M_{\max} r_e^2 \langle LP \rangle g_{\text{shot}}}$ directly from `(H, K, L, F, SIGF)`.

---

## 5. Out-of-Sample Forward Validation (`574` Held-Out `nanoBragg2` Datasets)

Evaluated strictly out-of-sample on held-out seeds `.06–.10` (`574` Pilatus 6M + `XDS` + `XDSCONV` datasets across `0.86–4.91 Å`; values show `real / sim (>3se)`):

| Metric | `0.0–1.2 Å` | `1.2–2.0 Å` | `2.0–3.0 Å` | `3.0–4.0 Å` | `4.0–9.0 Å` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`nF (all)`** | `39027 / 38999 (0.00)` | `11446 / 11288 (0.07)` | `3681 / 3518 (0.07)` | `1097 / 1102 (0.00)` | `531 / 522 (0.00)` |
| **`compl d>=dmin`** | `0.985 / 0.985 (0.00)` | `0.998 / 0.998 (0.17)` | `0.997 / 0.997 (0.00)` | `0.987 / 0.991 (0.00)` | `0.977 / 0.982 (0.00)` |
| **`dB vs truth` (Å²)** | `-1.26 / -1.27 (0.00)` | `-4.72 / -4.62 (0.05)` | `-10.18 / -10.06 (0.03)` | `-24.72 / -25.30 (0.05)` | `-40.54 / -40.19 (0.00)` |
| **`R_F d>=dmin %` (`k,B`)** | `6.05 / 6.14 (0.10)` | `6.12 / 6.14 (0.03)` | `6.03 / 6.04 (0.00)` | `6.67 / 7.23 (0.11)` | `6.37 / 6.85 (0.00)` |
| **`R_F shellscaled %`** | `4.72 / 4.68 (0.10)` | `3.00 / 2.98 (0.03)` | `2.58 / 2.63 (0.10)` | `2.93 / 2.95 (0.05)` | `3.09 / 2.99 (0.12)` |
| **`CC(I,Itrue) @ dmin`** | `0.676 / 0.688 (0.10)` | `0.690 / 0.695 (0.00)` | `0.688 / 0.686 (0.00)` | `0.516 / 0.505 (0.05)` | `0.409 / 0.329 (0.06)` |
| **`<I/sigI> @ dmin`** | `0.868 / 0.859 (0.10)` | `0.715 / 0.736 (0.03)` | `0.718 / 0.720 (0.03)` | `0.500 / 0.455 (0.05)` | `0.349 / 0.335 (0.12)` |
| **`u_law std shellscaled`** | `0.899 / 0.898 (0.00)` | `0.862 / 0.857 (0.12)` | `0.872 / 0.865 (0.03)` | `0.907 / 0.898 (0.00)` | `0.923 / 0.928 (0.00)` |
| **`zDANO std`** | `0.900 / 0.900 (0.00)` | `0.940 / 0.937 (0.03)` | `0.946 / 0.942 (0.03)` | `0.931 / 0.935 (0.00)` | `0.892 / 0.893 (0.00)` |

---

## 6. Repository Structure

- `sim_reso_mtz.py` — Forward simulator CLI & Python API (`--deduce` dispatches to the inverse solver).
- `deduce_mtz_params.py` — Blind inverse experimental & processing forensics CLI & Python API.
- `nanobragg2_calibration.json` — 115-point empirical calibration table (`B_Wilson = 18.2–518.2 Å²`).
- `build_calibration_v2.py` — Reproducible calibration extractor from raw `nanoBragg2` `FreeRed.mtz` runs.
- `pyproject.toml` & `setup.py` — PEP 621 package metadata, wheel data packaging (`nanobragg2_calibration.json`), and CLI entry points (`sim-reso-mtz`, `deduce-mtz-params`).
- `LICENSE` — MIT License.

