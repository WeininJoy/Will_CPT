# Why ρ < 1 at High-k Even When k_allowedK ≈ k_integerK, and the ρ → 1 Resonance Block

## Context

The CCA (`cca_sparse_2D_dk-penalty_analysis.py`) compares two bases:
- **allowedK**: k-values that are zeros of `vr^∞(k)` — the palindrome (CPT) boundary condition
- **integerK**: k = n√K for integer n — spatial quantization

Two puzzles arise from the CCA output:

1. At high k the two bases are nearly aligned in the heatmap (coefficient matrices ≈ identity), yet the canonical correlations remain ρ ≈ 0.91 rather than ρ → 1.
2. Near k_index ~33–43 (k_phys ≈ 5.2–6.7×10⁻³ Mpc⁻¹) the CCA heatmap shows a diagonal identity block with ρ → 1, coinciding with the PPS enhancement peak at k ≈ 6×10⁻³ Mpc⁻¹.

---

## Root Cause: Different FCB Boundary Conditions

Even when `k_allowedK ≈ k_integerK` (difference < 0.01), the two basis solutions are **physically different** because they carry different boundary conditions at the FCB (Final Crunch Bounce).

### Solution reconstruction in `generate_multi_perturbation_bases`

For every mode, the full time-domain solution is reconstructed as:

```
Y(η) = U_ABC(η) @ (X1 @ x_inf) + U_DEF(η) @ (X2 @ x_inf)
```

where `x_inf = [dr^∞, dm^∞, vr^∞, vm^∞]` is the state vector at the FCB, solved by:

```python
M_matrix = (A @ X1 + D @ X2)[2:6, :]   # rows: dr, dm, vr, vm
x_inf    = lstsq(M_matrix, x_rec)[0]
```

### The key difference: vr^∞ at the FCB

| Basis | Physical meaning | vr^∞ at FCB |
|-------|-----------------|-------------|
| **allowedK** | k is a root of vr^∞(k) — palindrome satisfied | `vr^∞ ≈ 0` by construction |
| **integerK** | k = n√K, spatial quantization only | `vr^∞ ≠ 0` in general |

Even when `k_allowedK ≈ k_integerK`, the vector `x_inf` differs between the two modes — most importantly `x_inf[2] = vr^∞`. This difference propagates through the entire conformal-time evolution via `X1 @ x_inf` and `X2 @ x_inf`, producing genuinely different solutions `Y(η)`.

---

## Why ρ ≈ 0.91 and Not → 1

The canonical correlation measures the time-domain overlap between matched solutions:

```
ρ ≈ ⟨f1_i(η) · f2_i(η)⟩ / √( ⟨f1_i²⟩ · ⟨f2_i²⟩ )
```

Because `x_inf` differs at the FCB, the two solutions have **different amplitude ratios** between `dr, dm, vr, vm` throughout the entire evolution — not merely a phase shift. The overlap integral therefore falls below 1 even when the oscillation frequencies (set by k) are nearly identical.

---

## Physical Interpretation of ρ

The value of ρ encodes genuine physics, not numerical noise:

- **ρ → 1**: The palindrome condition and integer spatial quantization are simultaneously satisfied — the mode is a valid solution for both bases.
- **ρ ≈ 0.91 at high k**: Real incompatibility between the two physical conditions even at nearly equal k. The FCB states differ and cannot be made identical by varying k alone.
- **Low-k modes with mixing**: At low k, k_allowedK and k_integerK differ more and vr^∞ varies more rapidly with k, so the mismatch is larger and a linear combination of several modes is needed to bridge the two bases.

---

## Why Mixing Persists in the High-k Heatmap

The off-diagonal elements in the CCA coefficient matrices at high k are **not** due to k-value proximity issues. They arise because an integerK mode carries a non-zero `vr^∞` component that must be cancelled or redistributed across neighbouring modes from the allowedK basis in order to construct a valid palindromic solution. The mixing is the CCA's way of encoding this physical mismatch.

---

## The ρ → 1 Resonance Block at k_index ~33–43

### Observation

The CCA heatmap shows a striking diagonal identity block (ρ → 1) near k_index ~33–43, corresponding to k_phys ≈ 5.2–6.7×10⁻³ Mpc⁻¹. This is precisely the region of the PPS enhancement peak at k ≈ 6×10⁻³ Mpc⁻¹.

### Resonance Hypothesis (Verified)

Running `plot_vrfcb_integerK.py` directly computes `vr^∞` for every integerK mode via the same `lstsq` procedure used in `compute_allowedK`. The output confirms:

```
k_index= 33  k=21.3672  k_phys=5.213e-03 Mpc^-1  vr^inf=-2.9960e-01
k_index= 34  k=21.9777  k_phys=5.362e-03 Mpc^-1  vr^inf= 2.4341e-01
k_index= 35  k=22.5881  k_phys=5.511e-03 Mpc^-1  vr^inf=-1.7714e-01
k_index= 36  k=23.1986  k_phys=5.660e-03 Mpc^-1  vr^inf= 1.2017e-01
k_index= 37  k=23.8091  k_phys=5.809e-03 Mpc^-1  vr^inf=-5.4361e-02
k_index= 38  k=24.4196  k_phys=5.958e-03 Mpc^-1  vr^inf=-3.1577e-03  ← near zero
k_index= 39  k=25.0301  k_phys=6.107e-03 Mpc^-1  vr^inf= 6.8488e-02
k_index= 40  k=25.6406  k_phys=6.256e-03 Mpc^-1  vr^inf=-1.2635e-01
k_index= 41  k=26.2511  k_phys=6.405e-03 Mpc^-1  vr^inf= 1.9115e-01
k_index= 42  k=26.8616  k_phys=6.554e-03 Mpc^-1  vr^inf=-2.4924e-01
k_index= 43  k=27.4721  k_phys=6.703e-03 Mpc^-1  vr^inf= 3.1350e-01
```

`vr^∞` oscillates like a sine wave across modes, passing through a node at **k_index=38** (k=24.42, k_phys=5.958×10⁻³ Mpc⁻¹) where `vr^∞ ≈ -3.2×10⁻³` — nearly zero. This is within the 5%-of-max threshold.

### Physical Interpretation

At k_index=38, the integerK mode `k = 24.42` **accidentally satisfies both boundary conditions simultaneously**:

1. `k = n√K` for integer n — spatial quantization condition met by definition
2. `vr^∞ ≈ 0` — palindrome (CPT) condition accidentally satisfied

This double coincidence means the integerK and allowedK bases assign nearly identical FCB states (`x_inf`) to this mode. Their full time-domain solutions therefore overlap with ρ → 1, producing the diagonal identity block observed in the CCA heatmap.

### Connection to the PPS Enhancement

Because the resonant integerK mode (k_index≈38, k_phys≈6×10⁻³ Mpc⁻¹) is simultaneously a valid palindromic mode, it receives the full power from both bases. Non-resonant modes at other k values must distribute their power across mixtures of basis functions. The net effect is a relative **enhancement of the primordial power spectrum** at k≈6×10⁻³ Mpc⁻¹ — the PPS peak is a direct consequence of the resonance.

This is a physically meaningful signal: the geometry of the closed CPT-symmetric universe selects out specific modes where the spatial quantization and the palindrome boundary condition coincide, imprinting a characteristic scale on the PPS.

### Comparison with Cosmological Characteristic Scales

Computed numerically using CLASS and analytic integration (`compute_characteristic_scales.py`) with the bestfit parameters:

| Scale | Value | Relation to k_res |
|-------|-------|-------------------|
| r_s(z_rec) | 144.0 Mpc | — |
| k_s1 = π/r_s(z_rec) | 2.18×10⁻² Mpc⁻¹ | k_res / k_s1 = **0.27** |
| r_s(z_drag) | 146.6 Mpc | — |
| k_drag = π/r_s(z_drag) | 2.14×10⁻² Mpc⁻¹ | k_res / k_drag = **0.28** |
| k_eq (matter-radiation equality) | 1.05×10⁻² Mpc⁻¹ | k_res / k_eq = **0.57** |
| **k_res** (vr^∞ node, verified) | **5.958×10⁻³ Mpc⁻¹** | — |
| l at k_res (CMB multipole) | **l ≈ 82** | between Sachs-Wolfe plateau and first peak |

Key observations:
- k_res is **not** at the first acoustic peak (l≈220, k≈0.022 Mpc⁻¹), but at **l≈82** — in the plateau-to-first-peak transition region
- k_res sits at **~1/4 of k_s1** and **~1/2 of k_eq**, suggesting a sub-horizon but pre-equality oscillation resonance
- z_rec (CLASS) = 1088.9,  z_eq = 3436,  a0 (present scale factor in internal units) = 8.08

Gemini's suggestion that the resonance is related to sound horizon crossing at matter-radiation equality (`k ≈ 1/r_s(t_eq)`) remains the leading physical hypothesis. The precise relationship `k_res ≈ 0.57 k_eq` and the verification step 3 (vary Ω_m, Ω_r to see if the resonance shifts) are the next tests to confirm this.

---

---

## Gemini's Full Analysis (consulted during investigation)

> *The following is Gemini's verbatim response when asked to explain the ρ<1 puzzle and the ρ→1 resonance block.*

### Summary of the Proposed Explanation

The most likely explanation is **physical, not numerical**. The identity block with ρ → 1 corresponds to a **resonant scale** in your cosmology. At this specific range of wavenumbers (k ≈ 6×10⁻³ Mpc⁻¹), the natural, unconstrained time evolution of the standard `integerK` modes **coincidentally satisfies** the palindromic boundary condition `vr^∞ ≈ 0`. For other `k`, this condition is not naturally met, hence the two bases differ and ρ < 1. The peak in the PPS is the physical consequence of this resonance, representing modes that are "doubly-quantized" or maximally self-consistent within your model's framework.

---

### 1. Why do specific modes have ρ → 1? What makes vr^∞ ≈ 0 for these specific integerK modes?

Your initial explanation for ρ < 1 is likely correct: the `allowedK` and `integerK` bases are constructed with different boundary conditions at the FCB, leading to different space-time evolutions and thus imperfect correlation.

The existence of a ρ → 1 block strongly suggests that for this specific range of `k` values, this difference vanishes.

**Physical Reasoning:**

The evolution of cosmological perturbations is governed by a set of coupled differential equations. The final value of `vr` at the FCB, `vr^∞(k)`, is a function of the entire cosmic history integrated for that specific mode `k`. It's plausible that there exists a special `k` or a narrow range of `k` for which the solution to the evolution equations *naturally* has a node (a zero) at the FCB time, `t_FCB`.

Think of it like a vibrating string fixed at one end (`t=0`, Big Bang) and "whipped" in a complex way (by the evolving background spacetime). You are comparing two scenarios:
1. **allowedK**: You only select the whipping frequencies (`k`) that result in the string being perfectly still at the other end (`t=t_FCB`).
2. **integerK**: You select standard harmonic frequencies (`k`) and see where the other end of the string lands.

For most harmonic frequencies (`integerK`), the end is still moving (`vr^∞ ≠ 0`). However, for a special set of harmonic frequencies (`k_index ~30-40`), the dynamics of the whip are such that the end of the string *just happens* to come to rest (`vr^∞ ≈ 0`).

**What could create such a "resonant" scale?**

- **Sound Horizon Crossing:** The most likely candidate is a relationship between the mode's wavenumber `k` and a characteristic physical scale of the universe, such as the sound horizon (`r_s`) at a key epoch like matter-radiation equality or recombination. The dynamics of perturbations change dramatically when they enter the sound horizon. It is conceivable that for modes with `k ≈ 1/r_s(t_eq)`, the subsequent oscillations are phased in just the right way to produce a velocity node at the FCB. **The peak you see at `k ≈ 6×10⁻³ Mpc⁻¹` is in the same broad range as the first CMB acoustic peak scale** (k_s1 = 2.18×10⁻² Mpc⁻¹, l≈220), though numerical calculation (see "Comparison with Cosmological Characteristic Scales" below) places k_res more precisely at **l≈82** — the plateau-to-first-peak transition — at about 1/4 of k_s1 and 1/2 of k_eq. In your palindromic model, the physics near the crunch might mirror the physics near the bang, setting up this resonance.

- **Effective Potential:** The perturbation equations can often be cast into a Schrödinger-like form: `u'' + (k² - V(η))u = 0`, where `V(η)` is an effective potential dependent on the background evolution. A specific `k` might correspond to a resonance with a feature (a well or barrier) in this potential, leading to a specific phase for the solution at the FCB.

**Conclusion for Q1:** The `integerK` modes in the `k_index ~30-40` range don't have `vr^∞ = 0` imposed on them. Rather, their natural evolution within your cosmological model results in `vr^∞ ≈ 0` anyway. For these modes, the two basis definitions become physically equivalent, leading to ρ → 1.

---

### 2. Is the PPS peak physically meaningful or a numerical artefact?

Given the reasoning above, the peak is very likely **physically meaningful**. It highlights the modes that are most "at home" in your universe, satisfying both the spatial quantization from its closed geometry and the temporal CPT symmetry from its palindromic nature.

**Why it's likely NOT a numerical artefact:**

- **CCA Regularisation** tends to *reduce* correlations slightly to ensure numerical stability. It is extremely unlikely to spontaneously create a perfect identity block where one doesn't exist.
- **Gram Matrix Structure:** The Gram matrix captures the inner product of the mode functions over their entire space-time history. For ρ to be 1, the modes must be functionally identical at *all times*, not just at the boundary. This is a very strong condition that is hard to produce by accident.
- **Sparse Rotation:** Methods like Varimax are applied *after* CCA. If the CCA finds a block where canonical correlations are already `(1, 1, ..., 1)` and coefficient matrices are already close to identity, the rotation simply "snaps" them to a perfect identity matrix. It reveals the underlying structure; it doesn't create it.

**Physical Interpretation of the Peak:**

These "doubly-quantized" modes are the natural resonant modes of your spacetime. It is plausible that the mechanism generating primordial perturbations in your model would preferentially excite these modes, leading to a peak in the PPS at that specific scale. This peak is a direct, potentially observable prediction of your model.

---

### 3. Unified Explanation (addendum, not a revision)

The ρ < 1 explanation is correct as a description of the *general case*; the ρ → 1 block is a *special case*:

| Case | What happens | ρ |
|------|-------------|---|
| Generic k | Palindrome and integer quantisation are independent; `integerK` has `vr^∞ ≠ 0` | < 1 |
| Resonant k_res | Background dynamics force `vr^∞(k_res) ≈ 0` naturally; both conditions met simultaneously | → 1 |

At the resonant scale, if integer `n` happens to give `n√K ≈ k_res`, then the `integerK` mode *is* the `allowedK` mode — the same physical solution, identical Gram matrices, CCA correctly reports ρ → 1.

---

### 4. Verification Steps (Gemini's suggestions)

1. **Plot `vr^∞(k)` for the `integerK` basis** — the most crucial test. The plot should show a significant dip approaching zero precisely at k_index ~30-40, and be non-zero elsewhere. *(Verified — see below.)*
2. **Analyze the effective potential `V(η)`** in the perturbation equations for a feature at k ≈ 6×10⁻³ Mpc⁻¹.
3. **Vary cosmological parameters** (Ω_m, Ω_r): if the resonance is tied to the sound horizon at matter-radiation equality, changing these parameters should **move the ρ → 1 block and PPS peak** predictably. If numerical, the block would stay at the same k_index regardless of physics.

---

## Diagnostic Check

To quantify the mismatch directly, print `x_inf[2]` (= vr^∞) for matched allowedK/integerK pairs at high k:

- For allowedK: `x_inf[2] ≈ 0` (palindrome condition satisfied)
- For integerK: `x_inf[2] ≠ 0` in general, but `x_inf[2] ≈ 0` near the resonance (k_index~38)

The magnitude of `x_inf[2]` for the integerK modes gives a direct measure of how far each integerK mode is from being a valid palindromic solution, which in turn sets the floor on (1 − ρ). The zero-crossing of `vr^∞(k)` for the integerK basis directly predicts the location of the ρ → 1 block and the PPS enhancement scale.

Script: `plot_vrfcb_integerK.py` — produces `./figures/vrfcb_integerK.pdf` and `./figures/xfcb_all_integerK.pdf`.

---

## Numerical Verification: k-convergence and Solution Comparison (`compare_perturbations_soln.py`)

Script: `compare_perturbations_soln.py` — produces four figures in `./figures/`.

### Part 1 — k-values do converge at high k (`compare_k_convergence.pdf`)

Both bases were generated from the same ordered integer-n sequence, so k_allowedK[i] and k_integerK[i] correspond to the same mode number n. The absolute and relative k-differences across all 105 modes are:

| k range | Δk/k (typical) |
|---------|----------------|
| Low k (first 5 modes) | 7–27% |
| High k (last 5 modes) | ~0.25% |

The relative difference shrinks by a factor of ~52 from the lowest to the highest modes, confirming that k_allowedK → k_integerK at high k.

Numerical output for the last 10 modes:

```
 idx       k_aK       k_iK       |Δk|     |Δk|/k       vr∞_aK       vr∞_iK
  95   59.06506   59.21757    0.15251    0.2579%  -9.03e-05   2.7355e+00
  96   59.67489   59.82806    0.15317    0.2564%  -3.53e-05  -2.7585e+00
  97   60.28465   60.43855    0.15390    0.2550%   5.66e-04   2.7816e+00
  98   60.89447   61.04904    0.15457    0.2535%  -3.67e-04  -2.8026e+00
  99   61.50422   61.65953    0.15531    0.2522%   7.61e-04   2.8235e+00
 100   62.11402   62.27002    0.15600    0.2508%  -2.43e-04  -2.8425e+00
 101   62.72377   62.88051    0.15675    0.2496%   3.71e-04   2.8614e+00
 102   63.33354   63.49100    0.15747    0.2483%  -1.64e-04  -2.8783e+00
 103   63.94326   64.10149    0.15823    0.2472%   4.54e-04   2.8952e+00
 104   64.55301   64.71198    0.15898    0.2460%  -5.92e-04  -2.9101e+00
```

Despite Δk/k < 0.26%, the vr^∞ values differ by a factor of ~10,000:

| Basis | |vr^∞| (high-k modes) |
|-------|------------------------|
| allowedK | max ≈ 5.25×10⁻² , mean ≈ 8.3×10⁻⁴ (≈ 0 by construction) |
| integerK | max ≈ 6.29, mean ≈ 1.54 (large, oscillating with alternating sign) |

### Part 2 — Even a 0.25% k-shift is enough to break the palindrome (`compare_perturbations_soln_highk.pdf`, `compare_vr_near_FCB.pdf`, `compare_perturbations_overlay.pdf`)

The full perturbation solutions (dr, dm, vr, vm) were reconstructed from recombination to the FCB for the last 5 high-k modes of each basis using:

```
x_inf  = lstsq( (A@X1 + D@X2 + GX3)[2:6, :],  recs[2:6] )
Y(η)   = einsum('ijt,j->it', U_ABC(η), X1@x_inf)
       + einsum('ijt,j->it', U_DEF(η), X2@x_inf)
```

**Key observation**: at high k, both bases oscillate with nearly identical frequency (Δk/k ≈ 0.25%), but the accumulation of many oscillation cycles between recombination and the FCB amplifies the tiny k-difference into a large phase error at the FCB. For allowedK modes, vr → 0 at the FCB (palindrome condition satisfied). For integerK modes at the same mode index, vr arrives at the FCB with amplitude |vr^∞| ≈ 2.9 — comparable to the mid-evolution oscillation amplitude.

This is visible in the near-FCB zoom (`compare_vr_near_FCB.pdf`): allowedK curves (solid blue) converge smoothly to zero at the FCB, while integerK curves (dashed red) oscillate with full amplitude right up to the bounce. The integerK solutions are **not** symmetric or anti-symmetric at the end of the universe — the CPT palindrome condition is violated.

**Why the factor of ~10,000 in vr^∞?** At high k, the mode completes many oscillations between recombination and the FCB. The phase accumulated is φ ≈ k·(η_FCB − η_rec). A shift of Δk shifts the final phase by Δφ ≈ Δk·(η_FCB − η_rec). Because Δk is small but (η_FCB − η_rec) is large, Δφ can be an arbitrary fraction of 2π, placing the integerK solution at a completely different phase at the FCB relative to the allowedK solution — hence vr^∞ ≠ 0 for integerK despite Δk/k being tiny.
