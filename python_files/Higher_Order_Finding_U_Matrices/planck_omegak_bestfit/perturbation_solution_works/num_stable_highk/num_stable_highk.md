# Numerical Stability at High-k: Diagnosis and Fix

## Context

`cal_allowedK_diff_nuspacing.py` computes `allowedK` values by:
1. Calling `compute_U_matrices` (in `Higher_Order_Finding_U_Matrices.py`) — solves ODEs
   (LSODA) for each k, building ABC/DEF/GHI/X1/X2 matrices.
2. Calling `compute_allowedK` (in `Higher_Order_Solving_for_Vrinf.py`) — evaluates
   `vrfcb(k)`, fits a cubic spline, and finds zero crossings (the allowed k values).

**Observed symptom:** Non-physical wiggles in the `allowedK` spacing at high k for
`nu_spacing > 4` (visible in `allowedK_diff_nuspacing.pdf`). The goal is to extend
the calculation to `k_max = 50` (currently limited to 22).

---

## Cause 1: Boltzmann Hierarchy Truncation (Primary)

**Location:** `Higher_Order_Finding_U_Matrices.py`, line ~500

```python
num_variables = 75  # l_max = 70
```

The photon multipole hierarchy is truncated at `l_max = 70` using the condition:

```python
lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
```

This approximates `f_{l+1} ≈ (2l+1)/(kt) · f_l`, which is only valid when `kt >> l_max`.
For this to hold at the truncation scale, we need `k · t_rec >> 70`.

- For `nu_spacing > 4`, the different cosmological parameters (`mt`, `kt`, `h`) produce
  different `fcb_time` and `recConformalTime`, changing the ratio `k · t_rec / l_max`.
  This is why wiggles appear specifically for `nu_spacing > 4`.
- For `k_max = 50`, the rule of thumb `l_max ≥ 1.5–2 × k × Δη_integration` requires
  **l_max ≈ 150 or more**. With `l_max = 70`, truncation errors are severe.

**This is the dominant source of the non-physical wiggles.**

---

## Cause 2: k-Grid Too Sparse for Root Finding (Secondary)

**Location:** `cal_allowedK_diff_nuspacing.py`, line ~128

```python
kvalues_all_k = np.linspace(1e-5, 22, num=200)  # Δk ≈ 0.11
```

At high k, the `allowedK` values are spaced approximately Δk ≈ 1 apart. With Δk_grid ≈ 0.11
there are ~9 grid points per zero-crossing interval — just sufficient for `k_max = 22`.

**Extending to `k_max = 50` with 200 points** gives Δk_grid ≈ 0.25, meaning only ~4 points
per zero-crossing interval. Some sign changes in `vrfcb` will be missed or mislocated,
causing missing or shifted `allowedK` values.

---

## Cause 3: Cubic Spline Interpolation Artefacts (Amplifier)

**Location:** `Higher_Order_Solving_for_Vrinf.py`, line ~143

```python
vr_inf_func = interp1d(k_grid, vrfcb_grid, kind='cubic', ...)
```

When `vrfcb` values carry small numerical noise from truncation errors (Cause 1), the
cubic spline introduces spurious oscillations between sample points (Runge's phenomenon).
This adds fake zero crossings or shifts real ones, compounding the wiggles.

---

## Parameters Not Responsible

- **`atol = rtol = 1e-13`**: Already very tight; ODE integration accuracy is not the issue.
- **`deltaeta` (adaptive)**: `deltaeta = min(k_deltaeta_target/k, deltaeta_max)` with
  `k_deltaeta_target = 0.005` keeps `k·deltaeta < 0.005` for all k. The Taylor expansion
  in X1/X2 is accurate.
- **`swaptime = 2`**: Hardcoded coordinate swap time. Not a significant contributor for
  the parameter sets considered here.

---

## Summary of Causes

| Cause | Impact on nu>4 wiggles | Impact for k_max=50 |
|---|---|---|
| `l_max = 70` truncation | **Primary** — `recConformalTime` differs per nu_spacing | Severe — need `l_max ≈ 150+` |
| k-grid `Δk ≈ 0.11` (200 pts to k=22) | Secondary — marginal coverage | Critical — `Δk ≈ 0.25` misses roots |
| Cubic spline artefacts | Amplifies truncation noise | Moderate |

---

## Recommended Fixes

### Fix 1: Increase Boltzmann Hierarchy Truncation (Critical)

In `Higher_Order_Finding_U_Matrices.py`:

```python
# Old
num_variables = 75  # l_max = 70

# New — safe for k_max = 50
num_variables = 205  # l_max = 200
```

Rule of thumb: `l_max ≥ 2 × k_max × Δη_integration`. For `k_max = 50` this is the
most important change. Note that increasing `num_variables` increases memory and
computation time proportionally; parallelisation over k (`n_processes`) mitigates this.

### Fix 2: Increase k-Grid Density and Range

In `cal_allowedK_diff_nuspacing.py`:

```python
# Old
kvalues_all_k = np.linspace(1e-5, 22, num=200)

# New — non-uniform: dense at low k and sufficient coverage at high k
kvalues_all_k = np.concatenate([
    np.linspace(1e-5,  3,   100),   # fine coverage at low k (non-uniform allowedK spacing)
    np.linspace(3,    50,  1000),   # Δk ≈ 0.047 → ~20 pts per zero-crossing interval
])
```

This ensures reliable sign-change detection in `vrfcb` across the full extended range.

### Fix 3: Switch to Linear Interpolation for Root Bracketing

In `Higher_Order_Solving_for_Vrinf.py`:

```python
# Old
vr_inf_func = interp1d(k_grid, vrfcb_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")

# New
vr_inf_func = interp1d(k_grid, vrfcb_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
```

With sufficient k-grid density (Fix 2), linear interpolation is safer — it cannot
introduce spurious wiggles between sample points. The `brentq` root finder provides
the precise root location within each bracket regardless of interpolation order.
