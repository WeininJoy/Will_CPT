# Matrix Consistency Verification Report

## Summary

✅ **VERIFIED**: The ABCDEF matrices in `/data_allowedK/` match the time series matrices in `/data_allowedK_timeseries/` at recombination time **within numerical precision**.

## Key Findings

### 1. Time Ordering Confirmed

From `Higher_Order_Finding_U_Matrices_TimeSeries_s-a.py`:
- Line 197: `t_grid = np.linspace(recConformalTime, endtime, num=num_time_points)`
- **`t_grid[0]`** = recConformalTime (η = 0.067361)
- **`t_grid[-1]`** = endtime (η = 5.104367, near FCB)

The integration proceeds **backward** from FCB to recombination:
1. `solve_ivp(dX2_dt, [endtime, swaptime], ...)` - FCB → swaptime
2. `solve_ivp(dX3_dt, [swaptime, recConformalTime], ...)` - swaptime → recombination

The stitching (line 251) concatenates solutions correctly:
```python
full_solution = np.concatenate((sol_on_a_grid, sol_on_s_grid), axis=1)
```
where `sol_on_a_grid` contains times near recombination (first) and `sol_on_s_grid` contains times near FCB (last).

### 2. Initial Conditions at FCB

At FCB (`t_grid[-1]`), the time series correctly starts from identity:
```
ABC_at_fcb[0:6, :] ≈ I₆ (identity matrix)
```
Maximum deviation from identity: **1.43×10⁻¹²** ✅

This confirms the basis vectors start with correct initial conditions:
- Basis vector 0: `[1, 0, 0, 0, 0, 0]` (phi=1, others=0)
- Basis vector 1: `[0, 1, 0, 0, 0, 0]` (psi=1, others=0)
- etc.

### 3. Comparison at Recombination

#### Absolute Differences (Initially Alarming)

Maximum absolute difference: **5.38×10³**

This initially appeared to be a huge discrepancy, but...

#### Relative Differences (The True Measure)

Maximum relative difference: **1.72×10⁻⁸** ✅

| Statistic | Value |
|-----------|-------|
| Mean relative error | 5.86×10⁻⁹ |
| Median relative error | 4.70×10⁻¹⁰ |
| 90th percentile | 1.72×10⁻⁸ |
| Maximum | 1.72×10⁻⁸ |

**No elements exceed 1×10⁻⁶ relative error threshold.**

#### Example: Basis Vector 1, dm variable

| Quantity | Value |
|----------|-------|
| Static value | 3.151083×10¹¹ |
| Timeseries value | 3.151083×10¹¹ |
| Absolute difference | 5.382×10³ |
| **Relative difference** | **1.71×10⁻⁸** |

The absolute difference looks large (5382), but it's tiny compared to the magnitude of the values (~10¹¹).

### 4. Integration Parameters

Both codes use identical integration parameters:
- Absolute tolerance: `atol = 1e-13`
- Relative tolerance: `rtol = 1e-13`
- Method: `LSODA`
- K values: Identical (verified: `np.allclose(kvalues_static, kvalues_timeseries)` ✅)

## Conclusion

### Question: Do the matrices match?

**YES.** The matrices match within **1.7×10⁻⁸ relative precision**, which is excellent given:
- Integration tolerances of 10⁻¹³
- Numerical propagation over ~70 time steps
- Double precision arithmetic (~10⁻¹⁶)

The relative error of 1.7×10⁻⁸ represents approximately **8 significant figures of accuracy**.

### Implication for Phi Mismatch Problem

**This is NOT the source of the phi mismatch issue.**

The phi constraint error (~0.065 for the first mode) is much larger than the matrix precision error (1.7×10⁻⁸). Therefore, the matrix consistency is verified, and the phi mismatch must have a different cause.

### Possible Alternative Causes for Phi Mismatch

Since the matrices match at recombination, the phi mismatch likely originates from:

1. **X1, X2 matrix approximations**: The Taylor expansion to 6th order in deltaeta may not be accurate enough
   - Recommendation: Test with higher order expansions

2. **The lstsq solution for x_inf**: Even though we proved high precision arithmetic doesn't help, the 4×4 system is ill-conditioned (κ~5×10¹⁴)
   - However, previous tests showed this is NOT the bottleneck

3. **The phi constraint itself**: The constraint (equation 14) may not be exactly satisfied by the perturbation equations during backward integration
   - The constraint is satisfied at FCB (~10⁻¹¹) but not at recombination (~0.065)
   - This 6×10⁹ amplification suggests error growth during propagation

4. **Recombination initial conditions**: The forward integration from Big Bang may have different assumptions than the backward reconstruction
   - Check whether `recValues` from forward integration satisfy the constraint

## Files Generated

1. `verify_matrix_consistency.py` - Initial verification showing "huge" absolute differences
2. `diagnose_matrix_mismatch.py` - Detailed diagnosis with time evolution plots
3. `find_which_column_differs.py` - Identification of which basis vectors differ
4. `check_relative_errors.py` - **Definitive proof that matrices match** (relative errors ~10⁻⁸)
5. `matrix_mismatch_diagnosis.pdf` - Time evolution plots

## Recommendation

**Move on from matrix consistency.** The matrices are verified to match. Focus investigation on:
1. Testing higher-order Taylor expansions for X1, X2
2. Understanding why phi constraint degrades during propagation
3. Verifying that forward integration initial conditions satisfy the constraint
