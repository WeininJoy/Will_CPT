# Adaptive Δη Implementation - Summary of Modifications

## Problem Identified

Based on `NumericalIssue.md`, the code had a **fixed** `deltaeta = 6.6e-4` for all k values. This causes numerical errors at high k because:

1. **Taylor Expansion Breakdown**: The Taylor expansion near the Future Conformal Boundary (FCB) contains terms like `k^4 * (Δη)^3`. For the expansion to be valid, we need `k * Δη ≪ 1`.
2. **Phase Error Accumulation**: High-k modes oscillate rapidly, accumulating phase errors during numerical integration.
3. **Physical Expectation**: In the WKB limit (k → ∞), allowed k values should approach equal spacing, but the fixed Δη causes them to diverge.

## Solution Implemented

### 1. Adaptive Δη (Main Fix)

**Formula**:
```
Δη(k) = min(k_Δη_target / k, Δη_max)
```

**Parameters**:
- `Δη_max = 6.6e-4`: Maximum value (maintains current behavior for low k)
- `k_Δη_target = 0.01`: Target for k·Δη product (ensures k·Δη < 0.01 for all k)

**Result**:
- For k < 15.15: uses Δη_max = 6.6e-4 (original behavior)
- For k ≥ 15.15: uses Δη = 0.01/k (adaptive)
- For k = 20: Δη = 5.0e-4 (k·Δη = 0.01)
- For k = 100: Δη = 1.0e-4 (k·Δη = 0.01)

### 2. Adaptive Tolerances (Secondary Fix)

**For high-k modes (k > 10)**:
- `atol_k = atol / 10 = 1e-14`
- `rtol_k = rtol / 10 = 1e-14`

This reduces phase error accumulation during numerical integration of rapidly oscillating solutions.

## Files Modified

### 1. `Higher_Order_Finding_U_Matrices.py`

**Changes**:
- Added `deltaeta_max` and `k_deltaeta_target` parameters
- Modified `process_single_k()` to compute adaptive Δη based on k
- Added adaptive tolerance calculation for k > 10
- Updated all `solve_ivp` calls to use `atol_k` and `rtol_k`
- Added diagnostic print statement showing k, Δη, and k·Δη
- Updated worker function to pass new parameters

**Key Code Sections**:
```python
# Lines 26-37: Adaptive Δη and tolerances computation
deltaeta = min(k_deltaeta_target / k, deltaeta_max)

if k > 10:
    atol_k = atol / 10
    rtol_k = rtol / 10
else:
    atol_k = atol
    rtol_k = rtol
```

### 2. `Higher_Order_Finding_U_Matrices_TimeSeries.py`

**Changes**:
- Added `deltaeta_max` and `k_deltaeta_target` parameters
- Computed conservative `endtime` using minimum k value (for common time grid)
- Added k-specific Δη calculation inside the main loop
- Updated diagnostic print to show Δη for each k

**Key Code Sections**:
```python
# Lines 194-198: Conservative endtime for time grid
k_min = np.min(kvalues)
deltaeta_for_grid = min(k_deltaeta_target / k_min, deltaeta_max)
endtime = fcb_time - deltaeta_for_grid

# Lines 219-220: K-specific Δη for boundary conditions
deltaeta = min(k_deltaeta_target / k, deltaeta_max)
```

## Expected Improvements

1. **High-k Accuracy**: The adaptive Δη ensures the Taylor expansion remains valid even for large k values (k > 20).

2. **Equal Spacing**: As k increases, the allowed k values should approach equal spacing (consistent with WKB limit), rather than showing oscillating deviations.

3. **Numerical Stability**: Tighter tolerances for high-k modes reduce phase error accumulation.

4. **Backward Compatibility**: For k < 15, behavior is identical to the original code.

## Testing

To verify the improvements:

1. Run `generate_data.py` to regenerate data with adaptive Δη
2. Compare new `data_allowedK/L70_kvalues.npy` with `data_integerK/L70_kvalues.npy`
3. The difference should decrease for high k (instead of increasing as before)

Expected result: The plot of allowedK vs integerK should show convergence at high k, with differences remaining small and not oscillating with increasing amplitude.

## References

- `NumericalIssue.md`: Original analysis identifying the problem
- Paper 2 (PhysRevD.113.023546): Section III.B and Appendix A describe the Taylor expansion method
