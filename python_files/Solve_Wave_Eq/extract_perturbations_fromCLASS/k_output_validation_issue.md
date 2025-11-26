# K_OUTPUT_VALUES Issue in Closed Universe (OmegaK < 0)

## Problem Summary

When using CLASS with a closed universe (OmegaK < 0, or K > 0), arbitrary `k_output_values` are NOT validated against the quantization condition required for closed universes. This causes the code to fail or produce incorrect results.

## Root Cause

### 1. How k-values are generated for closed universes

In `source/primordial.c` (lines 720-755), CLASS correctly generates quantized k values for closed universes:

```c
// For closed universe (sgnK > 0):
nu_min = 3;
q_min = nu_min * sqrt(K);
q_step = nu_spacing * sqrt(K);  // Line 731

// Loop to generate k values
while (index_k < k_size_max) {
    q = nu*sqrt(K);              // Line 749
    nu = nu + nu_spacing;        // Line 750 - increment by nu_spacing
    k = sqrt(q*q - K*(m+1.));    // Line 751
    ppm->lnk[index_k] = log(k);
    index_k++;
}
```

**Key quantization**: ν = k/√K, where ν must be an integer multiple of `nu_spacing`

### 2. How k_output_values are handled

In `source/input.c` (lines 5445-5463), user-provided k_output_values are simply read and sorted:

```c
class_call(parser_read_list_of_doubles(pfc,"k_output_values",&int1,&pointer1,&flag1,errmsg), ...);
if (flag1 == _TRUE_) {
    ppt->k_output_values_num = int1;
    for (i=0; i<int1; i++) {
        ppt->k_output_values[i] = pointer1[i];  // No validation!
    }
    qsort(ppt->k_output_values, ppt->k_output_values_num, sizeof(double), compare_doubles);
}
```

**No validation is performed** to check if these k values satisfy the quantization condition.

### 3. How k_output_values are added to the k list

In `source/perturbations.c` (lines 2558-2616), the k_output_values are directly merged into the existing k array:

```c
if (ppt->k_output_values_num > 0) {
    for (index_mode=0; index_mode<ppt->md_size; index_mode++){
        newk_size = ppt->k_size[index_mode]+ppt->k_output_values_num;
        // ... merge k_output_values into ppt->k[index_mode] ...
        // No validation against quantization condition!
    }
}
```

## Why This Causes Problems

For closed universes, the hyperspherical Bessel functions (in `tools/hyperspherical.c`) are computed with the assumption that:

- β (beta) = ν is an integer multiple of nu_spacing
- sqrtK[l] = sqrt(β² - l²) for K=1 (line 110)

When arbitrary k values are used that don't satisfy the quantization, the mathematical structure breaks down because:
1. The eigenfunctions are only defined for integer ν values
2. The boundary conditions at the spatial "horizon" are violated
3. The orthogonality relations between different modes break down

## Solution Required

Add validation in `source/input.c` after reading k_output_values:

```c
if (flag1 == _TRUE_) {
    ppt->k_output_values_num = int1;
    for (i=0; i<int1; i++) {
        ppt->k_output_values[i] = pointer1[i];

        // Add validation for closed universe
        if (pba->sgnK > 0) {
            double q = sqrt(ppt->k_output_values[i]*ppt->k_output_values[i] + pba->K);
            double nu = q / sqrt(pba->K);
            double nu_rounded = round(nu / pba->nu_spacing) * pba->nu_spacing;

            // Check if nu is close to an allowed value
            if (fabs(nu - nu_rounded) > 1e-6) {
                class_stop(errmsg,
                    "k_output_values[%d] = %e not allowed for closed universe.\n"
                    "For OmegaK < 0, k values must satisfy: nu = k/sqrt(K) = integer multiple of nu_spacing.\n"
                    "Computed nu = %e, nearest allowed nu = %e (nu_spacing = %d)",
                    i, ppt->k_output_values[i], nu, nu_rounded, pba->nu_spacing);
            }
        }
    }
    // ... rest of code ...
}
```

## Workaround

Until the code is fixed, you must manually ensure your k_output_values satisfy:

1. Calculate: q = √(k² + K)
2. Calculate: ν = q/√K
3. Verify: ν is an integer multiple of nu_spacing (e.g., 3, 6, 9, ... if nu_spacing=3)

## Files Involved

- `source/primordial.c`: Lines 720-755 (k generation for closed universes)
- `source/input.c`: Lines 5445-5463 (k_output_values reading - **needs validation**)
- `source/perturbations.c`: Lines 2558-2616 (k_output_values insertion)
- `tools/hyperspherical.c`: Hyperspherical Bessel function computation
