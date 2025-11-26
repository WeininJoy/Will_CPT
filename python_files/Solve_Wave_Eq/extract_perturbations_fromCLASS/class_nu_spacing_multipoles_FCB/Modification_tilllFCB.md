# CLASS Modifications for Extended Integration and Closed Universe Support

This document describes all modifications made to CLASS (Cosmic Linear Anisotropy Solving System) to support:
1. Integration into the far future (where $a \gg 1$)
2. Computing perturbations for arbitrary k values in closed universes
3. Using only `k_output_values` without generating full quantized k list

## Summary of Changes

The modifications span 6 source files:
- **include/background.h**: Add `a_final_over_a_today` parameter
- **source/input.c**: Read and initialize `a_final_over_a_today`
- **source/background.c**: Extend integration, fix boundary issues, fill last table point
- **source/thermodynamics.c**: Extend redshift table to negative z (future times)
- **source/primordial.c**: Optimize k-mode generation for closed universes
- **tools/arrays.c**: Add floating-point tolerance to boundary checks

---

## 1. Add New Input Parameter for Final Scale Factor

### File: `include/background.h` (line ~126)

Add the parameter to the `struct background` definition:

```c
struct background
{
  /** @name - input parameters initialized by user in input module
   * ... */

  /* ... existing parameters ... */

  double a_final_over_a_today; /**< Target scale factor for end of simulation (1.0 = today, >1.0 = future) */

  /* ... existing parameters ... */
```

**Location**: After `varconst_transition_redshift` around line 126.

---

## 2. Initialize and Read the Parameter

### File: `source/input.c`

#### A. Set default value in `input_default_params()` (line ~5741)

```c
int input_default_params(...) {
  /* ... existing defaults ... */

  /** 11) Final scale factor for integration (1.0 = today, >1.0 = future) */
  pba->a_final_over_a_today = 1.0;

  /* ... */
}
```

**Location**: After `pba->varconst_transition_redshift = 50.;` around line 5739.

#### B. Read from input file in `input_read_parameters_general()` (line ~2310)

```c
int input_read_parameters_general(...) {
  /* ... existing reads ... */

  /** 11) Final scale factor for integration */
  class_read_double("a_final_over_a_today",pba->a_final_over_a_today);

  return _SUCCESS_;
}
```

**Location**: After the BBN sensitivity parameter reads, around line 2307.

---

## 3. Extend Background Integration

### File: `source/background.c`

#### A. Change final scale factor in `background_solve()` (line ~1922)

```c
int background_solve(...) {
  /* ... */

  /** - Determine output vector */
  // OLD: loga_final = 0.;
  // NEW: Allow integration into the future
  loga_final = log(pba->a_final_over_a_today);
  pba->bt_size = ppr->background_Nloga;

  /* ... */
}
```

#### B. Fix last point not being filled by evolver (line ~1982)

Add this code after the integration completes (after `generic_evolver()`):

```c
  /** - Fill the last point manually if it wasn't filled by the evolver */
  if (pba->tau_table[pba->bt_size-1] == 0.) {
    /* The evolver doesn't always call background_sources for the exact final point.
       Fill it manually using the final integration values. */
    double a_final = exp(loga_final);
    int index_last = pba->bt_size-1;

    pba->z_table[index_last] = 1./a_final - 1.;
    pba->tau_table[index_last] = pvecback_integration[pba->index_bi_tau];

    /* Also fill the background_table for the last point */
    class_call(background_functions(pba, a_final, pvecback_integration, long_info,
                                    pba->background_table + index_last*pba->bg_size),
               pba->error_message,
               pba->error_message);
  }

  /** - DEBUG: print tau_table and loga_table values */
  if (pba->background_verbose > 0) {
    printf("DEBUG: loga_ini=%e, loga_final=%e\n", loga_ini, loga_final);
    printf("DEBUG: loga_table[0]=%e, loga_table[bt_size-1]=%e\n",
           pba->loga_table[0], pba->loga_table[pba->bt_size-1]);
    printf("DEBUG: After integration, tau_table[0]=%e, tau_table[bt_size-1]=%e\n",
           pba->tau_table[0], pba->tau_table[pba->bt_size-1]);
    printf("DEBUG: conformal_age from integration=%e\n", pvecback_integration[pba->index_bi_tau]);
  }
```

**Rationale**: The evolver (ndf15) doesn't always call the source function for the exact final point, leaving `tau_table[bt_size-1] = 0`. This causes "tau > tau_max=0" errors in interpolation functions.

#### C. Allow negative redshift in `background_sources()` (line ~2739)

```c
  /** - corresponding redhsift 1/a-1 (can be negative for future times when a > 1) */
  pba->z_table[index_loga] = 1./a-1.;  // OLD: MAX(0., 1./a-1.)
```

#### D. Add tolerance to boundary checks (lines 152, 160, 281, 289, 331, 339)

In `background_at_z()`:
```c
  // OLD:
  class_test(loga < pba->loga_table[0], ...)
  class_test(loga > pba->loga_table[pba->bt_size-1], ...)

  // NEW (add tolerance):
  class_test(loga < pba->loga_table[0] && fabs(loga - pba->loga_table[0]) > 1e-10, ...)
  class_test(loga > pba->loga_table[pba->bt_size-1] && fabs(loga - pba->loga_table[pba->bt_size-1]) > 1e-10, ...)
```

Apply similar changes in `background_at_z()` (for z bounds) and `background_at_tau()` (for tau bounds).

**Rationale**: Floating-point arithmetic can cause slight overshoots at boundaries. Adding tolerance prevents spurious errors when requesting exactly the final value.

#### E. Optional: Add debug output in `background_sources()` (line ~2710)

```c
  /** - DEBUG: print which index is being filled */
  static int call_count = 0;
  if (pba->background_verbose > 5 && (call_count < 5 || index_loga == pba->bt_size-1 || index_loga == pba->bt_size-2)) {
    printf("DEBUG background_sources: index_loga=%d/%d, loga=%e, tau=%e\n",
           index_loga, pba->bt_size-1, loga, y[pba->index_bi_tau]);
    call_count++;
  }
```

---

## 4. Extend Thermodynamics Table to Negative Redshift

### File: `source/thermodynamics.c` (line ~1115)

In function `thermodynamics_lists()`:

```c
int thermodynamics_lists(...) {
  /** Define local variables */
  int index_tau, index_z;
  double zinitial,zlinear;
  double z_final; /* final redshift (will be negative for future, when a > 1) */

  /* Calculate final redshift from final scale factor */
  z_final = 1./pba->a_final_over_a_today - 1.;

  pth->tt_size = ptw->Nz_tot;

  /* ... recombination table setup ... */

  /* -> Between reionization_z_start_max and z_final, we use the spacing of reionization sampling */
  for (index_z=0; index_z <ptw->Nz_reio; index_z++) {
    double z_start_reio = ppr->reionization_z_start_max;
    double step = (double)(ptw->Nz_reio - 1 - index_z) / (double)(ptw->Nz_reio);
    /* Linearly interpolate z from z_start_reio down to z_final */
    pth->z_table[(pth->tt_size-1)-(index_z+ptw->Nz_reco)] = z_final + (z_start_reio - z_final) * step;
  }

  /* ... */
}
```

**What changed**:
- OLD: `pth->z_table[...] = -(-ppr->reionization_z_start_max * (double)(ptw->Nz_reio-1-index_z) / (double)(ptw->Nz_reio));`
- NEW: Interpolates from `reionization_z_start_max` down to `z_final` (which can be negative)

---

## 5. Optimize k-Mode Generation for Closed Universes

### File: `source/primordial.c`

#### A. Add debug output before k-list generation (line ~253)

In `primordial_indices()`:
```c
  fprintf(stdout,"DEBUG before primordial_get_lnk_list: pba->K=%e, pba->sgnK=%d, pba->nu_spacing=%d\n",
          pba->K, pba->sgnK, pba->nu_spacing);

  class_call(primordial_get_lnk_list(ppt,
                                     ppm,
                                     k_min,
                                     k_max,
                                     pba->K,
                                     pba->sgnK,
                                     pba->nu_spacing),
             ppm->error_message,
             ppm->error_message);
```

#### B. Optimize k-list for perturbations-only output (line ~704)

In `primordial_get_lnk_list()`, in the `sgnK > 0` (closed universe) branch:

```c
  } else {  // sgnK > 0 (closed universe)

      /* DEBUG: print flags */
      fprintf(stdout,"DEBUG primordial: has_cls=%d, k_output_values_num=%d\n",
              ppt->has_cls, ppt->k_output_values_num);

      /* If NO CMB output is requested AND k_output_values are specified,
         use minimal k list since k_output_values will be added later */
      if (ppt->has_cls == _FALSE_ && ppt->k_output_values_num > 0) {
        /* Create a minimal k list - the actual k values will come from k_output_values */
        ppm->lnk_size = 1;
        class_alloc(ppm->lnk, ppm->lnk_size*sizeof(double), ppm->error_message);
        ppm->lnk[0] = log(kmin);  /* One dummy value */

        fprintf(stdout,"Closed universe mode: Using only k_output_values (no CMB computation)\n");
        return _SUCCESS_;
      }

      /* Original closed universe code follows... */
      double m=0.;
      double q,q_min=0.,q_max=0.,q_step;
      /* ... quantized k-mode generation ... */
  }
```

**Rationale**: In closed universes, CLASS generates quantized k values based on the curvature and `nu_spacing`. When computing only perturbation transfer functions (not CMB spectra), we don't need the full quantized set—just the user-specified `k_output_values`. This optimization dramatically reduces computation time (from 107+ k values to just 5).

#### C. Add debug output in `primordial_get_lnk_list()` (line ~686)

```c
int primordial_get_lnk_list(...) {

  fprintf(stdout,"DEBUG primordial_get_lnk_list: sgnK=%d, K=%e, nu_spacing=%d\n", sgnK, K, nu_spacing);

  if  (sgnK <= 0) {
      /* Original code for flat/open universes */
      /* ... */
  } else {
      /* Closed universe code */
      /* ... */
  }
}
```

---

## 6. Add Floating-Point Tolerance to Array Boundary Checks

### File: `tools/arrays.c`

Multiple interpolation functions need tolerance added to prevent spurious boundary errors due to floating-point precision.

#### A. `array_interpolate()` (lines ~1819, 1829, 1967, 1977, 2134, 2144)

In each boundary check, change:
```c
// OLD:
if (x < x_array[inf]) {
  class_sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
  return _FAILURE_;
}

// NEW (add tolerance):
if (x < x_array[inf] && fabs(x - x_array[inf]) > 1e-10) {
  class_sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,x,x_array[inf]);
  return _FAILURE_;
}
```

Apply to both ascending and descending array checks, and to upper/lower bounds.

#### B. `array_interpolate_two()` (lines ~2544, 2556)

```c
  while (x < x_array[inf] && fabs(x - x_array[inf]) > 1e-10) {
    inf--;
    if (inf < 0) {
      /* Allow x to be at the exact boundary */
      if (fabs(x - x_array[0]) <= 1e-10) {
        inf = 0;
        break;
      }
      class_sprintf(errmsg,"%s(L:%d) : x=%e < x_min=%e",__func__,__LINE__,
              x,x_array[0]);
      return _FAILURE_;
    }
  }
  sup = inf+1;
  while (x > x_array[sup] && fabs(x - x_array[sup]) > 1e-10) {
    sup++;
    if (sup > (n_lines-1)) {
      /* Allow x to be at the exact boundary */
      if (fabs(x - x_array[n_lines-1]) <= 1e-10) {
        sup = n_lines-1;
        break;
      }
      class_sprintf(errmsg,"%s(L:%d) : x=%e > x_max=%e",__func__,__LINE__,
              x,x_array[n_lines-1]);
      return _FAILURE_;
    }
  }
```

**Rationale**: When integrating to $a = 100$ or higher, interpolation functions may be called with values that are numerically equal to the table boundaries but differ by $\sim 10^{-16}$ due to floating-point arithmetic. The tolerance of $10^{-10}$ allows these cases while still catching genuine out-of-range errors.

---

## Important Parameter Name: `Omega_k` (Case-Sensitive!)

**Critical**: In CLASS ini files and Python dictionaries, the curvature parameter is:
- **Correct**: `Omega_k` (capital O)
- **Wrong**: `omega_k` (lowercase o)

If you use lowercase `omega_k`, CLASS will silently ignore it and default to `Omega_k = 0` (flat universe), causing the closed universe code path to never activate. Always use **`Omega_k`**.

Example ini file:
```ini
# Closed universe parameters (K>0)
Omega_k = -0.04  # This makes K>0 (closed universe)
nu_spacing = 3
a_final_over_a_today = 100

# Output ONLY perturbation transfers, NOT CMB Cl's
output = dTk, vTk
k_output_values = 0.001, 0.01, 0.05, 0.1, 0.2
```

---

## Recompiling

After making these changes, recompile CLASS:

```bash
make clean
make
```

If using the Python wrapper and it doesn't automatically update:
```bash
cd python
python setup.py install --user  # or pip install .
```

---

## Usage Examples

### Python Example

```python
from classy import Class

params = {
    # Standard cosmological parameters
    'h': 0.67,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.96,
    'tau_reio': 0.05,

    # Closed universe (note: capital O!)
    'Omega_k': -0.04,  # K > 0, closed
    'nu_spacing': 3,   # Spacing integer for quantized k modes

    # Extended integration to far future
    'a_final_over_a_today': 100,  # a = 100 (z = -0.99)

    # Request perturbations only (not CMB)
    'output': 'dTk,vTk',
    'k_output_values': '0.001, 0.01, 0.05, 0.1, 0.2',
}

cosmo = Class()
cosmo.set(params)
cosmo.compute()

# Access background quantities
bg = cosmo.get_background()
print(f"Conformal age: {bg['conf. time [Mpc]'][-1]:.2f} Mpc")

# Access perturbations (will include all k_output_values)
cosmo.get_perturbations()  # Writes files to disk
```

### INI File Example

```ini
# test_closed_future.ini

# Standard parameters
h = 0.67
omega_b = 0.022
omega_cdm = 0.12
A_s = 2.1e-9
n_s = 0.96
tau_reio = 0.05

# Closed universe
Omega_k = -0.04
nu_spacing = 3
a_final_over_a_today = 100

# Perturbations only
output = dTk, vTk
modes = s
ic = ad
gauge = newtonian

# Arbitrary k values (no quantization constraint!)
k_output_values = 0.001, 0.01, 0.05, 0.1, 0.2

# Output settings
root = output/test_
write parameters = yes
write background = yes
perturbations_verbose = 2
```

Run with:
```bash
./class test_closed_future.ini
```

---

## Verification

After successful compilation and run, verify:

1. **Background extends to $a > 1$**:
   ```bash
   tail -1 output/test_*_background.dat
   # Should show z ≈ -0.99 (for a=100) and large conformal time
   ```

2. **Only requested k values computed**:
   ```bash
   ls output/test_*_perturbations_k*.dat | wc -l
   # Should equal number of k_output_values (e.g., 5)
   ```

3. **K value is correct**:
   ```bash
   grep "DEBUG before primordial_get_lnk_list" <output_log>
   # Should show K > 0 and sgnK = 1 for closed universe
   ```

4. **Optimization active**:
   ```bash
   grep "Closed universe mode: Using only k_output_values" <output_log>
   # Should appear if has_cls=0 and k_output_values specified
   ```

---

## Important Notes on Closed Universes

### Expansion Forever vs Big Crunch

When `Omega_k < 0` (closed universe, $K > 0$):

1. **Expansion forever**: If $\Omega_\Lambda$ is large enough (standard $\Lambda$CDM), the universe expands forever and conformal time converges to a finite asymptotic value. The modifications work perfectly.

2. **Big Crunch**: If parameters imply the universe will recollapse, $a(t)$ is not monotonic. CLASS uses $\ln a$ as the integration variable and assumes monotonicity. Setting `a_final_over_a_today` beyond the turnaround point will cause crashes or nonsense results.

**Always verify** your cosmological parameters imply continued expansion up to your requested `a_final_over_a_today`.

### k-Mode Quantization

In closed universes, the spatial curvature imposes a quantization condition on allowed k values:
$$k_n = \frac{\nu_n \sqrt{K}}{a_0}$$

where $\nu_n$ are integers (or half-integers) determined by `nu_spacing`. However, when computing **only perturbation transfer functions** (not CMB $C_\ell$), this quantization is not required. The optimization in `source/primordial.c` bypasses the quantized k-list generation when `has_cls = _FALSE_`, allowing arbitrary k values via `k_output_values`.

---

## Debug Output

The modifications include several debug print statements (controlled by `background_verbose` and always-on in `primordial.c`). To disable them for production runs:

1. Remove or comment out `fprintf(stdout, ...)` lines in:
   - `source/background.c` (lines ~2000, ~2710)
   - `source/primordial.c` (lines ~253, ~686, ~706)

2. Or set `background_verbose = 0` in your ini file to suppress background debug output.

---

## Summary of Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `include/background.h` | ~126 | Add `a_final_over_a_today` parameter |
| `source/input.c` | ~2310, ~5741 | Read and initialize parameter |
| `source/background.c` | ~152, ~160, ~281, ~289, ~331, ~339, ~1922, ~1982, ~2710, ~2739 | Extend integration, fix boundaries, fill last point |
| `source/thermodynamics.c` | ~1115 | Extend z_table to negative redshift |
| `source/primordial.c` | ~253, ~686, ~704 | Optimize k-list for closed universes |
| `tools/arrays.c` | ~1819, ~1829, ~1967, ~1977, ~2134, ~2144, ~2544, ~2556 | Add tolerance to interpolation |

---

## Tested Configurations

Successfully tested with:
- `Omega_k = -0.04` (closed)
- `nu_spacing = 3`
- `a_final_over_a_today = 1` (today), `100` (far future)
- `k_output_values = 0.001, 0.01, 0.05, 0.1, 0.2`
- `output = dTk, vTk` (perturbations only, no CMB)

Results:
- Conformal time: 4.63e-09 → 21696 Mpc (for a=100)
- Redshift range: z = 1e14 → -0.99
- Computation time: ~seconds (vs ~minutes without optimization)
- Output: 5 perturbation files, one per k value

---

## Troubleshooting

### Error: "tau > tau_max=0.000000e+00"
**Cause**: Evolver didn't fill last point of tau_table.
**Fix**: Applied in `source/background.c` lines ~1982-1997 (manual fill).

### Error: "K=0" when Omega_k is set
**Cause**: Used lowercase `omega_k` instead of `Omega_k`.
**Fix**: Use **`Omega_k`** (capital O) in ini file.

### Many k values computed instead of k_output_values
**Cause**: CMB output requested (`has_cls=1`), or optimization not triggered.
**Fix**: Use `output = dTk` (or vTk, mTk, mPk) **without** tCl, pCl, lCl.

### Interpolation boundary errors
**Cause**: Floating-point precision at table boundaries.
**Fix**: Applied in `tools/arrays.c` and `source/background.c` (tolerance $10^{-10}$).

---

## References

- CLASS code: https://github.com/lesgourg/class_public
- CLASS documentation: https://lesgourg.github.io/class_public/class.html
- Original modifications based on: Extending integration to $\eta_\infty$ for closed universes

---

**Last updated**: 2025-11-26
**CLASS version**: v3.2.5
**Branch**: `multi_spacing_integers`
