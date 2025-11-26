# Solution: Extract Perturbations with Arbitrary k in Closed Universe

## Summary

**Problem**: CLASS fails when using arbitrary `k_output_values` with closed universes (OmegaK < 0) if you request CMB power spectrum output (`tCl`, `pCl`, etc.).

**Root Cause**: CMB C_ℓ computation requires hyperspherical Bessel functions, which are only defined for quantized k values: ν = k/√K must be an integer multiple of `nu_spacing`.

**Solution**: Request **only perturbation/transfer outputs**, NOT CMB C_ℓ outputs. This bypasses the harmonic/hyperspherical computation entirely.

## ✓ Correct Configuration (Works with Arbitrary k)

```ini
# Configuration for extracting perturbations in closed universe with ARBITRARY k values

h = 0.54
omega_b = 0.022032
omega_cdm = 0.12038
tau_reio = 0.05430
ln10^{10}A_s = 3.0448
n_s = 0.96605

omega_k = -0.04  # Closed universe (K>0)
nu_spacing = 3

# ✓ CORRECT: Request ONLY transfer/perturbation outputs
output = dTk, vTk

modes = s
ic = ad
gauge = newtonian

# ✓ These k values do NOT need to satisfy quantization!
k_output_values = 0.001, 0.01, 0.05, 0.1, 0.2

root = output/test_perturbations_closed_
write parameters = yes
write background = yes
```

### Output Files Generated

With `k_output_values = 0.001, 0.01, 0.05, 0.1, 0.2`, you'll get:

```
output/test_perturbations_closed_perturbations_k0_s.dat  # k = 0.001
output/test_perturbations_closed_perturbations_k1_s.dat  # k = 0.01
output/test_perturbations_closed_perturbations_k2_s.dat  # k = 0.05
output/test_perturbations_closed_perturbations_k3_s.dat  # k = 0.1
output/test_perturbations_closed_perturbations_k4_s.dat  # k = 0.2
```

Each file contains columns for: conformal time τ, proper time t, scale factor a, and perturbation variables (δ, θ, Φ, Ψ, etc.) for that specific k value.

## ✗ Incorrect Configuration (Fails with Arbitrary k)

```ini
omega_k = -0.04
nu_spacing = 3

# ✗ WRONG: Requesting CMB C_ℓ output requires quantized k
output = tCl

k_output_values = 0.001, 0.01, 0.05, 0.1, 0.2  # These will fail!
```

This fails because:
1. `output = tCl` requests CMB temperature power spectrum
2. CMB C_ℓ requires integrating perturbations with hyperspherical Bessel functions
3. Hyperspherical Bessel functions require ν = k/√K to be quantized

## Available Output Options

### Perturbation/Transfer Outputs (✓ Work with arbitrary k)
- `dTk` - Density transfer functions
- `vTk` - Velocity transfer functions
- `mTk` - Matter transfer functions
- `mPk` - Matter power spectrum P(k)

### CMB/Harmonic Outputs (✗ Require quantized k)
- `tCl` - CMB temperature power spectrum
- `pCl` - CMB polarization power spectrum (E, B modes)
- `lCl` - CMB lensing potential power spectrum
- `nCl` - Number count power spectrum
- `sCl` - Cosmic shear power spectrum

## Code Flow Explanation

### When you use `output = dTk, vTk` (perturbations only):

```
input.c → background.c → thermodynamics.c → perturbations.c → output.c
                                                    ↓
                                          Solves Einstein-Boltzmann
                                          equations for each k
                                                    ↓
                                          Stores solutions in
                                          ppt->sources[index_k][...]
                                                    ↓
                                          output.c writes directly
                                          to perturbations_k*.dat
                                          ✓ No hyperspherical Bessel needed!
```

### When you use `output = tCl` (CMB spectrum):

```
input.c → ... → perturbations.c → transfer.c → harmonic.c → output.c
                        ↓              ↓
                  Solves for k  →  transfer_init() calls
                                   hyperspherical_HIS_create()
                                          ↓
                                   ✗ FAILS if k not quantized!
```

## Technical Details

### Why Quantization is Required for CMB

The CMB C_ℓ calculation involves: source/transfer.c:262

```c
class_call(hyperspherical_HIS_create(pba->sgnK,
                                     nu,           // nu = k/√K
                                     l_size_max,
                                     l,
                                     ...),
           ...);
```

In closed universes (K>0), the hyperspherical Bessel functions Φ_ℓ(x) satisfy:

```
d²Φ_ℓ/dx² + 2cotK(x) dΦ_ℓ/dx + [ν² - ℓ(ℓ+1)/sin²K(x)]Φ_ℓ = 0
```

where ν = k/√K must be an integer (or integer multiple of nu_spacing) for the boundary conditions at the "spatial horizon" x = π to be satisfied.

### k_output_values Bypasses harmonic.c

The `k_output_values` mechanism (source/perturbations.c:2558-2616) adds your requested k values directly to the perturbation solver's k array. When you request only transfer outputs:

- `ppt->has_cls = _FALSE_` (set in source/input.c:1764)
- transfer.c checks this flag (line 170) and skips harmonic computation
- output.c writes perturbations directly (lines 1168-1201)

## Validation of k Values (If You Need CMB)

If you DO need CMB spectra, you must use quantized k values. Here's how to check:

```python
import numpy as np

omega_k = -0.04
h = 0.54
nu_spacing = 3

# Calculate K
K = -omega_k * (100 * h)**2 / 299792.458**2  # (Mpc)^-2
sqrtK = np.sqrt(K)

k = 0.1  # Your k value to check

# Check quantization
q = np.sqrt(k**2 + K)
nu = q / sqrtK
nu_rounded = np.round(nu / nu_spacing) * nu_spacing

print(f"k = {k}")
print(f"nu = {nu:.6f}")
print(f"Nearest valid nu = {nu_rounded:.1f}")
print(f"Valid? {np.abs(nu - nu_rounded) < 1e-6}")

# Calculate valid k
q_valid = nu_rounded * sqrtK
k_valid = np.sqrt(q_valid**2 - K)
print(f"Nearest valid k = {k_valid:.6e}")
```

## Recommendations

1. **For perturbation solutions only**: Use the configuration shown above with `output = dTk, vTk`
   - ✓ Works with any k values
   - ✓ No quantization restriction
   - ✓ Direct access to δ(k,τ), θ(k,τ), Φ(k,τ), etc.

2. **For CMB power spectra**: You must either:
   - Use flat universe (omega_k = 0), OR
   - Use only quantized k values where ν = n × nu_spacing (n = 3, 4, 5, ...)

3. **For both**: Run CLASS twice:
   - Once with arbitrary k for perturbations
   - Once with quantized k grid for CMB (if needed)

## Files Involved

- `source/input.c`: Lines 1774-1824 (output type parsing)
- `source/perturbations.c`: Lines 2558-2616 (k_output_values insertion)
- `source/transfer.c`: Lines 170-178 (has_cls check), 262-267 (hyperspherical creation)
- `source/output.c`: Lines 1168-1201 (perturbation output)
- `tools/hyperspherical.c`: Hyperspherical Bessel function computation

## Example: What You Get in Output Files

From `output/test_perturbations_closed_perturbations_k0_s.dat`:

```
# scalar perturbations for mode k = 1.000000e-03 Mpc^(-1)
# tau (Mpc)    t (Gyr)     a         delta_g    theta_g    shear_g    delta_b    ...
1.234e+01     1.456e-05   1.234e-06  -3.456e-04  2.345e-04  1.234e-08  -2.345e-04  ...
```

You can then analyze these perturbations at arbitrary k values without any quantization restriction!
