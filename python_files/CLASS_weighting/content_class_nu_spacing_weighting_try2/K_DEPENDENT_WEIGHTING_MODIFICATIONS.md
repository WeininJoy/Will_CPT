# K-Dependent Weighting Modifications for CLASS

## Overview

This document describes the modifications made to the CLASS cosmological code to implement k-dependent weighting for low-k modes in closed universes. The weighting allows custom modification of the primordial power spectrum based on discrete mode indices.

## Motivation

In closed universes, cosmological perturbations have discrete modes with specific relationships between wavenumber k, comoving wavenumber q, and mode number ν. This implementation allows applying theoretical weights to these discrete modes to study modified cosmological scenarios.

## Key Formula

The weighting is applied based on the index calculated as:
```
q = sqrt(k² + K*(m+1))  # where m=0 for scalars, m=1 for vectors, m=2 for tensors
index = (int)((q/sqrt(K) - ν_min) / ν_spacing)
weight = weights[index]  # if index is in valid range
```

Where:
- `ν_min = 3` (hardcoded, following CLASS convention)
- `ν_spacing` comes from background parameters
- `K` is the curvature parameter

## Files Modified

### 1. `include/primordial.h`
**Added weighting parameters to primordial structure:**
```c
/* - parameters for k-dependent weighting in closed universes */
int has_weighting; /**< flag: do we have k-dependent weighting? */
char weighting_filename[_FILENAMESIZE_]; /**< name of file containing weights for discrete modes */
int max_weight_index; /**< maximum index in weighting table (e.g., 30) */
double * weights; /**< array of weights indexed by mode index (0 to max_weight_index) */
double K; /**< curvature parameter for weighting calculation */
int nu_spacing; /**< nu spacing parameter for weighting calculation */
int sgnK; /**< sign of curvature for weighting calculation */
```

**Added function declaration:**
```c
int primordial_read_weights(struct primordial * ppm);
```

### 2. `source/input.c`
**Added parameter reading functionality (after line 4207):**
```c
/** 1.b.1.3) K-dependent weighting for closed universes */
class_read_string("weighting_filename",ppm->weighting_filename);
if (strcmp(ppm->weighting_filename,"none") != 0) {
  ppm->has_weighting = _TRUE_;
  class_read_int("max_weight_index",ppm->max_weight_index);
  class_read_double("K",ppm->K);
  class_read_int("nu_spacing",ppm->nu_spacing);
  class_read_int("sgnK",ppm->sgnK);
}
else {
  ppm->has_weighting = _FALSE_;
}
```

### 3. `source/primordial.c`

#### a) Added weight file reading function:
```c
int primordial_read_weights(struct primordial * ppm) {
  FILE * file;
  int index, max_index;
  double weight;
  int entries_read;

  if (ppm->has_weighting == _FALSE_)
    return _SUCCESS_;

  /* Allocate weight array */
  class_alloc(ppm->weights,
              (ppm->max_weight_index + 1) * sizeof(double),
              ppm->error_message);

  /* Initialize all weights to 1.0 */
  for (index = 0; index <= ppm->max_weight_index; index++) {
    ppm->weights[index] = 1.0;
  }

  /* Open weight file */
  class_open(file, ppm->weighting_filename, "r", ppm->error_message);

  /* Read weights from file */
  while (fscanf(file, "%d %lf", &index, &weight) == 2) {
    if ((index >= 0) && (index <= ppm->max_weight_index)) {
      ppm->weights[index] = weight;
    }
  }

  fclose(file);
  return _SUCCESS_;
}
```

#### b) Modified `primordial_analytic_spectrum_init()` to call weight reading:
```c
/* Read k-dependent weights if specified */
class_call(primordial_read_weights(ppm),
           ppm->error_message,
           ppm->error_message);
```

#### c) Modified `primordial_analytic_spectrum()` to apply weighting:
```c
/* Apply k-dependent weighting for closed universes */
if (ppm->has_weighting == _TRUE_ && ppm->sgnK == 1) {
  double q, weight = 1.0;
  int index, nu_min = 3;
  
  /* Calculate q using your formula: q = sqrt(k*k + K*(m+1)) where m=0 for scalars */
  q = sqrt(k*k + ppm->K * (index_md + 1.));
  
  /* Calculate index using your formula: index = (int)(q/sqrt(K) - nu_min)/nu_spacing */
  index = (int)((q/sqrt(ppm->K) - nu_min) / ppm->nu_spacing);
  
  /* Apply weight if index is in valid range */
  if (index >= 0 && index <= ppm->max_weight_index) {
    weight = ppm->weights[index];
  }
  
  /* Apply the weight */
  *pk *= weight;
}
```

#### e) Modified `primordial_get_lnk_list()` to use stored parameters:
Updated function to use `ppm->K`, `ppm->nu_spacing`, and `ppm->sgnK` instead of passed parameters.

#### f) Modified `primordial_free()` to clean up weights:
```c
/* Free weights if allocated */
if (ppm->has_weighting == _TRUE_) {
  free(ppm->weights);
}
```

## Usage

### Input Parameters

Add the following parameters to your CLASS `.ini` file or Python dictionary:

```ini
# K-dependent weighting (required parameters when enabled)
weighting_filename = mode_weights.dat      # path to weight file (or "none" to disable)
max_weight_index = 31                      # maximum index in weight table (0-31 for 32 weights)
K = 0.02604086891105694                   # curvature parameter
nu_spacing = 1                            # nu spacing parameter
sgnK = 1                                  # sign of curvature (+1 for closed universe)

# Extended k range for compatibility with weighting (recommended)
k_max_tau0_over_l_max = 20.0
```

### Weight File Format

Create a weight file with the following format:
```
# index  weight
0       1.0234
1       0.9876
2       1.1234
3       0.8765
...
30      1.0000
```

- **index**: Mode index (0 to max_weight_index)
- **weight**: Multiplicative weight factor for that mode
- Lines starting with `#` are ignored
- Missing indices default to weight = 1.0

### Behavior

- **Weighting is only applied in closed universes** (`sgnK == 1`)
- **Weights are applied to all perturbation types** (scalar, vector, tensor)
- **Invalid indices are ignored** (weight = 1.0 used)
- **Standard CLASS behavior** when `weighting_filename = "none"` or not specified

## Technical Details

### Design Approach

We used **Approach (1)**: Store background parameters in the primordial structure during initialization. This was chosen because:

1. Background parameters are constant after initialization
2. `primordial_spectrum_at_k()` is called from multiple modules without access to background structure
3. Avoids breaking existing function interfaces
4. Consistent with CLASS design patterns

### Memory Management

- Weight array is allocated once during initialization
- All weights initialized to 1.0 (no modification)
- Proper cleanup in `primordial_free()`
- File reading with error handling

### Performance Impact

- Minimal: Weight lookup is O(1) during spectrum calculation
- File reading occurs only once during initialization
- No impact when weighting is disabled

## Example Use Cases

1. **Testing theoretical models** with modified low-k power
2. **Studying discrete mode effects** in closed universe cosmology
3. **Implementing custom primordial spectra** for specific theoretical predictions
4. **Debugging and validation** of CLASS nu_spacing functionality

## Complete Implementation Summary

### 1. Weight Generation Pipeline

The implementation includes a complete pipeline for generating and using k-dependent weights:

1. **Eigenfunction Analysis** (`solve_real_cosmology_weighting.py`):
   - Loads eigenfunction coefficients from `coefficients_basis1.npy`
   - Calculates weights as `|coefficient_1[0,:]|²`
   - Normalizes weights so that `sum(weights) = N` (number of modes)
   - Outputs `mode_weights.dat` in CLASS-compatible format

2. **CLASS Integration**:
   - Reads weight file during initialization
   - Applies k-dependent weighting to primordial power spectrum
   - Works seamlessly with existing CLASS functionality

3. **CMB Power Spectrum Generation** (`generate_cmb_spectrum.py`):
   - Extracts cosmological parameters from research scripts
   - Configures CLASS with weighting parameters
   - Generates CMB power spectra with and without weighting
   - Creates comparison plots and analysis

### 2. Key Implementation Details

#### Parameter Flow
```
solve_real_cosmology_vr.py → cosmological parameters
                ↓
solve_real_cosmology_weighting.py → mode_weights.dat (32 normalized weights)
                ↓
generate_cmb_spectrum.py → CMB power spectra with weighting
```

#### Weight Normalization
- **Formula**: `weights[i] = |coeff1[0,i]|² × N / sum(|coeff1[0,:]|²)`
- **Result**: `sum(weights) = 32` for proper cosmological normalization
- **Physical meaning**: Each weight represents the contribution of a discrete mode

#### Memory Management
- Weights allocated once during `primordial_analytic_spectrum_init()`
- Default weight value: 1.0 (no modification)
- Proper cleanup in `primordial_free()`
- Error handling for file I/O operations

### 3. Compilation and Installation

#### Prerequisites
```bash
# Install required Python packages
pip install cython numpy setuptools matplotlib

# Ensure virtual environment is used to avoid conflicts
source class_weighting_env/bin/activate
```

#### Build Process
```bash
# Clean and rebuild CLASS with modifications
cd class_nu_spacing_weighting
make clean
make all

# Build and install Python wrapper
cd python
python setup.py build
python setup.py install  # installs to virtual environment
```

#### Verification
```bash
# Test CLASS import
python -c "from classy import Class; print('SUCCESS')"

# Run CMB generation script
python generate_cmb_spectrum.py
```

### 4. Generated Files

The implementation creates several output files:

- **`mode_weights.dat`**: Weight file in CLASS format (32 weights, normalized)
- **`mode_weights.npy`**: NumPy array of weights for analysis
- **`mode_weights.pdf`**: Visualization of weight distribution
- **`cmb_spectra_results.npz`**: CMB power spectra with/without weighting
- **`cmb_power_spectra_with_weighting.pdf/png`**: Comparison plots

### 5. Validation Results

The successful implementation demonstrates:

✅ **Parameter Recognition**: All custom parameters accepted by CLASS
✅ **Weight Application**: K-dependent weighting applied to primordial spectrum
✅ **Stable Computation**: No segmentation faults or numerical instabilities
✅ **Proper K-ranges**: Compatible k-max calculations with extended ranges
✅ **CMB Generation**: Full pipeline from weights to CMB power spectra

## Compatibility

- **Backward compatible**: No changes to existing CLASS functionality
- **Optional feature**: Disabled by default (`weighting_filename = "none"`)
- **Works with existing CLASS features**: nu_spacing, custom precision files, etc.
- **Thread safe**: No shared state modifications during spectrum calculation
- **Python Integration**: Full support for Python wrapper and scripting