# Modification Summary: Using Time Series Matrices for x_inf

## Date
2025-12-29

## File Modified
`generate_perturbation_solutions_transMatrix_timeseries_s-a.py`

## Change Description

Modified the script to use time series matrices at recombination (instead of static matrices from `data_allowedK/`) when solving for x_inf.

## Specific Changes

### 1. Data Loading (Lines 172-194)

**Before:**
```python
# Load the static matrices needed for boundary conditions
ABCmatrices = np.load(folder_path_matrices+'L70_ABCmatrices.npy')
DEFmatrices = np.load(folder_path_matrices+'L70_DEFmatrices.npy')
GHIvectors = np.load(folder_path_matrices+'L70_GHIvectors.npy')
X1matrices = np.load(folder_path_matrices + 'L70_X1matrices.npy')
X2matrices = np.load(folder_path_matrices + 'L70_X2matrices.npy')
recValues = np.load(folder_path_matrices + 'L70_recValues.npy')

# Extract all matrix components
Amatrices = ABCmatrices[:, 0:6, :]
Bmatrices = ABCmatrices[:, 6:8, :]
Cmatrices = ABCmatrices[:, 8:num_variables, :]
Dmatrices = DEFmatrices[:, 0:6, :]
Ematrices = DEFmatrices[:, 6:8, :]
Fmatrices = DEFmatrices[:, 8:num_variables, :]
```

**After:**
```python
# Load X1, X2, GHI, and recValues from static data (still needed)
# Note: ABCmatrices and DEFmatrices are NO LONGER used - we extract them from time series instead
GHIvectors = np.load(folder_path_matrices+'L70_GHIvectors.npy')
X1matrices = np.load(folder_path_matrices + 'L70_X1matrices.npy')
X2matrices = np.load(folder_path_matrices + 'L70_X2matrices.npy')
recValues = np.load(folder_path_matrices + 'L70_recValues.npy')

# Note: We no longer extract Amatrices, Bmatrices, etc. from static data
# Instead, we extract A, B, C, D, E, F from time series for each k mode individually
```

### 2. Matrix Extraction for Each k Mode (Lines 214-239)

**Before:**
```python
# Get exact matrices for this k
A = Amatrices[k_index]
B = Bmatrices[k_index]
C = Cmatrices[k_index]
D = Dmatrices[k_index]
E = Ematrices[k_index]
F = Fmatrices[k_index]
X1 = X1matrices[k_index]
X2 = X2matrices[k_index]
recs_vec = recValues[k_index]
```

**After:**
```python
# Extract full ABC and DEF matrices at recombination from time series
ABC_at_rec = all_ABC_solutions[k_index, :, :, 0]  # Shape: (75, 6)
DEF_at_rec = all_DEF_solutions[k_index, :, :, 0]  # Shape: (75, 2)

# Split into A, B, C and D, E, F components
A = ABC_at_rec[0:6, :]      # First 6 rows (phi, psi, dr, dm, vr, vm)
B = ABC_at_rec[6:8, :]      # Next 2 rows (fr2, fr3)
C = ABC_at_rec[8:num_variables, :]  # Remaining rows (higher multipoles)
D = DEF_at_rec[0:6, :]
E = DEF_at_rec[6:8, :]
F = DEF_at_rec[8:num_variables, :]

# Still use X1, X2 from static data (these are analytical, not from integration)
X1 = X1matrices[k_index]
X2 = X2matrices[k_index]
recs_vec = recValues[k_index]
```

## Rationale

1. **Consistency**: The entire reconstruction now uses data from the same source (time series)
2. **Verification**: We verified that static matrices and time series at recombination match within 1.7×10⁻⁸ relative precision (see `MATRIX_VERIFICATION_REPORT.md`)
3. **Expected Impact**: Since the matrices match so well, results should be essentially identical to the original version

## What Remains Unchanged

- X1, X2 matrices: Still loaded from static data (these are analytical Taylor expansions, not numerical integrations)
- GHIvectors: Still loaded from static data
- recValues: Still loaded from static data (forward integration results)
- All reconstruction logic: Unchanged
- Plotting: Unchanged

## Files Still Needed from data_allowedK/

- `L70_X1matrices.npy` (Taylor expansion matrices)
- `L70_X2matrices.npy` (Taylor expansion matrices)
- `L70_GHIvectors.npy` (Inhomogeneous solution vectors)
- `L70_recValues.npy` (Forward integration boundary conditions)

## Files Now Used from data_allowedK_timeseries/

- `L70_ABC_solutions.npy` - **NOW ALSO USED** for extracting A, B, C at recombination
- `L70_DEF_solutions.npy` - **NOW ALSO USED** for extracting D, E, F at recombination
- `L70_GHI_solutions.npy` - Used for reconstruction (unchanged)
- `t_grid.npy` - Time grid (unchanged)
- `L70_kvalues.npy` - K values (unchanged)

## Testing Recommendation

Run the modified script and compare:
1. The x_inf values before/after modification (should be nearly identical)
2. The phi constraint error at recombination (should be nearly identical)
3. The final plots (should be visually indistinguishable)

Any differences should be at the 10⁻⁸ level or smaller.
