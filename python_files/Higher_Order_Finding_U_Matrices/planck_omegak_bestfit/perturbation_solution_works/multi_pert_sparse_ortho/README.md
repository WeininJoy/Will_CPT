# Sparse and Orthonormal Valid Solutions

This directory contains implementations of two methods for finding **sparse and orthonormal** valid solutions for cosmological perturbations:

1. **Varimax Rotation** (classical method, fast, no hyperparameters)
2. **Sparse PCA via Manifold Optimization** (modern method, tunable sparsity)

Both methods maintain strict orthonormality while rotating eigenvectors to maximize sparsity.

---

## Background

Your `multi_perturbation_analysis.py` finds valid eigenvectors (with eigenvalue ≈ 1) by solving:

```
A = M @ M^† where M = <basis_1 | basis_2>
```

The eigenvectors form an **orthonormal basis** for the valid solution space. However, this basis is **arbitrary** - any rotation of these eigenvectors is equally valid physically.

The goal here is to find the **sparsest rotation** of this basis, which corresponds to:
- Localized wave numbers (single-k modes)
- Easier physical interpretation
- Simpler structure

---

## Files Overview

| File | Description |
|------|-------------|
| `varimax_sparse_rotation.py` | Varimax rotation method |
| `sparse_pca_rotation.py` | Sparse PCA via manifold optimization |
| `compare_methods.py` | Side-by-side comparison of both methods |
| `VarimaxRotation_sparsePCA.md` | Theoretical guidance document |
| `README.md` | This file |

---

## Method 1: Varimax Rotation

### Overview
- **Classical method** from factor analysis (1950s)
- Used in physics/chemistry for "localized orbitals"
- **No hyperparameters** to tune
- **Fast** (converges in ~10-50 iterations)
- **Strictly preserves orthogonality**

### Usage

```bash
python varimax_sparse_rotation.py
```

### Outputs
- `varimax_sparse_results.pickle` - Full results
- `varimax_sparse_coefficients_comparison.pdf` - Before/after comparison
- `varimax_sparse_coefficients_sorted.pdf` - Sorted by dominant k

### When to use
- Quick exploratory analysis
- When you need guaranteed convergence
- When you don't want to tune parameters
- Default choice for most applications

---

## Method 2: Sparse PCA via Manifold Optimization

### Overview
- **Modern method** (Erichson et al., 2020)
- Optimizes L1 norm directly over the Stiefel manifold
- **Tunable sparsity** via `sparsity_weight` parameter
- **More aggressive sparsity** than Varimax
- Requires `pymanopt` library

### Installation

```bash
pip install pymanopt
```

### Usage

```python
python sparse_pca_rotation.py
```

To tune sparsity, edit the `sparsity_weight` parameter in the script:

```python
sparse_results = sparse_pca_analysis(
    results,
    eigenvalue_threshold=0.9,
    sparsity_weight=2.0,  # Higher = more sparse (try 0.5, 1.0, 2.0, 5.0)
    epsilon=1e-8
)
```

### Outputs
- `sparse_pca_results.pickle` - Full results
- `sparse_pca_coefficients_comparison.pdf` - Before/after comparison
- `sparse_pca_coefficients_sorted.pdf` - Sorted by dominant k

### When to use
- When Varimax is not sparse enough
- When you need maximum sparsity
- For publication-quality results with tuned parameters
- When computational cost is not a concern

---

## Method Comparison

### Quick Comparison

Run both methods and compare results:

```bash
python compare_methods.py
```

This generates:
- `comparison_results.pickle` - Combined results
- `method_comparison.pdf` - Side-by-side coefficient plots
- `sparsity_comparison_bar.pdf` - Quantitative sparsity metrics

### Pros and Cons

| Feature | Varimax | Sparse PCA |
|---------|---------|------------|
| **Speed** | Very fast (~seconds) | Slower (~minutes) |
| **Hyperparameters** | None | `sparsity_weight`, `epsilon` |
| **Sparsity level** | Good | Excellent (tunable) |
| **Convergence** | Always | Usually (check `converged` flag) |
| **Dependencies** | NumPy/SciPy only | Requires `pymanopt` |
| **Ease of use** | Very easy | Moderate |

---

## Understanding the Results

### Key Outputs

Both methods produce:

1. **Sparse eigenvectors** (`eigenvecs_sparse`): Rotated orthonormal eigenvectors
2. **Rotation matrix** (`R`): The transformation applied
3. **Sparse coefficients** (`coefficients_sparse`): Coefficients in the k-basis

### Interpreting Coefficients

After rotation, each eigenfunction should have:
- **One dominant k-mode** (tall bar in plots)
- **Small contributions** from other modes (near-zero bars)

Example interpretation:
```
λ=0.987, dominant k=5
```
This means:
- Eigenvalue is 0.987 (valid solution)
- Primarily composed of k-mode #5
- This is a "pure" or "localized" mode

### Sparsity Metrics

The code reports several metrics:

1. **Fraction near-zero**: Percentage of coefficients with |c| < threshold
2. **L1 norm**: Sum of absolute values (lower = sparser)
3. **L0 norm**: Number of non-zero elements (lower = sparser)

---

## Workflow Integration

### Step 1: Run your main analysis

```bash
cd ..
python multi_perturbation_analysis.py
```

This saves `multi_perturbation_results.pickle` containing eigenvalues and eigenvectors.

### Step 2: Choose a sparsification method

**Option A: Quick analysis with Varimax**
```bash
cd multi_pert_sparse_ortho
python varimax_sparse_rotation.py
```

**Option B: Maximum sparsity with Sparse PCA**
```bash
cd multi_pert_sparse_ortho
pip install pymanopt  # If not installed
python sparse_pca_rotation.py
```

**Option C: Compare both methods**
```bash
cd multi_pert_sparse_ortho
python compare_methods.py
```

### Step 3: Analyze results

Load the pickle files in your own scripts:

```python
import pickle

# Load Varimax results
with open('varimax_sparse_results.pickle', 'rb') as f:
    varimax = pickle.load(f)

# Access sparse coefficients
coeffs = varimax['coefficients_sparse']['vr']  # For vr perturbation
eigenvals = varimax['eigenvals_valid']

# Print dominant modes
dominant_k = varimax['dominant_k']
print(f"Dominant k modes: {dominant_k}")
```

---

## Theoretical Justification

From your guidance document (`VarimaxRotation_sparsePCA.md`):

> You are looking for a **Rotation Matrix** R (K × K unitary matrix) such that the new basis C_new = C · R is sparse. Since R is unitary, if C was orthonormal, C_new will also be **strictly orthonormal**.

Both methods find this rotation R, but with different objectives:

- **Varimax**: Maximizes variance of squared coefficients (kurtosis)
- **Sparse PCA**: Minimizes smooth L1 norm on Stiefel manifold

Physical interpretation:
> The eigensolver returns an arbitrary rotation of the degenerate subspace (eigenvalue ≈ 1). Varimax/Sparse PCA rotates this to the basis of "Simple Structure," which in your physics context corresponds to localized wave numbers.

---

## Parameters to Tune

### Common Parameters

- `eigenvalue_threshold`: Minimum eigenvalue to include (default: 0.95)
  - Lower = include more modes
  - Higher = only include best matches

- `N_plot`: Number of modes to plot (default: 10)

### Sparse PCA Specific

- `sparsity_weight`: Sparsity penalty (default: 1.0)
  - Higher = more sparse (try 2.0, 5.0)
  - Lower = less sparse, closer to Varimax

- `epsilon`: L1 smoothing parameter (default: 1e-8)
  - Usually keep at default
  - Smaller = sharper but harder to optimize

---

## Troubleshooting

### Varimax not sparse enough
- Use Sparse PCA with higher `sparsity_weight`

### Sparse PCA not converging
- Check `converged` flag in results
- Try lower `sparsity_weight`
- Increase `max_iterations`

### pymanopt installation issues
```bash
# Try with pip
pip install pymanopt

# Or with conda
conda install -c conda-forge pymanopt
```

### Memory issues with large matrices
- Reduce `N_t` in `multi_perturbation_analysis`
- Process perturbation types separately

---

## Citation and References

If you use these methods in your thesis/paper:

**Varimax:**
- Kaiser, H. F. (1958). "The varimax criterion for analytic rotation in factor analysis." *Psychometrika*, 23(3), 187-200.

**Sparse PCA:**
- Erichson, N. B., et al. (2020). "Sparse Principal Component Analysis via Variable Projection." *SIAM Journal on Applied Mathematics*, 80(2), 977-1002.

**Manifold Optimization:**
- Townsend, J., Koep, N., & Weichwald, S. (2016). "Pymanopt: A Python toolbox for optimization on manifolds using automatic differentiation." *Journal of Machine Learning Research*, 17(137), 1-5.

---

## Questions?

For issues specific to these rotation methods:
1. Check the guidance document: `VarimaxRotation_sparsePCA.md`
2. Review the comparison results: `python compare_methods.py`
3. Adjust hyperparameters and re-run

For issues with the underlying analysis:
1. Check `../multi_perturbation_analysis.py`
2. Verify data files in `../data/`

---

## Summary

**Quick Start:**
```bash
# Install dependencies (if needed)
pip install pymanopt

# Run Varimax (fast, no tuning)
python varimax_sparse_rotation.py

# OR run Sparse PCA (more control)
python sparse_pca_rotation.py

# OR compare both
python compare_methods.py
```

**Key Points:**
- Both methods preserve orthonormality **exactly**
- Varimax: Fast, no tuning, good default choice
- Sparse PCA: Slower, tunable, maximum sparsity
- Sparser = easier to interpret physically
- Coefficients sorted by dominant k-mode

Good luck with your research!
