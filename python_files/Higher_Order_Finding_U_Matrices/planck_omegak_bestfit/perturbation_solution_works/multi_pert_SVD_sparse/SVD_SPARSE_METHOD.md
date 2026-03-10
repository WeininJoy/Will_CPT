# SVD-Based Sparse Joint Eigenfunction Method

## 1. Problem Statement

We have two sets of cosmological perturbation solutions:

- **Basis 1** (integerK): matrix `X_1^p` of shape `(N_t, N_k)` for each perturbation type `p ∈ {dr, dm, vr, vm}`
- **Basis 2** (allowedK): matrix `X_2^p` of shape `(N_t, N_k)` for the same perturbation types

Here `N_t = 1000` (time grid points) and `N_k = 335` (number of k-modes, including high-k extension).

**Goal**: Find linear combinations of k-modes — coefficients `c ∈ R^{N_k}` — such that the same
combination produces consistent time evolution in **both** bases, for **all** perturbation types
**simultaneously**. That is, find `c` such that `X_1^p @ c ≈ X_2^p @ c'` for all `p`, ideally with
`c ≈ c'`.

The final result should be a set of **sparse** coefficient vectors `c̃_j`, each concentrated on one
or a few k-modes, representing the "joint eigenfunctions" of the two bases.

---

## 2. Structure of the k-modes

The k-modes split naturally into two regimes:

| Regime | k range | # modes | Property |
|--------|---------|---------|----------|
| Low-k (core) | k ≤ ~25 | ~25 | integerK and allowedK differ — non-trivial mixing |
| High-k (extension) | k > ~25 | ~310 | **Identical** in both bases — trivially valid |

Because high-k modes are appended identically to both `basis_1` and `basis_2`, any coefficient
vector `c` that selects a single high-k mode is trivially a valid joint solution. All ~310 high-k
modes should be recovered as valid. Only the low-k modes require non-trivial analysis.

---

## 3. Why the QR Approach Fails

### 3.1 Practical Problem: Index Mixing

QR decomposition (Gram-Schmidt) processes columns left-to-right and produces an orthonormal basis
`Q` that is a global linear combination of all original k-modes. Eigenvectors in Q-space therefore
do not correspond to individual k-modes. The back-transform via `(R^p)^{-T}` should undo this, but
it is numerically unstable when R is ill-conditioned (see below).

### 3.2 Fundamental Problem: Rank Deficiency → Missing Modes

The data matrices have shape `(N_t, N_k) = (1000, 335)`, but many high-k modes produce nearly
identical time evolutions — they are physically degenerate. The effective rank is only ~155, far
below `N_k = 335`.

During QR, when column `j` is nearly in the span of the previous columns, the diagonal element
`R_{jj} ≈ 0`. The corresponding Q column becomes dominated by numerical noise, and the
back-transform amplifies it via `(R^{-T})_{jj} = 1/R_{jj} → ∞`. These noisy directions produce
small eigenvalues and are filtered out.

**Result**: Only ~155 modes survive instead of the expected ≥310. The ~180 "missing" modes are
not physically absent — they are lost to numerical instability in the QR step.

---

## 4. The SVD-Based Solution

### 4.1 Why SVD Handles Degeneracy Correctly

SVD explicitly identifies the rank. The singular values `σ_1 ≥ σ_2 ≥ ... ≥ σ_r >> σ_{r+1} ≈ 0`
directly reveal that the data lives in an r-dimensional subspace. Truncating to rank `r` gives a
clean, well-conditioned factorization. No modes are lost to instability.

### 4.2 Stacked SVD for a Universal Back-Transform

If we apply SVD per perturbation type separately, the back-transform `c_i = V^p Σ_p^{-1} e_i`
is perturbation-type-specific — different `p` gives different `c_i` for the same physical mode.
But physically, the **same** linear combination of k-modes must describe all perturbation types
simultaneously.

**Solution**: Stack all perturbation types (after Frobenius normalization) before applying SVD:

```
X_1_stacked = [ X_1^dr / ||X_1^dr||_F ]     shape: (4*N_t, N_k)
              [ X_1^dm / ||X_1^dm||_F ]
              [ X_1^vr / ||X_1^vr||_F ]
              [ X_1^vm / ||X_1^vm||_F ]
```

Each column of `X_1_stacked` now represents the **complete physical evolution** (all four fields)
of the j-th k-mode. The Frobenius normalization ensures all fields contribute equally regardless
of amplitude differences.

Compute the truncated SVD:

```
X_1_stacked = U_1 @ Σ_1 @ V_1^T     (rank-r truncated)
```

- `U_1` : `(4*N_t, r)` — left singular vectors (principal modes of time evolution, joint across all fields)
- `Σ_1` : `(r, r)` — diagonal matrix of singular values
- `V_1` : `(N_k, r)` — right singular vectors in k-space (**universal**, not field-specific)

The right singular vectors `V_1` live in k-space and are automatically universal. The
back-transform

```
c_i = V_1 @ Σ_1^{-1} @ e_i
```

gives a single coefficient vector valid for **all** perturbation types simultaneously.

**Derivation**: We want `c_i` such that `X_1_stacked @ c_i = U_1 @ e_i`. Substituting the SVD:

```
U_1 Σ_1 V_1^T c_i = U_1 e_i
→  Σ_1 V_1^T c_i = e_i          (left-multiply by U_1^T, use U_1^T U_1 = I)
→  V_1^T c_i = Σ_1^{-1} e_i
→  c_i = V_1 Σ_1^{-1} e_i       (project onto V_1 basis)
```

### 4.3 Mixing Matrix and Valid Modes

After computing the stacked SVD for both bases:

```
X_1_stacked = U_1 Σ_1 V_1^T
X_2_stacked = U_2 Σ_2 V_2^T
```

The mixing matrix measures overlap between the principal time evolution modes of the two bases:

```
M = U_1^T @ U_2       (r × r)
```

Compute eigenvalues of `M M^T`:

```
(M M^T) @ e_i = λ_i @ e_i
```

- `λ_i = 1`: perfect agreement — exact joint solution exists in both bases
- `λ_i ≈ 1`: high coherence — nearly valid joint mode
- `λ_i << 1`: incompatible mode — no consistent joint solution at this scale

Keep eigenvectors with `λ_i > threshold` (e.g., 0.99).

---

## 5. Handling the High-k Identical Modes: Deflation

Because ~310 high-k columns are **identical** in both bases, they produce a large degenerate
subspace with `λ = 1` exactly. Rather than feeding this into the SVD (where it would dominate
the singular value spectrum and make the valid low-k modes harder to find), we **deflate** it out
first.

### Deflation Procedure

**Step 1**: Extract the high-k columns and build an orthonormal basis for their subspace:

```python
X_high = X_stacked[:, high_k_indices]       # (4*N_t, N_high)
Q_high, _ = np.linalg.qr(X_high)            # Q_high: (4*N_t, N_high), orthonormal
```

**Step 2**: Project both stacked bases onto the orthogonal complement of the high-k subspace:

```python
P_high = Q_high @ Q_high.T                  # projection onto high-k subspace
X_1_low = (I - P_high) @ X_1_stacked        # (4*N_t, N_k), high-k signal removed
X_2_low = (I - P_high) @ X_2_stacked
```

**Step 3**: Run the SVD + eigenvalue analysis **only** on the deflated low-k matrices.

**Step 4**: After finding sparse low-k modes, **append** the N_high individual high-k modes
directly as identity-mapped modes (each selects a single k-index), with eigenvalue = 1.

This ensures:
- All ~310 high-k modes are recovered as valid (no missing modes)
- The SVD focuses only on the non-trivial low-k mixing
- The problem size is reduced from `(4*N_t, 335)` to roughly `(4*N_t, 25)` for the core analysis

---

## 6. Finding a Sparse Basis: Oblique Rotation

### 6.1 Why Varimax is Not Directly Applicable

The back-transformed vectors `c_i = V_1 Σ_1^{-1} e_i` are **not orthonormal** in k-space:

```
c_i^T c_j = e_i^T (Σ_1^{-1})^T V_1^T V_1 Σ_1^{-1} e_j = e_i^T Σ_1^{-2} e_j ≠ δ_ij
```

Varimax requires orthonormal input. Applying it directly to `c_i` would give incorrect results.

### 6.2 Goal: Sparse Linear Combinations Without Orthogonality Constraint

We want to find an invertible matrix `M` (`m × m`) such that:

```
c̃_j = Σ_i M_{ji} c_i      i.e.,   C̃ = C @ M^T      (N_k × m)
```

where each column of `C̃` is **sparse** (concentrated on one or few k-modes). Orthogonality of
`c̃_j` is **not** required — only sparsity.

### 6.3 Recommended Method: Oblique Rotation (Promax)

Oblique rotations generalize Varimax by relaxing orthogonality. They find `M` such that:
- Columns of `C̃ = C @ M^T` are sparse ("simple structure")
- `M` is invertible (the subspace S is preserved)
- `M` is NOT constrained to be orthogonal → more freedom → sparser solutions

**Implementation** using `factor_analyzer`:

```python
from factor_analyzer import FactorAnalyzer

# C is (N_k, m) matrix of back-transformed c_i vectors
fa = FactorAnalyzer(n_factors=m, rotation='promax')
fa.fit(C)
C_sparse = fa.loadings_    # (N_k, m) — sparse c̃_j vectors
```

### 6.4 Alternative: FastICA

ICA models the problem as blind source separation: the sparse `c̃_j` are "independent sources"
mixed by matrix `A` to produce the observed `c_i`. FastICA recovers the sources by maximizing
non-Gaussianity (kurtosis), which is equivalent to maximizing sparsity.

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=m, random_state=42)
C_sparse = ica.fit_transform(C)    # (N_k, m) — sparse c̃_j vectors
```

Set `random_state` for reproducibility.

### 6.5 Comparison

| Method | Orthogonality | Sparsity objective | Notes |
|--------|--------------|-------------------|-------|
| Varimax | Required (orthogonal M) | Maximize kurtosis of columns | Not applicable here |
| **Promax** | **Not required (oblique M)** | **Simple structure / kurtosis** | **Top recommendation** |
| FastICA | Not required | Maximize non-Gaussianity | Excellent alternative |
| ℓ1 + log-det | Not required | Direct ℓ1 minimization | Most rigorous, custom implementation |

---

## 7. Complete Pipeline

```
INPUT: X_1^p, X_2^p  for p ∈ {dr, dm, vr, vm}   shape (N_t, N_k) each

STEP 1 — Frobenius normalize and stack
    X_1_stacked = vstack(X_1^p / ||X_1^p||_F for each p)   shape (4*N_t, N_k)
    X_2_stacked = same for basis 2

STEP 2 — Deflate high-k subspace
    Identify high-k column indices (k > k_threshold)
    X_high = X_1_stacked[:, high_k_indices]
    Q_high, _ = qr(X_high)                                  # orthonormal high-k basis
    P = Q_high @ Q_high.T                                   # projector
    X_1_low = (I - P) @ X_1_stacked                        # deflated, shape (4*N_t, N_k)
    X_2_low = (I - P) @ X_2_stacked

STEP 3 — Stacked SVD on deflated matrices (rank-r truncated)
    X_1_low = U_1 @ Σ_1 @ V_1^T    (keep only r largest singular values)
    X_2_low = U_2 @ Σ_2 @ V_2^T

STEP 4 — Mixing matrix and valid eigenvectors
    M = U_1^T @ U_2                                         # (r, r)
    eigenvalues λ_i, eigenvectors e_i  of  M M^T
    keep e_i  with  λ_i > threshold (e.g., 0.99)           # m valid modes

STEP 5 — Universal back-transform to k-space
    c_i = V_1 @ Σ_1^{-1} @ e_i    for each valid i         # (N_k,) each
    C = hstack(c_i)                                         # (N_k, m)

STEP 6 — Oblique rotation for sparsity
    Apply Promax (or FastICA) to C
    C_sparse = C @ M_promax^T                               # (N_k, m), sparse columns

STEP 7 — Append high-k modes
    For each high-k index j:
        c̃_j = standard basis vector e_j                    # selects k-mode j exactly
    Append these N_high vectors to C_sparse

OUTPUT: C_sparse  of shape  (N_k, m + N_high)
    Each column c̃_j: sparse coefficient vector for a valid joint eigenfunction
    Low-k modes: found via SVD + Promax (m modes, λ close to 1)
    High-k modes: trivially valid, appended as identity (N_high modes, λ = 1 exactly)
```

---

## 8. Key Takeaways

1. **SVD over QR**: SVD handles rank deficiency cleanly by explicit rank truncation. QR amplifies
   numerical noise for degenerate columns via `R_{jj}^{-1}`, causing spurious missing modes.

2. **Stacking before SVD**: The right singular vectors `V` of the stacked matrix are universal
   — the back-transform `c_i = V Σ^{-1} e_i` gives k-space coefficients valid for all
   perturbation types simultaneously.

3. **Deflation before SVD**: Projecting out the identical high-k subspace before SVD ensures
   the analysis focuses on the physically non-trivial low-k modes, avoids the degenerate
   `λ = 1` cluster dominating the spectrum, and guarantees all high-k modes are recovered.

4. **Promax over Varimax**: The back-transformed `c_i` are not orthonormal, so Varimax (which
   requires orthogonal input) is replaced by Promax oblique rotation, which imposes no
   orthogonality constraint while still finding maximally sparse linear combinations.

5. **Missing modes explained**: The original QR + eigenvalue approach loses ~180 modes because
   the high-k columns are nearly linearly dependent (rank ≈ 155 < 335). SVD truncation makes
   this explicit and clean — the ~180 "missing" modes are in the null space of the stacked
   matrix and are correctly identified as redundant directions, not as missing physics.
