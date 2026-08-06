# Joint Eigenfunction Analysis: Method, Reasoning, and Discussion

## 1. Problem Statement

We have two sets of cosmological perturbation solutions computed by CLASS:

- **Basis 1** (integerK): solutions `Ψ1_p` of shape `(N_t, N_k)` for each perturbation type `p`
- **Basis 2** (allowedK): solutions `Ψ2_p` of shape `(N_t, N_k)` for the same time grid but different k-values

There are `P` perturbation types (δ_cdm, δ_b, θ_cdm, θ_b, Φ, photon multipoles, ...).

**Goal**: Find all "modes of maximum coherence" — linear combinations of integerK modes that are as close as possible to linear combinations of allowedK modes, across all perturbation types simultaneously. In the special case where the two bases perfectly overlap (cos²θ = 1), this gives exact joint solutions in S1 ∩ S2.

We additionally have **high-k modes** (k > N_core) computed identically in both bases (same physical setup, same k-values), so they are literally the same vectors in both Basis 1 and Basis 2. These always produce exact joint solutions (cos²θ = 1).

---

## 2. The Stacking Constraint

To enforce consistency across all perturbation types simultaneously, we stack them:

```
Ψ1_stacked = [Ψ1_p1 / N_p1 ; Ψ1_p2 / N_p2 ; ... ; Ψ1_pP / N_pP]   shape (P*N_t, N_k)
Ψ2_stacked = [Ψ2_p1 / N_p1 ; Ψ2_p2 / N_p2 ; ... ; Ψ2_pP / N_pP]   shape (P*N_t, N_k)
```

where `N_p = ‖Ψ_p‖_F` is the Frobenius norm of each perturbation type (see Section 4 on normalization).

This imposes the constraint that **the same linear combination coefficients `c_k` must work for ALL perturbation types simultaneously**. This is a strong physical constraint: each eigenfunction `e_i = Σ_k c_{i,k} ψ_{k,p}` must describe a consistent multi-field state across all fields.

---

## 3. What the Method Actually Computes

It is important to be precise about what the SVD/principal-angle pipeline finds.

Let the time-series produced by the back-mapped coefficients be:
- From Basis 1: `f1_i(t) = Ψ1 @ C1[:, i] = U1 @ Um[:, i]`
- From Basis 2: `f2_i(t) = Ψ2 @ C2[:, i] = U2 @ Vm[:, i]`

**When `Sm[i] = 1` exactly**: `f1_i = f2_i` — the same function lies in both subspaces. This is a **genuine valid joint solution** in S1 ∩ S2. This happens for high-k modes (identical in both bases) and for any directions the two bases share exactly.

**When `Sm[i] < 1`**: `f1_i ≠ f2_i`. These are the **principal vectors** — the closest pair of directions in S1 and S2 respectively, separated by angle θ_i = arccos(Sm[i]). They are NOT equal functions and NOT valid joint solutions. Instead, they represent the **mode of maximum coherence** at that principal angle.

### Physical interpretation

The value `Sm[i] = cos(θ_i)` directly measures how well the two bases agree on mode i:
- `Sm = 1.000`: perfect agreement — an exact joint solution exists
- `Sm = 0.999`: agreement to within ~2.6° — the two bases describe nearly identical physics
- `Sm = 0.900`: agreement to within ~26° — significant physical difference

For the integerK vs allowedK comparison, **exact agreement (Sm = 1) is not expected for low-k modes**. The physically meaningful result is the coherence spectrum `{Sm[i]}` and the corresponding linear combination coefficients `C1, C2`.

### Alternative: exact intersection via null space

If you need the *exact* subspace S1 ∩ S2 (only modes with Sm = 1), solve `Ψ1 @ c1 = Ψ2 @ c2` via the null space of the augmented matrix:

```python
Psi_aug = np.hstack([Psi1, -Psi2])   # (N_t, 2*N_k)
_, S_a, Vt_a = np.linalg.svd(Psi_aug)
# Null vectors (S_a ≈ 0) give [c1; c2] with Ψ1 @ c1 = Ψ2 @ c2 exactly
```

This is numerically fragile and gives no information about "nearly intersecting" directions; the GEP approach is superior in practice.

---

## 4. Why Missing Modes Can Be Physical

Because of the shared-coefficient constraint, not all k-modes can necessarily be represented as joint eigenfunctions. The principal angles between S1 = span(Ψ1) and S2 = span(Ψ2) measure how "compatible" the two bases are:

- **cos²(θ) ≈ 1**: a direction nearly common to both subspaces → high-coherence mode
- **cos²(θ) ≪ 1**: a direction in S1 nearly orthogonal to all of S2 → no coherent joint mode

A missing mode at k* means: the multi-field state at k* in integerK has no coherent counterpart in the allowedK basis. This is **not necessarily a numerical artifact** — it may reflect genuine physical inconsistency between the two bases at that scale.

### Diagnostic test

To distinguish numerical artifacts from physical missing modes, run the pipeline **per field** (unstacked) and compare singular values at k*:

- If `Sm_cdm(k*) ≈ 1`, `Sm_b(k*) ≈ 1`, but `Sm_{photon ℓ=10}(k*) = 0.85` → the photon multipoles are responsible. The mode is physically inconsistent between the two bases for that field.
- If Sm is low in ALL individual field analyses → likely numerical (ill-conditioned basis, near-linear-dependence).

Missing modes in **high-k** are expected to be numerical (since both bases have identical high-k columns). Missing modes in **low-k** are more likely physical.

---

## 5. Normalization Before Stacking

Different perturbation types have very different amplitude scales (e.g., Φ ~ 10⁻⁵ vs θ_b ~ 10⁻³). Without normalization, the SVD is dominated by large-amplitude fields, effectively ignoring smaller but physically equally important ones.

**Recommended normalization**: divide each block by its Frobenius norm before stacking:

```python
N_p = np.linalg.norm(Psi_p, 'fro')
Psi_p_norm = Psi_p / N_p
```

This ensures every perturbation type contributes equally to the inner product. Physically: a 1% change in the structure of Φ is treated as equally significant as a 1% change in θ_b.

**Note**: normalization defines the inner product on the stacked space. This is an unavoidable physical modeling choice (not an arbitrary numerical one). The subsequent analysis is canonical *given* this choice.

---

## 6. The Role of the Weighting Matrix G

### The current method as a special case of a general inner product

Stacking Frobenius-normalized bases is equivalent to defining a **diagonal weighting matrix G** in perturbation-type space. The inner product between two k-mode columns is:

```
<v, w>_G = Σ_{p,q} ψ_p^T G_{pq} ψ_q    where G_{pq} = (1/N_p²) δ_{pq}
```

This is a special case of a more general framework where G can be any positive-definite matrix.

### The GEP with general G

With an arbitrary positive-definite G, all Gram matrices become G-weighted:

```
G̃11 = Ψ1^T G Ψ1,    G̃22 = Ψ2^T G Ψ2,    G̃12 = Ψ1^T G Ψ2
```

and the GEP becomes:

```
(G̃12 G̃22⁻¹ G̃12^T) c1 = λ G̃11 c1
```

**The principal angles λ = cos²θ and eigenvectors c1 both depend explicitly on G.** There is no G-independent set of "valid modes" — the choice of G is a physical statement about what "coherence" or "similarity" means between the two bases.

### Physical motivation for different G choices

| G choice | Physical meaning | When to use |
|----------|-----------------|-------------|
| Diagonal, Frobenius-normalized: `G_{pq} = (1/N_p²)δ_{pq}` | Fields are independent; 1% fractional change in each field has equal weight | Best uninformative default — current method |
| Kinetic metric K from Lagrangian | Coherence defined by fundamental dynamics; fields coupled by equations of motion | Theory-first comparison, encodes physical coupling |
| Fisher information matrix F | Coherence from observational distinguishability; degenerate field combinations down-weighted | Observer-centric, CMB/LSS motivated |
| Inverse covariance C⁻¹ | Optimal statistical comparison weighted by measurement precision | Data-driven, accounts for noise |
| G = I (unweighted identity) | Assumes all fields have the same amplitude scale — **incorrect** | Dominated by large-amplitude fields; ignores Φ |

### Why the diagonal Frobenius G is the correct default

The G = I (unweighted) choice is physically wrong: raw amplitudes differ by orders of magnitude (Φ ~ 10⁻⁵ vs θ_b ~ 10⁻³), so the SVD would be completely dominated by high-amplitude fields and blind to physically crucial low-amplitude ones like the gravitational potential.

The Frobenius-normalized diagonal G is the best **uninformative default** because it:
1. Makes the analysis sensitive to **fractional changes** in every field equally
2. Assumes no prior knowledge about correlations between field types (maximum entropy choice)
3. Treats each physical observable as an independent degree of freedom

Off-diagonal G choices are physically justified only when additional structure is known:
- **Kinetic metric** (from the perturbation Lagrangian): non-diagonal, encodes how fields are dynamically coupled — arguably the most physically deep choice, but requires knowing the full Lagrangian
- **Fisher metric**: optimal if you have a specific observational model in mind
- **Inverse covariance**: optimal if you have data with known noise properties

**Conclusion**: The Frobenius-normalized diagonal G is not arbitrary — it is the most principled choice in the absence of a specific physical question that would motivate a different metric. The principal angles found with this G answer the question: *"Which modes of multi-field perturbation evolution are fractionally coherent across all fields simultaneously?"*

---

## 7. The Spacetime Inner Product and the S³ Spatial Basis

### Why the time-only inner product is insufficient

The time-only inner product `<ψ_k, ψ_k'> = Σ_t ψ_k(t) ψ_k'(t)` can produce spurious correlations between modes at very different k-scales if their time-series happen to overlap. This makes the Gram matrix dense and allows unphysical long-range k-mixing.

The physically correct inner product integrates over **both space and time**, using the natural spatial eigenfunctions of the closed universe geometry.

### Spatial basis on S³: scalar hyperspherical harmonics

In a closed universe (K > 0), spatial sections are 3-spheres (S³). The scalar eigenfunctions of the Laplace-Beltrami operator are the **scalar hyperspherical harmonics**:

```
Q_k(χ) ∝ sin((k+1)χ) / sin(χ)     k = 1, 2, 3, ... (must be integer)
```

where χ ∈ [0, π] is the radial hyperspherical coordinate. Volume element: `dV = sin²(χ) sin(θ) dχ dθ dφ`.

Their orthogonality on S³ is **exact** for integer k:
```
∫_{S³} Q_k(χ) Q_k'(χ) dV = (π/2) δ_{k,k'}
```

Normalization constant: `N_k = √(2/π)`, so normalized `Q̂_k = √(2/π) Q_k` satisfies `∫ Q̂_k² dV = 1`.

### The allowedK basis is not an eigenfunction on S³

Non-integer k modes like `Q_{1.3}(χ) = sin(2.3χ)/sin(χ)` are **not eigenfunctions of the S³ Laplacian**. The allowedK basis represents solutions of a different physical problem (different boundary conditions), expressible only as an infinite series of integer eigenmodes.

### Spatial overlap integral: derivation

The overlap between integerK mode Q_k (integer k) and allowedK mode Q_n (non-integer n) is:

```
I_{kn} = ∫₀^π Q_k(χ) Q_n(χ) sin²(χ) dχ
        = ∫₀^π sin((k+1)χ) sin((n+1)χ) dχ
```

(the sin²(χ) from the volume element cancels the sin²(χ) in the denominators of Q_k and Q_n).

Using the product-to-sum identity `sin(Aχ)sin(Bχ) = ½[cos((A-B)χ) − cos((A+B)χ)]` with A = k+1, B = n+1:

```
I_{kn} = ½ [ sin((k-n)π)/(k-n)  −  sin((k+n+2)π)/(k+n+2) ]
```

Since k is an integer: `sin((k-n)π) = (-1)^{k+1} sin(nπ)` and `sin((k+n+2)π) = (-1)^k sin(nπ)`.
Substituting and simplifying:

```
I_{kn} = (-1)^{k+1} (k+1) sin(nπ) / [(k-n)(k+n+2)]
```

The **normalized** overlap (dividing by I_{kk} = π/2):

```
Î_{kn} = (2/π) · (-1)^{k+1} (k+1) sin(nπ) / [(k-n)(k+n+2)]
```

**Key properties:**
- `Î_{k,k} = 1` (self-overlap, same integer mode)
- `Î_{k,k'} = 0` for integer k ≠ k' (exact orthogonality between integer eigenmodes, since `sin((k-k')π) = 0`)
- Decays as `1/|k-n|` for non-integer n → long-range k-mixing strongly suppressed
- Peaked near k ≈ n → the closest integer k dominates the overlap

**Concrete examples** (k=3):

| n    | Î_{3,n}  | Interpretation                          |
|------|----------|-----------------------------------------|
| 3.0  | 1.000    | same mode                               |
| 3.3  | 0.936    | nearby allowedK — high overlap          |
| 4.0  | 0.000    | exact orthogonality (both integers)     |
| 10.0 | 0.000    | exact orthogonality (both integers)     |
| 10.3 | −0.020   | distant allowedK — very small overlap   |

---

## 8. The Canonical Method: Spacetime GEP

### Why naively modifying M breaks the pipeline

A tempting shortcut is to multiply the time-overlap M matrix element-wise by the spatial overlap:
```
M_spacetime[i,j] = Î_{k_i, n_j} × (U1^T @ U2)[i,j]    ← WRONG
```

This is **mathematically inconsistent** and invalidates the back-mapping. The reason:

- U1, U2 are orthonormal bases defined by the **time-only** inner product
- `M_time = U1^T @ U2` is the inner product matrix between these bases — a well-defined geometric object
- `M_spacetime = Î ⊙ M_time` is no longer the inner product between any two sets of orthonormal vectors
- Its singular values are NOT cosines of principal angles between any well-defined subspaces
- The back-mapping `C1 = V1 @ diag(1/S1) @ Um` relies on the chain `Ψ1 @ C1 = U1 @ Um`, which **breaks** when M is modified this way

The reduction in singular values from multiplying by Î ≤ 1 would partly reflect genuine physics (suppressing unphysical spatial mixing) but also introduce mathematical artifacts from the inconsistent construction.

### The correct approach: define the spacetime inner product from the start

The spacetime inner product must be used **consistently throughout** — in the Gram matrices, not patched onto the M matrix afterwards. This leads to the **Generalized Eigenvalue Problem (GEP)** with spacetime Gram matrices.

**Step 1: Compute spacetime Gram matrices**

```python
# Time overlaps (from stacked, Frobenius-normalized bases Psi1, Psi2)
G11_t = Psi1.T @ Psi1   # (N_k, N_k)  integerK self-overlap
G22_t = Psi2.T @ Psi2   # (N_k, N_k)  allowedK self-overlap
G12_t = Psi1.T @ Psi2   # (N_k, N_k)  cross-overlap

# Spatial overlap matrices
# Î_11[i,j] = δ_{i,j}  (integerK vs integerK: exact orthogonality)
# Î_22[i,j] = (2/π)·(-1)^{n_i+1}(n_i+1)sin(n_j π)/[(n_i-n_j)(n_i+n_j+2)]  (allowedK vs allowedK)
# Î_12[i,j] = (2/π)·(-1)^{k_i+1}(k_i+1)sin(n_j π)/[(k_i-n_j)(k_i+n_j+2)]  (integerK vs allowedK)

# Spacetime Gram matrices (element-wise product)
G11_st = np.eye(N_k) * np.diag(G11_t)          # diagonal: Î_11 = I, so G11_st = diag(||ψ1_k||²)
G22_st = Ihat_22 * G22_t                         # non-diagonal: allowedK modes not orthogonal spatially
G12_st = Ihat_12 * G12_t                         # diagonally dominant cross-term
```

Note: since integerK modes are exact S³ eigenfunctions with integer k, `Î_11[i,j] = δ_{i,j}`, making **G11_st diagonal** — a major simplification.

**Step 2: Solve the GEP**

```python
# Generalized eigenvalue problem:
# (G12_st @ inv(G22_st) @ G12_st.T) @ c1 = λ @ G11_st @ c1
A = G12_st @ np.linalg.solve(G22_st, G12_st.T)
# Solve: A @ c1 = λ @ G11_st @ c1
eigenvalues, C1 = scipy.linalg.eigh(A, G11_st)
```

**Step 3: Filter and interpret**

```python
valid = eigenvalues > threshold    # e.g., threshold = 0.99² = 0.9801
C1_valid = C1[:, valid]            # (N_k, N_joint) — coefficients in integerK basis
# C2 obtained analogously from the right eigenvectors
```

- Eigenvalues λ = cos²(θ) are the principal angles under the **spacetime** inner product
- Eigenvectors C1 are directly in the original k-basis — **no back-mapping needed**
- The GEP is canonical: results are independent of any intermediate orthogonalization

### Comparison of the two approaches

| Aspect | SVD + back-mapping (time-only) | GEP (spacetime) |
|--------|-------------------------------|-----------------|
| Inner product | Time only | Space + time (S³ correct) |
| M matrix | M = U1^T U2 | Gram matrices G11_st, G12_st, G22_st |
| Eigenvalues | cos²(θ) in time-space | cos²(θ) in spacetime |
| Eigenvectors | Require back-mapping V1 @ diag(1/S1) @ Um | **Directly in k-basis** |
| k-mixing | Allows unphysical long-range mixing | Suppressed by Î_{kn} ∝ 1/|k-n| |
| Canonical? | Yes (for time-only inner product) | Yes (for spacetime inner product) |

---

## 9. The Sparsity Problem and k-Space Varimax

In the GEP approach, eigenvectors C1 are already in k-space. However, for the degenerate subspace (λ = 1, high-k modes), the GEP returns an arbitrary orthonormal basis. Varimax is applied directly to C1 and C2 to recover sparse, k-localized modes:

```
J(L) = Σ|C1 @ L|⁴ + Σ|C2 @ L|⁴     subject to L^T L = I
```

```python
C_stacked = np.vstack([C1, C2])
L = varimax_rotation(C_stacked)
C1_sparse = C1 @ L
C2_sparse = C2 @ L
```

---

## 10. High-k Modes: Varimax is Necessary

Since high-k modes are **identical** in both bases, they all have λ = 1 exactly. The GEP returns an arbitrary orthonormal basis for this degenerate subspace. Varimax is **necessary** to rotate back to the original individual k-modes.

| Perspective | Varimax needed? | Reason |
|-------------|-----------------|--------|
| Mathematical validity | No | Any basis for the degenerate subspace is valid |
| Recovery of original k-modes | **Yes** | GEP/SVD returns arbitrary rotation; Varimax undoes it |

---

## 11. Handling the High-k Extension (Subspace Deflation)

Even with reversed ordering, the off-diagonal blocks of the overlap matrix are non-zero (core modes differ between bases), contaminating the high-k eigenspace.

**Step 1**: Löwdin-orthogonalize the high-k modes among themselves:
```python
G_hk = high_k.T @ high_k
D, Vhk = np.linalg.eigh(G_hk)
Q_high = high_k @ (Vhk @ np.diag(D**-0.5) @ Vhk.T)
```

**Step 2**: Project core modes into the subspace orthogonal to Q_high:
```python
P_high = Q_high @ Q_high.T
core_1_perp = core_1 - P_high @ core_1
core_2_perp = core_2 - P_high @ core_2
```

**Step 3**: Run the GEP pipeline (Section 8) on `core_1_perp` and `core_2_perp` only.

**Step 4**: Combine high-k (columns of Q_high) and core results.

---

## 12. Summary: Final Recommended Pipeline

```
1.  Normalize each perturbation type by Frobenius norm
2.  Stack all perturbation types → Ψ1_stacked, Ψ2_stacked

3.  [If using high-k extension] Apply subspace deflation (Section 11)

4.  Compute spacetime Gram matrices:
      G11_st[i,i] = ||ψ1_i||²_t              (diagonal, integerK exact S³ eigenfunctions)
      G22_st[i,j] = Î_{n_i,n_j} × (Ψ2^T Ψ2)[i,j]    (allowedK self-overlap)
      G12_st[i,j] = Î_{k_i,n_j} × (Ψ1^T Ψ2)[i,j]    (cross-overlap, diagonally dominant)

      where Î_{k,n} = (2/π)·(-1)^{k+1}(k+1) sin(nπ) / [(k-n)(k+n+2)]

5.  Solve GEP: (G12_st @ G22_st⁻¹ @ G12_st^T) @ c1 = λ @ G11_st @ c1

6.  Filter: keep eigenvectors with λ > threshold (e.g., 0.99² ≈ 0.98)

7.  C1[:, valid] — coefficients directly in integerK k-basis (no back-mapping needed)
    C2 from right eigenvectors analogously

8.  k-space Varimax on [C1; C2] → sparse eigenfunctions in original k-basis

9.  Interpret results:
      λ = 1.00: exact joint solution (genuine S1 ∩ S2)
      λ ≈ 1.00: high-coherence mode (nearly shared between bases)
      λ << 1.00: missing mode (physical or numerical — diagnose per-field)

10. Diagnose missing modes using per-field eigenvalue analysis
```

---

## 13. Key Takeaways

1. **The method finds modes of maximum coherence, not exact joint solutions**: only at λ = 1 exactly does the method give f1_i = f2_i. For λ < 1, the GEP gives the closest pair of directions in S1 and S2 (principal vectors). This is the physically correct output.

2. **Missing modes in low-k can be physical**: they indicate directions where integerK and allowedK are geometrically incompatible under the shared-coefficient constraint. Use per-field analysis to diagnose which perturbation type is responsible.

3. **Missing modes in high-k are likely numerical**: both bases have identical high-k columns, so all combinations should have λ = 1. Missing high-k modes point to orthogonalization artifacts or near-linear-dependence.

4. **The canonical method is the spacetime GEP**: using spacetime Gram matrices (G11_st, G12_st, G22_st) incorporates the S³ spatial structure correctly and gives eigenvectors directly in k-space. Simply element-wise multiplying the time-only M matrix by Î_{kn} is mathematically inconsistent and invalidates the back-mapping.

5. **G11_st is diagonal**: because integerK modes are exact S³ eigenfunctions with integer k, `Î_{k,k'} = δ_{k,k'}`, simplifying the GEP significantly.

6. **Normalization is a physical modeling choice**: Frobenius-norm normalization per perturbation type gives equal weight to all fields when stacking.

7. **Spatial overlap Î_{kn} enforces k-locality**: it suppresses unphysical long-range k-mixing (decays as 1/|k-n|), while still allowing physically motivated mixing between nearby k-modes. Exact zeros at integer k ≠ k' enforce S³ orthogonality.

8. **k-space Varimax is necessary for high-k degenerate subspace**: the GEP returns an arbitrary basis for the λ = 1 subspace; Varimax recovers the original individual k-modes.

9. **The principal angles depend on G — there is no G-independent set of valid modes**: the choice of weighting matrix G is a physical statement, not a mathematical artifact. The Frobenius-diagonal G (`G_{pq} = (1/N_p²)δ_{pq}`) answers the question *"which modes are fractionally coherent across all fields simultaneously?"* Different physical questions (dynamical coupling, observational distinguishability, noise-weighted optimality) require different G. The current method's G is the correct uninformative default, but should be replaced with the kinetic metric or Fisher matrix if those physical frameworks are more appropriate for the application.
