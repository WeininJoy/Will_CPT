# Why High-k Modes Are More Mixed in the CCA Heatmap

## Observation

The CCA heatmap (`figures_cca/cca_heatmap.pdf`) of column-normalised sparse coefficients is
nearly an identity matrix overall — each mode is concentrated on one k-value. However,
**mixing terms persist at high-k modes**, where one might naively expect the clearest
separation.

## Failed Intuition

The two k-grids converge at high k: `integerK_i ≈ allowedK_i` for large i. One might
therefore expect:

> "Since the two bases agree at high k, each integerK mode matches exactly one allowedK
> mode, so CCA should find near-perfect, single-k joint eigenfunctions with *less* mixing."

This reasoning is incomplete because it focuses only on the **cross-basis** convergence and
ignores the **within-basis** degeneracy that convergence simultaneously implies.

---

## The Correct Mathematical Picture

### 1. Within-Basis Near-Degeneracy

At high k, adjacent modes **within the same basis** become nearly indistinguishable in their
time evolution:

$$X^p_\alpha(k_i, \tau) \approx X^p_\alpha(k_{i+1}, \tau) \quad \text{for large } k.$$

Rapid oscillations and Silk damping cause the time-series columns of neighbouring high-k
modes to become nearly collinear. This is the key effect the naive intuition misses.

### 2. Ill-Conditioned Gram Matrices

The Gram matrix element `G1_{ij} = Σ_p w_p ⟨X̄^p_1(k_i), X̄^p_1(k_j)⟩` measures the
time-averaged inner product between modes i and j. At low k, modes are well-separated so
`G1` is strongly diagonal. At high k, `G1_{i, i+1} → 1` because the two column vectors
are nearly parallel. The high-k block of G1 (and G2) becomes **nearly singular**, with all
entries close to 1.

Note: `G12_{ij}` is also large for nearby high-k pairs, not just on the diagonal.

### 3. Whitening Amplifies Numerical Noise

`G1^{-1/2}` whitening is designed to orthogonalise the modes. But when `G1` is
ill-conditioned, `G1^{-1/2}` has very large entries and **dramatically amplifies tiny
numerical differences** in the high-k block. This is the opposite of stable behaviour.

### 4. SVD Returns an Arbitrary Mixed Basis

In the whitened cross-Gram matrix `W = G1^{-1/2} G12 G2^{-1/2}`, the high-k block yields
a **cluster of singular values all close to 1** — a degenerate subspace. When singular
values are degenerate, the corresponding singular vectors `u_i`, `v_i` are **not unique**:
any rotation within the degenerate subspace is equally valid.

The numerical SVD returns an *arbitrary* orthonormal basis for this subspace. CCA correctly
identifies that "the subspace spanned by high-k modes in Basis 1 is nearly identical to the
subspace spanned by high-k modes in Basis 2," but **cannot disentangle individual modes**
within that subspace.

### 5. Promax Cannot Fix the Degeneracy

Promax oblique rotation tries to find a sparser representation of the canonical vectors. But
if the underlying physical modes are genuinely degenerate in their time evolution, **there
is no sparser basis to find**. The mixing is a real consequence of physical degeneracy, not
an artifact of the initial basis choice. Promax re-expresses the same mixed subspace
slightly differently, but cannot resolve the fundamental ambiguity.

---

## Summary

| Effect | Low k | High k |
|--------|-------|--------|
| Cross-basis agreement (`integerK_i ≈ allowedK_i`) | Moderate | Strong |
| Within-basis mode separation | Large | Nearly zero |
| Gram matrix condition number | Small (well-conditioned) | Large (ill-conditioned) |
| SVD singular vector uniqueness | Unique | Degenerate (arbitrary mixing) |
| Heatmap sparsity | Good | Mixed |

**One-line summary:** The convergence `integerK_i ≈ allowedK_i` ensures `ρ_i ≈ 1` (good
cross-basis matching), but it simultaneously guarantees `integerK_i ≈ integerK_{i+1}`
(within-basis degeneracy), making the high-k Gram matrices ill-conditioned. CCA can only
identify the shared degenerate subspace — any basis for that subspace is equally valid and
inevitably mixed.

**Analogy:** Having three photos of the same person in nearly identical shirts #20, #21, #22
— an algorithm correctly identifies "blue shirt" as a concept but cannot distinguish the
individual shirts, so it returns arbitrary mixtures of all three.

---

## Strategies to Solve the Mixing Problem

### 1. Pre-processing / Regularisation

#### Tikhonov Regularisation (simplest fix)

Replace `G1`, `G2` with regularised versions before computing the whitening `G^{-1/2}`:

$$G_{1,\text{reg}} = G_1 + \lambda_1 I, \qquad G_{2,\text{reg}} = G_2 + \lambda_2 I.$$

This bounds eigenvalues away from zero, stabilising the matrix inversion.

- ✅ Simple to implement; often sufficient as a first fix
- ⚠️ Introduces bias; requires careful tuning of `λ`

#### Truncated SVD Whitening

Decompose `G1 = UΣUᵀ`, keep only the `p` components above a threshold `ε`, and build
`G1^{-1/2}` from the stable subspace alone:

$$G_1^{-1/2} \approx U_p \Sigma_p^{-1/2} U_p^\top.$$

- ✅ Explicitly removes degenerate directions; effective noise filter
- ⚠️ Discards high-k modes entirely — likely unacceptable here

---

### 2. CCA Stage

#### Regularised CCA (RCCA)

Incorporate the regularisation directly into the CCA eigenvalue problem:

$$(G_1 + \lambda I)^{-1} G_{12} (G_2 + \lambda I)^{-1} G_{12}^\top u = \rho^2 u.$$

Handles ill-conditioning inside the CCA itself rather than as a separate pre-processing step.

- ✅ Theoretically clean, unified framework
- ⚠️ Still needs hyperparameter tuning; shrinks canonical vectors toward zero

#### Sparse CCA (strongest candidate)

Build sparsity directly into the CCA objective via an L1 penalty:

$$\max_{u,v} \; u^\top G_{12} v \quad \text{subject to} \quad u^\top G_1 u \leq 1, \; v^\top G_2 v \leq 1, \; \|u\|_1 \leq c_1, \; \|v\|_1 \leq c_2.$$

Within a degenerate subspace where multiple solutions exist, the L1 penalty **automatically
selects the single-k solution**, breaking the degeneracy in the desired direction.

- ✅ Directly encodes the sparsity goal; best chance of a clean identity heatmap
- ⚠️ More computationally expensive; sparsity parameters `c1`, `c2` need tuning;
  non-convex — iterative algorithms may find local optima

---

### 3. Post-processing / Rotation

#### ICA Rotation (better than Promax for degenerate modes)

Apply FastICA to the degenerate block of SVD vectors instead of Promax. ICA maximises
non-Gaussianity (sparsity is a strong form of non-Gaussianity) to unmix overlapping modes,
treating the mixed canonical vectors as "observed signals" and the sparse modes as "sources."

- ✅ Purpose-built for unmixing degenerate subspaces; often more powerful than Promax
- ⚠️ Permutation/scale ambiguity; assumes statistical independence of underlying sources

#### Direct L1 Rotation

Find an orthogonal rotation matrix `R` applied only to the degenerate block `U_d` that
minimises the entry-wise L1 norm:

$$\min_{R:\, R^\top R = I} \|U_d R\|_1.$$

- ✅ Directly optimises sparsity rather than a proxy (kurtosis)
- ⚠️ Non-convex optimisation on the Stiefel manifold; requires specialised solvers

---

### 4. Reformulation

#### Sparse Regression / Multi-task Lasso

Bypass CCA entirely. Directly solve for the sparse transformation matrix `C` that maps one
basis to the other:

$$\min_C \|B_2 - B_1 C\|_F^2 + \alpha \|C\|_1,$$

where `B1`, `B2` are matrices whose columns are the discretised basis functions. The matrix
`C` is exactly the heatmap we want to be near-identity; the L1 penalty enforces this
directly without any Gram matrix inversion or SVD.

- ✅ Avoids SVD/Gram inversion entirely; most direct path to the desired result
- ⚠️ Requires representing basis functions on a discrete grid; `α` tuning required

#### Joint Diagonalisation

Find a single transformation `A` that simultaneously diagonalises `G1` and makes `G12` as
diagonal as possible:

$$\min_A \; \text{off-diag}\!\left(A^\top G_1 A\right) + \text{off-diag}\!\left(A^\top G_{12} A\right).$$

- ✅ Single optimisation for the joint eigen-basis; no two-step instability
- ⚠️ Complex algorithms; only approximate since `G1` and `G12` generally do not commute

---

### Practical Recommendations (priority order)

| Priority | Method | Reason |
|----------|--------|--------|
| 1 | Tikhonov regularisation | Easiest to try; add `λI` to G1/G2 |
| 2 | Sparse CCA | Best theoretical fit; encodes sparsity in the objective |
| 3 | ICA rotation (replace Promax) | Better unmixing of degenerate SVD blocks |
| 4 | Multi-task Lasso | Complete reformulation; bypasses all ill-conditioning |

---

## Making High-k Modes More Distinguishable

The strategies above treat the ill-conditioning as a numerical problem to be regularised
away. A complementary angle is to ask: can we make the high-k modes **intrinsically more
distinguishable** before forming the Gram matrices? Two natural ideas were evaluated.

### Evaluated Idea 1: Denser Time Grid for High-k Modes

**Proposal:** Use $N_t(k) \propto k$ time points per mode so that high-k oscillations are
better sampled; the Gram matrix inner products should then be more accurate and modes
should become more distinguishable.

**Verdict: No — this does not help.**

Two sinusoids $\cos(\omega_1 \tau)$ and $\cos(\omega_2 \tau)$ become orthogonal only as
the integration length $T \to \infty$. Over a fixed interval $[0, \tau_{\text{fcb}}]$,
the orthogonality condition is:

$$\Delta k \cdot c_s \cdot \tau_{\text{fcb}} \gtrsim 2\pi.$$

This depends on $\tau_{\text{fcb}}$ (fixed), not on the number of sample points $N_t$.
Increasing $N_t$ improves the *numerical accuracy* of the inner product, but the
converged value is still close to 1. The degeneracy is a property of the integration
domain, not of the discretisation.

- ⚠️ Correct for Nyquist compliance (`Δτ < π / (k_{\max} c_s)`), but going beyond that
  does not help.

---

### Evaluated Idea 2: k-Dependent Weighting of G12 Only (Asymmetric)

**Proposal:** Apply a locality kernel $W(k_i, k_j)$ to suppress off-diagonal entries of
$G_{12}$ while leaving $G_1$ and $G_2$ unchanged.

**Verdict: No — breaks the CCA symmetry.**

Modifying $G_{12}$ without modifying $G_1$ and $G_2$ solves a different optimisation
problem (maximising a modified cross-covariance rather than correlation). The whitening
$G_1^{-1/2}$ is still computed from the original ill-conditioned $G_1$, so the numerical
instability remains. The canonical correlation values lose their $[0,1]$ interpretation.

---

### Alternative A: Symmetric k-Dependent Weighting (Corrected Idea 2)

Apply the locality kernel **to all three matrices simultaneously**:

$$G_1' = W \circ G_1, \quad G_2' = W \circ G_2, \quad G_{12}' = W \circ G_{12},$$

where $W_{ij} = \exp\!\left(-\alpha (k_i - k_j)^2 / k_{\text{ref}}^2\right)$ and
$\circ$ denotes the Hadamard (element-wise) product. This is equivalent to CCA in a
modified Hilbert space with a k-localised inner product, and correctly handles the
symmetry of the problem.

- ✅ Principled; suppresses off-diagonal Gram entries uniformly; improves condition number
- ⚠️ Encodes a prior that eigenfunctions are localised in k-space; `α` needs tuning

---

### Alternative B: Basis Pre-Orthogonalisation

Replace the raw high-k columns with their principal components **before** running CCA:

1. Isolate the ill-conditioned high-k block: $X_{1,\text{high-k}}$ shape $(N_t, m)$.
2. SVD: $X_{1,\text{high-k}} = U S V^\top$ — columns of $U$ form an orthonormal basis.
3. Build a **hybrid basis**: original single-k columns at low k, $U$ columns at high k.
4. Repeat for Basis 2, then run standard CCA on the two hybrid bases.

$G_1$ and $G_2$ are block-diagonal and well-conditioned by construction. No
hyperparameter is needed.

- ✅ Most direct and robust; no assumptions beyond what the data says; guaranteed stability
- ⚠️ Requires identifying the boundary between "well-conditioned" and "degenerate" k-blocks

---

### Alternative C: Derivative Regularisation / Sobolev Inner Product *(implemented)*

Modify the inner product to penalise directions where adjacent k-modes look similar,
by adding a k-derivative term to all three Gram matrices:

$$G_\alpha^{\text{reg}} = \sum_p w_p \left[
    (\bar{X}^p_\alpha)^\top \bar{X}^p_\alpha
    \;+\; \lambda \left(\frac{\partial \bar{X}^p_\alpha}{\partial k}\right)^\top
                      \frac{\partial \bar{X}^p_\alpha}{\partial k}
\right],$$

and analogously for $G_{12}^{\text{reg}}$. The derivative $\partial \bar{X}/\partial k$
is computed column-wise via `np.gradient` using the actual k-spacing.

**Key advantage over Tikhonov:** The derivative term is *data-adaptive* — it is large
where modes are nearly degenerate (high k) and small where modes are well-separated
(low k), so it regularises only where needed and does **not** suppress canonical
correlations for well-separated low-k modes.

Implementation: `cca_sparse_reg_analysis.py`

- ✅ Data-adaptive regularisation; preserves CCA symmetry; physically motivated
- ⚠️ Requires loading k-values for finite differences; `λ` still needs tuning

---

### Alternative D: Tikhonov with L-Curve

The original Tikhonov strategy (`+λI`) is sound but the choice of `λ` is critical.
Too large a `λ` suppresses all correlations (only 1 valid mode found in initial test).
The **L-curve method** finds the optimal trade-off:

Plot $\|\mathbf{a}\|^2$ (solution norm) vs $(1 - \sigma)$ (residual) for many $\lambda$
values. The curve forms an "L" shape; the optimal $\lambda$ is at the corner.

Implementation: `cca_sparse_noMix_analysis.py`

- ✅ Simple; `λI` is easy to implement; L-curve is a standard selection criterion
- ⚠️ Biases all canonical correlations uniformly (non-adaptive)

---

### Updated Priority Order

| Priority | Method | File | Notes |
|----------|--------|------|-------|
| 1 | **Derivative regularisation** (Alt C) | `cca_sparse_reg_analysis.py` | Data-adaptive; preserves rho |
| 2 | **Basis pre-orthogonalisation** (Alt B) | — | Most robust; no hyperparameter |
| 3 | **Symmetric k-weighting** (Alt A) | — | Good if locality prior is appropriate |
| 4 | **Tikhonov + L-curve** (Alt D) | `cca_sparse_noMix_analysis.py` | Simple; revisit with L-curve |
| 5 | Sparse CCA | — | Best theoretical fit; expensive |
