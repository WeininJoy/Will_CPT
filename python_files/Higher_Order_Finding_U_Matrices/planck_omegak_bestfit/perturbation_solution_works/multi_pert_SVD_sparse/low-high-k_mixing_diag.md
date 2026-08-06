# Mixing Blocks Diagnosis: `figures_2D/2D_heatmap.pdf`

**Script:** `cca_sparse_2D_analysis.py`
**Data:** `../data/data_integerK_timeseries/` and `../data/data_allowedK_timeseries/`
**Note:** Indices 35–56 (k > 22.5) are buggy — integerK and allowedK are exactly identical
there and must be excluded. The analysis below concerns indices 0–34 (k ≤ 22) only.

---

## Two Distinct Mixing Mechanisms

### 1. High-k mixing — indices 20–34 (k ≈ 13.5–22)

**Your hypothesis is correct.** The two bases converge as k increases: `dk = k_int − k_all`
shrinks from ~0.01 at index 20 to ~0.001 at index 27 and below. This makes the matched
time series nearly identical and the spatial overlap S12[i,i] approach 1, so CCA cannot
resolve the pair. Promax distributes weight across 2 nearby k-indices (e.g. top1 ≈ 0.50,
top2 ≈ 0.45), producing the 2-mode pairing blocks visible in the heatmap.

Degeneracy onset (time-series cosine similarity between matched integerK[i] and allowedK[i]):

| threshold | first index | k_int | dk |
|---|---|---|---|
| corr_time > 0.999  | 16 | 11.169 | 0.029 |
| corr_time > 0.9999 | 20 | 13.552 | 0.011 |
| corr_time > 0.99999 | 28 | 18.317 | −0.001 |

Selected mixed high-k modes (dominant k-index and weight distribution in A_sparse):

| dom_k idx | k_int  | dk      | corr_time  | top1  | top2  |
|-----------|--------|---------|------------|-------|-------|
| 20        | 13.552 | 0.01066 | 0.9999271  | 0.522 | 0.456 |
| 21        | 14.148 | 0.00947 | 0.9999468  | 0.619 | 0.381 |
| 26        | 17.126 | −0.00086| 0.9999868  | 0.613 | 0.318 |
| 28        | 18.317 | −0.00109| 0.9999909  | 0.494 | 0.454 |
| 29        | 18.913 | 0.00039 | 0.9999925  | 0.563 | 0.437 |
| 31        | 20.104 | 0.00121 | 0.9999944  | 0.553 | 0.357 |

The 2-mode mixing is the direct consequence of degeneracy: once `corr_time > 0.9999`,
CCA eigenvectors span a degenerate 2D subspace and Promax finds arbitrary rotations within it.

---

### 2. Low-k mixing — indices 0–13 (k ≈ 1.6–9.4)

**Different cause.** The two bases are distinguishable at low k (corr_time = 0.87–0.99,
`dk` = 0.05–0.19), yet the mixing here is far more severe: modes spread across 7–12
k-indices on average, with individual k-mode weights as low as 0.17–0.21. For example,
dom_k = 9 produces 8 separate CCA modes each with nearly equal weight.

The cause is **strong off-diagonal coupling in G12_t**: at low k the maximum off-diagonal
value in the normalised cross-Gram matrix is 0.47–0.85 of the diagonal. Many k-modes share
overlapping time-series structure (multiple low-k oscillation frequencies interfere), making
the cross-Gram block nearly dense. Promax cannot find sparse directions within a dense block.

---

## Summary Table (indices 0–34)

| k-range | indices | mixing type | cause |
|---|---|---|---|
| k ≲ 9.4  | 0–13  | severe many-mode spreading (7–12 k-modes) | strong off-diagonal G12 coupling |
| 9.4 ≲ k ≲ 13.5 | 14–19 | mild | transitional |
| k ≳ 13.5 | 20–34 | 2-mode pairing | near-degeneracy: dk ≲ 0.01, corr_time > 0.9999 |

---

## Practical Implication

Restricting to k ≤ 22 (indices 0–34) is necessary to remove the buggy identical modes
(indices 35–56), but it does not by itself resolve the mixing. Within indices 0–34:

- **Indices 20–34** will always show 2-mode pairing because `allowedK` has already
  converged to `integerK` — the two bases are physically indistinguishable there.
- **Indices 0–13** suffer from a different, structural problem (dense G12 at low k).

For clean sparse single-k modes, the usable regime is roughly **indices 0–19** (k ≲ 12),
where the two bases are still meaningfully separated (`dk` ≳ 0.017) and the off-diagonal
coupling in G12 is not yet overwhelming.

---

## Proposed Solutions (from Gemini consultation)

### Core problem: ill-conditioning

The mixing stems from the Gram matrices becoming ill-conditioned when modes are similar.
Near-degeneracy makes G12 approach identity, and tiny differences get amplified into
unreliable canonical vectors. A single method applied to the entire k-range will not work
well — the physics is different at low-k and high-k, so the analysis method should reflect that.

---

### Hybrid strategy

#### Step 1 — Formalise the partition

Compute `1 - corr_time[i]` for all matched pairs and plot vs. index. There will be a clear knee:

- **High-k plateau**: `1 - corr_time < 1e-5` → degenerate, force identity
- **Transition slope**: rapidly changing
- **Low-k**: large and variable → distinct modes, physical mixing

Choose a threshold (e.g. `corr_time > 0.99995`) to define `I_degen`. For all `i ∈ I_degen`,
skip CCA entirely and enforce a 1-to-1 identity mapping in the coefficient block.

#### Step 2 — Regularised CCA for `I_mixed`

For the remaining indices (low-k + transition), use **Regularised CCA**:

```
G1_reg = G1 + r * I
G2_reg = G2 + r * I
```

This stabilises the inversion and prevents spurious high-correlation directions from tiny
numerical differences in the transition region, while having minimal effect on the
well-conditioned low-k block.

#### Step 3 — Physics-motivated `dk`-based diagonal penalty

Instead of a uniform `r * I`, use a diagonal penalty matrix `P`:

```
P[i,i] = c / (dk_i² + ε)
```

Large penalty for small `dk` (near-degenerate) → small penalty for large `dk` (distinct).
This injects the physical prior that modes with `dk → 0` should not be trusted by CCA,
without discarding them entirely. `ε` prevents division by zero; `c` is a scaling constant
chosen by cross-validation or physically motivated by the noise floor.

---

### Alternative methods

| Method | Best for | Notes |
|---|---|---|
| Regularised CCA (`r * I`) | Transition region | Simplest fix; uniform stabilisation |
| `dk`-based diagonal penalty | Transition region | Most physically principled |
| Sparse CCA (L1 penalty) | Low-k region | Forces clean sparse mixing; eliminates minor cross-correlations automatically |
| Partial CCA | — | Not applicable here (no third control set) |

---

### Recommended workflow

1. Compute `corr_time[i]` for all matched pairs; choose threshold to split `I_degen` / `I_mixed`
2. **High-k (`I_degen`)**: set coefficient block = identity (no CCA)
3. **Low-k / transition (`I_mixed`)**: extract sub-matrices `G1_sub`, `G2_sub`, `G12_sub`
   and run regularised CCA (Option A: uniform `r`; Option B: `dk`-based penalty; Option C: Sparse CCA)
4. Apply Promax rotation only to the `I_mixed` block
5. Assemble final coefficient matrix: identity block (high-k) + rotated sparse block (low-k/transition)

---

## Mathematical Derivations

### Preliminaries: Standard CCA

The CCA objective is to find vectors **a** and **b** maximising the correlation between
the projected variables `u = X1 a` and `v = X2 b`:

$$
\rho = \frac{\mathbf{a}^T \mathbf{G_{12}} \mathbf{b}}{\sqrt{(\mathbf{a}^T \mathbf{G_1} \mathbf{a})(\mathbf{b}^T \mathbf{G_2} \mathbf{b})}}
$$

Under the unit-variance constraints `a^T G1 a = 1` and `b^T G2 b = 1`, the Lagrangian is:

$$
\mathcal{L} = \mathbf{a}^T \mathbf{G_{12}} \mathbf{b}
  - \frac{\lambda}{2}(\mathbf{a}^T \mathbf{G_1} \mathbf{a} - 1)
  - \frac{\mu}{2}(\mathbf{b}^T \mathbf{G_2} \mathbf{b} - 1)
$$

Setting partial derivatives to zero gives the coupled system `G12 b = ρ G1 a` and
`G12^T a = ρ G2 b`, which reduces to the generalised eigenvalue problems:

$$
\mathbf{G_{12}} \mathbf{G_2}^{-1} \mathbf{G_{12}}^T \mathbf{a} = \rho^2 \mathbf{G_1} \mathbf{a}
$$
$$
\mathbf{G_{12}}^T \mathbf{G_1}^{-1} \mathbf{G_{12}} \mathbf{b} = \rho^2 \mathbf{G_2} \mathbf{b}
$$

Equivalently via the whitened SVD: `G1^{-1/2} G12 G2^{-1/2} = U S V^T`,
with `a_i = G1^{-1/2} u_i`, `b_i = G2^{-1/2} v_i`, `rho_i = S_ii`.

The problem arises when `G1` or `G2` are ill-conditioned (near-degenerate modes), making
their inverses numerically unstable.

---

### Method 1: Regularised CCA (rCCA)

#### Modified objective

Replace the unit-variance constraints with regularised quadratic forms:

$$
\begin{align}
\text{maximise} \quad & \mathbf{a}^T \mathbf{G_{12}} \mathbf{b} \\
\text{subject to} \quad & \mathbf{a}^T (\mathbf{G_1} + r_1 \mathbf{I}) \mathbf{a} \le 1 \\
& \mathbf{b}^T (\mathbf{G_2} + r_2 \mathbf{I}) \mathbf{b} \le 1
\end{align}
$$

Define the regularised Gram matrices:

$$
\mathbf{G_{1,reg}} = \mathbf{G_1} + r_1 \mathbf{I}, \qquad
\mathbf{G_{2,reg}} = \mathbf{G_2} + r_2 \mathbf{I}
$$

The problem is now identical to standard CCA with `G1,reg` and `G2,reg` in place of `G1`
and `G2`. Since `eig(G_reg) = eig(G) + r`, all eigenvalues are now bounded below by `r > 0`,
guaranteeing invertibility regardless of how near-degenerate the original modes are.

#### Modified whitened SVD

1. Compute regularised whitening matrices: `W1 = G1_reg^{-1/2}`, `W2 = G2_reg^{-1/2}`
2. Form `M_reg = W1 G12 W2`
3. SVD: `M_reg = U S V^T`
4. Regularised canonical correlations: `ρ_reg,i = S_ii`
5. Canonical vectors:

$$
\mathbf{a}_i = \mathbf{G_{1,reg}}^{-1/2} \mathbf{u}_i, \qquad
\mathbf{b}_i = \mathbf{G_{2,reg}}^{-1/2} \mathbf{v}_i
$$

#### Geometric effect

Adding `r*I` **shrinks** the covariance ellipsoid toward a sphere (isotropic covariance).
It inflates the shortest axes — the directions of low variance corresponding to near-collinear
modes — preventing CCA from exploiting unstable low-variance directions.
The resulting `ρ_reg` are smaller than the unregularised `ρ`: a classic bias-variance
tradeoff (less variance in the solution, at the cost of slightly underestimating correlations).

#### Choosing `r`

- **Cross-validation**: split the `N_t` time axis into K folds; choose `(r1, r2)` that
  maximises hold-out correlation
- **Ledoit-Wolf**: analytical optimal shrinkage intensity applied separately to `G1` and `G2`
- **Trace heuristic**: `r = α · trace(G) / N_k` with `α ~ 1e-3 – 1e-4`

#### Physical interpretation

Every mode direction carries a small isotropic "noise floor" `r`. CCA is prevented from
finding spurious perfect correlations based on numerical artifacts in near-degenerate subspaces.

#### Limitation

The penalty `r*I` is **uniform** — it penalises all mode directions equally, including the
well-separated low-k modes where physical mixing is the signal of interest.

---

### Method 2: Sparse CCA (L1 penalty)

#### Modified objective

Add L1 constraints directly on the canonical vectors:

$$
\begin{align}
\text{maximise}_{\mathbf{a},\mathbf{b}} \quad & \mathbf{a}^T \mathbf{G_{12}} \mathbf{b} \\
\text{subject to} \quad & \mathbf{a}^T \mathbf{G_1} \mathbf{a} \le 1, \quad
                           \mathbf{b}^T \mathbf{G_2} \mathbf{b} \le 1 \\
& \|\mathbf{a}\|_1 \le c_1, \quad \|\mathbf{b}\|_1 \le c_2
\end{align}
$$

The L1 constraints have no closed-form eigendecomposition and must be solved iteratively.

#### Iterative algorithm (Penalised Matrix Decomposition, Witten et al. 2009)

For the first canonical pair:

1. Initialise **b** (e.g. first right singular vector of `G12`)
2. **Repeat until convergence:**
   - Fix **b**, update **a**:
     - Compute `x = G12 b`
     - Solve `a ← argmax_a (a^T x)` subject to `a^T G1 a = 1`, `||a||_1 ≤ c1`
     via **soft-thresholding** on the whitened `x`:
     `S(x, Δ) = sgn(x) · max(0, |x| - Δ)`
   - Fix **a**, update **b** symmetrically with `y = G12^T a`
3. For subsequent pairs, deflate: `G12 ← G12 - (a1^T G12 b1) · G1 a1 b1^T G2`,
   then repeat

The soft-thresholding operator produces **exact zeros**, inducing genuine sparsity.

#### Contrast with Promax rotation

| | Promax | Sparse CCA |
|---|---|---|
| When | Post-hoc rotation of dense CCA output | Sparse directions found during optimisation |
| Zeros | Approximate (small, not zero) | Exact (by construction) |
| Orthogonality | Oblique (non-orthogonal) | G-orthogonal across pairs |
| Stability | Inherits ill-conditioning of CCA | Can be combined with rCCA pre-conditioning |

#### Choosing `c1`, `c2`

- `c = sqrt(N_k)` → no L1 penalty (standard CCA)
- `c = 1` → maximum sparsity (only one non-zero component)
- **Permutation test**: permute rows of `X2` to break true correlations; choose the
  smallest `c` that gives `ρ` significantly above the permutation null distribution

#### Physical interpretation

Imposes the prior that physical mixing is **sparse**: each `allowedK` mode is a combination
of only a few nearby `integerK` modes, not a dense mix of all k-values. Particularly
powerful for interpreting the low-k regime where you want to understand the mixing structure.

#### Limitation

Still requires whitening and is sensitive to ill-conditioned `G1`, `G2`.
Best practice: apply rCCA pre-conditioning to stabilise the Gram matrices first, then
run Sparse CCA on the regularised matrices.

---

### Method 3: `dk`-based Diagonal Penalty

#### Motivation

Standard rCCA penalises all directions uniformly. But we have mode-specific prior
knowledge: the physical degeneracy of mode `i` is directly measured by
`dk_i = |k_int_i - k_all_i|`. This motivates a **Tikhonov regularisation** with
an anisotropic, physically-motivated penalty.

#### Penalty matrix

Define a diagonal matrix **P**:

$$
P_{ii} = \frac{c}{dk_i^2 + \varepsilon}
$$

- Large `dk_i` (well-separated modes, low-k region) → small `P_ii` → minimal regularisation
- Small `dk_i` (near-degenerate modes, high-k region) → large `P_ii` → strong regularisation

The modified Gram matrices are:

$$
\mathbf{G_{1,dk}} = \mathbf{G_1} + \mathbf{P}, \qquad
\mathbf{G_{2,dk}} = \mathbf{G_2} + \mathbf{P}
$$

#### Modified whitened SVD

1. Compute `G1,dk = G1 + P` and `G2,dk = G2 + P`
2. Whitening: `W1 = G1,dk^{-1/2}`, `W2 = G2,dk^{-1/2}`
3. `M_dk = W1 G12 W2`
4. SVD: `M_dk = U S V^T`
5. Canonical vectors: `a_i = G1,dk^{-1/2} u_i`, `b_i = G2,dk^{-1/2} v_i`

#### Geometric effect

The penalty `P` adds a large **variance floor** specifically along the k-directions
where the two bases nearly coincide. CCA is heavily penalised for finding correlations
in those directions and is forced to look for signal in the subspace where the bases
are physically distinct. Directions with large `dk_i` are barely affected.

This is the key difference from rCCA:

| | rCCA (`r*I`) | `dk`-penalty (`P`) |
|---|---|---|
| Penalty type | Isotropic (uniform) | Anisotropic (mode-specific) |
| Effect on low-k | Suppresses physical mixing slightly | Negligible effect |
| Effect on high-k | Suppresses degeneracy (with low-k side-effect) | Selectively suppresses degeneracy |
| Prior knowledge used | None | `dk_i` for each mode |

#### Choosing `c` and `ε`

- **`ε`**: set to `min(dk_i^2)` over non-zero `dk_i`, or to machine precision scale
- **`c`**: choose so that the penalty on the most degenerate mode equals a fraction `α`
  of the average eigenvalue of `G1`:

$$
\frac{c}{dk_{min}^2 + \varepsilon} \approx \alpha \cdot \frac{\text{trace}(\mathbf{G_1})}{N_k}
\quad \Rightarrow \quad
c \approx \alpha \cdot (dk_{min}^2 + \varepsilon) \cdot \frac{\text{trace}(\mathbf{G_1})}{N_k}
$$

Tune the dimensionless `α` (e.g. `α ~ 0.1 – 10`) by cross-validation or by inspecting
the canonical correlations of the near-degenerate modes.

#### Physical interpretation

The penalty directly encodes the physical belief: *modes where the two quantisation
schemes agree (dk ≈ 0) carry no information about the difference between the bases.*
The algorithm is explicitly told to distrust those directions, concentrating its
explanatory power on the modes where `integerK` and `allowedK` genuinely differ.

#### Limitation

- Requires `dk_i` to be a reliable proxy for numerical instability (it is, in this problem)
- Still produces dense canonical vectors; Promax rotation is still needed afterwards,
  but now acts on a much more stable and physically meaningful input
