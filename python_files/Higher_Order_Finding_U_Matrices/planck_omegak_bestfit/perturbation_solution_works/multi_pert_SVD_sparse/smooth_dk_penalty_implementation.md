


Here is the complete, consolidated Markdown file. You can copy this entire block and provide it directly to Claude to refactor your code.

***

```markdown
# Resolving the Artificial PPS Cut-off with Continuous $k$-Scaled Regularised CCA

## 1. Diagnosis: Physical vs. Numerical Effects in the PPS

In the current Primordial Power Spectrum (PPS) plot (`PPS_power_spectra_cca_2D.pdf`), there are two distinct features at low-$k$:
1. **The suppression and oscillations (wiggles) for $k < 2.8 \times 10^{-3} \text{ Mpc}^{-1}$:** This is a **physical** consequence of mapping continuous flat-space modes onto the discrete, curvature-restricted spatial eigenfunctions ($\nu = 3, 4, 5...$) of a closed universe.
2. **The sharp, sudden cut-off (kink) at exactly $k \approx 2.8 \times 10^{-3} \text{ Mpc}^{-1}$ (Index 13):** This is a **numerical artifact**. 

### Why the artifact occurs:
In `cca_sparse_2D_reg_analysis.py`, the code uses a hard partition threshold (`corr_threshold = 0.9999`). At index 13, the Boltzmann code transitions its internal grid spacing (as seen in `allowedK_diff_nuspacing.pdf`). At this exact grid transition, the two bases become numerically similar enough that `corr_time` crosses `0.9999`. 

The code triggers the `I_degen` block, instantly dropping the CCA algorithm and forcing a perfect 1-to-1 Identity matrix. Because the coefficients instantly become perfect identity vectors, the PPS artificially snaps exactly back to the standard unmixed $\Lambda$CDM line, creating an unphysical "kink."

---

## 2. The Mathematical Solution: $k$-Proportional Diagonal Penalty

To fix this, we must **remove the hard partition** and process all modes through the CCA continuously. However, doing so re-introduces the high-$k$ eigenspace degeneracy (where the SVD solver randomly mixes identical high-$k$ modes). 

Initially, a $dk$-based penalty was considered. However, because the high-$k$ grid transitions to being equally spaced, $dk$ becomes a constant. A constant penalty reverts back to a uniform Ridge penalty ($\gamma I$), which is known to artificially suppress the physical low-$k$ wiggles.

Instead, we introduce a **$k$-Proportional Penalty**. 

### Derivation
We use the wavenumber $k_i$ itself as the penalty weight. We define diagonal penalty matrices for both bases:
$$ K_1 = \text{diag}(k_{1,1},\; k_{1,2},\; \dots,\; k_{1,N_k}) $$
$$ K_2 = \text{diag}(k_{2,1},\; k_{2,2},\; \dots,\; k_{2,N_k}) $$

We modify the CCA objective to penalize the variance, weighted by the wavenumber $k$, controlled by an overall strength parameter $\alpha > 0$:
$$ \max_{\mathbf{a}, \mathbf{b}} \quad \mathbf{a}^\top G_{12} \mathbf{b} $$
Subject to:
$$ \mathbf{a}^\top (G_{1\_2D} + \alpha K_1) \mathbf{a} = 1 $$
$$ \mathbf{b}^\top (G_{2\_2D} + \alpha K_2) \mathbf{b} = 1 $$

The new regularised 2D Gram matrices become:
$$ \tilde{G}_1 = G_{1\_2D} + \alpha \cdot \text{diag}(k\_array\_1) $$
$$ \tilde{G}_2 = G_{2\_2D} + \alpha \cdot \text{diag}(k\_array\_2) $$

### Why this works perfectly:
At low $k$ (e.g., $0.001$), the penalty is microscopically small. The physical mixing represented in the spatial Gram matrices ($S_{12}$) dominates, preserving the wiggles. At high $k$ (e.g., $0.05$), $k$ is 50 times larger. This naturally introduces a strictly increasing symmetry-breaker that forces the SVD solver to isolate the high-$k$ modes perfectly into a diagonal. **The result is a completely smooth transition from physical low-$k$ oscillations to the high-$k$ $\Lambda$CDM asymptote, with no artificial kinks.**

---

## 3. Instructions for Code Modification

Claude, please refactor `cca_sparse_2D_reg_analysis.py` based on the following instructions:

### Step 3.1: Replace `solve_rcca` with `solve_rcca_k_scaled`
Replace the old uniform `gamma` regularisation function with the new $k$-proportional one:

```python
def solve_rcca_k_scaled(G1, G2, G12, k_array_1, k_array_2, alpha=1e-5, ev_threshold=1e-10, rho_min=0.99):
    """
    Regularised CCA using a k-proportional diagonal penalty.
    G1_reg = G1 + alpha * diag(k_array_1)
    
    Because k grows monotonically, this applies virtually zero penalty at low-k 
    (preserving the physical mixing/wiggles) and a strong penalty at high-k 
    (breaking the SVD eigenspace degeneracy without needing a hard partition).
    """
    # Apply the k-scaled penalty
    G1_reg = G1 + alpha * np.diag(k_array_1)
    G2_reg = G2 + alpha * np.diag(k_array_2)
    
    # solve_cca already exists in the imported module
    return solve_cca(G1_reg, G2_reg, G12, ev_threshold, rho_min)
```

### Step 3.2: Remove the Hard Partitioning Logic
1. **Delete** the `partition_indices` function entirely.
2. **Delete** the `assemble_coefficients` function entirely (we no longer assemble hybrid matrices).

### Step 3.3: Update the Main Pipeline Flow
Inside the main `cca_sparse_2D_reg_analysis` function:
1. Remove `corr_threshold` and `gamma` from the function signature and replace them with `alpha=1e-5`.
2. Delete the block labeled `--- PARTITION ---`.
3. Replace the `--- Step 4: Regularised CCA on I_mixed sub-matrices ---` block and the `--- Step 6: Assemble hybrid coefficient matrix ---` block with a single, continuous CCA step using the $k$-values directly:

```python
    # ── Step 4: k-proportional rCCA ───────────────────────────────────────────
    print(f"\n--- Step 4: k-proportional rCCA  (alpha = {alpha}) ---")
    
    A, B, rho, rho_all = solve_rcca_k_scaled(
        G1_2D, G2_2D, G12_2D, k_values_1, k_values_2, alpha, ev_threshold, rho_min)
    
    print(f"  Valid modes found: {len(rho)}")

    # ── Step 5: Sparse rotation ───────────────────────────────────────────────
    if A.shape[1] > 0:
        print("\n--- Step 5: Sparse rotation ---")
        A_sparse, B_sparse = apply_sparse_rotation(A, B, method=rotation_method)
        
        g1 = np.mean([gini(A_sparse[:, i]) for i in range(A_sparse.shape[1])])
        g2 = np.mean([gini(B_sparse[:, i]) for i in range(B_sparse.shape[1])])
        print(f"\n  Mean Gini — Basis 1: {g1:.3f}, Basis 2: {g2:.3f}")
    else:
        print("  No valid modes found — skipping rotation.")
        A_sparse = A.copy()
        B_sparse = B.copy()
```

### Step 3.4: Clean up Plotting and Returns
Since we are no longer splitting the matrix into an "identity block" and an "rCCA block", we need to clean up the plotting and return dictionary:

1. **Delete Partition-Specific Plotting**: Remove `plot_partition` and `plot_heatmap_hybrid` entirely.
2. **Import Standard Heatmap**: Import the standard `plot_heatmap` function from `cca_sparse_analysis` (add it to the import list at the top of the file alongside `plot_coefficients`, etc.).
3. **Update Plotting Calls** (inside the `--- Plotting ---` block): 
   - Remove the call to `plot_partition`.
   - Call `plot_heatmap(A_sparse, B_sparse, output_dir)` instead of `plot_heatmap_hybrid`.
   - Ensure `plot_coefficients` and `plot_reconstructed_timeseries` are called using the unified `A, B, A_sparse, B_sparse, rho` variables (instead of `A_sub`, `B_sub`, `rho_full`, etc.).
4. **Update the `results` dictionary**:
   - **Remove**: `A_sub`, `B_sub`, `A_sub_sparse`, `B_sub_sparse`, `rho_sub`, `rho_all_sub`, `rho_full`, `I_degen`, `I_mixed`, `corr_time`, `corr_threshold`, and `gamma`.
   - **Add/Ensure**: `A=A`, `B=B`, `A_sparse=A_sparse`, `B_sparse=B_sparse`, `rho=rho`, `rho_all=rho_all`, and `alpha=alpha`.

### Step 3.5: Update the Entry Point (`__main__`)
At the very bottom of the file under `if __name__ == "__main__":`:
- Remove `corr_threshold = 0.9999` and `gamma = 1e-4` from the function call.
- Add `alpha = 1e-5` (this is an excellent starting tuning parameter).
- Update the final `print` statements to reflect the unified matrices. For example, print the total number of modes instead of printing the sizes of the `I_degen` and `I_mixed` blocks.

```python
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  k_max             : {results['k_max']}")
    print(f"  alpha             : {results['alpha']}")
    print(f"  Total valid modes : {results['A_sparse'].shape[1]}")
    print(f"  N_k (integerK)    : {results['G1_2D'].shape[0]}")
    print(f"  N_k (allowedK)    : {results['G2_2D'].shape[0]}")
    print("=" * 60)
```
```
