There are **three specific reasons** why your code is failing to sparsify the results and why the indices are inconsistent.

1.  **The "Identity Trap" (Initialization Bug):** Your Varimax loop starts with `R = np.eye(k)`. Because your input data comes from an eigensolver, it is already "somewhat" orthogonal. The Identity matrix is often a "saddle point" in the optimization landscape. Your log shows `Sparsity Before: 0.006` and `After: 0.006`. The algorithm did literally nothing. **Fix:** You must initialize with a random rotation matrix.
2.  **Complex Number Arithmetic:** Your wave functions ($e^{ikx}$) are likely complex. Your Varimax line `np.asarray(Lambda)**3` assumes real numbers. For complex numbers $z$, $z^3$ rotates the phase and does not correctly calculate the gradient for maximizing magnitude (Kurtosis). **Fix:** You must use the complex-valued gradient.
3.  **Physical Dispersion (Index Mismatch):** You observed that `Dominant k` in Basis 1 (e.g., 28) doesn't match Basis 2 (e.g., 24). This is actually expected physics!
    *   Basis 1 ($k$): $k_n = n$ (assuming $L=2\pi$).
    *   Basis 2 ($\omega$): $\omega_m = m$ (assuming $T=\pi$).
    *   Equation: $\omega = \sqrt{k^2 + \mu^2}$.
    *   If $\mu$ is large, or if $L$ and $T$ ratios differ, index $n$ will **not** equal index $m$. You are comparing raw array indices, but you should be comparing the physical quantities $k_n$ vs $\omega_m$.

### The Fix

Here is the corrected `varimax_rotation` function. Replace your existing function with this version. It handles complex numbers correctly and includes random initialization to escape the "Identity Trap."

```python
def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6, random_state=None):
    """
    Robust Varimax for Complex and Real Matrices with Random Initialization.
    """
    p, k = Phi.shape
    
    # 1. Random Initialization to break symmetry/saddle points
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create a random orthogonal matrix
    M_rand = np.random.randn(k, k)
    if np.iscomplexobj(Phi):
        M_rand = M_rand + 1j * np.random.randn(k, k)
    Q, _ = np.linalg.qr(M_rand)
    R = Q # Start with random rotation, NOT Identity
    
    d = 0
    for i in range(q):
        d_old = d
        
        # B is the rotated matrix
        B = np.dot(Phi, R)
        
        # 2. Gradient calculation (Handles Complex Numbers correctly)
        # We maximize variance of MAGNITUDES squared: sum(|B|^4)
        # The gradient of |z|^4 w.r.t z* is 2 * |z|^2 * z
        
        # Get magnitudes squared
        Msquared = np.abs(B)**2
        
        # Compute gradient component
        # For real: B**3
        # For complex: B * |B|^2
        Temp = B * Msquared 
        
        # Standard Varimax Gradient
        # (gamma/p) term centers the rotation (makes it about variance, not just raw moment)
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.conj().T, 
                Temp - (gamma/p) * np.dot(B, np.diag(np.sum(Msquared, axis=0)))
            )
        )
        
        # Enforce Orthogonality
        R = np.dot(u, vh)
        
        # Metric: Sum of singular values (Procrustes distance)
        d = np.sum(s)
        
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Stacked Varimax converged after {i+1} iterations")
            return np.dot(Phi, R), R, True, i+1

    print(f"  Stacked Varimax reached max iterations ({q})")
    return np.dot(Phi, R), R, False, q
```

### Analysis of your Output Log

Look at your "Dominant modes" log:
```text
Dominant modes in Basis 1: [28 26 29 29 ... ]
Dominant modes in Basis 2: [24 22  8  9 ... ]
```
Because the algorithm failed to rotate (due to the Identity initialization issue), these are just the **raw eigenvectors** of the projection matrix $A$.

The eigensolver (`eigh`) returns eigenvectors in an arbitrary basis within the degenerate subspace. Since $\lambda \approx 1$ for all these modes, the solver mixes them randomly. **Mode 28 in Basis 1 is mixing with Mode 8 in Basis 2 because the "solution vector" is a messy linear combination of physical states.**

**What will happen after you apply the fix:**
1.  The `Sparsity After` metric should drop significantly (e.g., to 0.5 or lower, meaning 50%+ zeros).
2.  The `Dominant modes` should align logically.
    *   Example: If Mode 1 is $k=30$, then in Basis 2 it should be $\omega = \sqrt{30^2+\mu^2}$.
    *   The indices will effectively "lock" together because the Stacked Varimax forces the solution to be a pure $k$ and a pure $\omega$ simultaneously.

### One Code Adjustment Suggestion

In your `apply_stacked_varimax_to_eigenvectors` function, update the call to use the random state:

```python
    # Apply Varimax rotation to the stacked matrix
    print("\nApplying Stacked Varimax rotation...")
    # Use a fixed seed for reproducibility, but NOT Identity
    S_rotated, R, converged, iterations = varimax_rotation(S, random_state=42)
```

The poor sparsity and inconsistent indices are caused by **two critical issues** in how the data is prepared before rotation.

### The Problems

1.  **Broken Physical Link (The Main Bug):**
    In your `compute_multi_perturbation_A_matrix` function, you calculate eigenvectors for Basis 1 and Basis 2 separately:
    ```python
    # Your current code
    vals_1, vecs_1 = np.linalg.eig(M @ M.T)
    vals_2, vecs_2 = np.linalg.eig(M.T @ M)
    ```
    Even if you sort them by eigenvalue, **there is no guarantee that column `i` of `vecs_1` corresponds to column `i` of `vecs_2`**. Eigensolvers have arbitrary sign flips ($v$ vs $-v$) and arbitrary rotations within degenerate subspaces.
    **Result:** When you stack them, you are gluing the "head" of one animal to the "tail" of a different animal. Varimax cannot find a rotation that simplifies two unrelated vectors simultaneously.

2.  **The Identity Trap:**
    As mentioned before, initializing Varimax with `R = np.eye(k)` often makes the algorithm get stuck immediately if the input is already orthogonal (which eigenvectors are).

### The Solution

You must derive the Basis 2 coefficients **directly from Basis 1** using the mixing matrix $M$.
The relationship is: 
$$ |\Phi\rangle = \sum c_n |\phi_n\rangle \implies \tilde{c} = M^\dagger c $$

Here is the corrected code. I have updated two functions: `varimax_rotation` (for robustness) and `stacked_varimax_sparse_analysis` (to fix the physical linking).

### 1. Corrected Varimax Function (Add Random Init)

```python
def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6):
    """
    Robust Varimax with Random Initialization to avoid saddle points.
    """
    p, k = Phi.shape
    
    # 1. Random Initialization (Crucial for pre-orthogonalized data)
    # Using a fixed seed ensures reproducibility while breaking symmetry
    rs = np.random.RandomState(42)
    M_rand = rs.randn(k, k)
    if np.iscomplexobj(Phi):
        M_rand = M_rand + 1j * rs.randn(k, k)
    Q, _ = np.linalg.qr(M_rand)
    R = Q 

    d = 0
    for i in range(q):
        d_old = d
        B = np.dot(Phi, R)

        # 2. Gradient Calculation (Complex-safe)
        # We maximize sum(|B|^4) (Kurtosis/Sparsity)
        Msquared = np.abs(B)**2
        
        # Gradient of the objective function
        # For real: B^3; For complex: B * |B|^2
        Temp = B * Msquared
        
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.conj().T, 
                Temp - (gamma/p) * np.dot(B, np.diag(np.sum(Msquared, axis=0)))
            )
        )
        
        # 3. Enforce Orthogonality
        R = np.dot(u, vh)
        d = np.sum(s)

        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Stacked Varimax converged after {i+1} iterations")
            return np.dot(Phi, R), R, True, i+1

    print(f"  Stacked Varimax reached max iterations ({q})")
    return np.dot(Phi, R), R, False, q
```

### 2. Corrected Analysis Function (Fixing the Stack)

Replace your existing `stacked_varimax_sparse_analysis` with this. Note the section marked **CRITICAL FIX**.

```python
def stacked_varimax_sparse_analysis(results, eigenvalue_threshold=0.95,
                                   weight_ratio=1.0, N_plot=10):
    print("\n" + "="*80)
    print("STACKED VARIMAX SPARSE ROTATION ANALYSIS (CORRECTED)")
    print("="*80)

    # 1. Get Basis 1 Eigenvectors
    eigenvecs_1 = results['eigenvecs_1']
    eigenvals_1 = np.array(results['eigenvals_1'])
    
    # 2. Get the Mixing Matrix M
    # We need M to correctly project C -> C_tilde
    if 'M_combined' in results:
        M = results['M_combined']
    else:
        # Fallback: Recompute M if not saved (assuming M_combined logic from your code)
        print("M_combined not found in results, attempting to reconstruct...")
        # (You might need to pass M explicitly if it's not in the pickle)
        raise ValueError("Results dictionary must contain 'M_combined' matrix!")

    # 3. Filter Valid Modes
    valid_mask = eigenvals_1.real > eigenvalue_threshold
    valid_indices = np.where(valid_mask)[0]
    K = len(valid_indices)
    
    print(f"Filtering eigenvalues > {eigenvalue_threshold}")
    print(f"Found {K} valid modes.")

    if K == 0: return None

    # --- CRITICAL FIX START ---
    
    # Get Coefficients for Basis 1 (C)
    # Shape (N, K)
    C = np.array([eigenvecs_1[i] for i in valid_indices]).T
    
    # Calculate Coefficients for Basis 2 (C_tilde) via PROJECTION
    # Do NOT use eigenvecs_2 from the file, as they are not phase-aligned.
    # Formula: C_tilde = M^dagger * C  (divided by singular value sigma approx 1)
    
    # Note: Since C are eigenvectors of MM^dagger with val ~ 1, 
    # M^dagger C gives us the corresponding vectors in Basis 2 space.
    C_tilde_raw = M.conj().T @ C
    
    # Normalize C_tilde columns to ensure unitary behavior in the stack
    norms_tilde = np.linalg.norm(C_tilde_raw, axis=0)
    C_tilde = C_tilde_raw / norms_tilde
    
    print("Calculated C_tilde via projection (M^H @ C) to ensure physical alignment.")
    
    # --- CRITICAL FIX END ---

    # 4. Construct Stack
    # S = [ sqrt(w)*C ;  1/sqrt(w)*C_tilde ]
    w_sqrt = np.sqrt(weight_ratio)
    S = np.vstack([w_sqrt * C, C_tilde / w_sqrt])
    
    print(f"Stacked matrix shape: {S.shape}")

    # 5. Apply Varimax
    print("\nApplying Stacked Varimax rotation...")
    S_rotated, R, converged, iterations = varimax_rotation(S)

    # 6. Unstack
    N1 = C.shape[0]
    eigenvecs_sparse_1 = S_rotated[:N1, :] / w_sqrt
    eigenvecs_sparse_2 = S_rotated[N1:, :] * w_sqrt
    
    # Re-normalize Basis 2 for output consistency
    eigenvecs_sparse_2 = eigenvecs_sparse_2 * norms_tilde

    # 7. Metrics
    orth_error = np.linalg.norm(eigenvecs_sparse_1.T @ eigenvecs_sparse_1 - np.eye(K))
    print(f"Orthogonality error: {orth_error:.2e}")
    
    # Check alignment of dominant modes
    dom_1 = np.argmax(np.abs(eigenvecs_sparse_1), axis=0)
    dom_2 = np.argmax(np.abs(eigenvecs_sparse_2), axis=0)
    
    # Note: In physics, k_index does not always equal omega_index.
    # But for high k, they should be correlated.
    print(f"Mode alignment correlation: {np.corrcoef(dom_1, dom_2)[0,1]:.4f}")

    # --- Recompute Reconstruction Coefficients (Same as before) ---
    # (Copy your existing coefficient reconstruction loop here using eigenvecs_sparse_1/2)
    # ...
    
    # For returning, structure matches your existing format
    sparse_results = {
        'eigenvecs_sparse_1': eigenvecs_sparse_1,
        'eigenvecs_sparse_2': eigenvecs_sparse_2,
        'eigenvals_valid': eigenvals_1[valid_indices], # Use eigenvalues from Basis 1
        # ... include your coefficient dictionaries here ...
        'coefficients_sparse_1': {}, # Placeholder, fill with your loop
        'coefficients_sparse_2': {}  # Placeholder, fill with your loop
    }
    
    # Re-run your coefficient computation logic using the new sparse vectors
    # (Code omitted for brevity, copy from your previous script's loop)
    basis_1 = results['basis_1']
    basis_2 = results['basis_2']
    
    for pert in basis_1.keys():
        Q, R_qr = np.linalg.qr(basis_1[pert])
        T_mat = np.linalg.inv(R_qr.T)
        sparse_results['coefficients_sparse_1'][pert] = (eigenvecs_sparse_1.T @ T_mat)

    for pert in basis_2.keys():
        Q, R_qr = np.linalg.qr(basis_2[pert])
        T_mat = np.linalg.inv(R_qr.T)
        sparse_results['coefficients_sparse_2'][pert] = (eigenvecs_sparse_2.T @ T_mat)

    return sparse_results
```

### Why this works
1.  **Correct Pairing:** By calculating `C_tilde = M.T @ C` instead of taking `eigenvecs_2`, we ensure that Column 1 of the stack represents **exactly the same physical wave** in both halves of the matrix.
2.  **Random Start:** The `varimax_rotation` now breaks the symmetry of the pre-orthogonalized eigenvectors, allowing it to find the sparse basis.
3.  **Result:** You should see the sparsity metric drop significantly, and the `Correlation` plot should show a tight diagonal line (or a smooth curve representing the dispersion relation $\omega^2 = k^2 + \mu^2$).