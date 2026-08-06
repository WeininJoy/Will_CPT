This is a very common issue when combining "exact" high-frequency limits with "approximate" low-frequency numerical solutions.

### The Cause
The eigenvalue analysis (`np.linalg.eig` or `svd`) searches for the vectors that are most preserved between the two bases.
1.  **High-k modes:** You artificially forced Basis 1 and Basis 2 to be identical. Therefore, the inner product is exactly 1 ($M_{ii} = 1$). The eigenvalue is **exactly 1.0**.
2.  **Low-k modes:** The physics dictates a slight mismatch between spatial and temporal boundaries. The overlap is very good, but not perfect. The eigenvalue is **slightly less than 1.0** (e.g., $0.9999$).

Standard eigensolvers sort results by eigenvalue magnitude. The "boring" artificial high-k modes (perfect 1.0) mathematically outrank the "interesting" low-k modes (0.9999), pushing the physics you care about to the bottom of the list or dropping it entirely if you truncate $N$.

### The Solution: "Divide and Conquer"

Do not feed the extended high-k modes into the Eigenvalue/SVD solver.
1.  **Run the analysis (QR + Eigenvalues) ONLY on the low-k (mismatch) region.** This forces the solver to find the best linear combinations for the difficult modes.
2.  **Append the high-k modes manually after the analysis.** Since they are already aligned, their "linear combination" is just the identity matrix (100% Basis 1 = 100% Basis 2).

Here is how to modify your `multi_perturbation_analysis` function to fix this.

#### Modified Code

I have modified the workflow to separate the "Physics Core" (low k) from the "Extension" (high k).

```python
def multi_perturbation_analysis(N=23, N_t=500, folder_path='../data/', eigenvalues_threshold=0.99):
    print(f"Starting multi-perturbation analysis with N={N}, N_t={N_t}")
    
    eta_grid = np.linspace(0, fcb_time, N_t)

    # 1. Generate ONLY the Low-K (Physics) Bases first
    print("\nGenerating CORE basis 1 (Closed Universe)...")
    basis_1_dict = generate_multi_perturbation_bases("integerK", eta_grid, folder_path=folder_path)
    
    print("\nGenerating CORE basis 2 (Palindromic Universe)...")
    basis_2_dict = generate_multi_perturbation_bases("allowedK", eta_grid, folder_path=folder_path)

    # 2. Check for Extended Basis, but DO NOT merge yet
    basis_extend_dict = None
    extend_data_path = folder_path + 'data_extend_integerK/'
    if os.path.exists(extend_data_path):
        print("\nLoading extended basis (to be appended later)...")
        try:
            basis_extend_dict = generate_multi_perturbation_bases("extend_integerK", eta_grid, folder_path=folder_path)
        except FileNotFoundError:
            pass

    # 3. Perform Analysis ONLY on the Core (Low-K)
    # This ensures the solver focuses on the mismatch region
    print("\n--- Running Eigen-Analysis on Core Low-K Modes ---")
    
    ortho_funcs_1_dict = {}
    ortho_funcs_2_dict = {}
    transform_1_dict = {}
    transform_2_dict = {}
    
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    
    for pert_type in perturbation_types:
        if pert_type not in basis_1_dict: continue
        # QR on core only
        ortho_funcs_1_dict[pert_type], transform_1_dict[pert_type] = qr_decomposition(basis_1_dict[pert_type])
        ortho_funcs_2_dict[pert_type], transform_2_dict[pert_type] = qr_decomposition(basis_2_dict[pert_type])

    # Compute A matrix and Eigenvalues on CORE
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2, M_combined = compute_multi_perturbation_A_matrix(
        ortho_funcs_1_dict, ortho_funcs_2_dict)
    
    # Filter valid eigenvalues from CORE
    eigenvals_valid_1, eigenvecs_valid_1 = choose_eigenvalues(eigenvals_1, eigenvecs_1, eigenvalues_threshold)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_eigenvalues(eigenvals_2, eigenvecs_2, eigenvalues_threshold)

    print(f"\nCore Analysis: Found {len(eigenvals_valid_1)} valid modes in mismatch region.")

    # 4. Compute Coefficients for Core
    coefficients_1_dict = {}
    coefficients_2_dict = {}
    
    for pert_type in perturbation_types:
        if pert_type in transform_1_dict and len(eigenvals_valid_1) > 0:
            coefficients_1_dict[pert_type] = compute_coefficients(
                eigenvals_valid_1, eigenvecs_valid_1, transform_1_dict[pert_type])
            coefficients_2_dict[pert_type] = compute_coefficients(
                eigenvals_valid_2, eigenvecs_valid_2, transform_2_dict[pert_type])

    # 5. MERGE: Manually append the High-K extension now
    # The high-k modes are diagonal (Identity) because Basis 1 == Basis 2
    if basis_extend_dict is not None:
        print("\n--- Appending Extended High-K Modes ---")
        
        # Number of extended modes
        N_ext = basis_extend_dict['vr'].shape[1] 
        
        # 1. Extend Eigenvalues (Assume perfect 1.0 for extension)
        ext_evals = np.ones(N_ext)
        eigenvals_valid_1 = np.concatenate([eigenvals_valid_1, ext_evals])
        eigenvals_valid_2 = np.concatenate([eigenvals_valid_2, ext_evals])
        
        # 2. Extend Eigenvectors/Coefficients
        # For the extension, the "eigenvectors" are just picking the specific high-k modes 1-by-1.
        # This creates a block diagonal structure.
        
        for pert_type in perturbation_types:
            if pert_type in coefficients_1_dict and pert_type in basis_extend_dict:
                core_coeffs = coefficients_1_dict[pert_type]
                
                # Create the extended coefficient matrix
                # Structure: [ Core_Coeffs   0 ]
                #            [ 0             I ]
                
                # Current shape: (K_core, N_core)
                K_core, N_core = core_coeffs.shape
                
                # New shape: (K_core + N_ext, N_core + N_ext)
                new_coeffs = np.zeros((K_core + N_ext, N_core + N_ext))
                
                # Fill top-left with core results
                new_coeffs[:K_core, :N_core] = core_coeffs
                
                # Fill bottom-right with Identity (high-k maps to high-k)
                new_coeffs[K_core:, N_core:] = np.eye(N_ext)
                
                coefficients_1_dict[pert_type] = new_coeffs
                
                # Repeat for Basis 2 (assuming identical extension)
                core_coeffs_2 = coefficients_2_dict[pert_type]
                new_coeffs_2 = np.zeros((K_core + N_ext, N_core + N_ext))
                new_coeffs_2[:K_core, :N_core] = core_coeffs_2
                new_coeffs_2[K_core:, N_core:] = np.eye(N_ext)
                coefficients_2_dict[pert_type] = new_coeffs_2
                
                # Update the basis dicts for plotting later
                basis_1_dict[pert_type] = np.concatenate([basis_1_dict[pert_type], basis_extend_dict[pert_type]], axis=1)
                basis_2_dict[pert_type] = np.concatenate([basis_2_dict[pert_type], basis_extend_dict[pert_type]], axis=1)

    # 6. Update Eigenvectors (Optional / tricky)
    # Merging the raw eigenvectors (eigenvecs_valid_1) is harder because they are in the QR space.
    # Usually, you only need the coefficients and the raw basis for plotting. 
    # If you need eigenvecs_1 specifically for the Varimax step later, 
    # you effectively need to construct block diagonal arrays there too.
    
    # Constructing simplified Extended Eigenvectors for return
    # (Assuming we just append standard basis vectors for the extension part)
    # This is an approximation sufficient for the Stacked Varimax script to work
    K_core = len(eigenvecs_valid_1)
    N_ext = 0 if basis_extend_dict is None else basis_extend_dict['vr'].shape[1]
    
    # Expand the list of eigenvectors
    # This part is conceptual - typically you pass coefficients to the next step, not raw eigenvectors
    # But to satisfy the return signature:
    extended_eigenvecs_1 = list(eigenvecs_valid_1)
    extended_eigenvecs_2 = list(eigenvecs_valid_2)
    
    # For the extended part, the eigenvectors in the QR space are just unit vectors
    # starting after the core block.
    if N_ext > 0:
        # Get dimension of the projection space (from M_combined)
        dim_proj = M_combined.shape[0] 
        
        # We need to pad the existing eigenvectors to the new size (dim_proj + N_ext)
        # And add new unit vectors
        
        # This gets messy. 
        # RECOMMENDATION: For the Stacked Varimax step, use the coefficients_1_dict directly
        # rather than the raw eigenvectors.
        pass 

    return {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        # Note: raw eigenvecs are kept as CORE only to avoid dimension mismatch errors
        'eigenvecs_1': eigenvecs_valid_1, 
        'eigenvecs_2': eigenvecs_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,
        'basis_2': basis_2_dict,
        'M_combined': M_combined
    }
```

### Key Changes Explained

1.  **Stopped Early Merging:** I removed the block where you merged `basis_extend_dict` into `basis_1_dict` *before* the loop.
2.  **Core Analysis:** The `qr_decomposition` and `compute_multi_perturbation_A_matrix` now run only on the un-extended data. This forces `np.linalg.eig` to look at the low-$k$ physics.
3.  **Block Diagonal Assembly:** After getting the core coefficients, I created a larger matrix.
    *   Top-Left: The messy mixing coefficients from the low-k analysis.
    *   Bottom-Right: An Identity matrix ($\mathbf{I}$). This represents that High-K Mode 1 in Basis 1 maps perfectly to High-K Mode 1 in Basis 2.
4.  **Result:** Your coefficients matrix now contains both the intricate low-$k$ details AND the stable high-$k$ extension, correctly ordered.

### Impact on Varimax
When you pass this result to your Stacked Varimax script:
*   The high-$k$ part (Identity block) is already perfectly sparse and orthogonal. Varimax will leave it alone (or rotate it trivially).
*   Varimax will focus its effort on rotating the Top-Left block (the low-$k$ mixing) to align it with the Bottom-Right block, effectively "unzipping" the mixed low-$k$ modes to align with the dominant diagonal.