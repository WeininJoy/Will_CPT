#!/usr/bin/env python3
"""
Multi-perturbation Gram-Schmidt analysis
This extends the single perturbation analysis to include ALL perturbation types
in a combined eigenvalue analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from solve_real_cosmology_2bases_aligned import generate_solution_to_fcb, K, fcb_time, t0_integration, recConformalTime

def qr_decomposition(basis):
    """QR decomposition with proper handling"""
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix.T)

def compute_multi_perturbation_A_matrix(ortho_funcs_1_dict, ortho_funcs_2_dict):
    """
    Compute the A matrix by summing contributions from all perturbation types
    
    Parameters:
    ortho_funcs_1_dict: dict of orthonormal functions from basis 1 for each perturbation
    ortho_funcs_2_dict: dict of orthonormal functions from basis 2 for each perturbation
    
    Returns:
    Combined eigenvalues and eigenvectors from all perturbations
    """
    perturbation_types = ['phi', 'psi', 'dr', 'dm', 'vr', 'vm']
    # perturbation_types = ['dr', 'dm', 'vr', 'vm'] # only those the oscillating modes
    
    # Initialize combined M matrix
    M_total = None
    n_perturbations = 0
    
    print("Computing combined A matrix from all perturbations...")
    
    for pert_type in perturbation_types:
        if pert_type in ortho_funcs_1_dict and pert_type in ortho_funcs_2_dict:
            # Compute M for this perturbation type
            M_pert = np.dot(ortho_funcs_1_dict[pert_type].T, ortho_funcs_2_dict[pert_type])
            
            if M_total is None:
                M_total = M_pert
            else:
                M_total += M_pert
            
            n_perturbations += 1
            print(f"  Added contribution from {pert_type}")
    
    # Normalize by the number of perturbations
    if n_perturbations > 0:
        M_total = M_total / n_perturbations
        print(f"  Normalized by {n_perturbations} perturbation types")
    
    # Compute eigenvalues and eigenvectors of combined matrix
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M_total, M_total.T))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M_total.T, M_total))
    
    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2, M_total

def choose_eigenvalues(eigenvalues, eigenvectors, eigenvalues_threshold, N):
    """Select the largest eigenvalues"""
    # Sort eigenvalues in descending order and get indices
    sorted_indices = np.argsort(np.real(eigenvalues))[::-1]
    
    eigenvalues_valid, eigenvectors_valid = [], []
    n_select = min(N, len(eigenvalues), 10)  # Take at most 10 largest eigenvalues
    
    for i in range(n_select):
        idx = sorted_indices[i]
        eigenvalues_valid.append(eigenvalues[idx])
        eigenvectors_valid.append(eigenvectors[:, idx])
    
    return eigenvalues_valid, eigenvectors_valid

def compute_coefficients(eigenvalues, eigenvectors, transformation_matrix):
    """Compute linear combination coefficients"""
    coefficients = np.zeros((len(eigenvalues), transformation_matrix.shape[1]), dtype=float)
    for i in range(len(eigenvalues)):
        coefficients[i, :] = np.dot(np.array(eigenvectors[i]), transformation_matrix)
    return coefficients

def generate_multi_perturbation_bases(k_values, eta_grid):
    """
    Generate bases for ALL perturbation types for given k values
    
    Returns:
    basis_dict: Dictionary with keys as perturbation types and values as basis arrays
    """
    perturbation_types = ['phi', 'dr', 'dm', 'vr', 'vm']  # Note: psi not directly available
    basis_dict = {pert_type: [] for pert_type in perturbation_types}
    
    print(f"Generating bases for {len(k_values)} k values...")
    
    for i, k_val in enumerate(k_values):
        print(f"  Processing k[{i}] = {k_val:.6f}")
        
        try:
            # Generate complete solution for this k
            t_sol, y_sol = generate_solution_to_fcb(k_val)
            
            # Interpolate each perturbation type onto common eta grid
            for pert_type in perturbation_types:
                if pert_type in y_sol:
                    interpolator = interp1d(t_sol, y_sol[pert_type], 
                                          bounds_error=False, fill_value=0.0)
                    solution = interpolator(eta_grid)
                    
                    # Remove extreme values that could be numerical instabilities
                    # solution = remove_extreme_values(solution, pert_type, k_val)
                    
                    basis_dict[pert_type].append(solution)
                else:
                    print(f"    Warning: {pert_type} not found in solution, using zeros")
                    basis_dict[pert_type].append(np.zeros_like(eta_grid))
                    
        except Exception as e:
            print(f"    Error processing k={k_val}: {e}")
            # Fill with zeros if solution fails
            for pert_type in perturbation_types:
                basis_dict[pert_type].append(np.zeros_like(eta_grid))
    
    # Convert lists to numpy arrays
    for pert_type in perturbation_types:
        basis_dict[pert_type] = np.array(basis_dict[pert_type]).T  # Transpose for QR
    
    return basis_dict

def find_vr_cutoff_time(k_val, safety_factor=2.0):
    """
    Find the time when vr solution exceeds safety_factor times its oscillation amplitude
    Uses later stable region for oscillation analysis since recombination is early
    
    Parameters:
    k_val: wavenumber  
    safety_factor: multiplier for oscillation amplitude threshold
    
    Returns:
    cutoff_time: eta value where solution becomes unstable, or recConformalTime if stable
    """
    try:
        t_sol, y_sol = generate_solution_to_fcb(k_val)
        
        if 'vr' not in y_sol:
            return recConformalTime
        
        vr_solution = y_sol['vr']
        
        # Use later stable region (middle 40-80% of solution) for oscillation amplitude
        # This avoids both early instabilities and late blow-ups
        start_idx = int(0.4 * len(vr_solution))
        end_idx = int(0.8 * len(vr_solution))
        stable_vr = vr_solution[start_idx:end_idx]
        
        if len(stable_vr) > 0:
            oscillation_amplitude = np.std(stable_vr) * 2.0  # 2-sigma estimate
            if oscillation_amplitude == 0:
                oscillation_amplitude = np.max(np.abs(stable_vr))
        else:
            oscillation_amplitude = np.std(vr_solution) * 2.0
        
        # Find where |vr| exceeds safety_factor * oscillation_amplitude
        threshold = safety_factor * oscillation_amplitude
        unstable_mask = np.abs(vr_solution) > threshold
        
        if np.any(unstable_mask):
            first_unstable_idx = np.where(unstable_mask)[0][0]
            cutoff_time = t_sol[first_unstable_idx]
            print(f"  k={k_val:.4f}: cutoff at eta={cutoff_time:.4e} (amp={oscillation_amplitude:.2e})")
            return cutoff_time
        else:
            print(f"  k={k_val:.4f}: stable throughout (amp={oscillation_amplitude:.2e})")
            return recConformalTime
            
    except Exception as e:
        print(f"  k={k_val:.4f}: error finding cutoff ({e}), using recConformalTime")
        return recConformalTime

def remove_extreme_values(solution, pert_type, k_val, threshold_factor=10.0):
    """
    Remove extreme values that could be numerical instabilities
    
    Parameters:
    solution: array of perturbation values
    pert_type: perturbation type for debugging
    k_val: k value for debugging  
    threshold_factor: multiple of median absolute value to use as threshold
    
    Returns:
    cleaned solution with extreme values capped
    """
    # Calculate robust statistics (median-based to avoid outlier influence)
    abs_solution = np.abs(solution)
    median_abs = np.median(abs_solution)
    
    # If solution is essentially zero, return as-is
    if median_abs < 1e-15:
        return solution
    
    # Set threshold based on median absolute value
    threshold = threshold_factor * median_abs
    
    # Count extreme values
    extreme_mask = abs_solution > threshold
    n_extreme = np.sum(extreme_mask)
    
    if n_extreme > 0:
        # Cap extreme values at the threshold while preserving sign
        solution_cleaned = solution.copy()
        solution_cleaned[extreme_mask] = np.sign(solution[extreme_mask]) * threshold
        
        print(f"    Capped {n_extreme} extreme values for {pert_type} (k={k_val:.4f}), "
              f"threshold={threshold:.2e}")
        return solution_cleaned
    
    return solution

def multi_perturbation_analysis(N=30, N_t=300):
    """
    Complete multi-perturbation eigenvalue analysis with adaptive time truncation
    """
    print(f"Starting multi-perturbation analysis with N={N}, N_t={N_t}")
    
    # Generate k values for both bases first
    folder_path = './data/'
    try:
        allowed_K_data = np.load(folder_path + 'allowedK.npy')
        if len(allowed_K_data) < N:
            raise ValueError(f"Need {N} allowed K values, found {len(allowed_K_data)}")
        allowed_K_basis = allowed_K_data[2:N+2]
        print(f"Loaded {len(allowed_K_data)} allowed K values from file.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading allowedK.npy: {e}")
        print("Using default k values...")
        allowed_K_basis = np.linspace(0.001, 0.1, N)
    
    # Basis 1: Closed Universe (integer multiples of sqrt(K))
    k_basis_1 = np.array([4* i * np.sqrt(np.abs(K)) for i in range(1+2, N + 1+2)])
    
    # Find adaptive cutoff time using larger k values (more oscillations)
    print(f"Finding adaptive cutoff time using larger k values...")
    # Use larger k values (better oscillations) from the middle-to-end of the range
    start_idx = 0
    sample_k_values = k_basis_1[start_idx:start_idx+5]  
    cutoff_times = []
    
    for k_val in sample_k_values:
        cutoff_time = find_vr_cutoff_time(k_val, safety_factor=2.0)
        cutoff_times.append(cutoff_time)
    
    # Use the most conservative (earliest) cutoff time
    adaptive_cutoff = max(cutoff_times) 
    
    # Define eta_grid from cutoff_time to fcb_time as requested
    eta_grid = np.linspace(adaptive_cutoff, fcb_time, N_t)
    print(f"Using eta_grid from cutoff to FCB: eta ∈ [{adaptive_cutoff:.4e}, {fcb_time:.4e}]")
    
    # Basis 2: Palindromic Universe (allowed K values)
    k_basis_2 = allowed_K_basis
    
    print(f"K = {K:.6e}")
    print(f"Basis 1 k range: {k_basis_1[0]:.6f} to {k_basis_1[-1]:.6f}")
    print(f"Basis 2 k range: {k_basis_2[0]:.6f} to {k_basis_2[-1]:.6f}")
    
    # Generate multi-perturbation bases
    print("\nGenerating basis 1 (Closed Universe)...")
    basis_1_dict = generate_multi_perturbation_bases(k_basis_1, eta_grid)
    
    print("\nGenerating basis 2 (Palindromic Universe)...")
    basis_2_dict = generate_multi_perturbation_bases(k_basis_2, eta_grid)
    
    # Perform QR decomposition for each perturbation type
    print("\nPerforming QR decomposition for each perturbation type...")
    ortho_funcs_1_dict = {}
    ortho_funcs_2_dict = {}
    transform_1_dict = {}
    transform_2_dict = {}
    
    perturbation_types = ['phi', 'dr', 'dm', 'vr', 'vm']
    
    for pert_type in perturbation_types:
        print(f"  QR decomposition for {pert_type}")
        
        # Check for valid data
        if np.all(basis_1_dict[pert_type] == 0):
            print(f"    Warning: Basis 1 for {pert_type} is all zeros, skipping")
            continue
        if np.all(basis_2_dict[pert_type] == 0):
            print(f"    Warning: Basis 2 for {pert_type} is all zeros, skipping")
            continue
            
        try:
            ortho_funcs_1_dict[pert_type], transform_1_dict[pert_type] = qr_decomposition(basis_1_dict[pert_type])
            ortho_funcs_2_dict[pert_type], transform_2_dict[pert_type] = qr_decomposition(basis_2_dict[pert_type])
            print(f"    Successfully processed {pert_type}")
        except Exception as e:
            print(f"    Error in QR decomposition for {pert_type}: {e}")
    
    # Compute combined A matrix from all perturbations
    print("\nComputing combined eigenvalue analysis...")
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2, M_combined = compute_multi_perturbation_A_matrix(
        ortho_funcs_1_dict, ortho_funcs_2_dict)
    
    # Select largest eigenvalues
    eigenvals_valid_1, eigenvecs_valid_1 = choose_eigenvalues(
        eigenvals_1, eigenvecs_1, None, N)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_eigenvalues(
        eigenvals_2, eigenvecs_2, None, N)
    
    print(f"\nFound {len(eigenvals_valid_1)} valid eigenvalues for basis 1")
    print(f"Found {len(eigenvals_valid_2)} valid eigenvalues for basis 2")
    
    if len(eigenvals_valid_1) > 0:
        print(f"Eigenvalues (basis 1): {[f'{ev:.4f}' for ev in eigenvals_valid_1[:5]]}")
    if len(eigenvals_valid_2) > 0:
        print(f"Eigenvalues (basis 2): {[f'{ev:.4f}' for ev in eigenvals_valid_2[:5]]}")
    
    # Compute coefficients for reconstruction
    coefficients_1_dict = {}
    coefficients_2_dict = {}
    
    for pert_type in perturbation_types:
        if pert_type in transform_1_dict and len(eigenvals_valid_1) > 0:
            coefficients_1_dict[pert_type] = compute_coefficients(
                eigenvals_valid_1, eigenvecs_valid_1, transform_1_dict[pert_type])
        if pert_type in transform_2_dict and len(eigenvals_valid_2) > 0:
            coefficients_2_dict[pert_type] = compute_coefficients(
                eigenvals_valid_2, eigenvecs_valid_2, transform_2_dict[pert_type])
    
    return {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,
        'basis_2': basis_2_dict,
        'M_combined': M_combined,
        'k_basis_1': k_basis_1,
        'k_basis_2': k_basis_2
    }

def plot_multi_perturbation_results(results, N_plot=3):
    """
    Plot the combined eigenfunctions from multi-perturbation analysis
    """
    eta_grid = results['eta_grid']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']
    coefficients_2 = results['coefficients_2']
    basis_1 = results['basis_1']
    basis_2 = results['basis_2']
    
    N_plot = min(N_plot, len(eigenvals_1))
    if N_plot == 0:
        print("No valid eigenfunctions to plot")
        return
    
    perturbation_types = ['phi', 'dr', 'dm', 'vr', 'vm']
    n_pert = len(perturbation_types)
    
    fig, axes = plt.subplots(N_plot, n_pert, figsize=(16, 2*N_plot), 
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Multi-Perturbation Common Eigenfunctions", fontsize=16)
    
    for i in range(N_plot):
        for j, pert_type in enumerate(perturbation_types):
            ax = axes[i, j]
            
            if pert_type not in coefficients_1 or pert_type not in coefficients_2:
                ax.text(0.5, 0.5, f"No data\nfor {pert_type}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{pert_type} (eigenval {i+1})")
                continue
            
            # Reconstruct solutions from both bases
            solution_1 = np.zeros_like(eta_grid)
            solution_2 = np.zeros_like(eta_grid)
            
            N = len(coefficients_1[pert_type][i])
            for k in range(N):
                solution_1 += coefficients_1[pert_type][i, k] * basis_1[pert_type][:, k]
                solution_2 += coefficients_2[pert_type][i, k] * basis_2[pert_type][:, k]
            
            # Sign alignment
            if np.dot(solution_1, solution_2) < 0:
                solution_2 *= -1
            
            # Plot
            ax.plot(eta_grid, solution_1, 'r-', linewidth=2.5, 
                   label='Basis 1 (Closed)', alpha=0.8)
            ax.plot(eta_grid, solution_2, 'g--', linewidth=2.0, 
                   label='Basis 2 (Palindromic)', alpha=0.8)
            
            ax.set_title(f"{pert_type} (λ={eigenvals_1[i]:.3f})")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
            if i == N_plot - 1:
                ax.set_xlabel("Conformal Time η")
    
    plt.savefig("multi_perturbation_eigenfunctions_k2.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_basis1_diagnostic(N_k=5, N_t=300):
    """
    Plot individual solutions from basis 1 to identify instabilities
    """
    print(f"Diagnostic analysis for first {N_k} k-modes in basis 1")
    
    # Create time grid - start from fcb_time/100 to avoid Big Bang instabilities
    eta_start = fcb_time / 100.0
    eta_grid = np.linspace(eta_start, fcb_time, N_t)
    print(f"Time range: eta ∈ [{eta_start:.6e}, {fcb_time:.6e}]")
    
    # Generate first few k values from basis 1 (Closed Universe)
    k_values = np.array([4 * i * np.sqrt(np.abs(K)) for i in range(1, N_k + 1)])
    print(f"K = {K:.6e}")
    print(f"k values: {k_values}")
    
    perturbation_types = ['phi', 'dr', 'dm', 'vr', 'vm']
    
    fig, axes = plt.subplots(N_k, len(perturbation_types), figsize=(20, 4*N_k))
    if N_k == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Basis 1 Individual Solutions - Diagnostic Plot", fontsize=16)
    
    for i in range(N_k):
        k_val = k_values[i]
        print(f"Processing k[{i}] = {k_val:.6f}")
        
        try:
            # Generate solution for this k
            t_sol, y_sol = generate_solution_to_fcb(k_val)
            
            for j, pert_type in enumerate(perturbation_types):
                ax = axes[i, j] if N_k > 1 else axes[j]
                
                if pert_type in y_sol:
                    # Plot full solution
                    ax.semilogy(t_sol, np.abs(y_sol[pert_type]), 'b-', alpha=0.7, label='|solution|')
                    
                    # Interpolate onto truncated eta_grid
                    interpolator = interp1d(t_sol, y_sol[pert_type], bounds_error=False, fill_value=0.0)
                    solution = interpolator(eta_grid)
                    
                    # Mark extreme values
                    abs_solution = np.abs(solution)
                    median_abs = np.median(abs_solution)
                    threshold = 10.0 * median_abs
                    extreme_mask = abs_solution > threshold
                    
                    if np.any(extreme_mask):
                        ax.scatter(eta_grid[extreme_mask], abs_solution[extreme_mask], 
                                 c='red', s=20, alpha=0.8, label=f'Extreme (>{threshold:.1e})')
                    
                    # Mark important time points
                    ax.axvline(t0_integration, color='red', linestyle='-', alpha=0.5, label='t0_integration')
                    ax.axvline(eta_grid[0], color='green', linestyle='--', alpha=0.7, label='Truncation start')
                    ax.axvline(fcb_time/100.0, color='orange', linestyle='--', alpha=0.7, label='fcb_time/100')
                    
                    # Show where solution might blow up
                    max_val = np.max(abs_solution)
                    min_val = np.min(abs_solution[abs_solution > 0]) if np.any(abs_solution > 0) else 1e-20
                    print(f"  {pert_type}: range [{min_val:.2e}, {max_val:.2e}], extreme points: {np.sum(extreme_mask)}")
                    
                else:
                    ax.text(0.5, 0.5, f"No {pert_type}\ndata", ha='center', va='center')
                
                ax.set_title(f"k[{i}]={k_val:.4f}, {pert_type}")
                ax.set_ylabel("Amplitude")
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend()
                if i == N_k - 1:
                    ax.set_xlabel("Conformal Time η")
                    
        except Exception as e:
            print(f"Error processing k[{i}] = {k_val}: {e}")
            for j, pert_type in enumerate(perturbation_types):
                ax = axes[i, j] if N_k > 1 else axes[j]
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
    
    plt.savefig("basis1_diagnostic_plot.pdf", dpi=300, bbox_inches='tight')
    print("Diagnostic plot saved as basis1_diagnostic_plot.pdf")
    plt.show()


if __name__ == "__main__":
    # Run the analysis
    results = multi_perturbation_analysis(N=30, N_t=300)
    
    # # Plot diagnostic for basis 1 first
    # print("\nGenerating diagnostic plots for first few k values in basis 1...")
    # plot_basis1_diagnostic(results['k_basis_1'], results['eta_grid'], N_plot=5)


    # Plot results
    if len(results['eigenvals_1']) > 0:
        plot_multi_perturbation_results(results, N_plot=5)
    else:
        print("No eigenvalues found for plotting")
    
    print("\nAnalysis complete!")
    print(f"Combined M matrix shape: {results['M_combined'].shape}")
    print(f"Number of common eigenfunctions found: {len(results['eigenvals_1'])}")