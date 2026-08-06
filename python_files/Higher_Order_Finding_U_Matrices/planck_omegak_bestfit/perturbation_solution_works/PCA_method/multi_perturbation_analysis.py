# -*- coding: utf-8 -*-
"""
PCA-based Multi-Perturbation Eigenvalue Analysis

This script uses PCA (Principal Component Analysis) instead of QR decomposition
to find common eigenfunctions between two bases without orthonormalization.

Key differences from QR approach:
1. Normalize perturbation solutions (preserve relative amplitudes)
2. Use PCA on the mixing matrix M
3. Eigenvalues represent explained variance, not overlap (values won't be ~1.0)
4. Avoid degeneracy issues from QR decomposition

Based on PCA_solution.md instructions and PCA_original_basis.py reference.
"""
import json
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from sklearn.decomposition import PCA

# =============================================================================
# 1. SETUP: Parameters and Constants
# =============================================================================
nu_spacing = 4

print("--- Setting up parameters and functions ---")

## --- Best-fit parameters ---
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5  # photon density
Neff = 3.046

def cosmological_parameters(mt, kt, h):
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3)) * Omega_gamma_h2/h**2

    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(Omega_r, rt, mt, kt)
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return Omega_lambda, Omega_m, Omega_K

###############################################################################
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 409.969398,1.459351,0.163514,0.547313,2.095762,0.972835,0.053017

OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1063.4075  # calculated based on the calculate_z_rec() output
###############################################################################

# Background equations
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 75
swaptime = 2
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda)
Hinf = H0*np.sqrt(OmegaLambda)
a0 = 1
K = -OmegaK * a0**2 * H0**2

def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5

a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2))
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol_a = solve_ivp(da_dt, [t0,swaptime], [a_Bang], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
sol = solve_ivp(ds_dt, [swaptime, 12], [1./sol_a.y[0][-1]], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    fcb_time = None

if fcb_time is not None:
    print(f"Further processing with fcb_time = {fcb_time}")
else:
    print(f"No fcb_time available for further processing.")

endtime = fcb_time - deltaeta

# Recombination conformal time
a_rec = 1./(1+z_rec)
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    sigmadot = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)+OmegaR*np.exp(2*sigma))
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4)
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 2. BASIS GENERATION (reused from QR version)
# =============================================================================
def generate_multi_perturbation_bases(discreteK_type, eta_grid, folder_path='../data/'):
    """
    Generate perturbation bases for different k-modes.
    This is identical to the QR version - we just won't orthonormalize them.
    """
    folder_path_matrices = folder_path + f'data_{discreteK_type}/'
    folder_path_timeseries = folder_path + f'data_{discreteK_type}_timeseries/'

    print(f"\n--- Loading pre-computed time-dependent transfer matrices for {discreteK_type} ---")
    try:
        t_grid = np.load(folder_path_timeseries + 't_grid.npy')
        allowedK = np.load(folder_path_timeseries + 'L70_kvalues.npy')
        all_ABC_solutions = np.load(folder_path_timeseries + 'L70_ABC_solutions.npy')
        all_DEF_solutions = np.load(folder_path_timeseries + 'L70_DEF_solutions.npy')
        all_GHI_solutions = np.load(folder_path_timeseries + 'L70_GHI_solutions.npy')

        ABCmatrices = np.load(folder_path_matrices+'L70_ABCmatrices.npy')
        DEFmatrices = np.load(folder_path_matrices+'L70_DEFmatrices.npy')
        GHIvectors = np.load(folder_path_matrices+'L70_GHIvectors.npy')
        X1matrices = np.load(folder_path_matrices + 'L70_X1matrices.npy')
        X2matrices = np.load(folder_path_matrices + 'L70_X2matrices.npy')
        recValues = np.load(folder_path_matrices + 'L70_recValues.npy')

        print(f"Loaded solution histories for {len(allowedK)} K values.")
        print(f"Time grid has {len(t_grid)} points from eta={t_grid[0]:.2f} to eta={t_grid[-1]:.2f}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit()

    Amatrices = ABCmatrices[:, 0:6, :]
    Bmatrices = ABCmatrices[:, 6:8, :]
    Cmatrices = ABCmatrices[:, 8:num_variables, :]
    Dmatrices = DEFmatrices[:, 0:6, :]
    Ematrices = DEFmatrices[:, 6:8, :]
    Fmatrices = DEFmatrices[:, 8:num_variables, :]

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    basis_dict = {pert_type: [] for pert_type in perturbation_types}

    for i in range(len(allowedK)):
        k_index = i
        k = allowedK[k_index]

        A = Amatrices[k_index]
        B = Bmatrices[k_index]
        C = Cmatrices[k_index]
        D = Dmatrices[k_index]
        E = Ematrices[k_index]
        F = Fmatrices[k_index]
        X1 = X1matrices[k_index]
        X2 = X2matrices[k_index]
        recs_vec = recValues[k_index]

        GX3 = np.zeros((6,4))
        GX3[:,2] = GHIvectors[k_index][0:6]

        M_matrix = (A @ X1 + D @ X2)[2:6, :]
        x_rec = recs_vec[2:6]
        x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]

        x_prime_coeffs = X1 @ x_inf
        y_prime_coeffs = X2 @ x_inf

        ABC_sols_k = all_ABC_solutions[k_index]
        DEF_sols_k = all_DEF_solutions[k_index]
        GHI_sols_k = all_GHI_solutions[k_index]

        Y_reconstructed = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                        np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs)

        s_background = np.interp(t_grid, sol.t, sol.y[0])
        Y_backward = np.vstack([s_background, Y_reconstructed])

        t_full_unsorted = np.concatenate((t_grid[::-1], [fcb_time]))

        solutions_unsorted = {
            'dr': np.concatenate((Y_backward[3, ::-1], [x_inf[0]])),
            'dm': np.concatenate((Y_backward[4, ::-1], [x_inf[1]])),
            'vr': np.concatenate((Y_backward[5, ::-1], [x_inf[2]])),
            'vm': np.concatenate((Y_backward[6, ::-1], [(X1 @ x_inf)[3]]))
        }

        sort_indices = np.argsort(t_full_unsorted)
        t_sol = t_full_unsorted[sort_indices]
        y_sol = {key: value[sort_indices] for key, value in solutions_unsorted.items()}

        for pert_type in perturbation_types:
            if pert_type in y_sol:
                interpolator = interp1d(t_sol, y_sol[pert_type],
                                      bounds_error=False, fill_value=0.0)
                solution = interpolator(eta_grid)
                basis_dict[pert_type].append(solution)
            else:
                basis_dict[pert_type].append(np.zeros_like(eta_grid))

    # Convert lists to numpy arrays (each column is a basis function)
    for pert_type in perturbation_types:
        basis_dict[pert_type] = np.array(basis_dict[pert_type]).T

    return basis_dict

# =============================================================================
# 3. PCA-BASED ANALYSIS
# =============================================================================
def normalize_basis(basis_dict, save_path=None):
    """
    Normalize each perturbation solution separately.
    Save normalization constants for reconstruction.

    Parameters:
    -----------
    basis_dict : dict
        Dictionary of basis functions for each perturbation type
        Each is shape (N_time, N_modes)
    save_path : str, optional
        Path to save normalization constants

    Returns:
    --------
    normalized_basis_dict : dict
        Dictionary of normalized basis functions
    norm_constants_dict : dict
        Dictionary of normalization constants (L2 norms)
    """
    normalized_basis_dict = {}
    norm_constants_dict = {}

    perturbation_types = ['dr', 'dm', 'vr', 'vm']

    print("\n--- Normalizing perturbation solutions ---")

    for pert_type in perturbation_types:
        if pert_type not in basis_dict:
            continue

        basis = basis_dict[pert_type]  # Shape: (N_time, N_modes)
        N_time, N_modes = basis.shape

        # Compute L2 norm for each mode (column)
        norms = np.linalg.norm(basis, axis=0)  # Shape: (N_modes,)

        # Avoid division by zero
        norms[norms < 1e-15] = 1.0

        # Normalize each column
        normalized_basis = basis / norms[np.newaxis, :]

        normalized_basis_dict[pert_type] = normalized_basis
        norm_constants_dict[pert_type] = norms

        print(f"  {pert_type}: normalized {N_modes} modes")
        print(f"    Norm range: [{norms.min():.2e}, {norms.max():.2e}]")

    # Save normalization constants
    if save_path is not None:
        np.savez(save_path,
                 dr_norms=norm_constants_dict.get('dr', np.array([])),
                 dm_norms=norm_constants_dict.get('dm', np.array([])),
                 vr_norms=norm_constants_dict.get('vr', np.array([])),
                 vm_norms=norm_constants_dict.get('vm', np.array([])))
        print(f"\nSaved normalization constants to {save_path}")

    return normalized_basis_dict, norm_constants_dict

def compute_multi_perturbation_M_matrix(basis_1_dict, basis_2_dict):
    """
    Compute the mixing matrix M by summing contributions from all perturbation types.

    Similar to QR version, but NO orthonormalization.
    M = sum_pert_types (basis_1[pert].T @ basis_2[pert])

    Parameters:
    -----------
    basis_1_dict : dict
        Normalized basis 1 for each perturbation type
    basis_2_dict : dict
        Normalized basis 2 for each perturbation type

    Returns:
    --------
    M_total : ndarray
        Combined mixing matrix summed over perturbation types
    """
    perturbation_types = ['dr', 'dm', 'vr', 'vm']

    M_total = None
    n_perturbations = 0

    print("\n--- Computing combined M matrix from all perturbations ---")

    for pert_type in perturbation_types:
        if pert_type in basis_1_dict and pert_type in basis_2_dict:
            # Compute M for this perturbation type
            # M_pert = basis_1.T @ basis_2  (N_modes x N_modes)
            M_pert = np.dot(basis_1_dict[pert_type].T, basis_2_dict[pert_type])

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

    print(f"  M_total shape: {M_total.shape}")

    return M_total

def pca_eigenvalue_analysis(M_matrix, n_components=None):
    """
    Perform PCA on the mixing matrix M.

    PCA on M.T @ M gives us the principal components (eigenvectors)
    and explained variance (related to eigenvalues).

    Parameters:
    -----------
    M_matrix : ndarray
        Mixing matrix (N_modes x N_modes)
    n_components : int, optional
        Number of components to keep (default: all)

    Returns:
    --------
    eigenvalues_1 : ndarray
        Eigenvalues from PCA on M @ M.T
    eigenvectors_1 : ndarray
        Eigenvectors (columns) from PCA on M @ M.T
    eigenvalues_2 : ndarray
        Eigenvalues from PCA on M.T @ M
    eigenvectors_2 : ndarray
        Eigenvectors (columns) from PCA on M.T @ M
    """
    print("\n--- Performing PCA eigenvalue analysis ---")

    # PCA on M @ M.T (for basis 1)
    X_1 = M_matrix @ M_matrix.T
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(X_1)

    # PCA on M.T @ M (for basis 2)
    X_2 = M_matrix.T @ M_matrix
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(X_2)

    # Sort by eigenvalues (descending)
    sorted_idx_1 = np.argsort(np.real(eigenvalues_1))[::-1]
    sorted_idx_2 = np.argsort(np.real(eigenvalues_2))[::-1]

    eigenvalues_1 = eigenvalues_1[sorted_idx_1]
    eigenvectors_1 = eigenvectors_1[:, sorted_idx_1]

    eigenvalues_2 = eigenvalues_2[sorted_idx_2]
    eigenvectors_2 = eigenvectors_2[:, sorted_idx_2]

    print(f"  Computed {len(eigenvalues_1)} eigenvalues for basis 1")
    print(f"  Computed {len(eigenvalues_2)} eigenvalues for basis 2")
    print(f"  Top 5 eigenvalues (basis 1): {eigenvalues_1[:5].real}")
    print(f"  Top 5 eigenvalues (basis 2): {eigenvalues_2[:5].real}")

    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2

def choose_valid_eigenvalues(eigenvalues, eigenvectors, eigenvalue_ratio_threshold=0.5):
    """
    Select valid eigenvalues using a relative threshold.

    Since PCA eigenvalues represent explained variance (not overlap),
    we use a different criterion: keep eigenvalues within some ratio
    of the maximum eigenvalue.

    Parameters:
    -----------
    eigenvalues : ndarray
        Eigenvalues (sorted descending)
    eigenvectors : ndarray
        Eigenvectors (columns)
    eigenvalue_ratio_threshold : float
        Ratio relative to max eigenvalue (default: 0.5)

    Returns:
    --------
    eigenvalues_valid : list
        Valid eigenvalues
    eigenvectors_valid : list
        Valid eigenvectors
    """
    max_eigenval = np.real(eigenvalues[0])
    threshold_value = max_eigenval * eigenvalue_ratio_threshold

    eigenvalues_valid = []
    eigenvectors_valid = []

    for i in range(len(eigenvalues)):
        if np.real(eigenvalues[i]) < threshold_value:
            break
        eigenvalues_valid.append(eigenvalues[i])
        eigenvectors_valid.append(eigenvectors[:, i])

    print(f"\n  Kept {len(eigenvalues_valid)} modes with eigenvalue >= {threshold_value:.2e}")
    print(f"  (ratio >= {eigenvalue_ratio_threshold} of max)")

    return eigenvalues_valid, eigenvectors_valid

def compute_coefficients_pca(eigenvalues, eigenvectors):
    """
    Compute linear combination coefficients from PCA eigenvectors.

    For PCA, the eigenvectors ARE the coefficients (no transformation matrix needed).

    Parameters:
    -----------
    eigenvalues : list
        Valid eigenvalues
    eigenvectors : list
        Valid eigenvectors

    Returns:
    --------
    coefficients : ndarray
        Coefficient matrix (K_valid x N_modes)
    """
    K_valid = len(eigenvalues)
    N_modes = len(eigenvectors[0])

    coefficients = np.zeros((K_valid, N_modes), dtype=float)

    for i in range(K_valid):
        coefficients[i, :] = np.real(eigenvectors[i])

    return coefficients

# =============================================================================
# 4. MAIN ANALYSIS FUNCTION
# =============================================================================
def multi_perturbation_pca_analysis(N=23, N_t=500, folder_path='../data/',
                                     eigenvalue_ratio_threshold=0.5,
                                     save_norms_path='normalization_constants.npz'):
    """
    Complete multi-perturbation PCA analysis.

    Parameters:
    -----------
    N : int
        Number of modes (for compatibility)
    N_t : int
        Number of time points
    folder_path : str
        Path to data folders
    eigenvalue_ratio_threshold : float
        Threshold for selecting valid eigenvalues (ratio of max)
    save_norms_path : str
        Path to save normalization constants

    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    print(f"\n{'='*80}")
    print("STARTING PCA-BASED MULTI-PERTURBATION ANALYSIS")
    print(f"{'='*80}")
    print(f"N={N}, N_t={N_t}")

    # Define eta_grid
    eta_grid = np.linspace(0, fcb_time, N_t)
    print(f"Using eta_grid: eta ∈ [0, {fcb_time:.4e}]")

    # 1. Generate bases (same as QR version)
    print("\nGenerating basis 1 (Closed Universe - integerK)...")
    basis_1_dict = generate_multi_perturbation_bases("integerK", eta_grid, folder_path=folder_path)

    print("\nGenerating basis 2 (Palindromic Universe - allowedK)...")
    basis_2_dict = generate_multi_perturbation_bases("allowedK", eta_grid, folder_path=folder_path)

    # 2. Normalize bases
    print("\nNormalizing basis 1...")
    basis_1_normalized, norms_1 = normalize_basis(basis_1_dict,
                                                   save_path=save_norms_path.replace('.npz', '_basis1.npz'))

    print("\nNormalizing basis 2...")
    basis_2_normalized, norms_2 = normalize_basis(basis_2_dict,
                                                   save_path=save_norms_path.replace('.npz', '_basis2.npz'))

    # 3. Compute combined M matrix
    M_total = compute_multi_perturbation_M_matrix(basis_1_normalized, basis_2_normalized)

    # 4. PCA eigenvalue analysis
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2 = pca_eigenvalue_analysis(M_total)

    # 5. Select valid eigenvalues
    eigenvals_valid_1, eigenvecs_valid_1 = choose_valid_eigenvalues(
        eigenvals_1, eigenvecs_1, eigenvalue_ratio_threshold)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_valid_eigenvalues(
        eigenvals_2, eigenvecs_2, eigenvalue_ratio_threshold)

    print(f"\nPCA Analysis: Found {len(eigenvals_valid_1)} valid modes")

    # 6. Compute coefficients (for normalized bases)
    coefficients_1_norm = compute_coefficients_pca(eigenvals_valid_1, eigenvecs_valid_1)
    coefficients_2_norm = compute_coefficients_pca(eigenvals_valid_2, eigenvecs_valid_2)

    # 7. Create coefficient dictionaries (same coefficients for all pert types)
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    coefficients_1_dict = {pert: coefficients_1_norm for pert in perturbation_types}
    coefficients_2_dict = {pert: coefficients_2_norm for pert in perturbation_types}

    print(f"\nCreated coefficient dictionaries for {len(perturbation_types)} perturbation types")

    # Return results
    results = {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        'eigenvecs_1': eigenvecs_valid_1,
        'eigenvecs_2': eigenvecs_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,  # Original (non-normalized) for reconstruction
        'basis_2': basis_2_dict,
        'basis_1_normalized': basis_1_normalized,
        'basis_2_normalized': basis_2_normalized,
        'norms_1': norms_1,
        'norms_2': norms_2,
        'M_combined': M_total,
        'method': 'PCA'
    }

    print(f"\n{'='*80}")
    print("PCA ANALYSIS COMPLETE")
    print(f"{'='*80}")

    return results

# =============================================================================
# 5. PLOTTING (reuse from QR version with minor adaptations)
# =============================================================================
def plot_multi_perturbation_results(results, N_plot=3):
    """Plot the combined eigenfunctions from PCA analysis"""
    eta_grid = results['eta_grid']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']
    coefficients_2 = results['coefficients_2']
    basis_1 = results['basis_1_normalized']  # Use normalized basis
    basis_2 = results['basis_2_normalized']

    N_plot = min(N_plot, len(eigenvals_1))
    if N_plot == 0:
        print("No valid eigenfunctions to plot")
        return

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    n_pert = len(perturbation_types)

    fig, axes = plt.subplots(N_plot, n_pert, figsize=(16, 2*N_plot),
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("PCA Multi-Perturbation Common Eigenfunctions", fontsize=16)

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

            ax.set_title(f"{pert_type} (λ={np.real(eigenvals_1[i]):.2e})")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
            if i == N_plot - 1:
                ax.set_xlabel("Conformal Time η")

    plt.savefig("pca_multi_perturbation_eigenfunctions.pdf", dpi=300, bbox_inches='tight')
    print("\nSaved plot: pca_multi_perturbation_eigenfunctions.pdf")

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    folder_path = '../data/'

    # Check if allowedK data exists
    allowedK_path = folder_path + 'data_allowedK/L70_kvalues.npy'
    if os.path.exists(allowedK_path):
        allowedK = np.load(allowedK_path)
        N = len(allowedK)
        print(f"Found {N} allowedK values")
    else:
        N = 23  # Default
        print(f"Using default N={N}")

    # Run PCA analysis
    results = multi_perturbation_pca_analysis(
        N=N,
        N_t=1000,
        folder_path=folder_path,
        eigenvalue_ratio_threshold=1.e-3,
        save_norms_path='normalization_constants.npz'
    )

    # Save results
    results_file = "pca_multi_perturbation_results.pickle"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_file}")

    # Plot results
    if len(results['eigenvals_1']) > 0:
        plot_multi_perturbation_results(results, N_plot=5)
    else:
        print("No eigenvalues found for plotting")

    print("\nPCA Analysis complete!")
    print(f"Combined M matrix shape: {results['M_combined'].shape}")
    print(f"Number of common eigenfunctions found: {len(results['eigenvals_1'])}")
