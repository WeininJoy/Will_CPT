# -*- coding: utf-8 -*-
"""
This script implements Symmetric Löwdin Orthogonalization (Method 2) to generate
sparse and orthogonal perturbation solutions.

Method 2: Symmetric Löwdin Orthogonalization
--------------------------------------------
This method preserves the identity of modes while ensuring strict orthogonality.
It finds orthogonal solutions that are closest to pure Fourier modes.

Algorithm:
1. Project pure Fourier modes onto the valid subspace
2. Compute the overlap matrix S
3. Apply symmetric orthogonalization: Φ = Ψ S^{-1/2}

This gives us the "Löwdin-orthogonalized Quasi-Fourier basis" - strictly orthogonal
valid solutions that minimize distance to the standard Fourier basis.
"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.linalg import sqrtm, fractional_matrix_power

# =============================================================================
# 1. SETUP: Parameters and Constants (Same as original)
# =============================================================================
nu_spacing = 4

print("--- Setting up parameters and functions ---")

## --- Best-fit parameters ---
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density
Neff = 3.046

def cosmological_parameters(mt, kt, h):

    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

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
z_rec = 1063.4075 # calculated based on the calculate_z_rec() output
###############################################################################

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#set tolerances
atol = 1e-13;
rtol = 1e-13;
stol = 1e-10;
num_variables = 75; # number of pert variables
swaptime = 2; #set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda);
a0=1; K=-OmegaK * a0**2 * H0**2

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5;

a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda));
a2 = OmegaM/(12*OmegaLambda);
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2));
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2);
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2));
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3);
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6;

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol_a = solve_ivp(da_dt, [t0,swaptime], [a_Bang], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
sol = solve_ivp(ds_dt, [swaptime, 12], [1./sol_a.y[0][-1]], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Check if t_events[0] is not empty before trying to access its elements
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

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
a_rec = 1./(1+z_rec)  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol_a.y[0] - a_rec) #take difference between s values and s_rec to find where s=s_rec
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

# Perfect fluid ODE for early times (unchanged)
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

def generate_multi_perturbation_bases(discreteK_type, eta_grid):
    """
    Generate perturbation solutions for a given basis type
    (Same as original code)
    """
    # =============================================================================
    # 2. DATA LOADING
    # =============================================================================
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

        print(f"Loaded solution histories for {len(allowedK)} allowed K values.")
        print(f"Time grid has {len(t_grid)} points from eta={t_grid[0]:.2f} to eta={t_grid[-1]:.2f}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have run the data generation script first.")
        exit()

    # Extract all matrix components
    Amatrices = ABCmatrices[:, 0:6, :]
    Bmatrices = ABCmatrices[:, 6:8, :]
    Cmatrices = ABCmatrices[:, 8:num_variables, :]
    Dmatrices = DEFmatrices[:, 0:6, :]
    Ematrices = DEFmatrices[:, 6:8, :]
    Fmatrices = DEFmatrices[:, 8:num_variables, :]

    print(f"Matrix shapes: A={Amatrices.shape}, B={Bmatrices.shape}, C={Cmatrices.shape}")

    # =============================================================================
    # 3. CALCULATE AND RECONSTRUCT SOLUTIONS
    # =============================================================================

    print(f"\n--- Reconstructing {len(allowedK)} modes for {discreteK_type} ---")

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    basis_dict = {pert_type: [] for pert_type in perturbation_types}

    for i in range(len(allowedK)):
        k_index = i
        k = allowedK[k_index]

        if i < 3 or i % 5 == 0:  # Print progress every 5 modes
            print(f"Processing mode n={i+1}/{len(allowedK)} with k={k:.6f}")

        # Get matrices for this k
        A = Amatrices[k_index]
        B = Bmatrices[k_index]
        C = Cmatrices[k_index]
        D = Dmatrices[k_index]
        E = Ematrices[k_index]
        F = Fmatrices[k_index]
        X1 = X1matrices[k_index]
        X2 = X2matrices[k_index]
        recs_vec = recValues[k_index]

        # Calculate x^∞
        GX3 = np.zeros((6,4))
        GX3[:,2] = GHIvectors[k_index][0:6]

        M_matrix = (A @ X1 + D @ X2)[2:6, :]
        x_rec = recs_vec[2:6]
        x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]

        # Calculate x' and y' at endtime using X1 and X2
        x_prime_coeffs = X1 @ x_inf
        y_prime_coeffs = X2 @ x_inf

        # Reconstruct solution from pre-computed basis
        ABC_sols_k = all_ABC_solutions[k_index]
        DEF_sols_k = all_DEF_solutions[k_index]
        GHI_sols_k = all_GHI_solutions[k_index]

        Y_reconstructed = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                        np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs)

        t_backward = t_grid
        s_background = np.interp(t_grid, sol.t, sol.y[0])
        Y_backward = np.vstack([s_background, Y_reconstructed])

        # Prepare solutions
        t_full_unsorted = np.concatenate((t_backward[::-1], [fcb_time]))

        solutions_unsorted = {
            'dr': np.concatenate((Y_backward[3, ::-1], [x_inf[0]])),
            'dm': np.concatenate((Y_backward[4, ::-1], [x_inf[1]])),
            'vr': np.concatenate((Y_backward[5, ::-1], [x_inf[2]])),
            'vm': np.concatenate((Y_backward[6, ::-1], [(X1 @ x_inf)[3]]))
        }

        # Sort by time
        sort_indices = np.argsort(t_full_unsorted)
        t_sol = t_full_unsorted[sort_indices]
        y_sol = {key: value[sort_indices] for key, value in solutions_unsorted.items()}

        # Interpolate onto common eta grid
        for pert_type in perturbation_types:
            if pert_type in y_sol:
                interpolator = interp1d(t_sol, y_sol[pert_type],
                                        bounds_error=False, fill_value=0.0)
                solution = interpolator(eta_grid)
                basis_dict[pert_type].append(solution)
            else:
                basis_dict[pert_type].append(np.zeros_like(eta_grid))

    # Convert lists to numpy arrays
    for pert_type in perturbation_types:
        basis_dict[pert_type] = np.array(basis_dict[pert_type]).T  # Transpose: (N_t, N_modes)

    return basis_dict, allowedK

# =============================================================================
# METHOD 2: SYMMETRIC LÖWDIN ORTHOGONALIZATION
# =============================================================================

def lowdin_orthogonalization(basis_dict, M_valid, eta_grid):
    """
    Perform Symmetric Löwdin Orthogonalization to generate sparse and orthogonal solutions.

    This method:
    1. Takes the M_valid valid eigenfunctions from the subspace
    2. Computes the overlap matrix S
    3. Applies symmetric orthogonalization: Φ = Ψ S^{-1/2}

    Parameters:
    -----------
    basis_dict : dict
        Dictionary containing basis functions for each perturbation type
        Each entry is shape (N_t, N_modes)
    M_valid : int
        Number of valid modes to orthogonalize
    eta_grid : array
        Time grid

    Returns:
    --------
    ortho_basis_dict : dict
        Orthogonalized basis functions
    overlap_matrix : ndarray
        The overlap matrix S
    """
    print("\n" + "="*80)
    print("SYMMETRIC LÖWDIN ORTHOGONALIZATION (Method 2)")
    print("="*80)

    perturbation_types = ['dr', 'dm', 'vr', 'vm']

    # Step 1: Extract the first M_valid modes from the basis
    # These are our "projected Fourier modes" ψ_k
    print(f"\nStep 1: Selecting {M_valid} modes from the valid subspace...")

    psi_dict = {}
    for pert_type in perturbation_types:
        if pert_type in basis_dict:
            # Select first M_valid columns
            psi_dict[pert_type] = basis_dict[pert_type][:, :M_valid]
            print(f"  {pert_type}: shape {psi_dict[pert_type].shape}")

    # Step 2: Compute the overlap matrix S
    # S_ij = ⟨ψ_i|ψ_j⟩ = sum over all perturbation types
    print("\nStep 2: Computing overlap matrix S...")
    print("  S_ij = ⟨ψ_i|ψ_j⟩ (summed over all perturbation types)")

    S = np.zeros((M_valid, M_valid))

    for pert_type in perturbation_types:
        if pert_type in psi_dict:
            # Compute inner product: ψ^T · ψ (integrated over time via dot product)
            # Note: We need to normalize by the time grid spacing for proper integration
            dt = eta_grid[1] - eta_grid[0]  # assuming uniform grid
            S_pert = psi_dict[pert_type].T @ psi_dict[pert_type] * dt
            S += S_pert
            print(f"  Added contribution from {pert_type}")

    # Normalize by number of perturbation types to keep scale reasonable
    S = S / len(perturbation_types)

    print(f"\n  Overlap matrix S shape: {S.shape}")
    print(f"  Diagonal elements (should be ~1 for normalized modes):")
    print(f"    S_ii = {np.diag(S)[:5]}")
    print(f"  Condition number of S: {np.linalg.cond(S):.2e}")

    # Step 3: Compute S^{-1/2} (inverse square root)
    print("\nStep 3: Computing S^{-1/2} (inverse square root of overlap matrix)...")

    # Method: Use eigenvalue decomposition S = V Λ V^T, then S^{-1/2} = V Λ^{-1/2} V^T
    eigenvals, eigenvecs = np.linalg.eigh(S)  # eigh for symmetric matrix

    print(f"  Eigenvalues of S (smallest 5): {eigenvals[:5]}")
    print(f"  Eigenvalues of S (largest 5): {eigenvals[-5:]}")

    # Check for small eigenvalues (numerical instability)
    min_eigenval = np.min(eigenvals)
    if min_eigenval < 1e-10:
        print(f"  WARNING: Small eigenvalue detected ({min_eigenval:.2e})")
        print(f"  Regularizing with threshold...")
        eigenvals = np.maximum(eigenvals, 1e-10)

    # Compute S^{-1/2}
    S_inv_sqrt = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T

    print(f"  S^{{-1/2}} computed successfully")

    # Step 4: Apply symmetric orthogonalization: Φ = Ψ S^{-1/2}
    print("\nStep 4: Applying Löwdin orthogonalization Φ = Ψ S^{-1/2}...")

    ortho_basis_dict = {}

    for pert_type in perturbation_types:
        if pert_type in psi_dict:
            # Matrix multiplication: (N_t × M) @ (M × M) = (N_t × M)
            ortho_basis_dict[pert_type] = psi_dict[pert_type] @ S_inv_sqrt
            print(f"  {pert_type}: orthogonalized, shape {ortho_basis_dict[pert_type].shape}")

    # Verification: Check orthogonality
    print("\nStep 5: Verifying orthogonality...")
    S_ortho = np.zeros((M_valid, M_valid))
    for pert_type in perturbation_types:
        if pert_type in ortho_basis_dict:
            dt = eta_grid[1] - eta_grid[0]
            S_ortho += ortho_basis_dict[pert_type].T @ ortho_basis_dict[pert_type] * dt

    S_ortho = S_ortho / len(perturbation_types)

    print(f"  Overlap matrix after orthogonalization:")
    print(f"    Diagonal (should be ~1): {np.diag(S_ortho)[:5]}")
    print(f"    Off-diagonal max (should be ~0): {np.max(np.abs(S_ortho - np.diag(np.diag(S_ortho)))):.2e}")

    print("\n" + "="*80)
    print("LÖWDIN ORTHOGONALIZATION COMPLETE")
    print("="*80)

    return ortho_basis_dict, S, S_inv_sqrt

# =============================================================================
# ANALYSIS AND PLOTTING FUNCTIONS
# =============================================================================

def compute_combined_coefficients(ortho_basis_dict, original_basis_dict, eta_grid):
    """
    Compute combined coefficients across all perturbation types.
    This gives a general measure of how each orthogonalized mode decomposes
    into the original k modes, averaged over all perturbation types.

    Returns:
    --------
    coefficients : ndarray (M_valid, N_modes)
        Coefficient matrix where coefficients[i, j] indicates the contribution
        of original mode j to orthogonalized mode i
    """
    perturbation_types = ['dr', 'dm', 'vr', 'vm']

    M_valid = ortho_basis_dict['vr'].shape[1]
    N_modes = original_basis_dict['vr'].shape[1]

    # Initialize combined coefficient matrix
    coefficients_combined = np.zeros((M_valid, N_modes))

    # Compute time integration factor
    dt = eta_grid[1] - eta_grid[0]

    print("\nComputing combined coefficients across all perturbation types...")

    for pert_type in perturbation_types:
        if pert_type in ortho_basis_dict and pert_type in original_basis_dict:
            # Compute inner products: <ortho_i | original_j>
            # Shape: (M_valid, N_modes)
            coeffs_pert = ortho_basis_dict[pert_type].T @ original_basis_dict[pert_type] * dt
            coefficients_combined += coeffs_pert
            print(f"  Added coefficients from {pert_type}")

    # Normalize by number of perturbation types
    coefficients_combined = coefficients_combined / len(perturbation_types)

    print(f"  Combined coefficient matrix shape: {coefficients_combined.shape}")

    return coefficients_combined

def plot_lowdin_coefficients_by_dominant_k(ortho_basis_dict, original_basis_dict,
                                           k_values, eta_grid, N_plot=10):
    """
    Plot sparse coefficients sorted by dominant k mode.
    Similar to plot_coefficients_by_dominant_k() in the original code,
    but for Löwdin-orthogonalized modes.

    This shows the general decomposition (combined over all perturbation types)
    of each orthogonalized mode in terms of the original k modes.
    """
    print("\n" + "="*80)
    print("COMPUTING SPARSE COEFFICIENTS (COMBINED OVER ALL PERTURBATIONS)")
    print("="*80)

    # Compute combined coefficients
    coefficients = compute_combined_coefficients(ortho_basis_dict, original_basis_dict, eta_grid)

    M_valid = coefficients.shape[0]
    N_modes = coefficients.shape[1]

    # Find dominant k for each orthogonalized mode
    dominant_k_idx = np.argmax(np.abs(coefficients), axis=1)

    # Sort by dominant k (from small to large)
    sorted_order = np.argsort(dominant_k_idx)

    print(f"\nDominant k indices (sorted): {dominant_k_idx[sorted_order]}")
    print(f"Corresponding k values: {k_values[dominant_k_idx[sorted_order]]}")

    # Compute sparsity measure (inverse participation ratio)
    sparsity = 1.0 / np.sum(coefficients**4, axis=1)
    print(f"\nSparsity measure (IPR):")
    print(f"  Mean: {np.mean(sparsity):.2f}")
    print(f"  Min: {np.min(sparsity):.2f}")
    print(f"  Max: {np.max(sparsity):.2f}")
    print(f"  (Lower = more sparse, 1.0 = perfectly sparse)")

    # Create plot
    N_plot = min(N_plot, M_valid)
    fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
    if N_plot == 1:
        axs = [axs]

    fig.suptitle(f"Löwdin-Orthogonalized Modes: Sparse Coefficients (sorted by dominant k)",
                 fontsize=10)

    # Plot each mode's coefficients
    for i in range(N_plot):
        plot_idx = sorted_order[i]

        coeffs = coefficients[plot_idx, :].real
        dominant_k = dominant_k_idx[plot_idx]
        sparse_measure = sparsity[plot_idx]

        # Create bar plot
        k_indices = np.arange(1, N_modes + 1)
        axs[i].bar(k_indices, coeffs, width=0.4)
        # axs[i].set_xlim(0, min(20, N_modes + 1))
        # axs[i].set_ylim(-1, 1.)

        # Add title with dominant k and sparsity
        axs[i].set_title(f"Mode {plot_idx+1}: dominant k={k_values[dominant_k]:.3f} (idx={dominant_k+1}), sparse={sparse_measure:.2f}",
                        fontsize=8, loc='right')

        # Highlight the dominant k
        axs[i].axvline(dominant_k + 1, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

        # Format
        axs[i].xaxis.set_tick_params(labelsize=8)
        axs[i].yaxis.set_tick_params(labelsize=8)
        axs[i].label_outer()
        axs[i].grid(True, alpha=0.3, axis='y')

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1].set_xlabel(r"$k$ mode index", fontsize=8)
    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

    fig.tight_layout()

    # Save figure
    output_filename = "lowdin_sparse_coefficients_sorted.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved coefficient plot to {output_filename}")
    plt.show()

    return {
        'coefficients': coefficients,
        'dominant_k_idx': dominant_k_idx,
        'sorted_order': sorted_order,
        'sparsity': sparsity
    }

def plot_lowdin_eigenfunctions(ortho_basis_dict, eta_grid, k_values, N_plot=6):
    """
    Plot the orthogonalized eigenfunctions in time domain.
    """
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    n_pert = len(perturbation_types)

    # Determine which modes to plot (evenly spaced)
    M_valid = ortho_basis_dict['vr'].shape[1]
    N_plot = min(N_plot, M_valid)
    plot_indices = np.linspace(0, M_valid-1, N_plot, dtype=int)

    fig, axes = plt.subplots(N_plot, n_pert, figsize=(16, 2*N_plot),
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Löwdin-Orthogonalized Quasi-Fourier Modes", fontsize=14)

    for i, mode_idx in enumerate(plot_indices):
        for j, pert_type in enumerate(perturbation_types):
            ax = axes[i, j]

            if pert_type not in ortho_basis_dict:
                ax.text(0.5, 0.5, f"No data\nfor {pert_type}",
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Get the mode
            mode = ortho_basis_dict[pert_type][:, mode_idx]

            # Plot
            ax.plot(eta_grid, mode, 'b-', linewidth=1.5)

            # Add title
            if mode_idx < len(k_values):
                ax.set_title(f"{pert_type} (mode {mode_idx+1}, k≈{k_values[mode_idx]:.3f})",
                           fontsize=9)
            else:
                ax.set_title(f"{pert_type} (mode {mode_idx+1})", fontsize=9)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if i == N_plot - 1:
                ax.set_xlabel("Conformal Time η", fontsize=8)

    plt.savefig("lowdin_orthogonal_eigenfunctions.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved plot to lowdin_orthogonal_eigenfunctions.pdf")
    plt.show()

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def lowdin_sparse_orthogonal_analysis(N_valid=23, N_t=1000):
    """
    Complete Löwdin orthogonalization analysis for sparse and orthogonal solutions.

    Parameters:
    -----------
    N_valid : int
        Number of valid modes to use from the subspace
    N_t : int
        Number of time grid points
    """
    print("\n" + "="*80)
    print("LÖWDIN SPARSE ORTHOGONAL ANALYSIS")
    print("="*80)
    print(f"Parameters: N_valid={N_valid}, N_t={N_t}")

    # Define eta grid
    eta_grid = np.linspace(0, fcb_time, N_t)
    print(f"\nUsing eta_grid: η ∈ [0, {fcb_time:.4e}] with {N_t} points")

    # Generate basis from the valid subspace (use allowedK as the valid subspace)
    print("\n" + "="*80)
    print("Generating valid basis (allowedK)...")
    print("="*80)
    basis_dict, k_values = generate_multi_perturbation_bases("allowedK", eta_grid)

    # Determine number of valid modes
    actual_N_valid = min(N_valid, basis_dict['vr'].shape[1])
    print(f"\nActual number of modes to orthogonalize: {actual_N_valid}")

    # Apply Löwdin orthogonalization
    ortho_basis_dict, S, S_inv_sqrt = lowdin_orthogonalization(
        basis_dict, actual_N_valid, eta_grid)

    # Save results
    print("\nSaving results...")
    results = {
        'ortho_basis': ortho_basis_dict,
        'original_basis': basis_dict,
        'k_values': k_values,
        'eta_grid': eta_grid,
        'overlap_matrix': S,
        'S_inv_sqrt': S_inv_sqrt,
        'N_valid': actual_N_valid
    }

    with open("lowdin_sparse_ortho_results.pickle", 'wb') as f:
        pickle.dump(results, f)
    print("Saved results to lowdin_sparse_ortho_results.pickle")

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set folder path
    folder_path = f'./data/'

    # Load to determine size
    allowedK = np.load(folder_path + 'data_allowedK/L70_kvalues.npy')
    print(f"Found {len(allowedK)} allowed K values")

    # Run Löwdin orthogonalization
    results = lowdin_sparse_orthogonal_analysis(N_valid=len(allowedK), N_t=1000)

    # Plot results
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # Plot the orthogonalized eigenfunctions
    plot_lowdin_eigenfunctions(
        results['ortho_basis'],
        results['eta_grid'],
        results['k_values'],
        N_plot=6
    )

    # Plot sparse coefficients (combined over all perturbation types)
    coeff_results = plot_lowdin_coefficients_by_dominant_k(
        results['ortho_basis'],
        results['original_basis'],
        results['k_values'],
        results['eta_grid'],
        N_plot=10
    )

    # Save coefficient results
    results['coefficient_analysis'] = coeff_results
    with open("lowdin_sparse_ortho_results.pickle", 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - lowdin_sparse_ortho_results.pickle (saved results)")
    print("  - lowdin_orthogonal_eigenfunctions.pdf (orthogonal modes)")
    print("  - lowdin_sparse_coefficients_sorted.pdf (sparse coefficients)")
    print("\nThe Löwdin-orthogonalized basis provides:")
    print("  1. Strict orthogonality (required for quantum operators)")
    print("  2. Maximum sparsity (closest to pure Fourier modes)")
    print("  3. Physical meaning (each mode labeled by dominant k)")
