# -*- coding: utf-8 -*-
"""
This script performs a Gram-Schmidt orthonormalization and eigenvalue analysis
on two different bases of the COMPLETE cosmological perturbation solutions for
the radiation velocity, v_r(eta).

MODIFIED VERSION: Includes weighting calculation for k-dependent CLASS modifications.

The solution generator is based on the user's verified script for plotting
the full palindromic evolution.

- basis_1 is constructed from wavenumbers 'k' in a closed universe model.
- basis_2 is constructed from the 'allowed' wavenumbers for a palindromic flat universe.

It identifies and compares the common eigenfunctions of v_r found in both bases.

NEW FEATURES:
- Calculates weighting(i) = |coefficient_1(i)|^2 * N / sum(|coefficient_1|^2)
- Normalized so that sum(weights) = N (number of modes)
- Outputs weighting as .dat file in format compatible with modified CLASS code
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# SECTION 1: FULL SOLUTION GENERATION ENGINE (MODIFIED FOR v_r)
# =============================================================================
start_time = time.time()
folder_path = './data/'
# --- Cosmological Parameters and Setup (No changes here) ---
h = 0.5409; Omega_gamma_h2 = 2.47e-5; Neff = 3.046
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
OmegaM, OmegaK = 0.483, -0.0438
OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
z_rec = 1089.411

# assume s0 = 1/a0 = 1
a0 = 1; s0 = 1/a0; H0 = np.sqrt(1 / (3 * OmegaLambda)); Hinf = H0 * np.sqrt(OmegaLambda)
K=-OmegaK * a0**2 * H0**2
atol, rtol = 1e-12, 1e-12
num_variables_boltzmann, l_max = 75, 69
t0_integration = 1e-8 * s0
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4)); szero = -OmegaM/s0**3/(4*OmegaR/s0**4)
s_bang_init = smin1/t0_integration + szero
swaptime = 2.0 * s0
def ds_dt(t, s):
    s_abs = np.abs(s); return -H0 * np.sqrt(OmegaLambda + OmegaK*(s_abs**2/s0**2) + OmegaM*(s_abs**3/s0**3) + OmegaR*(s_abs**4/s0**4))
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True
sol_bg = solve_ivp(ds_dt, [t0_integration, 12 * s0], [s_bang_init], events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
fcb_time = sol_bg.t_events[0][0]; deltaeta = 6.6e-4 * s0
endtime = fcb_time - deltaeta
s_rec = 1 + z_rec; recConformalTime = sol_bg.t[np.argmin(np.abs(sol_bg.y[0] - s_rec))]
# --- ODE Functions (No changes here) ---
def dX_boltzmann_s(t, X, k):
    s, phi, psi, dr, dm, vr, vm = X[0:7]; fr_all = X[7:]
    sdot, H = ds_dt(t, s), ds_dt(t, s)/s
    rho_m, rho_r = 3*(H0**2)*OmegaM*(np.abs(s/s0)**3), 3*(H0**2)*OmegaR*(np.abs(s/s0)**4)
    phidot = H*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2, fr3 = fr_all[0], fr_all[1]; fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*fr3
    psidot = phidot - (3*H0**2*OmegaR/s0**4/k**2)*(2*s*sdot*fr2 + s**2*fr2dot)
    drdot, dmdot, vrdot, vmdot = (4/3)*(3*phidot+(k**2)*vr), 3*phidot+k**2*vm, -(psi+dr/4), H*vm-psi
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    for j in range(1, l_max - 2):
        l = j + 2; derivatives.append((k/(2*l+1))*(l*fr_all[j-1]-(l+1)*fr_all[j+1]))
    derivatives.append((k*l_max/(2*l_max+1))*fr_all[l_max-3])
    return derivatives
def dX_boltzmann_sigma(t, X, k):
    sigma, phi, psi, dr, dm, vr, vm = X[0:7]; fr_all = X[7:]
    H = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaK/s0**2+OmegaM/s0**3*np.exp(sigma)+OmegaR/s0**4*np.exp(2*sigma))
    rho_m, rho_r = 3*(H0**2)*OmegaM/s0**3*(np.exp(3*sigma)), 3*(H0**2)*OmegaR/s0**4*(np.exp(4*sigma))
    phidot = H*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2, fr3 = fr_all[0], fr_all[1]; fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*fr3
    psidot = phidot - (3*H0**2*OmegaR/s0**4/k**2)*np.exp(2*sigma)*(2*H*fr2+fr2dot)
    drdot, dmdot, vrdot, vmdot = (4/3)*(3*phidot+(k**2)*vr), 3*phidot+k**2*vm, -(psi+dr/4), H*vm-psi
    derivatives = [H, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    for j in range(1, l_max-2):
        l = j+2; derivatives.append((k/(2*l+1))*(l*fr_all[j-1]-(l+1)*fr_all[j+1]))
    derivatives.append((k*l_max/(2*l_max+1))*fr_all[l_max-3])
    return derivatives
def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    H=-(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaK/s0**2+OmegaM/s0**3*np.exp(sigma)+OmegaR/s0**4*np.exp(2*sigma))
    rho_m, rho_r = 3*(H0**2)*OmegaM/s0**3*(np.exp(3*sigma)), 3*(H0**2)*OmegaR/s0**4*(np.exp(4*sigma))
    phidot = H*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot, dmdot, vrdot, vmdot = (4/3)*(3*phidot+k**2*vr), 3*phidot+k**2*vm, -(phi+dr/4), H*vm-phi
    return [H, phidot, drdot, dmdot, vrdot, vmdot]
# --- Data Loading (No changes here) ---
try:
    k_grid_data = np.load(folder_path + 'L70_kvalues.npy'); ABCmatrices = np.load(folder_path + 'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path + 'L70_DEFmatrices.npy'); X1matrices = np.load(folder_path + 'L70_X1matrices.npy')
    X2matrices = np.load(folder_path + 'L70_X2matrices.npy'); recValues = np.load(folder_path + 'L70_recValues.npy')
except FileNotFoundError as e:
    print(f"Error loading necessary data files: {e}"); exit()
Amatrices, Dmatrices = ABCmatrices[:, 0:6, :], DEFmatrices[:, 0:6, :]
def create_matrix_interpolator(k, m): return lambda k_val: np.array([[interp1d(k, m[:,i,j], bounds_error=False, fill_value="extrapolate")(k_val) for j in range(m.shape[2])] for i in range(m.shape[1])])
def create_vector_interpolator(k, v): return lambda k_val: np.array([interp1d(k, v[:,i], bounds_error=False, fill_value="extrapolate")(k_val) for i in range(v.shape[1])])
get_A, get_D, get_X1, get_X2, get_recs = create_matrix_interpolator(k_grid_data, Amatrices), create_matrix_interpolator(k_grid_data, Dmatrices), create_matrix_interpolator(k_grid_data, X1matrices), create_matrix_interpolator(k_grid_data, X2matrices), create_vector_interpolator(k_grid_data, recValues)

# --- MODIFIED Full Solution Generation Function for v_r ---
def generate_vr_solution_to_fcb(k):
    """
    Generates the COMPLETE v_r(eta) solution from Big Bang to FCB for a given k.
    Returns a time array and the corresponding v_r solution array.
    """
    # Get transfer matrices & rec values for this k
    A, D, X1, X2, recs_vec = get_A(k), get_D(k), get_X1(k), get_X2(k), get_recs(k)
    
    # Solve for x_inf = [dr_inf, dm_inf, vr_inf, vmdot_inf]
    M_matrix, x_rec_subset = (A @ X1 + D @ X2)[2:6, :], recs_vec[2:6]
    try:
        x_inf = np.linalg.solve(M_matrix, x_rec_subset)
    except np.linalg.LinAlgError:
        print(f"Warning: Could not solve for x_inf for k={k}. Using zeros."); x_inf = np.zeros(4)
    
    # *** CHANGE: The value at the FCB is now vr_inf ***
    vr_inf = x_inf[2]
    
    # Calculate initial conditions at eta'
    x_prime, y_prime_2_4 = X1 @ x_inf, X2 @ x_inf
    s_prime_val = np.interp(endtime, sol_bg.t, sol_bg.y[0])
    Y_prime = np.zeros(num_variables_boltzmann); Y_prime[0], Y_prime[1:7], Y_prime[7:9] = s_prime_val, x_prime, y_prime_2_4
    
    # Integrate backwards
    sol_part1 = solve_ivp(dX_boltzmann_s, [endtime, swaptime], Y_prime, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    Y_swap = sol_part1.y[:, -1]; Y_swap[0] = np.log(Y_swap[0])
    sol_part2 = solve_ivp(dX_boltzmann_sigma, [swaptime, recConformalTime], Y_swap, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    t_backward = np.concatenate((sol_part1.t, sol_part2.t)); Y_backward_sigma = np.concatenate((sol_part1.y, sol_part2.y), axis=1)
    Y_backward = Y_backward_sigma.copy(); mask = t_backward >= swaptime; Y_backward[0, mask] = np.exp(Y_backward_sigma[0, mask])
    
    # Get solution from Big Bang to Recombination (perfect fluid)
    # The state vector for perfect fluid is [sigma, phi, dr, dm, vr, vm]
    phi1, _ = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0, (1/60)*(-2*k**2+(9*OmegaM**2)/(16*OmegaLambda*OmegaR*s0**2))-2*OmegaK/(15*OmegaLambda*s0**2)
    _, _ = -OmegaM/(4*np.sqrt(3*OmegaR*OmegaLambda))/s0, (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(240*s0**2*OmegaR*OmegaLambda)-8*OmegaK/(15*OmegaLambda*s0**2)
    _, _ = -np.sqrt(3)*OmegaM/(16*s0*np.sqrt(OmegaR*OmegaLambda)), (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(320*s0**2*OmegaR*OmegaLambda)-2*OmegaK/(5*OmegaLambda*s0**2)
    vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-OmegaM**2+8*s0**2*OmegaR*OmegaLambda*k**2)/(160*s0**2*OmegaR*OmegaLambda)+4*OmegaK/(45*OmegaLambda*s0**2)
    _, _, _ = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-3*OmegaM**2+4*s0**2*OmegaR*OmegaLambda*k**2)/(480*s0**2*OmegaR*OmegaLambda)+17*OmegaK/(360*OmegaLambda*s0**2)
    sigma0 = np.log(s_bang_init); phi0 = 1+phi1*t0_integration # simplified
    dr0, dm0 = 0, 0 # simplified
    vr0, vm0 = vr1*t0_integration+vr2*t0_integration**2+vr3*t0_integration**3, 0 # simplified
    Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0_integration, recConformalTime], Y0_perfect, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Stitch the v_r solution from BB to FCB
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1], [fcb_time]))
    
    # *** CHANGE: Extract v_r (index 4 in perfect, 5 in Boltzmann) ***
    vr_perfect = sol_perfect.y[4, :]
    vr_backward = Y_backward[5, ::-1]
    
    vr_left = np.concatenate((vr_perfect, vr_backward, [vr_inf]))
    
    # Sort by time to ensure monotonicity for interpolation
    sort_indices = np.argsort(t_left); t_sorted = t_left[sort_indices]; vr_sorted = vr_left[sort_indices]
    
    return t_sorted, vr_sorted

# =============================================================================
# SECTION 2: GRAM-SCHMIDT ANALYSIS (MODIFIED FOR v_r)
# =============================================================================

def qr_decomposition(basis):
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix.T)
def compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2):
    M = np.dot(orthonormal_functions_1.T, orthonormal_functions_2)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M, M.T))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M.T, M))
    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2
def choose_eigenvalues(eigenvalues, eigenvectors, eigenvalues_threshold, N):
    eigenvalues_valid, eigenvectors_valid = [], []
    for i in range(N):
        if np.abs(eigenvalues[i] - 1.) < eigenvalues_threshold:
            eigenvalues_valid.append(eigenvalues[i])
            eigenvectors_valid.append(eigenvectors[:, i])
    return eigenvalues_valid, eigenvectors_valid
def compute_coefficients(eigenvalues, eigenvectors, transformation_matrix):
    coefficients = np.zeros((len(eigenvalues), transformation_matrix.shape[1]), dtype=float)
    for i in range(len(eigenvalues)):
        coefficients[i, :] = np.dot(np.array(eigenvectors[i]), transformation_matrix)
    return coefficients

# =============================================================================
# SECTION 3: NEW WEIGHTING CALCULATION FUNCTIONS
# =============================================================================

def calculate_weights(coefficients_1):
    """
    Calculate weights for each mode based on the first eigenfunction coefficients.
    Normalized so that sum(weights) = number of weights.
    
    Parameters:
    -----------
    coefficients_1 : np.ndarray
        Coefficient matrix where coefficients_1[i, j] is the coefficient of mode j 
        in eigenfunction i for basis 1
    
    Returns:
    --------
    weights : np.ndarray
        Array of weights for each mode index, normalized so sum = N
    """
    if len(coefficients_1) == 0:
        print("Warning: No valid coefficients found. Returning unit weights.")
        return np.ones(32)  # Default unit weights
    
    # Use the first eigenfunction coefficients
    first_eigenfunction_coeffs = coefficients_1[0, :]
    
    # Calculate |coefficient_1(i)|^2 for each mode i
    raw_weights = np.abs(first_eigenfunction_coeffs)**2
    
    # Normalize so that sum(weights) = number of weights
    N = len(raw_weights)
    weights = raw_weights * N / np.sum(raw_weights)
    
    return weights

def save_weights_for_class(weights, filename="mode_weights.dat", max_index=None):
    """
    Save weights in the format required by the modified CLASS code.
    
    Parameters:
    -----------
    weights : np.ndarray
        Array of weights for each mode
    filename : str
        Output filename
    max_index : int
        Maximum index to save (if None, saves all weights)
    """
    if max_index is None:
        max_index = len(weights) - 1
    
    with open(filename, 'w') as f:
        f.write("# Mode weights for CLASS k-dependent weighting\n")
        f.write("# Format: index  weight\n")
        f.write("# Generated from solve_real_cosmology_weighting.py\n")
        f.write("#\n")
        
        for i in range(min(len(weights), max_index + 1)):
            f.write(f"{i:3d}    {weights[i]:.8e}\n")
    
    print(f"Weights saved to {filename}")
    print(f"Saved {min(len(weights), max_index + 1)} weight values")
    print(f"Weight range: [{np.min(weights):.6e}, {np.max(weights):.6e}]")
    print(f"Weight sum: {np.sum(weights[:min(len(weights), max_index + 1)]):.8f}")

def plot_weights(weights, filename="mode_weights.pdf"):
    """
    Create a plot of the calculated weights.
    """
    plt.figure(figsize=(10, 6))
    mode_indices = np.arange(len(weights))
    
    plt.bar(mode_indices, weights, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Mode Index')
    plt.ylabel('Weight')
    plt.title('K-dependent Mode Weights for CLASS\n' + 
              r'$w_i = |c_1(i)|^2 \cdot N / \sum_j |c_1(j)|^2$, $\sum w_i = N$')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale since weights might vary significantly
    
    # Add statistics text
    stats_text = f'Total modes: {len(weights)}\n'
    stats_text += f'Weight sum: {np.sum(weights):.6f}\n'
    stats_text += f'Max weight: {np.max(weights):.3e}\n'
    stats_text += f'Min weight: {np.min(weights):.3e}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Weight plot saved to {filename}")

if __name__=="__main__":
    N = 32; N_t = 500
    eigenvalues_threshold = 1.e-1; N_plot = 5
    eta_grid = np.linspace(t0_integration, fcb_time, N_t)
    try:
        allowed_K = np.load(folder_path + 'allowedK.npy')
        print(f"Loaded {len(allowed_K)} allowed K values from file.")
        if len(allowed_K) < N: raise ValueError(f"Need {N} allowed K values, found {len(allowed_K)}.")
        allowed_K_basis = allowed_K[:N]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}"); exit()

    basis_1, basis_2 = [], []
    print("\nGenerating basis 1: Closed Universe Model (v_r solutions)...")
    for i in range(1, N + 1):
        k_effective = i * np.sqrt(K) * 4
        print(f"  Generating v_r solution for k_eff = {k_effective:.4f} (i={i})")
        # *** CHANGE: Call the new function ***
        t_sol, vr_sol = generate_vr_solution_to_fcb(k_effective)
        vr_interpolated = interp1d(t_sol, vr_sol, bounds_error=False, fill_value=0.0)
        basis_1.append(vr_interpolated(eta_grid))

    print("\nGenerating basis 2: Palindromic Flat Universe Model (v_r solutions)...")
    for k_val in allowed_K_basis:
        print(f"  Generating v_r solution for allowed_k = {k_val:.4f}")
        # *** CHANGE: Call the new function ***
        t_sol, vr_sol = generate_vr_solution_to_fcb(k_val)
        vr_interpolated = interp1d(t_sol, vr_sol, bounds_error=False, fill_value=0.0)
        basis_2.append(vr_interpolated(eta_grid))
    
    print("\nPerforming QR decomposition and eigenvalue analysis...")
    orthonormal_functions_1, transformation_matrix_1 = qr_decomposition(np.array(basis_1).T)
    orthonormal_functions_2, transformation_matrix_2 = qr_decomposition(np.array(basis_2).T)
    eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2)
    eigenvalues_valid_1, eigenvectors_valid_1 = choose_eigenvalues(eigenvalues_1, eigenvectors_1, eigenvalues_threshold, N)
    eigenvalues_valid_2, eigenvectors_valid_2 = choose_eigenvalues(eigenvalues_2, eigenvectors_2, eigenvalues_threshold, N)
    coefficients_1 = compute_coefficients(eigenvalues_valid_1, eigenvectors_valid_1, transformation_matrix_1)
    coefficients_2 = compute_coefficients(eigenvalues_valid_2, eigenvectors_valid_2, transformation_matrix_2)

    # =============================================================================
    # SECTION 4: NEW WEIGHTING CALCULATION AND OUTPUT
    # =============================================================================
    
    print("\n" + "="*60)
    print("CALCULATING K-DEPENDENT WEIGHTS FOR CLASS")
    print("="*60)
    
    # Calculate weights based on the first eigenfunction of basis 1
    weights = calculate_weights(coefficients_1)
    
    # Print weight summary
    print(f"\nWeight calculation summary:")
    print(f"Number of modes: {len(weights)}")
    print(f"Number of valid eigenfunctions found: {len(coefficients_1)}")
    print(f"Weight sum: {np.sum(weights):.8f}")
    print(f"Maximum weight: {np.max(weights):.6e} (mode {np.argmax(weights)})")
    print(f"Minimum weight: {np.min(weights):.6e} (mode {np.argmin(weights)})")
    
    # Show first few weights
    print(f"\nFirst 10 weights:")
    for i in range(min(10, len(weights))):
        print(f"  Mode {i:2d}: {weights[i]:.8e}")
    
    # Save weights for CLASS
    save_weights_for_class(weights, "mode_weights.dat", max_index=31)
    
    # Create weight plot
    plot_weights(weights, "mode_weights.pdf")
    
    # Also save raw coefficients for analysis
    np.save("coefficients_basis1.npy", coefficients_1)
    np.save("coefficients_basis2.npy", coefficients_2)
    np.save("mode_weights.npy", weights)
    print(f"\nRaw data saved:")
    print(f"  coefficients_basis1.npy - Basis 1 coefficients")
    print(f"  coefficients_basis2.npy - Basis 2 coefficients") 
    print(f"  mode_weights.npy - Calculated weights")

    print("\nPlotting results for comparison...")
    if N_plot > len(eigenvalues_valid_1): N_plot = len(eigenvalues_valid_1)
    if N_plot > 0:
        fig, axs = plt.subplots(N_plot, 1, figsize=(10, 2*N_plot), sharex=True, constrained_layout=True)
        if N_plot == 1: axs = [axs]
        fig.suptitle(r"Comparison of Common Eigenfunctions of $v_r(\eta)$" + "\n(Used for Weight Calculation)")
        for i in range(N_plot):
            ax = axs[i]; solution_1, solution_2 = np.zeros_like(eta_grid), np.zeros_like(eta_grid)
            for j in range(N):
                solution_1 += coefficients_1[i, j] * basis_1[j]
                solution_2 += coefficients_2[i, j] * basis_2[j]

            # --- NEW: Robust sign alignment using dot product ---
            # This ensures that solution_2 is flipped if it's anti-aligned with solution_1
            if np.dot(solution_1, solution_2) < 0:
                solution_2 *= -1

            ax.plot(eta_grid, solution_1, color='red', linewidth=2.5, label='From Basis 1 (Closed)')
            ax.plot(eta_grid, solution_2, color='green', linestyle='--', linewidth=2.0, label='From Basis 2 (Flat)')
            ax.grid(True, linestyle='--', alpha=0.6); ax.set_ylabel(f"Eigenfunc. {i+1}"); ax.legend()
            
            # Add weight information for first eigenfunction
            if i == 0:
                weight_info = f"Weights calculated from this eigenfunction"
                ax.text(0.02, 0.95, weight_info, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        axs[-1].set_xlabel(r"Conformal Time $\eta$")
        plt.savefig(f"full_vr_eigenfunction_match_N{N:d}_Nt{N_t:d}_aligned_with_weights.pdf")
        plt.show()
    else:
        print("No valid eigenfunctions found to plot.")

print("--- Total execution time: %s seconds ---" % (time.time() - start_time))
print("\n" + "="*60)
print("WEIGHT CALCULATION COMPLETE!")
print("="*60)
print("Files generated:")
print("  - mode_weights.dat      (for CLASS input)")
print("  - mode_weights.pdf      (visualization)")
print("  - mode_weights.npy      (raw weights)")
print("  - coefficients_basis*.npy (raw coefficients)")
print("\nTo use with CLASS, add to your .ini file:")
print("  weighting_filename = mode_weights.dat")
print("  max_weight_index = 31")