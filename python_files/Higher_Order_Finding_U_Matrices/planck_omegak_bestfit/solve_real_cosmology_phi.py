# -*- coding: utf-8 -*-
"""
This script performs a Gram-Schmidt orthonormalization and eigenvalue analysis
on two different bases of the COMPLETE cosmological perturbation solutions for phi(eta).

The solution generator is based on the user's verified script for plotting
the full palindromic evolution.

- basis_1 is constructed from wavenumbers 'k' in a closed universe model.
- basis_2 is constructed from the 'allowed' wavenumbers for a palindromic flat universe.

The analysis compares the common eigenfunctions found in both bases.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# SECTION 1: FULL SOLUTION GENERATION ENGINE
# This section contains the full logic to generate the complete phi(eta)
# solution from the Big Bang to the FCB, based on your working script.
# =============================================================================
start_time = time.time()
folder_path = './data/'

# --- Cosmological Parameters and Setup ---
# NOTE: Using parameters consistent with your code(2) and code(3), which
# likely generated the data files. Your provided plotting script had slightly
# different (unscaled by h**2) values for OmegaM, which I have corrected here.
h = 0.5409
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
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

# --- ODE Functions (Copied from your script) ---
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

# --- Data Loading and Interpolation for Transfer Matrices ---
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

# --- Encapsulated Full Solution Generation Function ---
def generate_phi_solution_to_fcb(k):
    """
    Generates the COMPLETE phi(eta) solution from Big Bang to FCB for a given k.
    This function combines all the logic from the verified plotting script.
    Returns a time array and the corresponding phi solution array.
    """
    # Get transfer matrices & rec values for this k
    A, D, X1, X2, recs_vec = get_A(k), get_D(k), get_X1(k), get_X2(k), get_recs(k)
    
    # Solve for x_inf
    M_matrix, x_rec_subset = (A @ X1 + D @ X2)[2:6, :], recs_vec[2:6]
    try:
        x_inf = np.linalg.solve(M_matrix, x_rec_subset)
    except np.linalg.LinAlgError:
        print(f"Warning: Could not solve for x_inf for k={k}. Using zeros."); x_inf = np.zeros(4)

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
    phi1, phi2 = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0, (1/60)*(-2*k**2+(9*OmegaM**2)/(16*OmegaLambda*OmegaR*s0**2))-2*OmegaK/(15*OmegaLambda*s0**2)
    dr1, dr2 = -OmegaM/(4*np.sqrt(3*OmegaR*OmegaLambda))/s0, (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(240*s0**2*OmegaR*OmegaLambda)-8*OmegaK/(15*OmegaLambda*s0**2)
    dm1, dm2 = -np.sqrt(3)*OmegaM/(16*s0*np.sqrt(OmegaR*OmegaLambda)), (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(320*s0**2*OmegaR*OmegaLambda)-2*OmegaK/(5*OmegaLambda*s0**2)
    vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-OmegaM**2+8*s0**2*OmegaR*OmegaLambda*k**2)/(160*s0**2*OmegaR*OmegaLambda)+4*OmegaK/(45*OmegaLambda*s0**2)
    vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-3*OmegaM**2+4*s0**2*OmegaR*OmegaLambda*k**2)/(480*s0**2*OmegaR*OmegaLambda)+17*OmegaK/(360*OmegaLambda*s0**2)
    sigma0 = np.log(s_bang_init)
    phi0, dr0, dm0 = 1+phi1*t0_integration+phi2*t0_integration**2, -2+dr1*t0_integration+dr2*t0_integration**2, -1.5+dm1*t0_integration+dm2*t0_integration**2
    vr0, vm0 = vr1*t0_integration+vr2*t0_integration**2+vr3*t0_integration**3, vm1*t0_integration+vm2*t0_integration**2+vm3*t0_integration**3
    Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0_integration, recConformalTime], Y0_perfect, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    # Stitch the solution from BB to FCB, including the FCB point itself
    phi_inf = (X1 @ x_inf)[0]
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1], [fcb_time]))
    phi_left = np.concatenate((sol_perfect.y[1, :], Y_backward[1, ::-1], [phi_inf]))  
    
    # Sort by time to ensure monotonicity for interpolation
    sort_indices = np.argsort(t_left); t_sorted = t_left[sort_indices]; phi_sorted = phi_left[sort_indices]
    
    return t_sorted, phi_sorted

# =============================================================================
# SECTION 2: GRAM-SCHMIDT ANALYSIS
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

if __name__=="__main__":
    N = 3; N_t = 500
    eigenvalues_threshold = 1.e-1; N_plot = 5
    eta_grid = np.linspace(t0_integration, fcb_time, N_t)
    try:
        allowed_K = np.load(folder_path + 'allowedK.npy')
        if len(allowed_K) < N: raise ValueError(f"Need {N} allowed K values, found {len(allowed_K)}.")
        allowed_K_basis = allowed_K[:N]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}"); exit()

    basis_1, basis_2 = [], []
    # print("\nGenerating basis 1: Closed Universe Model (Full Evolution)...")
    # for i in range(1, N + 1):
    #     k_effective = i * np.sqrt(K)
    #     print(f"  Generating solution for k_eff = {k_effective:.4f} (i={i})")
    #     t_sol, phi_sol = generate_phi_solution_to_fcb(k_effective)
    #     phi_interpolated = interp1d(t_sol, phi_sol, bounds_error=False, fill_value=0.0)
    #     basis_1.append(phi_interpolated(eta_grid))

    print("\nGenerating basis 2: Palindromic Flat Universe Model (Full Evolution)...")
    for k_val in allowed_K_basis:
        print(f"  Generating solution for allowed_k = {k_val:.4f}")
        t_sol, phi_sol = generate_phi_solution_to_fcb(k_val)
        phi_interpolated = interp1d(t_sol, phi_sol, bounds_error=False, fill_value=0.0)
        basis_2.append(phi_interpolated(eta_grid))
        plt.plot(eta_grid, phi_interpolated(eta_grid), label=f'k={k_val:.4f}', alpha=0.5)
    plt.title("Palindromic Flat Universe Solutions")
    plt.xlabel(r"Conformal Time $\eta$"); plt.ylabel(r"$\Phi(\eta)$")
    plt.ylim(-2.1, 2.2)
    plt.grid(True); plt.legend(); plt.savefig(f"palindromic_universe_solutions_N{N:d}_Nt{N_t:d}.pdf")

#     print("\nPerforming QR decomposition and eigenvalue analysis...")
#     orthonormal_functions_1, transformation_matrix_1 = qr_decomposition(np.array(basis_1).T)
#     orthonormal_functions_2, transformation_matrix_2 = qr_decomposition(np.array(basis_2).T)
#     eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2)
#     eigenvalues_valid_1, eigenvectors_valid_1 = choose_eigenvalues(eigenvalues_1, eigenvectors_1, eigenvalues_threshold, N)
#     eigenvalues_valid_2, eigenvectors_valid_2 = choose_eigenvalues(eigenvalues_2, eigenvectors_2, eigenvalues_threshold, N)
#     coefficients_1 = compute_coefficients(eigenvalues_valid_1, eigenvectors_valid_1, transformation_matrix_1)
#     coefficients_2 = compute_coefficients(eigenvalues_valid_2, eigenvectors_valid_2, transformation_matrix_2)

#     print("\nPlotting results for comparison...")
#     if N_plot > len(eigenvalues_valid_1):
#         N_plot = len(eigenvalues_valid_1)
#         print(f"Warning: Only {N_plot} valid eigenvalues found. Adjusting N_plot.")

#     if N_plot > 0:
#         fig, axs = plt.subplots(N_plot, 1, figsize=(10, 2*N_plot), sharex=True, constrained_layout=True)
#         if N_plot == 1: axs = [axs]
#         fig.suptitle(r"Comparison of Common Eigenfunctions $\Phi(\eta)$")
#         for i in range(N_plot):
#             ax = axs[i]
#             solution_1 = np.zeros_like(eta_grid); solution_2 = np.zeros_like(eta_grid)
#             for j in range(N):
#                 solution_1 += coefficients_1[i, j] * basis_1[j]
#                 solution_2 += coefficients_2[i, j] * basis_2[j]
#             if solution_1[0] < 0: solution_1 *= -1
#             if solution_2[0] < 0: solution_2 *= -1
#             ax.plot(eta_grid, solution_1, color='red', linewidth=2.5, label='From Basis 1 (Closed)')
#             ax.plot(eta_grid, solution_2, color='green', linestyle='--', linewidth=2.0, label='From Basis 2 (Flat)')
#             ax.grid(True, linestyle='--', alpha=0.6); ax.set_ylabel(f"Eigenfunc. {i+1}"); ax.legend()
#         axs[-1].set_xlabel(r"Conformal Time $\eta$")
#         plt.savefig(f"full_eigenfunction_match_N{N:d}_Nt{N_t:d}.pdf")
#     else:
#         print("No valid eigenfunctions found to plot.")

# print("--- Total execution time: %s seconds ---" % (time.time() - start_time))