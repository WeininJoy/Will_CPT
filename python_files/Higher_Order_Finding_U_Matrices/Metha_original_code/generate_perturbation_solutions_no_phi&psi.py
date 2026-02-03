# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model using modified ODEs where phi and psi are eliminated using equations (14) and (15)
from the Evidence_for_a_Palindromic_Universe.pdf.

The ODEs now only contain x=[dr,dm,vr,vm] and y terms, with phi and psi computed
analytically from the other variables when needed.

Based on Metha's original flat universe formulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# 1. SETUP: Parameters, Constants, and ODE Functions
# =============================================================================

print("--- Setting up parameters and functions ---")
folder_path = './data_all_k/'

# Metha's original parameters (flat universe)
OmegaLambda = 0.679
OmegaM = 0.321
OmegaR = 9.24e-5
H0 = 1/np.sqrt(3*OmegaLambda)
z_rec = 1090.30

# Tolerances and constants
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 73  # Reduced by 2 since we removed phi and psi
swaptime = 2
endtime = 6.15
deltaeta = 6.150659839680297 - endtime
Hinf = H0*np.sqrt(OmegaLambda)

# Background evolution
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8

# Initial conditions for background
smin1 = np.sqrt(3*OmegaLambda/OmegaR)
szero = - OmegaM/(4*OmegaR)
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3))
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2)
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3))/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
s4 = -(OmegaM**5)/(9216*(OmegaR**3)*(OmegaLambda**2))

s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4

# Background integration
print('Performing Initial Background Integration')
sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Find FCB and recombination times
idxfcb = np.where(np.diff(np.sign(sol.y[0])) != 0)[0]
fcb_time = 0.5*(sol.t[idxfcb[0]] + sol.t[idxfcb[0] + 1])
s_rec = 1+z_rec
recScaleFactorDifference = abs(sol.y[0] - s_rec)
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

print(f"FCB Time (eta_FCB): {fcb_time:.4f}")
print(f"Recombination Time (eta_rec): {recConformalTime:.4f}")
print(f"Integration Start Time (eta'): {endtime:.4f}")

# Pre-computed constants for performance
H0_sq = H0**2
three_H0_sq = 3*H0_sq
four_thirds = 4/3
eight_fifteenths = 8/15
three_fifths = 3/5

# Create optimized ODE functions with pre-computed k-dependent constants
def create_optimized_s_ode(k):
    """Create optimized s-evolution ODE with pre-computed k-dependent constants"""
    k_sq = k**2
    phi_coeff = -three_H0_sq/(2*k_sq)
    psi_coeff = -(three_H0_sq*OmegaR/k_sq)
    fr2dot_coeff = -eight_fifteenths*k_sq
    point_six_k = 0.6*k
    
    def dX_boltzmann_s_optimized(t, X):
        s, dr, dm, vr, vm, fr2 = X[0:6]
        
        # Pre-compute powers
        s_cubed = s**3
        s_fourth = s_cubed * s
        s_sq = s**2
        abs_s = abs(s)
        abs_s_cubed = abs(s_cubed)
        abs_s_fourth = abs(s_fourth)
        
        sdot = -H0*np.sqrt((OmegaLambda + OmegaM*abs_s_cubed + OmegaR*abs_s_fourth))
        sdot_over_s = sdot/s
        
        # Inline phi and psi calculations
        phi = phi_coeff * (OmegaM*abs_s*(dm + 3*sdot_over_s*vm) + OmegaR*s_sq*(dr + 4*sdot_over_s*vr))
        psi = phi + psi_coeff*s_sq*fr2
        
        # Compute phidot efficiently
        rho_terms = (four_thirds*three_H0_sq*OmegaR*abs_s_fourth*vr + three_H0_sq*OmegaM*abs_s_cubed*vm)
        phidot = sdot_over_s*psi - rho_terms/(2*s_sq)
        
        fr2dot = fr2dot_coeff*vr - point_six_k*X[6]
        
        drdot = four_thirds*(3*phidot + k_sq*vr)
        dmdot = 3*phidot + vm*k_sq
        vrdot = -(psi + dr*0.25) + fr2*0.5
        vmdot = sdot_over_s*vm - psi
        
        derivatives = [sdot, drdot, dmdot, vrdot, vmdot, fr2dot]
        
        # For l>2 terms
        for j in range(6, num_variables-1):
            l = j - 3
            derivatives.append(k/(2*l+1)*(l*X[j-1] - (l+1)*X[j+1]))
        
        lastderiv = k*X[num_variables-2] - ((num_variables-2)*X[num_variables-1])/t
        derivatives.append(lastderiv)
        return derivatives
    
    return dX_boltzmann_s_optimized

def create_optimized_sigma_ode(k):
    """Create optimized sigma-evolution ODE with pre-computed k-dependent constants"""
    k_sq = k**2
    phi_coeff = -three_H0_sq/(2*k_sq)
    psi_coeff = -(three_H0_sq*OmegaR/k_sq)
    fr2dot_coeff = -eight_fifteenths*k_sq
    three_fifths_k = three_fifths*k
    
    def dX_boltzmann_sigma_optimized(t, X):
        sigma, dr, dm, vr, vm, fr2 = X[0:6]
        
        # Pre-compute exponentials
        exp_neg2sigma = np.exp(-2*sigma)
        exp_sigma = np.exp(sigma)
        exp_2sigma = np.exp(2*sigma)
        exp_3sigma = exp_sigma * exp_2sigma
        exp_4sigma = exp_2sigma * exp_2sigma
        
        sigmadot = -H0*np.sqrt((OmegaLambda*exp_neg2sigma + OmegaM*exp_sigma + OmegaR*exp_2sigma))
        
        # Inline phi and psi calculations
        phi = phi_coeff * (OmegaM*exp_sigma*(dm + 3*sigmadot*vm) + OmegaR*exp_2sigma*(dr + 4*sigmadot*vr))
        psi = phi + psi_coeff*exp_2sigma*fr2
        
        # Compute phidot efficiently
        rho_terms = (four_thirds*three_H0_sq*OmegaR*exp_4sigma*vr + three_H0_sq*OmegaM*exp_3sigma*vm)
        phidot = sigmadot*psi - rho_terms/(2*exp_2sigma)
        
        fr2dot = fr2dot_coeff*vr - three_fifths_k*X[6]
        
        drdot = four_thirds*(3*phidot + k_sq*vr)
        dmdot = 3*phidot + vm*k_sq
        vrdot = -(psi + dr*0.25) + fr2*0.5
        vmdot = sigmadot*vm - psi
        
        derivatives = [sigmadot, drdot, dmdot, vrdot, vmdot, fr2dot]
        
        # For l>2 terms
        for j in range(6, num_variables-1):
            l = j - 3
            derivatives.append(k/(2*l+1)*(l*X[j-1] - (l+1)*X[j+1]))
        
        lastderiv = k*X[num_variables-2] - ((num_variables-2)*X[num_variables-1])/t
        derivatives.append(lastderiv)
        return derivatives
    
    return dX_boltzmann_sigma_optimized

def create_optimized_perfect_ode(k):
    """Create optimized perfect fluid ODE with pre-computed k-dependent constants"""
    k_sq = k**2
    phi_coeff = -three_H0_sq/(2*k_sq)
    
    def dX_perfect_sigma_optimized(t, X):
        sigma, dr, dm, vr, vm = X
        
        # Pre-compute exponentials
        exp_neg2sigma = np.exp(-2*sigma)
        exp_sigma = np.exp(sigma)
        exp_2sigma = np.exp(2*sigma)
        exp_3sigma = exp_sigma * exp_2sigma
        exp_4sigma = exp_2sigma * exp_2sigma
        
        sigmadot = -H0*np.sqrt((OmegaLambda*exp_neg2sigma + OmegaM*exp_sigma + OmegaR*exp_2sigma))
        
        # Inline phi calculation (fr2=0, so psi=phi)
        phi = phi_coeff * (OmegaM*exp_sigma*(dm + 3*sigmadot*vm) + OmegaR*exp_2sigma*(dr + 4*sigmadot*vr))
        
        # Compute derivatives efficiently
        rho_terms = (four_thirds*three_H0_sq*OmegaR*exp_4sigma*vr + three_H0_sq*OmegaM*exp_3sigma*vm)
        phidot = sigmadot*phi - rho_terms/(2*exp_2sigma)
        
        drdot = four_thirds*(3*phidot + k_sq*vr)
        dmdot = 3*phidot + k_sq*vm
        vrdot = -(phi + dr*0.25)
        vmdot = sigmadot*vm - phi
        
        return [sigmadot, drdot, dmdot, vrdot, vmdot]
    
    return dX_perfect_sigma_optimized

# Helper functions for computing phi and psi (for plotting only)
def compute_phi_fast(s, dr, dm, vr, vm, k):
    """Fast phi computation for plotting"""
    sdot = -H0*np.sqrt((OmegaLambda + OmegaM*abs(s**3) + OmegaR*abs(s**4)))
    return -(three_H0_sq/(2*k**2)) * (OmegaM*abs(s)*(dm + 3*(sdot/s)*vm) + OmegaR*s**2*(dr + 4*(sdot/s)*vr))

def compute_psi_fast(s, dr, dm, vr, vm, fr2, k):
    """Fast psi computation for plotting"""
    phi = compute_phi_fast(s, dr, dm, vr, vm, k)
    return phi - (three_H0_sq*OmegaR/k**2)*s**2*fr2

def compute_phi_sigma_fast(sigma, dr, dm, vr, vm, k):
    """Fast phi computation with sigma coordinate for plotting"""
    sigmadot = -H0*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)+OmegaR*np.exp(2*sigma)))
    return -(three_H0_sq/(2*k**2)) * (OmegaM*np.exp(sigma)*(dm + 3*sigmadot*vm) + OmegaR*np.exp(2*sigma)*(dr + 4*sigmadot*vr))

# Modified ODE system for full Boltzmann hierarchy (sigma-evolution) - no phi, psi
def dX_boltzmann_sigma_no_phi_psi(t, X, k):
    sigma, dr, dm, vr, vm, fr2 = X[0:6]
    
    # Pre-compute exponentials and constants
    exp_neg2sigma = np.exp(-2*sigma)
    exp_sigma = np.exp(sigma)
    exp_2sigma = np.exp(2*sigma)
    exp_3sigma = exp_sigma * exp_2sigma
    exp_4sigma = exp_2sigma * exp_2sigma
    k_sq = k**2
    
    sigmadot = -H0*np.sqrt((OmegaLambda*exp_neg2sigma + OmegaM*exp_sigma + OmegaR*exp_2sigma))
    
    # Inline phi calculation from equation (14)
    phi_coeff = -three_H0_sq/(2*k_sq)
    phi = phi_coeff * (OmegaM*exp_sigma*(dm + 3*sigmadot*vm) + OmegaR*exp_2sigma*(dr + 4*sigmadot*vr))
    
    # Inline psi calculation from equation (15)
    psi = phi - (three_H0_sq*OmegaR/k_sq)*exp_2sigma*fr2
    
    # Compute phidot efficiently
    rho_terms = (four_thirds*three_H0_sq*OmegaR*exp_4sigma*vr + three_H0_sq*OmegaM*exp_3sigma*vm)
    phidot = sigmadot*psi - rho_terms/(2*exp_2sigma)
    
    fr2dot = -eight_fifteenths*k_sq*vr - three_fifths*k*X[6]
    
    drdot = four_thirds*(3*phidot + k_sq*vr)
    dmdot = 3*phidot + vm*k_sq
    vrdot = -(psi + dr*0.25) + fr2*0.5
    vmdot = sigmadot*vm - psi
    
    derivatives = [sigmadot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    # For l>2 terms - indices shifted down by 2
    for j in range(6, num_variables-1):
        l = j - 3
        derivatives.append(k/(2*l+1)*(l*X[j-1] - (l+1)*X[j+1]))
    
    # Final term
    lastderiv = k*X[num_variables-2] - ((num_variables-2)*X[num_variables-1])/t
    derivatives.append(lastderiv)
    return derivatives

# Modified ODE system for perfect fluid (sigma-evolution) - no phi, psi
def dX_perfect_sigma_no_phi_psi(t, X, k):
    sigma, dr, dm, vr, vm = X
    
    # Pre-compute exponentials and constants
    exp_neg2sigma = np.exp(-2*sigma)
    exp_sigma = np.exp(sigma)
    exp_2sigma = np.exp(2*sigma)
    exp_3sigma = exp_sigma * exp_2sigma
    exp_4sigma = exp_2sigma * exp_2sigma
    k_sq = k**2
    
    sigmadot = -H0*np.sqrt((OmegaLambda*exp_neg2sigma + OmegaM*exp_sigma + OmegaR*exp_2sigma))
    
    # Inline phi calculation (fr2=0 for perfect fluid, so psi=phi)
    phi_coeff = -three_H0_sq/(2*k_sq)
    phi = phi_coeff * (OmegaM*exp_sigma*(dm + 3*sigmadot*vm) + OmegaR*exp_2sigma*(dr + 4*sigmadot*vr))
    
    # Compute derivatives efficiently
    rho_terms = (four_thirds*three_H0_sq*OmegaR*exp_4sigma*vr + three_H0_sq*OmegaM*exp_3sigma*vm)
    phidot = sigmadot*phi - rho_terms/(2*exp_2sigma)
    
    drdot = four_thirds*(3*phidot + k_sq*vr)
    dmdot = 3*phidot + k_sq*vm
    vrdot = -(phi + dr*0.25)
    vmdot = sigmadot*vm - phi
    
    return [sigmadot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 2. DATA LOADING AND INTERPOLATION
# =============================================================================

print("\n--- Loading and interpolating pre-computed data ---")

try:
    kvalues = np.load(folder_path+'L70_kvalues.npy')
    ABCmatrices = np.load(folder_path+'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path+'L70_DEFmatrices.npy')
    GHIvectors = np.load(folder_path+'L70_GHIvectors.npy')
    X1matrices = np.load(folder_path+'L70_X1matrices.npy')
    X2matrices = np.load(folder_path+'L70_X2matrices.npy')
    recValues = np.load(folder_path+'L70_recValues.npy')
    allowedK = np.load('allowedK.npy')
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the current directory.")
    exit()

# Extract sub-matrices - only need rows 2:6 since we removed phi, psi
Amatrices = ABCmatrices[:, 2:6, :]  # Remove phi, psi rows
Dmatrices = DEFmatrices[:, 2:6, :]  # Remove phi, psi rows

# Create interpolation functions for each element of the matrices
def create_matrix_interpolator(k_grid, matrix_array):
    rows, cols = matrix_array.shape[1], matrix_array.shape[2]
    interp_funcs = [[interp1d(k_grid, matrix_array[:, i, j], bounds_error=False, fill_value="extrapolate") for j in range(cols)] for i in range(rows)]
    def get_matrix(k):
        return np.array([[func(k) for func in row] for row in interp_funcs])
    return get_matrix

def create_vector_interpolator(k_grid, vector_array):
    cols = vector_array.shape[1]
    interp_funcs = [interp1d(k_grid, vector_array[:, i], bounds_error=False, fill_value="extrapolate") for i in range(cols)]
    def get_vector(k):
        return np.array([func(k) for func in interp_funcs])
    return get_vector
    
get_A = create_matrix_interpolator(kvalues, Amatrices)
get_D = create_matrix_interpolator(kvalues, Dmatrices)
get_X1 = create_matrix_interpolator(kvalues, X1matrices)
get_X2 = create_matrix_interpolator(kvalues, X2matrices)
get_recs = create_vector_interpolator(kvalues, recValues)

# =============================================================================
# 3. CALCULATE AND INTEGRATE SOLUTIONS
# =============================================================================

print("\n--- Calculating solutions for the first 3 allowed modes ---")

solutions = []
num_modes_to_plot = min(3, len(allowedK))

for i in range(num_modes_to_plot):
    k = allowedK[i+2]
    print(f"\nProcessing mode n={i+1} with k={k:.4f}")

    # --- a) Get interpolated matrices for this k ---
    A = get_A(k)
    D = get_D(k)
    X1 = get_X1(k)
    X2 = get_X2(k)
    recs_vec = get_recs(k)

    # --- b) Solve for X_inf using only dr, dm, vr, vm components ---
    M_matrix = (A @ X1 + D @ X2)  # Now 4x4 since we removed phi, psi
    x_rec_subset = recs_vec[2:6]  # Only dr, dm, vr, vm
    
    try:
        # Use more robust solver
        x_inf = np.linalg.lstsq(M_matrix, x_rec_subset, rcond=None)[0]
        cond_num = np.linalg.cond(M_matrix)
        print(f"Condition number: {cond_num:.2e}")
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Skipping mode.")
        continue

    # --- c) Calculate initial conditions at eta' (endtime) ---
    x_prime_reduced = X1[2:6, :] @ x_inf  # Only dr, dm, vr, vm from X1
    y_prime_2_4 = X2 @ x_inf
    s_prime_val = np.interp(endtime, sol.t, sol.y[0])

    # Construct full state vector (without phi, psi)
    Y_prime = np.zeros(num_variables)
    Y_prime[0] = s_prime_val
    Y_prime[1:5] = x_prime_reduced  # [dr, dm, vr, vm]
    Y_prime[5:7] = y_prime_2_4  # [F_2, F_3]
    
    # --- d) Create optimized ODE functions for this k ---
    ode_s_func = create_optimized_s_ode(k)
    ode_sigma_func = create_optimized_sigma_ode(k)
    ode_perfect_func = create_optimized_perfect_ode(k)
    
    # --- e) Integrate backwards from endtime to recConformalTime ---
    sol_part1 = solve_ivp(ode_s_func, [endtime, swaptime], Y_prime, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol)
    
    Y_swap = sol_part1.y[:, -1]
    Y_swap[0] = np.log(Y_swap[0])  # Convert s to sigma
    sol_part2 = solve_ivp(ode_sigma_func, [swaptime, recConformalTime], Y_swap, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol)

    t_backward = np.concatenate((sol_part1.t, sol_part2.t))
    Y_backward_sigma = np.concatenate((sol_part1.y, sol_part2.y), axis=1)
    Y_backward = Y_backward_sigma.copy()
    mask = t_backward >= swaptime
    Y_backward[0, mask] = np.exp(Y_backward_sigma[0, mask])  # Convert sigma back to s

    # --- e) Get solution from Big Bang to Recombination (perfect fluid) ---
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))
    
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda)
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda)
    
    vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda)
    vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda)
    
    sigma0 = np.log(s0)
    dr0 = -2 + dr1*t0 + dr2*t0**2
    dm0 = -1.5 + dm1*t0 + dm2*t0**2
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3
    
    Y0_perfect = [sigma0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(ode_perfect_func, [t0, recConformalTime], Y0_perfect,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol)

    # --- f) Stitch and Reflect ---
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1]))

    # Construct full solution with phi and psi computed from other variables
    Y_perfect_full = np.zeros((num_variables + 2, len(sol_perfect.t)))  # +2 for phi, psi
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])  # s
    
    # Compute phi and psi for perfect fluid solution
    for idx in range(len(sol_perfect.t)):
        sigma = sol_perfect.y[0, idx]
        dr, dm, vr, vm = sol_perfect.y[1:5, idx]
        phi = compute_phi_sigma_fast(sigma, dr, dm, vr, vm, k)
        Y_perfect_full[1, idx] = phi  # phi
        Y_perfect_full[2, idx] = phi  # psi = phi for perfect fluid
    
    Y_perfect_full[3:7,:] = sol_perfect.y[1:5,:]  # dr, dm, vr, vm
    # Higher order terms remain zero for perfect fluid

    # For backward solution, compute phi and psi
    Y_backward_full = np.zeros((num_variables + 2, Y_backward.shape[1]))
    Y_backward_full[0, :] = Y_backward[0, :]  # s
    
    for idx in range(Y_backward.shape[1]):
        s_val = Y_backward[0, idx]
        dr, dm, vr, vm = Y_backward[1:5, idx]
        fr2 = Y_backward[5, idx]
        phi = compute_phi_fast(s_val, dr, dm, vr, vm, k)
        psi = compute_psi_fast(s_val, dr, dm, vr, vm, fr2, k)
        Y_backward_full[1, idx] = phi
        Y_backward_full[2, idx] = psi
    
    Y_backward_full[3:7, :] = Y_backward[1:5, :]  # dr, dm, vr, vm
    Y_backward_full[7:, :] = Y_backward[5:, :]    # Higher order terms
    
    Y_left = np.concatenate((Y_perfect_full, Y_backward_full[:, ::-1]), axis=1)

    t_right = 2 * fcb_time - t_left[::-1]
    
    # Symmetry matrix for reflection - adjusted for original indexing
    symm = np.ones(num_variables + 2)
    symm[[6, 7]] = -1  # vm and fr2 are antisymmetric (adjusted indices)
    for l_idx in range(num_variables - 6):
        l = l_idx + 3
        if l % 2 != 0: symm[8 + l_idx] = -1
    S = np.diag(symm)
    
    Y_right = S @ Y_left[:, ::-1]

    t_full = np.concatenate((t_left, t_right))
    Y_full = np.concatenate((Y_left, Y_right), axis=1)
    
    solutions.append({'t': t_full, 'Y': Y_full})

# =============================================================================
# 4. PLOTTING
# =============================================================================

print("\n--- Plotting solutions ---")

fig, axes = plt.subplots(num_modes_to_plot, 1, figsize=(10, 8), sharex=True)
if num_modes_to_plot == 1: axes = [axes]

labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$', r'$\phi$', r'$\psi$']
indices = [6, 3, 7, 4, 1, 2]  # Adjusted indices: vr, dr, vm, dm, phi, psi
colors = ['blue', 'red', 'green', 'orange', 'magenta', 'cyan']

for i, sol in enumerate(solutions):
    ax = axes[i]
    for label, index, color in zip(labels, indices, colors):
        ax.plot(sol['t'], sol['Y'][index, :], label=label, color=color, linewidth=1.2)

    ax.axvline(fcb_time, color='k', linestyle='--', label='FCB')
    ax.axvline(2 * fcb_time, color='k', linestyle='--', label='Big Crunch')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(f'n = {i+1}', fontsize=12)
    ax.set_xlim(0, 2 * fcb_time * 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3)
axes[-1].set_xlabel(r'$\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Evolution of Perturbation Solutions (No φ, ψ in ODEs)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('perturbation_solutions_no_phi_psi.pdf')
plt.show()

print("\n--- Analysis Complete ---")
print("Phi and psi were computed algebraically from equations (14) and (15)")
print("ODEs only evolved dr, dm, vr, vm, and higher-order terms")