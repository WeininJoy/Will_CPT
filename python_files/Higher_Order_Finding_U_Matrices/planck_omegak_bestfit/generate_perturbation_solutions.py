# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model, following the methodology outlined in the paper "Evidence for a Palindromic Universe".

It uses pre-computed data from transfer matrix calculations to solve for the
boundary conditions at the Future Conformal Boundary (FCB) for each allowed wavenumber k.
It then integrates the perturbation equations to obtain the full evolution of the
variables and plots the results.

CORRECTED VERSION: Fixes the IndexError in the Boltzmann hierarchy ODE function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# 1. SETUP: Parameters, Constants, and ODE Functions
# =============================================================================

print("--- Setting up parameters and functions ---")

# Data folder
folder_path = './data/'

# Cosmological Parameters (Planck 2018)
# Using definitions from code(2) and code(3) for consistency.
# Note: code(1) was missing the h**2 scaling for OmegaM and OmegaK.
h = 0.5409
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
OmegaM, OmegaK = 0.483, -0.0438
OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
z_rec = 1089.411

# Working units: 8piG = c = hbar = 1, and s0 = 1 for numerical stability.
s0 = 1.0
H0 = np.sqrt(1 / (3 * OmegaLambda))
Hinf = H0 * np.sqrt(OmegaLambda)

# Tolerances and constants
atol = 1e-12
rtol = 1e-12
num_variables_boltzmann = 75
l_max = 69 # Derived from num_variables_boltzmann = 7 + (l_max - 2 + 1)
num_variables_perfect = 6

# Time constants
t0_integration = 1e-8 * s0
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
szero = -OmegaM/s0**3/(4*OmegaR/s0**4)
s_bang_init = smin1/t0_integration + szero
deltaeta = 6.6e-4 * s0
swaptime = 2.0 * s0

# Define background evolution to find key times
def ds_dt(t, s):
    s_abs = np.abs(s)
    return -H0 * np.sqrt(OmegaLambda + OmegaK * (s_abs**2 / s0**2) + OmegaM * (s_abs**3 / s0**3) + OmegaR * (s_abs**4 / s0**4))

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

# Find FCB time, recombination time, etc.
sol_bg = solve_ivp(ds_dt, [t0_integration, 12 * s0], [s_bang_init], events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
fcb_time = sol_bg.t_events[0][0]
endtime = fcb_time - deltaeta
s_rec = 1 + z_rec
recConformalTime = sol_bg.t[np.argmin(np.abs(sol_bg.y[0] - s_rec))]

print(f"FCB Time (eta_FCB): {fcb_time:.4f}")
print(f"Recombination Time (eta_rec): {recConformalTime:.4f}")
print(f"Integration Start Time (eta'): {endtime:.4f}")

# CORRECTED ODE system for full Boltzmann hierarchy (s-evolution)
def dX_boltzmann_s(t, X, k):
    s, phi, psi, dr, dm, vr, vm = X[0:7]
    fr_all = X[7:] # Holds F_2 to F_lmax

    sdot = ds_dt(t, s)
    H = sdot/s

    rho_m = 3 * (H0**2) * OmegaM * (np.abs(s / s0)**3)
    rho_r = 3 * (H0**2) * OmegaR * (np.abs(s / s0)**4)

    phidot = H * psi - ((4/3) * rho_r * vr + rho_m * vm) / (2 * s**2)
    fr2 = fr_all[0]
    fr3 = fr_all[1]
    fr2dot = -(8/15) * (k**2) * vr - (3/5) * k * fr3
    psidot = phidot - (3*H0**2*OmegaR/s0**4/k**2)*(2*s*sdot*fr2 + s**2*fr2dot)
    
    drdot = (4/3) * (3 * phidot + (k**2) * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(psi + dr / 4)
    vmdot = H * vm - psi
    
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    # Loop for F_l' where l = 3 to l_max-1 (i.e., 3 to 68)
    # This corresponds to j = l-2, so j runs from 1 to 66.
    for j in range(1, l_max - 2): # In Python range, this goes up to j=l_max-3=66.
        l = j + 2
        derivatives.append((k / (2 * l + 1)) * (l * fr_all[j-1] - (l + 1) * fr_all[j+1]))
        
    # Truncation for d(F_lmax)/dt, i.e., d(F_69)/dt
    # Free-streaming approx: dF_lmax/dt = k*lmax/(2*lmax+1) * F_{lmax-1}
    # F_{lmax-1} = F_68 is at fr_all index lmax-1-2 = 66
    last_deriv = (k * l_max / (2 * l_max + 1)) * fr_all[l_max - 3]
    derivatives.append(last_deriv)
    
    return derivatives

# CORRECTED ODE system for full Boltzmann hierarchy (sigma-evolution)
def dX_boltzmann_sigma(t, X, k):
    sigma, phi, psi, dr, dm, vr, vm = X[0:7]
    fr_all = X[7:]
    
    H = -(H0) * np.sqrt(OmegaLambda * np.exp(-2 * sigma) + OmegaK / s0**2 + OmegaM / s0**3 * np.exp(sigma) + OmegaR / s0**4 * np.exp(2 * sigma))
    
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (np.exp(3 * sigma))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (np.exp(4 * sigma))

    phidot = H * psi - ((4/3) * rho_r * vr + rho_m * vm) / (2 * np.exp(2 * sigma))
    fr2 = fr_all[0]
    fr3 = fr_all[1]
    fr2dot = -(8/15) * (k**2) * vr - (3/5) * k * fr3
    psidot = phidot - (3*H0**2*OmegaR/s0**4/k**2)*np.exp(2*sigma)*(2*H*fr2 + fr2dot)
    
    drdot = (4/3) * (3 * phidot + (k**2) * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(psi + dr / 4)
    vmdot = H * vm - psi
    
    derivatives = [H, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    # Loop for F_l' where l = 3 to l_max-1 (i.e., 3 to 68)
    for j in range(1, l_max - 2):
        l = j + 2
        derivatives.append((k / (2 * l + 1)) * (l * fr_all[j-1] - (l + 1) * fr_all[j+1]))

    # Truncation for d(F_lmax)/dt, i.e., d(F_69)/dt
    last_deriv = (k * l_max / (2 * l_max + 1)) * fr_all[l_max - 3]
    derivatives.append(last_deriv)
    
    return derivatives

# ODE system for perfect fluid (sigma-evolution)
def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    H = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma) + OmegaK/s0**2 + OmegaM/s0**3*np.exp(sigma) + OmegaR/s0**4*np.exp(2*sigma))
    
    rho_m = 3*(H0**2)*OmegaM/s0**3*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR/s0**4*(np.exp(4*sigma))
    
    phidot = H*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4)
    vmdot = H*vm - phi
    return [H, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 2. DATA LOADING AND INTERPOLATION
# =============================================================================

print("\n--- Loading and interpolating pre-computed data ---")

try:
    kvalues = np.load(folder_path + 'L70_kvalues.npy')
    ABCmatrices = np.load(folder_path + 'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path + 'L70_DEFmatrices.npy')
    X1matrices = np.load(folder_path + 'L70_X1matrices.npy')
    X2matrices = np.load(folder_path + 'L70_X2matrices.npy')
    recValues = np.load(folder_path + 'L70_recValues.npy')
    allowedK = np.load(folder_path + 'allowedK.npy')
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the 'data' directory.")
    exit()

# Extract sub-matrices
Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

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
num_modes_to_plot = 3

for i in range(num_modes_to_plot):
    k = allowedK[i]
    print(f"\nProcessing mode n={i+1} with k={k:.4f}")

    # --- a) Get interpolated matrices for this k ---
    A = get_A(k)
    D = get_D(k)
    X1 = get_X1(k)
    X2 = get_X2(k)
    recs_vec = get_recs(k)

    # --- b) Solve for X_inf ---
    M_matrix = (A @ X1 + D @ X2)[2:6, :]
    x_rec_subset = recs_vec[2:6]
    
    try:
        x_inf = np.linalg.solve(M_matrix, x_rec_subset)
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Skipping mode.")
        continue

    # --- c) Calculate initial conditions at eta' (endtime) ---
    x_prime = X1 @ x_inf
    y_prime_2_4 = X2 @ x_inf
    s_prime_val = np.interp(endtime, sol_bg.t, sol_bg.y[0])

    Y_prime = np.zeros(num_variables_boltzmann)
    Y_prime[0] = s_prime_val
    Y_prime[1:7] = x_prime
    Y_prime[7:9] = y_prime_2_4

    # --- d) Integrate backwards from endtime to recConformalTime ---
    sol_part1 = solve_ivp(dX_boltzmann_s, [endtime, swaptime], Y_prime, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    Y_swap = sol_part1.y[:, -1]
    Y_swap[0] = np.log(Y_swap[0])
    sol_part2 = solve_ivp(dX_boltzmann_sigma, [swaptime, recConformalTime], Y_swap, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    t_backward = np.concatenate((sol_part1.t, sol_part2.t))
    Y_backward_sigma = np.concatenate((sol_part1.y, sol_part2.y), axis=1)
    Y_backward = Y_backward_sigma.copy()
    mask = t_backward >= swaptime
    Y_backward[0, mask] = np.exp(Y_backward_sigma[0, mask])

    # --- e) Get solution from Big Bang to Recombination (perfect fluid) ---
    phi1, phi2 = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0, (1/60)*(-2*k**2+(9*OmegaM**2)/(16*OmegaLambda*OmegaR*s0**2))-2*OmegaK/(15*OmegaLambda*s0**2)
    dr1, dr2 = -OmegaM/(4*np.sqrt(3*OmegaR*OmegaLambda))/s0, (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(240*s0**2*OmegaR*OmegaLambda)-8*OmegaK/(15*OmegaLambda*s0**2)
    dm1, dm2 = -np.sqrt(3)*OmegaM/(16*s0*np.sqrt(OmegaR*OmegaLambda)), (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(320*s0**2*OmegaR*OmegaLambda)-2*OmegaK/(5*OmegaLambda*s0**2)
    vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-OmegaM**2+8*s0**2*OmegaR*OmegaLambda*k**2)/(160*s0**2*OmegaR*OmegaLambda)+4*OmegaK/(45*OmegaLambda*s0**2)
    vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-3*OmegaM**2+4*s0**2*OmegaR*OmegaLambda*k**2)/(480*s0**2*OmegaR*OmegaLambda)+17*OmegaK/(360*OmegaLambda*s0**2)
    
    sigma0 = np.log(s_bang_init)
    phi0, dr0, dm0 = 1+phi1*t0_integration+phi2*t0_integration**2, -2+dr1*t0_integration+dr2*t0_integration**2, -1.5+dm1*t0_integration+dm2*t0_integration**2
    vr0, vm0 = vr1*t0_integration+vr2*t0_integration**2+vr3*t0_integration**3, vm1*t0_integration+vm2*t0_integration**2+vm3*t0_integration**3
    
    Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0_integration, recConformalTime], Y0_perfect,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    # --- f) Stitch and Reflect ---
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1]))
    Y_perfect_full = np.zeros((num_variables_boltzmann, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
    Y_perfect_full[1,:] = sol_perfect.y[1,:]
    Y_perfect_full[2,:] = sol_perfect.y[1,:]
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]
    
    Y_left = np.concatenate((Y_perfect_full, Y_backward[:, ::-1]), axis=1)

    t_right = 2 * fcb_time - t_left[::-1]
    
    symm = np.ones(num_variables_boltzmann)
    symm[[5, 6]] = -1
    for l_idx in range(l_max - 1):
        l = l_idx + 2
        if l % 2 != 0: symm[7 + l_idx] = -1
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
# indices = [5, 3, 6, 4, 1, 2]
indices = [5]
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
    ax.set_ylim(-2.1, 2.2)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3)
axes[-1].set_xlabel(r'$\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Evolution of Perturbation Solutions for Allowed Modes', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig(folder_path + 'VR_solutions.pdf')