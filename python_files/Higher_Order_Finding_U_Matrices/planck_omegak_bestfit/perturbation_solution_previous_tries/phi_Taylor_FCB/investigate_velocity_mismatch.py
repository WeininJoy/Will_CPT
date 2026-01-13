# -*- coding: utf-8 -*-
"""
Detailed investigation of velocity mismatch at recombination.
This script traces through the full calculation to identify where the discrepancy arises.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =============================================================================
# SETUP
# =============================================================================
folder_path = './'
folder_path_matrices = folder_path + 'data_allowedK/'
folder_path_timeseries = folder_path + 'data_allowedK_timeseries/'

lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
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

mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915

atol = 1e-13
rtol = 1e-13
num_variables = 75
H0 = 1/np.sqrt(3*OmegaLambda)

# Background
def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4

sol_a = solve_ivp(da_dt, [t0, 2], [a_Bang], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)

a_rec = 1./(1+z_rec)
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]

print("=" * 80)
print("INVESTIGATING VELOCITY MISMATCH AT RECOMBINATION")
print("=" * 80)
print(f"Recombination time: {recConformalTime}")
print(f"a_rec: {a_rec}")

# Load data
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

Amatrices = ABCmatrices[:, 0:6, :]
Bmatrices = ABCmatrices[:, 6:8, :]
Dmatrices = DEFmatrices[:, 0:6, :]
Ematrices = DEFmatrices[:, 6:8, :]

# Focus on first mode
k_index = 3
k = allowedK[k_index]

print(f"\n{'=' * 80}")
print(f"MODE: k={k:.6f} (index {k_index})")
print(f"{'=' * 80}")

# Get matrices
A = Amatrices[k_index]
D = Dmatrices[k_index]
X1 = X1matrices[k_index]
X2 = X2matrices[k_index]
recs_vec = recValues[k_index]

print(f"\nStored recombination values:")
print(f"  recs_vec = {recs_vec}")
print(f"  phi = {recs_vec[0]:.10f}")
print(f"  psi = {recs_vec[1]:.10f}")
print(f"  dr  = {recs_vec[2]:.10f}")
print(f"  dm  = {recs_vec[3]:.10f}")
print(f"  vr  = {recs_vec[4]:.10f}")
print(f"  vm  = {recs_vec[5]:.10f}")

# According to equation (23) in the paper:
# x* = A*x^∞ + B*y^∞
# But we want y^∞ = 0, so:
# x* = A*x^∞
# But the code uses equation (26):
# x* = (A*X1 + D*X2)*X^∞

print(f"\n{'=' * 80}")
print(f"STEP 1: Calculate x^∞ from equation (26)")
print(f"{'=' * 80}")

# The equation is: x* = (A*X1 + D*X2)*X^∞
# x* contains [phi, psi, dr, dm, vr, vm]
# But we only use [phi, dr, dm, vr, vm] (exclude psi at index 1)
# X^∞ contains [dr^∞, dm^∞, vr^∞, vm_dot^∞]

M_matrix = (A @ X1 + D @ X2)[[0,2,3,4,5], :]
x_rec = recs_vec[1:6]  # [phi, dr, dm, vr, vm] - excluding a

print(f"\nM_matrix shape: {M_matrix.shape}")
print(f"M_matrix = \n{M_matrix}")

print(f"\nx_rec (from stored recValues):")
print(f"  {x_rec}")

# Solve for X^∞
x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]
print(f"\nSolved x^∞:")
print(f"  x_inf = {x_inf}")

print(f"\n{'=' * 80}")
print(f"STEP 2: Verify reconstruction at recombination")
print(f"{'=' * 80}")

# Calculate x' and y' coefficients
x_prime_coeffs = X1 @ x_inf
y_prime_coeffs = X2 @ x_inf

print(f"\nx' coefficients (from X1 @ x_inf):")
print(f"  x_prime = {x_prime_coeffs}")

print(f"\ny' coefficients (from X2 @ x_inf):")
print(f"  y_prime = {y_prime_coeffs}")

# Now verify: does (A*X1 + D*X2)*x_inf give back x_rec?
x_reconstructed = M_matrix @ x_inf
print(f"\nReconstruction check (should equal x_rec):")
print(f"  M_matrix @ x_inf = {x_reconstructed}")
print(f"  x_rec            = {x_rec}")
print(f"  Difference       = {x_reconstructed - x_rec}")
print(f"  Max error        = {np.max(np.abs(x_reconstructed - x_rec))}")

print(f"\n{'=' * 80}")
print(f"STEP 3: Check values at recombination from time series")
print(f"{'=' * 80}")

# Get solutions at first time point (which should be recombination)
ABC_sols_k = all_ABC_solutions[k_index]
DEF_sols_k = all_DEF_solutions[k_index]

print(f"\nt_grid[0] = {t_grid[0]}")
print(f"recConformalTime = {recConformalTime}")
print(f"Difference = {t_grid[0] - recConformalTime}")

# Reconstruct at first time point
rec_idx = 0
Y_at_t0 = np.einsum('ij,j->i', ABC_sols_k[:, :, rec_idx], x_prime_coeffs) + \
          np.einsum('ij,j->i', DEF_sols_k[:, :, rec_idx], y_prime_coeffs)

print(f"\nReconstructed solution at t_grid[0]:")
print(f"  phi = {Y_at_t0[0]:.10f}")
print(f"  psi = {Y_at_t0[1]:.10f}")
print(f"  dr  = {Y_at_t0[2]:.10f}")
print(f"  dm  = {Y_at_t0[3]:.10f}")
print(f"  vr  = {Y_at_t0[4]:.10f}")
print(f"  vm  = {Y_at_t0[5]:.10f}")

print(f"\nComparison with stored recValues:")
print(f"  Δphi = {Y_at_t0[0] - recs_vec[0]:.6e}")
print(f"  Δdr  = {Y_at_t0[2] - recs_vec[2]:.6e}")
print(f"  Δdm  = {Y_at_t0[3] - recs_vec[3]:.6e}")
print(f"  Δvr  = {Y_at_t0[4] - recs_vec[4]:.6e}")
print(f"  Δvm  = {Y_at_t0[5] - recs_vec[5]:.6e}")

print(f"\n{'=' * 80}")
print(f"STEP 4: Check the transfer matrix equation directly")
print(f"{'=' * 80}")

# According to equation (24):
# [x*]   = [A D G] [x']
# [y*_2:4] = [B E H] [y']
# [y*_4:]  = [C F I] [0]

# At recombination (first point in time series), we should have:
# x* = A*x' + D*y'

x_star_from_matrices = A @ x_prime_coeffs + D @ y_prime_coeffs

print(f"\nDirect calculation: x* = A @ x' + D @ y'")
print(f"  Result:")
for i, name in enumerate(['a', 'phi', 'psi', 'dr', 'dm', 'vr', 'vm']):
    if i < len(x_star_from_matrices):
        print(f"    {name:3s} = {x_star_from_matrices[i]:.10f}")

print(f"\nComparison with stored recValues:")
print(f"  Δphi = {x_star_from_matrices[0] - recs_vec[0]:.6e}")
print(f"  Δpsi = {x_star_from_matrices[1] - recs_vec[1]:.6e}")
print(f"  Δdr  = {x_star_from_matrices[2] - recs_vec[2]:.6e}")
print(f"  Δdm  = {x_star_from_matrices[3] - recs_vec[3]:.6e}")
print(f"  Δvr  = {x_star_from_matrices[4] - recs_vec[4]:.6e}")
print(f"  Δvm  = {x_star_from_matrices[5] - recs_vec[5]:.6e}")

print(f"\n{'=' * 80}")
print(f"CONCLUSION")
print(f"{'=' * 80}")

print(f"\nThe issue is that:")
print(f"  1. x_rec is extracted from recValues (excluding index 0 which is 'a')")
print(f"  2. We solve (A*X1 + D*X2)*x_inf = x_rec to get x_inf")
print(f"  3. This reconstruction gives back x_rec perfectly")
print(f"  4. BUT when we use A @ x' + D @ y' at recombination,")
print(f"     it does NOT match the stored recValues!")
print(f"\nThis suggests the matrices A, D, X1, X2 may not be consistent")
print(f"with the stored recValues, OR there's an indexing issue.")
