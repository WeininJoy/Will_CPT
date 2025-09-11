# -*- coding: utf-8 -*-
"""
Verification Script: Transfer Matrix vs ODE Backward Integration

This script checks whether x* and y* calculated from x' y' using the 
transfer matrices ABCDEF are the same as those obtained by integrating 
backward from x' y' using the ODEs.

This is a crucial consistency check for the transfer matrix method.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =============================================================================
# 1. SETUP: Parameters and Functions
# =============================================================================

print("--- Setting up parameters and functions ---")
folder_path = './data/'


## --- Best-fit parameters for nu_spacing =4 ---
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
    s0 = 1/a0
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return s0, Omega_lambda, Omega_m, Omega_K

# Best-fit parameters from nu_spacing=4
mt, kt, Omegab_ratio, h, A_s, n_s, tau_reio = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583, 1.9375648884116028, 0.9787493821596979, 0.019760560255556746
s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1065.0  # the actual value still needs to be checked
###########################

# Working units: 8piG = c = hbar = 1, and s0 = 1 for numerical stability.
s0 = 1.0
H0 = np.sqrt(1 / (3 * OmegaLambda))
Hinf = H0 * np.sqrt(OmegaLambda)

# Tolerances and constants
atol = 1e-13
rtol = 1e-13
stol = 1e-10 * s0
num_variables_boltzmann = num_variables = 75
l_max = 69 # Derived from num_variables_boltzmann = 7 + (l_max - 2 + 1)
num_variables_perfect = 6

# Time constants
t0_integration = 1e-8 * s0
deltaeta = 6.6e-4 * s0
swaptime = 2*s0

# Background evolution
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8

# Initial conditions for background
# Big Bang initial condition of s
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
szero = - OmegaM/s0**3/(4*OmegaR/s0**4)
s1 = (OmegaM)**2/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = - (OmegaM**3)/(192*s0*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*s0*OmegaLambda*OmegaR) 
s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2

# Background integration
print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12* s0], [s_bang], max_step = 0.25e-4 * s0, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Find FCB and recombination times
idxfcb = np.where(np.diff(np.sign(sol.y[0])) != 0)[0]
fcb_time = 0.5*(sol.t[idxfcb[0]] + sol.t[idxfcb[0] + 1])
endtime = fcb_time - deltaeta
s_rec = 1+z_rec
recScaleFactorDifference = abs(sol.y[0] - s_rec)
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

print(f"FCB Time (eta_FCB): {fcb_time:.4f}")
print(f"Recombination Time (eta_rec): {recConformalTime:.4f}")

def dX1_dt(t, X):
    """
    Evolves the perfect fluid perturbation equations using the scale factor 'a'
    as the background variable for improved numerical stability near the Big Bang.

    State vector X: [a, phi, dr, dm, vr, vm]
    """
    a, phi, dr, dm, vr, vm = X
    
    # Calculate the physical conformal Hubble parameter H_phys = adot/a
    # We use abs(a) in the OmegaM term for robustness if a ever becomes tiny and negative due to numerical error.
    H_phys = H0 * np.sqrt(OmegaLambda * a**2 + (OmegaK / s0**2) + (OmegaM / (s0**3 * abs(a))) + (OmegaR / (s0**4 * a**2)))
    adot = a * H_phys

    # Calculate background densities using 'a'
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (a**(-3))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (a**(-4))

    # Evolution equations
    # Note: H from the original code is -H_phys
    phidot = -H_phys * phi - 0.5 * a**2 * ((4/3) * rho_r * vr + rho_m * vm)
    
    drdot = (4/3) * (3 * phidot + k**2 * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(phi + dr / 4) 
    vmdot = -H_phys * vm - phi
    
    return [adot, phidot, drdot, dmdot, vrdot, vmdot]


def dX2_dt(t,X):
    #print(t);
    s,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2/s0**2)))+ OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(abs(s/s0)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s/s0)**4)
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR/s0**4*s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK/s0**2*H0**2/k**2)*fr2/2
    vmdot = (sdot/s)*vm - psi
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    #for l>2 terms, add derivates to above list
    for j in range(8,num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    #now add final term
    """
    lmax = num_variables - 5;
    lastderiv = (k*lmax*X[num_variables-1])/(2*lmax + 1);
    """
    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    
    derivatives.append(lastderiv)
    return derivatives

def dX3_dt(t, X):
    """
    Evolves the full Boltzmann hierarchy using the scale factor 'a'
    as the background variable for improved numerical stability.

    State vector X: [a, phi, psi, dr, dm, vr, vm, F_2, F_3, ..., F_lmax]
    """
    # Unpack state vector
    a, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
    
    # Calculate the physical conformal Hubble parameter H_phys = adot/a
    H_phys = H0 * np.sqrt(OmegaLambda * a**2 + (OmegaK / s0**2) + (OmegaM / (s0**3 * abs(a))) + (OmegaR / (s0**4 * a**2)))
    adot = a * H_phys

    # Calculate background densities using 'a'
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (a**(-3))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (a**(-4))
    
    # --- Evolution Equations ---
    # Note: H from the original code is -H_phys
    phidot = -H_phys * psi - 0.5 * a**2 * ((4/3) * rho_r * vr + rho_m * vm)
    
    fr2dot = -(8/15) * (k**2) * vr - (3/5) * k * X[8]
    
    psidot = phidot - (3 * H0**2 * OmegaR / (s0**4 * k**2)) * a**(-2) * (-2 * H_phys * fr2 + fr2dot)
    
    drdot = (4/3) * (3 * phidot + k**2 * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(psi + dr / 4) + (1 + 3 * OmegaK / s0**2 * H0**2 / k**2) * fr2 / 2
    vmdot = -H_phys * vm - psi
    
    # --- Assemble derivative vector ---
    derivatives = [adot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    #for l>2 terms, add derivates to above list
    for j in range(8,num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    #now add final term
    """
    lmax = num_variables - 5;
    lastderiv = (k*lmax*X[num_variables-1])/(2*lmax + 1);
    """
    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    
    derivatives.append(lastderiv)
    return derivatives


# =============================================================================
# 2. LOAD TRANSFORMATION MATRICES
# =============================================================================

try:
    allowedK = np.load(folder_path+'allowedK.npy')
    ABCmatrices = np.load(folder_path+'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path+'L70_DEFmatrices.npy')
    # GHIvectors = np.load(folder_path+'L70_GHIvectors.npy')
    X1matrices = np.load(folder_path+'L70_X1matrices.npy')
    X2matrices = np.load(folder_path+'L70_X2matrices.npy')
    recValues = np.load(folder_path+'L70_recValues.npy')
    print(f"Loaded exact transformation matrices for {len(allowedK)} allowed K values")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Extract all matrix components
Amatrices = ABCmatrices[:, 0:6, :]
Bmatrices = ABCmatrices[:, 6:8, :]
Cmatrices = ABCmatrices[:, 8:num_variables, :]
Dmatrices = DEFmatrices[:, 0:6, :]
Ematrices = DEFmatrices[:, 6:8, :]
Fmatrices = DEFmatrices[:, 8:num_variables, :]

print(f"Matrix shapes: A={Amatrices.shape}, B={Bmatrices.shape}, C={Cmatrices.shape}")
print(f"               D={Dmatrices.shape}, E={Ematrices.shape}, F={Fmatrices.shape}")

# =============================================================================
# 3. VERIFICATION FOR EACH ALLOWED MODE
# =============================================================================

print("\n=== VERIFICATION: Transfer Matrix vs ODE Integration ===")

k_index = 4
k = allowedK[k_index]
print(f"\nTesting mode with k={k:.6f} (index {k_index})")

# Get exact matrices for this k
A = Amatrices[k_index]
B = Bmatrices[k_index] 
C = Cmatrices[k_index]
D = Dmatrices[k_index]
E = Ematrices[k_index]
F = Fmatrices[k_index]
X1 = X1matrices[k_index]
X2 = X2matrices[k_index]
recs_vec = recValues[k_index]

print(f"Matrix shapes for k={k:.6f}:")
print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")
print(f"D: {D.shape}, E: {E.shape}, F: {F.shape}")
print(f"X1: {X1.shape}, X2: {X2.shape}")
print(f"Stored recValues: {recs_vec}")

# Calculate x^∞ 
# GX3 = np.zeros((6,4))
# GX3[:,2] = GHIvectors[k_index][0:6]

M_matrix = (A @ X1 + D @ X2)[2:6, :]  # Only use independent components [dr, dm, vr, vm]
x_rec = recs_vec[2:6]  # Only use independent components from stored values
# x_inf = np.linalg.solve(M_matrix, x_rec)
x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]
print(f"x^∞ = {x_inf}")

# Calculate x' and y' at endtime using X1 and X2 (equation 25)
x_prime = X1 @ x_inf
y_prime_2_4 = X2 @ x_inf
s_prime_val = np.interp(endtime, sol.t, sol.y[0])

print(f"x' (from X1 @ x^∞) = {x_prime}")
print(f"y'_{{2:4}} (from X2 @ x^∞) = {y_prime_2_4}")
print(f"s' = {s_prime_val}")

# =============================================================================
# 4. METHOD 1: Calculate x* y* using Transfer Matrices ABCDEF
# =============================================================================

print(f"\n--- METHOD 1: Transfer Matrix Calculation ---")

# According to the transfer matrix method:
# The full solution vector at recombination should be:
# [x*, y*] = [A, B; D, E; C, F] @ [X1; X2] @ x^∞ + [G; H; I] @ x3

# Calculate x* using A matrix: x* = A @ X1 @ x^∞ + D @ X2 @ x^∞ + G @ x3
# x_star_transfer = A @ X1 @ x_inf + D @ X2 @ x_inf + GX3[:, 2]
x_star_transfer = A @ x_prime + D @ y_prime_2_4 

# Calculate y*_{2:4} using B matrix: y*_{2:4} = B @ X1 @ x^∞ + E @ X2 @ x^∞ + H @ x3
# HX3 = np.zeros(2)
# HX3 = GHIvectors[k_index][6:8]  # H vector
# y_star_2_4_transfer = B @ X1 @ x_inf + E @ X2 @ x_inf + HX3
y_star_2_4_transfer = B @ x_prime + E @ y_prime_2_4 

# Calculate y*_{4:} using C matrix: y*_{4:} = C @ X1 @ x^∞ + F @ X2 @ x^∞ + I @ x3
# IX3 = GHIvectors[k_index][8:num_variables]  # I vector
# y_star_4plus_transfer = C @ X1 @ x_inf + F @ X2 @ x_inf + IX3
y_star_4plus_transfer = C @ x_prime + F @ y_prime_2_4

print(f"x* (transfer matrix) = {x_star_transfer}")
print(f"y*_{{2:4}} (transfer matrix) = {y_star_2_4_transfer}")
print(f"y*_{{4:}} length = {len(y_star_4plus_transfer)}")

# =============================================================================
# 5. METHOD 2: Calculate x* y* using ODE Backward Integration
# =============================================================================

print(f"\n--- METHOD 2: ODE Backward Integration ---")

# Start from x' y' at endtime and integrate backward to recombination
Y_endtime = np.zeros(num_variables+1) # +1 for s variable
Y_endtime[0] = s_prime_val
Y_endtime[1:7] = x_prime  # [phi, psi, dr, dm, vr, vm]
Y_endtime[7:9] = y_prime_2_4  # [F_2, F_3]
# Higher order terms remain zero

print(f"Initial conditions at endtime (eta'):")
print(f"s' = {Y_endtime[0]:.8f}")
print(f"x' = {Y_endtime[1:7]}")
print(f"y'_{{2:4}} = {Y_endtime[7:9]}")

# Phase 1: Backward integration from endtime to swaptime (s-evolution)
sol_back1 = solve_ivp(dX2_dt, [endtime, swaptime], Y_endtime, 
                      dense_output=True, method='LSODA', atol=atol, rtol=rtol)

print(f"Phase 1 (endtime -> swaptime): {len(sol_back1.t)} points")

# Phase 2: Backward integration from swaptime to recombination (sigma-evolution)
Y_swap = sol_back1.y[:, -1].copy()
Y_swap[0] = 1./ Y_swap[0]  # Convert s to a

sol_back2 = solve_ivp(dX3_dt, [swaptime, recConformalTime], Y_swap,
                      dense_output=True, method='LSODA', atol=atol, rtol=rtol)

print(f"Phase 2 (swaptime -> recombination): {len(sol_back2.t)} points")

# Extract x* and y* from backward integration at recombination
Y_rec_ode = sol_back2.y[:, -1].copy()
Y_rec_ode[0] = np.exp(Y_rec_ode[0])  # Convert sigma back to s

x_star_ode = Y_rec_ode[1:7]  # [phi, psi, dr, dm, vr, vm]
y_star_2_4_ode = Y_rec_ode[7:9]  # [F_2, F_3]
y_star_4plus_ode = Y_rec_ode[9:]  # [F_4, F_5, ..., F_69]

print(f"x* (ODE integration) = {x_star_ode}")
print(f"y*_{{2:4}} (ODE integration) = {y_star_2_4_ode}")
print(f"y*_{{4:}} length = {len(y_star_4plus_ode)}")

# # =============================================================================
# # 6. METHOD 3: Calculate x* y* using ODE Forward Integration
# # =============================================================================

# # Now compare with perfect fluid solution integrated forward
# print(f"\nPerfect fluid forward integration to recombination...")

# # Perfect fluid initial conditions 
# phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
# phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))

# dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
# dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda)

# dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
# dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda)

# vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda)
# vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda)

# sigma0 = np.log(s0)
# phi0 = 1 + phi1*t0 + phi2*t0**2
# dr0 = -2 + dr1*t0 + dr2*t0**2
# dm0 = -1.5 + dm1*t0 + dm2*t0**2
# vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
# vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3

# Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
# sol_perfect = solve_ivp(dX_perfect_sigma, [t0, recConformalTime], Y0_perfect,
#                         dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

# # Values at recombination from forward perfect fluid integration
# Y_rec_from_perfect = sol_perfect.y[:, -1]
# s_rec_perfect = np.exp(Y_rec_from_perfect[0])

# print(f"Values at recombination from PERFECT FLUID integration:")
# print(f"s = {s_rec_perfect:.6f}")
# print(f"phi = {Y_rec_from_perfect[1]:.6f}")  
# print(f"psi = {Y_rec_from_perfect[1]:.6f} (=phi for perfect fluid)")
# print(f"dr = {Y_rec_from_perfect[2]:.6f}")
# print(f"dm = {Y_rec_from_perfect[3]:.6f}")
# print(f"vr = {Y_rec_from_perfect[4]:.6f}")
# print(f"vm = {Y_rec_from_perfect[5]:.6f}")


# # =============================================================================
# # 7. COMPARISON
# # =============================================================================

# print(f"\n=== COMPARISON: Transfer Matrix vs ODE Integration ===")

# print(f"\nCOMPARING x* (6 components):")
# for i, (label, transfer, ode) in enumerate(zip(['phi', 'psi', 'dr', 'dm', 'vr', 'vm'], 
#                                                x_star_transfer, x_star_ode)):
#     abs_diff = abs(transfer - ode)
#     if abs(ode) > 1e-15:
#         rel_diff = abs_diff / abs(ode) * 100
#         print(f"  {label}: {transfer:.8f} (transfer) vs {ode:.8f} (ODE) -> {rel_diff:.2e}% rel diff")
#     else:
#         print(f"  {label}: {transfer:.8f} (transfer) vs {ode:.8f} (ODE) -> {abs_diff:.2e} abs diff")

# print(f"\nCOMPARING y*_{{2:4}} (2 components):")
# for i, (transfer, ode) in enumerate(zip(y_star_2_4_transfer, y_star_2_4_ode)):
#     abs_diff = abs(transfer - ode)
#     if abs(ode) > 1e-15:
#         rel_diff = abs_diff / abs(ode) * 100
#         print(f"  F_{i+2}: {transfer:.8f} (transfer) vs {ode:.8f} (ODE) -> {rel_diff:.2e}% rel diff")
#     else:
#         print(f"  F_{i+2}: {transfer:.8f} (transfer) vs {ode:.8f} (ODE) -> {abs_diff:.2e} abs diff")

# print(f"\nCOMPARING y*_{{4:}} (first 5 components of {len(y_star_4plus_transfer)}):")
# for i in range(min(5, len(y_star_4plus_transfer))):
#     transfer = y_star_4plus_transfer[i]
#     ode = y_star_4plus_ode[i]
#     abs_diff = abs(transfer - ode)
#     if abs(ode) > 1e-15:
#         rel_diff = abs_diff / abs(ode) * 100
#         print(f"  F_{i+4}: {transfer:.8f} (transfer) vs {ode:.8f} (ODE) -> {rel_diff:.2e}% rel diff")
#     else:
#         print(f"  F_{i+4}: {transfer:.8f} (transfer) vs {ode:.8f} (ODE) -> {abs_diff:.2e} abs diff")

# # Statistical summary
# all_x_diffs = [abs(t - o) for t, o in zip(x_star_transfer, x_star_ode)]
# all_y24_diffs = [abs(t - o) for t, o in zip(y_star_2_4_transfer, y_star_2_4_ode)]
# all_y4plus_diffs = [abs(t - o) for t, o in zip(y_star_4plus_transfer, y_star_4plus_ode)]

# print(f"\n=== STATISTICAL SUMMARY ===")
# print(f"x* differences: max={max(all_x_diffs):.2e}, mean={np.mean(all_x_diffs):.2e}")
# print(f"y*_{{2:4}} differences: max={max(all_y24_diffs):.2e}, mean={np.mean(all_y24_diffs):.2e}")
# print(f"y*_{{4:}} differences: max={max(all_y4plus_diffs):.2e}, mean={np.mean(all_y4plus_diffs):.2e}")

# # Check if they match within numerical tolerance
# tolerance = 1e-10
# x_match = all(diff < tolerance for diff in all_x_diffs)
# y24_match = all(diff < tolerance for diff in all_y24_diffs)
# y4plus_match = all(diff < tolerance for diff in all_y4plus_diffs)

# print(f"\n=== CONSISTENCY CHECK (tolerance = {tolerance}) ===")
# print(f"x* matches: {x_match}")
# print(f"y*_{{2:4}} matches: {y24_match}")
# print(f"y*_{{4:}} matches: {y4plus_match}")
# print(f"Overall consistency: {x_match and y24_match and y4plus_match}")

# if not (x_match and y24_match and y4plus_match):
#     print(f"\n⚠️  WARNING: Transfer matrix and ODE integration give different results!")
#     print(f"This indicates a problem with either:")
#     print(f"1. The transfer matrix calculation")
#     print(f"2. The backward ODE integration")
#     print(f"3. The boundary conditions")
# else:
#     print(f"\n✅ SUCCESS: Transfer matrix and ODE integration are consistent!")

# # =============================================================================
# # 7. VERIFICATION WITH STORED RECOMBINATION VALUES
# # =============================================================================

# print(f"\n=== VERIFICATION WITH STORED RECOMBINATION VALUES ===")
# print(f"Stored recs_vec = {recs_vec}")
# print(f"Transfer matrix x* = {x_star_transfer}")
# print(f"ODE integration x* = {x_star_ode}")

# print(f"\nComparing with stored recombination values:")
# for i, (label, stored, transfer, ode) in enumerate(zip(['phi', 'psi', 'dr', 'dm', 'vr', 'vm'], 
#                                                        recs_vec, x_star_transfer, x_star_ode)):
#     print(f"  {label}: {stored:.8f} (stored) vs {transfer:.8f} (transfer) vs {ode:.8f} (ODE)")
#     print(f"    stored-transfer: {abs(stored-transfer):.2e}")
#     print(f"    stored-ODE: {abs(stored-ode):.2e}")