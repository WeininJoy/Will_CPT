# -*- coding: utf-8 -*-
"""
Debug script to investigate the discontinuity at recombination.

This script checks if the perfect fluid solution and Boltzmann hierarchy solution
properly match at the recombination boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# 1. SETUP: Parameters and Functions (from Metha's original code)
# =============================================================================

print("--- Setting up parameters and functions ---")
folder_path = './data_allowedK/'

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
num_variables = 75
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

# ODE systems
def dX_boltzmann_s(t, X, k):
    s,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = (sdot/s)*vm - psi
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    for j in range(8,num_variables-1):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    
    lastderiv = k*X[(num_variables-1)-1] - (((num_variables-1)-5 + 1)*X[(num_variables-1)])/t
    derivatives.append(lastderiv)
    return derivatives

def dX_boltzmann_sigma(t, X, k):
    sigma,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))
    
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    
    phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = (sigmadot)*vm - psi
    derivatives = [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    for j in range(8,num_variables-1):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    
    lastderiv = k*X[(num_variables-1)-1] - (((num_variables-1)-5 + 1)*X[(num_variables-1)])/t
    derivatives.append(lastderiv)
    return derivatives

def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))
    
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4) 
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 2. LOAD DATA
# =============================================================================

try:
    allowedK = np.load('allowedK.npy')
    ABCmatrices = np.load(folder_path+'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path+'L70_DEFmatrices.npy')
    GHIvectors = np.load(folder_path+'L70_GHIvectors.npy')
    X1matrices = np.load(folder_path+'L70_X1matrices.npy')
    X2matrices = np.load(folder_path+'L70_X2matrices.npy')
    recValues = np.load(folder_path+'L70_recValues.npy')
    print(f"Loaded exact transformation matrices for {len(allowedK)} allowed K values")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

# =============================================================================
# 3. DEBUG: Compare Solutions at Recombination
# =============================================================================

print("\n=== DEBUGGING RECOMBINATION BOUNDARY MATCHING ===")

k_index = 2  # Test first allowed mode
k = allowedK[k_index]
print(f"\nDebugging mode with k={k:.6f}")

# Get exact matrices for this k
A = Amatrices[k_index]
D = Dmatrices[k_index] 
X1 = X1matrices[k_index]
X2 = X2matrices[k_index]
recs_vec = recValues[k_index]

print(f"recs_vec from data_allowedK: {recs_vec}")

# Solve for X_inf
GX3 = np.zeros((6,4))
GX3[:,2] = GHIvectors[k_index][0:6]

M_matrix = (A @ X1 + D @ X2 + GX3)[2:6, :]
x_rec_subset = recs_vec[2:6]

x_inf = np.linalg.solve(M_matrix, x_rec_subset)
print(f"x_inf = {x_inf}")

# Calculate boundary conditions using transfer matrices
x_prime = X1 @ x_inf
y_prime_2_4 = X2 @ x_inf
s_prime_val = np.interp(endtime, sol.t, sol.y[0])

print(f"Transfer matrix boundary conditions at endtime:")
print(f"x_prime = {x_prime}")
print(f"y_prime_2_4 = {y_prime_2_4}")
print(f"s_prime_val = {s_prime_val}")

# Integrate backward from endtime to recombination (Boltzmann hierarchy)
Y_prime = np.zeros(num_variables)
Y_prime[0] = s_prime_val
Y_prime[1:7] = x_prime
Y_prime[7:9] = y_prime_2_4

print(f"\nBackward integration from endtime to recombination...")

# Phase 1: s-evolution (endtime to swaptime)
sol_back1 = solve_ivp(dX_boltzmann_s, [endtime, swaptime], Y_prime, 
                      dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

# Phase 2: sigma-evolution (swaptime to recombination)  
Y_swap = sol_back1.y[:, -1].copy()
Y_swap[0] = np.log(Y_swap[0])
sol_back2 = solve_ivp(dX_boltzmann_sigma, [swaptime, recConformalTime], Y_swap,
                      dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

# Values at recombination from backward integration
Y_rec_from_backward = sol_back2.y[:, -1].copy()
Y_rec_from_backward[0] = np.exp(Y_rec_from_backward[0])  # Convert sigma to s

print(f"Values at recombination from BACKWARD integration:")
print(f"s = {Y_rec_from_backward[0]:.6f}")
print(f"phi = {Y_rec_from_backward[1]:.6f}")  
print(f"psi = {Y_rec_from_backward[2]:.6f}")
print(f"dr = {Y_rec_from_backward[3]:.6f}")
print(f"dm = {Y_rec_from_backward[4]:.6f}")
print(f"vr = {Y_rec_from_backward[5]:.6f}")
print(f"vm = {Y_rec_from_backward[6]:.6f}")

# Now compare with perfect fluid solution integrated forward
print(f"\nPerfect fluid forward integration to recombination...")

# Perfect fluid initial conditions 
phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))

dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda)

dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda)

vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda)
vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda)

sigma0 = np.log(s0)
phi0 = 1 + phi1*t0 + phi2*t0**2
dr0 = -2 + dr1*t0 + dr2*t0**2
dm0 = -1.5 + dm1*t0 + dm2*t0**2
vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3

Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
sol_perfect = solve_ivp(dX_perfect_sigma, [t0, recConformalTime], Y0_perfect,
                        dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

# Values at recombination from forward perfect fluid integration
Y_rec_from_perfect = sol_perfect.y[:, -1]
s_rec_perfect = np.exp(Y_rec_from_perfect[0])

print(f"Values at recombination from PERFECT FLUID integration:")
print(f"s = {s_rec_perfect:.6f}")
print(f"phi = {Y_rec_from_perfect[1]:.6f}")  
print(f"psi = {Y_rec_from_perfect[1]:.6f} (=phi for perfect fluid)")
print(f"dr = {Y_rec_from_perfect[2]:.6f}")
print(f"dm = {Y_rec_from_perfect[3]:.6f}")
print(f"vr = {Y_rec_from_perfect[4]:.6f}")
print(f"vm = {Y_rec_from_perfect[5]:.6f}")

# Compare the values
print(f"\n=== COMPARISON AT RECOMBINATION ===")
print(f"s:   {s_rec_perfect:.6f} (perfect) vs {Y_rec_from_backward[0]:.6f} (backward)")
print(f"phi: {Y_rec_from_perfect[1]:.6f} (perfect) vs {Y_rec_from_backward[1]:.6f} (backward)")
print(f"psi: {Y_rec_from_perfect[1]:.6f} (perfect) vs {Y_rec_from_backward[2]:.6f} (backward)")
print(f"dr:  {Y_rec_from_perfect[2]:.6f} (perfect) vs {Y_rec_from_backward[3]:.6f} (backward)")
print(f"dm:  {Y_rec_from_perfect[3]:.6f} (perfect) vs {Y_rec_from_backward[4]:.6f} (backward)")
print(f"vr:  {Y_rec_from_perfect[4]:.6f} (perfect) vs {Y_rec_from_backward[5]:.6f} (backward)")
print(f"vm:  {Y_rec_from_perfect[5]:.6f} (perfect) vs {Y_rec_from_backward[6]:.6f} (backward)")

# Calculate relative differences
variables = ['s', 'phi', 'psi', 'dr', 'dm', 'vr', 'vm']
perfect_vals = [s_rec_perfect, Y_rec_from_perfect[1], Y_rec_from_perfect[1], 
                Y_rec_from_perfect[2], Y_rec_from_perfect[3], Y_rec_from_perfect[4], Y_rec_from_perfect[5]]
backward_vals = [Y_rec_from_backward[0], Y_rec_from_backward[1], Y_rec_from_backward[2],
                 Y_rec_from_backward[3], Y_rec_from_backward[4], Y_rec_from_backward[5], Y_rec_from_backward[6]]

print(f"\n=== RELATIVE DIFFERENCES ===")
for var, perfect, backward in zip(variables, perfect_vals, backward_vals):
    if abs(perfect) > 1e-15:
        rel_diff = abs(backward - perfect) / abs(perfect) * 100
        print(f"{var}: {rel_diff:.2e}% relative difference")
    else:
        abs_diff = abs(backward - perfect)
        print(f"{var}: {abs_diff:.2e} absolute difference (perfect value ~0)")

# Check what the stored recValues actually represent
print(f"\n=== STORED RECOMBINATION VALUES ===")
print(f"recs_vec (from data_allowedK): {recs_vec}")
print(f"Expected length: {len(recs_vec)} (should be 6 for [phi, psi, dr, dm, vr, vm])")

# Check if recs_vec matches either perfect fluid or backward integration values
print(f"\n=== COMPARING WITH STORED RECOMBINATION VALUES ===")
perfect_rec_6vars = [Y_rec_from_perfect[1], Y_rec_from_perfect[1], Y_rec_from_perfect[2], 
                     Y_rec_from_perfect[3], Y_rec_from_perfect[4], Y_rec_from_perfect[5]]
backward_rec_6vars = [Y_rec_from_backward[1], Y_rec_from_backward[2], Y_rec_from_backward[3],
                      Y_rec_from_backward[4], Y_rec_from_backward[5], Y_rec_from_backward[6]]

print("Stored vs Perfect fluid:")
for i, (stored, perfect) in enumerate(zip(recs_vec, perfect_rec_6vars)):
    diff = abs(stored - perfect)
    print(f"  Component {i}: {stored:.6f} (stored) vs {perfect:.6f} (perfect) -> diff = {diff:.2e}")

print("Stored vs Backward integration:")
for i, (stored, backward) in enumerate(zip(recs_vec, backward_rec_6vars)):
    diff = abs(stored - backward)
    print(f"  Component {i}: {stored:.6f} (stored) vs {backward:.6f} (backward) -> diff = {diff:.2e}")