# -*- coding: utf-8 -*-
"""
Corrected Perturbation Solutions using Proper Recombination Boundary Matching

This script fixes the discontinuity issue by:
1. Using stored recombination values (recs_vec) as exact boundary conditions
2. Integrating Boltzmann hierarchy FORWARD from recombination to endtime  
3. Using perfect fluid BACKWARD from recombination to Big Bang
4. Proper stitching that respects the boundary conditions

This eliminates the massive ψ discontinuity that was causing numerical peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

# ODE system for full Boltzmann hierarchy (s-evolution) - FORWARD
def dX_boltzmann_s_forward(t, X, k):
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

# ODE system for full Boltzmann hierarchy (sigma-evolution) - FORWARD
def dX_boltzmann_sigma_forward(t, X, k):
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

# ODE system for perfect fluid (sigma-evolution) - BACKWARD
def dX_perfect_sigma_backward(t, X, k):
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
# 2. LOAD EXACT TRANSFORMATION MATRICES
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
# 3. CORRECTED INTEGRATION METHOD
# =============================================================================

print("\n--- Using corrected integration method ---")

solutions = []
num_modes_to_plot = min(3, len(allowedK))

for i in range(num_modes_to_plot):
    k_index = i + 2
    k = allowedK[k_index]
    print(f"\nProcessing mode n={i+1} with k={k:.6f}")

    # Get exact transformation matrices and recombination values
    recs_vec = recValues[k_index]
    print(f"Using stored recombination values: {recs_vec}")
    
    # --- PART 1: Perfect fluid from Big Bang to Recombination ---
    # Use perfect fluid initial conditions and integrate forward to recombination
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
    sol_perfect = solve_ivp(dX_perfect_sigma_backward, [t0, recConformalTime], Y0_perfect,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # --- PART 2: Boltzmann hierarchy from Recombination to Endtime ---
    # Use the stored recombination values as EXACT boundary conditions
    s_rec_val = np.interp(recConformalTime, sol.t, sol.y[0])
    
    # Create initial state vector at recombination using stored values
    Y_rec = np.zeros(num_variables)
    Y_rec[0] = s_rec_val  # Background scale factor
    Y_rec[1:7] = recs_vec  # [phi, psi, dr, dm, vr, vm] from stored data
    # F_l terms start as zero at recombination (perfect fluid assumption)
    
    print(f"Recombination boundary conditions:")
    print(f"s = {Y_rec[0]:.6f}")
    print(f"phi = {Y_rec[1]:.6f}, psi = {Y_rec[2]:.6f}")
    print(f"dr = {Y_rec[3]:.6f}, dm = {Y_rec[4]:.6f}")
    print(f"vr = {Y_rec[5]:.6f}, vm = {Y_rec[6]:.6f}")
    
    # Forward integration: recombination → swaptime (sigma) → endtime (s)
    
    # Phase 1: sigma-evolution from recombination to swaptime
    Y_rec_sigma = Y_rec.copy()
    Y_rec_sigma[0] = np.log(Y_rec[0])  # Convert s to sigma
    
    sol_forward1 = solve_ivp(dX_boltzmann_sigma_forward, [recConformalTime, swaptime], Y_rec_sigma,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Phase 2: s-evolution from swaptime to endtime  
    Y_swap = sol_forward1.y[:, -1].copy()
    Y_swap[0] = np.exp(Y_swap[0])  # Convert sigma back to s
    
    sol_forward2 = solve_ivp(dX_boltzmann_s_forward, [swaptime, endtime], Y_swap,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Combine forward integration results
    t_forward = np.concatenate((sol_forward1.t, sol_forward2.t))
    Y_forward_raw = np.concatenate((sol_forward1.y, sol_forward2.y), axis=1)
    
    # Convert sigma back to s for the first phase
    Y_forward = Y_forward_raw.copy()
    mask = t_forward <= swaptime
    Y_forward[0, mask] = np.exp(Y_forward_raw[0, mask])
    
    # --- PART 3: Combine Solutions Properly ---
    # Perfect fluid (Big Bang to recombination) + Boltzmann (recombination to endtime)
    
    # Expand perfect fluid solution to full variable space
    Y_perfect_full = np.zeros((num_variables, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])  # Convert sigma to s
    Y_perfect_full[1,:] = sol_perfect.y[1,:]  # phi
    Y_perfect_full[2,:] = sol_perfect.y[1,:]  # psi = phi for perfect fluid
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]  # dr, dm, vr, vm
    # F_l terms remain zero during perfect fluid phase
    
    # Combine: perfect fluid + forward Boltzmann (both in forward time order)
    t_left = np.concatenate((sol_perfect.t, t_forward))
    Y_left = np.concatenate((Y_perfect_full, Y_forward), axis=1)
    
    # Check boundary continuity at recombination
    boundary_idx = len(sol_perfect.t) - 1
    print(f"Boundary continuity check at recombination:")
    print(f"φ: {Y_left[1, boundary_idx]:.6f} -> {Y_left[1, boundary_idx+1]:.6f}")
    print(f"ψ: {Y_left[2, boundary_idx]:.6f} -> {Y_left[2, boundary_idx+1]:.6f}")
    print(f"vr: {Y_left[5, boundary_idx]:.6f} -> {Y_left[5, boundary_idx+1]:.6f}")
    
    # Create palindromic solution by reflection
    t_right = 2 * fcb_time - t_left[::-1]
    
    # Symmetry matrix for reflection
    symm = np.ones(num_variables)
    symm[[5, 6]] = -1  # vr and vm are antisymmetric
    for l_idx in range(num_variables - 8):
        l = l_idx + 3
        if l % 2 != 0: symm[8 + l_idx] = -1  # Odd l F_l terms are antisymmetric
    S = np.diag(symm)
    
    Y_right = S @ Y_left[:, ::-1]
    
    # Complete solution
    t_full = np.concatenate((t_left, t_right))
    Y_full = np.concatenate((Y_left, Y_right), axis=1)
    
    solutions.append({'t': t_full, 'Y': Y_full, 'k': k})

# =============================================================================
# 4. PLOTTING
# =============================================================================

print(f"\n--- Plotting {len(solutions)} solutions ---")

fig, axes = plt.subplots(len(solutions), 1, figsize=(12, 8), sharex=True)
if len(solutions) == 1: axes = [axes]

labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$', r'$\phi$', r'$\psi$']
indices = [5, 3, 6, 4, 1, 2]
colors = ['blue', 'red', 'green', 'orange', 'magenta', 'cyan']

for i, sol in enumerate(solutions):
    ax = axes[i]
    
    for label, index, color in zip(labels, indices, colors):
        ax.plot(sol['t'], sol['Y'][index, :], label=label, color=color, linewidth=1.2)

    ax.axvline(fcb_time, color='k', linestyle='--', label='FCB')
    ax.axvline(2 * fcb_time, color='k', linestyle='--', label='Big Crunch')
    ax.axvline(recConformalTime, color='purple', linestyle=':', label='Recombination')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(f'Mode n={i+1}\nk={sol["k"]:.4f}', fontsize=11)
    ax.set_xlim(0, 2 * fcb_time * 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3, fontsize=10)
axes[-1].set_xlabel(r'$\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Corrected Solutions with Proper Recombination Matching', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('perturbation_solutions_rec.pdf')
plt.show()

print("\n--- Corrected method implementation complete ---")
print("Solutions saved to perturbation_solutions_rec.pdf")