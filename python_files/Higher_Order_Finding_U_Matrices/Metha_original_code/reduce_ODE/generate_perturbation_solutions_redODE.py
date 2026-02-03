# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model using Metha's original code ODEs and initial conditions.

It uses pre-computed data from transfer matrix calculations to solve for the
boundary conditions at the Future Conformal Boundary (FCB) for each allowed wavenumber k.
It then integrates the perturbation equations to obtain the full evolution of the
variables and plots the results.

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
folder_path = '../data_allowedK/'

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
num_variables = 75 - 2 # without phi and psi
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

# ODE system for full Boltzmann hierarchy (s-evolution)
def dX_boltzmann_s(t, X, k):
    s,dr,dm,vr,vm,fr2 = X[0:6]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))
    phi = -3*H0**2/(2*k**2) * ( OmegaM*abs(s)* (dm+3*sdot/s * vm) + OmegaR*s**2 * ( dr+4*sdot/s * vr) )
    psi = phi - 3*H0**2 * OmegaR/ (k**2) * s**2 * fr2

    rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[6]
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = (sdot/s)*vm - psi
    derivatives = [sdot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    # For l>2 terms (F_3 to F_{lmax-1})
    for j in range(6, num_variables-1):  # Start from index 6 (F_3) in reduced array
        l = j - 3  # l = 3 when j = 6
        if j < len(X) - 1:  # Safety check
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    
    # Final term for F_lmax
    if len(X) >= num_variables:
        lastderiv = k*X[num_variables-2] - (((num_variables-6)+1)*X[num_variables-1])/t
        derivatives.append(lastderiv)
    return derivatives

# ODE system for full Boltzmann hierarchy (sigma-evolution)  
def dX_boltzmann_sigma(t, X, k):
    sigma,dr,dm,vr,vm,fr2 = X[0:6]
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))
    phi = -3*H0**2/(2*k**2) * ( OmegaM*np.exp(sigma)* (dm+3*sigmadot * vm) + OmegaR*np.exp(2*sigma) * ( dr+4*sigmadot * vr) )
    psi = phi - 3*H0**2 * OmegaR/ (k**2) * np.exp(2*sigma) * fr2
    
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    
    phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[6]
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = (sigmadot)*vm - psi
    derivatives = [sigmadot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    # For l>2 terms (F_3 to F_{lmax-1})
    for j in range(6, num_variables-1):  # Start from index 6 (F_3) in reduced array
        l = j - 3  # l = 3 when j = 6
        if j < len(X) - 1:  # Safety check
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    
    # Final term for F_lmax
    if len(X) >= num_variables:
        lastderiv = k*X[num_variables-2] - (((num_variables-6)+1)*X[num_variables-1])/t
        derivatives.append(lastderiv)
    return derivatives

# ODE system for perfect fluid (sigma-evolution)
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
# 2. DATA LOADING AND INTERPOLATION
# =============================================================================

print("\n--- Loading and interpolating pre-computed data ---")

try:
    # Load exact transformation matrices for allowed K values only
    allowedK = np.load('../allowedK.npy')
    ABCmatrices = np.load(folder_path+'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path+'L70_DEFmatrices.npy')
    GHIvectors = np.load(folder_path+'L70_GHIvectors.npy')
    X1matrices = np.load(folder_path+'L70_X1matrices.npy')
    X2matrices = np.load(folder_path+'L70_X2matrices.npy')
    recValues = np.load(folder_path+'L70_recValues.npy')
    print(f"Loaded exact transformation matrices for {len(allowedK)} allowed K values")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the data_allowedK directory.")
    exit()

# Extract exact matrices (no interpolation needed)
Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

print(f"Using exact transformation matrices for {len(allowedK)} modes")
print(f"Matrix shapes: ABC={ABCmatrices.shape}, DEF={DEFmatrices.shape}")
print(f"X1={X1matrices.shape}, X2={X2matrices.shape}")


# =============================================================================
# 3. CALCULATE AND INTEGRATE SOLUTIONS
# =============================================================================

print("\n--- Calculating solutions for the first 3 allowed modes ---")

solutions = []
num_modes_to_plot = min(3, len(allowedK))

for i in range(num_modes_to_plot):
    k_index = i + 2  # Index in the allowedK array
    k = allowedK[k_index]
    print(f"\nProcessing mode n={i+1} with k={k:.6f} (index {k_index})")

    # --- a) Get exact matrices for this k (no interpolation) ---
    A = Amatrices[k_index]
    D = Dmatrices[k_index] 
    X1 = X1matrices[k_index]
    X2 = X2matrices[k_index]
    recs_vec = recValues[k_index]

    # --- b) Solve for X_inf ---
    GX3 = np.zeros((6,4))
    # Use exact GHI vector (no interpolation)
    GX3[:,2] = GHIvectors[k_index][0:6]
    
    M_matrix = (A @ X1 + D @ X2 + GX3)[2:6, :]
    x_rec_subset = recs_vec[2:6]
    
    try:
        x_inf = np.linalg.solve(M_matrix, x_rec_subset)
        print(f"Solved for x_inf: {x_inf}")
        # x_inf[2] = 0 # vr^\inf=0 for allowed k
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Skipping mode.")
        continue

    # --- c) Calculate initial conditions at eta' (endtime) ---
    x_prime = X1 @ x_inf
    y_prime_2_4 = X2 @ x_inf
    s_prime_val = np.interp(endtime, sol.t, sol.y[0])

    Y_prime = np.zeros(num_variables)
    Y_prime[0] = s_prime_val
    Y_prime[1:5] = x_prime[2:6]  # only dr, dm, vr, vm
    Y_prime[5:7] = y_prime_2_4   # fr2, fr3
    
    # --- d) Integrate backwards from endtime to recConformalTime ---
    sol_part1 = solve_ivp(dX_boltzmann_s, [endtime, swaptime], Y_prime, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    print("Finish integrating backwards till swaptime")

    Y_swap = sol_part1.y[:, -1]
    Y_swap[0] = np.log(Y_swap[0])
    sol_part2 = solve_ivp(dX_boltzmann_sigma, [swaptime, recConformalTime], Y_swap, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    t_backward = np.concatenate((sol_part1.t, sol_part2.t))
    Y_backward_sigma = np.concatenate((sol_part1.y, sol_part2.y), axis=1)
    Y_backward = Y_backward_sigma.copy()
    mask = t_backward >= swaptime
    Y_backward[0, mask] = np.exp(Y_backward_sigma[0, mask])

    print("Finish integrating backwards till recombination time")

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
    phi0 = 1 + phi1*t0 + phi2*t0**2
    dr0 = -2 + dr1*t0 + dr2*t0**2
    dm0 = -1.5 + dm1*t0 + dm2*t0**2
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3
    
    Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0, recConformalTime], Y0_perfect,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    print("Finish integrating from t0 to recConformalTime")

    # --- f) Stitch and Reflect ---
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1]))

    Y_perfect_full = np.zeros((num_variables, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
    Y_perfect_full[1,:] = sol_perfect.y[1,:]
    Y_perfect_full[2,:] = sol_perfect.y[1,:]
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]

    Y_left = np.concatenate((Y_perfect_full, Y_backward[:, ::-1]), axis=1)

    t_right = 2 * fcb_time - t_left[::-1]
    
    # Symmetry matrix for reflection
    symm = np.ones(num_variables)
    symm[[3, 4]] = -1  # vr and vm are antisymmetric (indices 3,4 in reduced array)
    for l_idx in range(num_variables - 6):  # Adjusted for reduced system
        l = l_idx + 3
        if l % 2 != 0: symm[6 + l_idx] = -1  # Adjusted indices
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

labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$']
indices = [3, 1, 4, 2]
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
    ax.set_ylim(-10, 10)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3)
axes[-1].set_xlabel(r'$\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Perturbation Solutions using Exact Transformation Matrices (v2)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('perturbation_solutions_redODE.pdf')
plt.show()