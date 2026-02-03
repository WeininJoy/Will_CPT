# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021
Modified to save the full time evolution of the transfer matrix basis vectors.

This script pre-calculates the time-dependent transfer matrices U(t, t_prime)
by integrating basis vectors from a point near the FCB (t_prime = endtime)
back to recombination and saving the full solution history for each.

@author: MRose (Modified by AI)
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# --- Basic Setup (Unchanged) ---
# working in units 8piG = Lambda = c = hbar = kB = 1 throughout
folder_path = './data_allowedK_timeseries/'
import os
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# set cosmological parameters from Planck baseline
OmegaLambda = 0.679
OmegaM = 0.321
OmegaR = 9.24e-5
H0 = 1/np.sqrt(3*OmegaLambda) # we are working in units of Lambda=c=1

# set tolerances and constants
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 75 # number of pert variables
swaptime = 2.0 # set time when we swap from s to sigma
endtime = 6.15
deltaeta = 6.150659839680297 - endtime
Hinf = H0*np.sqrt(OmegaLambda)

# --- Background and Recombination Time (Unchanged) ---
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8
smin1 = np.sqrt(3*OmegaLambda/OmegaR)
szero = - OmegaM/(4*OmegaR)
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3))
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2)
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3))/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
s4 = -(OmegaM**5)/(9216*(OmegaR**3)*(OmegaLambda**2))
s0_init = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4

print('Performing Initial Background Integration')
sol_bg = solve_ivp(ds_dt, [t0, 12], [s0_init], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

z_rec = 1090.30
s_rec = 1 + z_rec
recConformalTime = sol_bg.t[abs(sol_bg.y[0] - s_rec).argmin()]
print(f"Recombination Time: {recConformalTime:.4f}, Swap Time: {swaptime:.4f}, End Time: {endtime:.4f}")

# --- Derivative Functions (Unchanged, but ensure lmax is handled correctly) ---
l_max = num_variables - 7 + 2 # Assuming this is the logic
def dX2_dt(t, X, k): # s-evolution
    s,phi,psi,dr,dm,vr,vm = X[0:7]
    fr2 = X[7]
    sdot = ds_dt(t, s)
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
    #for l>2 terms, add derivates to above list
    for j in range(8,num_variables):
        l = j - 5;
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]));
    #now add final term
    """
    lmax = num_variables - 5;
    lastderiv = (k*lmax*X[num_variables-1])/(2*lmax + 1);
    """
    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t;
    
    derivatives.append(lastderiv);
    return derivatives

def dX3_dt(t, X, k): # sigma-evolution
    sigma,phi,psi,dr,dm,vr,vm = X[0:7]
    fr2 = X[7]
    sigmadot = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)+OmegaR*np.exp(2*sigma))
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    phidot = sigmadot*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = sigmadot*vm - psi
    derivatives = [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    #for l>2 terms, add derivates to above list
    for j in range(8,num_variables):
        l = j - 5;
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]));
    #now add final term
    """
    lmax = num_variables - 5;
    lastderiv = (k*lmax*X[num_variables-1])/(2*lmax + 1);
    """
    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t;
    
    derivatives.append(lastderiv);
    return derivatives

# --- Main Calculation Loop ---

kvalues = np.load('allowedK.npy')
# kvalues = kvalues[0:2] # For testing: only run for a few k-values

# *** NEW: Define a common time grid ***
num_time_points = 500
t_grid = np.linspace(recConformalTime, endtime, num=num_time_points)

# Split grid for the two integration domains
t_grid_sigma = t_grid[t_grid < swaptime]
t_grid_s = t_grid[t_grid >= swaptime]

# *** NEW: Initialize lists to hold the full solution tensors ***
all_ABC_solutions = []
all_DEF_solutions = []
all_GHI_solutions = []

# This is the starting value of s at `endtime`, used for all initial conditions
s_init = np.interp(endtime, sol_bg.t, sol_bg.y[0])

for i in range(len(kvalues)):
    k = kvalues[i]
    print(f"\nProcessing k = {k:.6f} ({i+1}/{len(kvalues)})")
    
    # *** NEW: Initialize tensors to store solutions for this k ***
    # Shape: (num_variables, num_basis_vectors, num_time_steps)
    ABC_sols_k = np.zeros((num_variables, 6, num_time_points))
    DEF_sols_k = np.zeros((num_variables, 2, num_time_points))
    GHI_sols_k = np.zeros((num_variables, num_time_points))

    # --- A: Calculate solutions for base variable basis vectors (forms ABC) ---
    for n in range(6):
        print(f"  Calculating basis vector {n+1}/6...")
        # Initial conditions at 'endtime' for this basis vector

        x0 = np.zeros(num_variables)
        x0[n] = 1
        inits_s = np.concatenate(([s_init], x0))
        
        # Integrate from endtime to swaptime (s-domain)
        sol_s = solve_ivp(dX2_dt, [endtime, swaptime], inits_s, dense_output=True,
                          method='LSODA', atol=atol, rtol=rtol, args=(k,))
        
        # Prepare for next integration leg
        inits_sigma = sol_s.y[:, -1]
        inits_sigma[0] = np.log(inits_sigma[0])
        
        # Integrate from swaptime to recombination (sigma-domain)
        sol_sigma = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits_sigma, dense_output=True,
                              method='LSODA', atol=atol, rtol=rtol, args=(k,))

        # *** NEW: Evaluate solutions on the grid and stitch them ***
        # Evaluate on the respective parts of the grid
        sol_on_s_grid = sol_s.sol(t_grid_s)
        sol_on_sigma_grid = sol_sigma.sol(t_grid_sigma)
        
        # Convert sigma back to s for consistency
        sol_on_sigma_grid[0, :] = np.exp(sol_on_sigma_grid[0, :])
        
        # Stitch the solutions together
        full_solution = np.concatenate((sol_on_sigma_grid, sol_on_s_grid), axis=1)
        
        # Store the perturbation part (ignoring the background variable)
        ABC_sols_k[:, n, :] = full_solution[1:, :]

    all_ABC_solutions.append(ABC_sols_k)

    # --- B: Calculate solutions for anisotropic basis vectors (forms DEF) ---
    for j in range(2):
        print(f"  Calculating anisotropic vector {j+1}/2...")
        x0 = np.zeros(num_variables)
        x0[j+7] = 1 # Initial conditions for F_2 and F_3
        inits_s = np.concatenate(([s_init], x0))
        
        sol_s = solve_ivp(dX2_dt, [endtime, swaptime], inits_s, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
        inits_sigma = sol_s.y[:, -1]
        inits_sigma[0] = np.log(inits_sigma[0])
        sol_sigma = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits_sigma, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

        sol_on_s_grid = sol_s.sol(t_grid_s)
        sol_on_sigma_grid = sol_sigma.sol(t_grid_sigma)
        sol_on_sigma_grid[0, :] = np.exp(sol_on_sigma_grid[0, :])
        full_solution = np.concatenate((sol_on_sigma_grid, sol_on_s_grid), axis=1)
        DEF_sols_k[:, j, :] = full_solution[1:, :]
        
    all_DEF_solutions.append(DEF_sols_k)

    # --- C: Calculate inhomogeneous solution part (forms GHI) ---
    print("  Calculating inhomogeneous vector...")
    x0 = np.zeros(num_variables)
    x3 = -(16/945) * (k**4) * (deltaeta**3)
    x0[9] = x3  # Inhomogeneous term from v_r^\infty
    inits_s = np.concatenate(([s_init], x0))

    sol_s = solve_ivp(dX2_dt, [endtime, swaptime], inits_s, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    inits_sigma = sol_s.y[:, -1]
    inits_sigma[0] = np.log(inits_sigma[0])
    sol_sigma = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits_sigma, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    sol_on_s_grid = sol_s.sol(t_grid_s)
    sol_on_sigma_grid = sol_sigma.sol(t_grid_sigma)
    sol_on_sigma_grid[0, :] = np.exp(sol_on_sigma_grid[0, :])
    full_solution = np.concatenate((sol_on_sigma_grid, sol_on_s_grid), axis=1)
    GHI_sols_k[:, :] = full_solution[1:, :]
    
    all_GHI_solutions.append(GHI_sols_k)


# --- Save the results ---
print("\nSaving all solution tensors to disk...")

# Convert lists of tensors to single large numpy arrays
# Final shape for ABC: (num_k, num_vars, 6, num_times)
# Final shape for DEF: (num_k, num_vars, 2, num_times)
# Final shape for GHI: (num_k, num_vars, num_times)
all_ABC_solutions = np.array(all_ABC_solutions)
all_DEF_solutions = np.array(all_DEF_solutions)
all_GHI_solutions = np.array(all_GHI_solutions)

# Save the time grid - this is essential!
np.save(folder_path + 't_grid.npy', t_grid)

# Save the solution tensors
np.save(folder_path + 'L70_kvalues.npy', kvalues)
np.save(folder_path + 'L70_ABC_solutions.npy', all_ABC_solutions)
np.save(folder_path + 'L70_DEF_solutions.npy', all_DEF_solutions)
np.save(folder_path + 'L70_GHI_solutions.npy', all_GHI_solutions)
  

print("Done.")