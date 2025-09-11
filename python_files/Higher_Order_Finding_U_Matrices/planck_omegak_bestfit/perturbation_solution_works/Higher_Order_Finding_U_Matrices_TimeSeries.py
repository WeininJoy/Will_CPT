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
from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt

# --- Basic Setup (Unchanged) ---
# working in units 8piG = Lambda = c = hbar = kB = 1 throughout
folder_path = './nu_spacing4/data_allowedK_timeseries/'
allowedK_path = './nu_spacing4/data_all_k/allowedK.npy'
import os
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# # set cosmological parameters from Planck baseline
# OmegaLambda = 0.679
# OmegaM = 0.321
# OmegaR = 9.24e-5
# H0 = 1/np.sqrt(3*OmegaLambda) # we are working in units of Lambda=c=1

## --- Best-fit parameters for nu_spacing =8 ---
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
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return Omega_lambda, Omega_m, Omega_K

# # Best-fit parameters from nu_spacing=8
# mt, kt, Omegab_ratio, h = 374.09852041763577, 0.3513780000350142, 0.16073476133403286, 0.6449862051259417
# Best-fit parameters from nu_spacing=4
mt, kt, Omegab_ratio, h = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1065.0  # the actual value still needs to be checked
###############################################################################

#set tolerances
atol = 1e-13;
rtol = 1e-13;
stol = 1e-10;
num_variables = 75; # number of pert variables
swaptime = 2; #set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda);

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8;

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR);
szero = - OmegaM/(4*OmegaR);
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Check if t_events[0] is not empty before trying to access its elements
if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    # You might want to assign a default value or 'None' to fcb_time here
    fcb_time = None # Or np.nan, or some other indicator

# Rest of your code that uses fcb_time would go here
# For example:
if fcb_time is not None:
    print(f"Further processing with fcb_time = {fcb_time}")
else:
    print(f"No fcb_time available for further processing.")

endtime = fcb_time - deltaeta

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
s_rec = 1+z_rec  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - s_rec) #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

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
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2 #+ fr2/2
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
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2 #+ fr2/2
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

kvalues = np.load(allowedK_path)

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
s_init = np.interp(endtime, sol.t, sol.y[0])

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