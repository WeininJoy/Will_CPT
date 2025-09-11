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
folder_path = './data/best-fit_nu_spacing4_allowedK/'


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
###############################################################################

print('s0, OmegaLambda, OmegaR, OmegaM, OmegaK=', s0, OmegaLambda, OmegaR, OmegaM, OmegaK)    


s0 = 1 # first set s0 to 1, for numerical stability. Will transfer discerete wave vector back to the correct value later.

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10 * s0 # 1e-10
num_variables = 75 # number of pert variables, 75 for original code
H0 = 1/np.sqrt(3*OmegaLambda)
Hinf = H0*np.sqrt(OmegaLambda)
#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs((s**2/s0**2)) + OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

t0 = 1e-8 * s0

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
szero = - OmegaM/s0**3/(4*OmegaR/s0**4)
s1 = OmegaM**2/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = - (OmegaM**3)/(192*s0*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*s0*OmegaLambda*OmegaR) 

s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2


print('Performing Initial Background Integration')

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12* s0], [s_bang], max_step = 0.25e-4 * s0, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
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

deltaeta = 6.e-4 * s0 # integrating from endtime-deltaeta to recombination time, instead of from FCB -> prevent numerical issues
endtime = fcb_time - deltaeta
swaptime = s0 #set time when we swap from s to sigma

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
s_rec = 1+z_rec  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - s_rec) #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

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
# x_inf[2] = 0

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

print(f"x* (transfer matrix) = {x_star_transfer}")
