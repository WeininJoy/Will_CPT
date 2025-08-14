# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:22:53 2021
@author: MRose

MODIFIED VERSION: This script finds the allowed wavenumbers 'k' by first
calculating vr_inf for a grid of k-values, then uses a cubic spline
interpolation and a robust numerical root-finder (scipy.optimize.root_scalar)
to determine the k-values with much higher precision.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# --- All setup code is the same as your original code(3) ---

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
folder_path = './data/'

try:
    kvalues = np.load(folder_path + f'L70_kvalues.npy')
    print(f'Successfully loaded kvalues.')
except FileNotFoundError as e:
    print(f"Error: {e}"); exit()

# Parameters
s0 = 1 # working unit
num_variables = 75

# Load pre-computed matrices
ABCmatrices = np.load(folder_path + f'L70_ABCmatrices.npy')
DEFmatrices = np.load(folder_path + f'L70_DEFmatrices.npy')
X1matrices = np.load(folder_path + f'L70_X1matrices.npy')
X2matrices = np.load(folder_path + f'L70_X2matrices.npy')
recValues = np.load(folder_path + f'L70_recValues.npy')

Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

# Calculate vr_inf (which you called vrfcb) for the grid of kvalues
vrfcb_list = []
print("Calculating vr_inf across the k-grid...")
for j in range(len(kvalues)):
    A = Amatrices[j]
    D = Dmatrices[j]
    X1 = X1matrices[j]
    X2 = X2matrices[j]
    recs = recValues[j]
    
    matrix_M = (A @ X1 + D @ X2)[2:6, :]
    xrecs_subset = [recs[2], recs[3], recs[4], recs[5]]
    
    try:
        xinf = np.linalg.solve(matrix_M, xrecs_subset)
        vr_infinity = xinf[2] # This is the value we want to be zero
        vrfcb_list.append(vr_infinity)
    except np.linalg.LinAlgError:
        # Handle cases where the matrix is singular for a given k
        vrfcb_list.append(np.nan)

# Convert to numpy array and remove any NaNs that might have occurred
vrfcb = np.array(vrfcb_list)
valid_indices = ~np.isnan(vrfcb)
k_grid = kvalues[valid_indices]
vrfcb_grid = vrfcb[valid_indices]


# --- NEW: Accurate Root Finding ---
print("\nFinding refined allowed K values using interpolation and root finding...")

# 1. Create a continuous function vr_inf(k) using a cubic spline
vr_inf_func = interp1d(k_grid, vrfcb_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")

# 2. Find the brackets where the sign changes (same as before)
idxzeros = np.where(np.diff(np.sign(vrfcb_grid)) != 0)[0]

allowedK_refined = []
for idx in idxzeros:
    # Define a narrow bracket around the zero-crossing for the root-finder
    k_bracket = (k_grid[idx], k_grid[idx+1])
    
    try:
        # 3. Use a robust root-finding algorithm to find the precise k
        sol = root_scalar(vr_inf_func, bracket=k_bracket, method='brentq')
        
        if sol.converged:
            refined_k = sol.root
            # Note: Your original code scaled by s0 here. If your physical s0 is not 1,
            # you should re-introduce that scaling. Since we set s0=1, this is fine.
            allowedK_refined.append(refined_k)
            print(f"Found root at k = {refined_k:.8f} in bracket {k_bracket}")
        else:
            print(f"Warning: Root finding failed to converge for bracket {k_bracket}")
            
    except ValueError as e:
        # This can happen if the values at the bracket endpoints have the same sign
        # due to interpolation artifacts.
        print(f"Skipping bracket {k_bracket} due to ValueError: {e}")

allowedK_refined = np.array(allowedK_refined)

# Save the new, highly accurate wavenumbers
output_filename = folder_path + 'allowedK.npy'
np.save(output_filename, allowedK_refined)
print(f"\nSaved {len(allowedK_refined)} refined allowed K values to {output_filename}")

# Optional: Plot for verification
plt.figure(figsize=(10, 6))
plt.plot(k_grid, vrfcb_grid, 'b-', label=r'$v_r^\infty(k)$ from grid')
plt.plot(allowedK_refined, np.zeros_like(allowedK_refined), 'ro', label='Refined Roots')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Comoving Wavenumber K')
plt.ylabel(r'$v_r^\infty$')
plt.title('Finding Allowed Wavenumbers')
plt.legend()
plt.grid(True)
plt.ylim(-2, 2) # Adjust ylim to see the oscillations clearly
plt.savefig(folder_path + 'vrfcb_refined_roots.pdf')