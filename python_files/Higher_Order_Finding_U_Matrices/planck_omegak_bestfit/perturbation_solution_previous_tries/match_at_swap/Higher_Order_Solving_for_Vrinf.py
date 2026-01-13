# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:22:53 2021

@author: MRose
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

nu_spacing = 4
folder = f'./data/'
folder_path = folder + 'data_all_k/'

num_variables = 75;

# Load matrices for matching at swaptime
kvalues = np.load(folder_path+'L70_kvalues.npy');
ABCmatrices = np.load(folder_path+'L70_ABCmatrices.npy');
DEFmatrices = np.load(folder_path+'L70_DEFmatrices.npy');
GHIvectors = np.load(folder_path+'L70_GHIvectors.npy');
JKLmatrices = np.load(folder_path+'L70_JKLmatrices.npy');
MNOmatrices = np.load(folder_path+'L70_MNOmatrices.npy');
PQRmatrices = np.load(folder_path+'L70_PQRmatrices.npy');
X1matrices = np.load(folder_path+'L70_X1matrices.npy');
X2matrices = np.load(folder_path+'L70_X2matrices.npy');
recValues = np.load(folder_path+'L70_recValues.npy');

print(f"Loaded matrices for {len(kvalues)} k values")
print(f"Matrix shapes: ABC={ABCmatrices[0].shape}, JKL={JKLmatrices[0].shape}, PQR={PQRmatrices[0].shape}")

# ===============================================================================
# Solve for xinf using matching at swaptime
# ===============================================================================
# The key equation is: v_rec = (JKLMNOPQR)^(-1) * (ABCDEFGHI) * (X1 X2 0)^T * xinf
# - JKLMNOPQR: (75 x 75) matrix mapping v_rec to v_swap
# - ABCDEFGHI: (75 x 9) matrix mapping FCB boundary to v_swap
# - X1, X2, 0: (9 x 4) combined matrix for power series expansion at FCB
# - xinf: (4,) vector [δr∞, δm∞, vr∞, vm_dot∞]

vrfcb = [];
drfcb = [];
dmfcb = [];
vmdotfcb = [];

for j in range(len(kvalues)):

    # Load matrices from FCB (backward integration to swaptime)
    ABC = ABCmatrices[j]  # (75, 6)
    DEF = DEFmatrices[j]  # (75, 2)

    # Load matrices from recombination (forward integration to swaptime)
    JKL = JKLmatrices[j]  # (75, 6)
    MNO = MNOmatrices[j]  # (75, 2)
    PQR = PQRmatrices[j]  # (75, 67)

    # Load X matrices and recombination values
    X1 = X1matrices[j]  # (6, 4)
    X2 = X2matrices[j]  # (2, 4)
    recs = recValues[j]  # (75,) perturbation values at recombination

    # =========================================================================
    # Construct the equation: v_rec = (JKLMNOPQR)^(-1) * v_swap
    #                         v_swap = ABCDEF * (X1 X2)^T * xinf
    # =========================================================================

    # Step 1: Construct JKLMNOPQR matrix (75 x 75) from recombination to swaptime
    # v_swap = JKLMNOPQR * v_rec
    JKLMNOPQR = np.hstack([JKL, MNO, PQR])  # (75, 75)

    # Step 2: Construct ABCDEF matrix (75 x 8) from FCB to swaptime
    # Note: We don't include GHI because y_{4:}' = 0 at FCB
    ABCDEF = np.hstack([ABC, DEF])  # (75, 8)

    # Step 3: Construct (X1 X2)^T matrix (8 x 4) for FCB power series expansion
    # - Rows 0-5: X1 (base variables)
    # - Rows 6-7: X2 (Fr2, Fr3)
    X_combined = np.vstack([X1, X2])  # (8, 4)

    # Step 4: Compute v_swap in terms of xinf
    # v_swap = ABCDEF * X_combined * xinf
    # Then: v_rec = (JKLMNOPQR)^(-1) * v_swap
    #             = (JKLMNOPQR)^(-1) * ABCDEF * X_combined * xinf

    # Compute M where: v_rec = M * xinf
    # This solves: JKLMNOPQR * M = ABCDEF * X_combined
    # Using np.linalg.solve(A, B) which solves A*X = B, giving X = A^(-1)*B
    M_full = np.linalg.solve(JKLMNOPQR, ABCDEF @ X_combined)  # (75, 4)

    # Step 5: Solve for xinf using a subset of equations
    # We use rows corresponding to δr, δm, vr, vm (indices 2, 3, 4, 5)
    # because φ and ψ can be derived from other variables
    M_reduced = M_full[[2, 3, 4, 5], :]  # (4, 4)
    recs_reduced = recs[[2, 3, 4, 5]]  # (4,)

    # Solve: M_reduced * xinf = recs_reduced
    xinf = np.linalg.solve(M_reduced, recs_reduced)

    # Extract components: xinf = [δr∞, δm∞, vr∞, vm_dot∞]
    vrfcb.append(xinf[2])      # vr∞ - this should be zero for allowed modes
    drfcb.append(xinf[0])      # δr∞
    dmfcb.append(xinf[1])      # δm∞
    vmdotfcb.append(xinf[3])   # vm_dot∞

#np.save('L70_vrfcb', vrfcb);


# Convert to numpy array and remove any NaNs that might have occurred
vrfcb = np.array(vrfcb)
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
print(f"allowed K values: {allowedK_refined}")

# Save the new, highly accurate wavenumbers
output_filename = 'allowedK.npy'
np.save(folder_path+output_filename, allowedK_refined)
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