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

folder_path = './nu_spacing4/data_all_k/'

num_variables = 75;

kvalues = np.load(folder_path+'L70_kvalues.npy');
ABCmatrices = np.load(folder_path+'L70_ABCmatrices.npy');
DEFmatrices = np.load(folder_path+'L70_DEFmatrices.npy');
GHIvectors = np.load(folder_path+'L70_GHIvectors.npy');
X1matrices = np.load(folder_path+'L70_X1matrices.npy');
X2matrices = np.load(folder_path+'L70_X2matrices.npy');
recValues = np.load(folder_path+'L70_recValues.npy');

#first extract A and D matrices from results

Amatrices = [];
Bmatrices = [];
Cmatrices = [];
Dmatrices = [];
Ematrices = [];
Fmatrices = [];
GX3matrices = [];
HX3matrices = [];
IX3matrices = [];

for i in range(len(kvalues)):
    
    ABC = ABCmatrices[i];
    DEF = DEFmatrices[i];
    GHI = GHIvectors[i];
    
    A = ABC[0:6, 0:6];
    B = ABC[6:8, 0:6];
    C = ABC[8:num_variables, 0:6];
    D = DEF[0:6, 0:2]; 
    E = DEF[6:8, 0:2];
    F = DEF[8:num_variables, 0:2];
    G = GHI[0:6];
    H = GHI[6:8];
    I = GHI[8:num_variables];
    
    #create zero arrays for G,H and I matrices
    GX3mat = np.zeros(shape=(6,4));
    GX3mat[:,2] = G;
    HX3mat = np.zeros(shape=(2,4));
    HX3mat[:,2] = H;
    IX3mat = np.zeros(shape=(num_variables-8, 4));
    IX3mat[:,2] = I;
    
    Amatrices.append(A);
    Bmatrices.append(B);
    Cmatrices.append(C);
    Dmatrices.append(D);
    Ematrices.append(E);
    Fmatrices.append(F);
    GX3matrices.append(GX3mat);
    HX3matrices.append(HX3mat);
    IX3matrices.append(IX3mat);
    
#now set up matrix equations to solve for xinf
vrfcb = [];
drfcb = [];
dmfcb = [];
vmdotfcb = [];
yrecs = [];

for j in range(len(kvalues)):
    
    A = Amatrices[j];
    D = Dmatrices[j];
    GX3 = GX3matrices[j];
    B = Bmatrices[j];
    E = Ematrices[j];
    HX3 = HX3matrices[j];
    C = Cmatrices[j];
    F = Fmatrices[j];
    IX3 = IX3matrices[j];
    X1 = X1matrices[j];
    X2 = X2matrices[j];
    recs = recValues[j];
    
    #calculate full matrix but then remove top two rows
    AX1 = A.reshape(6,6) @ X1.reshape(6,4);
    DX2 = D.reshape(6,2) @ X2.reshape(2,4);
    matrixog = AX1 + DX2 + GX3;
    matrix = matrixog[[2,3,4,5], :];
    
    xrecs = [recs[2], recs[3], recs[4], recs[5]];

    xinf = np.linalg.lstsq(matrix, xrecs, rcond=None)[0];
    vrfcb.append(xinf[2]);
    drfcb.append(xinf[0]);
    dmfcb.append(xinf[1]);
    vmdotfcb.append(xinf[3]);
    
    #mat1 = B.reshape(2,6) @ X1.reshape(6,4);
    #mat2 = E.reshape(2,2) @ X2.reshape(2,4);
    #yrec = (mat1 + mat2).reshape(2,4) @ xinf.reshape(4,1);
    
    #yrecs.append(yrec);

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