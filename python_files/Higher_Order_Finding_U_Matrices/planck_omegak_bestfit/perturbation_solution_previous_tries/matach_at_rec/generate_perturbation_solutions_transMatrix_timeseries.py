# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model using pre-computed time-dependent transfer matrices.

This method avoids unstable backward integration by reconstructing the solution
from a basis of pre-calculated evolutions, making it numerically robust.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# =============================================================================
# 1. SETUP: Parameters and Constants (Mostly Unchanged)
# =============================================================================
nu_spacing = 4

print("--- Setting up parameters and functions ---")
# *** NEW: Point to the time-series data directory ***
folder_path = f'./data/nu_spacing{nu_spacing}_s-a/'
folder_path_matrices = folder_path + 'data_allowedK/' # For X1, X2, recValues
folder_path_timeseries = folder_path + 'data_allowedK_timeseries/' # For solution histories
# folder_path_matrices = folder_path + 'data_integerK/' # For X1, X2, recValues
# folder_path_timeseries = folder_path + 'data_integerK_timeseries/' # For solution histories

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

###############################################################################
# Best-fit parameters from nu_spacing=4 (first try)
# mt, kt, Omegab_ratio, h = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915 # calculated based on the calculate_z_rec() output
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
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-8;

#set coefficients for initial conditions
# smin1 = np.sqrt(3*OmegaLambda/OmegaR);
# szero = - OmegaM/(4*OmegaR);
# s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
# s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
# s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
# s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

# s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda));
a2 = OmegaM/(12*OmegaLambda);
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2));
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2);
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4; 

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol_a = solve_ivp(da_dt, [t0,swaptime], [a_Bang], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
sol = solve_ivp(ds_dt, [swaptime, 12], [1./sol_a.y[0][-1]], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
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
a_rec = 1./(1+z_rec)  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol_a.y[0] - a_rec) #take difference between s values and s_rec to find where s=s_rec
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

# Perfect fluid ODE for early times (unchanged)
def dX_perfect(t, X, k):
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k**2)*X[4]);
    dmdot = 3*phidot + X[5]*(k**2);
    vrdot = -(X[1] + X[2]/4);
    vmdot = - (adot/X[0])*X[5] - X[1];

    return [adot, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 2. DATA LOADING
# =============================================================================

print("\n--- Loading pre-computed time-dependent transfer matrices ---")
try:
    # *** NEW: Load the time-series data ***
    t_grid = np.load(folder_path_timeseries + 't_grid.npy')
    allowedK = np.load(folder_path_timeseries + 'L70_kvalues.npy')
    all_ABC_solutions = np.load(folder_path_timeseries + 'L70_ABC_solutions.npy')
    all_DEF_solutions = np.load(folder_path_timeseries + 'L70_DEF_solutions.npy')
    all_GHI_solutions = np.load(folder_path_timeseries + 'L70_GHI_solutions.npy')
    
    # Load the static matrices needed for boundary conditions
    ABCmatrices = np.load(folder_path_matrices+'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path_matrices+'L70_DEFmatrices.npy')
    GHIvectors = np.load(folder_path_matrices+'L70_GHIvectors.npy')
    X1matrices = np.load(folder_path_matrices + 'L70_X1matrices.npy')
    X2matrices = np.load(folder_path_matrices + 'L70_X2matrices.npy')
    recValues = np.load(folder_path_matrices + 'L70_recValues.npy')
    
    print(f"Loaded solution histories for {len(allowedK)} allowed K values.")
    print(f"Time grid has {len(t_grid)} points from eta={t_grid[0]:.2f} to eta={t_grid[-1]:.2f}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure you have run 'Higher_Order_Finding_U_Matrices_TimeSeries.py' first.")
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
# 3. CALCULATE AND RECONSTRUCT SOLUTIONS
# =============================================================================

print("\n--- Reconstructing solutions for the first 3 allowed modes ---")

solutions = []
num_modes_to_plot = min(3, len(allowedK))

for i in range(num_modes_to_plot):
    # This logic now selects the first few modes for simplicity
    k_index = i + 3
    k = allowedK[k_index]
    print(f"\nProcessing mode n={i+1} with k={k:.6f} (index {k_index})")

    # --- a) Get matrices for solving boundary conditions ---
    # *** NEW: Extract endpoint matrices from the time-series data ***
    # The saved arrays are from rec_time to end_time, so the rec_time values are at index 0.

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
    GX3 = np.zeros((6,4))
    GX3[:,2] = GHIvectors[k_index][0:6]

    M_matrix = (A @ X1 + D @ X2)[2:6, :]  # Only use independent components [dr, dm, vr, vm]
    x_rec = recs_vec[2:6]  # Only use independent components from stored values
    # x_inf = np.linalg.solve(M_matrix, x_rec)
    x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]
    print(f"x^∞ = {x_inf}")
    # x_inf[2] = 0

    # Calculate x' and y' at endtime using X1 and X2 (equation 25)
    x_prime_coeffs = X1 @ x_inf
    y_prime_coeffs = X2 @ x_inf

    # --- d) RECONSTRUCT solution from recTime to endTime ---
    # *** THIS ENTIRE SECTION REPLACES THE BACKWARD INTEGRATION ***
    print("  Reconstructing solution from pre-computed basis...")
    ABC_sols_k = all_ABC_solutions[k_index] # Shape: (num_vars, 6, num_times)
    DEF_sols_k = all_DEF_solutions[k_index] # Shape: (num_vars, 2, num_times)
    GHI_sols_k = all_GHI_solutions[k_index] # Shape: (num_vars, num_times)
    
    # Linearly combine the basis solutions using the calculated coefficients
    # Y(t) = U_ABC(t) * x_prime_coeffs + U_DEF(t) * y_prime_coeffs + Y_GHI(t)
    Y_reconstructed = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                      np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs) # + GHI_sols_k

    # The time array for this solution is simply the loaded t_grid
    t_backward = t_grid
    
    # Add the background 's' variable back for stitching and plotting
    s_background = np.interp(t_grid, np.concatenate((sol_a.t, sol.t)), np.concatenate(([1./sol_a.y[0][i] for i in range(len(sol_a.y[0]))], sol.y[0])))
    Y_backward = np.vstack([s_background, Y_reconstructed])
    
    # --- e) Get solution from Big Bang to Recombination (unchanged) ---
    # (Using a simplified initial condition for this example)
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda);
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda);
    
    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda);
    
    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda);
    
    #set initial conditions
    t0 = 1e-8;
    phi0 = 1 + phi1*t0 + phi2*t0**2; #t0 from above in "background equations section"
    dr0 = -2 + dr1*t0 + dr2*t0**2;
    dm0 = -1.5 + dm1*t0 + dm2*t0**2;
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3;
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3;

    Y0_perfect = [a_Bang, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect, [t0, recConformalTime], Y0_perfect,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    # --- f) Stitch and Reflect ---
    # *** MODIFIED SECTION: PREPARE AND STORE HALVES SEPARATELY ***
    
    # Create the left part (Big Bang -> FCB)
    t_left_unsorted = np.concatenate((sol_perfect.t, t_backward[::-1]))

    Y_perfect_full = np.zeros((num_variables + 1, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
    Y_perfect_full[1,:] = sol_perfect.y[1,:]
    Y_perfect_full[2,:] = sol_perfect.y[1,:]
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]
    Y_left_unsorted = np.concatenate((Y_perfect_full, Y_backward[:, ::-1]), axis=1)

    # *** NEW FIX: Sort the left-hand data by time to prevent plotting artifacts ***
    sort_indices = np.argsort(t_left_unsorted)
    t_left = t_left_unsorted[sort_indices]
    Y_left = Y_left_unsorted[:, sort_indices]

    # Create the right part (FCB -> Big Crunch) by reflecting the left part
    # We use the sorted left part to ensure the right part is also sorted correctly.
    t_right = 2 * fcb_time - t_left[::-1]
    
    # Symmetry matrix for reflection
    symm = np.ones(num_variables + 1)
    symm[5] = -1 # vm
    symm[6] = -1 # fr2
    # This loop correctly sets symmetries for odd-l multipoles
    for l_idx in range(num_variables - 8):
        l = l_idx + 3
        if l % 2 != 0:
            symm[8 + l_idx] = -1
    S = np.diag(symm)
    Y_right = S @ Y_left[:, ::-1]
    
    # Store the separate halves for plotting
    solutions.append({
        't_left': t_left,
        'Y_left': Y_left[1:, :],  # Store only perturbation variables
        't_right': t_right,
        'Y_right': Y_right[1:, :] # Store only perturbation variables
    })


# =============================================================================
# 4. PLOTTING (MODIFIED SECTION)
# =============================================================================

print("\n--- Plotting solutions ---")
fig, axes = plt.subplots(num_modes_to_plot, 1, figsize=(12, 6), sharex=True, gridspec_kw={'hspace': 0})
if num_modes_to_plot == 1: axes = [axes]

# Define labels, colors, and the corresponding indices in the perturbation vector
labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$', r'$\phi$', r'$\psi$']
indices = [4, 2, 5, 3, 0, 1]  # Order: vr, dr, vm, dm, phi, psi
colors = ['blue', 'red', 'green', 'black', 'magenta', 'cyan']

for i, sol in enumerate(solutions):
    ax = axes[i]
    for j, (label, index, color) in enumerate(zip(labels, indices, colors)):
        current_label = label if i == 0 else None # Only label the top subplot for the legend

        # Plot the first half (Big Bang to FCB)
        ax.plot(sol['t_left'], sol['Y_left'][index, :], label=current_label, color=color, linewidth=1.5)

        # Plot the second half (FCB to Big Crunch)
        ax.plot(sol['t_right'], sol['Y_right'][index, :], color=color, linewidth=1.5)

    # --- Formatting for each subplot ---
    # Vertical boundary lines
    ax.axvline(fcb_time, color='k', linestyle='--')
    ax.axvline(0, color='cyan', linestyle='--') # Big Bang line
    ax.axvline(2 * fcb_time, color='cyan', linestyle='--') # Big Crunch line
    
    # Horizontal center line
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, dashes=(5, 5))
    
    # Y-axis label
    ax.set_ylabel(f'$n = {i+3+1}$', rotation=90, labelpad=15, va='center', fontsize=14)
    
    # Y-axis ticks and limits
    if i == 0: ax.set_ylim(-0.3, 0.3) 
    elif i == 1: ax.set_ylim(-0.6, 0.6)
    else:ax.set_ylim(-1., 1.)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False) # Hide y-ticks and labels

# --- Final plot formatting ---
axes[-1].set_xlim(-0.1, 2 * fcb_time + 0.1)
axes[-1].set_xlabel(r'$\eta\sqrt{\Lambda}$', fontsize=16)
axes[-1].tick_params(axis='x', labelsize=12)
axes[-1].set_xticks(np.arange(0, 12, 2)) # Set x-ticks to be 0, 2, 4, ... 8

# Top text labels
axes[0].text(0, 1.1, 'Big Bang', ha='center', va='center', transform=axes[0].transAxes, fontsize=14)
axes[0].text(0.5, 1.1, 'Future Conformal Boundary', ha='center', va='center', transform=axes[0].transAxes, fontsize=14)
axes[0].text(1, 1.1, 'Big Crunch', ha='center', va='center', transform=axes[0].transAxes, fontsize=14)

# Legend above the entire plot
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=True, fontsize=14)

# Adjust layout to prevent overlap and make space for top elements
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])

plt.savefig(folder_path + 'perturbation_solutions_transMatrix_timeseries_s-a.pdf')
plt.show()