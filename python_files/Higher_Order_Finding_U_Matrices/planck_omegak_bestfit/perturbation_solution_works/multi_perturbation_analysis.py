# -*- coding: utf-8 -*-
"""
This script performs a Gram-Schmidt orthonormalization and eigenvalue analysis
on two different bases of the COMPLETE cosmological perturbation solutions for
the radiation velocity, v_r(eta).

The solution generator is based on the user's verified script for plotting
the full palindromic evolution.

- basis_1 is constructed from wavenumbers 'k' in a closed universe model.
- basis_2 is constructed from the 'allowed' wavenumbers for a palindromic flat universe.

It identifies and compares the common eigenfunctions of v_r found in both bases.
"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# =============================================================================
# 1. SETUP: Parameters and Constants (Mostly Unchanged)
# =============================================================================
nu_spacing = 4

print("--- Setting up parameters and functions ---")

## --- Best-fit parameters ---
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
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 409.969398,1.459351,0.163514,0.547313,2.095762,0.972835,0.053017

OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1063.4075 # calculated based on the calculate_z_rec() output
###############################################################################

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#set tolerances
atol = 1e-13;
rtol = 1e-13;
stol = 1e-10;
num_variables = 75; # number of pert variables
swaptime = 2; #set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda);
a0=1; K=-OmegaK * a0**2 * H0**2

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5;

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
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2));
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3);
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6; 

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
def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    sigmadot = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)+OmegaR*np.exp(2*sigma))
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4) 
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

def generate_multi_perturbation_bases(discreteK_type, eta_grid): # discreteK_type = 'allowedK' or 'integerK'
    # =============================================================================
    # 2. DATA LOADING
    # =============================================================================
    # *** Point to the time-series data directory ***
    folder_path_matrices = folder_path + f'data_{discreteK_type}/' # For X1, X2, recValues
    folder_path_timeseries = folder_path + f'data_{discreteK_type}_timeseries/' # For solution histories

    print("\n--- Loading pre-computed time-dependent transfer matrices ---")
    try:
        # *** NEW: Load the time-series data ***
        t_grid = np.load(folder_path_timeseries + 't_grid.npy')
        allowedK = np.load(folder_path_timeseries + 'L70_kvalues.npy')
        print(f"K values from {discreteK_type}:", allowedK)
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

    perturbation_types = ['dr', 'dm', 'vr', 'vm']  # Note: phi and psi are not directly considered, since they can be derived from other variables
    basis_dict = {pert_type: [] for pert_type in perturbation_types}

    for i in range(len(allowedK)):
        # This logic now selects the first few modes for simplicity
        k_index = i 
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
        s_background = np.interp(t_grid, sol.t, sol.y[0])
        Y_backward = np.vstack([s_background, Y_reconstructed])
        
        # # --- e) Get solution from Big Bang to Recombination (unchanged) ---
        # # (Using a simplified initial condition for this example)
        # phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
        # phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
        
        # dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
        # dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda);
        
        # dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
        # dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda);
        
        # vr1 = -1/2;
        # vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
        # vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda);
        
        # vm1 = -1/2;
        # vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
        # vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda);
        
        # #set initial conditions
        # t0 = 1e-8;
        # s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3;
        # sigma0 = np.log(s0)
        # phi0 = 1 + phi1*t0 + phi2*t0**2; #t0 from above in "background equations section"
        # dr0 = -2 + dr1*t0 + dr2*t0**2;
        # dm0 = -1.5 + dm1*t0 + dm2*t0**2;
        # vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3;
        # vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3;

        # Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
        # sol_perfect = solve_ivp(dX_perfect_sigma, [t0, recConformalTime], Y0_perfect,
        #                         dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

        # # --- f) Stitch and Reflect ---
        # # *** MODIFIED SECTION: PREPARE AND STORE HALVES SEPARATELY ***

        # Y_perfect_full = np.zeros((num_variables + 1, len(sol_perfect.t)))
        # Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
        # Y_perfect_full[1,:] = sol_perfect.y[1,:]
        # Y_perfect_full[2,:] = sol_perfect.y[1,:]
        # Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]

        # First, create the common time array
        # t_full_unsorted = np.concatenate((sol_perfect.t, t_backward[::-1], [fcb_time]))
        t_full_unsorted = np.concatenate((t_backward[::-1], [fcb_time]))

        # Now, assemble each solution vector individually
        # Note: Indices come from your original if/elif block
        # Perfect fluid state: [sigma, phi, dr, dm, vr, vm] -> indices 0, 1, 2, 3, 4, 5
        # Boltzmann state: [s, phi, phidot, dr, dm, vr, vm, ...] -> indices 0, 1, 2, 3, 4, 5, 6
        
        solutions_unsorted = {
            'dr': np.concatenate((
                # sol_perfect.y[2, :],          # Perfect fluid dr
                Y_backward[3, ::-1],          # Boltzmann dr
                [x_inf[0]]                    # Value at FCB (dr_inf)
            )),
            'dm': np.concatenate((
                # sol_perfect.y[3, :],          # Perfect fluid dm
                Y_backward[4, ::-1],          # Boltzmann dm
                [x_inf[1]]                    # Value at FCB (dm_inf)
            )),
            'vr': np.concatenate((
                # sol_perfect.y[4, :],          # Perfect fluid vr
                Y_backward[5, ::-1],          # Boltzmann vr
                [x_inf[2]]                    # Value at FCB (vr_inf)
            )),
            'vm': np.concatenate((
                # sol_perfect.y[5, :],          # Perfect fluid vm
                Y_backward[6, ::-1],          # Boltzmann vm
                [(X1 @ x_inf)[3]]             # Value at FCB (vm_inf)
            ))
        }
        
        # Sort by time to ensure monotonicity for interpolation
        sort_indices = np.argsort(t_full_unsorted)
        t_sol = t_full_unsorted[sort_indices]
        
        # Apply the same sorting to all solution arrays in the dictionary
        y_sol = {key: value[sort_indices] for key, value in solutions_unsorted.items()}
            
        # Interpolate each perturbation type onto common eta grid
        for pert_type in perturbation_types:
            if pert_type in y_sol:
                interpolator = interp1d(t_sol, y_sol[pert_type], 
                                        bounds_error=False, fill_value=0.0)
                solution = interpolator(eta_grid)
                
                basis_dict[pert_type].append(solution)
            else:
                print(f"    Warning: {pert_type} not found in solution, using zeros")
                basis_dict[pert_type].append(np.zeros_like(eta_grid))
    
    # Convert lists to numpy arrays
    for pert_type in perturbation_types:
        basis_dict[pert_type] = np.array(basis_dict[pert_type]).T  # Transpose for QR

    return basis_dict

# =============================================================================
# SECTION 2: GRAM-SCHMIDT ANALYSIS 
# =============================================================================
def qr_decomposition(basis):
    """QR decomposition with proper handling"""
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix.T)

def compute_multi_perturbation_A_matrix(ortho_funcs_1_dict, ortho_funcs_2_dict):
    """
    Compute the A matrix by summing contributions from all perturbation types
    
    Parameters:
    ortho_funcs_1_dict: dict of orthonormal functions from basis 1 for each perturbation
    ortho_funcs_2_dict: dict of orthonormal functions from basis 2 for each perturbation
    
    Returns:
    Combined eigenvalues and eigenvectors from all perturbations
    """
    perturbation_types = ['dr', 'dm', 'vr', 'vm'] # only those the oscillating modes
    
    # Initialize combined M matrix
    M_total = None
    n_perturbations = 0
    
    print("Computing combined A matrix from all perturbations...")
    
    for pert_type in perturbation_types:
        if pert_type in ortho_funcs_1_dict and pert_type in ortho_funcs_2_dict:
            # Compute M for this perturbation type
            M_pert = np.dot(ortho_funcs_1_dict[pert_type].T, ortho_funcs_2_dict[pert_type])
            
            if M_total is None:
                M_total = M_pert
            else:
                M_total += M_pert
            
            n_perturbations += 1
            print(f"  Added contribution from {pert_type}")
    
    # Normalize by the number of perturbations
    if n_perturbations > 0:
        M_total = M_total / n_perturbations
        print(f"  Normalized by {n_perturbations} perturbation types")
    
    # Compute eigenvalues and eigenvectors of combined matrix
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M_total, M_total.T))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M_total.T, M_total))
    
    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2, M_total

def choose_eigenvalues(eigenvalues, eigenvectors, eigenvalues_threshold=0.99):
    """Select the largest eigenvalues"""
    # Sort eigenvalues in descending order and get indices
    sorted_indices = np.argsort(np.real(eigenvalues))[::-1]
    
    eigenvalues_valid, eigenvectors_valid = [], []
    
    for i in range(len(eigenvalues)):
        idx = sorted_indices[i]
        if np.real(eigenvalues[idx]) < eigenvalues_threshold:
            break
        eigenvalues_valid.append(eigenvalues[idx])
        eigenvectors_valid.append(eigenvectors[:, idx])
    
    return eigenvalues_valid, eigenvectors_valid

def compute_coefficients(eigenvalues, eigenvectors, transformation_matrix):
    """Compute linear combination coefficients"""
    coefficients = np.zeros((len(eigenvalues), transformation_matrix.shape[1]), dtype=float)
    for i in range(len(eigenvalues)):
        coefficients[i, :] = np.dot(np.array(eigenvectors[i]), transformation_matrix)
    return coefficients

def multi_perturbation_analysis(N=23, N_t=500, eigenvalues_threshold=0.99):
    """
    Complete multi-perturbation eigenvalue analysis with adaptive time truncation
    """
    print(f"Starting multi-perturbation analysis with N={N}, N_t={N_t}, eigenvalues_threshold={eigenvalues_threshold}")
    
    # Define eta_grid from cutoff_time to fcb_time as requested
    eta_grid = np.linspace(0, fcb_time, N_t)
    print(f"Using eta_grid from cutoff to FCB: eta ∈ [0, {fcb_time:.4e}]")

    # Generate multi-perturbation bases
    print("\nGenerating basis 1 (Closed Universe)...")
    basis_1_dict = generate_multi_perturbation_bases("integerK", eta_grid)
    
    print("\nGenerating basis 2 (Palindromic Universe)...")
    basis_2_dict = generate_multi_perturbation_bases("allowedK", eta_grid)
    
    # Perform QR decomposition for each perturbation type
    print("\nPerforming QR decomposition for each perturbation type...")
    ortho_funcs_1_dict = {}
    ortho_funcs_2_dict = {}
    transform_1_dict = {}
    transform_2_dict = {}
    
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    
    for pert_type in perturbation_types:
        print(f"  QR decomposition for {pert_type}")
        
        # Check for valid data
        if np.all(basis_1_dict[pert_type] == 0):
            print(f"    Warning: Basis 1 for {pert_type} is all zeros, skipping")
            continue
        if np.all(basis_2_dict[pert_type] == 0):
            print(f"    Warning: Basis 2 for {pert_type} is all zeros, skipping")
            continue
            
        try:
            ortho_funcs_1_dict[pert_type], transform_1_dict[pert_type] = qr_decomposition(basis_1_dict[pert_type])
            ortho_funcs_2_dict[pert_type], transform_2_dict[pert_type] = qr_decomposition(basis_2_dict[pert_type])
            print(f"    Successfully processed {pert_type}")
        except Exception as e:
            print(f"    Error in QR decomposition for {pert_type}: {e}")
    
    # Compute combined A matrix from all perturbations
    print("\nComputing combined eigenvalue analysis...")
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2, M_combined = compute_multi_perturbation_A_matrix(
        ortho_funcs_1_dict, ortho_funcs_2_dict)
    
    # Select largest eigenvalues
    eigenvals_valid_1, eigenvecs_valid_1 = choose_eigenvalues(
        eigenvals_1, eigenvecs_1, eigenvalues_threshold)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_eigenvalues(
        eigenvals_2, eigenvecs_2, eigenvalues_threshold)
    
    print(f"\nFound {len(eigenvals_valid_1)} valid eigenvalues for basis 1")
    print(f"Found {len(eigenvals_valid_2)} valid eigenvalues for basis 2")
    
    if len(eigenvals_valid_1) > 0:
        print(f"Eigenvalues (basis 1): {[f'{ev:.4f}' for ev in eigenvals_valid_1[:5]]}")
    if len(eigenvals_valid_2) > 0:
        print(f"Eigenvalues (basis 2): {[f'{ev:.4f}' for ev in eigenvals_valid_2[:5]]}")
    
    # Compute coefficients for reconstruction
    coefficients_1_dict = {}
    coefficients_2_dict = {}
    
    for pert_type in perturbation_types:
        if pert_type in transform_1_dict and len(eigenvals_valid_1) > 0:
            coefficients_1_dict[pert_type] = compute_coefficients(
                eigenvals_valid_1, eigenvecs_valid_1, transform_1_dict[pert_type])
        if pert_type in transform_2_dict and len(eigenvals_valid_2) > 0:
            coefficients_2_dict[pert_type] = compute_coefficients(
                eigenvals_valid_2, eigenvecs_valid_2, transform_2_dict[pert_type])
    
    # # save the coefficients
    # with open("multi_perturbation_coefficients1.pickle", 'wb') as f:
    #     pickle.dump(coefficients_1_dict, f)

    return {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        'eigenvecs_1': eigenvecs_valid_1,
        'eigenvecs_2': eigenvecs_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,
        'basis_2': basis_2_dict,
        'M_combined': M_combined
    }

def plot_multi_perturbation_results(results, N_plot=3):
    """
    Plot the combined eigenfunctions from multi-perturbation analysis
    """
    eta_grid = results['eta_grid']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']
    coefficients_2 = results['coefficients_2']
    basis_1 = results['basis_1']
    basis_2 = results['basis_2']
    
    N_plot = min(N_plot, len(eigenvals_1))
    if N_plot == 0:
        print("No valid eigenfunctions to plot")
        return
    
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    n_pert = len(perturbation_types)
    
    fig, axes = plt.subplots(N_plot, n_pert, figsize=(16, 2*N_plot), 
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Multi-Perturbation Common Eigenfunctions", fontsize=16)
    
    for i in range(N_plot):
        for j, pert_type in enumerate(perturbation_types):
            ax = axes[i, j]
            
            if pert_type not in coefficients_1 or pert_type not in coefficients_2:
                ax.text(0.5, 0.5, f"No data\nfor {pert_type}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{pert_type} (eigenval {i+1})")
                continue
            
            # Reconstruct solutions from both bases
            solution_1 = np.zeros_like(eta_grid)
            solution_2 = np.zeros_like(eta_grid)
            
            N = len(coefficients_1[pert_type][i])
            for k in range(N):
                solution_1 += coefficients_1[pert_type][i, k] * basis_1[pert_type][:, k]
                solution_2 += coefficients_2[pert_type][i, k] * basis_2[pert_type][:, k]
            
            # Sign alignment
            if np.dot(solution_1, solution_2) < 0:
                solution_2 *= -1
            
            # Plot
            ax.plot(eta_grid, solution_1, 'r-', linewidth=2.5, 
                   label='Basis 1 (Closed)', alpha=0.8)
            ax.plot(eta_grid, solution_2, 'g--', linewidth=2.0, 
                   label='Basis 2 (Palindromic)', alpha=0.8)
            
            ax.set_title(f"{pert_type} (λ={eigenvals_1[i]:.3f})")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
            if i == N_plot - 1:
                ax.set_xlabel("Conformal Time η")

    plt.savefig("multi_perturbation_eigenfunctions_after_recombination.pdf", dpi=300, bbox_inches='tight')

def plot_coefficients_by_dominant_k(results, eigenvalue_threshold=0.99, N_plot=10):
    """
    Plot coefficients of eigenfunctions sorted by dominant k mode

    Parameters:
    -----------
    results : dict
        Results dictionary from multi_perturbation_analysis
    eigenvalue_threshold : float
        Minimum eigenvalue to include (default: 0.95)
    N_plot : int
        Maximum number of eigenfunctions to plot
    """
    eigenvals_1 = np.array(results['eigenvals_1'])
    eigenvecs_1 = results['eigenvecs_1']

    # Step 1: Filter eigenvectors with eigenvalues > threshold
    valid_mask = eigenvals_1.real > eigenvalue_threshold
    eigenvals_valid = eigenvals_1[valid_mask]

    print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
    print(f"Found {len(eigenvals_valid)} eigenvectors above threshold out of {len(eigenvals_1)} total")

    if len(eigenvals_valid) == 0:
        print("No eigenvectors found above threshold. Try lowering the threshold.")
        return

    # Get valid indices
    valid_indices = np.where(valid_mask)[0]

    # Step 2: Sort by dominant k mode
    # Extract coefficients for the reference perturbation
    # Convert list of eigenvectors to numpy array first
    eigenvecs_1_array = np.array([eigenvecs_1[i] for i in range(len(eigenvecs_1))])
    coeffs_ref = eigenvecs_1_array[valid_indices]

    # Find the index of the maximum coefficient for each eigenfunction (dominant k)
    max_indices = np.argmax(np.abs(coeffs_ref.real), axis=1)

    # Sort eigenfunctions based on the dominant k mode (from small to large)
    sorted_order = np.argsort(max_indices)

    print(f"Dominant k indices: {max_indices[sorted_order]}")
    print(f"Corresponding eigenvalues: {eigenvals_valid[sorted_order].real}")

    # Step 3: Create bar chart plots
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis = coeffs_ref.shape[1]  # Number of basis functions

    # Create figure with subplots
    fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
    if N_plot == 1:
        axs = [axs]

    fig.suptitle(f"Coefficients sorted by dominant k", fontsize=10)

    # Plot each eigenfunction's coefficients
    for i in range(N_plot):
        plot_idx = sorted_order[i]  # Index in the filtered arrays
        global_idx = valid_indices[plot_idx]  # Index in the original arrays

        coefficients = coeffs_ref[plot_idx, :].real
        dominant_k = max_indices[plot_idx]
        eigenval = eigenvals_valid[plot_idx].real

        # Create bar plot
        k_values = np.arange(1, N_basis + 1)
        axs[i].bar(k_values, coefficients, width=0.4)
        axs[i].set_xlim(0, min(20, N_basis + 1))
        axs[i].set_ylim(-1, 1.)

        # Add title with dominant k and eigenvalue
        axs[i].set_title(f"λ={eigenval:.3f}, dominant k={dominant_k+1}",
                        fontsize=8, loc='right')

        # Highlight the dominant k
        axs[i].axvline(dominant_k + 1, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

    # Format axes
    for ax in axs:
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.label_outer()
        ax.grid(True, alpha=0.3, axis='y')

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1].set_xlabel(r"$k$ mode index", fontsize=8)
    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

    fig.tight_layout()

    # Save figure
    output_filename = f"multi_perturbation_coefficients_sorted.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved coefficient plot to {output_filename}")
    plt.show()

    return {
        'valid_indices': valid_indices,
        'sorted_order': sorted_order,
        'dominant_k': max_indices,
        'eigenvals_valid': eigenvals_valid
    }


# def plot_coefficients_by_dominant_k_pert_type(results, eigenvalue_threshold=0.95, N_plot=10,
#                                     reference_perturbation='vr'):
#     """
#     Plot coefficients of eigenfunctions sorted by dominant k mode

#     Parameters:
#     -----------
#     results : dict
#         Results dictionary from multi_perturbation_analysis
#     eigenvalue_threshold : float
#         Minimum eigenvalue to include (default: 0.95)
#     N_plot : int
#         Maximum number of eigenfunctions to plot
#     reference_perturbation : str
#         Which perturbation type to use for determining dominant k (default: 'vr')
#     """
#     eigenvals_1 = np.array(results['eigenvals_1'])
#     coefficients_1 = results['coefficients_1']

#     if reference_perturbation not in coefficients_1:
#         print(f"Error: {reference_perturbation} not found in coefficients")
#         print(f"Available perturbation types: {list(coefficients_1.keys())}")
#         return

#     # Step 1: Filter eigenvectors with eigenvalues > threshold
#     valid_mask = eigenvals_1.real > eigenvalue_threshold
#     eigenvals_valid = eigenvals_1[valid_mask]

#     print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
#     print(f"Found {len(eigenvals_valid)} eigenvectors above threshold out of {len(eigenvals_1)} total")

#     if len(eigenvals_valid) == 0:
#         print("No eigenvectors found above threshold. Try lowering the threshold.")
#         return

#     # Get valid indices
#     valid_indices = np.where(valid_mask)[0]

#     # Step 2: Sort by dominant k mode
#     # Extract coefficients for the reference perturbation
#     coeffs_ref = coefficients_1[reference_perturbation][valid_indices]

#     # Find the index of the maximum coefficient for each eigenfunction (dominant k)
#     max_indices = np.argmax(np.abs(coeffs_ref.real), axis=1)

#     # Sort eigenfunctions based on the dominant k mode (from small to large)
#     sorted_order = np.argsort(max_indices)

#     print(f"Dominant k indices: {max_indices[sorted_order]}")
#     print(f"Corresponding eigenvalues: {eigenvals_valid[sorted_order].real}")

#     # Step 3: Create bar chart plots
#     N_plot = min(N_plot, len(eigenvals_valid))
#     N_basis = coeffs_ref.shape[1]  # Number of basis functions

#     # Create figure with subplots
#     fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
#     if N_plot == 1:
#         axs = [axs]

#     fig.suptitle(f"Coefficients sorted by dominant k ({reference_perturbation})", fontsize=10)

#     # Plot each eigenfunction's coefficients
#     for i in range(N_plot):
#         plot_idx = sorted_order[i]  # Index in the filtered arrays
#         global_idx = valid_indices[plot_idx]  # Index in the original arrays

#         coefficients = coeffs_ref[plot_idx, :].real
#         dominant_k = max_indices[plot_idx]
#         eigenval = eigenvals_valid[plot_idx].real

#         # Create bar plot
#         k_values = np.arange(1, N_basis + 1)
#         axs[i].bar(k_values, coefficients, width=0.4)
#         axs[i].set_xlim(0, min(20, N_basis + 1))
#         axs[i].set_ylim(-1, 1.)

#         # Add title with dominant k and eigenvalue
#         axs[i].set_title(f"λ={eigenval:.3f}, dominant k={dominant_k+1}",
#                         fontsize=8, loc='right')

#         # Highlight the dominant k
#         axs[i].axvline(dominant_k + 1, color='red', linestyle='--',
#                       alpha=0.5, linewidth=1)

#     # Format axes
#     for ax in axs:
#         ax.xaxis.set_tick_params(labelsize=8)
#         ax.yaxis.set_tick_params(labelsize=8)
#         ax.label_outer()
#         ax.grid(True, alpha=0.3, axis='y')

#     axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[-1].set_xlabel(r"$k$ mode index", fontsize=8)
#     fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

#     fig.tight_layout()

#     # Save figure
#     output_filename = f"multi_perturbation_coefficients_{reference_perturbation}_sorted.pdf"
#     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#     print(f"\nSaved coefficient plot to {output_filename}")
#     plt.show()

#     return {
#         'valid_indices': valid_indices,
#         'sorted_order': sorted_order,
#         'dominant_k': max_indices,
#         'eigenvals_valid': eigenvals_valid
#     }

if __name__ == "__main__":
    # Run the analysis
    folder_path = f'./data/'
    allowedK = np.load(folder_path + 'data_allowedK/L70_kvalues.npy')
    results = multi_perturbation_analysis(N=len(allowedK), N_t=1000)

    # Plot results
    if len(results['eigenvals_1']) > 0:
        plot_multi_perturbation_results(results, N_plot=5)

        # Plot coefficients sorted by dominant k mode
        print("\n" + "="*80)
        print("Plotting coefficients sorted by dominant k mode")
        print("="*80)
        sorting_results = plot_coefficients_by_dominant_k(
            results,
            eigenvalue_threshold=0.9,
            N_plot=6
        )
    else:
        print("No eigenvalues found for plotting")

    print("\nAnalysis complete!")
    print(f"Combined M matrix shape: {results['M_combined'].shape}")
    print(f"Number of common eigenfunctions found: {len(results['eigenvals_1'])}")