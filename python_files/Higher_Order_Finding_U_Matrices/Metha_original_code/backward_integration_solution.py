# -*- coding: utf-8 -*-
"""
Backward Integration Solution for Palindromic Universe Perturbations

This script implements the backward integration method:
1. Solve x^∞ using Higher_Order_Solving_for_Vrinf.py approach
2. Map x^∞ to x', y'_{2:4} using equation (25) 
3. Integrate backward from endtime to recombination to get x*, y*
4. Integrate backward from recombination to Big Bang using perfect fluid ODE

This avoids connection issues at the end of the universe by ensuring 
proper boundary conditions through backward integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# 1. SETUP: Parameters and Functions (from Metha's original code)
# =============================================================================

print("--- Setting up parameters and functions ---")

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

# ODE systems for backward integration
def dX_boltzmann_s_backward(t, X, k):
    """Backward integration in s-parameterization (endtime to swaptime)"""
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
    
    # For l>2 terms
    for j in range(8,num_variables-1):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    
    # Final term
    lastderiv = k*X[(num_variables-1)-1] - (((num_variables-1)-5 + 1)*X[(num_variables-1)])/t
    derivatives.append(lastderiv)
    return derivatives

def dX_boltzmann_sigma_backward(t, X, k):
    """Backward integration in sigma-parameterization (swaptime to recombination)"""
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
    
    # For l>2 terms
    for j in range(8,num_variables-1):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    
    # Final term
    lastderiv = k*X[(num_variables-1)-1] - (((num_variables-1)-5 + 1)*X[(num_variables-1)])/t
    derivatives.append(lastderiv)
    return derivatives

def dX_perfect_sigma_backward(t, X, k):
    """Backward integration with perfect fluid (recombination to Big Bang)"""
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
# 2. LOAD TRANSFORMATION MATRICES
# =============================================================================

print("\n--- Loading transformation matrices ---")

try:
    kvalues = np.load('L70_kvalues.npy')
    ABCmatrices = np.load('L70_ABCmatrices.npy')
    DEFmatrices = np.load('L70_DEFmatrices.npy')
    GHIvectors = np.load('L70_GHIvectors.npy')
    X1matrices = np.load('L70_X1matrices.npy')
    X2matrices = np.load('L70_X2matrices.npy')
    recValues = np.load('L70_recValues.npy')
    allowedK = np.load('allowedK.npy')
    print("All transformation matrices loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Extract A and D matrices 
Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

# Create interpolation functions
def create_matrix_interpolator(k_grid, matrix_array):
    rows, cols = matrix_array.shape[1], matrix_array.shape[2]
    interp_funcs = [[interp1d(k_grid, matrix_array[:, i, j], bounds_error=False, fill_value="extrapolate") for j in range(cols)] for i in range(rows)]
    def get_matrix(k):
        return np.array([[func(k) for func in row] for row in interp_funcs])
    return get_matrix

get_A = create_matrix_interpolator(kvalues, Amatrices)
get_D = create_matrix_interpolator(kvalues, Dmatrices)
get_X1 = create_matrix_interpolator(kvalues, X1matrices)
get_X2 = create_matrix_interpolator(kvalues, X2matrices)

# =============================================================================
# 3. BACKWARD INTEGRATION METHOD
# =============================================================================

print("\n--- Applying backward integration method ---")

def solve_x_infinity(k):
    """STEP 1: Solve for x^∞ using the method from Higher_Order_Solving_for_Vrinf.py"""
    
    # Get interpolated matrices for this k
    A = get_A(k)
    D = get_D(k)
    X1 = get_X1(k)
    X2 = get_X2(k)
    
    # Get GHI vector through interpolation
    GHI_interp = interp1d(kvalues, np.array([GHIvectors[j][0:6] for j in range(len(kvalues))]).T, 
                         bounds_error=False, fill_value="extrapolate")
    GX3 = np.zeros((6,4))
    GX3[:,2] = GHI_interp(k)
    
    # Get recombination values from interpolation
    rec_interp = interp1d(kvalues, recValues, axis=0, bounds_error=False, fill_value="extrapolate")
    recs_vec = rec_interp(k)
    
    # Build matrix equation: (A*X1 + D*X2 + G*X3) * x^∞ = x*_rec
    # Use rows 2:6 corresponding to [dr, dm, vr, vm]
    M_matrix = (A @ X1 + D @ X2 + GX3)[2:6, :]
    x_rec_subset = recs_vec[2:6]  # [dr, dm, vr, vm] at recombination
    
    try:
        x_inf = np.linalg.solve(M_matrix, x_rec_subset)
        return x_inf
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Matrix is singular.")
        return None

def map_xinf_to_endtime(k, x_inf):
    """STEP 2: Use equation (25) to map x^∞ to x', y'_{2:4}"""
    
    X1 = get_X1(k)
    X2 = get_X2(k)
    
    # Equation (25): x' = X1 * x^∞, y'_{2:4} = X2 * x^∞
    x_prime = X1 @ x_inf
    y_prime_2_4 = X2 @ x_inf
    
    return x_prime, y_prime_2_4

def backward_integrate_to_recombination(k, x_prime, y_prime_2_4):
    """STEP 3: Integrate backward from endtime to recombination"""
    
    # Background value at endtime
    s_endtime = np.interp(endtime, sol.t, sol.y[0])
    
    # Create full state vector at endtime
    Y_endtime = np.zeros(num_variables)
    Y_endtime[0] = s_endtime
    Y_endtime[1:7] = x_prime  # [phi, psi, dr, dm, vr, vm]
    Y_endtime[7:9] = y_prime_2_4  # [F_2, F_3]
    # Higher order terms remain zero
    
    print(f"Initial conditions at endtime: s={s_endtime:.6f}, phi={x_prime[0]:.6f}, vr={x_prime[4]:.6f}")
    
    # Phase 1: Backward integration from endtime to swaptime (s-evolution)
    sol_back1 = solve_ivp(dX_boltzmann_s_backward, [endtime, swaptime], Y_endtime, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Phase 2: Backward integration from swaptime to recombination (sigma-evolution)
    Y_swap = sol_back1.y[:, -1].copy()
    Y_swap[0] = np.log(Y_swap[0])  # Convert s to sigma
    
    sol_back2 = solve_ivp(dX_boltzmann_sigma_backward, [swaptime, recConformalTime], Y_swap,
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Extract x* and y* at recombination
    Y_rec_sigma = sol_back2.y[:, -1]
    Y_rec = Y_rec_sigma.copy()
    Y_rec[0] = np.exp(Y_rec_sigma[0])  # Convert sigma back to s
    
    x_star = Y_rec[1:7]  # [phi, psi, dr, dm, vr, vm]
    y_star_2_4 = Y_rec[7:9]  # [F_2, F_3]
    
    # Combine backward integration results
    t_backward = np.concatenate((sol_back1.t, sol_back2.t))
    Y_backward_raw = np.concatenate((sol_back1.y, sol_back2.y), axis=1)
    
    # Convert sigma back to s for the second phase
    Y_backward = Y_backward_raw.copy()
    mask = t_backward >= swaptime
    Y_backward[0, mask] = np.exp(Y_backward_raw[0, mask])
    
    return x_star, y_star_2_4, t_backward, Y_backward

def backward_integrate_to_bigbang(k, x_star):
    """STEP 4: Integrate backward from recombination to Big Bang with perfect fluid"""
    
    # Perfect fluid initial conditions at recombination
    # Convert to sigma parameterization for perfect fluid integration
    s_rec_val = np.interp(recConformalTime, sol.t, sol.y[0])
    sigma_rec = np.log(s_rec_val)
    
    # x_star = [phi, psi, dr, dm, vr, vm] - for perfect fluid, psi = phi
    X_rec_perfect = [sigma_rec, x_star[0], x_star[2], x_star[3], x_star[4], x_star[5]]
    
    # Backward integration using perfect fluid equations
    sol_perfect = solve_ivp(dX_perfect_sigma_backward, [recConformalTime, t0], X_rec_perfect,
                           dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    return sol_perfect

# =============================================================================
# 4. MAIN COMPUTATION: Apply Backward Integration Method
# =============================================================================

solutions = []
num_modes_to_plot = min(3, len(allowedK))

for i in range(num_modes_to_plot):
    k = allowedK[i+2]  # Skip first few modes
    print(f"\n=== Processing mode n={i+1} with k={k:.6f} ===")
    
    # STEP 1: Solve x^∞ 
    print("Step 1: Solving for x^∞...")
    x_inf = solve_x_infinity(k)
    if x_inf is None:
        print(f"Skipping mode k={k} due to singular matrix")
        continue
    print(f"x^∞ = {x_inf}")
    
    # STEP 2: Map x^∞ to endtime conditions
    print("Step 2: Mapping x^∞ to x', y'_{2:4}...")
    x_prime, y_prime_2_4 = map_xinf_to_endtime(k, x_inf)
    print(f"x' = {x_prime[:3]}... (first 3 components)")
    print(f"y'_{2:4} = {y_prime_2_4}")
    
    # STEP 3: Backward integrate from endtime to recombination
    print("Step 3: Backward integration from endtime to recombination...")
    x_star, y_star_2_4, t_back_post_rec, Y_back_post_rec = backward_integrate_to_recombination(k, x_prime, y_prime_2_4)
    print(f"x* at recombination = {x_star[:3]}... (first 3 components)")
    
    # STEP 4: Backward integrate from recombination to Big Bang  
    print("Step 4: Backward integration from recombination to Big Bang...")
    sol_perfect = backward_integrate_to_bigbang(k, x_star)
    
    # Combine all backward integration results (Big Bang to endtime)
    t_perfect = sol_perfect.t
    t_left = np.concatenate((t_perfect, t_back_post_rec))
    
    # Expand perfect fluid solution to full variable space
    Y_perfect_full = np.zeros((num_variables, len(t_perfect)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])  # Convert sigma to s
    Y_perfect_full[1,:] = sol_perfect.y[1,:]  # phi
    Y_perfect_full[2,:] = sol_perfect.y[1,:]  # psi = phi for perfect fluid
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]  # dr, dm, vr, vm
    # F_l terms remain zero during perfect fluid phase
    
    Y_left = np.concatenate((Y_perfect_full, Y_back_post_rec), axis=1)
    
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
    print(f"Mode n={i+1} completed successfully")

# =============================================================================
# 5. PLOTTING
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
fig.suptitle('Backward Integration Solutions (Palindromic Universe)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('backward_integration_solutions.pdf')
plt.show()

print("\n--- Backward integration method implementation complete ---")
print("Solutions saved to backward_integration_solutions.pdf")