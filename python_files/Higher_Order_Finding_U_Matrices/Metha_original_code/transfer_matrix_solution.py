# -*- coding: utf-8 -*-
"""
Transfer Matrix Solution for Palindromic Universe Perturbations

This script implements the 5-step transfer matrix method from 
"Evidence for a Palindromic Universe" to avoid numerical discontinuities:

1. Calculate x* and y*_{2:4} to recombination using Higher_Order_Finding_Xrecs.py
2. Solve x^∞ using Higher_Order_Solving_for_Vrinf.py  
3. Map x^∞ to x', y'_{2:4} using equation (25)
4. Calculate y*_{4:} using equation (24) and transformation functions
5. Integrate forward from recombination to end using complete solution

This eliminates the problematic stitching that causes numerical peaks.
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

# ODE system for perfect fluid evolution (sigma parameterization)
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

# ODE system for full Boltzmann hierarchy (sigma parameterization)
def dX_boltzmann_sigma(t, X, k):
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

# =============================================================================
# 2. STEP 1: Calculate x* and y*_{2:4} to recombination (from Higher_Order_Finding_Xrecs.py)
# =============================================================================

print("\n--- STEP 1: Calculating solutions to recombination time ---")

# Load allowed k values
try:
    allowedK = np.load('allowedK.npy')
    print(f"Loaded {len(allowedK)} allowed k values")
except FileNotFoundError:
    print("allowedK.npy not found. Please run Higher_Order_Solving_for_Vrinf.py first.")
    exit()

# Function to calculate x* and y*_{2:4} for each allowed k
def calculate_xstar_ystar(k):
    """
    Calculate x* and y*_{2:4} to recombination time using perfect fluid evolution
    This follows the approach in Higher_Order_Finding_Xrecs.py
    """
    
    # Perfect fluid initial conditions (from Higher_Order_Finding_Xrecs.py)
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))
    
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda)
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda)
    
    vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda)
    vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)), (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda)
    
    # Set initial conditions at Big Bang
    sigma0 = np.log(s0)
    phi0 = 1 + phi1*t0 + phi2*t0**2
    dr0 = -2 + dr1*t0 + dr2*t0**2
    dm0 = -1.5 + dm1*t0 + dm2*t0**2
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3
    
    X0 = [sigma0, phi0, dr0, dm0, vr0, vm0]
    
    # Integrate perfect fluid equations to recombination
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0,recConformalTime], X0, 
                           method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Extract final values (x* and conversion to y*_{2:4})
    sigma_rec = sol_perfect.y[0,-1]
    s_rec_val = np.exp(sigma_rec)
    phi_rec = sol_perfect.y[1,-1]
    dr_rec = sol_perfect.y[2,-1]
    dm_rec = sol_perfect.y[3,-1]
    vr_rec = sol_perfect.y[4,-1]
    vm_rec = sol_perfect.y[5,-1]
    
    # x* = [phi, psi, dr, dm, vr, vm] at recombination
    # For perfect fluid: psi = phi
    x_star = np.array([phi_rec, phi_rec, dr_rec, dm_rec, vr_rec, vm_rec])
    
    # y*_{2:4} = [F_2, F_3] at recombination (zero for perfect fluid)
    y_star_2_4 = np.array([0.0, 0.0])
    
    return x_star, y_star_2_4, s_rec_val

# =============================================================================
# 3. STEP 2: Solve x^∞ (from Higher_Order_Solving_for_Vrinf.py)
# =============================================================================

print("\n--- STEP 2: Loading x^∞ solutions ---")

# Load pre-computed transformation matrices
try:
    kvalues = np.load('L70_kvalues.npy')
    ABCmatrices = np.load('L70_ABCmatrices.npy')
    DEFmatrices = np.load('L70_DEFmatrices.npy')
    GHIvectors = np.load('L70_GHIvectors.npy')
    X1matrices = np.load('L70_X1matrices.npy')
    X2matrices = np.load('L70_X2matrices.npy')
    recValues = np.load('L70_recValues.npy')
    print("All transformation matrices loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading transformation matrices: {e}")
    print("Please run Higher_Order_Finding_U_Matrices.py first.")
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

def solve_x_infinity(k, x_star):
    """
    Solve for x^∞ using the transfer matrix approach from Higher_Order_Solving_for_Vrinf.py
    """
    
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
    
    # Build matrix equation: (A*X1 + D*X2 + G*X3) * x^∞ = x*
    # Use rows 2:6 corresponding to [dr, dm, vr, vm]
    M_matrix = (A @ X1 + D @ X2 + GX3)[2:6, :]
    x_star_subset = x_star[2:6]  # [dr, dm, vr, vm] components
    
    try:
        x_inf = np.linalg.solve(M_matrix, x_star_subset)
        return x_inf
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Matrix is singular.")
        return None

# =============================================================================
# 4. STEP 3 & 4: Apply Equations (25) and (24)
# =============================================================================

def apply_transfer_matrix_mapping(k, x_inf):
    """
    Steps 3 & 4: Use equations (25) and (24) to map x^∞ to initial conditions
    
    Equation (25): x' = X1 * x^∞  (maps FCB to eta')
    Equation (24): y*_{4:} = X2 * x^∞  (maps FCB to higher order terms)
    """
    
    X1 = get_X1(k)
    X2 = get_X2(k)
    
    # Equation (25): x' = X1 * x^∞
    x_prime = X1 @ x_inf
    
    # Equation (24): y*_{4:} = X2 * x^∞  
    y_star_4_plus = X2 @ x_inf
    
    return x_prime, y_star_4_plus

# =============================================================================
# 5. STEP 5: Forward Integration with Complete Solution
# =============================================================================

def integrate_complete_solution(k, x_star, y_star_2_4, x_prime, y_star_4_plus):
    """
    Step 5: Integrate forward from recombination to end using complete solution
    
    This uses x* and y*_{2:4} from step 1 and y*_{4:} from step 4
    to create a complete initial condition at recombination time,
    then integrates forward to the end of the universe.
    """
    
    # Create complete state vector at recombination
    Y_rec = np.zeros(num_variables)
    
    # Background value at recombination
    _, _, s_rec_val = calculate_xstar_ystar(k)
    Y_rec[0] = s_rec_val  # s
    Y_rec[1:7] = x_star   # [phi, psi, dr, dm, vr, vm]
    Y_rec[7:9] = y_star_2_4  # [F_2, F_3] = [0, 0] for perfect fluid
    
    # The key insight: use y*_{4:} from transfer matrix for higher order terms
    # y*_{4:} represents [F_2, F_3] from the transfer matrix calculation
    Y_rec[7:9] = y_star_4_plus  # Replace with transfer matrix values
    
    # Convert to sigma parameterization for integration
    Y_rec_sigma = Y_rec.copy()
    Y_rec_sigma[0] = np.log(Y_rec[0])  # Convert s to sigma
    
    # Forward integration from recombination through swaptime to endtime
    sol_forward1 = solve_ivp(dX_boltzmann_sigma, [recConformalTime, swaptime], Y_rec_sigma,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Convert back to s parameterization for final phase
    Y_swap = sol_forward1.y[:, -1].copy()
    Y_swap[0] = np.exp(Y_swap[0])  # Convert sigma back to s
    
    # ODE system for s-evolution (from swaptime to endtime)
    def dX_boltzmann_s(t, X, k):
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
    
    # Forward integration from swaptime to endtime
    sol_forward2 = solve_ivp(dX_boltzmann_s, [swaptime, endtime], Y_swap,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # Combine the solutions
    t_forward = np.concatenate((sol_forward1.t, sol_forward2.t))
    Y_forward_raw = np.concatenate((sol_forward1.y, sol_forward2.y), axis=1)
    
    # Convert sigma back to s for the first phase
    Y_forward = Y_forward_raw.copy()
    mask = t_forward <= swaptime
    Y_forward[0, mask] = np.exp(Y_forward_raw[0, mask])
    
    return t_forward, Y_forward

# =============================================================================
# 6. MAIN COMPUTATION: Apply 5-Step Method
# =============================================================================

print("\n--- Applying 5-step transfer matrix method ---")

solutions = []
num_modes_to_plot = min(3, len(allowedK))

for i in range(num_modes_to_plot):
    k = allowedK[i+2]  # Skip first few modes
    print(f"\nProcessing mode n={i+1} with k={k:.6f}")
    
    # STEP 1: Calculate x* and y*_{2:4} to recombination
    x_star, y_star_2_4, _ = calculate_xstar_ystar(k)
    print(f"Step 1 complete: x* calculated to recombination")
    
    # STEP 2: Solve x^∞ 
    x_inf = solve_x_infinity(k, x_star)
    if x_inf is None:
        print(f"Skipping mode k={k} due to singular matrix")
        continue
    print(f"Step 2 complete: x^∞ = {x_inf}")
    
    # STEPS 3 & 4: Apply transfer matrix mapping
    x_prime, y_star_4_plus = apply_transfer_matrix_mapping(k, x_inf)
    print(f"Steps 3&4 complete: Transfer matrix mapping applied")
    
    # STEP 5: Forward integration with complete solution
    t_left, Y_left = integrate_complete_solution(k, x_star, y_star_2_4, x_prime, y_star_4_plus)
    print(f"Step 5 complete: Forward integration finished")
    
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

# =============================================================================
# 7. PLOTTING
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
fig.suptitle('Transfer Matrix Solutions (No Stitching Discontinuities)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('transfer_matrix_solutions.pdf')
plt.show()

print("\n--- Transfer matrix method implementation complete ---")
print("Solutions saved to transfer_matrix_solutions.pdf")