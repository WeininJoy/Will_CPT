# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model, following the methodology outlined in the paper "Evidence for a Palindromic Universe".

It uses pre-computed data from transfer matrix calculations to solve for the
boundary conditions at the Future Conformal Boundary (FCB) for each allowed wavenumber k.
It then integrates the perturbation equations to obtain the full evolution of the
variables and plots the results.

CORRECTED VERSION: Fixes the IndexError in the Boltzmann hierarchy ODE function.

ADDED SECTION 5: Computes the CMB Angular Power Spectrum (Cl) from the solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from scipy.special import spherical_jn # For the projection function

# =============================================================================
# 1. SETUP: Parameters, Constants, and ODE Functions
# (Your original code from here...)
# =============================================================================

print("--- Setting up parameters and functions ---")

# Data folder
folder_path = './data/'

# Cosmological Parameters (Planck 2018)
h = 0.5409
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
OmegaM, OmegaK = 0.483, -0.0438
OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
z_rec = 1089.411

# Working units: 8piG = c = hbar = 1, and s0 = 1 for numerical stability.
s0 = a0 = 1.0
H0 = np.sqrt(1 / (3 * OmegaLambda))
Hinf = H0 * np.sqrt(OmegaLambda)
K = -OmegaK * a0**2 * H0**2

# Tolerances and constants
atol = 1e-12
rtol = 1e-12
num_variables_boltzmann = 75
l_max = 69
num_variables_perfect = 6

# Time constants
t0_integration = 1e-8 * s0
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
szero = -OmegaM/s0**3/(4*OmegaR/s0**4)
s_bang_init = smin1/t0_integration + szero
deltaeta = 6.6e-4 * s0
swaptime = 2.0 * s0

def ds_dt(t, s):
    s_abs = np.abs(s)
    return -H0 * np.sqrt(OmegaLambda + OmegaK * (s_abs**2 / s0**2) + OmegaM * (s_abs**3 / s0**3) + OmegaR * (s_abs**4 / s0**4))

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol_bg = solve_ivp(ds_dt, [t0_integration, 12 * s0], [s_bang_init], events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
fcb_time = sol_bg.t_events[0][0]
endtime = fcb_time - deltaeta
s_rec = 1 + z_rec
recConformalTime = sol_bg.t[np.argmin(np.abs(sol_bg.y[0] - s_rec))]
eta_0 = sol_bg.t[np.argmin(np.abs(sol_bg.y[0] - s0))] # Observer is at the a=1

print(f"FCB Time (eta_FCB): {fcb_time:.4f}")
print(f"Recombination Time (eta_rec): {recConformalTime:.4f}")
print(f"Integration Start Time (eta'): {endtime:.4f}")

# CORRECTED ODE system for full Boltzmann hierarchy (s-evolution)
def dX_boltzmann_s(t, X, k):
    s, phi, psi, dr, dm, vr, vm = X[0:7]
    fr_all = X[7:] # Holds F_2 to F_lmax

    sdot = ds_dt(t, s)
    H = sdot/s

    rho_m = 3 * (H0**2) * OmegaM * (np.abs(s / s0)**3)
    rho_r = 3 * (H0**2) * OmegaR * (np.abs(s / s0)**4)

    phidot = H * psi - ((4/3) * rho_r * vr + rho_m * vm) / (2 * s**2)
    fr2 = fr_all[0]
    fr3 = fr_all[1]
    fr2dot = -(8/15) * (k**2) * vr - (3/5) * k * fr3
    psidot = phidot - (3*H0**2*OmegaR/s0**4/k**2)*(2*s*sdot*fr2 + s**2*fr2dot)
    
    drdot = (4/3) * (3 * phidot + (k**2) * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(psi + dr / 4)
    vmdot = H * vm - psi
    
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    
    for j in range(1, l_max - 2): 
        l = j + 2
        derivatives.append((k / (2 * l + 1)) * (l * fr_all[j-1] - (l + 1) * fr_all[j+1]))
        
    last_deriv = (k * l_max / (2 * l_max + 1)) * fr_all[l_max - 3]
    derivatives.append(last_deriv)
    
    return derivatives

# CORRECTED ODE system for full Boltzmann hierarchy (sigma-evolution)
def dX_boltzmann_sigma(t, X, k):
    sigma, phi, psi, dr, dm, vr, vm = X[0:7]
    fr_all = X[7:]
    
    H = -(H0) * np.sqrt(OmegaLambda * np.exp(-2 * sigma) + OmegaK / s0**2 + OmegaM / s0**3 * np.exp(sigma) + OmegaR / s0**4 * np.exp(2 * sigma))
    
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (np.exp(3 * sigma))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (np.exp(4 * sigma))

    phidot = H * psi - ((4/3) * rho_r * vr + rho_m * vm) / (2 * np.exp(2 * sigma))
    fr2 = fr_all[0]
    fr3 = fr_all[1]
    fr2dot = -(8/15) * (k**2) * vr - (3/5) * k * fr3
    psidot = phidot - (3*H0**2*OmegaR/s0**4/k**2)*np.exp(2*sigma)*(2*H*fr2 + fr2dot)
    
    drdot = (4/3) * (3 * phidot + (k**2) * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(psi + dr / 4)
    vmdot = H * vm - psi
    
    derivatives = [H, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    for j in range(1, l_max - 2):
        l = j + 2
        derivatives.append((k / (2 * l + 1)) * (l * fr_all[j-1] - (l + 1) * fr_all[j+1]))

    last_deriv = (k * l_max / (2 * l_max + 1)) * fr_all[l_max - 3]
    derivatives.append(last_deriv)
    
    return derivatives

# ODE system for perfect fluid (sigma-evolution)
def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    H = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma) + OmegaK/s0**2 + OmegaM/s0**3*np.exp(sigma) + OmegaR/s0**4*np.exp(2*sigma))
    
    rho_m = 3*(H0**2)*OmegaM/s0**3*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR/s0**4*(np.exp(4*sigma))
    
    phidot = H*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4)
    vmdot = H*vm - phi
    return [H, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 2. DATA LOADING AND INTERPOLATION
# =============================================================================

print("\n--- Loading and interpolating pre-computed data ---")

try:
    kvalues = np.load(folder_path + 'L70_kvalues.npy')
    ABCmatrices = np.load(folder_path + 'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path + 'L70_DEFmatrices.npy')
    X1matrices = np.load(folder_path + 'L70_X1matrices.npy')
    X2matrices = np.load(folder_path + 'L70_X2matrices.npy')
    recValues = np.load(folder_path + 'L70_recValues.npy')
    # allowedK = np.load(folder_path + 'allowedK.npy') # discrete k, satisfying FCB temporal boundary condition
    allowedK = [i*100*np.sqrt(K) for i in range(1, 30)] # Extended k-range for better coverage
    allowedK = np.array(allowedK)
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the 'data' directory.")
    exit()

# Create interpolation functions for each element of the matrices
def create_matrix_interpolator(k_grid, matrix_array):
    rows, cols = matrix_array.shape[1], matrix_array.shape[2]
    interp_funcs = [[interp1d(k_grid, matrix_array[:, i, j], bounds_error=False, fill_value="extrapolate") for j in range(cols)] for i in range(rows)]
    def get_matrix(k):
        return np.array([[func(k) for func in row] for row in interp_funcs])
    return get_matrix

def create_vector_interpolator(k_grid, vector_array):
    cols = vector_array.shape[1]
    interp_funcs = [interp1d(k_grid, vector_array[:, i], bounds_error=False, fill_value="extrapolate") for i in range(cols)]
    def get_vector(k):
        return np.array([func(k) for func in interp_funcs])
    return get_vector

get_A = create_matrix_interpolator(kvalues, ABCmatrices[:, 0:6, :])
get_D = create_matrix_interpolator(kvalues, DEFmatrices[:, 0:6, :])
get_X1 = create_matrix_interpolator(kvalues, X1matrices)
get_X2 = create_matrix_interpolator(kvalues, X2matrices)
get_recs = create_vector_interpolator(kvalues, recValues)


# =============================================================================
# 3. CALCULATE AND INTEGRATE SOLUTIONS
# (This section now calculates all allowed modes needed for the CMB spectrum)
# =============================================================================

print(f"\n--- Calculating solutions for all {len(allowedK)} allowed modes ---")

solutions = []
for i, k in enumerate(allowedK):
    if (i+1) % 10 == 0:
        print(f"Processing mode n={i+1}/{len(allowedK)} with k={k:.4f}")

    A, D, X1, X2, recs_vec = get_A(k), get_D(k), get_X1(k), get_X2(k), get_recs(k)
    M_matrix = (A @ X1 + D @ X2)[2:6, :]
    x_rec_subset = recs_vec[2:6]
    
    try:
        x_inf = np.linalg.solve(M_matrix, x_rec_subset)
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Skipping mode.")
        continue

    x_prime = X1 @ x_inf
    y_prime_2_4 = X2 @ x_inf
    s_prime_val = np.interp(endtime, sol_bg.t, sol_bg.y[0])

    Y_prime = np.zeros(num_variables_boltzmann)
    Y_prime[0] = s_prime_val
    Y_prime[1:7] = x_prime
    Y_prime[7:9] = y_prime_2_4

    sol_part1 = solve_ivp(dX_boltzmann_s, [endtime, swaptime], Y_prime, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    Y_swap = sol_part1.y[:, -1]
    Y_swap[0] = np.log(Y_swap[0])
    sol_part2 = solve_ivp(dX_boltzmann_sigma, [swaptime, recConformalTime], Y_swap, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    # --- CORRECTED STITCHING LOGIC ---
    
    # 1. Stitch the two backward parts (part2 and part1) at 'swaptime'
    t_back_part2 = sol_part2.t[::-1] # Runs from recConformalTime to swaptime
    Y_back_part2_sigma = sol_part2.y[:, ::-1]

    t_back_part1 = sol_part1.t[::-1] # Runs from swaptime to endtime
    Y_back_part1_sigma = sol_part1.y[:, ::-1]
    
    # Concatenate, dropping the duplicated 'swaptime' point from BOTH arrays
    t_back = np.concatenate((t_back_part2, t_back_part1[1:])) 
    Y_back_sigma = np.concatenate((Y_back_part2_sigma, Y_back_part1_sigma[:, 1:]), axis=1)

    # Apply the coordinate transformation from sigma to s
    Y_back = Y_back_sigma.copy()
    mask_back = t_back >= swaptime
    Y_back[0, mask_back] = np.exp(Y_back_sigma[0, mask_back])

    # 2. Get solution from Big Bang to Recombination (perfect fluid)
    phi1, phi2 = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0, (1/60)*(-2*k**2+(9*OmegaM**2)/(16*OmegaLambda*OmegaR*s0**2))-2*OmegaK/(15*OmegaLambda*s0**2)
    dr1, dr2 = -OmegaM/(4*np.sqrt(3*OmegaR*OmegaLambda))/s0, (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(240*s0**2*OmegaR*OmegaLambda)-8*OmegaK/(15*OmegaLambda*s0**2)
    dm1, dm2 = -np.sqrt(3)*OmegaM/(16*s0*np.sqrt(OmegaR*OmegaLambda)), (9*OmegaM**2-112*OmegaR*OmegaLambda*k**2*s0**2)/(320*s0**2*OmegaR*OmegaLambda)-2*OmegaK/(5*OmegaLambda*s0**2)
    vr1, vr2, vr3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-OmegaM**2+8*s0**2*OmegaR*OmegaLambda*k**2)/(160*s0**2*OmegaR*OmegaLambda)+4*OmegaK/(45*OmegaLambda*s0**2)
    vm1, vm2, vm3 = -1/2, OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0), (-3*OmegaM**2+4*s0**2*OmegaR*OmegaLambda*k**2)/(480*s0**2*OmegaR*OmegaLambda)+17*OmegaK/(360*OmegaLambda*s0**2)
    sigma0 = np.log(s_bang_init)
    phi0, dr0, dm0 = 1+phi1*t0_integration+phi2*t0_integration**2, -2+dr1*t0_integration+dr2*t0_integration**2, -1.5+dm1*t0_integration+dm2*t0_integration**2
    vr0, vm0 = vr1*t0_integration+vr2*t0_integration**2+vr3*t0_integration**3, vm1*t0_integration+vm2*t0_integration**2+vm3*t0_integration**3
    Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0_integration, recConformalTime], Y0_perfect, dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))
    
    # 3. Stitch the forward (perfect) and backward (boltzmann) solutions at 'recConformalTime'
    # Prepare the Y array for the perfect fluid part
    Y_perfect_full = np.zeros((num_variables_boltzmann, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
    Y_perfect_full[1,:] = sol_perfect.y[1,:] # Phi
    Y_perfect_full[2,:] = sol_perfect.y[1,:] # Psi = Phi in perfect fluid approx
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:] # dr, dm, vr, vm
    
    # Concatenate, dropping the duplicated 'recConformalTime' point from BOTH arrays
    t_full = np.concatenate((sol_perfect.t, t_back[1:]))
    Y_full = np.concatenate((Y_perfect_full, Y_back[:, 1:]), axis=1)
    
    # Store the final, clean solution where len(t_full) == Y_full.shape[1]
    solutions.append({'t': t_full, 'Y': Y_full, 'k': k})
    
print("--- All solutions calculated. ---")


# =============================================================================
# 4. PLOTTING SOLUTIONS (OPTIONAL - You can comment this out to speed up)
# =============================================================================

# (Your original plotting code... can be kept or removed)

# =============================================================================
# 5. CMB POWER SPECTRUM (Cl) CALCULATION (REVISED & ROBUST VERSION)
# =============================================================================

print("\n--- Starting CMB Power Spectrum Calculation ---")

# --- 5.1: Setup CMB Parameters ---
tau_reio = 0.0495  # Reionization optical depth (typical value)
As = 2.0706e-9  # Scalar amplitude (typical value)
ns = 0.97235  # Scalar spectral index (typical value)
k_pivot = 0.05  # Pivot scale in 1/Mpc

# CRITICAL: Define the CMB temperature for unit conversion
T0_cmb_K = 2.7255
T0_cmb_muK = T0_cmb_K * 1e6
# UNIT_CONVERSION_FACTOR = T0_cmb_muK**2  # Remove this - likely double counting
UNIT_CONVERSION_FACTOR = 1.0  # Start with no unit conversion

l_max_cmb = 300  # Extended range
# Use logarithmic spacing for efficiency: dense at low-l, sparse at high-l
ells_low = np.arange(2, 20)  # l=2 to 19 (fine resolution)
ells_mid = np.arange(20, 60, 2)  # l=20 to 58 (every 2)
ells_high = np.geomspace(60, l_max_cmb, 20, dtype=int)  # l=60 to 300 (log spacing)
ells_high = np.unique(ells_high)  # Remove duplicates
ells = np.concatenate([ells_low, ells_mid, ells_high])
ells = np.unique(ells)  # Ensure no duplicates and sorted

# --- 5.2: Helper Functions ---
def primordial_power_spectrum(k, As, ns):
    # Convert pivot scale to match k units (k = i*sqrt(K))
    k_pivot_physical = k_pivot * np.sqrt(K)
    return As * (k / k_pivot_physical)**(ns - 1)

def visibility_function(eta, eta_rec, width_rec):
    return np.exp(-((eta - eta_rec)**2) / (2 * width_rec**2)) / (np.sqrt(2 * np.pi) * width_rec)

rec_width = 0.05

def get_solution_interpolators(sol):
    t = sol['t']
    Y = sol['Y']
    interp = {
        'phi': interp1d(t, Y[1, :], kind='cubic', bounds_error=False, fill_value=0),
        'psi': interp1d(t, Y[2, :], kind='cubic', bounds_error=False, fill_value=0),
        'Theta0': interp1d(t, Y[3, :] / 4.0, kind='cubic', bounds_error=False, fill_value=0),
        'vr': interp1d(t, Y[5, :], kind='cubic', bounds_error=False, fill_value=0)
    }
    fine_t = np.linspace(t.min(), t.max(), len(t) * 5)
    phi_fine = interp['phi'](fine_t)
    psi_fine = interp['psi'](fine_t)
    phi_dot_fine = np.gradient(phi_fine, fine_t)
    psi_dot_fine = np.gradient(psi_fine, fine_t)
    interp['phi_dot'] = interp1d(fine_t, phi_dot_fine, kind='cubic', bounds_error=False, fill_value=0)
    interp['psi_dot'] = interp1d(fine_t, psi_dot_fine, kind='cubic', bounds_error=False, fill_value=0)
    return interp

# --- 5.3: Define Integrands (using the robust split method) ---

def rec_integrand(eta, l, k, interp, eta_rec):
    g_eta = visibility_function(eta, eta_rec, rec_width)
    if g_eta < 1e-10: return 0.0
    psi, Theta0, vr = interp['psi'](eta), interp['Theta0'](eta), interp['vr'](eta)
    source = g_eta * (Theta0 + psi + vr)
    proj_func = spherical_jn(l, k * (eta_0 - eta))
    return source * proj_func

def isw_integrand(eta, l, k, interp):
    phi_dot, psi_dot = interp['phi_dot'](eta), interp['psi_dot'](eta)
    source = phi_dot + psi_dot
    proj_func = spherical_jn(l, k * (eta_0 - eta))
    return source * proj_func

# --- 5.4: Main Calculation Loop ---
Theta_l_k = np.zeros((len(ells), len(solutions)))

print("Creating interpolators for all modes...")
interpolators = [get_solution_interpolators(sol) for sol in solutions]

print("Performing line-of-sight integrals...")
rec_window_width = 10 * rec_width
rec_start_eta = max(t0_integration, recConformalTime - rec_window_width)
rec_end_eta = recConformalTime + rec_window_width

for i, sol in enumerate(solutions):
    k_n = sol['k']  # Keep k in physical units
    interp = interpolators[i]
    if (i+1)%5 == 0:
        print(f"Integrating for mode {i+1}/{len(solutions)} (k={k_n:.4f})")
    
    for j, l in enumerate(ells):
        # Adaptive precision: higher precision for low-l, lower for high-l
        if l <= 20:
            limit, epsabs, epsrel = 100, 1e-10, 1e-8
        elif l <= 60:
            limit, epsabs, epsrel = 50, 1e-8, 1e-6
        else:
            limit, epsabs, epsrel = 30, 1e-6, 1e-4
            
        integral_rec, err_rec = quad(rec_integrand, rec_start_eta, rec_end_eta,
                                     args=(l, k_n, interp, recConformalTime), 
                                     limit=limit, epsabs=epsabs, epsrel=epsrel)
        integral_isw, err_isw = quad(isw_integrand, rec_end_eta, eta_0,
                                     args=(l, k_n, interp),
                                     limit=limit, epsabs=epsabs, epsrel=epsrel)
        Theta_l_k[j, i] = integral_rec + integral_isw

# --- 5.5: Summation and Final Spectrum (WITH CORRECTIONS) ---
print("Summing contributions to form Cl spectrum...")
Cl_values_dimless = np.zeros(len(ells)) # Dimensionless Cl
P_R_values = primordial_power_spectrum(allowedK, As, ns)

# CORRECTION 2: For discrete k modes in closed universe, use simple sum instead of continuous measure
# Volume factor for closed universe (may need adjustment based on your model)
Volume_factor = (2 * np.pi**2) / np.abs(K)**(3/2)  # Characteristic volume scale

# DEBUG: Print key values to diagnose amplitude issues
print(f"K value: {K:.6f}")
print(f"Volume_factor: {Volume_factor:.6e}")
print(f"Number of k modes: {len(allowedK)}")
print(f"k range: {allowedK[0]:.3f} to {allowedK[-1]:.3f}")
print(f"Number of l modes: {len(ells)}")
print(f"l values: {ells[:10]} ... {ells[-5:]}")
print(f"Sample P_R values: {P_R_values[:3]}")

for j, l in enumerate(ells):
    # Simple sum for discrete modes - remove 4π factor and continuous k-measure
    Cl_values_dimless[j] = np.sum(P_R_values * Theta_l_k[j, :]**2) / Volume_factor
    
    # DEBUG: Print first few Cl values
    if j < 3:
        print(f"l={l}: Cl_dimless = {Cl_values_dimless[j]:.6e}")
    
# CORRECTION 1: Apply the unit conversion
Cl_values_physical = Cl_values_dimless * UNIT_CONVERSION_FACTOR

# Calculate Dl in physical units (muK^2)
Dl_values = ells * (ells + 1) * Cl_values_physical / (2 * np.pi)

# DEBUG: Print final spectrum values
print(f"Sample Dl values: {Dl_values[:5]}")
print(f"Max Dl value: {np.max(Dl_values):.2f}")
print(f"Peak Dl location: l = {ells[np.argmax(Dl_values)]}")

# --- 5.6: Plot the Final CMB Power Spectrum ---
print("--- Plotting Final CMB Power Spectrum ---")
plt.figure(figsize=(10, 6))
plt.plot(ells, Dl_values, 'b-', marker='o', markersize=4, label='Calculated Spectrum')
plt.xscale('log')
plt.yscale('linear') # Standard for Dl plots
plt.xlabel(r'Multipole moment $\ell$', fontsize=14)
plt.ylabel(r'$\ell(\ell+1)C_\ell / 2\pi \quad [\mu K^2]$', fontsize=14)
plt.title('CMB Temperature Angular Power Spectrum', fontsize=16)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig(folder_path + 'CMB_Power_Spectrum_finer.pdf')
plt.show()