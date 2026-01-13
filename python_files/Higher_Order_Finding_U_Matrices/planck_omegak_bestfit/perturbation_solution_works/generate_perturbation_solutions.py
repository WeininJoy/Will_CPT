# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model, following the methodology outlined in the paper "Evidence for a Palindromic Universe".

It uses pre-computed data from transfer matrix calculations to solve for the
boundary conditions at the Future Conformal Boundary (FCB) for each allowed wavenumber k.
It then integrates the perturbation equations to obtain the full evolution of the
variables and plots the results.

CORRECTED VERSION: Fixes the IndexError in the Boltzmann hierarchy ODE function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# =============================================================================
# 1. SETUP: Parameters, Constants, and ODE Functions
# =============================================================================

print("--- Setting up parameters and functions ---")

# Data folder
nu_spacing = 4
folder = f'./data/'
# folder_path = folder + 'data_allowedK/'
# folder_path = folder + 'data_integerK/'
folder_path = folder + 'data_all_k/'

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
    return Omega_lambda, Omega_m, Omega_K

###############################################################################
# Best-fit parameters from nu_spacing=4 (first try)
# mt, kt, Omegab_ratio, h = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915  # the actual value still needs to be checked
print(f"Recombination redshift z_rec = {z_rec}") # z_rec = 1061.915 based on the calculate_z_rec() output
###############################################################################

# Working units: 8piG = c = hbar = 1, and s0 = 1 for numerical stability.

#set tolerances
atol = 1e-13;
rtol = 1e-13;
stol = 1e-10;
num_variables = 75; # number of pert variables
l_max = 69 # Derived from num_variables_boltzmann = 7 + (l_max - 2 + 1)
swaptime = 2; #set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda);

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

def dX1_dt(t, X):
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

    poisson_residual = X[0] + 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (-3*adot/X[0]*X[5] + X[3])/X[0]*OmegaM + (-4*adot/X[0]*X[4] + X[2])/X[0]**2*OmegaR);
    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k**2)*X[4]);
    dmdot = 3*phidot + X[5]*(k**2);
    vrdot = -(X[1] + X[2]/4);
    vmdot = - (adot/X[0])*X[5] - X[1];

    return [adot, poisson_residual, drdot, dmdot, vrdot, vmdot]

# Define derivative functions with closure over k and cosmological parameters
def dX2_dt_local(t, X):
    s, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(s**3) + OmegaR*abs(s**4)))

    rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)

    poisson_residual = phi + 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (3*sdot/s*vm + dm)*s*OmegaM + (4*sdot/s*vr + dr)*s**2*OmegaR);
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
    vmdot = (sdot/s)*vm - psi
    derivatives = [sdot, poisson_residual, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    for j in range(8, num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    derivatives.append(lastderiv)
    return derivatives

def dX3_dt_local(t, X):
    a, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
    adot = a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(a**2) + OmegaM/abs(a**3) + OmegaR/abs(a**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(a)**3)
    rho_r = 3*(H0**2)*OmegaR/(abs(a)**4)

    poisson_residual = phi + 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (-3*adot/a*vm + dm)/a*OmegaM + (-4*adot/a*vr + dr)/a**2*OmegaR);
    phidot = - (adot/a)*psi - ((4/3)*rho_r*vr + rho_m*vm)*(a**2/2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR/a)*(-adot*fr2/a**2 + 0.5*fr2dot/a)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
    vmdot = (-adot/a)*vm - psi
    derivatives = [adot, poisson_residual, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    for j in range(8, num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    derivatives.append(lastderiv)
    return derivatives

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
    allowedK = np.load(folder_path + 'allowedK.npy')
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the 'data' directory.")
    exit()

# Extract sub-matrices
Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

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
    
get_A = create_matrix_interpolator(kvalues, Amatrices)
get_D = create_matrix_interpolator(kvalues, Dmatrices)
get_X1 = create_matrix_interpolator(kvalues, X1matrices)
get_X2 = create_matrix_interpolator(kvalues, X2matrices)
get_recs = create_vector_interpolator(kvalues, recValues)


# =============================================================================
# 3. CALCULATE AND INTEGRATE SOLUTIONS
# =============================================================================

print("\n--- Calculating solutions for the first 3 allowed modes ---")

solutions = []
num_modes_to_plot = 3

for i in range(num_modes_to_plot):
    k = allowedK[i+2]
    print(f"\nProcessing mode n={i+1} with k={k:.4f}")

    # --- a) Get interpolated matrices for this k ---
    A = get_A(k)
    D = get_D(k)
    X1 = get_X1(k)
    X2 = get_X2(k)
    recs_vec = get_recs(k)

    AX1 = A.reshape(6,6) @ X1.reshape(6,4);
    DX2 = D.reshape(6,2) @ X2.reshape(2,4);
    matrixog = AX1 + DX2 # + GX3;
    matrix = matrixog[[2,3,4,5], :]; #take rows corresponding to dr, dm, vr, vm
    
    xrecs = [recs_vec[2], recs_vec[3], recs_vec[4], recs_vec[5]];


    # # --- b) Solve for X_inf ---
    # M_matrix = (A @ X1 + D @ X2)[1:6, :]
    # x_rec_subset = recs_vec[1:6]
    
    try:
        x_inf = np.linalg.lstsq(matrix, xrecs, rcond=None)[0];
        # x_inf = np.linalg.solve(M_matrix, x_rec_subset)
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Skipping mode.")
        continue

    # --- c) Calculate initial conditions at eta' (endtime) ---
    x_prime = X1 @ x_inf
    y_prime_2_4 = X2 @ x_inf
    s_prime_val = np.interp(endtime, sol.t, sol.y[0])

    Y_prime = np.zeros(num_variables+1) # +1 for s
    Y_prime[0] = s_prime_val
    Y_prime[1:7] = x_prime
    Y_prime[7:9] = y_prime_2_4

    # check Y_prime satisfies the constraint
    phi_prime, psi_prime, dr_prime, dm_prime, vr_prime, vm_prime = x_prime
    dsdt_prime = ds_dt(endtime,Y_prime[0])
    phi_derived = -3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (3*dsdt_prime/s_prime_val*vm_prime + dm_prime)*s_prime_val*OmegaM + (4*dsdt_prime/s_prime_val*vr_prime + dr_prime)*s_prime_val**2*OmegaR);
    print("phi_prime, phi_derived, difference:", phi_prime, phi_derived, abs(phi_prime - phi_derived))
    
    # --- d) Integrate backwards from endtime to recConformalTime ---
    M = np.diag([0 if i==1 else 1 for i in range(num_variables+1)]) # zero out poisson eq row for solver stability
    sol_part1 = solve_ivp(dX2_dt_local, [endtime, swaptime], Y_prime, 
                          dense_output=True, method='Radau', mass_matrix=M, atol=atol, rtol=rtol)
    
    Y_swap = sol_part1.y[:, -1]
    Y_swap[0] = 1./ Y_swap[0] # swap s to a
    sol_part2 = solve_ivp(dX3_dt_local, [swaptime, recConformalTime], Y_swap, 
                          dense_output=True, method='Radau', mass_matrix=M, atol=atol, rtol=rtol)
    
    # check the backward solution satisfies the constraint at recombination
    a_rec, phi_rec, psi_rec, dr_rec, dm_rec, vr_rec, vm_rec = sol_part2.y[:7, -1]
    print("phi, dr, dm, vr, vm at recombination (backward integration):", phi_rec, dr_rec, dm_rec, vr_rec, vm_rec)
    dadt_rec = da_dt(recConformalTime,a_rec)
    phi_derived = -3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (-3*dadt_rec/a_rec*vm_rec + dm_rec)/a_rec*OmegaM + (-4*dadt_rec/a_rec*vr_rec + dr_rec)/a_rec**2*OmegaR);
    print("phi_rec (backward), phi_derived, difference:", phi_rec, phi_derived, abs(phi_rec - phi_derived))
    t_backward = np.concatenate((sol_part1.t, sol_part2.t))
    Y_backward_a = sol_part2.y
    Y_backward_a[0, :] = 1./Y_backward_a[0, :] # swap back a to s where needed
    Y_backward = np.concatenate((sol_part1.y, Y_backward_a), axis=1)

    # --- e) Get solution from Big Bang to Recombination (perfect fluid) ---
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda);
    dr3 = (OmegaM*OmegaR*(696*OmegaK + 404*k**2*OmegaLambda) - 63*OmegaM**3)/(4320*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2));
    dr4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1924*k**2*OmegaLambda) + 48*OmegaR**2*(160*OmegaK**2 + 176*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-80*OmegaR + 23*k**4*OmegaLambda)))/(181440*OmegaR**2*OmegaLambda**2);

    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda);
    dm3 = (OmegaM*OmegaR*(404*k**2*OmegaLambda + 696*OmegaK) - 63*OmegaM**3)/(5760*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
    dm4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1924*k**2*OmegaLambda) + 24*OmegaR**2*(320*OmegaK**2 - 480*OmegaR*OmegaLambda + 247*k**2*OmegaK*OmegaLambda + 33*k**4*OmegaLambda**2))/(241920*OmegaR**2*OmegaLambda**2);

    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda);
    vr4 = (63*OmegaM**3 - 8*OmegaM*OmegaR*(87*OmegaK + 43*k**2*OmegaLambda))/(34560*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2));
    vr5 = (-63*OmegaM**4 + OmegaM**2*OmegaR*(783*OmegaK + 347*k**2*OmegaLambda) - 24*OmegaR**2*(64*OmegaK**2 + 48*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-32*OmegaR + 5*k**4*OmegaLambda)))/(362880*OmegaR**2*OmegaLambda**2);

    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda);
    vm4 = (63*OmegaM**3 - 32*OmegaM*OmegaR*(15*OmegaK + 4*k**2*OmegaLambda))/(34560*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2));
    vm5 = (-63*OmegaM**4 + 2*OmegaM**2*OmegaR*(297*OmegaK + 79*k**2*OmegaLambda) - 24*OmegaR**2*(43*OmegaK**2 + 13*k**2*OmegaK*OmegaLambda + OmegaLambda*(-96*OmegaR + k**4*OmegaLambda)))/(362880*OmegaR**2*OmegaLambda**2);

    # Set fixed initial conditions for dr, dm, vr, vm
    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4;
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4;
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5;
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5;

    # Initial guess for phi0 using analytical expansion
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)
    phi0 = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4;

    X0 = [a_Bang, phi0, dr0, dm0, vr0, vm0];
    M = np.diag([1, 0, 1, 1, 1, 1])
    sol_perfect = solve_ivp(dX1_dt, [t0, recConformalTime], X0, method='Radau', mass_matrix=M, atol=atol, rtol=rtol);
    a_rec, phi_rec, dr_rec, dm_rec, vr_rec, vm_rec = sol_perfect.y[:6, -1]
    print("phi, dr, dm, vr, vm at recombination (perfect fluid):", phi_rec, dr_rec, dm_rec, vr_rec, vm_rec)
    dadt_rec = da_dt(recConformalTime,a_rec)
    phi_derived = -3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (-3*dadt_rec/a_rec*vm_rec + dm_rec)/a_rec*OmegaM + (-4*dadt_rec/a_rec*vr_rec + dr_rec)/a_rec**2*OmegaR);
    print("phi_rec (perfect), phi_derived, difference:", phi_rec, phi_derived, abs(phi_rec - phi_derived))
    
    # --- f) Stitch and Reflect ---
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1]))

    Y_perfect_full = np.zeros((num_variables+1, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
    Y_perfect_full[1,:] = sol_perfect.y[1,:]
    Y_perfect_full[2,:] = sol_perfect.y[1,:]
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]
    
    Y_left = np.concatenate((Y_perfect_full, Y_backward[:, ::-1]), axis=1)

    t_right = 2 * fcb_time - t_left[::-1]
    
    symm = np.ones(num_variables+1)
    symm[[5, 6]] = -1
    for l_idx in range(l_max - 1):
        l = l_idx + 2
        if l % 2 != 0: symm[7 + l_idx] = -1
    S = np.diag(symm)
    
    Y_right = S @ Y_left[:, ::-1]

    t_full = np.concatenate((t_left, t_right))
    Y_full = np.concatenate((Y_left, Y_right), axis=1)
    
    solutions.append({'t': t_full, 'Y': Y_full})

# =============================================================================
# 4. PLOTTING
# =============================================================================

print("\n--- Plotting solutions ---")

fig, axes = plt.subplots(num_modes_to_plot, 1, figsize=(10, 8), sharex=True)
if num_modes_to_plot == 1: axes = [axes]

labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$', r'$\phi$', r'$\psi$']
indices = [5, 3, 6, 4, 1, 2]
colors = ['blue', 'red', 'green', 'orange', 'magenta', 'cyan']

for i, sol in enumerate(solutions):
    ax = axes[i]
    for label, index, color in zip(labels, indices, colors):
        ax.plot(sol['t'], sol['Y'][index, :], label=label, color=color, linewidth=1.2)

    ax.axvline(fcb_time, color='k', linestyle='--', label='FCB')
    ax.axvline(2 * fcb_time, color='k', linestyle='--', label='Big Crunch')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(f'n = {i+1}', fontsize=12)
    ax.set_xlim(0, 2 * fcb_time * 1.05)
    # ax.set_ylim(-3, 3)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3)
axes[-1].set_xlabel(r'$\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Evolution of Perturbation Solutions for Allowed Modes', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig(folder_path + 'perturbation_solutions_DAE.pdf')
plt.show()