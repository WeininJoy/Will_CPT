# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model, following the methodology outlined in the paper "Evidence for a Palindromic Universe".

MATCHING AT SWAPTIME VERSION:
- Integrates backward from FCB to swaptime
- Integrates forward from recombination to swaptime
- Matches both solutions at swaptime to determine boundary conditions
- Combines with perfect fluid solution from Big Bang to recombination
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
folder = './data/'
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
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915
print(f"Recombination redshift z_rec = {z_rec}")
###############################################################################

# Working units: 8piG = c = hbar = 1, and s0 = 1 for numerical stability.

# Set tolerances
atol = 1e-13
rtol = 1e-13
atol_forward = 1e-10  # Relaxed tolerance for forward integration
rtol_forward = 1e-10  # Relaxed tolerance for forward integration
stol = 1e-10
num_variables = 75 # number of pert variables
l_max = 69 # Derived from num_variables_boltzmann = 7 + (l_max - 2 + 1)
swaptime = 2 # set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda) # we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

# Write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5

# Set coefficients for initial conditions
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4

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
    fcb_time = None

if fcb_time is not None:
    print(f"Further processing with fcb_time = {fcb_time}")
else:
    print(f"No fcb_time available for further processing.")

endtime = fcb_time - deltaeta

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

# Find conformal time at recombination
a_rec = 1./(1+z_rec)

# Take difference between a values and a_rec to find where a=a_rec
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

# Perfect fluid ODE (Big Bang to recombination)
def dX1_dt(t, X):
    """
    Perfect fluid approximation (6 variables)
    X = [a, phi, dr, dm, vr, vm]
    """
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k**2)*X[4])
    dmdot = 3*phidot + X[5]*(k**2)
    vrdot = -(X[1] + X[2]/4)
    vmdot = - (adot/X[0])*X[5] - X[1]

    return [adot, phidot, drdot, dmdot, vrdot, vmdot]

# Define derivative functions for Boltzmann hierarchy
def dX2_dt_local(t, X):
    """
    Backward integration from FCB to swaptime using s (inverse scale factor)
    X = [s, phi, psi, dr, dm, vr, vm, fr2, fr3, ..., fr_lmax]
    """
    s, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(s**3) + OmegaR*abs(s**4)))

    rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)

    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
    vmdot = (sdot/s)*vm - psi
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    for j in range(8, num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    derivatives.append(lastderiv)
    return derivatives

def dX3_dt_local(t, X):
    """
    Forward integration from recombination to swaptime using a (scale factor)
    X = [a, phi, psi, dr, dm, vr, vm, fr2, fr3, ..., fr_lmax]
    """
    a, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
    adot = a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(a**2) + OmegaM/abs(a**3) + OmegaR/abs(a**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(a)**3)
    rho_r = 3*(H0**2)*OmegaR/(abs(a)**4)

    phidot = - (adot/a)*psi - ((4/3)*rho_r*vr + rho_m*vm)*(a**2/2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR/a)*(-adot*fr2/a**2 + 0.5*fr2dot/a)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
    vmdot = (-adot/a)*vm - psi
    derivatives = [adot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    for j in range(8, num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    derivatives.append(lastderiv)
    return derivatives

# =============================================================================
# 2. DATA LOADING
# =============================================================================

print("\n--- Loading pre-computed data ---")

try:
    kvalues = np.load(folder_path + 'L70_kvalues.npy')
    ABCmatrices = np.load(folder_path + 'L70_ABCmatrices.npy')
    DEFmatrices = np.load(folder_path + 'L70_DEFmatrices.npy')
    JKLmatrices = np.load(folder_path + 'L70_JKLmatrices.npy')
    MNOmatrices = np.load(folder_path + 'L70_MNOmatrices.npy')
    PQRmatrices = np.load(folder_path + 'L70_PQRmatrices.npy')
    X1matrices = np.load(folder_path + 'L70_X1matrices.npy')
    X2matrices = np.load(folder_path + 'L70_X2matrices.npy')
    recValues = np.load(folder_path + 'L70_recValues.npy')
    allowedK = np.load(folder_path + 'allowedK.npy')
    print("All data files loaded successfully.")
    print(f"Number of k values: {len(kvalues)}")
    print(f"Number of allowed K values: {len(allowedK)}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the 'data_all_k' directory.")
    exit()

# =============================================================================
# 3. CALCULATE AND INTEGRATE SOLUTIONS
# =============================================================================

print("\n--- Calculating solutions for allowed modes at indices 2, 3, 4 ---")

solutions = []
mode_indices = [2, 3, 4]  # Indices of allowed modes to plot
num_modes_to_plot = len(mode_indices)

for mode_idx in mode_indices:
    k = allowedK[mode_idx]
    print(f"\n{'='*80}")
    print(f"Processing mode index {mode_idx} with k={k:.6f}")
    print(f"{'='*80}")

    # Find the closest k value in the kvalues array for matrix lookup
    k_idx = np.argmin(np.abs(kvalues - k))
    print(f"Using matrices from k_idx={k_idx}, k={kvalues[k_idx]:.6f}")

    # Load matrices for this k value (needed for all steps)
    ABC = ABCmatrices[k_idx]  # (75, 6)
    DEF = DEFmatrices[k_idx]  # (75, 2)
    JKL = JKLmatrices[k_idx]  # (75, 6)
    MNO = MNOmatrices[k_idx]  # (75, 2)
    PQR = PQRmatrices[k_idx]  # (75, 67)
    X1 = X1matrices[k_idx]    # (6, 4)
    X2 = X2matrices[k_idx]    # (2, 4)

    # Construct matrices for matching equation
    JKLMNOPQR = np.hstack([JKL, MNO, PQR])  # (75, 75)
    ABCDEF = np.hstack([ABC, DEF])  # (75, 8)
    X_combined = np.vstack([X1, X2])  # (8, 4)

    # Compute M_full where: v_rec = M_full * xinf
    M_full = np.linalg.solve(JKLMNOPQR, ABCDEF @ X_combined)  # (75, 4)
    M_reduced = M_full[[2, 3, 4, 5], :]  # (4, 4) for dr, dm, vr, vm

    # ==========================================================================
    # STEP 1: Calculate perfect fluid solution from Big Bang to recombination
    # ==========================================================================
    print("\nStep 1: Calculating perfect fluid solution from Big Bang to recombination...")

    # Power series coefficients for initial conditions
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda)
    dr3 = (OmegaM*OmegaR*(696*OmegaK + 404*k**2*OmegaLambda) - 63*OmegaM**3)/(4320*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2))
    dr4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1924*k**2*OmegaLambda) + 48*OmegaR**2*(160*OmegaK**2 + 176*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-80*OmegaR + 23*k**4*OmegaLambda)))/(181440*OmegaR**2*OmegaLambda**2)

    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda)
    dm3 = (OmegaM*OmegaR*(404*k**2*OmegaLambda + 696*OmegaK) - 63*OmegaM**3)/(5760*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2))
    dm4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1924*k**2*OmegaLambda) + 24*OmegaR**2*(320*OmegaK**2 - 480*OmegaR*OmegaLambda + 247*k**2*OmegaK*OmegaLambda + 33*k**4*OmegaLambda**2))/(241920*OmegaR**2*OmegaLambda**2)

    vr1 = -1/2
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda)
    vr4 = (63*OmegaM**3 - 8*OmegaM*OmegaR*(87*OmegaK + 43*k**2*OmegaLambda))/(34560*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2))
    vr5 = (-63*OmegaM**4 + OmegaM**2*OmegaR*(783*OmegaK + 347*k**2*OmegaLambda) - 24*OmegaR**2*(64*OmegaK**2 + 48*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-32*OmegaR + 5*k**4*OmegaLambda)))/(362880*OmegaR**2*OmegaLambda**2)

    vm1 = -1/2
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda)
    vm4 = (63*OmegaM**3 - 32*OmegaM*OmegaR*(15*OmegaK + 4*k**2*OmegaLambda))/(34560*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2))
    vm5 = (-63*OmegaM**4 + 2*OmegaM**2*OmegaR*(297*OmegaK + 79*k**2*OmegaLambda) - 24*OmegaR**2*(43*OmegaK**2 + 13*k**2*OmegaK*OmegaLambda + OmegaLambda*(-96*OmegaR + k**4*OmegaLambda)))/(362880*OmegaR**2*OmegaLambda**2)

    # Set fixed initial conditions for dr, dm, vr, vm
    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5

    # Initial guess for phi0 using analytical expansion
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2))
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)
    phi0_guess = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4

    # Define helper function: solve ODE for given phi0
    def solve_for_phi0(phi0_input):
        """Solve ODE with given phi0 and return solution at recombination."""
        X0_temp = [a_Bang, phi0_input, dr0, dm0, vr0, vm0]
        sol_temp = solve_ivp(dX1_dt, [t0, recConformalTime], X0_temp,
                            method='LSODA', atol=atol, rtol=rtol)
        return sol_temp

    # Define residual function for root-finding
    def residual(phi0_input):
        """
        Compute residual: phi_rec - phi_constraint
        The constraint should be satisfied at recombination.
        """
        # Solve ODE with this phi0
        sol_temp = solve_for_phi0(phi0_input)

        # Extract values at recombination (last point)
        a_rec_temp = sol_temp.y[0, -1]
        phi_rec_temp = sol_temp.y[1, -1]
        dr_rec_temp = sol_temp.y[2, -1]
        dm_rec_temp = sol_temp.y[3, -1]
        vr_rec_temp = sol_temp.y[4, -1]
        vm_rec_temp = sol_temp.y[5, -1]
        t_rec = sol_temp.t[-1]

        # Compute the constraint value at recombination
        adot_rec_temp = da_dt(t_rec, a_rec_temp)
        phi_constraint = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
            (-3*adot_rec_temp/a_rec_temp*vm_rec_temp + dm_rec_temp)*OmegaM/a_rec_temp +
            (-4*adot_rec_temp/a_rec_temp*vr_rec_temp + dr_rec_temp)*OmegaR/a_rec_temp**2
        )

        # Return residual
        return phi_rec_temp - phi_constraint

    # Use root-finding to find the correct phi0
    print(f"  Starting root-finding for phi0...")
    print(f"  Initial guess: phi0_guess = {phi0_guess:.6e}")

    try:
        # Try bracketing method first
        result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5],
                            method='brentq', xtol=1e-10)
        phi0_optimal = result.root
        print(f"  Root-finding converged!")
        print(f"  Optimal phi0 = {phi0_optimal:.6e}")
    except ValueError:
        # If bracket doesn't work, try secant method
        print("  Bracket method failed, trying secant method...")
        result = root_scalar(residual, x0=phi0_guess, x1=phi0_guess*1.1,
                            method='secant', xtol=1e-10)
        phi0_optimal = result.root
        print(f"  Root-finding converged!")
        print(f"  Optimal phi0 = {phi0_optimal:.6e}")

    # Solve with optimal phi0
    sol_perfect = solve_for_phi0(phi0_optimal)

    # Extract values at recombination
    a_rec_final = sol_perfect.y[0, -1]
    phi_rec_final = sol_perfect.y[1, -1]
    dr_rec_final = sol_perfect.y[2, -1]
    dm_rec_final = sol_perfect.y[3, -1]
    vr_rec_final = sol_perfect.y[4, -1]
    vm_rec_final = sol_perfect.y[5, -1]

    # Verify constraint is satisfied
    adot_rec_final = da_dt(recConformalTime, a_rec_final)
    phi_constraint_final = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
        (-3*adot_rec_final/a_rec_final*vm_rec_final + dm_rec_final)*OmegaM/a_rec_final +
        (-4*adot_rec_final/a_rec_final*vr_rec_final + dr_rec_final)*OmegaR/a_rec_final**2
    )

    print(f"\n  Verification at recombination:")
    print(f"    phi_rec = {phi_rec_final:.6e}")
    print(f"    phi_constraint = {phi_constraint_final:.6e}")
    print(f"    Difference = {abs(phi_rec_final - phi_constraint_final):.6e}")
    print(f"  Perfect fluid integration complete: {len(sol_perfect.t)} points")
    print(f"    a = {a_rec_final:.6e}, z = {1./a_rec_final - 1:.2f}")
    print(f"    dr = {dr_rec_final:.6e}, dm = {dm_rec_final:.6e}")
    print(f"    vr = {vr_rec_final:.6e}, vm = {vm_rec_final:.6e}")

    # ==========================================================================
    # STEP 2: Solve for xinf using perfect fluid values at recombination
    # ==========================================================================
    print("\nStep 2: Solving for boundary conditions xinf at FCB...")

    # Use perfect fluid values at recombination to solve for xinf
    pf_rec = np.array([dr_rec_final, dm_rec_final, vr_rec_final, vm_rec_final])

    # Solve: M_reduced * xinf = pf_rec
    xinf = np.linalg.solve(M_reduced, pf_rec)
    print(f"Solved xinf = {xinf}")
    print(f"  δr_inf = {xinf[0]:.6e}, δm_inf = {xinf[1]:.6e}")
    print(f"  vr_inf = {xinf[2]:.6e}, vm_dot_inf = {xinf[3]:.6e}")

    # Compute v_rec = M_full * xinf (all 75 variables at recombination for Boltzmann)
    v_rec = M_full @ xinf

    # CRITICAL FIX: Enforce phi continuity at recombination
    # The M_full matrix gives phi from the matching equations, but this doesn't
    # guarantee continuity with the perfect fluid solution. Since phi must be
    # continuous at recombination, we enforce this by replacing v_rec[0] with
    # the perfect fluid value.
    phi_bol_before_fix = v_rec[0]
    v_rec[0] = phi_rec_final  # Enforce phi continuity

    print(f"\n  Continuity check at recombination:")
    print(f"    Perfect fluid -> Boltzmann hierarchy:")
    print(f"    phi: pf={phi_rec_final:.6e}, bol_before_fix={phi_bol_before_fix:.6e}, bol_after_fix={v_rec[0]:.6e}")
    print(f"    dr:  pf={dr_rec_final:.6e}, bol={v_rec[2]:.6e}, diff={abs(dr_rec_final-v_rec[2]):.6e}")
    print(f"    dm:  pf={dm_rec_final:.6e}, bol={v_rec[3]:.6e}, diff={abs(dm_rec_final-v_rec[3]):.6e}")
    print(f"    vr:  pf={vr_rec_final:.6e}, bol={v_rec[4]:.6e}, diff={abs(vr_rec_final-v_rec[4]):.6e}")
    print(f"    vm:  pf={vm_rec_final:.6e}, bol={v_rec[5]:.6e}, diff={abs(vm_rec_final-v_rec[5]):.6e}")
    print(f"    psi: bol={v_rec[1]:.6e} (allowed to jump from pf where psi=phi)")
    print(f"  NOTE: phi was manually set to perfect fluid value to enforce continuity")

    # ==========================================================================
    # STEP 3: Integrate backward from FCB to swaptime
    # ==========================================================================
    print("\nStep 3: Integrating backward from FCB to swaptime...")

    # Calculate v_prime = (X1 X2 0)^T * xinf
    x_prime = X1 @ xinf  # (6,) for base variables
    y_prime_2_4 = X2 @ xinf  # (2,) for Fr2, Fr3

    # Get s value at endtime
    s_prime_val = np.interp(endtime, sol.t, sol.y[0])

    # Construct full initial condition vector at endtime
    Y_prime = np.zeros(num_variables + 1)  # +1 for s
    Y_prime[0] = s_prime_val
    Y_prime[1:7] = x_prime  # phi, psi, dr, dm, vr, vm
    Y_prime[7:9] = y_prime_2_4  # fr2, fr3
    # Y_prime[9:] remain zero (higher multipoles)

    print(f"Initial conditions at FCB (eta={endtime:.6f}):")
    print(f"  s = {Y_prime[0]:.6e}")
    print(f"  phi = {Y_prime[1]:.6e}, psi = {Y_prime[2]:.6e}")
    print(f"  dr = {Y_prime[3]:.6e}, dm = {Y_prime[4]:.6e}")
    print(f"  vr = {Y_prime[5]:.6e}, vm = {Y_prime[6]:.6e}")

    # Integrate backward from endtime to swaptime
    sol_fcb_to_swap = solve_ivp(dX2_dt_local, [endtime, swaptime], Y_prime,
                                 dense_output=True, method='LSODA', atol=atol, rtol=rtol)

    print(f"Backward integration complete: {len(sol_fcb_to_swap.t)} points")
    print(f"  Values at swaptime (s-coordinates):")
    Y_swap_from_fcb = sol_fcb_to_swap.y[:, -1]
    print(f"    s = {Y_swap_from_fcb[0]:.6e}, a = {1./Y_swap_from_fcb[0]:.6e}")
    print(f"    phi = {Y_swap_from_fcb[1]:.6e}, psi = {Y_swap_from_fcb[2]:.6e}")
    print(f"    dr = {Y_swap_from_fcb[3]:.6e}, dm = {Y_swap_from_fcb[4]:.6e}")
    print(f"    vr = {Y_swap_from_fcb[5]:.6e}, vm = {Y_swap_from_fcb[6]:.6e}")

    # ==========================================================================
    # STEP 4: Integrate forward from recombination to swaptime
    # ==========================================================================
    print("\nStep 4: Integrating forward from recombination to swaptime...")

    # v_rec was already computed in STEP 2
    # Construct full initial condition vector at recombination
    Y_rec = np.zeros(num_variables + 1)  # +1 for a
    Y_rec[0] = a_rec
    Y_rec[1:] = v_rec

    print(f"Initial conditions at recombination (eta={recConformalTime:.6f}):")
    print(f"  a = {Y_rec[0]:.6e}, z = {1./Y_rec[0] - 1:.2f}")
    print(f"  phi = {Y_rec[1]:.6e}, psi = {Y_rec[2]:.6e}")
    print(f"  dr = {Y_rec[3]:.6e}, dm = {Y_rec[4]:.6e}")
    print(f"  vr = {Y_rec[5]:.6e}, vm = {Y_rec[6]:.6e}")

    # Integrate forward from recombination to swaptime using BDF
    sol_rec_to_swap = solve_ivp(dX3_dt_local, [recConformalTime, swaptime], Y_rec,
                                 dense_output=True, method='BDF',
                                 atol=atol_forward, rtol=rtol_forward, max_step=1e-2)

    print(f"Forward integration complete: {len(sol_rec_to_swap.t)} points")
    print(f"  Values at swaptime (a-coordinates):")
    Y_swap_from_rec = sol_rec_to_swap.y[:, -1]
    print(f"    a = {Y_swap_from_rec[0]:.6e}, s = {1./Y_swap_from_rec[0]:.6e}")
    print(f"    phi = {Y_swap_from_rec[1]:.6e}, psi = {Y_swap_from_rec[2]:.6e}")
    print(f"    dr = {Y_swap_from_rec[3]:.6e}, dm = {Y_swap_from_rec[4]:.6e}")
    print(f"    vr = {Y_swap_from_rec[5]:.6e}, vm = {Y_swap_from_rec[6]:.6e}")

    # Check matching at swaptime
    print(f"\n  Checking matching at swaptime:")
    a_swap_from_fcb = 1. / Y_swap_from_fcb[0]
    a_swap_from_rec = Y_swap_from_rec[0]
    print(f"    a from FCB: {a_swap_from_fcb:.6e}")
    print(f"    a from rec: {a_swap_from_rec:.6e}")
    print(f"    Difference: {abs(a_swap_from_fcb - a_swap_from_rec):.6e}")

    # Compare perturbation values at swaptime
    print(f"\n  Perturbation values at swaptime:")
    for i, label in enumerate(['phi', 'psi', 'dr', 'dm', 'vr', 'vm']):
        val_fcb = Y_swap_from_fcb[i+1]
        val_rec = Y_swap_from_rec[i+1]
        diff = abs(val_fcb - val_rec)
        rel_diff = diff / (abs(val_rec) + 1e-20)
        print(f"    {label}: FCB={val_fcb:.6e}, rec={val_rec:.6e}, diff={diff:.6e}, rel={rel_diff:.6e}")

    # DIAGNOSTIC: Check if matrix predictions match actual integrations
    print(f"\n  Diagnostic: Checking matrix predictions vs actual integrations:")
    # Predicted v_swap from FCB side
    v_prime_full = np.zeros(8)
    v_prime_full[:6] = x_prime
    v_prime_full[6:8] = y_prime_2_4
    v_swap_predicted_fcb = ABCDEF @ v_prime_full
    print(f"    From FCB - Matrix prediction vs actual integration:")
    for i, label in enumerate(['phi', 'psi', 'dr', 'dm', 'vr', 'vm']):
        predicted = v_swap_predicted_fcb[i]
        actual = Y_swap_from_fcb[i+1]
        print(f"      {label}: predicted={predicted:.6e}, actual={actual:.6e}, diff={abs(predicted-actual):.6e}")

    # Predicted v_swap from rec side
    v_swap_predicted_rec = JKLMNOPQR @ v_rec
    print(f"    From rec - Matrix prediction vs actual integration:")
    for i, label in enumerate(['phi', 'psi', 'dr', 'dm', 'vr', 'vm']):
        predicted = v_swap_predicted_rec[i]
        actual = Y_swap_from_rec[i+1]
        print(f"      {label}: predicted={predicted:.6e}, actual={actual:.6e}, diff={abs(predicted-actual):.6e}")

    # ==========================================================================
    # STEP 5: Combine the three solutions and prepare for plotting
    # ==========================================================================
    print("\nStep 5: Combining solutions and preparing for reflection...")

    # Convert perfect fluid solution to full variable array
    Y_perfect_full = np.zeros((num_variables+1, len(sol_perfect.t)))
    Y_perfect_full[0,:] = sol_perfect.y[0,:]  # a (scale factor)
    Y_perfect_full[1,:] = sol_perfect.y[1,:]  # phi
    Y_perfect_full[2,:] = sol_perfect.y[1,:]  # psi = phi for perfect fluid
    Y_perfect_full[3,:] = sol_perfect.y[2,:]  # dr
    Y_perfect_full[4,:] = sol_perfect.y[3,:]  # dm
    Y_perfect_full[5,:] = sol_perfect.y[4,:]  # vr
    Y_perfect_full[6,:] = sol_perfect.y[5,:]  # vm
    # Y_perfect_full[7:,:] remain zero (no multipoles before recombination)

    # Convert forward solution (a-coordinates) to s-coordinates for consistency
    Y_rec_to_swap_s = sol_rec_to_swap.y.copy()
    Y_rec_to_swap_s[0, :] = 1. / Y_rec_to_swap_s[0, :]  # Convert a to s

    # Combine: Big Bang -> recombination -> swaptime -> FCB
    t_left = np.concatenate((sol_perfect.t, sol_rec_to_swap.t, sol_fcb_to_swap.t[::-1]))
    Y_left = np.concatenate((Y_perfect_full, Y_rec_to_swap_s, sol_fcb_to_swap.y[:, ::-1]), axis=1)

    print(f"Combined left half: {Y_left.shape[1]} points from eta=0 to FCB")

    # Create reflection for palindromic universe
    t_right = 2 * fcb_time - t_left[::-1]

    # Define symmetry matrix for reflection
    symm = np.ones(num_variables+1)
    symm[[5, 6]] = -1  # vr and vm are antisymmetric
    for l_idx in range(l_max - 1):
        l = l_idx + 2
        if l % 2 != 0: symm[7 + l_idx] = -1  # Odd multipoles are antisymmetric
    S = np.diag(symm)

    Y_right = S @ Y_left[:, ::-1]

    # Combine left and right
    t_full = np.concatenate((t_left, t_right))
    Y_full = np.concatenate((Y_left, Y_right), axis=1)

    print(f"Full solution: {Y_full.shape[1]} points from Big Bang to Big Crunch")

    solutions.append({'t': t_full, 'Y': Y_full, 'k': k, 'mode_idx': mode_idx})

# =============================================================================
# 4. PLOTTING
# =============================================================================

print("\n--- Plotting solutions ---")

fig, axes = plt.subplots(num_modes_to_plot, 1, figsize=(12, 10), sharex=True)
if num_modes_to_plot == 1: axes = [axes]

labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$', r'$\phi$', r'$\psi$']
indices = [5, 3, 6, 4, 1, 2]
colors = ['blue', 'red', 'green', 'orange', 'magenta', 'cyan']

for i, sol_dict in enumerate(solutions):
    ax = axes[i]
    sol = sol_dict

    for label, index, color in zip(labels, indices, colors):
        ax.plot(sol['t'], sol['Y'][index, :], label=label, color=color, linewidth=1.2)

    ax.axvline(fcb_time, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='FCB')
    ax.axvline(recConformalTime, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Rec')
    ax.axvline(swaptime, color='brown', linestyle='-.', linewidth=1.5, alpha=0.7, label='Swap')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(f'n={sol["mode_idx"]} (k={sol["k"]:.4f})', fontsize=11)
    ax.set_xlim(0, 2 * fcb_time * 1.05)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[0].legend(loc='upper right', ncol=4, fontsize=9)
axes[-1].set_xlabel(r'Conformal Time $\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Perturbation Solutions: Matching at Swaptime', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

output_file = folder_path + 'perturbation_solutions.pdf'
plt.savefig(output_file)
print(f"\nFigure saved to: {output_file}")
plt.show()

print("\n" + "="*80)
print("Calculation complete!")
print("="*80)
