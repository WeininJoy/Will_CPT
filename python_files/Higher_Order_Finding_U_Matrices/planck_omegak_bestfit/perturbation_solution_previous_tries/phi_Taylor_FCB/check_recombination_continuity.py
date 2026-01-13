# -*- coding: utf-8 -*-
"""
Diagnostic script to check continuity of perturbation solutions at recombination.
This compares the solutions from:
1. Perfect fluid integration (Big Bang → Recombination)
2. Backward integration using transfer matrices (FCB → Recombination)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =============================================================================
# SETUP: Parameters and Constants
# =============================================================================
nu_spacing = 4

print("=" * 80)
print("CHECKING CONTINUITY AT RECOMBINATION")
print("=" * 80)

folder_path = './'
folder_path_matrices = folder_path + 'data_allowedK/'
folder_path_timeseries = folder_path + 'data_allowedK_timeseries/'

lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
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

mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915

atol = 1e-13
rtol = 1e-13
num_variables = 75
swaptime = 2
H0 = 1/np.sqrt(3*OmegaLambda)

# Background equations
def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5

a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4

print('Performing background integration...')
sol_a = solve_ivp(da_dt, [t0, swaptime], [a_Bang], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)
print('Background integration done')

a_rec = 1./(1+z_rec)
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")
print(f"a_rec = {a_rec}")

# Perfect fluid ODE
def dX1_dt(t, X, k):
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k**2)*X[4])
    dmdot = 3*phidot + X[5]*(k**2)
    vrdot = -(X[1] + X[2]/4)
    vmdot = - (adot/X[0])*X[5] - X[1]

    return [adot, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# Load data
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

t_grid = np.load(folder_path_timeseries + 't_grid.npy')
allowedK = np.load(folder_path_timeseries + 'L70_kvalues.npy')
all_ABC_solutions = np.load(folder_path_timeseries + 'L70_ABC_solutions.npy')
all_DEF_solutions = np.load(folder_path_timeseries + 'L70_DEF_solutions.npy')
all_GHI_solutions = np.load(folder_path_timeseries + 'L70_GHI_solutions.npy')

ABCmatrices = np.load(folder_path_matrices+'L70_ABCmatrices.npy')
DEFmatrices = np.load(folder_path_matrices+'L70_DEFmatrices.npy')
GHIvectors = np.load(folder_path_matrices+'L70_GHIvectors.npy')
X1matrices = np.load(folder_path_matrices + 'L70_X1matrices.npy')
X2matrices = np.load(folder_path_matrices + 'L70_X2matrices.npy')
recValues = np.load(folder_path_matrices + 'L70_recValues.npy')

Amatrices = ABCmatrices[:, 0:6, :]
Bmatrices = ABCmatrices[:, 6:8, :]
Dmatrices = DEFmatrices[:, 0:6, :]
Ematrices = DEFmatrices[:, 6:8, :]

print(f"Loaded {len(allowedK)} k values")
print(f"Time grid: {len(t_grid)} points from eta={t_grid[0]:.4f} to eta={t_grid[-1]:.4f}")

# =============================================================================
# Check first 3 modes
# =============================================================================
print("\n" + "=" * 80)
print("CHECKING CONTINUITY FOR FIRST 3 MODES")
print("=" * 80)

num_modes_to_check = min(3, len(allowedK))

for i in range(num_modes_to_check):
    k_index = i + 3
    k = allowedK[k_index]

    print(f"\n{'-' * 80}")
    print(f"MODE n={i+1}, k={k:.6f} (index {k_index})")
    print(f"{'-' * 80}")

    # Get matrices
    A = Amatrices[k_index]
    D = Dmatrices[k_index]
    X1 = X1matrices[k_index]
    X2 = X2matrices[k_index]
    recs_vec = recValues[k_index]

    # Calculate x^∞
    GX3 = np.zeros((6,4))
    GX3[:,2] = GHIvectors[k_index][0:6]

    M_matrix = (A @ X1 + D @ X2)[[0,2,4,5], :]  # Only use independent components [phi,dr,vr,vm]
    x_rec = [recs_vec[1], recs_vec[2], recs_vec[4], recs_vec[5]]  # Only use independent components from stored values
    print(f"M_matrix: {M_matrix}, x_rec: {x_rec}")
    x_inf = np.linalg.solve(M_matrix, x_rec)
    print(f"x^∞ = {x_inf}")

    # Calculate coefficients
    x_prime_coeffs = X1 @ x_inf
    y_prime_coeffs = X2 @ x_inf

    # Reconstruct solution at recombination from backward integration
    ABC_sols_k = all_ABC_solutions[k_index]
    DEF_sols_k = all_DEF_solutions[k_index]

    # Interpolate to get values at exact recombination time
    from scipy.interpolate import interp1d

    print(f"  Recombination time: {recConformalTime:.6f}")
    print(f"  t_grid range: [{t_grid[0]:.6f}, {t_grid[-1]:.6f}]")

    # Check if recombination time is within t_grid range
    if recConformalTime < t_grid[0]:
        print(f"  WARNING: Recombination time is BEFORE t_grid starts!")
        print(f"    Need to extrapolate or extend t_grid")
        # Use the first point as approximation
        rec_idx = 0
        Y_backward_at_rec = np.einsum('ij,j->i', ABC_sols_k[:, :, rec_idx], x_prime_coeffs) + \
                            np.einsum('ij,j->i', DEF_sols_k[:, :, rec_idx], y_prime_coeffs)
    elif recConformalTime > t_grid[-1]:
        print(f"  WARNING: Recombination time is AFTER t_grid ends!")
        rec_idx = -1
        Y_backward_at_rec = np.einsum('ij,j->i', ABC_sols_k[:, :, rec_idx], x_prime_coeffs) + \
                            np.einsum('ij,j->i', DEF_sols_k[:, :, rec_idx], y_prime_coeffs)
    else:
        # Reconstruct full solution at all times
        Y_backward_full = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                         np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs)

        # Interpolate each variable to recombination time
        Y_backward_at_rec = np.zeros(num_variables)
        for var_idx in range(num_variables):
            interp_func = interp1d(t_grid, Y_backward_full[var_idx, :], kind='cubic', fill_value='extrapolate')
            Y_backward_at_rec[var_idx] = interp_func(recConformalTime)

        print(f"  Interpolated to exact recombination time")

    # The backward solution gives [phi, psi, dr, dm, vr, vm, Fr2, Fr3, ...]
    # Extract the base variables
    phi_backward = Y_backward_at_rec[0]
    psi_backward = Y_backward_at_rec[1]
    dr_backward = Y_backward_at_rec[2]
    dm_backward = Y_backward_at_rec[3]
    vr_backward = Y_backward_at_rec[4]
    vm_backward = Y_backward_at_rec[5]

    print(f"\n  BACKWARD INTEGRATION AT RECOMBINATION:")
    print(f"    phi = {phi_backward:.10f}")
    print(f"    psi = {psi_backward:.10f}")
    print(f"    dr  = {dr_backward:.10f}")
    print(f"    dm  = {dm_backward:.10f}")
    print(f"    vr  = {vr_backward:.10f}")
    print(f"    vm  = {vm_backward:.10f}")

    # ===================================================================
    # PERFECT FLUID INTEGRATION
    # ===================================================================

    # Taylor expansion coefficients
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

    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5

    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2))
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)
    phi0_guess = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4

    def solve_for_phi0(phi0_input):
        X0 = [a_Bang, phi0_input, dr0, dm0, vr0, vm0]
        sol3 = solve_ivp(lambda t, X: dX1_dt(t, X, k), [t0, recConformalTime], X0,
                        method='LSODA', atol=atol, rtol=rtol)
        return sol3

    def residual(phi0_input):
        sol3 = solve_for_phi0(phi0_input)
        a_rec = sol3.y[0, -1]
        phi_rec = sol3.y[1, -1]
        dr_rec = sol3.y[2, -1]
        dm_rec = sol3.y[3, -1]
        vr_rec = sol3.y[4, -1]
        vm_rec = sol3.y[5, -1]
        t_rec = sol3.t[-1]

        adot_rec = da_dt(t_rec, a_rec)
        phi_constraint = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
            (-3*adot_rec/a_rec*vm_rec + dm_rec)*OmegaM/a_rec +
            (-4*adot_rec/a_rec*vr_rec + dr_rec)*OmegaR/a_rec**2
        )

        return phi_rec - phi_constraint

    try:
        result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5], method='brentq', xtol=1e-10)
        phi0_optimal = result.root
    except ValueError:
        result = root_scalar(residual, x0=phi0_guess, x1=phi0_guess*1.1, method='secant', xtol=1e-10)
        phi0_optimal = result.root

    sol_perfect = solve_for_phi0(phi0_optimal)

    a_perfect = sol_perfect.y[0, -1]
    phi_perfect = sol_perfect.y[1, -1]
    dr_perfect = sol_perfect.y[2, -1]
    dm_perfect = sol_perfect.y[3, -1]
    vr_perfect = sol_perfect.y[4, -1]
    vm_perfect = sol_perfect.y[5, -1]

    # Calculate psi from perfect fluid
    adot_perfect = da_dt(recConformalTime, a_perfect)
    psi_perfect = phi_perfect  # In perfect fluid, psi = phi (no anisotropic stress)

    print(f"\n  PERFECT FLUID AT RECOMBINATION:")
    print(f"    a   = {a_perfect:.10f}")
    print(f"    phi = {phi_perfect:.10f}")
    print(f"    psi = {psi_perfect:.10f}")
    print(f"    dr  = {dr_perfect:.10f}")
    print(f"    dm  = {dm_perfect:.10f}")
    print(f"    vr  = {vr_perfect:.10f}")
    print(f"    vm  = {vm_perfect:.10f}")

    # ===================================================================
    # COMPARISON
    # ===================================================================

    print(f"\n  COMPARISON (Backward - Perfect):")
    print(f"    Δphi = {phi_backward - phi_perfect:.6e}  (rel: {abs(phi_backward - phi_perfect)/abs(phi_perfect):.6e})")
    print(f"    Δpsi = {psi_backward - psi_perfect:.6e}  (rel: {abs(psi_backward - psi_perfect)/abs(psi_perfect):.6e})")
    print(f"    Δdr  = {dr_backward - dr_perfect:.6e}  (rel: {abs(dr_backward - dr_perfect)/abs(dr_perfect):.6e})")
    print(f"    Δdm  = {dm_backward - dm_perfect:.6e}  (rel: {abs(dm_backward - dm_perfect)/abs(dm_perfect):.6e})")
    print(f"    Δvr  = {vr_backward - vr_perfect:.6e}  (rel: {abs(vr_backward - vr_perfect)/abs(vr_perfect):.6e})")
    print(f"    Δvm  = {vm_backward - vm_perfect:.6e}  (rel: {abs(vm_backward - vm_perfect)/abs(vm_perfect):.6e})")

    # Also compare with stored recValues
    print(f"\n  STORED RECVALUES:")
    print(f"    phi = {recs_vec[0]:.10f}")
    print(f"    psi = {recs_vec[1]:.10f}")
    print(f"    dr  = {recs_vec[2]:.10f}")
    print(f"    dm  = {recs_vec[3]:.10f}")
    print(f"    vr  = {recs_vec[4]:.10f}")
    print(f"    vm  = {recs_vec[5]:.10f}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
