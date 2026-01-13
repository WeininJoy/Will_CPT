# -*- coding: utf-8 -*-
"""
This script checks the continuity of perturbation solutions at recombination time.
It compares the perfect fluid solution (forward from Big Bang) with the
reconstructed solution (backward from FCB) at the recombination conformal time.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =============================================================================
# 1. SETUP: Parameters and Constants (Same as the original script)
# =============================================================================
nu_spacing = 4

print("--- Setting up parameters and functions ---")
folder_path = f'./nu_spacing{nu_spacing}_intK/'
folder_path_matrices = folder_path + 'data_allowedK/'
folder_path_timeseries = folder_path + 'data_allowedK_timeseries/'

## --- Best-fit parameters ---
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

# Best-fit parameters
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915

# Set tolerances
atol = 1e-13
rtol = 1e-13
num_variables = 75
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda)
Hinf = H0*np.sqrt(OmegaLambda)

# =============================================================================
# 2. BACKGROUND EQUATIONS
# =============================================================================

def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8

# Set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR)
szero = - OmegaM/(4*OmegaR)
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR)
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    fcb_time = None

endtime = fcb_time - deltaeta

# =============================================================================
# 3. RECOMBINATION CONFORMAL TIME
# =============================================================================

s_rec = 1+z_rec
recScaleFactorDifference = abs(sol.y[0] - s_rec)
recConformalTime = sol.t[recScaleFactorDifference.argmin()]
print(f'Conformal time at recombination: {recConformalTime}')

# Perfect fluid ODE
def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    sigmadot = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaK+OmegaM*np.exp(sigma)+OmegaR*np.exp(2*sigma))
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4)
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

# =============================================================================
# 4. DATA LOADING
# =============================================================================

print("\n--- Loading pre-computed time-dependent transfer matrices ---")
try:
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

    print(f"Loaded solution histories for {len(allowedK)} allowed K values.")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

Amatrices = ABCmatrices[:, 0:6, :]
Bmatrices = ABCmatrices[:, 6:8, :]
Cmatrices = ABCmatrices[:, 8:num_variables, :]
Dmatrices = DEFmatrices[:, 0:6, :]
Ematrices = DEFmatrices[:, 6:8, :]
Fmatrices = DEFmatrices[:, 8:num_variables, :]

# =============================================================================
# 5. CHECK CONTINUITY AT RECOMBINATION
# =============================================================================

print("\n" + "="*70)
print("CHECKING CONTINUITY AT RECOMBINATION TIME")
print("="*70)

num_modes_to_check = min(10, len(allowedK))
continuity_results = []

for i in range(num_modes_to_check):
    k_index = i + 3
    k = allowedK[k_index]
    print(f"\n{'='*70}")
    print(f"Mode n={i+1} with k={k:.6f} (index {k_index})")
    print(f"{'='*70}")

    # Get matrices
    A = Amatrices[k_index]
    B = Bmatrices[k_index]
    C = Cmatrices[k_index]
    D = Dmatrices[k_index]
    E = Ematrices[k_index]
    F = Fmatrices[k_index]
    X1 = X1matrices[k_index]
    X2 = X2matrices[k_index]
    recs_vec = recValues[k_index]

    # Calculate x^∞
    GX3 = np.zeros((6,4))
    GX3[:,2] = GHIvectors[k_index][0:6]

    M_matrix = (A @ X1 + D @ X2)[2:6, :]
    x_rec = recs_vec[2:6]
    x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]

    # Calculate coefficients
    x_prime_coeffs = X1 @ x_inf
    y_prime_coeffs = X2 @ x_inf

    # RECONSTRUCT backward solution
    ABC_sols_k = all_ABC_solutions[k_index]
    DEF_sols_k = all_DEF_solutions[k_index]
    GHI_sols_k = all_GHI_solutions[k_index]

    # Test with GHI term included
    Y_reconstructed_with_GHI = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                                np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs) + GHI_sols_k

    # Test without GHI term (as in original code)
    Y_reconstructed_without_GHI = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                                   np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs)

    # Get value at recombination time (first point in t_grid)
    # The t_grid should start at recConformalTime
    rec_idx = 0  # First point in the time series
    t_at_rec = t_grid[rec_idx]

    print(f"t_grid[0] = {t_grid[0]:.10f}")
    print(f"recConformalTime = {recConformalTime:.10f}")
    print(f"Difference in time: {abs(t_grid[0] - recConformalTime):.2e}")

    # Values from reconstructed solution at recombination
    # Y_reconstructed has shape (num_vars, num_times)
    # The variables are: [phi, psi, dr, dm, vr, vm, fr2, ..., higher multipoles]
    backward_at_rec_with_GHI = Y_reconstructed_with_GHI[:, rec_idx]
    backward_at_rec_without_GHI = Y_reconstructed_without_GHI[:, rec_idx]

    # FORWARD solution from perfect fluid
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))

    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda)

    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda)

    vr1 = -1/2
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda)

    vm1 = -1/2
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda)

    s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3
    sigma0 = np.log(s0)
    phi0 = 1 + phi1*t0 + phi2*t0**2
    dr0 = -2 + dr1*t0 + dr2*t0**2
    dm0 = -1.5 + dm1*t0 + dm2*t0**2
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3

    Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
    sol_perfect = solve_ivp(dX_perfect_sigma, [t0, recConformalTime], Y0_perfect,
                            dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

    # Evaluate forward solution at recombination time
    forward_at_rec = sol_perfect.sol(recConformalTime)

    # The forward solution has variables: [sigma, phi, dr, dm, vr, vm]
    # The backward solution has variables: [phi, psi, dr, dm, vr, vm, fr2, ...]
    # We need to compare: phi, dr, dm, vr, vm (psi = phi in perfect fluid)

    # Map variables between the two solutions
    # forward_at_rec: [sigma, phi, dr, dm, vr, vm]
    # backward_at_rec: [phi, psi, dr, dm, vr, vm, ...]

    variables = ['phi', 'psi', 'dr', 'dm', 'vr', 'vm']
    forward_indices = [1, 1, 2, 3, 4, 5]  # psi = phi in forward solution
    backward_indices = [0, 1, 2, 3, 4, 5]

    # Compare WITHOUT GHI term first
    print(f"\n--- WITHOUT GHI term (current implementation) ---")
    print(f"{'Variable':<10} {'Forward':<20} {'Backward':<20} {'Difference':<20} {'Rel. Error':<15}")
    print("-" * 85)

    differences_no_GHI = []
    rel_errors_no_GHI = []

    for var, fwd_idx, bwd_idx in zip(variables, forward_indices, backward_indices):
        fwd_val = forward_at_rec[fwd_idx]
        bwd_val = backward_at_rec_without_GHI[bwd_idx]
        diff = fwd_val - bwd_val

        max_abs = max(abs(fwd_val), abs(bwd_val))
        if max_abs > 1e-15:
            rel_err = abs(diff) / max_abs
        else:
            rel_err = abs(diff)

        differences_no_GHI.append(diff)
        rel_errors_no_GHI.append(rel_err)

        print(f"{var:<10} {fwd_val:<20.10e} {bwd_val:<20.10e} {diff:<20.10e} {rel_err:<15.6e}")

    print(f"Maximum relative error: {max(rel_errors_no_GHI):.6e}")

    # Compare WITH GHI term
    print(f"\n--- WITH GHI term (testing fix) ---")
    print(f"{'Variable':<10} {'Forward':<20} {'Backward':<20} {'Difference':<20} {'Rel. Error':<15}")
    print("-" * 85)

    differences_with_GHI = []
    rel_errors_with_GHI = []

    for var, fwd_idx, bwd_idx in zip(variables, forward_indices, backward_indices):
        fwd_val = forward_at_rec[fwd_idx]
        bwd_val = backward_at_rec_with_GHI[bwd_idx]
        diff = fwd_val - bwd_val

        max_abs = max(abs(fwd_val), abs(bwd_val))
        if max_abs > 1e-15:
            rel_err = abs(diff) / max_abs
        else:
            rel_err = abs(diff)

        differences_with_GHI.append(diff)
        rel_errors_with_GHI.append(rel_err)

        print(f"{var:<10} {fwd_val:<20.10e} {bwd_val:<20.10e} {diff:<20.10e} {rel_err:<15.6e}")

    print(f"Maximum relative error: {max(rel_errors_with_GHI):.6e}")

    # Store results for both cases
    continuity_results.append({
        'k': k,
        'k_index': k_index,
        'variables': variables,
        'forward': [forward_at_rec[idx] for idx in forward_indices],
        'backward_no_GHI': [backward_at_rec_without_GHI[idx] for idx in backward_indices],
        'backward_with_GHI': [backward_at_rec_with_GHI[idx] for idx in backward_indices],
        'differences_no_GHI': differences_no_GHI,
        'rel_errors_no_GHI': rel_errors_no_GHI,
        'max_rel_error_no_GHI': max(rel_errors_no_GHI),
        'differences_with_GHI': differences_with_GHI,
        'rel_errors_with_GHI': rel_errors_with_GHI,
        'max_rel_error_with_GHI': max(rel_errors_with_GHI)
    })

# =============================================================================
# 6. SUMMARY AND VISUALIZATION
# =============================================================================

print("\n" + "="*70)
print("SUMMARY OF CONTINUITY CHECK")
print("="*70)

print("\nWITHOUT GHI term (current implementation):")
for result in continuity_results:
    print(f"k = {result['k']:.6f} (n={result['k_index']-2}): Max rel. error = {result['max_rel_error_no_GHI']:.6e}")

print("\nWITH GHI term (proposed fix):")
for result in continuity_results:
    print(f"k = {result['k']:.6f} (n={result['k_index']-2}): Max rel. error = {result['max_rel_error_with_GHI']:.6e}")

# Plot the relative errors for all modes
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Plot 1: Relative errors WITHOUT GHI term
ax1 = axes[0]
k_values = [r['k'] for r in continuity_results]
for i, var in enumerate(variables):
    rel_errs = [r['rel_errors_no_GHI'][i] for r in continuity_results]
    ax1.plot(k_values, rel_errs, 'o-', label=var, linewidth=2, markersize=6)

ax1.set_xlabel('k', fontsize=14)
ax1.set_ylabel('Relative Error', fontsize=14)
ax1.set_title('WITHOUT GHI term: Relative Errors at Recombination', fontsize=16)
ax1.set_yscale('log')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(1e-13, color='gray', linestyle='--', alpha=0.5)

# Plot 2: Relative errors WITH GHI term
ax2 = axes[1]
for i, var in enumerate(variables):
    rel_errs = [r['rel_errors_with_GHI'][i] for r in continuity_results]
    ax2.plot(k_values, rel_errs, 'o-', label=var, linewidth=2, markersize=6)

ax2.set_xlabel('k', fontsize=14)
ax2.set_ylabel('Relative Error', fontsize=14)
ax2.set_title('WITH GHI term: Relative Errors at Recombination', fontsize=16)
ax2.set_yscale('log')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(1e-13, color='gray', linestyle='--', alpha=0.5)

# Plot 3: Comparison of maximum errors
ax3 = axes[2]
max_errors_no_GHI = [r['max_rel_error_no_GHI'] for r in continuity_results]
max_errors_with_GHI = [r['max_rel_error_with_GHI'] for r in continuity_results]
ax3.plot(k_values, max_errors_no_GHI, 'ro-', linewidth=2, markersize=8, label='Without GHI')
ax3.plot(k_values, max_errors_with_GHI, 'go-', linewidth=2, markersize=8, label='With GHI')
ax3.set_xlabel('k', fontsize=14)
ax3.set_ylabel('Maximum Relative Error', fontsize=14)
ax3.set_title('Comparison: Maximum Relative Error per Mode', fontsize=16)
ax3.set_yscale('log')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.axhline(1e-13, color='gray', linestyle='--', alpha=0.5, label='rtol/atol')

plt.tight_layout()
plt.savefig(folder_path + 'continuity_check_at_recombination.pdf')
print(f"\nPlot saved to: {folder_path}continuity_check_at_recombination.pdf")
plt.show()

print("\n" + "="*70)
print("Continuity check complete!")
print("="*70)
