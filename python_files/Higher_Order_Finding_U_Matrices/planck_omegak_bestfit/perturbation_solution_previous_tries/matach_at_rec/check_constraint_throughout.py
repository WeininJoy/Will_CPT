# -*- coding: utf-8 -*-
"""
Check whether the constraint is satisfied throughout the evolution
and identify where it breaks down
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root_scalar
import numpy as np
from math import *

folder = './'

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

mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915

atol = 1e-13
rtol = 1e-13
H0 = 1/np.sqrt(3*OmegaLambda)

def da_dt(t, a):
    return H0*np.sqrt((OmegaLambda*a**4 + OmegaK*a**2 + OmegaM*a + OmegaR))

t0 = 1e-7

# Background initial conditions
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2))
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6

# Get recombination time
sol_a = solve_ivp(da_dt, [t0, 2], [a_Bang], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)
a_rec = 1./(1+z_rec)
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

def dX1_dt(t, X, k_value):
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k_value**2)*X[4])
    dmdot = 3*phidot + X[5]*(k_value**2)
    vrdot = -(X[1] + X[2]/4)
    vmdot = - (adot/X[0])*X[5] - X[1]

    return [adot, phidot, drdot, dmdot, vrdot, vmdot]

def compute_constraint(t, a, phi, dr, dm, vr, vm, k_value):
    """Compute the constraint value."""
    adot = da_dt(t, a)
    phi_constraint = - 3*H0**2 / (2*(k_value**2 + 3*OmegaK*H0**2)) * (
        (-3*adot/a*vm + dm)*OmegaM/a +
        (-4*adot/a*vr + dr)*OmegaR/a**2
    )
    return phi_constraint

# Test for k=0.5 and k=1.0
k_test_values = [0.5, 1.0]

print(f"\n{'='*80}")
print(f"CHECKING CONSTRAINT SATISFACTION THROUGHOUT EVOLUTION")
print(f"{'='*80}\n")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

for plot_idx, k in enumerate(k_test_values):
    print(f"\nProcessing k = {k}")

    # Get initial conditions
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
                         method='LSODA', atol=atol, rtol=rtol, dense_output=True)
        return sol3

    def residual(phi0_input):
        sol3 = solve_for_phi0(phi0_input)
        a_rec_temp = sol3.y[0, -1]
        phi_rec_temp = sol3.y[1, -1]
        dr_rec_temp = sol3.y[2, -1]
        dm_rec_temp = sol3.y[3, -1]
        vr_rec_temp = sol3.y[4, -1]
        vm_rec_temp = sol3.y[5, -1]
        t_rec_temp = sol3.t[-1]

        phi_constraint = compute_constraint(t_rec_temp, a_rec_temp, phi_rec_temp,
                                           dr_rec_temp, dm_rec_temp, vr_rec_temp,
                                           vm_rec_temp, k)
        return phi_rec_temp - phi_constraint

    # Find optimal phi0
    try:
        result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5], method='brentq', xtol=1e-10)
        phi0_optimal = result.root
    except ValueError:
        result = root_scalar(residual, x0=phi0_guess, x1=phi0_guess*1.1, method='secant', xtol=1e-10)
        phi0_optimal = result.root

    # Solve with optimal phi0
    sol3_full = solve_for_phi0(phi0_optimal)

    # Create dense time array
    t_full = np.logspace(np.log10(t0), np.log10(recConformalTime), 2000)
    X_full = sol3_full.sol(t_full)

    # Find plateau and rescale
    phi_dot_full = np.zeros_like(t_full)
    for i, t in enumerate(t_full):
        X_at_t = X_full[:, i]
        derivs = dX1_dt(t, X_at_t, k)
        phi_dot_full[i] = derivs[1]

    abs_phi_dot = np.abs(phi_dot_full)
    plateau_idx = np.argmin(abs_phi_dot)
    t_plateau = t_full[plateau_idx]
    phi_plateau = X_full[1, plateau_idx]
    scale_factor = 1.0 / phi_plateau

    # Rescale
    X_rescaled = X_full.copy()
    X_rescaled[1, :] *= scale_factor  # phi
    X_rescaled[2, :] *= scale_factor  # dr
    X_rescaled[3, :] *= scale_factor  # dm
    X_rescaled[4, :] *= scale_factor  # vr
    X_rescaled[5, :] *= scale_factor  # vm

    print(f"  Plateau at t = {t_plateau:.6e}, scale factor = {scale_factor:.6f}")

    # Compute constraint along the solution
    phi_constraint_array = np.zeros_like(t_full)
    constraint_error = np.zeros_like(t_full)
    relative_error = np.zeros_like(t_full)

    for i, t in enumerate(t_full):
        phi_constraint_array[i] = compute_constraint(t, X_rescaled[0, i], X_rescaled[1, i],
                                                     X_rescaled[2, i], X_rescaled[3, i],
                                                     X_rescaled[4, i], X_rescaled[5, i], k)
        constraint_error[i] = abs(X_rescaled[1, i] - phi_constraint_array[i])
        if abs(X_rescaled[1, i]) > 1e-15:
            relative_error[i] = constraint_error[i] / abs(X_rescaled[1, i])
        else:
            relative_error[i] = 0

    # Find where constraint breaks down (relative error > 1%)
    breakdown_mask = relative_error > 0.01
    if np.any(breakdown_mask):
        breakdown_idx = np.where(breakdown_mask)[0][0]
        t_breakdown = t_full[breakdown_idx]
        print(f"  Constraint breaks down (>1% error) at t = {t_breakdown:.6e}")
    else:
        t_breakdown = None
        print(f"  Constraint satisfied throughout (error < 1%)")

    # Plot phi vs phi_constraint
    ax1 = fig.add_subplot(gs[0, plot_idx])
    ax1.semilogx(t_full, X_rescaled[1], 'b-', linewidth=2, label=r'$\phi$ (solution)')
    ax1.semilogx(t_full, phi_constraint_array, 'r--', linewidth=2, alpha=0.7, label=r'$\phi$ (constraint)')
    ax1.axvline(t_plateau, color='green', linestyle=':', alpha=0.5, label=f'Plateau')
    ax1.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5, label='Recombination')
    if t_breakdown:
        ax1.axvline(t_breakdown, color='red', linestyle='-.', alpha=0.5, label='Breakdown')
    ax1.set_ylabel(r'$\phi$', fontsize=12)
    ax1.set_title(f'k = {k}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=8)

    # Plot absolute error
    ax2 = fig.add_subplot(gs[1, plot_idx])
    ax2.loglog(t_full, constraint_error, 'purple', linewidth=2)
    ax2.axvline(t_plateau, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5)
    if t_breakdown:
        ax2.axvline(t_breakdown, color='red', linestyle='-.', alpha=0.5)
    ax2.set_ylabel(r'$|\phi - \phi_{\rm constraint}|$', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot relative error
    ax3 = fig.add_subplot(gs[2, plot_idx])
    ax3.loglog(t_full, relative_error, 'orange', linewidth=2)
    ax3.axhline(0.01, color='red', linestyle=':', alpha=0.5, label='1% threshold')
    ax3.axvline(t_plateau, color='green', linestyle=':', alpha=0.5)
    ax3.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5)
    if t_breakdown:
        ax3.axvline(t_breakdown, color='red', linestyle='-.', alpha=0.5)
    ax3.set_ylabel('Relative error', fontsize=12)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=8)

    # Plot scale factor a
    ax4 = fig.add_subplot(gs[3, plot_idx])
    ax4.loglog(t_full, X_rescaled[0], 'purple', linewidth=2)
    ax4.axvline(t_plateau, color='green', linestyle=':', alpha=0.5, label='Plateau')
    ax4.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5, label='Recombination')
    if t_breakdown:
        ax4.axvline(t_breakdown, color='red', linestyle='-.', alpha=0.5, label='Breakdown')
    ax4.set_xlabel('Conformal time (log scale)', fontsize=12)
    ax4.set_ylabel('Scale factor a', fontsize=12)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=8)

    # Print statistics
    print(f"  Error statistics:")
    print(f"    At plateau: absolute = {constraint_error[plateau_idx]:.6e}, relative = {relative_error[plateau_idx]:.6e}")
    print(f"    At recombination: absolute = {constraint_error[-1]:.6e}, relative = {relative_error[-1]:.6e}")
    print(f"    Maximum absolute error: {np.max(constraint_error):.6e}")
    print(f"    Maximum relative error: {np.max(relative_error):.6e}")

plt.suptitle('Constraint Satisfaction Throughout Evolution\n(φ vs φ_constraint)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(folder + 'constraint_check_throughout.pdf', dpi=300, bbox_inches='tight')
print(f"\n\nPlot saved to {folder}constraint_check_throughout.pdf")
plt.show()

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
