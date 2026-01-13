# -*- coding: utf-8 -*-
"""
Test starting at t0 = 1e-5 (in the plateau) instead of t0 = 1e-7
to see if constraint is better satisfied
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

# Background initial conditions coefficients
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2))
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3)

# Get recombination time
t0_bg = 1e-7
a_Bang_bg = a1*t0_bg + a2*t0_bg**2 + a3*t0_bg**3 + a4*t0_bg**4 + a5*t0_bg**5 + a6*t0_bg**6
sol_a = solve_ivp(da_dt, [t0_bg, 2], [a_Bang_bg], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)
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

# Function to compute initial conditions at given t0
def get_initial_conditions(t0, k):
    """Get initial conditions using power series expansion."""

    # Background
    a_t0 = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6

    # Perturbation coefficients
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2))
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)

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

    # Initial values
    phi0_guess = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4
    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5

    return a_t0, phi0_guess, dr0, dm0, vr0, vm0

# Test with different starting times
t0_values = [1e-7, 1e-6, 1e-5]
k_test_values = [0.5, 1.0]

print(f"\n{'='*80}")
print(f"COMPARING DIFFERENT STARTING TIMES")
print(f"{'='*80}\n")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

for col, t0 in enumerate(t0_values):
    print(f"\n{'='*80}")
    print(f"STARTING TIME: t0 = {t0:.0e}")
    print(f"{'='*80}")

    for row_offset, k in enumerate(k_test_values):
        print(f"\n  k = {k}")

        # Get initial conditions
        a_t0, phi0_guess, dr0, dm0, vr0, vm0 = get_initial_conditions(t0, k)

        print(f"    Initial conditions:")
        print(f"      a(t0) = {a_t0:.6e}")
        print(f"      phi0_guess = {phi0_guess:.6f}")

        def solve_for_phi0(phi0_input):
            X0 = [a_t0, phi0_input, dr0, dm0, vr0, vm0]
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

        print(f"    Optimal phi0 = {phi0_optimal:.6f}")

        # Solve with optimal phi0
        sol3_full = solve_for_phi0(phi0_optimal)

        # Create time array
        t_full = np.logspace(np.log10(t0), np.log10(recConformalTime), 1000)
        X_full = sol3_full.sol(t_full)

        # Compute constraint error
        relative_error = np.zeros_like(t_full)
        for i, t in enumerate(t_full):
            phi_constraint = compute_constraint(t, X_full[0, i], X_full[1, i],
                                               X_full[2, i], X_full[3, i],
                                               X_full[4, i], X_full[5, i], k)
            if abs(X_full[1, i]) > 1e-15:
                relative_error[i] = abs(X_full[1, i] - phi_constraint) / abs(X_full[1, i])
            else:
                relative_error[i] = 0

        # Check constraint at t0
        phi_constraint_t0 = compute_constraint(t0, a_t0, phi0_optimal, dr0, dm0, vr0, vm0, k)
        error_t0 = abs(phi0_optimal - phi_constraint_t0) / abs(phi0_optimal) if abs(phi0_optimal) > 1e-15 else 0
        print(f"    At t0: phi = {phi0_optimal:.6f}, constraint = {phi_constraint_t0:.6f}, rel. error = {error_t0:.6e}")

        # Check constraint at recombination
        print(f"    At recombination: rel. error = {relative_error[-1]:.6e}")

        # Find maximum error and where
        max_error = np.max(relative_error)
        max_idx = np.argmax(relative_error)
        print(f"    Maximum rel. error = {max_error:.6e} at t = {t_full[max_idx]:.6e}")

        # Plot relative error for this case
        if row_offset == 0:  # k = 0.5
            ax = fig.add_subplot(gs[0, col])
        else:  # k = 1.0
            ax = fig.add_subplot(gs[2, col])

        ax.loglog(t_full, relative_error, linewidth=2, color='blue')
        ax.axhline(0.01, color='red', linestyle=':', alpha=0.5, label='1% threshold')
        ax.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5, label='Recomb')
        ax.set_ylabel(f'Rel. error (k={k})', fontsize=10)
        if row_offset == 0:
            ax.set_title(f't₀ = {t0:.0e}', fontsize=12, fontweight='bold')
        if row_offset == 1:
            ax.set_xlabel('Conformal time', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8)

print(f"\n\nPlot saved to {folder}comparison_starting_times_detailed.pdf")
plt.suptitle('Constraint Satisfaction vs Starting Time', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(folder + 'comparison_starting_times_detailed.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
