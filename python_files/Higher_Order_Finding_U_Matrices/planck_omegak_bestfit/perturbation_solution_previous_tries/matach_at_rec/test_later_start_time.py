# -*- coding: utf-8 -*-
"""
Test starting at a later time (t0 = 1e-6 instead of 1e-7)
to avoid numerical issues near the singularity (a->0)
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root_scalar
import numpy as np
from math import *

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
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

atol = 1e-13;
rtol = 1e-13;
H0 = 1/np.sqrt(3*OmegaLambda);

def da_dt(t, a):
    return H0*np.sqrt((OmegaLambda*a**4 + OmegaK*a**2 + OmegaM*a + OmegaR))

# Background initial conditions coefficients
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda));
a2 = OmegaM/(12*OmegaLambda);
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2));
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2);
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2));
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3);

def dX1_dt(t, X, k_value):
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k_value**2)*X[4]);
    dmdot = 3*phidot + X[5]*(k_value**2);
    vrdot = -(X[1] + X[2]/4);
    vmdot = - (adot/X[0])*X[5] - X[1];

    return [adot, phidot, drdot, dmdot, vrdot, vmdot]

# Function to compute initial conditions and derivatives
def compute_initial_conditions(t0, k):
    """Compute initial conditions at given t0 using power series."""

    # Background
    a_t0 = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6
    adot_t0 = a1 + 2*a2*t0 + 3*a3*t0**2 + 4*a4*t0**3 + 5*a5*t0**4 + 6*a6*t0**5

    # Perturbations
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)

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

    # Initial values
    phi0 = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4
    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5

    # Derivatives
    phi_dot = phi1 + 2*phi2*t0 + 3*phi3*t0**2 + 4*phi4*t0**3
    dr_dot = dr1 + 2*dr2*t0 + 3*dr3*t0**2 + 4*dr4*t0**3
    dm_dot = dm1 + 2*dm2*t0 + 3*dm3*t0**2 + 4*dm4*t0**3
    vr_dot = vr1 + 2*vr2*t0 + 3*vr3*t0**2 + 4*vr4*t0**3 + 5*vr5*t0**4
    vm_dot = vm1 + 2*vm2*t0 + 3*vm3*t0**2 + 4*vm4*t0**3 + 5*vm5*t0**4

    return {
        'a': a_t0, 'phi': phi0, 'dr': dr0, 'dm': dm0, 'vr': vr0, 'vm': vm0,
        'adot': adot_t0, 'phidot': phi_dot, 'drdot': dr_dot, 'dmdot': dm_dot,
        'vrdot': vr_dot, 'vmdot': vm_dot
    }

# Get recombination time using t0=1e-7 for background
t0_bg = 1e-7
a_Bang_bg = a1*t0_bg + a2*t0_bg**2 + a3*t0_bg**3 + a4*t0_bg**4 + a5*t0_bg**5 + a6*t0_bg**6
sol_a = solve_ivp(da_dt, [t0_bg, 2], [a_Bang_bg], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)
a_rec = 1./(1+z_rec)
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

# Test with different starting times
t0_values = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
k = 1.0

print(f"\n{'='*80}")
print(f"COMPARING INITIAL CONDITIONS AT DIFFERENT STARTING TIMES")
print(f"k = {k}")
print(f"{'='*80}\n")

print(f"{'t0':<12} {'phi(t0)':<12} {'phi_dot(t0)':<15} {'a(t0)':<15} {'|phi_dot|/|phi-1|':<15}")
print(f"{'-'*80}")

for t0 in t0_values:
    ic = compute_initial_conditions(t0, k)
    ratio = abs(ic['phidot']) / abs(ic['phi'] - 1) if abs(ic['phi'] - 1) > 1e-15 else float('inf')
    print(f"{t0:<12.2e} {ic['phi']:<12.6f} {ic['phidot']:<15.6e} {ic['a']:<15.6e} {ratio:<15.2f}")

print(f"\nNote: Theoretically, phi should start at 1 with phi_dot=0.")
print(f"The ratio |phi_dot|/|phi-1| should be small for a good starting point.")

# Now plot solutions starting from t0=1e-6 vs t0=1e-7
print(f"\n{'='*80}")
print(f"COMPARING SOLUTIONS: t0=1e-7 vs t0=1e-6")
print(f"{'='*80}\n")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for col, t0_test in enumerate([1e-7, 1e-6]):
    print(f"\nSolving for t0 = {t0_test:.0e}")

    ic = compute_initial_conditions(t0_test, k)

    # Initial conditions for ODE
    dr0 = ic['dr']
    dm0 = ic['dm']
    vr0 = ic['vr']
    vm0 = ic['vm']
    a_Bang = ic['a']
    phi0_guess = ic['phi']

    print(f"  Initial conditions:")
    print(f"    a(t0) = {a_Bang:.6e}")
    print(f"    phi(t0) = {phi0_guess:.6f}, phi_dot(t0) = {ic['phidot']:.6e}")

    def solve_for_phi0(phi0_input):
        X0 = [a_Bang, phi0_input, dr0, dm0, vr0, vm0];
        sol3 = solve_ivp(lambda t, X: dX1_dt(t, X, k), [t0_test, recConformalTime], X0,
                         method='LSODA', atol=atol, rtol=rtol, dense_output=True);
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

        adot_rec = da_dt(t_rec_temp, a_rec_temp)
        phi_constraint = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
            (-3*adot_rec/a_rec_temp*vm_rec_temp + dm_rec_temp)*OmegaM/a_rec_temp +
            (-4*adot_rec/a_rec_temp*vr_rec_temp + dr_rec_temp)*OmegaR/a_rec_temp**2
        )
        return phi_rec_temp - phi_constraint

    # Find optimal phi0
    try:
        result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5], method='brentq', xtol=1e-10)
        phi0_optimal = result.root
    except ValueError:
        result = root_scalar(residual, x0=phi0_guess, x1=phi0_guess*1.1, method='secant', xtol=1e-10)
        phi0_optimal = result.root

    print(f"  After root-finding: phi0 = {phi0_optimal:.6f}")

    # Solve with optimal phi0
    sol3_plot = solve_for_phi0(phi0_optimal)

    # Create time array
    t_plot = np.logspace(np.log10(t0_test), np.log10(recConformalTime), 2000)
    X_plot = sol3_plot.sol(t_plot)

    # Compute phi_dot along the solution
    phi_dot_plot = np.zeros_like(t_plot)
    for i, t in enumerate(t_plot):
        X_at_t = X_plot[:, i]
        derivs = dX1_dt(t, X_at_t, k)
        phi_dot_plot[i] = derivs[1]

    # Plot phi
    axes[0, col].semilogx(t_plot, X_plot[1], 'b-', linewidth=2)
    axes[0, col].scatter([t0_test], [phi0_optimal], color='red', s=100, zorder=5)
    axes[0, col].axhline(1, color='gray', linestyle=':', alpha=0.5, label='φ=1')
    axes[0, col].axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5)
    axes[0, col].set_ylabel(r'$\phi$', fontsize=14)
    axes[0, col].set_title(f't₀ = {t0_test:.0e}', fontsize=14, fontweight='bold')
    axes[0, col].grid(True, alpha=0.3, which='both')
    axes[0, col].legend()

    # Plot phi_dot
    axes[1, col].semilogx(t_plot, phi_dot_plot, 'r-', linewidth=2)
    axes[1, col].scatter([t0_test], [ic['phidot']], color='red', s=100, zorder=5)
    axes[1, col].axhline(0, color='gray', linestyle=':', alpha=0.5, label='φ̇=0')
    axes[1, col].axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5)
    axes[1, col].set_ylabel(r'$\dot{\phi}$', fontsize=14)
    axes[1, col].grid(True, alpha=0.3, which='both')
    axes[1, col].legend()

    # Plot density perturbations
    axes[2, col].semilogx(t_plot, X_plot[2], 'g-', linewidth=2, label=r'$\delta_r$')
    axes[2, col].semilogx(t_plot, X_plot[3], 'm-', linewidth=2, label=r'$\delta_m$')
    axes[2, col].axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5)
    axes[2, col].set_xlabel('Conformal time (log scale)', fontsize=12)
    axes[2, col].set_ylabel('Density perturbations', fontsize=12)
    axes[2, col].grid(True, alpha=0.3, which='both')
    axes[2, col].legend()

plt.suptitle(f'Comparison: Starting at t₀=10⁻⁷ vs t₀=10⁻⁶ (k={k})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(folder + 'comparison_starting_times.pdf', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {folder}comparison_starting_times.pdf")
plt.show()

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
