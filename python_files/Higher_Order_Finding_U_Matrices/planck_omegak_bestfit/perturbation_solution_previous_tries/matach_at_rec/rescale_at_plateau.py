# -*- coding: utf-8 -*-
"""
Find the plateau where phi_dot ≈ 0, rescale so phi=1 there,
and plot from the plateau onwards
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root_scalar
import numpy as np
from math import *

folder = './'
folder_path = folder + 'data_all_k/'

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

# Test for k=0.5 and k=1.0
k_test_values = [0.5, 1.0]
colors = ['blue', 'red']

print(f"\n{'='*80}")
print(f"FINDING PLATEAU AND RESCALING")
print(f"{'='*80}\n")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

for plot_idx, k in enumerate(k_test_values):
    print(f"\nProcessing k = {k}")

    # Get initial conditions using standard power series
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

        adot_rec = da_dt(t_rec_temp, a_rec_temp)
        phi_constraint = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
            (-3*adot_rec/a_rec_temp*vm_rec_temp + dm_rec_temp)*OmegaM/a_rec_temp +
            (-4*adot_rec/a_rec_temp*vr_rec_temp + dr_rec_temp)*OmegaR/a_rec_temp**2
        )
        return phi_rec_temp - phi_constraint

    # Find optimal phi0 with root-finding
    try:
        result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5], method='brentq', xtol=1e-10)
        phi0_optimal = result.root
    except ValueError:
        result = root_scalar(residual, x0=phi0_guess, x1=phi0_guess*1.1, method='secant', xtol=1e-10)
        phi0_optimal = result.root

    print(f"  phi0 (before rescaling) = {phi0_optimal:.6f}")

    # Solve with optimal phi0
    sol3_full = solve_for_phi0(phi0_optimal)

    # Create dense time array
    t_full = np.logspace(np.log10(t0), np.log10(recConformalTime), 2000)
    X_full = sol3_full.sol(t_full)

    # Compute phi_dot to find plateau
    phi_dot_full = np.zeros_like(t_full)
    for i, t in enumerate(t_full):
        X_at_t = X_full[:, i]
        derivs = dX1_dt(t, X_at_t, k)
        phi_dot_full[i] = derivs[1]

    # Find plateau: where |phi_dot| is minimized
    abs_phi_dot = np.abs(phi_dot_full)
    plateau_idx = np.argmin(abs_phi_dot)
    t_plateau = t_full[plateau_idx]
    phi_plateau = X_full[1, plateau_idx]

    print(f"  Plateau found at t = {t_plateau:.6e}")
    print(f"  phi at plateau = {phi_plateau:.6f}")
    print(f"  phi_dot at plateau = {phi_dot_full[plateau_idx]:.6e}")

    # Calculate rescaling factor
    scale_factor = 1.0 / phi_plateau

    print(f"  Rescaling factor = {scale_factor:.6f}")

    # Rescale all perturbations
    X_rescaled = X_full.copy()
    X_rescaled[1, :] *= scale_factor  # phi
    X_rescaled[2, :] *= scale_factor  # dr
    X_rescaled[3, :] *= scale_factor  # dm
    X_rescaled[4, :] *= scale_factor  # vr
    X_rescaled[5, :] *= scale_factor  # vm
    phi_dot_rescaled = phi_dot_full * scale_factor

    print(f"  After rescaling at plateau:")
    print(f"    phi = {X_rescaled[1, plateau_idx]:.6f}")
    print(f"    dr = {X_rescaled[2, plateau_idx]:.6f}")
    print(f"    dm = {X_rescaled[3, plateau_idx]:.6f}")
    print(f"    vr = {X_rescaled[4, plateau_idx]:.6e}")
    print(f"    vm = {X_rescaled[5, plateau_idx]:.6e}")

    # Plot from plateau onwards
    plot_start_idx = plateau_idx
    t_plot = t_full[plot_start_idx:]
    X_plot = X_rescaled[:, plot_start_idx:]
    phi_dot_plot = phi_dot_rescaled[plot_start_idx:]

    # Plot phi
    ax_phi = fig.add_subplot(gs[0, plot_idx])
    ax_phi.semilogx(t_plot, X_plot[1], color=colors[plot_idx], linewidth=2)
    ax_phi.scatter([t_plateau], [1.0], color='red', s=100, zorder=5, label=f'Plateau (t={t_plateau:.2e})')
    ax_phi.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax_phi.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Recombination')
    ax_phi.set_ylabel(r'$\phi$', fontsize=14)
    ax_phi.set_title(f'k = {k}', fontsize=14, fontweight='bold')
    ax_phi.grid(True, alpha=0.3, which='both')
    ax_phi.legend(fontsize=9)

    # Plot dr and dm
    ax_d = fig.add_subplot(gs[1, plot_idx])
    ax_d.semilogx(t_plot, X_plot[2], 'g-', linewidth=2, label=r'$\delta_r$')
    ax_d.semilogx(t_plot, X_plot[3], 'm-', linewidth=2, label=r'$\delta_m$')
    ax_d.scatter([t_plateau], [X_rescaled[2, plateau_idx]], color='green', s=80, zorder=5, marker='o')
    ax_d.scatter([t_plateau], [X_rescaled[3, plateau_idx]], color='magenta', s=80, zorder=5, marker='o')
    ax_d.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax_d.set_ylabel('Density perturbations', fontsize=12)
    ax_d.grid(True, alpha=0.3, which='both')
    ax_d.legend(fontsize=10)

    # Plot vr and vm
    ax_v = fig.add_subplot(gs[2, plot_idx])
    ax_v.semilogx(t_plot, X_plot[4], 'c-', linewidth=2, label=r'$v_r$')
    ax_v.semilogx(t_plot, X_plot[5], 'orange', linewidth=2, label=r'$v_m$')
    ax_v.scatter([t_plateau], [X_rescaled[4, plateau_idx]], color='cyan', s=80, zorder=5, marker='o')
    ax_v.scatter([t_plateau], [X_rescaled[5, plateau_idx]], color='orange', s=80, zorder=5, marker='o')
    ax_v.axvline(recConformalTime, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax_v.set_xlabel('Conformal time (log scale)', fontsize=12)
    ax_v.set_ylabel('Velocity perturbations', fontsize=12)
    ax_v.grid(True, alpha=0.3, which='both')
    ax_v.legend(fontsize=10)

plt.suptitle('Rescaled Perturbation Evolution from Plateau to Recombination\n(φ = 1 at plateau where φ̇ ≈ 0)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(folder + 'perturbation_evolution_rescaled_from_plateau.pdf', dpi=300, bbox_inches='tight')
print(f"\n\nPlot saved to {folder}perturbation_evolution_rescaled_from_plateau.pdf")
plt.show()

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
