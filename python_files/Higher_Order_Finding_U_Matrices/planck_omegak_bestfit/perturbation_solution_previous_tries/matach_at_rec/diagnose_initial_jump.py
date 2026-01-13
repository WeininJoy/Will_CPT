# -*- coding: utf-8 -*-
"""
Diagnostic script to investigate the jump at the Big Bang

This script examines:
1. Initial conditions and their derivatives
2. Very early time evolution (first few steps)
3. Consistency between initial conditions and ODE
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root_scalar
import numpy as np
from math import *

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
nu_spacing = 4
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

# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915

#set tolerances
atol = 1e-13;
rtol = 1e-13;
H0 = 1/np.sqrt(3*OmegaLambda);

def da_dt(t, a):
    return H0*np.sqrt((OmegaLambda*a**4 + OmegaK*a**2 + OmegaM*a + OmegaR))

t0 = 1e-7;

# Background initial conditions
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda));
a2 = OmegaM/(12*OmegaLambda);
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2));
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2);
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2));
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3);
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6;

#-------------------------------------------------------------------------------
# Define derivative functions
#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
# Analytical derivatives at t=t0
#-------------------------------------------------------------------------------
def analytical_derivatives_at_t0(k):
    """Compute analytical time derivatives at t=t0 using power series."""

    # Coefficients for perturbation variables
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

    # Analytical derivative at t=t0 (first derivative of power series)
    phi_dot_analytical = phi1 + 2*phi2*t0 + 3*phi3*t0**2 + 4*phi4*t0**3
    dr_dot_analytical = dr1 + 2*dr2*t0 + 3*dr3*t0**2 + 4*dr4*t0**3
    dm_dot_analytical = dm1 + 2*dm2*t0 + 3*dm3*t0**2 + 4*dm4*t0**3
    vr_dot_analytical = vr1 + 2*vr2*t0 + 3*vr3*t0**2 + 4*vr4*t0**3 + 5*vr5*t0**4
    vm_dot_analytical = vm1 + 2*vm2*t0 + 3*vm3*t0**2 + 4*vm4*t0**3 + 5*vm5*t0**4

    a_dot_analytical = a1 + 2*a2*t0 + 3*a3*t0**2 + 4*a4*t0**3 + 5*a5*t0**4 + 6*a6*t0**5

    return {
        'a_dot': a_dot_analytical,
        'phi_dot': phi_dot_analytical,
        'dr_dot': dr_dot_analytical,
        'dm_dot': dm_dot_analytical,
        'vr_dot': vr_dot_analytical,
        'vm_dot': vm_dot_analytical
    }

#-------------------------------------------------------------------------------
# Test for specific k value
#-------------------------------------------------------------------------------

k = 1.0  # Test k value

print(f"\n{'='*70}")
print(f"DIAGNOSING INITIAL CONDITIONS AND JUMP AT BIG BANG")
print(f"Testing for k = {k}")
print(f"{'='*70}\n")

# Get initial conditions
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

dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4;
dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4;
vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5;
vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5;

# Get phi0 using analytical expansion (before root-finding correction)
phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)
phi0_analytical = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4;

# Get analytical derivatives
analytical_derivs = analytical_derivatives_at_t0(k)

# Get ODE derivatives at t=t0 with analytical phi0
X0_analytical = [a_Bang, phi0_analytical, dr0, dm0, vr0, vm0]
ode_derivs_analytical = dX1_dt(t0, X0_analytical, k)

print(f"INITIAL CONDITIONS AT t0 = {t0:.2e}:")
print(f"  a(t0) = {a_Bang:.6e}")
print(f"  phi(t0) = {phi0_analytical:.6e}")
print(f"  dr(t0) = {dr0:.6e}")
print(f"  dm(t0) = {dm0:.6e}")
print(f"  vr(t0) = {vr0:.6e}")
print(f"  vm(t0) = {vm0:.6e}")

print(f"\nDERIVATIVES COMPARISON AT t0:")
print(f"{'Variable':<10} {'Analytical':<20} {'ODE':<20} {'Relative Diff':<15}")
print(f"{'-'*70}")
print(f"{'a_dot':<10} {analytical_derivs['a_dot']:<20.6e} {ode_derivs_analytical[0]:<20.6e} {abs((analytical_derivs['a_dot']-ode_derivs_analytical[0])/analytical_derivs['a_dot']):<15.6e}")
print(f"{'phi_dot':<10} {analytical_derivs['phi_dot']:<20.6e} {ode_derivs_analytical[1]:<20.6e} {abs((analytical_derivs['phi_dot']-ode_derivs_analytical[1])/analytical_derivs['phi_dot']) if analytical_derivs['phi_dot'] != 0 else abs(ode_derivs_analytical[1]):<15.6e}")
print(f"{'dr_dot':<10} {analytical_derivs['dr_dot']:<20.6e} {ode_derivs_analytical[2]:<20.6e} {abs((analytical_derivs['dr_dot']-ode_derivs_analytical[2])/analytical_derivs['dr_dot']):<15.6e}")
print(f"{'dm_dot':<10} {analytical_derivs['dm_dot']:<20.6e} {ode_derivs_analytical[3]:<20.6e} {abs((analytical_derivs['dm_dot']-ode_derivs_analytical[3])/analytical_derivs['dm_dot']):<15.6e}")
print(f"{'vr_dot':<10} {analytical_derivs['vr_dot']:<20.6e} {ode_derivs_analytical[4]:<20.6e} {abs((analytical_derivs['vr_dot']-ode_derivs_analytical[4])/analytical_derivs['vr_dot']):<15.6e}")
print(f"{'vm_dot':<10} {analytical_derivs['vm_dot']:<20.6e} {ode_derivs_analytical[5]:<20.6e} {abs((analytical_derivs['vm_dot']-ode_derivs_analytical[5])/analytical_derivs['vm_dot']):<15.6e}")

# Now solve and look at very early evolution
print(f"\n{'='*70}")
print(f"EARLY TIME EVOLUTION (first 10 time steps)")
print(f"{'='*70}\n")

# Integrate with many points near the beginning
sol = solve_ivp(lambda t, X: dX1_dt(t, X, k), [t0, 10*t0], X0_analytical,
                method='LSODA', atol=atol, rtol=rtol, dense_output=True)

# Get first few time steps
print(f"{'Time':<15} {'a':<15} {'phi':<15} {'dr':<15} {'dm':<15}")
print(f"{'-'*75}")
for i in range(min(10, len(sol.t))):
    print(f"{sol.t[i]:<15.6e} {sol.y[0,i]:<15.6e} {sol.y[1,i]:<15.6e} {sol.y[2,i]:<15.6e} {sol.y[3,i]:<15.6e}")

# Plot very early evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

t_early = np.linspace(t0, 100*t0, 1000)
X_early = sol.sol(t_early)

axes[0,0].plot(t_early, X_early[1], 'b-', linewidth=2)
axes[0,0].scatter([t0], [phi0_analytical], color='red', s=100, zorder=5, label=f'IC: {phi0_analytical:.4f}')
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel(r'$\phi$')
axes[0,0].set_title('Phi - Very Early Evolution')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

axes[0,1].plot(t_early, X_early[2], 'g-', linewidth=2)
axes[0,1].scatter([t0], [dr0], color='red', s=100, zorder=5, label=f'IC: {dr0:.4f}')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel(r'$\delta_r$')
axes[0,1].set_title('Delta_r - Very Early Evolution')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

axes[0,2].plot(t_early, X_early[3], 'm-', linewidth=2)
axes[0,2].scatter([t0], [dm0], color='red', s=100, zorder=5, label=f'IC: {dm0:.4f}')
axes[0,2].set_xlabel('Time')
axes[0,2].set_ylabel(r'$\delta_m$')
axes[0,2].set_title('Delta_m - Very Early Evolution')
axes[0,2].grid(True, alpha=0.3)
axes[0,2].legend()

axes[1,0].plot(t_early, X_early[4], 'c-', linewidth=2)
axes[1,0].scatter([t0], [vr0], color='red', s=100, zorder=5, label=f'IC: {vr0:.6f}')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel(r'$v_r$')
axes[1,0].set_title('v_r - Very Early Evolution')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend()

axes[1,1].plot(t_early, X_early[5], 'orange', linewidth=2)
axes[1,1].scatter([t0], [vm0], color='red', s=100, zorder=5, label=f'IC: {vm0:.6f}')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel(r'$v_m$')
axes[1,1].set_title('v_m - Very Early Evolution')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()

axes[1,2].plot(t_early, X_early[0], 'purple', linewidth=2)
axes[1,2].scatter([t0], [a_Bang], color='red', s=100, zorder=5, label=f'IC: {a_Bang:.6e}')
axes[1,2].set_xlabel('Time')
axes[1,2].set_ylabel('a')
axes[1,2].set_title('Scale Factor - Very Early Evolution')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].legend()

plt.suptitle(f'Very Early Time Evolution (t0 to 100×t0) for k = {k}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(folder + 'early_time_evolution_diagnostic.pdf', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {folder}early_time_evolution_diagnostic.pdf")
plt.show()

print(f"\n{'='*70}")
print(f"DIAGNOSTIC COMPLETE")
print(f"{'='*70}")
