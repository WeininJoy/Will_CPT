# -*- coding: utf-8 -*-
"""
Diagnostic: Why does v_rec not match when integrated forward?

This script tests if the issue is with:
1. The v_rec values themselves
2. The forward integration
3. The JKLMNOPQR matrix
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Load data
folder_path = './data/data_all_k/'
kvalues = np.load(folder_path + 'L70_kvalues.npy')
ABCmatrices = np.load(folder_path + 'L70_ABCmatrices.npy')
DEFmatrices = np.load(folder_path + 'L70_DEFmatrices.npy')
JKLmatrices = np.load(folder_path + 'L70_JKLmatrices.npy')
MNOmatrices = np.load(folder_path + 'L70_MNOmatrices.npy')
PQRmatrices = np.load(folder_path + 'L70_PQRmatrices.npy')
X1matrices = np.load(folder_path + 'L70_X1matrices.npy')
X2matrices = np.load(folder_path + 'L70_X2matrices.npy')
allowedK = np.load(folder_path + 'allowedK.npy')

# Parameters
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
mt, kt, h = 427.161507, 1.532563, 0.543442

def cosmological_parameters(mt, kt, h):
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3)) * Omega_gamma_h2/h**2
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

OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915
H0 = 1/np.sqrt(3*OmegaLambda)
swaptime = 2
num_variables = 75

# Compute recConformalTime
def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(a**2) + OmegaM/abs(a**3) + OmegaR/abs(a**4)))

t0 = 1e-5
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4

atol = 1e-13
rtol = 1e-13
sol_a = solve_ivp(da_dt, [t0, swaptime], [a_Bang], max_step=0.25e-4, method='LSODA', atol=atol, rtol=rtol)
a_rec = 1./(1+z_rec)
recScaleFactorDifference = abs(sol_a.y[0] - a_rec)
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]

print("="*80)
print("DIAGNOSTIC: Why v_rec Doesn't Match at Swaptime")
print("="*80)

# Test with mode index 2
mode_idx = 2
k = allowedK[mode_idx]
k_idx = np.argmin(np.abs(kvalues - k))
print(f"\nTesting mode_idx={mode_idx}, k={k:.6f}")
print(f"Using matrices from k_idx={k_idx}, k={kvalues[k_idx]:.6f}")

# Load matrices
ABC = ABCmatrices[k_idx]
DEF = DEFmatrices[k_idx]
JKL = JKLmatrices[k_idx]
MNO = MNOmatrices[k_idx]
PQR = PQRmatrices[k_idx]
X1 = X1matrices[k_idx]
X2 = X2matrices[k_idx]

# Construct matching matrices
JKLMNOPQR = np.hstack([JKL, MNO, PQR])
ABCDEF = np.hstack([ABC, DEF])
X_combined = np.vstack([X1, X2])

M_full = np.linalg.solve(JKLMNOPQR, ABCDEF @ X_combined)
M_reduced = M_full[[2, 3, 4, 5], :]

print(f"\nMatrix shapes:")
print(f"  JKLMNOPQR: {JKLMNOPQR.shape}")
print(f"  M_full: {M_full.shape}")
print(f"  M_reduced: {M_reduced.shape}")

# Compute perfect fluid solution
def dX1_dt(t, X):
    adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))
    rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
    rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))
    phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
    drdot = (4/3)*(3*phidot + (k**2)*X[4])
    dmdot = 3*phidot + X[5]*(k**2)
    vrdot = -(X[1] + X[2]/4)
    vmdot = - (adot/X[0])*X[5] - X[1]
    return [adot, phidot, drdot, dmdot, vrdot, vmdot]

# Power series coefficients
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
    X0_temp = [a_Bang, phi0_input, dr0, dm0, vr0, vm0]
    sol_temp = solve_ivp(dX1_dt, [t0, recConformalTime], X0_temp, method='LSODA', atol=atol, rtol=rtol)
    return sol_temp

def residual(phi0_input):
    sol_temp = solve_for_phi0(phi0_input)
    a_rec_temp = sol_temp.y[0, -1]
    phi_rec_temp = sol_temp.y[1, -1]
    dr_rec_temp = sol_temp.y[2, -1]
    dm_rec_temp = sol_temp.y[3, -1]
    vr_rec_temp = sol_temp.y[4, -1]
    vm_rec_temp = sol_temp.y[5, -1]
    t_rec = sol_temp.t[-1]
    adot_rec_temp = da_dt(t_rec, a_rec_temp)
    phi_constraint = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
        (-3*adot_rec_temp/a_rec_temp*vm_rec_temp + dm_rec_temp)*OmegaM/a_rec_temp +
        (-4*adot_rec_temp/a_rec_temp*vr_rec_temp + dr_rec_temp)*OmegaR/a_rec_temp**2
    )
    return phi_rec_temp - phi_constraint

result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5], method='brentq', xtol=1e-10)
phi0_optimal = result.root
sol_perfect = solve_for_phi0(phi0_optimal)

dr_rec_pf = sol_perfect.y[2, -1]
dm_rec_pf = sol_perfect.y[3, -1]
vr_rec_pf = sol_perfect.y[4, -1]
vm_rec_pf = sol_perfect.y[5, -1]
phi_rec_pf = sol_perfect.y[1, -1]

print(f"\nPerfect fluid values at recombination:")
print(f"  phi = {phi_rec_pf:.6e}")
print(f"  dr = {dr_rec_pf:.6e}")
print(f"  dm = {dm_rec_pf:.6e}")
print(f"  vr = {vr_rec_pf:.6e}")
print(f"  vm = {vm_rec_pf:.6e}")

# Solve for xinf
pf_rec = np.array([dr_rec_pf, dm_rec_pf, vr_rec_pf, vm_rec_pf])
xinf = np.linalg.solve(M_reduced, pf_rec)
print(f"\nSolved xinf:")
print(f"  δr_inf = {xinf[0]:.6e}")
print(f"  δm_inf = {xinf[1]:.6e}")
print(f"  vr_inf = {xinf[2]:.6e}")
print(f"  vm_dot_inf = {xinf[3]:.6e}")

# Compute v_rec
v_rec = M_full @ xinf
print(f"\nBoltzmann values at recombination (v_rec = M_full @ xinf):")
print(f"  phi = {v_rec[0]:.6e}")
print(f"  psi = {v_rec[1]:.6e}")
print(f"  dr = {v_rec[2]:.6e}")
print(f"  dm = {v_rec[3]:.6e}")
print(f"  vr = {v_rec[4]:.6e}")
print(f"  vm = {v_rec[5]:.6e}")
print(f"  fr2 = {v_rec[6]:.6e}")

# Define forward ODE
def dX3_dt_local(t, X):
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

# Integrate forward
Y_rec = np.zeros(num_variables + 1)
Y_rec[0] = a_rec
Y_rec[1:] = v_rec

atol_forward = 1e-10
rtol_forward = 1e-10

print(f"\n" + "-"*80)
print("Forward integration from recombination to swaptime")
print("-"*80)

sol_rec_to_swap = solve_ivp(dX3_dt_local, [recConformalTime, swaptime], Y_rec,
                             dense_output=True, method='BDF',
                             atol=atol_forward, rtol=rtol_forward, max_step=1e-2)

Y_swap_integrated = sol_rec_to_swap.y[:, -1]
v_swap_integrated = Y_swap_integrated[1:]  # Remove scale factor

print(f"Integration succeeded with {len(sol_rec_to_swap.t)} points")
print(f"\nIntegrated values at swaptime:")
print(f"  a = {Y_swap_integrated[0]:.6e}")
print(f"  phi = {v_swap_integrated[0]:.6e}")
print(f"  psi = {v_swap_integrated[1]:.6e}")
print(f"  dr = {v_swap_integrated[2]:.6e}")
print(f"  dm = {v_swap_integrated[3]:.6e}")
print(f"  vr = {v_swap_integrated[4]:.6e}")
print(f"  vm = {v_swap_integrated[5]:.6e}")

# Matrix prediction
v_swap_predicted = JKLMNOPQR @ v_rec
print(f"\nMatrix predicted values at swaptime (JKLMNOPQR @ v_rec):")
print(f"  phi = {v_swap_predicted[0]:.6e}")
print(f"  psi = {v_swap_predicted[1]:.6e}")
print(f"  dr = {v_swap_predicted[2]:.6e}")
print(f"  dm = {v_swap_predicted[3]:.6e}")
print(f"  vr = {v_swap_predicted[4]:.6e}")
print(f"  vm = {v_swap_predicted[5]:.6e}")

print(f"\n" + "-"*80)
print("COMPARISON: Matrix prediction vs Integration")
print("-"*80)
for i, label in enumerate(['phi', 'psi', 'dr', 'dm', 'vr', 'vm']):
    pred = v_swap_predicted[i]
    integ = v_swap_integrated[i]
    diff = abs(pred - integ)
    rel_diff = diff / (abs(pred) + 1e-20)
    match_str = "✓" if rel_diff < 1e-3 else "✗"
    print(f"  {label}: pred={pred:.6e}, integ={integ:.6e}, diff={diff:.6e}, rel={rel_diff:.6e} {match_str}")

print("\n" + "="*80)
