# -*- coding: utf-8 -*-
"""
Diagnostic script to investigate forward integration mismatch at swaptime.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Load data
folder_path = './data/data_all_k/'
kvalues = np.load(folder_path + 'L70_kvalues.npy')
JKLmatrices = np.load(folder_path + 'L70_JKLmatrices.npy')
MNOmatrices = np.load(folder_path + 'L70_MNOmatrices.npy')
PQRmatrices = np.load(folder_path + 'L70_PQRmatrices.npy')

# Parameters (same as main script)
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

# Tolerances
atol_forward = 1e-10
rtol_forward = 1e-10

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
print("DIAGNOSTIC: Forward Integration Mismatch Investigation")
print("="*80)
print(f"recConformalTime = {recConformalTime}")
print(f"swaptime = {swaptime}")
print(f"a_rec = {a_rec}")

# Test for a single k value
test_k_idx = len(kvalues)//2
k = kvalues[test_k_idx]
print(f"\nTesting k_idx={test_k_idx}, k={k:.6e}")

JKL = JKLmatrices[test_k_idx]
MNO = MNOmatrices[test_k_idx]
PQR = PQRmatrices[test_k_idx]

print(f"Matrix shapes: JKL={JKL.shape}, MNO={MNO.shape}, PQR={PQR.shape}")

# Define ODE
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

print("\n" + "-"*80)
print("TEST 1: Verify JKL matrix by direct integration")
print("-"*80)

# Test each column of JKL matrix
max_errors = []
for n in range(6):
    print(f"\nColumn {n} (variable {n}):")

    # Set up initial condition with nth variable = 1
    x0 = np.zeros(num_variables)
    x0[n] = 1
    inits_rec = np.concatenate(([a_rec], x0))

    # Direct integration
    sol_direct = solve_ivp(dX3_dt_local, [recConformalTime, swaptime], inits_rec,
                           method='BDF', atol=atol_forward, rtol=rtol_forward, max_step=1e-2)

    # Matrix prediction
    X_matrix = JKL[:, n]

    # Compare (remove scale factor from integration result)
    X_direct = sol_direct.y[1:, -1]

    error = np.abs(X_direct - X_matrix)
    max_error = np.max(error)
    max_error_idx = np.argmax(error)
    rel_error = max_error / (np.max(np.abs(X_direct)) + 1e-20)

    max_errors.append(max_error)

    print(f"  Max absolute error: {max_error:.6e} at index {max_error_idx}")
    print(f"  Max relative error: {rel_error:.6e}")
    print(f"  Direct value at max error: {X_direct[max_error_idx]:.6e}")
    print(f"  Matrix value at max error: {X_matrix[max_error_idx]:.6e}")

    # Check if integration succeeded
    if sol_direct.status != 0:
        print(f"  WARNING: Integration failed with status {sol_direct.status}")
    else:
        print(f"  Integration succeeded with {len(sol_direct.t)} points")

print("\n" + "-"*80)
print("SUMMARY:")
print("-"*80)
print(f"Maximum errors across all JKL columns: {max(max_errors):.6e}")
print(f"Average error: {np.mean(max_errors):.6e}")

if max(max_errors) > 1e-6:
    print("\n⚠ WARNING: Errors exceed 1e-6!")
    print("This suggests:")
    print("1. The JKL matrix was computed with insufficient precision")
    print("2. The forward integration is numerically unstable")
    print("3. There may be a bug in the matrix computation or integration")
else:
    print("\n✓ JKL matrix matches direct integration well")

print("\n" + "-"*80)
print("TEST 2: Check if different tolerances help")
print("-"*80)

# Try tighter tolerances
atol_tight = 1e-13
rtol_tight = 1e-13

n = 2  # Test with dr variable
x0 = np.zeros(num_variables)
x0[n] = 1
inits_rec = np.concatenate(([a_rec], x0))

sol_loose = solve_ivp(dX3_dt_local, [recConformalTime, swaptime], inits_rec,
                      method='BDF', atol=atol_forward, rtol=rtol_forward, max_step=1e-2)
sol_tight = solve_ivp(dX3_dt_local, [recConformalTime, swaptime], inits_rec,
                      method='BDF', atol=atol_tight, rtol=rtol_tight, max_step=1e-4)

X_loose = sol_loose.y[1:, -1]
X_tight = sol_tight.y[1:, -1]
X_matrix = JKL[:, n]

print(f"\nComparison for column {n} with different tolerances:")
print(f"Loose (1e-10): max error vs matrix = {np.max(np.abs(X_loose - X_matrix)):.6e}")
print(f"Tight (1e-13): max error vs matrix = {np.max(np.abs(X_tight - X_matrix)):.6e}")
print(f"Loose vs Tight: max difference = {np.max(np.abs(X_loose - X_tight)):.6e}")

print("\n" + "="*80)
print("Diagnostic complete")
print("="*80)
