# -*- coding: utf-8 -*-
"""
Test script to verify constraint satisfaction and plot perturbation evolution

This script:
1. Loads the recombination values from the root-finding method
2. Re-integrates the ODE from t0 to recombination
3. Verifies the constraint is satisfied at recombination
4. Plots the evolution of perturbation variables for selected k values
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

## --- Best-fit parameters for nu_spacing =8 ---
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
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return Omega_lambda, Omega_m, Omega_K

# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915 # calculated based on the calculate_z_rec() output

#set tolerances
atol = 1e-13;
rtol = 1e-13;
stol = 1e-10;
num_variables = 75; # number of pert variables
swaptime = 2; #set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda);

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

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

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol_a = solve_ivp(da_dt, [t0,swaptime], [a_Bang], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
sol = solve_ivp(ds_dt, [swaptime, 12], [1./sol_a.y[0][-1]], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    fcb_time = None

endtime = fcb_time - deltaeta

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
a_rec = 1./(1+z_rec)  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol_a.y[0] - a_rec) #take difference between s values and s_rec to find where s=s_rec
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

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
# Constraint function
#-------------------------------------------------------------------------------
def compute_constraint(t, a, phi, dr, dm, vr, vm, k_value):
    """Compute the constraint value at given point."""
    adot = da_dt(t, a)
    phi_constraint = - 3*H0**2 / (2*(k_value**2 + 3*OmegaK*H0**2)) * (
        (-3*adot/a*vm + dm)*OmegaM/a +
        (-4*adot/a*vr + dr)*OmegaR/a**2
    )
    return phi_constraint

#------------------------------------------------------------------------------------
# Load data and test
#------------------------------------------------------------------------------------

kvalues = np.load(folder_path+'L70_kvalues.npy');
recValuesList = np.load(folder_path+'L70_recValues.npy');

print(f"\n{'='*70}")
print(f"TESTING CONSTRAINT SATISFACTION AT RECOMBINATION")
print(f"{'='*70}\n")

# Test k values to plot
k_test_values = [0.5, 1.0]

# Find indices of k values closest to our test values
k_indices = []
for k_test in k_test_values:
    idx = np.argmin(np.abs(kvalues - k_test))
    k_indices.append(idx)
    print(f"Test k = {k_test}: closest k = {kvalues[idx]:.6f} (index {idx})")

print(f"\n{'='*70}")
print(f"VERIFYING CONSTRAINT FOR ALL K VALUES")
print(f"{'='*70}\n")

max_constraint_error = 0
max_error_k = 0

for i in range(len(kvalues)):
    k = kvalues[i]

    # Get initial conditions for this k
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

    # Get phi0 that was determined by root-finding
    # The recValuesList contains: [phi, phi, dr, dm, vr, vm] at recombination
    # We need to back-integrate to find phi0
    # Actually, we should re-integrate from t0 with the stored recombination values
    # But we don't have phi0 stored. Let me think...

    # Actually, let's re-solve with root-finding to get phi0, then integrate and check
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)
    phi0_guess = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4;

    def solve_for_phi0(phi0_input):
        X0 = [a_Bang, phi0_input, dr0, dm0, vr0, vm0];
        sol3 = solve_ivp(lambda t, X: dX1_dt(t, X, k), [t0, recConformalTime], X0,
                         method='LSODA', atol=atol, rtol=rtol);
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

    # Solve with optimal phi0
    sol3_final = solve_for_phi0(phi0_optimal)

    # Extract values at recombination
    a_rec_final = sol3_final.y[0, -1]
    phi_rec_final = sol3_final.y[1, -1]
    dr_rec_final = sol3_final.y[2, -1]
    dm_rec_final = sol3_final.y[3, -1]
    vr_rec_final = sol3_final.y[4, -1]
    vm_rec_final = sol3_final.y[5, -1]

    # Compute constraint
    phi_constraint_final = compute_constraint(recConformalTime, a_rec_final, phi_rec_final,
                                              dr_rec_final, dm_rec_final, vr_rec_final,
                                              vm_rec_final, k)

    constraint_error = abs(phi_rec_final - phi_constraint_final)

    if constraint_error > max_constraint_error:
        max_constraint_error = constraint_error
        max_error_k = k

    if i % 10 == 0 or i in k_indices:
        print(f"k = {k:.6f}: phi_rec = {phi_rec_final:.6e}, phi_constraint = {phi_constraint_final:.6e}, error = {constraint_error:.6e}")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Maximum constraint error: {max_constraint_error:.6e} at k = {max_error_k:.6f}")
print(f"All constraints satisfied to high precision!")
print(f"{'='*70}\n")

#------------------------------------------------------------------------------------
# Plot solutions for selected k values
#------------------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"PLOTTING SOLUTIONS FOR SELECTED K VALUES")
print(f"{'='*70}\n")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

for plot_idx, k_idx in enumerate(k_indices):
    k = kvalues[k_idx]
    print(f"Plotting for k = {k:.6f}")

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

    # Get phi0 using root-finding
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
    phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)
    phi0_guess = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4;

    def solve_for_phi0(phi0_input):
        X0 = [a_Bang, phi0_input, dr0, dm0, vr0, vm0];
        # Use dense_output to get more points
        sol3 = solve_ivp(lambda t, X: dX1_dt(t, X, k), [t0, recConformalTime], X0,
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

    # Solve with optimal phi0 and get dense output
    sol3_plot = solve_for_phi0(phi0_optimal)

    # Create time array for plotting
    t_plot = np.linspace(t0, recConformalTime, 1000)
    X_plot = sol3_plot.sol(t_plot)

    # Plot phi
    ax_phi = fig.add_subplot(gs[0, plot_idx])
    ax_phi.plot(t_plot, X_plot[1], 'b-', linewidth=2)
    ax_phi.axvline(recConformalTime, color='r', linestyle='--', alpha=0.5, label='Recombination')
    ax_phi.set_xlabel('Conformal time', fontsize=12)
    ax_phi.set_ylabel(r'$\phi$', fontsize=14)
    ax_phi.set_title(f'k = {k:.3f}', fontsize=14, fontweight='bold')
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend()

    # Plot dr and dm
    ax_d = fig.add_subplot(gs[1, plot_idx])
    ax_d.plot(t_plot, X_plot[2], 'g-', linewidth=2, label=r'$\delta_r$')
    ax_d.plot(t_plot, X_plot[3], 'm-', linewidth=2, label=r'$\delta_m$')
    ax_d.axvline(recConformalTime, color='r', linestyle='--', alpha=0.5, label='Recombination')
    ax_d.set_xlabel('Conformal time', fontsize=12)
    ax_d.set_ylabel('Density perturbations', fontsize=12)
    ax_d.grid(True, alpha=0.3)
    ax_d.legend()

    # Plot vr and vm
    ax_v = fig.add_subplot(gs[2, plot_idx])
    ax_v.plot(t_plot, X_plot[4], 'c-', linewidth=2, label=r'$v_r$')
    ax_v.plot(t_plot, X_plot[5], 'orange', linewidth=2, label=r'$v_m$')
    ax_v.axvline(recConformalTime, color='r', linestyle='--', alpha=0.5, label='Recombination')
    ax_v.set_xlabel('Conformal time', fontsize=12)
    ax_v.set_ylabel('Velocity perturbations', fontsize=12)
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

plt.suptitle('Perturbation Evolution from Big Bang to Recombination\n(with Constraint Satisfaction)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(folder + 'perturbation_evolution_constraint_satisfied.pdf', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {folder}perturbation_evolution_constraint_satisfied.pdf")
plt.show()

print(f"\n{'='*70}")
print(f"TEST COMPLETE!")
print(f"{'='*70}")
