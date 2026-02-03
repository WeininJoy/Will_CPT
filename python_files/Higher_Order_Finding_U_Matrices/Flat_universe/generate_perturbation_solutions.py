# -*- coding: utf-8 -*-
"""
This script generates and plots the perturbation solutions for a palindromic universe
model, following the methodology outlined in the paper "Evidence for a Palindromic Universe".

It uses pre-computed data from transfer matrix calculations to solve for the
boundary conditions at the Future Conformal Boundary (FCB) for each allowed wavenumber k.
It then integrates the perturbation equations to obtain the full evolution of the
variables and plots the results.

CORRECTED VERSION: Fixes the IndexError in the Boltzmann hierarchy ODE function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

#set cosmological parameters from Planck baseline
OmegaLambda = 0.679
OmegaM = 0.321
OmegaR = 9.24e-5
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 75 # number of pert variables, 75 for original code
l_max = 69 # Derived from num_variables_boltzmann = 7 + (l_max - 2 + 1)
swaptime = 2 #set time when we swap from s to sigma
endtime = 6.15 # integrating from endtime to recombination time, instead of from FCB -> prevent numerical issues
deltaeta = 6.150659839680297 - endtime 
deltaeta = 6.6e-4
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR)
szero = - OmegaM/(4*OmegaR)
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3))
s2 = -(OmegaM**3)/(192*OmegaLambda*OmegaR**2)  # chamge sign here
# s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3))/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
# s4 = -(OmegaM**5)/(9216*(OmegaR**3)*(OmegaLambda**2))

# s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4
s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 

print('Performing Initial Background Integration')
sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

idxfcb = np.where(np.diff(np.sign(sol.y[0])) != 0)[0]
fcb_time = 0.5*(sol.t[idxfcb[0]] + sol.t[idxfcb[0] + 1])
print('FCB time = ',fcb_time)

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
z_rec = 1090.30
s_rec = 1+z_rec #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - s_rec) #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

#-------------------------------------------------------------------------------
# Define derivative functions
#-------------------------------------------------------------------------------

def dX1_dt(t,X):
    sigma, phi, dr, dm, vr, vm = X  # sigma is log(s)
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))
    
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(phi + dr/4)
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

def dX2_dt(t,X):
    #print(t);
    s,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = (sdot/s)*vm - psi
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    #for l>2 terms, add derivates to above list
    for j in range(8,num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    #now add final term
    """
    lmax = num_variables - 5;
    lastderiv = (k*lmax*X[num_variables-1])/(2*lmax + 1);
    """
    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    
    derivatives.append(lastderiv)
    return derivatives

def dX3_dt(t,X):
    sigma,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    
    phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + fr2/2
    vmdot = (sigmadot)*vm - psi
    derivatives = [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
    #for l>2 terms, add derivates to above list
    for j in range(8,num_variables):
        l = j - 5
        derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
    #now add final term
    """
    lmax = num_variables - 5;
    lastderiv = (k*lmax*X[num_variables-1])/(2*lmax + 1);
    """
    lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
    
    derivatives.append(lastderiv)
    return derivatives


# =============================================================================
# 2. DATA LOADING AND INTERPOLATION
# =============================================================================

print("\n--- Loading and interpolating pre-computed data ---")

try:
    kvalues = np.load('./Origin/L70_kvalues.npy')
    ABCmatrices = np.load('./Origin/L70_ABCmatrices.npy')
    DEFmatrices = np.load('./Origin/L70_DEFmatrices.npy')
    GHIvectors = np.load('./Origin/L70_GHIvectors.npy')
    X1matrices = np.load('./Origin/L70_X1matrices.npy')
    X2matrices = np.load('./Origin/L70_X2matrices.npy')
    recValues = np.load('./Origin/L70_recValues.npy')
    allowedK = np.load('./Origin/allowedK.npy')
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all necessary .npy files are in the 'data' directory.")
    exit()

# Extract sub-matrices
Amatrices = ABCmatrices[:, 0:6, :]
Dmatrices = DEFmatrices[:, 0:6, :]

# Create interpolation functions for each element of the matrices
def create_matrix_interpolator(k_grid, matrix_array):
    rows, cols = matrix_array.shape[1], matrix_array.shape[2]
    interp_funcs = [[interp1d(k_grid, matrix_array[:, i, j], bounds_error=False, fill_value="extrapolate") for j in range(cols)] for i in range(rows)]
    def get_matrix(k):
        return np.array([[func(k) for func in row] for row in interp_funcs])
    return get_matrix

def create_vector_interpolator(k_grid, vector_array):
    cols = vector_array.shape[1]
    interp_funcs = [interp1d(k_grid, vector_array[:, i], bounds_error=False, fill_value="extrapolate") for i in range(cols)]
    def get_vector(k):
        return np.array([func(k) for func in interp_funcs])
    return get_vector
    
get_A = create_matrix_interpolator(kvalues, Amatrices)
get_D = create_matrix_interpolator(kvalues, Dmatrices)
get_X1 = create_matrix_interpolator(kvalues, X1matrices)
get_X2 = create_matrix_interpolator(kvalues, X2matrices)
get_recs = create_vector_interpolator(kvalues, recValues)


# =============================================================================
# 3. CALCULATE AND INTEGRATE SOLUTIONS
# =============================================================================

print("\n--- Calculating solutions for the first 3 allowed modes ---")

solutions = []
num_modes_to_plot = 3

for i in range(num_modes_to_plot):
    k = allowedK[i+2]
    print(f"\nProcessing mode n={i+1} with k={k:.4f}")

    # --- a) Get interpolated matrices for this k ---
    A = get_A(k)
    D = get_D(k)
    X1 = get_X1(k)
    X2 = get_X2(k)
    recs_vec = get_recs(k)

    AX1 = A.reshape(6,6) @ X1.reshape(6,4);
    DX2 = D.reshape(6,2) @ X2.reshape(2,4);
    matrixog = AX1 + DX2 # + GX3;
    matrix = matrixog[[2,3,4,5], :]; #take rows corresponding to phi, dr, dm, vr, vm
    
    xrecs = [recs_vec[2], recs_vec[3], recs_vec[4], recs_vec[5]];


    # # --- b) Solve for X_inf ---
    # M_matrix = (A @ X1 + D @ X2)[1:6, :]
    # x_rec_subset = recs_vec[1:6]
    
    try:
        x_inf = np.linalg.lstsq(matrix, xrecs, rcond=None)[0];
        # x_inf = np.linalg.solve(M_matrix, x_rec_subset)
    except np.linalg.LinAlgError:
        print(f"Could not solve for x_inf for k={k}. Skipping mode.")
        continue

    # --- c) Calculate initial conditions at eta' (endtime) ---
    x_prime = X1 @ x_inf
    y_prime_2_4 = X2 @ x_inf
    s_prime_val = np.interp(endtime, sol.t, sol.y[0])

    Y_prime = np.zeros(num_variables+1) # +1 for s
    Y_prime[0] = s_prime_val
    Y_prime[1:7] = x_prime
    Y_prime[7:9] = y_prime_2_4

    # check Y_prime satisfies the constraint
    phi_prime, psi_prime, dr_prime, dm_prime, vr_prime, vm_prime = x_prime
    dsdt_prime = ds_dt(endtime,Y_prime[0])
    phi_derived = -3*H0**2 / (2*k**2 ) * ( (3*dsdt_prime/s_prime_val*vm_prime + dm_prime)*s_prime_val*OmegaM + (4*dsdt_prime/s_prime_val*vr_prime + dr_prime)*s_prime_val**2*OmegaR);
    print("phi_prime, phi_derived, difference:", phi_prime, phi_derived, abs(phi_prime - phi_derived))
    
    # --- d) Integrate backwards from endtime to recConformalTime ---
    sol_part1 = solve_ivp(dX2_dt, [endtime, swaptime], Y_prime, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol)
    
    Y_swap = sol_part1.y[:, -1]
    Y_swap[0] = np.log(Y_swap[0]) # swap s to sigma
    sol_part2 = solve_ivp(dX3_dt, [swaptime, recConformalTime], Y_swap, 
                          dense_output=True, method='LSODA', atol=atol, rtol=rtol)
    
    # check the backward solution satisfies the constraint at recombination
    sigma_rec, phi_rec, psi_rec, dr_rec, dm_rec, vr_rec, vm_rec = sol_part2.y[:7, -1]
    print("phi, dr, dm, vr, vm at recombination (backward integration):", phi_rec, dr_rec, dm_rec, vr_rec, vm_rec)
    dsdt_rec = ds_dt(recConformalTime,np.exp(sigma_rec))
    phi_derived = -3*H0**2 / (2*k**2) * ( (3*dsdt_rec/np.exp(sigma_rec)*vm_rec + dm_rec)*np.exp(sigma_rec)*OmegaM + (4*dsdt_rec/np.exp(sigma_rec)*vr_rec + dr_rec)*np.exp(2*sigma_rec)*OmegaR);
    print("phi_rec (backward), phi_derived, difference:", phi_rec, phi_derived, abs(phi_rec - phi_derived))
    t_backward = np.concatenate((sol_part1.t, sol_part2.t))
    Y_backward_a = sol_part2.y
    Y_backward_a[0, :] = 1./Y_backward_a[0, :] # swap back a to s where needed
    Y_backward = np.concatenate((sol_part1.y, Y_backward_a), axis=1)

    # --- e) Get solution from Big Bang to Recombination (perfect fluid) ---
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5))
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR))
    
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5))
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda)
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda)
    
    vr1 = -1/2
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda)
    
    vm1 = -1/2
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda)
    
    #set initial conditions
    t0 = 1e-8
    # s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3
    sigma0 = np.log(s0)
    phi0 = 1 + phi1*t0 + phi2*t0**2 #t0 from above in "background equations section"
    dr0 = -2 + dr1*t0 + dr2*t0**2
    dm0 = -1.5 + dm1*t0 + dm2*t0**2
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3
    
    X0 = [sigma0, phi0, dr0, dm0, vr0, vm0]

    sol_perfect = solve_ivp(dX1_dt, [t0, recConformalTime], X0, method='LSODA', atol=atol, rtol=rtol);
    sigma_rec, phi_rec, dr_rec, dm_rec, vr_rec, vm_rec = sol_perfect.y[:6, -1]
    print("phi, dr, dm, vr, vm at recombination (perfect fluid):", phi_rec, dr_rec, dm_rec, vr_rec, vm_rec)
    dsdt_rec = ds_dt(recConformalTime,np.exp(sigma_rec))
    phi_derived = -3*H0**2 / (2*k**2) * ( (3*dsdt_rec/np.exp(sigma_rec)*vm_rec + dm_rec)*np.exp(sigma_rec)*OmegaM + (4*dsdt_rec/np.exp(sigma_rec)*vr_rec + dr_rec)*np.exp(2*sigma_rec)*OmegaR);
    print("phi_rec (perfect), phi_derived, difference:", phi_rec, phi_derived, abs(phi_rec - phi_derived))
    
    # --- f) Stitch and Reflect ---
    t_left = np.concatenate((sol_perfect.t, t_backward[::-1]))

    Y_perfect_full = np.zeros((num_variables+1, len(sol_perfect.t)))
    Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
    Y_perfect_full[1,:] = sol_perfect.y[1,:]
    Y_perfect_full[2,:] = sol_perfect.y[1,:]
    Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]
    
    Y_left = np.concatenate((Y_perfect_full, Y_backward[:, ::-1]), axis=1)

    t_right = 2 * fcb_time - t_left[::-1]
    
    symm = np.ones(num_variables+1)
    symm[[5, 6]] = -1
    for l_idx in range(l_max - 1):
        l = l_idx + 2
        if l % 2 != 0: symm[7 + l_idx] = -1
    S = np.diag(symm)
    
    Y_right = S @ Y_left[:, ::-1]

    t_full = np.concatenate((t_left, t_right))
    Y_full = np.concatenate((Y_left, Y_right), axis=1)
    
    solutions.append({'t': t_full, 'Y': Y_full})

# =============================================================================
# 4. PLOTTING
# =============================================================================

print("\n--- Plotting solutions ---")

fig, axes = plt.subplots(num_modes_to_plot, 1, figsize=(10, 8), sharex=True)
if num_modes_to_plot == 1: axes = [axes]

labels = [r'$v_r$', r'$\delta_r$', r'$v_m$', r'$\delta_m$', r'$\phi$', r'$\psi$']
indices = [5, 3, 6, 4, 1, 2]
colors = ['blue', 'red', 'green', 'orange', 'magenta', 'cyan']

for i, sol in enumerate(solutions):
    ax = axes[i]
    for label, index, color in zip(labels, indices, colors):
        ax.plot(sol['t'], sol['Y'][index, :], label=label, color=color, linewidth=1.2)

    ax.axvline(fcb_time, color='k', linestyle='--', label='FCB')
    ax.axvline(2 * fcb_time, color='k', linestyle='--', label='Big Crunch')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(f'n = {i+1}', fontsize=12)
    ax.set_xlim(0, 2 * fcb_time * 1.05)
    # ax.set_ylim(-3, 3)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3)
axes[-1].set_xlabel(r'$\eta \sqrt{\Lambda}$', fontsize=14)
fig.suptitle('Evolution of Perturbation Solutions for Allowed Modes', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('./Origin/perturbation_solutions.pdf')
plt.show()