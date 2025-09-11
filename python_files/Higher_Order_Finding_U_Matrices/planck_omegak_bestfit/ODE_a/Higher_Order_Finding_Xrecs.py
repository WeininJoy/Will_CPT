# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:21:56 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from math import *
import sys
#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

folder_path = './data/'

# try to load kvalues for the current input_number
try:
    kvalues = np.load(folder_path + f'L70_kvalues.npy')
    print(f'Successfully loaded kvalues.')

except FileNotFoundError:
    print(f"FileNotFoundError: No file found at {folder_path}L70_kvalues.npy")
except Exception as e:
    # Catch any other unexpected errors during file loading or subsequent calculations
    print(f"An unexpected error occurred: {e}")

# -- Metha's parameters (flat universe)
# OmegaLambda = 0.679
# OmegaM = 0.321
# OmegaR = 9.24e-5
# OmegaK = 0.0
# h = 0.701
# z_rec = 1090.30
# s0 = 1

# -- best-fit parameters from Planck 2018 base_omegak_plikHM_TTTEEE_lowl_lowE (see https://wiki.cosmos.esa.int/planck-legacy-archive/images/4/43/Baseline_params_table_2018_68pc_v2.pdf)
# rt = 1
# Omega_gamma_h2 = 2.47e-5 # photon density 
# Neff = 3.046
# h = 0.5409
# OmegaR = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2
# OmegaM, OmegaK = 0.483, -0.0438
# OmegaLambda = 1 - OmegaM - OmegaK - OmegaR # total density parameter
# a0 = (OmegaLambda/OmegaR)**(1/4) 
# s0 = 1/a0
# z_rec = 1089.411 # redshift at recombination

## --- Best-fit parameters for nu_spacing =4 ---
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
    s0 = 1/a0
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return s0, Omega_lambda, Omega_m, Omega_K

# Best-fit parameters from nu_spacing=4
mt, kt, Omegab_ratio, h, A_s, n_s, tau_reio = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583, 1.9375648884116028, 0.9787493821596979, 0.019760560255556746
s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1065.0  # the actual value still needs to be checked
###############################################################################

print('s0, OmegaLambda, OmegaR, OmegaM, OmegaK=', s0, OmegaLambda, OmegaR, OmegaM, OmegaK) 

s0 = 1 # first set s0 to 1, for numerical stability. Will transfer discerete wave vector back to the correct value later.

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10 * s0
num_variables = 6 # number of perf pert variables
H0 = 1/np.sqrt(3*OmegaLambda)
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2/s0**2))) + OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

def da_dt(t, a):
    return a * H0 * np.sqrt(OmegaLambda * a**2 + (OmegaK / s0**2) + (OmegaM / (s0**3 * abs(a))) + (OmegaR / (s0**4 * a**2)))

t0 = 1e-8 * s0

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
szero = - OmegaM/s0**3/(4*OmegaR/s0**4)
s1 = (OmegaM)**2/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = - (OmegaM**3)/(192*s0*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*s0*OmegaLambda*OmegaR) 

s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2
print('s_bang=', s_bang)

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12*s0], [s_bang], max_step = 0.25e-4*s0, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

fcb_time = sol.t_events[0]
print('FCB time=', fcb_time)

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
s_rec = 1+z_rec #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - s_rec) #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

#-------------------------------------------------------------------------------
# Define derivative functions
#-------------------------------------------------------------------------------

def dX1_dt(t, X):
    """
    Evolves the perfect fluid perturbation equations using the scale factor 'a'
    as the background variable for improved numerical stability near the Big Bang.

    State vector X: [a, phi, dr, dm, vr, vm]
    """
    a, phi, dr, dm, vr, vm = X
    
    # Calculate the physical conformal Hubble parameter H_phys = adot/a
    # We use abs(a) in the OmegaM term for robustness if a ever becomes tiny and negative due to numerical error.
    H_phys = H0 * np.sqrt(OmegaLambda * a**2 + (OmegaK / s0**2) + (OmegaM / (s0**3 * abs(a))) + (OmegaR / (s0**4 * a**2)))
    adot = a * H_phys

    # Calculate background densities using 'a'
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (a**(-3))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (a**(-4))

    # Evolution equations
    # Note: H from the original code is -H_phys
    phidot = -H_phys * phi - 0.5 * a**2 * ((4/3) * rho_r * vr + rho_m * vm)
    
    drdot = (4/3) * (3 * phidot + k**2 * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(phi + dr / 4) 
    vmdot = -H_phys * vm - phi
    
    return [adot, phidot, drdot, dmdot, vrdot, vmdot]


#------------------------------------------------------------------------------------
# Now find and store rec values
#------------------------------------------------------------------------------------

recValuesList = []

for i in range(len(kvalues)):
    
    k = kvalues[i]
    print(k)
    
    ## ----------------------------------------------------------------------------------
    # 1. Define Taylor Expansion Coefficients from Mathematica Results
    # ----------------------------------------------------------------------------------
    # These are the coefficients for the power series: X(eta) = X0 + X1*eta + X2*eta^2 + ...

    # Background coefficients for a(eta) = a1*eta + a2*eta^2 + a3*eta^3 + ...
    a1 = np.sqrt(OmegaR / (3 * s0**4 * OmegaLambda))
    a2 = OmegaM / (12 * s0**3 * OmegaLambda)
    a3 = (OmegaR * OmegaK) / (18*np.sqrt(3) * s0**6 * np.sqrt(OmegaR/(s0**4 * OmegaLambda)) * OmegaLambda**2)

    phi1 = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR*s0**2)) - 2*OmegaK/(15*OmegaLambda*s0**2)
    
    dr1 = -OmegaM/ (4* np.sqrt(3*OmegaR*OmegaLambda)) / s0
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2*s0**2)/(240*s0**2*OmegaR*OmegaLambda) - 8*OmegaK/(15*OmegaLambda*s0**2)
    
    dm1 = - np.sqrt(3) * OmegaM / (16 * s0* np.sqrt(OmegaR*OmegaLambda))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2*s0**2)/(320*s0**2*OmegaR*OmegaLambda) - 2*OmegaK/(5*OmegaLambda*s0**2)
    
    vr1 = -1/2
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0)
    vr3 = (-OmegaM**2 + 8*s0**2*OmegaR*OmegaLambda*k**2)/(160*s0**2*OmegaR*OmegaLambda) + 4*OmegaK/(45*OmegaLambda*s0**2)
    
    vm1 = -1/2
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda)*s0)
    vm3 = (-3*OmegaM**2 + 4*s0**2*OmegaR*OmegaLambda*k**2)/(480*s0**2*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda*s0**2)

    #set initial conditions
    t0 = 1e-8
    a_bang = a1*t0 + a2*t0**2 + a3*t0**3
    adot_bang = da_dt(t0, a_bang)
    phi0 = 1 + phi1*t0 + phi2*t0**2 #t0 from above in "background equations section"
    dr0 = -2 + dr1*t0 + dr2*t0**2
    dm0 = -1.5 + dm1*t0 + dm2*t0**2
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3

    phi_derived = -3*H0**2/(2*k**2) * ( OmegaM/a_bang * (dm0-3*adot_bang/a_bang * vm0) + OmegaR/a_bang**2 * ( dr0-4*adot_bang/a_bang * vr0) )
    
    X0 = [a_bang, phi0, dr0, dm0, vr0, vm0]
    
    #solve perfect fluid equations up to recombination
    sol3 = solve_ivp(dX1_dt, [t0,recConformalTime], X0, method='LSODA', atol=atol, rtol=rtol)
    
    #set initial conditions are recombination time (final entries of above integration)
    X00 = [np.exp(sol3.y[0,-1]), sol3.y[1,-1], sol3.y[1,-1], sol3.y[2,-1], sol3.y[3,-1], 
           sol3.y[4,-1], sol3.y[5,-1]]
    
    recValues = X00[1:num_variables+1]
    recValuesList.append(recValues)
    
np.save(folder_path + f'L70_recValues', recValuesList)
