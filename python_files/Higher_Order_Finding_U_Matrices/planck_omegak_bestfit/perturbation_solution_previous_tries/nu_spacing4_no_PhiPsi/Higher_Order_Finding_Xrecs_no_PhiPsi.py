# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:21:56 2021

@author: MRose
"""


from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root_scalar
import numpy as np
from math import *

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
nu_spacing = 4
folder = f'./data/nu_spacing{nu_spacing}_no_PhiPsi/'
# folder_path = folder + 'data_allowedK/'
# folder_path = folder + 'data_integerK/'
folder_path = folder + 'data_all_k/'

# #set cosmological parameters from Planck baseline
# OmegaLambda = 0.679;
# OmegaM = 0.321;
# OmegaR = 9.24e-5;
# H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1

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

###############################################################################
# Best-fit parameters from nu_spacing=4 (first try)
# mt, kt, Omegab_ratio, h = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915 # calculated based on the calculate_z_rec() output
###############################################################################

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

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8;

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR);
szero = - OmegaM/(4*OmegaR);
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Check if t_events[0] is not empty before trying to access its elements
if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    # You might want to assign a default value or 'None' to fcb_time here
    fcb_time = None # Or np.nan, or some other indicator

# Rest of your code that uses fcb_time would go here
# For example:
if fcb_time is not None:
    print(f"Further processing with fcb_time = {fcb_time}")
else:
    print(f"No fcb_time available for further processing.")

endtime = fcb_time - deltaeta

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
s_rec = 1+z_rec  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - s_rec) #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()]

#-------------------------------------------------------------------------------
# Define derivative functions
#-------------------------------------------------------------------------------

def dX1_dt(t,X):
    sigma , dr, dm, vr, vm = X[0], X[1], X[2], X[3], X[4];
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)));

    phi = -3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (3*sigmadot*vm + dm)*np.exp(sigma)*OmegaM + (4*sigmadot*vr + dr)*np.exp(2*sigma)*OmegaR);
    phidot = sigmadot*phi - H0**2 * ( (3/2)*OmegaM*np.exp(sigma)*vm + 2*OmegaR*np.exp(2*sigma)*vr );
    drdot = (4/3)*(3*phidot + (k**2)*vr);    
    dmdot = 3*phidot + vm*(k**2);
    vrdot = -(phi + dr/4);
    vmdot = sigmadot*vm - phi;
    return [sigmadot, drdot, dmdot, vrdot, vmdot]

#------------------------------------------------------------------------------------
# Now find and store rec values
#------------------------------------------------------------------------------------

kvalues = np.load(folder_path+'L70_kvalues.npy');
recValuesList = [];

for i in range(len(kvalues)):
    
    k = kvalues[i];
    print(k)
    
    #------------------------------------------------------------------------------
    # Set up actual recombination values
    #------------------------------------------------------------------------------
    
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    
    dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda);
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda);
    
    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda);
    
    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda);
    
    #set initial conditions
    t0 = 1e-8;
    s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3;
    sigma0 = np.log(s0)
    phi0 = 1 + phi1*t0 + phi2*t0**2; #t0 from above in "background equations section"
    sigmadot0 = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma0)+OmegaK+OmegaM*np.exp(sigma0)
                            +OmegaR*np.exp(2*sigma0)));
    dm0 = -1.5 + dm1*t0 + dm2*t0**2;
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3;
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3;
    dr0 = -2 + dr1*t0 + dr2*t0**2;
    phi0_derived = -3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * ( (3*sigmadot0*vm0 + dm0)*np.exp(sigma0)*OmegaM + (4*sigmadot0*vr0 + dr0)*np.exp(2*sigma0)*OmegaR);
    print('Difference in phi initial conditions:', abs(phi0 - phi0_derived))
    
    X0 = [sigma0, dr0, dm0, vr0, vm0];
    
    #solve perfect fluid equations up to recombination
    sol3 = solve_ivp(dX1_dt, [t0,recConformalTime], X0, method='Radau', atol=atol, rtol=rtol);
    
    #set initial conditions are recombination time (final entries of above integration)
    X00 = [np.exp(sol3.y[0,-1]), sol3.y[1,-1], sol3.y[1,-1], sol3.y[2,-1], sol3.y[3,-1], sol3.y[4,-1]];
    recValues = X00[1:num_variables+1];
    recValuesList.append(recValues);

np.save(folder_path+'L70_recValues', recValuesList)