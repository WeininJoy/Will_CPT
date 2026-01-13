# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from multiprocessing import Pool, cpu_count
from functools import partial
import classy
import os

nu_spacing = 4

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
folder = f'./data/'
# folder_path = folder + 'data_allowedK/'
folder_path = folder + 'data_integerK/'
# folder_path = folder + 'data_all_k/'
# folder_path = folder + 'data_small_k/'

# #set cosmological parameters from Planck baseline
# OmegaLambda = 0.679;
# OmegaM = 0.321;
# OmegaR = 9.24e-5;
# H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1

lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density 
Neff = 3.046
N_ncdm = 1  # number of massive neutrino species
m_ncdm = 0.06  # mass of massive neutrino species in e

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

##################
# calculate recombination redshift
def calculate_z_rec(mt, kt, omega_b_ratio, h, As, ns, tau):
    Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt, kt, h)
    
    CMB_params = {
        'output': 'tCl',  # Request only thermodynamics outputs
        'h': h,
        'Omega_b': omega_b_ratio*Omega_m,
        'Omega_cdm': (1.-omega_b_ratio) *Omega_m,
        'Omega_k': float(Omega_K),
        'A_s': As*1e-9, 
        'n_s': ns,
        'N_ncdm': N_ncdm,      # number of massive neutrino
        'm_ncdm': m_ncdm,      # mass of massive neutrino species in eV
        'N_ur': Neff - N_ncdm, # Effective number of MASSLESS neutrino species, N_eff = N_ncdm + N_ur
        'tau_reio': tau,
        'lensing': 'no' #Turn off lensing.
    }

    # Initialize and compute CLASS
    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()

    # Get thermodynamics data
    thermo = cosmo.get_thermodynamics()
    z = thermo['z']
    free_electron_fraction = thermo['x_e']

    # find z_recombination when xe = 0.1
    z_rec = z[np.argmin(np.abs(free_electron_fraction - 0.1))]
    cosmo.struct_cleanup()
    cosmo.empty()

    return z_rec

###############################################################################
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 409.969398,1.459351,0.163514,0.547313,2.095762,0.972835,0.053017
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = calculate_z_rec(mt, kt, Omegab_ratio, h, As, ns, tau)  # the actual value still needs to be checked
print(f"Recombination redshift z_rec = {z_rec}") # z_rec = 1063.4075 based on the calculate_z_rec() output
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
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5;

#set coefficients for initial conditions
# smin1 = np.sqrt(3*OmegaLambda/OmegaR);
# szero = - OmegaM/(4*OmegaR);
# s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
# s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
# s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
# s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

# s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

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
a_rec = 1./(1+z_rec)  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol_a.y[0] - a_rec) #take difference between s values and s_rec to find where s=s_rec
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

#print(dX2_dt(recConformalTime, inits))
#terminate at fcb (ie when s is at its minimum positive value)
def at_fcb(t,X):
    if X[0]<stol:
        X[0] = 0
        #print(t)
    return X[0]

at_fcb.terminal = True

#-------------------------------------------------------------------------------
# Worker function for parallel processing
#-------------------------------------------------------------------------------
def process_single_k(k, s_init, endtime, swaptime, recConformalTime, deltaeta,
                     Hinf, OmegaM, OmegaLambda, OmegaR, OmegaK,
                     num_variables, atol, rtol, H0):
    """
    Process a single k value to compute ABC, DEF matrices and GHI vector.
    This function encapsulates the per-k computation for parallel processing.

    Parameters
    ----------
    k : float
        The wavenumber to process
    s_init : float
        Initial value of s
    endtime : float
        End time for integration
    swaptime : float
        Time when we swap from s to sigma
    recConformalTime : float
        Conformal time at recombination
    deltaeta : float
        Time step parameter
    Hinf, OmegaM, OmegaLambda, OmegaR, OmegaK : float
        Cosmological parameters
    num_variables : int
        Number of perturbation variables
    atol, rtol : float
        Integration tolerances
    H0 : float
        Hubble constant

    Returns
    -------
    tuple
        (ABC_matrix, DEF_matrix, GHI_vector, X1, X2)
    """

    print(f"Processing k = {k}")

    # Define derivative functions with closure over k and cosmological parameters
    def dX2_dt_local(t, X):
        s, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
        sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(s**3) + OmegaR*abs(s**4)))

        rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
        rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)

        phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
        fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
        psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
        drdot = (4/3)*(3*phidot + (k**2)*vr)
        dmdot = 3*phidot + vm*(k**2)
        vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot = (sdot/s)*vm - psi
        derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

        for j in range(8, num_variables):
            l = j - 5
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

        lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
        derivatives.append(lastderiv)
        return derivatives

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

    #---------------------------------------------------------------------------------------
    # For each K, find ABCmatrix
    #---------------------------------------------------------------------------------------

    ABC_matrix = np.zeros(shape=(num_variables, 6))

    for n in range(6):
        x0 = np.zeros(num_variables)
        x0[n] = 1
        inits = np.concatenate(([s_init], x0))

        solperf = solve_ivp(dX2_dt_local, [endtime, swaptime], inits, method='LSODA', atol=atol, rtol=rtol)

        inits2 = solperf.y[:, -1]
        inits2[0] = 1./inits2[0] # switch s to a

        solperf2 = solve_ivp(dX3_dt_local, [swaptime, recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)

        nth_col = []
        for m in range(1, num_variables+1):
            nth_col.append(solperf2.y[m, -1])

        ABC_matrix[:, n] = nth_col

    #---------------------------------------------------------------------------
    # FOR each K, find DEF matrix
    #---------------------------------------------------------------------------

    DEF_matrix = np.zeros(shape=(num_variables, 2))

    for j in range(0, 2):
        x0 = np.zeros(num_variables)
        inits = np.concatenate(([s_init], x0))
        inits[j+7] = 1

        sol3 = solve_ivp(dX2_dt_local, [endtime, swaptime], inits, method='LSODA', atol=atol, rtol=rtol)

        inits2 = sol3.y[:, -1]
        inits2[0] = 1./inits2[0]

        sol4 = solve_ivp(dX3_dt_local, [swaptime, recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)

        nthcol = sol4.y[:, -1]
        nthcol = np.array(nthcol)
        nthcol = np.delete(nthcol, 0)

        DEF_matrix[:, j] = nthcol

    # #----------------------------------------------------------------------------
    # # Now find GHIx3 vectors by setting v_r^\infty to 1
    # #----------------------------------------------------------------------------

    x3 = -(16/945)*(k**4)*(deltaeta**3)
    x0 = np.zeros(num_variables)
    inits = np.concatenate(([s_init], x0))
    inits[9] = x3

    sol5 = solve_ivp(dX2_dt_local, [endtime, swaptime], inits, method='LSODA', atol=atol, rtol=rtol)

    inits2 = sol5.y[:, -1]
    inits2[0] = 1./inits2[0]

    sol6 = solve_ivp(dX3_dt_local, [swaptime, recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)

    vec = np.array(sol6.y[:, -1])
    vec = np.delete(vec, 0)
    GHI_vector = vec

    # # ----------------------------------------------------------------------------
    # # Define X1, X2 matrices (order deltaeta^6)
    # # ----------------------------------------------------------------------------

    #define coefficients that pop up often
    coeff1 = (Hinf**3)*(OmegaM/OmegaLambda)
    coeff2 = (Hinf**4)*(OmegaR/OmegaLambda)
    coeff3 = (Hinf**2)*(OmegaK/OmegaLambda)
    denom = k**2 + 3*coeff3

    de = deltaeta
    de2 = deltaeta**2
    de3 = deltaeta**3
    de4 = deltaeta**4
    de5 = deltaeta**5
    de6 = deltaeta**6
    
    #define rows and cols for X1
    #-----------------------------------------------------------------------------------------------
    # ROW 0: Auxiliary (Phi-like structure)
    #-----------------------------------------------------------------------------------------------
    phi4_dr = (8*k**2*coeff2 + 24*coeff2*coeff3)/(80*denom)
    
    phi4_dm = -75*coeff1**2 / (80*denom)
    phi5_dm = -35*coeff1*(56*coeff2 + coeff3**2) / (2800*denom)
    
    phi4_vr = 300*coeff1*coeff2 / (80*denom)
    phi5_vr = -(260*k**2*coeff2 + 644*coeff2*coeff3 - 140*coeff2*(coeff3**2 + 56*coeff2)) / (2800*denom)

    x00 = phi4_dr*de4
    
    x01 = -(3*coeff1)/(2*denom)*de - (5*coeff1*coeff3)/(20*denom)*de3 + phi4_dm*de4 + phi5_dm*de5
    
    x02 = (6*coeff2)/denom*de + (4*k**2*coeff2 + 32*coeff2*coeff3)/(20*denom)*de3 + phi4_vr*de4 + phi5_vr*de5
    
    x03 = -(9*coeff1)/(2*denom)*de - (15*k**2*coeff1 + 60*coeff1*coeff3)/(20*denom)*de3 + 3*phi4_dm*de4 + (-(525*coeff1*coeff3)/(2800*denom) + 3*phi5_dm)*de5
    
    #-----------------------------------------------------------------------------------------------
    # ROW 1: Psi (Identical to Row 0 except for shear term in x12)
    #-----------------------------------------------------------------------------------------------
    x10 = x00
    x11 = x01
    x12 = x02 + (24*k**2*coeff2 + 32*coeff2*coeff3)/(20*denom)*de3
    x13 = x03

    #-----------------------------------------------------------------------------------------------
    # ROW 2: delta_r
    #-----------------------------------------------------------------------------------------------
    dr5_dm = (210*k**2*coeff1*coeff3 - 630*k**4*coeff1 - 1575*coeff1*(56*coeff2 + coeff3**2)) / (31500*denom)
    dr6_dm = -(k**2*coeff1**2)/(12*denom)

    dr5_vr = -(150*k**8 + 858*k**6*coeff3 - 10080*coeff2*(-8*coeff3**2 + 35*coeff2) + 48*k**2*coeff3*(14*coeff3**2 + 1195*coeff2) + 4*k**4*(362*coeff3**2 + 1665*coeff2)) / (31500*denom)
    dr6_vr = -(4*k**2*coeff1*coeff2)/(12*denom)

    dr5_vm = (-13545*k**2*coeff1*coeff3 + 1260*k**4*coeff1 - 1575*coeff1*(168*coeff2 + 48*coeff3**2)) / (31500*denom)

    x20 = 1 - (1/6)*k**2*de2 + ((1/120)*k**4 + (k**2*coeff3)/90 + 0.4*coeff2)*de4 - (75*k**6 + 204*k**4*coeff3 + 112*k**2*coeff3**2)/378000*de6
    
    x21 = -(6*coeff1)/denom*de + (30*k**2*coeff1 - 45*coeff1*coeff3)/(45*denom)*de3 + dr5_dm*de5 + dr6_dm*de6
    
    x22 = -(4*k**4 + 12*k**2*coeff3 - 72*coeff2)/(3*denom)*de + (6*k**6 + 26*k**4*coeff3 + 288*coeff2*coeff3 + 12*k**2*(2*coeff3**2 - 7*coeff2))/(45*denom)*de3 + dr5_vr*de5 + dr6_vr*de6
    
    x23 = -(18*coeff1)/denom*de - (k**2*coeff1 + 12*coeff1*coeff3)/denom*de3 + dr5_vm*de5 + 3*dr6_dm*de6

    #-----------------------------------------------------------------------------------------------
    # ROW 3: delta_m
    #-----------------------------------------------------------------------------------------------
    dm5_dm = -105*coeff1*(-4*k**2*coeff3 + 3*(coeff3**2 + 56*coeff2)) / (8400*denom)
    dm6_dm = -(k**2*coeff1**2)/(16*denom)

    x30 = (72*coeff2/240)*de4 + (3*k**2*coeff2/720)*de6
    
    x31 = 1 - (9*coeff1)/(2*denom)*de + (10*k**2*coeff1 - 15*coeff1*coeff3)/(20*denom)*de3 + dm5_dm*de5 + dm6_dm*de6
    
    x32 = (18*coeff2)/denom*de - (28*k**2*coeff2 - 96*coeff2*coeff3)/(20*denom)*de3 + (-(1556*k**2*coeff2 + 5796*coeff2*coeff3) + 420*coeff2*(-4*k**2*coeff3 + 3*(coeff3**2 + 56*coeff2))/denom)/8400*de5
    
    x33 = -(27*coeff1)/(2*denom)*de + 0.5*k**2*de2 - (15*k**2*coeff1 + 180*coeff1*coeff3)/(20*denom)*de3 + (k**2*coeff3/24)*de4 + ((630*k**2*coeff1 - 4725*coeff1*coeff3)/8400 + 3*dm5_dm)*de5 + (k**2*(coeff3**2 + 12*coeff2)/720 + 3*dm6_dm)*de6

    #-----------------------------------------------------------------------------------------------
    # ROW 4: v_r
    #-----------------------------------------------------------------------------------------------
    vr5_dm = 3*coeff1**2 / (8*denom)
    vr6_dm = -(7875*k**4*coeff1 + 10395*k**2*coeff1*coeff3 + 2205*coeff1*(200*coeff2 + 7*coeff3**2)) / (4410000*denom)

    vr6_vr_num = 1935*k**8 + 13605*k**6*coeff3 + 36*k**4*(902*coeff3**2 - 2485*coeff2) + 112*k**2*coeff3*(271*coeff3**2 + 2880*coeff2) + 4704*(2*coeff3**4 + 330*coeff2*coeff3**2 - 375*coeff2**2)
    
    vr6_vm = (9450*k**4*coeff1 + 163485*k**2*coeff1*coeff3 + 2205*coeff1*(600*coeff2 + 336*coeff3**2)) / (4410000*denom)

    x40 = 0.25*de - (3*k**2 + 4*coeff3)/120*de3 + (75*k**4 + 204*k**2*coeff3 + 112*coeff3**2)/84000*de5
    
    x41 = -(15*coeff1)/(10*denom)*de2 + (315*k**2*coeff1 - 105*coeff1*coeff3)/(4200*denom)*de4 + vr5_dm*de5 + vr6_dm*de6
    
    x42 = 1 - (3*k**4 + 13*k**2*coeff3 + 12*(coeff3**2 - 5*coeff2))/(10*denom)*de2 + (75*k**6 + 429*k**4*coeff3 + 4*k**2*(181*coeff3**2 - 630*coeff2) + 336*coeff3*(coeff3**2 - 10*coeff2))/(4200*denom)*de4 + (12*coeff1*coeff2)/(8*denom)*de5 - vr6_vr_num/(4410000*denom)*de6
    
    x43 = -(45*coeff1)/(10*denom)*de2 - (630*k**2*coeff1 + 5040*coeff1*coeff3)/(4200*denom)*de4 + 3*vr5_dm*de5 + vr6_vm*de6

    #-----------------------------------------------------------------------------------------------
    # ROW 5: v_m
    #-----------------------------------------------------------------------------------------------
    vm5_dm = 45*coeff1**2 / (120*denom)
    
    x50 = -(3*coeff2/120)*de5
    
    x51 = -(3*coeff1)/(2*denom)*de2 - (30*coeff1*coeff3)/(120*denom)*de4 + vm5_dm*de5
    
    x52 = (6*coeff2)/denom*de2 - (56*k**2*coeff2 + 48*coeff2*coeff3)/(120*denom)*de4 + (180*coeff1*coeff2)/(120*denom)*de5
    
    x53 = -de - (9*coeff1)/(2*denom)*de2 - (coeff3/6)*de3 - (45*k**2*coeff1 + 225*coeff1*coeff3)/(120*denom)*de4 + (-(coeff3**2 + 12*coeff2)/120 + 3*vm5_dm)*de5
    
    X1 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13], [x20, x21, x22, x23], 
                  [x30, x31, x32, x33], [x40, x41, x42, x43], [x50, x51, x52, x53]])
    
    #-----------------------------------------------------------------------------------------------
    # Define X2 matrix (Fr2, Fr3)
    #-----------------------------------------------------------------------------------------------
    
    fr2_5_dm = (3150*k**2*coeff1 - 735*coeff1*coeff3) / (275625*denom)
    fr2_5_vr = (795*k**6 + 4065*k**4*coeff3 + 28*k**2*(208*coeff3**2 - 765*coeff2) + 2352*coeff3*(coeff3**2 - 10*coeff2)) / (275625*denom)
    fr2_5_vm = (-1575*k**2*coeff1 - 735*48*coeff1*coeff3) / (275625*denom)
    fr2_6_dm = k**2*coeff1**2 / (30*denom)

    x00 = (1/15)*k**2*de2 + k**2*(-15*k**2 - 14*coeff3)/3150*de4 + (795*k**6 + 1680*k**4*coeff3 + 784*k**2*coeff3**2)/6615000*de6
    
    x01 = -(4*k**2*coeff1)/(15*denom)*de3 + fr2_5_dm*de5 + fr2_6_dm*de6
    
    x02 = (8/15)*k**2*de - 4*k**2*(30*k**4 + 118*k**2*coeff3 + 84*(coeff3**2 - 5*coeff2))/(1575*denom)*de3 + fr2_5_vr*de5 + (4*k**2*coeff1*coeff2)/(30*denom)*de6
    
    x03 = -(12*k**2*coeff1)/(15*denom)*de3 + fr2_5_vm*de5 + 3*fr2_6_dm*de6
    
    fr3_6_dm = k**3*(3150*k**2*coeff1 - 735*coeff1*coeff3 + 2205*coeff1*(200*coeff2 + 7*coeff3**2)) / (3858750*denom)
    fr3_6_vr_num = 795*k**6 + 4065*k**4*coeff3 + 28*k**2*(208*coeff3**2 - 765*coeff2) + 2352*coeff3*(coeff3**2 - 10*coeff2)
    fr3_6_vm = k**3*(1575*k**2*coeff1 + 735*48*coeff1*coeff3 - 2205*coeff1*(600*coeff2 + 336*coeff3**2)) / (3858750*denom)

    x10 = -(1/105)*k**3*de3 + k**3*(15*k**2 + 14*coeff3)/36750*de5
    
    x11 = (k**3/35)*coeff1/denom*de4 + fr3_6_dm*de6
    
    x12 = -(4/35)*k**3*de2 + k**3*(30*k**4 + 118*k**2*coeff3 + 84*(coeff3**2 - 5*coeff2))/(3675*denom)*de4 - k**3*fr3_6_vr_num/(3858750*denom)*de6
    
    x13 = (3*k**3*coeff1)/(35*denom)*de4 + fr3_6_vm*de6

    X2 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13]])

    return ABC_matrix, DEF_matrix, GHI_vector, X1, X2

#-------------------------------------------------------------------------------
# For each K, find ACmatrices, BDvectors and Xmatrices
#-------------------------------------------------------------------------------

# kvalues = np.linspace(1e-5,15,num=300); # data_all_k
# kvalues = np.linspace(1e-5,3,num=300); # data_small_k

## kvalues based on allowedK
# small_allowedK = np.load(folder + 'data_small_k/allowedK.npy')
# allowedK_path = folder + 'data_all_k/allowedK.npy'
# kvalues = np.load(allowedK_path); # only caluclate for the allowed K values
# kvalues = kvalues[kvalues > small_allowedK[-1] + 0.5* np.diff(kvalues).mean()]  # remove k values that are already in small_allowedK
# kvalues = np.concatenate((small_allowedK[-3:], kvalues))  # add a few small K values for better resolution at low k

## kvalues based on allowedK_integer
small_allowedK_integer = np.load(folder + 'data_small_k/allowedK_integer.npy')
allowedK_integer_path = folder + 'data_all_k/allowedK_integer.npy'
allowedK_integer = np.load(allowedK_integer_path);
allowedK_integer = allowedK_integer[allowedK_integer > small_allowedK_integer[-1] + 0.5* np.diff(allowedK_integer).mean()]  # remove k values that are already in small_allowedK
allowedK_integer = np.concatenate((small_allowedK_integer[-3:], allowedK_integer))  # add a few small K values for better resolution at low k
a0=1; K=-OmegaK * a0**2 * H0**2
kvalues = [(round(allowedK_integer[-1])-nu_spacing*i)*np.sqrt(K) for i in range(len(allowedK_integer))]
kvalues = kvalues[::-1]  # reverse to ascending order

ABCmatrices = [];
DEFmatrices = [];
GHIvectors = [];
X1matrices = [];
X2matrices = [];

#first run background integration to determine s_init
sol2_a = solve_ivp(da_dt, [t0,swaptime], [a_Bang], method='LSODA', atol=atol, rtol=rtol);
sol2 = solve_ivp(ds_dt, [swaptime,endtime], [1./sol2_a.y[0][-1]], method='LSODA', events=at_fcb,
                atol=atol, rtol=rtol);
s_init = sol2.y[0,-1];

#-------------------------------------------------------------------------------
# Parallel Processing Setup
#-------------------------------------------------------------------------------

# Detect number of CPUs (HPC or local)
# n_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
n_processes = 4 # There are only 4 physical cores on my laptop
print(f"\nUsing {n_processes} parallel processes")

# Print HPC info if available
if 'SLURM_JOB_ID' in os.environ:
    print(f"Running on HPC - Job ID: {os.environ.get('SLURM_JOB_ID')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}")

print(f"Processing {len(kvalues)} k values in parallel...")

# Set up the worker function with fixed parameters
worker_func = partial(
    process_single_k,
    s_init=s_init,
    endtime=endtime,
    swaptime=swaptime,
    recConformalTime=recConformalTime,
    deltaeta=deltaeta,
    Hinf=Hinf,
    OmegaM=OmegaM,
    OmegaLambda=OmegaLambda,
    OmegaR=OmegaR,
    OmegaK=OmegaK, 
    num_variables=num_variables,
    atol=atol,
    rtol=rtol,
    H0=H0
)

# Run parallel computation
with Pool(processes=n_processes) as pool:
    results = pool.map(worker_func, kvalues)

print("Parallel computation completed!")

# Unpack results
for ABC_matrix, DEF_matrix, GHI_vector, X1, X2 in results:
    ABCmatrices.append(ABC_matrix)
    DEFmatrices.append(DEF_matrix)
    GHIvectors.append(GHI_vector)
    X1matrices.append(X1)
    X2matrices.append(X2)

print(f"Processed {len(kvalues)} k values successfully!")

np.save(folder_path+'L70_kvalues', kvalues);
np.save(folder_path+'L70_ABCmatrices', ABCmatrices);
np.save(folder_path+'L70_DEFmatrices', DEFmatrices);
np.save(folder_path+'L70_GHIvectors', GHIvectors);
np.save(folder_path+'L70_X1matrices', X1matrices);
np.save(folder_path+'L70_X2matrices', X2matrices);
