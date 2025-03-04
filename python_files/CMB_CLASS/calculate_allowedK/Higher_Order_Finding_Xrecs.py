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

folder_path = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS/calculate_allowedK/'

def findind_Xrecs(mt, kt, h):

    # Parameters
    lam = 1
    rt = 1
    Omega_gamma_h2 = 2.47e-5 # photon density 
    Neff = 3
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

    # calculate present scale factor a0 and energy densities
    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    def transform(Omega_r, rt, mt, kt):
        a0 = solve_a0(Omega_r, rt, mt, kt)
        s0 = 1/a0
        Omega_lambda = Omega_r * a0**4
        Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
        Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
        return s0, Omega_lambda, Omega_r, Omega_m, Omega_K

    s0, OmegaLambda, OmegaR, OmegaM, OmegaK = transform(Omega_r, rt, mt, kt)
    # print('s0, OmegaLambda, OmegaR, OmegaM, OmegaK=', s0, OmegaLambda, OmegaR, OmegaM, OmegaK)    

    s0 = 1

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

    t0 = 1e-8 * s0

    #set coefficients for initial conditions
    smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
    szero = - OmegaM/s0**3/(4*OmegaR/s0**4)
    s1 = (OmegaM)**2/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
    s2 = - (OmegaM**3)/(192*s0*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*s0*OmegaLambda*OmegaR) 

    s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2

    def reach_FCB(t, s): return s[0]
    reach_FCB.terminal = True

    sol = solve_ivp(ds_dt, [t0,12*s0], [s_bang], max_step = 0.25e-4*s0, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)

    fcb_time = sol.t_events[0]

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
        sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*X[0])+OmegaK/s0**2+OmegaM/s0**3*np.exp(X[0])
                                +OmegaR/s0**4*np.exp(2*X[0])))
        
        #calculate densities of matter and radiation
        rho_m = 3*(H0**2)*OmegaM/s0**3*(np.exp(3*X[0]))
        rho_r = 3*(H0**2)*OmegaR/s0**4*(np.exp(4*X[0]))
        
        phidot = sigmadot*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])/(2*np.exp(2*X[0]))
        drdot = (4/3)*(3*phidot + (k**2)*X[4])
        dmdot = 3*phidot + X[5]*(k**2)
        vrdot = -(X[1] + X[2]/4)
        vmdot = sigmadot*X[5] - X[1]
        return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

    #------------------------------------------------------------------------------------
    # Now find and store rec values
    #------------------------------------------------------------------------------------

    kvalues = np.load(folder_path + f'L70_kvalues.npy')
    recValuesList = []

    for i in range(len(kvalues)):
        
        k = kvalues[i]
        
        #------------------------------------------------------------------------------
        # Set up actual recombination values
        #------------------------------------------------------------------------------
        
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
        s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2
        sigma0 = np.log(s_bang)
        phi0 = 1 + phi1*t0 + phi2*t0**2 #t0 from above in "background equations section"
        dr0 = -2 + dr1*t0 + dr2*t0**2
        dm0 = -1.5 + dm1*t0 + dm2*t0**2
        vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3
        vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3
        
        X0 = [sigma0, phi0, dr0, dm0, vr0, vm0]
        
        #solve perfect fluid equations up to recombination
        sol3 = solve_ivp(dX1_dt, [t0,recConformalTime], X0, method='LSODA', atol=atol, rtol=rtol)
        
        #set initial conditions are recombination time (final entries of above integration)
        X00 = [np.exp(sol3.y[0,-1]), sol3.y[1,-1], sol3.y[1,-1], sol3.y[2,-1], sol3.y[3,-1], 
            sol3.y[4,-1], sol3.y[5,-1]]
        
        recValues = X00[1:num_variables+1]
        recValuesList.append(recValues)
        
    np.save(folder_path + f'L70_recValues', recValuesList)
    print("Finding Xrecs Done.")
