# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
import numpy as np
from scipy import optimize

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

#set cosmological parameters from Planck baseline
OmegaLambda = 0.68 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.321
OmegaR = 9.24e-5
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1

#set tolerances
atol = 1e-13
rtol = 1e-13
tol = 1e-8

def find_curvature(x):

    OmegaK = x[0]
    #```````````````````````````````````````````````````````````````````````````````
    #BACKGROUND EQUATIONS
    #```````````````````````````````````````````````````````````````````````````````

    #write derivative function for background
    def ds_dt(t, s):
        return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs((s**2)) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    t0 = 1e-8

    #set coefficients for initial conditions
    smin1 = np.sqrt(3*OmegaLambda/OmegaR)
    szero = - OmegaM/(4*OmegaR)
    s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
    s2 = - (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) 

    s0 = smin1/t0 + szero + s1*t0 + s2*t0**2

    print('Performing Initial Background Integration')
    sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
    print('Initial Background Integration Done')

    idxfcb = np.where(np.diff(np.sign(sol.y[0])) != 0)[0]
    fcb_time = 0.5*(sol.t[idxfcb[0]] + sol.t[idxfcb[0] + 1])

    #####################
    # Convert to actual units
    #####################
    from astropy import units as u
    from astropy.constants import c

    H0_actual = 70 * u.km/u.s/u.Mpc # 70
    Lambda = OmegaLambda * (3*H0_actual**2) / c**2
    devide_sqrtLambda = np.sqrt(1/Lambda).si.to(u.Mpc)
    Delta_k = np.sqrt(3) * np.pi / fcb_time / devide_sqrtLambda
    print("Delta_k=", Delta_k)

    K_bar = - np.sign(OmegaK) 
    a0_bar = c * np.sqrt(-K_bar/OmegaK)/H0_actual
    a0_bar = a0_bar.to(u.Mpc)
    integer_Delta_k = 1 / a0_bar
    print("integer_Delta_k=", integer_Delta_k)

    return Delta_k - 10 * integer_Delta_k

sol = optimize.root(find_curvature, [-0.001], method='hybr')
print("OmegaK=", sol.x)