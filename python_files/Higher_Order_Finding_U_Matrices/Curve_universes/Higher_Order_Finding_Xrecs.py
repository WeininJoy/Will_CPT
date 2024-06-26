# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:21:56 2021

@author: MRose
"""


from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from math import *

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

#set cosmological parameters from Planck baseline
OmegaLambda = 0.68 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.33 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 1 - OmegaLambda - OmegaM - OmegaR
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 6 # number of perf pert variables
swaptime = 2 #set time when we swap from s to sigma
deltaeta = 6.6e-7 # integrating from endtime-deltaeta to recombination time, instead of from FCB -> prevent numerical issues
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR)
szero = - OmegaM/(4*OmegaR)
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = - (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) 

s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

fcb_time = sol.t_events[0]
print('FCB time=', fcb_time)
endtime = fcb_time - deltaeta

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
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*X[0])+OmegaK+OmegaM*np.exp(X[0])
                            +OmegaR*np.exp(2*X[0])))
    
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*X[0]))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*X[0]))
    
    phidot = sigmadot*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])/(2*np.exp(2*X[0]))
    drdot = (4/3)*(3*phidot + (k**2)*X[4])
    dmdot = 3*phidot + X[5]*(k**2)
    vrdot = -(X[1] + X[2]/4)
    vmdot = sigmadot*X[5] - X[1]
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

#------------------------------------------------------------------------------------
# Now find and store rec values
#------------------------------------------------------------------------------------

kvalues = np.load('L70_kvalues.npy')
recValuesList = []

for i in range(len(kvalues)):
    
    k = kvalues[i]
    print(k)
    
    #------------------------------------------------------------------------------
    # Set up actual recombination values
    #------------------------------------------------------------------------------
    
    phi1 = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR)) - 2*OmegaK/(15*OmegaLambda)
    
    dr1 = -OmegaM/ (4* np.sqrt(3*OmegaR*OmegaLambda))
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda) - 8*OmegaK/(15*OmegaLambda)
    
    dm1 = - np.sqrt(3) * OmegaM / (16 * np.sqrt(OmegaR*OmegaLambda))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda) - 2*OmegaK/(5*OmegaLambda)
    
    vr1 = -1/2
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4*OmegaK/(45*OmegaLambda)
    
    vm1 = -1/2
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda)
    
    #set initial conditions
    t0 = 1e-8
    s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 
    sigma0 = np.log(s0)
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
    
np.save('L70_recValues', recValuesList)