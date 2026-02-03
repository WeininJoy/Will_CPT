# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:08:50 2022

@author: MRose
"""

"""
TESTING BACK_INT X_RECS
"""

from scipy.integrate import solve_ivp
import numpy as np
import config as c
from config import H0, OmegaM, OmegaR, OmegaLambda
import derivative_functions as d
import matplotlib.pyplot as plt

#functions_dict = {d.deaglan_s:"OgPerf", d.full_matrix:"FullMatrix", d.red_matrix:"RedMatrix"};
#function = d.function;
#variables_dict = {d.deaglan_s:6, d.full_matrix:6, d.red_matrix:4}

#integrate background equation to find s_rec and fcb_time
sol = solve_ivp(c.ds_dt, [c.t0,12], [c.s0], method='LSODA', atol=c.atol, rtol=c.rtol);
#find FCB by finding smallest absolute value in list and printing that time
s_abs = abs(sol.y[0]);
fcb_time = sol.t[s_abs.argmin()];
#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - c.s_rec); #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()];

#`````````````````````````````````````````````````````````````````````````````
# FINDING XRECS
#``````````````````````````````````````````````````````````````````````````````

kvalues = np.linspace(0.1, 5, 100);
x_recs_normal = [];
x_recs_full_matrix = [];
x_recs_full_matrix_a = [];
x_recs_red_matrix = [];

for i in range(len(kvalues)):
    
    k = kvalues[i];
    d.k = k;
    print(k)
    X0 = c.initial_conds(k);
    X0[0] = np.log(X0[0]); #to go from s0 to sigma0
    X0.pop(2);
  
    #set up derivative function with sigma=log(s) instead of s for stability at small t
    #X1 has form [sigma,phi,dr,dm,vr,vm]
    def dX1_dt(t,X):
        sigma,phi,dr,dm,vr,vm = X;
        sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                                +OmegaR*np.exp(2*sigma)));
        
        #calculate densities of matter and radiation
        rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma));
        rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma));
        
        phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma));
        drdot = (4/3)*(3*phidot + (k**2)*vr);
        dmdot = 3*phidot + vm*(k**2);
        vrdot = -(phi + dr/4);
        vmdot = sigmadot*vm - phi;
        return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]
    
    #solve perfect fluid equations up to recombination
    solperf = solve_ivp(dX1_dt, [c.t0,recConformalTime], X0, method='LSODA', atol=c.atol, rtol=c.rtol);
    solperf2 = solve_ivp(d.full_matrix, [c.t0, recConformalTime], c.initial_conds(k), method='LSODA', atol=c.atol, rtol=c.rtol);
    solperf3 = solve_ivp(d.full_matrix_a, [c.t0, recConformalTime], c.initial_conds_a(k), method='LSODA', atol=c.atol, rtol=c.rtol);
    
    #try with reduced matrix
    red_X0 = c.initial_conds(k);
    del red_X0[1:3];
    print("integrating red matrix")
    solperf4 = solve_ivp(d.red_matrix, [c.t0, recConformalTime], red_X0, method='LSODA', atol=c.atol, rtol=c.rtol);
    print("done integrating")
    #if variables_dict[function] == 6:
    x_rec_normal = [solperf.y[1,-1], solperf.y[1,-1], solperf.y[2,-1], solperf.y[3,-1],
             solperf.y[4,-1], solperf.y[5,-1]];
    x_rec_full_matrix = [solperf2.y[1,-1], solperf2.y[2,-1], solperf2.y[3,-1], solperf2.y[4,-1],
            solperf2.y[5,-1], solperf2.y[6,-1]];
    x_rec_full_matrix_a = [solperf3.y[1,-1], solperf3.y[2,-1], solperf3.y[3,-1], solperf3.y[4,-1],
           solperf3.y[5,-1], solperf3.y[6,-1]];
    x_rec_red_matrix = [solperf4.y[1, -1], solperf4.y[2,-1], solperf4.y[3,-1], solperf4.y[4,-1]]; #dr,dm,vr,vm
    """
    elif variables_dict[function] == 4:
        x_rec = [solperf.y[2,-1], solperf.y[3,-1],
                 solperf.y[4,-1], solperf.y[5,-1]];
    """
    x_recs_normal.append(x_rec_normal);
    x_recs_full_matrix.append(x_rec_full_matrix);
    x_recs_full_matrix_a.append(x_rec_full_matrix_a);
    x_recs_red_matrix.append(x_rec_red_matrix);
   
plt.figure(figsize=(10,7))
plt.plot(kvalues, [xrec[0] for xrec in x_recs_normal], '--b', label='usual')
plt.plot(kvalues, [xrec[0] for xrec in x_recs_full_matrix], '--r', label='full matrix')
plt.plot(kvalues, [xrec[0] for xrec in x_recs_full_matrix_a], '--g', label='full matrix a')
plt.legend()
plt.title("phi")

plt.figure(figsize=(10,7))
plt.plot(kvalues, [xrec[2] for xrec in x_recs_normal], '--b', label='usual')
plt.plot(kvalues, [xrec[2] for xrec in x_recs_full_matrix], '--r', label='full matrix')
plt.plot(kvalues, [xrec[2] for xrec in x_recs_full_matrix_a], '--g', label='full matrix a')
plt.plot(kvalues, [xrec[0] for xrec in x_recs_red_matrix], '--c', label='red_matrix')
plt.legend()
plt.title("dr")

plt.figure(figsize=(10,7))
plt.plot(kvalues, [xrec[3] for xrec in x_recs_normal], '--b', label='usual')
plt.plot(kvalues, [xrec[3] for xrec in x_recs_full_matrix], '--r', label='full matrix')
plt.plot(kvalues, [xrec[3] for xrec in x_recs_full_matrix_a], '--g', label='full matrix a')
plt.plot(kvalues, [xrec[1] for xrec in x_recs_red_matrix], '--c', label='red_matrix')
plt.legend()
plt.title("dm")

plt.figure(figsize=(10,7))
plt.plot(kvalues, [xrec[4] for xrec in x_recs_normal], '--b', label='usual')
plt.plot(kvalues, [xrec[4] for xrec in x_recs_full_matrix], '--r', label='full matrix')
plt.plot(kvalues, [xrec[4] for xrec in x_recs_full_matrix_a], '--g', label='full matrix a')
plt.plot(kvalues, [xrec[2] for xrec in x_recs_red_matrix], '--c', label='red_matrix')
plt.legend()
plt.title("vr")

plt.figure(figsize=(10,7))
plt.plot(kvalues, [xrec[5] for xrec in x_recs_normal], '--b', label='usual')
plt.plot(kvalues, [xrec[5] for xrec in x_recs_full_matrix], '--r', label='full matrix')
plt.plot(kvalues, [xrec[5] for xrec in x_recs_full_matrix_a], '--g', label='full matrix a')
plt.plot(kvalues, [xrec[3] for xrec in x_recs_red_matrix], '--c', label='red_matrix')
plt.legend()
plt.title("vm")
plt.show()