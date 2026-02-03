# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 09:02:48 2022

@author: MRose
"""

"""
CONFIG FILE - CONTAINS ALL HARD-CODED VARIABLES, INITIAL CONDITIONS,
AND DERIVATIVE FUNCTIONS
"""

import numpy as np

"""
HARD-CODED VALUES
"""
#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

#set cosmological parameters from Planck baseline
OmegaLambda = 0.679;
OmegaM = 0.321;
OmegaR = 9.24e-5;
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1

#set tolerances
atol = 1e-9;
rtol = 1e-9;
stol = 1e-10;
swaptime = 2; #set time when we swap from s to sigma
endtime = 6.140;
deltaeta = 6.150659839680297 - endtime;
Hinf = H0*np.sqrt(OmegaLambda);

t0 = 1e-4;

#set coefficients for initial conditions in s
smin1 = np.sqrt(3*OmegaLambda/OmegaR);
szero = - OmegaM/(4*OmegaR);
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3));
s2 = -(OmegaM**3)/(192*OmegaLambda*OmegaR**2);
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3))/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
s4 = -(OmegaM**5)/(9216*(OmegaR**3)*(OmegaLambda**2))
s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

#set coefficients for initial conditions in a
a1 = np.sqrt(OmegaR/(3*OmegaLambda));
a2 = OmegaM/(12*OmegaLambda);
a0 = a1*t0 + a2*t0**2;

#find conformal time at recombination
z_rec = 1090.30;
s_rec = 1+z_rec; #reciprocal scale factor at recombination

"""
FUNCTIONS
"""

#derivative function for background in s
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

#derivative function for background in a
def da_dt(t, a):
    return 1*H0*np.sqrt(OmegaLambda*abs(a**4) + OmegaM*abs(a) + OmegaR)

#calculate initial conditions as a function of k
def initial_conds(k):
    sigma0 = np.log(s0)
    
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phi0 = 1 + phi1*t0 + phi2*t0**2; #t0 from above in "background equations section"
    
    #dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr1 = 4*phi1;
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda);
    dr3 = np.sqrt(OmegaR*OmegaLambda)*(-63*OmegaM**3 \
                                       + 404*k**2*OmegaM*OmegaR*OmegaLambda)/(4320*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    dr4 = (315*OmegaM**4 - 1924*k**2*OmegaM**2*OmegaR*OmegaLambda \
           - 11520*OmegaR**3*OmegaLambda + 3312*k**4*OmegaR**2*OmegaLambda**2)/(181440*OmegaR**2*OmegaLambda**2);
    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4;
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda);
    dm3 = (3/4)*dr3;
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3;
    
    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda);
    vr4 = np.sqrt(OmegaR*OmegaLambda)*(63*OmegaM**3 \
                                       - 344*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    vr5 = (-63*OmegaM**4 + 347*k**2*OmegaM**2*OmegaR*OmegaLambda \
           + 2304*OmegaR**3*OmegaLambda - 360*k**4*OmegaR**2*OmegaLambda**2)/(362880*OmegaR**2*OmegaLambda**2);
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5;
    
    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda);
    vm4 = -(-63*OmegaM**3 + 128*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*(OmegaR*OmegaLambda)**1.5);
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4;
    
    #X0 = [sigma0, phi0, dr0, dm0, vr0, vm0];
    X0 = [s0, phi0, phi0, dr0, dm0, vr0, vm0];
    return X0

#calculate initial conditions as a function of k
def initial_conds_a(k):
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phi0 = 1 + phi1*t0 + phi2*t0**2; #t0 from above in "background equations section"
    
    #dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr1 = 4*phi1;
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda);
    dr3 = np.sqrt(OmegaR*OmegaLambda)*(-63*OmegaM**3 \
                                       + 404*k**2*OmegaM*OmegaR*OmegaLambda)/(4320*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    dr4 = (315*OmegaM**4 - 1924*k**2*OmegaM**2*OmegaR*OmegaLambda \
           - 11520*OmegaR**3*OmegaLambda + 3312*k**4*OmegaR**2*OmegaLambda**2)/(181440*OmegaR**2*OmegaLambda**2);
    dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4;
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda);
    dm3 = (3/4)*dr3;
    dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3;
    
    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda);
    vr4 = np.sqrt(OmegaR*OmegaLambda)*(63*OmegaM**3 \
                                       - 344*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    vr5 = (-63*OmegaM**4 + 347*k**2*OmegaM**2*OmegaR*OmegaLambda \
           + 2304*OmegaR**3*OmegaLambda - 360*k**4*OmegaR**2*OmegaLambda**2)/(362880*OmegaR**2*OmegaLambda**2);
    vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5;
    
    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda);
    vm4 = -(-63*OmegaM**3 + 128*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*(OmegaR*OmegaLambda)**1.5);
    vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4;
    
    X0 = [a0, phi0, phi0, dr0, dm0, vr0, vm0];
    #X0 = [s0, phi0, phi0, dr0, dm0, vr0, vm0];
    return X0

#calculate initial conditions as a function of k
def initial_grads(k):
    sdot0 = -smin1/(t0**2) + s1 + 2*s2*t0 + 3*s3*t0**2 + 4*s4*t0**3;
    sigmadot0 = sdot0/s0;
    
    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phidot0 = phi1 + 2*phi2*t0; #t0 from above in "background equations section"
    
    #dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr1 = 4*phi1;
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda);
    dr3 = np.sqrt(OmegaR*OmegaLambda)*(-63*OmegaM**3 \
                                       + 404*k**2*OmegaM*OmegaR*OmegaLambda)/(4320*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    dr4 = (315*OmegaM**4 - 1924*k**2*OmegaM**2*OmegaR*OmegaLambda \
           - 11520*OmegaR**3*OmegaLambda + 3312*k**4*OmegaR**2*OmegaLambda**2)/(181440*OmegaR**2*OmegaLambda**2);
    drdot0 = dr1 + 2*dr2*t0 + 3*dr3*t0**2 + 4*dr4*t0**3;
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda);
    dm3 = (3/4)*dr3;
    dmdot0 = dm1 + 2*dm2*t0 + 3*dm3*t0**2;
    
    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda);
    vr4 = np.sqrt(OmegaR*OmegaLambda)*(63*OmegaM**3 \
                                       - 344*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    vr5 = (-63*OmegaM**4 + 347*k**2*OmegaM**2*OmegaR*OmegaLambda \
           + 2304*OmegaR**3*OmegaLambda - 360*k**4*OmegaR**2*OmegaLambda**2)/(362880*OmegaR**2*OmegaLambda**2);
    vrdot0 = vr1 + 2*vr2*t0 + 3*vr3*t0**2 + 4*vr4*t0**3 + 5*vr5*t0**4;
    
    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda);
    vm4 = -(-63*OmegaM**3 + 128*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*(OmegaR*OmegaLambda)**1.5);
    vmdot0 = vm1 + 2*vm2*t0 + 3*vm3*t0**2 + 4*vm4*t0**3;
    
    #Xdot0 = [sigmadot0, phidot0, drdot0, dmdot0, vrdot0, vmdot0];
    Xdot0 = [sdot0, phidot0, phidot0, drdot0, dmdot0, vrdot0, vmdot0];
    return Xdot0

#calculate initial conditions as a function of k
def initial_grads_a(k):
    adot0 = a1 + 2*a2*t0;

    phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
    phi2 = (1/60)*(-2*k**2 + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
    phidot0 = phi1 + 2*phi2*t0; #t0 from above in "background equations section"
    
    #dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
    dr1 = 4*phi1;
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(240*OmegaR*OmegaLambda);
    dr3 = np.sqrt(OmegaR*OmegaLambda)*(-63*OmegaM**3 \
                                       + 404*k**2*OmegaM*OmegaR*OmegaLambda)/(4320*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    dr4 = (315*OmegaM**4 - 1924*k**2*OmegaM**2*OmegaR*OmegaLambda \
           - 11520*OmegaR**3*OmegaLambda + 3312*k**4*OmegaR**2*OmegaLambda**2)/(181440*OmegaR**2*OmegaLambda**2);
    drdot0 = dr1 + 2*dr2*t0 + 3*dr3*t0**2 + 4*dr4*t0**3;
    
    dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2)/(320*OmegaR*OmegaLambda);
    dm3 = (3/4)*dr3;
    dmdot0 = dm1 + 2*dm2*t0 + 3*dm3*t0**2;
    
    vr1 = -1/2;
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda);
    vr4 = np.sqrt(OmegaR*OmegaLambda)*(63*OmegaM**3 \
                                       - 344*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*OmegaR**2*OmegaLambda**2);
    vr5 = (-63*OmegaM**4 + 347*k**2*OmegaM**2*OmegaR*OmegaLambda \
           + 2304*OmegaR**3*OmegaLambda - 360*k**4*OmegaR**2*OmegaLambda**2)/(362880*OmegaR**2*OmegaLambda**2);
    vrdot0 = vr1 + 2*vr2*t0 + 3*vr3*t0**2 + 4*vr4*t0**3 + 5*vr5*t0**4;
    
    vm1 = -1/2;
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda);
    vm4 = -(-63*OmegaM**3 + 128*k**2*OmegaM*OmegaR*OmegaLambda)/(34560*np.sqrt(3)*(OmegaR*OmegaLambda)**1.5);
    vmdot0 = vm1 + 2*vm2*t0 + 3*vm3*t0**2 + 4*vm4*t0**3;
    
    Xdot0 = [adot0, phidot0, phidot0, drdot0, dmdot0, vrdot0, vmdot0];
    #X0 = [s0, phi0, phi0, dr0, dm0, vr0, vm0];
    return Xdot0
    
#terminate at fcb (ie when s is at its minimum positive value)

"""
def at_fcb(t,X):
    if X[0]<stol:
        X[0] = 0
    return X[0]
"""

def at_fcb(t,X):
    if X[0]<stol:
        X[0] = 0
        #if t != 6.150659839680297:
        #    print(t)
    return X[0]
        
at_fcb.terminal = True