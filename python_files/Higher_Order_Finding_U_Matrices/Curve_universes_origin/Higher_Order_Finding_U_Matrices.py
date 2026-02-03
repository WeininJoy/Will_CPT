# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import numpy as np

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

OmegaLambda = 0.679 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.321 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 0
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 75 # number of pert variables, 75 for original code
swaptime = 2 #set time when we swap from s to sigma
deltaeta = 6.6e-7 # integrating from endtime-deltaeta to recombination time, instead of from FCB -> prevent numerical issues
Hinf = H0*np.sqrt(OmegaLambda)

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

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

fcb_time = sol.t_events[0][0]
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
    sigma, phi, dr, dm, vr, vm = X  # sigma is log(s)
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK+OmegaM*np.exp(sigma)
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
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2)))+ OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    #calculate densities of matter and radiation
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
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    
    phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
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


#print(dX2_dt(recConformalTime, inits))
#terminate at fcb (ie when s is at its minimum positive value)
def at_fcb(t,X):
    if X[0]<stol:
        X[0] = 0
        #print(t)
    return X[0]

at_fcb.terminal = True

#-------------------------------------------------------------------------------
# For each K, find ACmatrices, BDvectors and Xmatrices
#-------------------------------------------------------------------------------

kvalues = np.linspace(1e-4,15,num=100) # originally: kvalues = np.linspace(1e-5,15,num=300)
ABCmatrices = []
DEFmatrices = []
GHIvectors = []
X1matrices = []
X2matrices = []

#first run background integration to determine s_init
sol2 = solve_ivp(ds_dt, [t0,endtime], [s0], method='LSODA', events=at_fcb, 
                atol=atol, rtol=rtol)
s_init = sol2.y[0,-1]

for i in range(len(kvalues)):
    
    #set k value and print
    k = kvalues[i]
    print(k)
    
    #---------------------------------------------------------------------------------------
    # For each K, find ABCmatrix 
    #---------------------------------------------------------------------------------------
    
    #create U matrix (nxn square matrix)
    ABC_matrix = np.zeros(shape=(num_variables, 6))
    
    #now run through each initial condition for base variables to obtain ABC
    for n in range(6):
        print(n)
        #define initial conditions
        x0 = np.zeros(num_variables)
        x0[n] = 1
        inits = np.concatenate(([s_init], x0))
        
        #first integrate from endtime to swaptime in s
        solperf = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)
        #print(solperf.y[:,-1])
        #now take end values as initial conditions for next integration
        #remember to change s to sigma!
        
        inits2 = solperf.y[:,-1]
        inits2[0] = np.log(inits2[0])
        
        #now integrate from swaptime to recombination
        solperf2 = solve_ivp(dX3_dt, [swaptime,recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)
        
        #obtain nth column of U matrix
        nth_col = []
        for m in range(1,num_variables+1):
            nth_col.append(solperf2.y[m,-1])
            
        #change nth column of U matrix to the above column vector
        ABC_matrix[:,n] = nth_col
        
    ABCmatrices.append(ABC_matrix)
    
    #---------------------------------------------------------------------------
    # FOR each K, find DEF matrix
    #---------------------------------------------------------------------------
    
    DEF_matrix = np.zeros(shape=(num_variables, 2))
    
    #run integration for F2 and F3's
    for j in range(0,2): 

        x0 = np.zeros(num_variables)
        inits = np.concatenate(([s_init], x0))
        inits[j+7] = 1
    
        #now perform integration and store this as BD vector (remember to remove s!)
        sol3 = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)
    
        #now take end values as initial conditions for next integration
        #remember to change s to sigma!
        
        inits2 = sol3.y[:,-1]
        inits2[0] = np.log(inits2[0])
    
        sol4 = solve_ivp(dX3_dt, [swaptime,recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)
    
        nthcol = sol4.y[:,-1]
        nthcol = np.array(nthcol)
        nthcol = np.delete(nthcol, 0)
        
        DEF_matrix[:,j] = nthcol
    
    DEFmatrices.append(DEF_matrix)
    
    #----------------------------------------------------------------------------
    # Now find GHIx3 vectors by setting v_r^\infty to 1
    #----------------------------------------------------------------------------
    
    x3 = -(16/945)*(k**4)*(deltaeta**3)
    x0 = np.zeros(num_variables)
    inits = np.concatenate(([s_init], x0))
    inits[9] = x3
    
    sol5 = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)
    
    inits2 = sol5.y[:,-1]
    inits2[0] = np.log(inits2[0])
    
    sol6 = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)
    
    #now add resulting vector apart from s to GHIvectors
    vec = np.array(sol6.y[:,-1])
    vec = np.delete(vec, 0)
    GHIvectors.append(vec)
    
    #-----------------------------------------------------------------------------------------------
    # Define X1 matrix for calculating x_endtimes from fcb values
    #-----------------------------------------------------------------------------------------------
    
    #define coefficients that pop up often
    coeff1 = (Hinf**3)*(OmegaM/OmegaLambda)
    coeff2 = (Hinf**4)*(OmegaR/OmegaLambda)
    de = deltaeta
    de2 = deltaeta**2
    de3 = deltaeta**3
    
    #define rows and cols
    x00 = 0
    x01 = -(3/(2*k**2))*coeff1*de
    x02 = (6/(k**2))*coeff2*de + (1/5)*coeff2*de3
    x03 = 3*x01 - (3/4)*coeff1*de3
    
    x10 = x00
    x11 = x01
    x12 = x02
    x13 = x03
    
    x20 = 1 - (1/6)*(k**2)*de2
    x21 = -(6/(k**2))*coeff1*de + (2/3)*coeff1*de3
    x22 = -(4/(3*(k**2)))*(k**4 - 18*coeff2)*de - ((28/15)*coeff2 - (2/15)*(k**4))*de3
    x23 = -(18/(k**2))*coeff1*de - coeff1*de3
    
    x30 = 0
    x31 = 1 - (9/(2*k**2))*coeff1*de + 0.5*coeff1*de3
    x32 = (18/(k**2))*coeff2*de - (7/5)*coeff2*de3
    x33 = -(27/(2*k**2))*coeff1*de + 0.5*(k**2)*de2 - (3/4)*coeff1*de3
    
    x40 = +0.25*de - (1/40)*(k**2)*de3
    x41 = -(3/(2*k**2))*coeff1*de2
    x42 = 1 + (1/k**2)*(6*coeff2 - 0.3*(k**4))*de2
    x43 = -(9/(2*k**2))*coeff1*de2
    
    x50 = 0
    x51 = -(3/(2*k**2))*coeff1*de2
    x52 = (6/(k**2))*coeff2*de2
    x53 = -de - (9/(2*k**2))*coeff1*de2
    
    X1 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13], [x20, x21, x22, x23], 
                  [x30, x31, x32, x33], [x40, x41, x42, x43], [x50, x51, x52, x53]])
    
    X1matrices.append(X1)
    
    #-----------------------------------------------------------------------------------------------
    # Define X2 matrix for calculating x_endtimes from fcb values
    #-----------------------------------------------------------------------------------------------
    
    de4 = de**4
    
    x00 = (1/15)*(k**2)*de2 - (1/210)*(k**4)*de4
    x01 = -(4/15)*coeff1*de3
    x02 = (8/15)*(k**2)*de - (16/15)*((1/14)*(k**4) - coeff2)*de3
    x03 = -(4/5)*coeff1*de3
    
    x10 = -(1/105)*(k**3)*de3
    x11 = (k/35)*coeff1*de4
    x12 = -(4/35)*(k**3)*de2 + (4/35)*((5/54)*(k**5) - coeff2)*de4
    x13 = (3*k/35)*coeff1*de4
    
    X2 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13]])
    
    X2matrices.append(X2)
    
np.save('L70_kvalues', kvalues)
np.save('L70_ABCmatrices', ABCmatrices)
np.save('L70_DEFmatrices', DEFmatrices)
np.save('L70_GHIvectors', GHIvectors)
np.save('L70_X1matrices', X1matrices)
np.save('L70_X2matrices', X2matrices)
