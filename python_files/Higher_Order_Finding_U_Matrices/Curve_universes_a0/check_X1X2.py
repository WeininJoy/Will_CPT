# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import numpy as np
from matplotlib import pyplot as plt

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

# set cosmological parameters
OmegaLambda = 0.679 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.321 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 0
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1

lam = rt = 1
a0 = (OmegaLambda/OmegaR)**(1./4.)
s0 = 1/a0
mt = OmegaM / (OmegaLambda**(1./4.) * OmegaR**(3./4.))
kt = - OmegaK / np.sqrt(OmegaLambda* OmegaR) / 3

# # Write the s0 value to the file s0.txt
# f = open("s0.txt", "a")
# f.write(str(s0)+" ")
# f.close()

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 75 # number of pert variables, 75 for original code
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs((s**2/s0**2)) + OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

t0 = 1e-8 

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR) * s0**2
szero = - OmegaM/(4*OmegaR) * s0
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = - (OmegaM**3)/(192*OmegaLambda*OmegaR**2 * s0) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR * s0) 

s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2

print('Performing Initial Background Integration')

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s_bang], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

fcb_time = sol.t_events[0][0]
print('FCB time=', fcb_time)
deltaeta = fcb_time * 1.5e-6 # integrating from endtime-deltaeta to recombination time, instead of from FCB -> prevent numerical issues
endtime = fcb_time - deltaeta
swaptime = fcb_time / 3. #set time when we swap from s to sigma

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
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK/s0**2+OmegaM/s0**3*np.exp(sigma)
                            +OmegaR/s0**4*np.exp(2*sigma)))
    
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM/s0**3 *(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR/s0**4 *(np.exp(4*sigma))
    
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(phi + dr/4)
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

def dX2_dt(t,X):
    #print(t);
    s,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2/s0**2)))+ OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(abs(s/s0)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s/s0)**4)
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR/s0**4 *s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK/s0**2*H0**2/k**2)*fr2/2
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
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK/s0**2+OmegaM/s0**3*np.exp(sigma)
                            +OmegaR/s0**4*np.exp(2*sigma)))
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM/s0**3*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR/s0**4*(np.exp(4*sigma))
    
    phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR/s0**4*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
    drdot = (4/3)*(3*phidot + (k**2)*vr)
    dmdot = 3*phidot + vm*(k**2)
    vrdot = -(psi + dr/4) + (1 + 3*OmegaK/s0**2*H0**2/k**2)*fr2/2
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


# kvalues = np.linspace(1e-4,20/s0,num=300) # originally: kvalues = np.linspace(1e-5,15,num=300)
kvalues = [15/s0]
variable_name = ['phi','psi','dr','dm','vr','vm','fr2','fr3']
color_list = ['b','g','r','c','m','y','k','orange']

# Find recombination values
recValuesList = []

for i in range(len(kvalues)):
    
    k = kvalues[i]
    print(k)
    
    #------------------------------------------------------------------------------
    # Set up actual recombination values
    #------------------------------------------------------------------------------
    
    phi1 = -OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0
    phi2 = 3*OmegaM**2/(320*s0**2*OmegaR*OmegaLambda) - (4*OmegaK+k**2*s0**2*OmegaLambda)/(30*s0**2*OmegaLambda)
    
    dr1 = -OmegaM/ (4* np.sqrt(3*OmegaR*OmegaLambda))/s0
    dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2*s0**2)/(240*OmegaR*OmegaLambda*s0**2) - 8*OmegaK/(15*OmegaLambda*s0**2)
    
    dm1 = - np.sqrt(3) * OmegaM / (16*s0* np.sqrt(OmegaR*OmegaLambda))
    dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2*s0**2)/(320*OmegaR*OmegaLambda*s0**2) - 2*OmegaK/(5*OmegaLambda*s0**2)
    
    vr1 = -1/2
    vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0
    vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2*s0**2)/(160*OmegaR*OmegaLambda*s0**2) + 4*OmegaK/(45*OmegaLambda*s0**2)
    
    vm1 = -1/2
    vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda))/s0
    vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2*s0**2)/(480*OmegaR*OmegaLambda*s0**2) + 17*OmegaK/(360*OmegaLambda*s0**2)
    
    #set initial conditions
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
    X00 = [sol3.y[0,-1], sol3.y[1,-1], sol3.y[1,-1], sol3.y[2,-1], sol3.y[3,-1], 
           sol3.y[4,-1], sol3.y[5,-1]]
    
    recValues = X00[0:num_variables+1]
    
    #now integrate from recombination to swaptime
    inits = np.concatenate(([float(x) for x in recValues], [0 for i in range(num_variables-len(recValues)+1)]))

    solperf = solve_ivp(dX3_dt, [recConformalTime, swaptime], inits, method='LSODA', atol=atol, rtol=rtol)
    #remember to change s to sigma!
    inits2 = solperf.y[:,-1]
    inits2[0] = np.exp(inits2[0])
    solperf2 = solve_ivp(dX2_dt, [swaptime, 12], inits2, events=at_fcb, method='LSODA', atol=atol, rtol=rtol)

    s,phi,psi,dr,dm,vr,vm = solperf2.y[[0,1,2,3,4,5,6],-1]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2/s0**2)))+ OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))
    vmdot = (sdot/s)*vm - psi
    X_infty = [dr,dm,vr,vmdot] 

    for i in range(8):
        if i==7:
            plt.plot(solperf2.t, solperf2.y[i+1], label=variable_name[i], color=color_list[i])

    #-----------------------------------------------------------------------------------------------
    # Define X1 matrix for calculating x_endtimes from fcb values
    #-----------------------------------------------------------------------------------------------
    
    #define coefficients that pop up often
    coeff1 = (Hinf**3)*(OmegaM/OmegaLambda) / s0**3
    coeff2 = (Hinf**4)*(OmegaR/OmegaLambda) / s0**4
    coeff3 = (Hinf**2)*(OmegaK/OmegaLambda) / s0**2
    de = deltaeta
    de2 = deltaeta**2
    de3 = deltaeta**3
    
    #define rows and cols
    x00 = 0
    x01 = - (3 / (2*k**2 + 6*coeff3))*coeff1*de - 5* coeff1*coeff3 / (20* (k**2+3*coeff3)) * de3
    x02 = - (12/(2*k**2 + 6*coeff3))*coeff2*de - ((4*k**2*coeff2 + 32*coeff2*coeff3) / (20*(k**2+3*coeff3)))*de3
    x03 = - 9/ (2*k**2 + 6*coeff3) *coeff1*de - (15*k**2*coeff1 + 60*coeff1*coeff3) / (20*(k**2+3*coeff3)) * de3
    
    x10 = x00
    x11 = x01
    x12 = x02 - 32*(k**2*coeff2+3*coeff2*coeff3) * de**3
    x13 = x03
    
    x20 = 1 - (1/6)*(k**2)*de2
    x21 = - (6*coeff1)/(k**2+3*coeff3)*de - 1/(3*(k**2+3*coeff3))*(coeff1*(3*coeff3-2*k**2))* de3
    x22 = (4*k**4+12*k**2*coeff3-72*coeff2)/(3*k**2+9*coeff3)*de - (6*k**6+26*k**4*coeff3+288*coeff2*coeff3+12*k**2*(2*coeff3**2-7*coeff2)) / (45*(k**2+3*coeff3)) *de3
    x23 = -(18/(k**2+3*coeff3))*coeff1*de - coeff1*(k**2+12*coeff3)/(k**2+3*coeff3)*de3
    
    x30 = 0
    x31 = 1 - (9/(2*k**2+6*coeff3))*coeff1*de + (2*k**2*coeff1-3*coeff1*coeff3)/(4*(k**2+3*coeff3))*de3
    x32 = -18/(k**2+3*coeff3)*coeff2*de - (24*coeff2*coeff3-7*k**2*coeff2)/(5*(k**2+3*coeff3))*de3
    x33 = -(27/(2*k**2+6*coeff3))*coeff1*de + 0.5*(k**2)*de2 - (15*k**2*coeff1+180*coeff1*coeff3)/(20*(k**2+3*coeff3))*de3
    
    x40 = -0.25*de + (3*k**2+4*coeff3)/120*de3
    x41 = 3*coeff1/(2*(k**2+3*coeff3))*de2
    x42 = 1 - (3*k**4+13*k**2*coeff3+12*(coeff3**2-5*coeff2))/(10*(k**2+3*coeff3))*de2
    x43 = (9*coeff1)/(2*(k**2+3*coeff3))*de2
    
    x50 = 0
    x51 = (3/(2*k**2+6*coeff3))*coeff1*de2
    x52 = (6/(k**2+3*coeff3))*coeff2*de2
    x53 = de + (9/(2*k**2+6*coeff3))*coeff1*de2 + coeff3/6*de3
    
    X1 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13], [x20, x21, x22, x23], 
                  [x30, x31, x32, x33], [x40, x41, x42, x43], [x50, x51, x52, x53]])
    
    #-----------------------------------------------------------------------------------------------
    # Define X2 matrix for calculating x_endtimes from fcb values
    #-----------------------------------------------------------------------------------------------
    
    de4 = de**4
    
    x00 = (1/15)*(k**2)*de2 - k**2/3150*(15*k**2+14*coeff3)*de4
    x01 = -(4*k**2*coeff1)/(15*(k**2+3*coeff3))*de3
    x02 = -(8/15)*(k**2)*de + 4*k**2/(1575*(k**2+3*coeff3))*(30*k**4+118*k**2*coeff3+84*(coeff3**2-5*coeff2))*de3
    x03 = -(4*coeff1*k**2)/(5*(k**2+3*coeff3))*de3
    
    x10 = (1/105)*(k**3)*de3
    x11 = -k**3*coeff1/(35*(k**2+3*coeff3))*de4
    x12 = -(4/35)*(k**3)*de2 + k**3*(30*k**4+118*k**2*coeff3+84*(coeff3**2-5*coeff2))/(3675*(k**2+3*coeff3))*de4
    x13 = -(3*k**3/(35*(k**2+3*coeff3)))*coeff1*de4
    
    X2 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13]])
    
    X_prime = X1.reshape(6,4) @ X_infty
    y_prime = X2.reshape(2,4) @ X_infty
    prime_values = np.concatenate((X_prime, y_prime))
    for i in range(8):
        if i ==7:
            plt.plot([endtime], [prime_values[i]], '.', color=color_list[i], markersize=5)

plt.legend(loc='upper right')
plt.xlim(endtime*(1.-1.e-7), fcb_time)
plt.ylim(-0.5, 0.5)
plt.savefig("check_X1X2.pdf")
