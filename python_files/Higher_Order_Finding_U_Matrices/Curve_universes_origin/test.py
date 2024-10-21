from scipy.integrate import solve_ivp
import numpy as np
from scipy.integrate import quad
from scipy.integrate import odeint

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

#set cosmological parameters from Planck baseline
OmegaLambda = 0.68 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.33 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 1 - OmegaLambda - OmegaM - OmegaR
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
deltaeta = 6.6e-7

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10

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


####################
# Calculate fcb time by a(eta) 
####################
def a_dot(a): 
    return H0*np.sqrt( OmegaR + OmegaM*a + OmegaK*a**2 + OmegaLambda* a**4 )

def eta_tot():
    result = quad(lambda a: 1./a_dot(a), 0, np.inf)
    return result[0]
    
print('eta_tot_a=', eta_tot())