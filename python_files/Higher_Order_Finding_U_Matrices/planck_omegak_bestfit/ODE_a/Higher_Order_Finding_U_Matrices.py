# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import numpy as np
import sys
#working in units 8piG = Lambda = c = hbar = kB = 1 throughout

folder_path = './data/'

# -- Metha's parameters (flat universe)
# OmegaLambda = 0.679
# OmegaM = 0.321
# OmegaR = 9.24e-5
# OmegaK = 0.0
# h = 0.701
# z_rec = 1090.30
# s0 = 1

# best-fit parameters from Planck 2018 base_omegak_plikHM_TTTEEE_lowl_lowE (see https://wiki.cosmos.esa.int/planck-legacy-archive/images/4/43/Baseline_params_table_2018_68pc_v2.pdf)
# rt = 1
# Omega_gamma_h2 = 2.47e-5 # photon density 
# Neff = 3.046
# h = 0.5409
# OmegaR = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2
# OmegaM, OmegaK = 0.483, -0.0438
# OmegaLambda = 1 - OmegaM - OmegaK - OmegaR # total density parameter
# a0 = (OmegaLambda/OmegaR)**(1/4) 
# s0 = 1/a0
# z_rec = 1089.411 # redshift at recombination


## --- Best-fit parameters for nu_spacing =4 ---
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
    s0 = 1/a0
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return s0, Omega_lambda, Omega_m, Omega_K

# Best-fit parameters from nu_spacing=4
mt, kt, Omegab_ratio, h, A_s, n_s, tau_reio = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583, 1.9375648884116028, 0.9787493821596979, 0.019760560255556746
s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1065.0  # the actual value still needs to be checked
###############################################################################

print('s0, OmegaLambda, OmegaR, OmegaM, OmegaK=', s0, OmegaLambda, OmegaR, OmegaM, OmegaK)    


s0 = 1 # first set s0 to 1, for numerical stability. Will transfer discerete wave vector back to the correct value later.

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10 * s0 # 1e-10
num_variables = 75 # number of pert variables, 75 for original code
H0 = 1/np.sqrt(3*OmegaLambda)
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs((s**2/s0**2)) + OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

t0 = 1e-8 * s0

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/(OmegaR/s0**4))
szero = - OmegaM/s0**3/(4*OmegaR/s0**4)
s1 = OmegaM**2/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = - (OmegaM**3)/(192*s0*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*s0*OmegaLambda*OmegaR) 

s_bang = smin1/t0 + szero + s1*t0 + s2*t0**2


print('Performing Initial Background Integration')

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12* s0], [s_bang], max_step = 0.25e-4 * s0, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
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

deltaeta = 6.e-4 * s0 # integrating from endtime-deltaeta to recombination time, instead of from FCB -> prevent numerical issues
endtime = fcb_time - deltaeta
swaptime = 2* s0 #set time when we swap from s to sigma

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

# ODE system for perfect fluid (a-evolution)
# Assumes psi = phi (no anisotropic stress)

def dX1_dt(t, X):
    """
    Evolves the perfect fluid perturbation equations using the scale factor 'a'
    as the background variable for improved numerical stability near the Big Bang.

    State vector X: [a, phi, dr, dm, vr, vm]
    """
    a, phi, dr, dm, vr, vm = X
    
    # Calculate the physical conformal Hubble parameter H_phys = adot/a
    # We use abs(a) in the OmegaM term for robustness if a ever becomes tiny and negative due to numerical error.
    H_phys = H0 * np.sqrt(OmegaLambda * a**2 + (OmegaK / s0**2) + (OmegaM / (s0**3 * abs(a))) + (OmegaR / (s0**4 * a**2)))
    adot = a * H_phys

    # Calculate background densities using 'a'
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (a**(-3))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (a**(-4))

    # Evolution equations
    # Note: H from the original code is -H_phys
    phidot = -H_phys * phi - 0.5 * a**2 * ((4/3) * rho_r * vr + rho_m * vm)
    
    drdot = (4/3) * (3 * phidot + k**2 * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(phi + dr / 4) 
    vmdot = -H_phys * vm - phi
    
    return [adot, phidot, drdot, dmdot, vrdot, vmdot]


def dX2_dt(t,X):
    #print(t);
    s,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2/s0**2)))+ OmegaM*abs(((s**3/s0**3))) + OmegaR*abs((s**4/s0**4))))

    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(abs(s/s0)**3)
    rho_r = 3*(H0**2)*OmegaR*(abs(s/s0)**4)
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR/s0**4*s)*(sdot*fr2 + 0.5*s*fr2dot)
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

def dX3_dt(t, X):
    """
    Evolves the full Boltzmann hierarchy using the scale factor 'a'
    as the background variable for improved numerical stability.

    State vector X: [a, phi, psi, dr, dm, vr, vm, F_2, F_3, ..., F_lmax]
    """
    # Unpack state vector
    a, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
    
    # Calculate the physical conformal Hubble parameter H_phys = adot/a
    H_phys = H0 * np.sqrt(OmegaLambda * a**2 + (OmegaK / s0**2) + (OmegaM / (s0**3 * abs(a))) + (OmegaR / (s0**4 * a**2)))
    adot = a * H_phys

    # Calculate background densities using 'a'
    rho_m = 3 * (H0**2) * OmegaM / s0**3 * (a**(-3))
    rho_r = 3 * (H0**2) * OmegaR / s0**4 * (a**(-4))
    
    # --- Evolution Equations ---
    # Note: H from the original code is -H_phys
    phidot = -H_phys * psi - 0.5 * a**2 * ((4/3) * rho_r * vr + rho_m * vm)
    
    fr2dot = -(8/15) * (k**2) * vr - (3/5) * k * X[8]
    
    psidot = phidot - (3 * H0**2 * OmegaR / (s0**4 * k**2)) * a**(-2) * (-2 * H_phys * fr2 + fr2dot)
    
    drdot = (4/3) * (3 * phidot + k**2 * vr)
    dmdot = 3 * phidot + k**2 * vm
    vrdot = -(psi + dr / 4) + (1 + 3 * OmegaK / s0**2 * H0**2 / k**2) * fr2 / 2
    vmdot = -H_phys * vm - psi
    
    # --- Assemble derivative vector ---
    derivatives = [adot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

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

kvalues = np.linspace(1e-4/s0, 20/s0,num=300) # originally: kvalues = np.linspace(1e-5,15,num=300)
# kvalues = np.load(folder_path+'allowedK.npy') # only calculate for the allowed K values
ABCmatrices = []
DEFmatrices = []
GHIvectors = []
X1matrices = []
X2matrices = []


#first run background integration to determine s_init
sol2 = solve_ivp(ds_dt, [t0,endtime], [s_bang], method='LSODA', events=at_fcb, 
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
        inits2[0] = 1./inits2[0] # swap s to a
        
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
        inits2[0] = 1./ inits2[0] # swap s to a
    
        sol4 = solve_ivp(dX3_dt, [swaptime,recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)
    
        nthcol = sol4.y[:,-1]
        nthcol = np.array(nthcol)
        nthcol = np.delete(nthcol, 0)
        
        DEF_matrix[:,j] = nthcol
    
    DEFmatrices.append(DEF_matrix)
    
    # # ----------------------------------------------------------------------------
    # # Now find GHIx3 vectors by setting v_r^\infty to 1
    # # ----------------------------------------------------------------------------
    
    # x3 = -(16/945)*(k**4)*(deltaeta**3)
    # x0 = np.zeros(num_variables)
    # inits = np.concatenate(([s_init], x0))
    # inits[9] = x3
    
    # sol5 = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)
    
    # inits2 = sol5.y[:,-1]
    # inits2[0] = np.log(inits2[0])
    
    # sol6 = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)
    
    # #now add resulting vector apart from s to GHIvectors
    # vec = np.array(sol6.y[:,-1])
    # vec = np.delete(vec, 0)
    # GHIvectors.append(vec)
    
    #-----------------------------------------------------------------------------------------------
    # Define X1 matrix for calculating x_endtimes from fcb values
    #-----------------------------------------------------------------------------------------------
    
    # #define coefficients that pop up often
    # coeff1 = (Hinf**3)*(OmegaM/OmegaLambda) / s0**3
    # coeff2 = (Hinf**4)*(OmegaR/OmegaLambda) / s0**4
    # coeff3 = (Hinf**2)*(OmegaK/OmegaLambda) / s0**2
    # de = deltaeta
    # de2 = deltaeta**2
    # de3 = deltaeta**3

    # #define rows and cols  
    # x00 = 0
    # x01 = - (3 / (2*k**2 + 6*coeff3))*coeff1*de - 5* coeff1*coeff3 / (20* (k**2+3*coeff3)) * de3
    # x02 = - (- (12/(2*k**2 + 6*coeff3))*coeff2*de - ((4*k**2*coeff2 + 32*coeff2*coeff3) / (20*(k**2+3*coeff3)))*de3) # - (12/(2*k**2 + 6*coeff3))*coeff2*de - ((4*k**2*coeff2 + 32*coeff2*coeff3) / (20*(k**2+3*coeff3)))*de3 # neg
    # x03 = - 9/ (2*k**2 + 6*coeff3) *coeff1*de - (15*k**2*coeff1 + 60*coeff1*coeff3) / (20*(k**2+3*coeff3)) * de3

    # x10 = x00
    # x11 = x01
    # x12 = x02  # x02 - 32*(k**2*coeff2+3*coeff2*coeff3) * de**3 # neg
    # x13 = x03
    
    # x20 = 1 - (1/6)*(k**2)*de2
    # x21 = - (6*coeff1)/(k**2+3*coeff3)*de - 1/(3*(k**2+3*coeff3))*(coeff1*(3*coeff3-2*k**2))* de3
    # x22 = - ((4*k**4+12*k**2*coeff3-72*coeff2)/(3*k**2+9*coeff3)*de - (6*k**6+26*k**4*coeff3+288*coeff2*coeff3+12*k**2*(2*coeff3**2-7*coeff2)) / (45*(k**2+3*coeff3)) *de3 ) # (4*k**4+12*k**2*coeff3-72*coeff2)/(3*k**2+9*coeff3)*de - (6*k**6+26*k**4*coeff3+288*coeff2*coeff3+12*k**2*(2*coeff3**2-7*coeff2)) / (45*(k**2+3*coeff3)) *de3 # neg
    # x23 = -(18/(k**2+3*coeff3))*coeff1*de - coeff1*(k**2+12*coeff3)/(k**2+3*coeff3)*de3

    # x30 = 0
    # x31 = 1 - (9/(2*k**2+6*coeff3))*coeff1*de + (2*k**2*coeff1-3*coeff1*coeff3)/(4*(k**2+3*coeff3))*de3
    # x32 = -(-18/(k**2+3*coeff3)*coeff2*de - (24*coeff2*coeff3-7*k**2*coeff2)/(5*(k**2+3*coeff3))*de3) # -18/(k**2+3*coeff3)*coeff2*de - (24*coeff2*coeff3-7*k**2*coeff2)/(5*(k**2+3*coeff3))*de3 # neg
    # x33 = -(27/(2*k**2+6*coeff3))*coeff1*de + 0.5*(k**2)*de2 - (15*k**2*coeff1+180*coeff1*coeff3)/(20*(k**2+3*coeff3))*de3
    
    # x40 = -(-0.25*de + (3*k**2+4*coeff3)/120*de3) # -0.25*de + (3*k**2+4*coeff3)/120*de3 # neg
    # x41 = -(3*coeff1/(2*(k**2+3*coeff3))*de2) #3*coeff1/(2*(k**2+3*coeff3))*de2 # neg
    # x42 = 1 - (3*k**4+13*k**2*coeff3+12*(coeff3**2-5*coeff2))/(10*(k**2+3*coeff3))*de2
    # x43 = -((9*coeff1)/(2*(k**2+3*coeff3))*de2) # (9*coeff1)/(2*(k**2+3*coeff3))*de2 # neg
    
    # x50 = 0
    # x51 = -((3/(2*k**2+6*coeff3))*coeff1*de2) # (3/(2*k**2+6*coeff3))*coeff1*de2 # neg
    # x52 = (6/(k**2+3*coeff3))*coeff2*de2 
    # x53 = -(de + (9/(2*k**2+6*coeff3))*coeff1*de2 + coeff3/6*de3) #de + (9/(2*k**2+6*coeff3))*coeff1*de2 + coeff3/6*de3 # neg
    
    # X1 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13], [x20, x21, x22, x23], 
    #             [x30, x31, x32, x33], [x40, x41, x42, x43], [x50, x51, x52, x53]])
    
    # X1matrices.append(X1)
    
    # #-----------------------------------------------------------------------------------------------
    # # Define X2 matrix for calculating x_endtimes from fcb values
    # #-----------------------------------------------------------------------------------------------
    
    # de4 = de**4
    
    # x00 = (1/15)*(k**2)*de2 - k**2/3150*(15*k**2+14*coeff3)*de4
    # x01 = -(4*k**2*coeff1)/(15*(k**2+3*coeff3))*de3
    # x02 = -(-(8/15)*(k**2)*de + 4*k**2/(1575*(k**2+3*coeff3))*(30*k**4+118*k**2*coeff3+84*(coeff3**2-5*coeff2))*de3) # -(8/15)*(k**2)*de + 4*k**2/(1575*(k**2+3*coeff3))*(30*k**4+118*k**2*coeff3+84*(coeff3**2-5*coeff2))*de3 # neg
    # x03 = -(4*coeff1*k**2)/(5*(k**2+3*coeff3))*de3
    
    # x10 = -((1/105)*(k**3)*de3) # (1/105)*(k**3)*de3 # neg
    # x11 = -(-k**3*coeff1/(35*(k**2+3*coeff3))*de4) # -k**3*coeff1/(35*(k**2+3*coeff3))*de4 # neg
    # x12 = -(4/35)*(k**3)*de2 + k**3*(30*k**4+118*k**2*coeff3+84*(coeff3**2-5*coeff2))/(3675*(k**2+3*coeff3))*de4 
    # x13 = -(-(3*k**3/(35*(k**2+3*coeff3)))*coeff1*de4) # -(3*k**3/(35*(k**2+3*coeff3)))*coeff1*de4 # nag
    
    # X2 = np.array([[x00, x01, x02, x03], [x10, x11, x12, x13]])
    
    # X2matrices.append(X2)


    # ----------------------------------------------------------------------------
    # Define X1, X2 matrices
    # ----------------------------------------------------------------------------
    k2 = k**2
    k3 = k**3
    k4 = k**4
    k6 = k**6

    # Powers of deltaeta (de)
    de = deltaeta
    de2 = deltaeta**2
    de3 = deltaeta**3
    de4 = deltaeta**4

    # These coefficients simplify the expressions by grouping common physical terms.
    coeff1 = (Hinf**3) * (OmegaM / OmegaLambda) / s0**3
    coeff2 = (Hinf**4) * (OmegaR / OmegaLambda) / s0**4
    coeff3 = (Hinf**2) * (OmegaK / OmegaLambda) / s0**2

    # Common denominator in many terms
    common_denom = (k2 + 3 * coeff3)

    # ----------------------------------------------------------------------------
    # Define X1 matrix elements
    # X1 maps X_inf = [dr_inf, dm_inf, vr_inf, vm_inf] to x_prime (base variables)
    # where x_prime = [phi', psi', dr', dm', vr', vm']
    # Note: X_inf columns are ordered [0]=dr, [1]=dm, [2]=vr, [3]=vm
    # ----------------------------------------------------------------------------

    # Row 0: phi' (x0j)
    x00 = 0.0

    phi1_dm = -3 * coeff1 / (2 * common_denom)
    phi3_dm = - (5 * coeff1 * coeff3) / (20 * common_denom)
    x01 = phi1_dm * de + phi3_dm * de3

    phi1_vr = -6 * coeff2 / common_denom
    phi3_vr = -(4 * k2 * coeff2 + 32 * coeff2 * coeff3) / (20 * common_denom)
    x02 = phi1_vr * de + phi3_vr * de3

    phi1_vm = -9 * coeff1 / (2 * common_denom)
    phi3_vm = -(15 * k2 * coeff1 + 60 * coeff1 * coeff3) / (20 * common_denom)
    x03 = phi1_vm * de + phi3_vm * de3

    # Row 1: psi' (x1j)
    psi4_dr = -(coeff2 / 5) * de4
    x10 = psi4_dr # phi[4] terms are assumed to be handled elsewhere

    psi1_dm = phi1_dm  # psi[1] term is the same as phi[1]
    psi3_dm = (5 * coeff1 * coeff3) / (20 * common_denom) # Note the sign difference from phi[3]
    x11 = psi1_dm * de + psi3_dm * de3

    psi1_vr = phi1_vr # psi[1] term is the same as phi[1]
    psi3_vr = (28 * k2 * coeff2 + 64 * coeff3 * coeff2) / (20 * common_denom)
    x12 = psi1_vr * de + psi3_vr * de3

    psi1_vm = phi1_vm # psi[1] term is the same as phi[1]
    psi3_vm = (15 * k2 * coeff1) / (20 * common_denom) # Note the different coeff3 part
    x13 = psi1_vm * de + psi3_vm * de3

    # Row 2: dr' (x2j)
    dr2_dr = -1/6 * k2
    dr4_dr = (1/120 * k4) + (k2 * coeff3 / 90)
    x20 = 1.0 + dr2_dr * de2 + dr4_dr * de4

    dr1_dm = -6 * coeff1 / common_denom
    dr3_dm = -(-2 * k2 * coeff1 + 45 * coeff1 * coeff3) / (45 * common_denom)
    x21 = dr1_dm * de + dr3_dm * de3

    dr1_vr = (4 * k4 + 12 * k2 * coeff3 - 72 * coeff2) / (3 * common_denom)
    dr3_vr = -(6 * k6 + 26 * k4 * coeff3 + 288 * coeff2 * coeff3 + 12 * k2 * (2 * coeff3**2 - 7 * coeff2)) / (45 * common_denom)
    x22 = dr1_vr * de + dr3_vr * de3

    dr1_vm = -18 * coeff1 / common_denom
    dr3_vm = -(3 * k2 * coeff1 + 540 * coeff1 * coeff3) / (45 * common_denom)
    x23 = dr1_vm * de + dr3_vm * de3

    # Row 3: dm' (x3j)
    x30 = 0.0

    dm1_dm = -9 * coeff1 / (2 * common_denom)
    dm3_dm = -(-10 * k2 * coeff1 + 15 * coeff1 * coeff3) / (20 * common_denom)
    x31 = 1.0 + dm1_dm * de + dm3_dm * de3

    dm1_vr = -18 * coeff2 / common_denom
    dm3_vr = -(-28 * k2 * coeff2 + 96 * coeff2 * coeff3) / (20 * common_denom)
    x32 = dm1_vr * de + dm3_vr * de3

    dm1_vm = -27 * coeff1 / (2 * common_denom)
    dm2_vm = 0.5 * k2
    dm3_vm = -(15 * k2 * coeff1 + 180 * coeff1 * coeff3) / (20 * common_denom)
    dm4_vm = (k2 * coeff3) / 24.0
    x33 = dm1_vm * de + dm2_vm * de2 + dm3_vm * de3 + dm4_vm * de4

    # Row 4: vr' (x4j)
    vr1_dr = -1/4
    vr3_dr = (3 * k2 + 4 * coeff3) / 120.0
    x40 = vr1_dr * de + vr3_dr * de3

    vr2_dm = (15 * coeff1) / (10 * common_denom)
    vr4_dm = (-315 * k2 * coeff1 + 105 * coeff1 * coeff3) / (4200 * common_denom)
    x41 = vr2_dm * de2 + vr4_dm * de4

    vr2_vr = - (3 * k4 + 13 * k2 * coeff3 + 12 * (coeff3**2 - 5 * coeff2)) / (10 * common_denom)
    vr4_vr = (75 * k6 + 429 * k4 * coeff3 + 4 * k2 * (181 * coeff3**2 - 630 * coeff2) + 336 * coeff3 * (coeff3**2 - 10 * coeff2)) / (4200 * common_denom)
    x42 = 1.0 + vr2_vr * de2 + vr4_vr * de4

    vr2_vm = (45 * coeff1) / (10 * common_denom)
    vr4_vm = (630 * k2 * coeff1 + 5040 * coeff1 * coeff3) / (4200 * common_denom)
    x43 = vr2_vm * de2 + vr4_vm * de4

    # Row 5: vm' (x5j)
    x50 = 0.0

    vm2_dm = 3 * coeff1 / (2 * common_denom)
    x51 = vm2_dm * de2

    vm2_vr = 6 * coeff2 / common_denom
    x52 = vm2_vr * de2

    vm3_vm = coeff3 / 6.0
    x53 = 1.0 * de + (9 * coeff1 / (2 * common_denom)) * de2 + vm3_vm * de3

    # Assemble X1 matrix
    X1 = np.array([
        [x00, x01, x02, x03],
        [x10, x11, x12, x13],
        [x20, x21, x22, x23],
        [x30, x31, x32, x33],
        [x40, x41, x42, x43],
        [x50, x51, x52, x53]
    ])

    # ----------------------------------------------------------------------------
    # Define X2 matrix elements
    # X2 maps X_inf to y_prime (anisotropic variables)
    # where y_prime = [fr2', fr3']
    # ----------------------------------------------------------------------------

    # Row 0: fr2' (y0j)
    fr2_2_dr = 1/15 * k2
    fr2_4_dr = (k2 * (-15 * k2 - 14 * coeff3)) / 3150.0
    y00 = fr2_2_dr * de2 + fr2_4_dr * de4

    fr2_3_dm = -(4 * k2 * coeff1) / (15 * common_denom)
    y01 = fr2_3_dm * de3

    fr2_1_vr = -8/15 * k2
    fr2_3_vr = (4 * k2 * (30 * k4 + 118 * k2 * coeff3 + 84 * (coeff3**2 - 5 * coeff2))) / (1575 * common_denom)
    y02 = fr2_1_vr * de + fr2_3_vr * de3

    fr2_3_vm = -(12 * k2 * coeff1) / (15 * common_denom)
    y03 = fr2_3_vm * de3

    # Row 1: fr3' (y1j)
    fr3_3_dr = 1/105 * k3
    y10 = fr3_3_dr * de3

    y11 = 0.0  # No dependence found

    fr3_2_vr = -4/35 * k3
    fr3_4_vr = (k3 * (30 * k4 + 118 * k2 * coeff3 + 84 * (coeff3**2 - 5 * coeff2))) / (3675 * common_denom)
    y12 = fr3_2_vr * de2 + fr3_4_vr * de4

    fr3_4_vm = -(k3 * 3 * coeff1) / (35 * common_denom)
    y13 = fr3_4_vm * de4

    # Assemble X2 matrix
    X2 = np.array([
        [y00, y01, y02, y03],
        [y10, y11, y12, y13]
    ])

    # Now you can use X1 and X2 in your main script, for example:
    X1matrices.append(X1)
    X2matrices.append(X2)
        
np.save(folder_path + f'L70_kvalues', kvalues)
np.save(folder_path + f'L70_ABCmatrices', ABCmatrices)
np.save(folder_path + f'L70_DEFmatrices', DEFmatrices)
np.save(folder_path + f'L70_GHIvectors', GHIvectors)
np.save(folder_path + f'L70_X1matrices', X1matrices)
np.save(folder_path + f'L70_X2matrices', X2matrices)
