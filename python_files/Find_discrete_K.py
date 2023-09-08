import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import cmath
from scipy.special import gamma

plt.style.use('seaborn-poster')

#################
# constants
#################

### cosmological constants
Lambda = 1.
C = Lambda / 3.
k_curved = 1  # close:+1, open:-1, flat:0
K = 2. / 3. * Lambda * k_curved  

### others
if k_curved == 0:
    eta_tot = cmath.sqrt(3./Lambda).real * gamma(0.25) * gamma(1.25) / gamma(0.5)
elif k_curved == -1:
    eta_tot = cmath.sqrt(3./Lambda).real * np.pi /2.
elif k_curved == 1: 
    a_tot = 1.
else:
    print('k_curved should be 0, -1, 1')

def f(t, y):
    a = y
    H = np.sqrt( Lambda / 3 + C / a**4 - K / a**2 )
    return [a**2 * H]

print(eta_tot_flat)
eta_eval = np.arange(1.e-5, eta_tot, 1.e-5)
sol = solve_ivp(f, [1.e-5, eta_tot], [1.e-5], t_eval=eta_eval)
plt.plot(sol.t, sol.y[0])
plt.yscale('log')
plt.show()

#################
# Phi(a): perturbation solution 
#################

def phi_sol_open(a, k):
    return 3*1j*((a**2 + 1j*a*cmath.sqrt(k**2 - 4) - 1)*np.exp(1j*cmath.sqrt(k**2 - 4)*cmath.atan(a))/(2*a**3*(k**2 - 8)*cmath.sqrt(k**2 - 4)) + 1j*(a**2 + 1j*a*cmath.sqrt(k**2 - 4) - 1)*(-1j*k**2*a**2 + 1j*a**4 + 2*a**3*cmath.sqrt(k**2 - 4) + 2*1j*a**2 - 2*a*cmath.sqrt(k**2 - 4) + 1j)*np.exp(-1j*cmath.sqrt(k**2 - 4)*cmath.atan(a))/(2*a**3*(k**2 - 8)*cmath.sqrt(k**2 - 4)*(k**2*a**2 + a**4 - 6*a**2 + 1)))

def phi_sol_close(a, k):
    return 3.*1j*(1./(2.*cmath.sqrt(4. + k**2)*(8. + k**2)) * ((1. - a)**(1./2. * cmath.sqrt(-4. - k**2)) * (1. + a)**(-(1./2.) * cmath.sqrt(-4. - k**2)) * (1. + a * (a + cmath.sqrt(-4. - k**2)))) / a**3   +   1j * ( (1. - a)**(-1./2. * cmath.sqrt(-4. - k**2)) * (1. + a)**((1./2.) * cmath.sqrt(-4. - k**2)) * (1. + a**2 + a* cmath.sqrt(-4. - k**2)) * (-1. + 2.*a**2 -a**4 + a**2*k**2 + 2.*a*cmath.sqrt(-4.-k**2)+ 2.*a**3*cmath.sqrt(-4.-k**2) ) ) / ( 2*a**3 * cmath.sqrt(-4.-k**2) *(8 + k**2) * (1. + 6.*a**2 + a**4 + a**2*k**2) )  )
    
# a_list = np.linspace(1.e-5, 0.9999, 1000)
# plt.plot(a_list, [phi_sol_close(a, 10) for a in a_list])
# plt.plot(sol.t, [phi_sol_open(a, 10) for a in sol.y[0]])
# plt.show()