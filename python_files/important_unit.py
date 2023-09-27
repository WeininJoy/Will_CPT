from astropy import units as u
from astropy.constants import c, G
from scipy.constants import physical_constants
import numpy as np
import math

H0 = 70 * u.km/u.s/u.Mpc
h = 0.7
Omega_lambda = 0.73
Omega_r = 2.47e-5 / h**2
k_c = 2.172

l_p = math.sqrt(1./8./np.pi)  # hbar = c = 8*pi*G = 1
a0 = (Omega_lambda/ Omega_r)**0.25  # suppose a = 1 when the energy density of radiation and the DE are equal.

# cosmological unit
Lambda = Omega_lambda * 3 * H0**2 / c**2   
r = Omega_r * 3 * H0**2 * a0**4 / (8*np.pi*G) 
K = 2.* Lambda / 3. * k_c
Omega_K = - K * c**2 / a0**2 / H0**2

print('a_0, cos = ', a0)
print('Lambda, cos = ', Lambda)
print('r, cos = ', r)
print('K, cos = ', K)
print('Omega_K, cos=', Omega_K.si.value)