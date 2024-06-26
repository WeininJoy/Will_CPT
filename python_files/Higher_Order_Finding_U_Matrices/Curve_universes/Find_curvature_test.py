import numpy as np
from astropy import units as u
from astropy.constants import c

fcb_time = 6.15 # flat universe
H0_actual = 70 * u.km/u.s/u.Mpc 
OmegaLambda = 0.68
Lambda = OmegaLambda * (3*H0_actual**2) / c**2
devide_sqrtLambda = np.sqrt(1/Lambda).si.to(u.Mpc)
Delta_k = np.sqrt(3) * np.pi / fcb_time / devide_sqrtLambda
print("Delta_k=", Delta_k)

OmegaK = -0.01
K_bar = - np.sign(OmegaK) 
a0_bar = c * np.sqrt(-K_bar/OmegaK)/H0_actual
a0_bar = a0_bar.to(u.Mpc)
integer_Delta_k = 1 / a0_bar
print("integer_Delta_k=", integer_Delta_k)
