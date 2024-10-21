import numpy as np
from astropy import units as u
from astropy.constants import c

OmegaLambda = 0.679 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.321 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 0
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1

lam = rt = 1
mt = OmegaM / (OmegaLambda**(1./4.) * OmegaR**(3./4.))
a0 = (OmegaLambda/OmegaR)**(1./4.)
kt = 1./3. * ( mt/a0 + rt/a0**2 - (1./OmegaLambda -1)*a0**2 ) 
print('a0=', a0)
print('mt=', mt)
print((2.11e-4/0.657)**2)

H0 = 66.88 * u.km/u.s/u.Mpc  # 66.86 km/s/Mpc
Lambda = OmegaLambda * 3 * H0**2 / c**2 
Lambda = Lambda.si.to(u.Mpc**-2).value
print('Lambda=', Lambda)