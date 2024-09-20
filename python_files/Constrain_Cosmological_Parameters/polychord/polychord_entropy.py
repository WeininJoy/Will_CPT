import numpy as np
import cmath
from mpmath import *
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
try:
    from mpi4py import MPI
except ImportError:
    pass

rt = 1 # set r=lambda, which fix the scale factor and the transformation of the parameters

# Define the roots a1, a2, a3, a4
def roots_a(rt, mt, kt):

    Pp = -(4./3.)* rt - kt**2
    Qq = -((mt**2) /2) + kt**3 - 4 *kt*rt
    xh = (-Qq + cmath.sqrt(Qq**2 + Pp**3))**(1./3.)
    # xh = max(xh, 1.e-6)
    z0 = 2*kt + xh - Pp/xh
    e0 = cmath.sqrt(z0)
    app = 1./2* ((+e0) + cmath.sqrt(-(+e0)**2 - 2*mt/(+e0) + 6 *kt))
    apm = 1./2* ((+e0) - cmath.sqrt(-(+e0)**2 - 2*mt/(+e0) + 6 *kt))
    amp = 1./2* ((-e0) + cmath.sqrt(-(-e0)**2 - 2*mt/(-e0) + 6 *kt))
    amm = 1./2* ((-e0) - cmath.sqrt(-(-e0)**2 - 2*mt/(-e0) + 6 *kt))

    a1 = app
    a2 = apm
    a3 = amp
    a4 = amm

    return a1, a2, a3, a4


def entropy_g(lam, rt, mt, kt):
    a1, a2, a3, a4 = roots_a(rt, mt, kt)
    C_K = (a1-a3) * ( (a3-a4)**2 - (a1-a2)**2 - 4./3. * (a1-a4)*(a2-a4) )
    C_E = 8 * kt * (a1-a3) * (a2-a4) / (a2-a3)
    C_pi = ( (a1-a3)**2 - (a2-a4)**2) * ((a1-a4) - (a2-a3))
    mm_bar = ((a1 - a2) * (a3 - a4)) / ((a1 - a3) * (a2 - a4))
    V3 = 2 * np.pi**2 * (kt*lam)**(-3./2)  # conformal spatial volume
    coefficient = cmath.sqrt(3*lam) * V3 * (a2-a3)/ 2./cmath.sqrt((a1-a3)*(a2-a4)) 
    entropy_g = coefficient * ( C_K* ellipk(mm_bar) + C_E * ellipe(mm_bar) + C_pi * ellippi((a1-a2)/(a1-a3), mm_bar))
    if entropy_g.real > 0:
        return entropy_g.real
    else:
        return - entropy_g.real

# def likelihood_dimless(theta):
#     lam, rt, mt, kt = theta
#     # return entropy(theta) theta: parameters, omega_m, omega_r, omega_k, omega_lambda
#     def entropy_divide(lam, rt, mt, kt):
#         entropy = entropy_g(lam, rt, mt, kt)
#         entropy_lam = 24* np.pi**2 / lam
#         return entropy / entropy_lam 
#     vfunc = np.vectorize(entropy_divide, otypes=[float])
#     result = vfunc(lam, rt, mt, kt).item()
#     try:
#         logl = result
#     except:
#         logl = settings.logzero
#     if np.isnan(logl):
#         logl = settings.logzero
#     print('logl='+str(logl))
#     return logl, []

# def likelihood_observable(theta):
#     omega_lambda, mt, kt = theta
#     omega_lambda, omega_r, omega_m, omega_kappa = transform(omega_lambda, rt, mt, kt)
    
#     def entropy_divide(lam, rt, mt, kt):
#         entropy = entropy_g(lam, rt, mt, kt)
#         entropy_lam = 24* np.pi**2 / lam
#         return entropy / entropy_lam 
#     vfunc = np.vectorize(entropy_divide, otypes=[float])
#     result = vfunc(lam, rt, mt, kt).item()
#     try:
#         logl = result
#     except:
#         logl = settings.logzero
#     if np.isnan(logl):
#         logl = settings.logzero
#     print('logl='+str(logl))
#     return logl, []

# #| Initialise the settings
# nDims = 3
# nDerived = 0
# settings = PolyChordSettings(nDims, nDerived)

# #| Define a box uniform prior from -1 to 1

# def prior(hypercube):
#     """ Uniform prior from [-1,1]^D. """
#     return UniformPrior([0.0, 0.0, 0.0, -1.0], [1.0,1.0,1.0,1.0])(hypercube) # omega_lambda, omega_m, omega_k 
#     # return [UniformPrior(1.e-3, 1.0)(hypercube[0]), UniformPrior(1.e-3, 5.0)(hypercube[2]), UniformPrior(1.e-3, 1.0)(hypercube[3])] # omega_lambda, mt, kt

# #| Parameter names
# #! This is a list of tuples (label, latex)
# #! Derived parameters should be followed by a *

# # paramnames = [('H0', 'H_0*'), ('omegar', '\\Omega_r*'), ('omegak', '\\Omega_K*'), ('omegal', '\\Omega_{\\Lambda}*')]
# paramnames = [('lam', '\\lambda'), ('rt', '\\tilde r'), ('mt', '\\tilde m'), ('kt', '\\tilde \\kappa')]

# #| Run PolyChord

# output = pypolychord.run(
#     likelihood,
#     nDims,
#     prior=prior,
#     file_root='entropy',
#     nlive=100,
#     do_clustering=True,
#     read_resume=False,
#     paramnames=paramnames,
# )

# #| Make an anesthetic plot 

# try:
#     from anesthetic import make_2d_axes
#     fig, ax = make_2d_axes(['lam', 'rt', 'mt', 'kt'], figsize=(3.375,3.375))
#     output.plot_2d(ax)
#     fig.savefig('entropy_posterior.pdf')
# except ImportError:
#     print("Install anesthetic for plotting examples.")
