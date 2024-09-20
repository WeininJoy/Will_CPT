import numpy as np
from scipy.optimize import root_scalar
from scipy.misc import derivative
import sympy
from mpmath import *
from mpmath import ellipfun
import cmath
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

class Universe:
    def __init__(self, lam): 
        
        ### cosmological constants
        self.lam = lam
        self.rt = 1.5**2  

    # Define the roots a1, a2, a3, a4
    def roots_a(self, kt, mt):  # kt = k/lam, rt = r/lam

        rt = self.rt
        Pp = -(4./3.)* rt - kt**2
        Qq = -((mt**2) /2) + kt**3 - 4 *kt*rt
        xh = (-Qq + cmath.sqrt(Qq**2 + Pp**3))**(1./3.)
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
    
    def a(self, eta, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        zeta = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        C = 1.j/2 * ellipk(1 - mm)
        np_ellipfun = np.frompyfunc(ellipfun, 3, 1)
        Jacobi_sn = np_ellipfun('sn', zeta*eta + C, mm)
        result = (a2*(a3-a1) - a1*(a3-a2)* Jacobi_sn**2) / ((a3 - a1) - (a3 - a2)* Jacobi_sn**2)
        return float(result.real)

    def a_switch_mt(self, eta, kt, mt):
        a_value = self.a(eta, kt, mt)
        if a_value > 0:
            mt_effective = mt
        else:
            mt_effective = - mt
        return self.a(eta, kt, mt_effective)

    # Function to compute the derivative of F_modified numerically
    def dadeta(self, eta, kt, mt):
        return derivative(lambda eta_val: self.a_switch_mt(eta_val, kt, mt), eta, dx=1e-6)
    
    def dadeta_difference(self, kt, mt):
        # Define a critical point where a(eta) = 0 
        def a(eta): return self.a(eta, kt, mt)
        sol = root_scalar(a, bracket=[-3, 0])
        eta0 = sol.root

        left_derivative = self.dadeta(eta0 - 1e-6, kt, mt)
        right_derivative = self.dadeta(eta0 + 1e-6, kt, mt)
        return left_derivative - right_derivative

    def plot_a(self, kt, mt):
        eta_list = np.linspace(-5, 5, 500) 
        def a_fun(eta): return self.a(eta, kt, mt)
        sol = root_scalar(a_fun, bracket=[-3, 0.0])
        eta0 = sol.root
        plt.plot(eta_list, [self.a_switch_mt(eta+eta0, kt, mt) for eta in eta_list])
        plt.ylim(-5, 5)
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$a(\eta)$')
        plt.show()
        

u = Universe(1)
u.plot_a(1., 1.5)