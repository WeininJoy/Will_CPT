import numpy as np
from scipy.optimize import root_scalar
from scipy.misc import derivative
import sympy
from scipy.integrate import quad
from scipy.integrate import odeint
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
    def __init__(self, lam, rt): 
        
        ### cosmological constants
        self.lam = lam
        self.rt = rt

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
    
    def a_anal(self, eta, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        zeta = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        C = 1.j/2 * ellipk(1 - mm)
        np_ellipfun = np.frompyfunc(ellipfun, 3, 1)
        Jacobi_sn = np_ellipfun('sn', zeta*eta + C, mm)
        result = (a2*(a3-a1) - a1*(a3-a2)* Jacobi_sn**2) / ((a3 - a1) - (a3 - a2)* Jacobi_sn**2)
        return float(result.real)
    
    def Eta_tot(self, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        zeta = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        Eta = ellipk(mm)/zeta
        print('Eta_tot=', Eta.real)
        print('Delta k=', np.sqrt(3)*np.pi/2/Eta)
        return Eta.real
    
    def find_a_0_inf(self, eta_list, a_list):
        idxzeros = np.where(np.diff(np.sign(a_list)) != 0)[0]
        eta_zeros = eta_list[idxzeros]
        def find_index(lst):
            idx_list = []
            for i in range(len(lst) - 1):  # Go up to len(lst) - 1 to avoid index out of range
                if abs(lst[i+1] - lst[i]) > 1000:
                    idx_list.append(i)
            return idx_list
        idxinf = find_index(a_list)
        eta_inf = eta_list[idxinf]
        return eta_zeros, eta_inf

    def a_switch_mt(self, eta, kt, mt):
        a_value = self.a_anal(eta, kt, mt)
        if a_value > 0:
            mt_effective = mt
        else:
            mt_effective = - mt
        return self.a_anal(eta, kt, mt_effective)

    # Function to compute the derivative of F_modified numerically
    def dadeta(self, eta, kt, mt):
        return derivative(lambda eta_val: self.a_switch_mt(eta_val, kt, mt), eta, dx=1e-6)
    
    def dadeta_difference(self, kt, mt):
        # Define a critical point where a(eta) = 0 
        def a(eta): return self.a_anal(eta, kt, mt)
        sol = root_scalar(a, bracket=[-3, 0])
        eta0 = sol.root

        left_derivative = self.dadeta(eta0 - 1e-6, kt, mt)
        right_derivative = self.dadeta(eta0 + 1e-6, kt, mt)
        return left_derivative - right_derivative

    def plot_a(self, kt, mt):
        eta_list = np.linspace(-5, 15, 1000) 
        # def a_fun(eta): return self.a_anal(eta, kt, mt)
        # sol = root_scalar(a_fun, bracket=[-, 0.0])
        # eta0 = sol.root
        eta_tot = self.Eta_tot(kt, mt)
        a_list = [self.a_anal(eta, kt, mt) for eta in eta_list]
        eta_zeros, eta_inf = self.find_a_0_inf(eta_list, a_list)
        print('eta_zeros:', eta_zeros)
        print('eta_diff:', np.diff(eta_zeros))
        print('Delta_k:', [np.sqrt(3)*np.pi/2/eta_diff for eta_diff in np.diff(eta_zeros)])
        plt.plot([eta-eta_zeros[0] for eta in eta_list], a_list)
        plt.vlines(eta_tot, -5, 5, colors='k', linestyles='dashed')
        plt.ylim(-3, 3)
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$a(\eta)$')

        

lam, rt = 1., 1.
u = Universe(lam, rt)
kt, mt = 0.0, 1.0
u.plot_a(kt, mt)
# u.plot_a_num()
plt.show()
# print(u.Eta_tot(kt, mt))