import numpy as np
from scipy import optimize
from mpmath import *
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
    def __init__(self, lam, rt, n): 
        
        ### cosmological constants
        self.lam = lam
        self.rt = rt    # dimensionless radiation = r/lam
        self.n = n

    # Define the roots a1, a2, a3, a4
    def roots_a(self, kt, mt):  # kt = k/lam, mt = m/lam

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

    def mm(self, a1, a2, a3, a4):
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        return mm

    def rho(self, a1, a2, a3, a4):
        rho = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        return rho

    def ReEta(self, mm, rho):
        ReEta = ellipk(mm)/rho
        return ReEta

    def ImEta(self, mm, rho):
        ImEta = ellipk(1 - mm)/rho
        return ImEta
    
    def divide(self, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
        mm = self.mm(a1, a2, a3, a4)
        rho = self.rho(a1, a2, a3, a4)
        ReEta = self.ReEta(mm, rho)
        ImEta = self.ImEta(mm, rho)
        divide = ReEta / ImEta
        return divide.real
    
    def slope(self, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
        mm = self.mm(a1, a2, a3, a4)
        rho = self.rho(a1, a2, a3, a4)
        ReEta = self.ReEta(mm, rho)
        slope = cmath.sqrt(kt)* ReEta/cmath.sqrt(3)
        return slope.real 
    
    def plot_eta_divide(self):
        
        # set plot 
        fig, ax = plt.subplots(1, figsize=(3.375,2.7))  
        kt_list = np.linspace(0, 1, 100)
        mt_list = np.linspace(0, 5, 100) 
        x, y = np.meshgrid(kt_list, mt_list)
        levels = [1/2,1,2]

        # Real eta divide Im eta contour 
        def fmt(x):
            N = x
            if N >= 1:
                N = int(N)
                return rf"N={N:d}"
            else:
                return rf"N={N:.1f}"
        divide_vfunc = np.vectorize(self.divide)
        Z1 = divide_vfunc(x, y)
        Z1 = np.array(Z1, dtype=float)
        CS1 = ax.contour(x, y, Z1, levels=levels, colors=['lightsteelblue', 'blue', 'darkblue'])
        ax.clabel(CS1, CS1.levels,fmt=fmt, inline=True, fontsize=8)

        # slope contour
        slope_levels = [N* np.pi/2 for N in levels]
        def fmt(x):
            N = 2/np.pi * x
            if N >= 1:
                N = int(N)
                return rf"N={N:d}"
            else:
                return rf"N={N:.1f}"
        slope_vfunc = np.vectorize(self.slope)
        Z2 = slope_vfunc(x, y)
        Z2 = np.array(Z2, dtype=float)
        CS2 = ax.contour(x, y, Z2, linestyles='dashed',levels=slope_levels, colors=['#C0C0C0', '#A0A0A0', '#808080'])
        ax.clabel(CS2, CS2.levels, fmt=fmt, inline=True, fontsize=8)
        plt.xlabel(r"$\kappa/\lambda=-\frac{\Omega_\kappa}{3\sqrt{\Omega_\lambda \Omega_r}}$")
        plt.ylabel(r"$m/\lambda=\frac{\Omega_m}{\Omega_\lambda^{1/4} \Omega_r^{3/4}}$")
        plt.savefig("eta_divide_slope_contour.pdf", bbox_inches="tight")

    def constraints(self, x):
        slope = self.slope(x[0], x[1])
        divide = self.divide(x[0], x[1])
        constraint_1 = slope.real - np.pi/2
        constraint_2 = divide.real - 1
        return [constraint_1, constraint_2]
    
    def find_k_m(self):
        sol = optimize.least_squares(self.constraints, np.array([0.3, 3.6]), bounds=([0.25, 0.35], [3.5, 4]))
        return sol.x

# first assume r=lam=1 -> fix scale factor a_eq=1
lam = 1
rt = 1 # dimensionless radiation = r/lam
universe = Universe(lam, rt, 0)
# universe.plot_eta_divide()
# print(universe.slope(0.3, 3.6)/np.pi*2)
# print(universe.divide(0.3, 3.6))
# kt, mt = universe.find_k_m()
# print('kt =', kt)
# print('mt =', mt)
