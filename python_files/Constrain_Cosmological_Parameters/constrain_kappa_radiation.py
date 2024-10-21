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
    def __init__(self, lam, mt, n): 
        
        ### cosmological constants
        self.lam = lam
        self.mt = mt    # dimensionless matter = m/lam
        self.n = n

    # Define the roots a1, a2, a3, a4
    def roots_a(self, kt, rt):  # kt = k/lam, rt = r/lam

        mt = self.mt
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

    def zeta(self, a1, a2, a3, a4):
        zeta = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        return zeta

    def ReEta(self, mm, zeta):
        ReEta = ellipk(mm)/zeta
        return ReEta

    def ImEta(self, mm, zeta):
        ImEta = ellipk(1 - mm)/zeta
        return ImEta
    
    def divide(self, kt, rt):
        a1, a2, a3, a4 = self.roots_a(kt, rt)
        mm = self.mm(a1, a2, a3, a4)
        zeta = self.zeta(a1, a2, a3, a4)
        ReEta = self.ReEta(mm, zeta)
        ImEta = self.ImEta(mm, zeta)
        divide = ReEta / ImEta
        return divide.real
    
    def slope(self, kt, rt):
        a1, a2, a3, a4 = self.roots_a(kt, rt)
        mm = self.mm(a1, a2, a3, a4)
        zeta = self.zeta(a1, a2, a3, a4)
        ReEta = self.ReEta(mm, zeta)
        slope = cmath.sqrt(kt)* ReEta/cmath.sqrt(3)
        return slope.real 
    
    def plot_eta_divide(self):
        
        # set plot 
        fig, ax = plt.subplots(1, figsize=(3.375,2.7))  
        kt_list = np.linspace(0, 1, 100)
        rt_list = np.linspace(0, 1, 100) 
        x, y = np.meshgrid(kt_list, rt_list)
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
        plt.xlabel(r"$\kappa/\lambda=-\frac{\Omega_\kappa}{3\Omega_m^{2/3} \Omega_\lambda^{1/3}}$")
        plt.ylabel(r"$r/\lambda=\frac{\Omega_r  \Omega_\lambda^{1/3}}{\Omega_m^{4/3}}$")
        plt.savefig("eta_divide_slope_contour_radiation.pdf", bbox_inches="tight")

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
mt = 1 # dimensionless matter = m/lam
universe = Universe(lam, mt, 0)
universe.plot_eta_divide()
# print(universe.slope(0.3, 3.6)/np.pi*2)
# print(universe.divide(0.3, 3.6))
# kt, mt = universe.find_k_m()
# print('kt =', kt)
# print('mt =', mt)

