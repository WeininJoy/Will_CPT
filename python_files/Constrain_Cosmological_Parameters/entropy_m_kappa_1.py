import numpy as np
from mpmath import *
import cmath
from scipy.optimize import brentq
from scipy import interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

class Universe:
    def __init__(self, lam, n): 
        
        ### cosmological constants
        self.lam = lam
        self.n = n
        if self.n == 0:
            self.kt = 1./3   # dimensionless curvature = k/lam 
        else:
            self.mt = 1    # dimensionless matter = m/lam 

    # Define the roots a1, a2, a3, a4
    def roots_a(self, rt, kt, mt):

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
    
    def gp(self, x): # x = kt*rt/mt**2
        gp = ( (1+8*x) + (1+8./3*x) * np.sqrt(1+ 32./3.*x) ) / (1 + 3*np.sqrt(1+32./3*x))
        return gp

    def gm(self, x): # x = kt*rt/mt**2
        gm = ( (1+8*x) - (1+8./3*x) * np.sqrt(1+ 32./3.*x) ) / (1 - 3*np.sqrt(1+32./3*x))
        return gm
    
    def partition_plot(self):
        # Create a grid of x and y values
        kt = np.linspace(0, 1, 400)
        mt = np.linspace(0, 5, 400)
        Kt, Mt = np.meshgrid(kt, mt)
        
        rt = self.rt
        def GP(kt,mt): return self.gp(kt*rt/mt**2)
        def GM(kt,mt): return self.gm(kt*rt/mt**2)
        
        def F(kt, mt): return 2* kt**3 / mt**2

        # Evaluate F(x, y)
        Z = F(Kt, Mt)

        # Define the conditions
        condition_1 = Z > GP(Kt,Mt)  # kt^3/mt^2 > gp(x) > gm(x)
        condition_2 = (GP(Kt,Mt) > Z) & (Z > GM(Kt,Mt))  # gp(x) > kt^3/mt^2 > gm(x)
        condition_3 = Z < GM(Kt,Mt)  # gp(x) > gm(x) > kt^3/mt^2 

        # Initialize an array for color coding based on conditions
        colors = np.zeros_like(Z)

        # Assign values based on conditions
        colors[condition_1] = 1  # First region
        colors[condition_2] = 2  # Second region
        colors[condition_3] = 3  # Third region

        # Create the plot
        plt.figure(figsize=(3.5,3))
        plt.contourf(Kt, Mt, colors, levels=[0, 1, 2, 3], cmap='rainbow')
        
        # add description words
        plt.text(0.8, 0.2, 'turnaround', fontsize=7, color='black', bbox=dict(facecolor='white', alpha=0.6))
        plt.text(0.35, 2., 'non-symmetric de-Sitter', fontsize=7, color='black', bbox=dict(facecolor='white', alpha=0.6))
        plt.text(0.05, 0.2, 'symmetric de-Sitter', fontsize=7, color='black', bbox=dict(facecolor='white', alpha=0.6))

        # slope contour
        kt_list = np.logspace(-3, 0, 1000)
        mt_list = np.logspace(-3, np.log10(5), 1000)
        kt, mt = np.meshgrid(kt_list, mt_list)
        levels = [1/2, 1, 2]
        slope_levels = [N* np.pi/2 for N in levels]
        def fmt(x):
            N = 2/np.pi * x
            if N >= 1:
                N = int(N)
                return rf"R={N:d}"
            else:
                return rf"R={N:.1f}"
        slope_vfunc = np.vectorize(self.slope)
        Z = slope_vfunc(kt, mt)
        Z = np.array(Z, dtype=float)
        CS = plt.contour(kt, mt, Z, levels=slope_levels, colors=['silver', 'grey', 'dimgrey'])
        plt.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=7)

        # Add labels and title
        plt.xlabel(r'$\tilde \kappa$')
        plt.ylabel(r'$\tilde m$')
        plt.title('Partition Plot for different universes', fontsize=10)
        plt.savefig("partition_plot.pdf", bbox_inches="tight")

    def solve_a0(self, omega_lambda, rt, mt, kt):
        def f(a0):
            return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
        a0 = brentq(f, 1., 3.)
        return a0.x[0]

    def transform_kt1_3(self, omega_lambda, rt, mt, kt):
        a0 = self.solve_a0(omega_lambda, rt, mt, kt)
        omega_kappa = - omega_lambda / a0**2
        omega_r = rt * omega_kappa**2 / omega_lambda
        omega_m = mt * abs(omega_kappa)**(3./2.) / np.sqrt(omega_lambda)
        return omega_lambda, omega_r, omega_m, omega_kappa
    
    def transform_mt1(self, omega_lambda, rt, mt, kt):
        a0 = self.solve_a0(omega_lambda, rt, mt, kt)
        omega_m = omega_lambda / a0**3
        omega_kappa = -3* kt * omega_m**(2./3) * omega_lambda**(1./3)
        omega_r = rt * omega_m**(4./3) / omega_lambda**(1./3)
        return omega_lambda, omega_r, omega_m, omega_kappa
    
    def slope(self, rt, kt, mt):
        a1, a2, a3, a4 = self.roots_a(rt, kt, mt)
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        rho = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        ReEta = 2 * ellipk(mm)/rho
        slope = cmath.sqrt(kt)* ReEta/cmath.sqrt(3)
        return slope.real
    
    def entropy_g(self, rt, kt, mt):
        a1, a2, a3, a4 = self.roots_a(rt, kt, mt)
        C_K = (a1-a3) * ( (a3-a4)**2 - (a1-a2)**2 - 4./3. * (a1-a4)*(a2-a4) )
        C_E = 8 * kt * (a1-a3) * (a2-a4) / (a2-a3)
        C_pi = ( (a1-a3)**2 - (a2-a4)**2) * ((a1-a4) - (a2-a3))
        mm_bar = ((a1 - a2) * (a3 - a4)) / ((a1 - a3) * (a2 - a4))
        V3 = 2 * np.pi**2 * (kt*self.lam)**(-3./2)  # conformal spatial volume
        coefficient = cmath.sqrt(3*self.lam) * V3 * (a2-a3)/ 2./cmath.sqrt((a1-a3)*(a2-a4)) 
        entropy_g = coefficient * ( C_K* ellipk(mm_bar) + C_E * ellipe(mm_bar) + C_pi * ellippi((a1-a2)/(a1-a3), mm_bar))
        # here needs to be fixed
        if entropy_g.real > 0:
            return entropy_g.real
        else:
            return - entropy_g.real

    def plot_entropy_3D_kt1_3(self):

        mt_list = np.linspace(1.e-1, 1, 20) 
        rt_list = np.linspace(1.e-1, 1, 20) 
        mt, rt = np.meshgrid(mt_list, rt_list)
        kt = self.kt

        def entropy_divide(mt,rt):
            entropy_g = self.entropy_g(rt, kt, mt)
            entropy_lam = 24* np.pi**2 / self.lam
            return entropy_g / entropy_lam
        
        vfunc = np.vectorize(entropy_divide)
        Z = vfunc(mt, rt)
        Z = np.array(Z, dtype=float)

        # set plot
        fig = plt.figure(figsize=(3.375,2.7))
        ax = fig.add_subplot(111)

        # entropy contour
        contour = ax.contourf(mt, rt, Z,cmap=plt.cm.jet)
        cbar = fig.colorbar(contour)
        
        # slope contour
        mt_list = np.linspace(mt_list[0], mt_list[-1] , 200)
        rt_list = np.linspace(rt_list[0], rt_list[-1], 200)
        mt, rt = np.meshgrid(mt_list, rt_list)
        levels = [1, 2, 3]
        slope_levels = [N* np.pi/2 for N in levels]
        def fmt(x):
            N = 2/np.pi * x
            if N >= 1:
                N = int(N)
                return rf"R={N:d}"
            else:
                return rf"R={N:.1f}"
        def slope(mt, rt):
            return self.slope(rt, kt, mt)
        slope_vfunc = np.vectorize(slope)
        Z = slope_vfunc(mt, rt)
        Z = np.array(Z, dtype=float)
        CS = ax.contour(mt, rt, Z, levels=slope_levels, colors=['silver', 'grey', 'dimgrey'])
        ax.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=8)
        ax.set_title(r"$S_g/S_{\Lambda}$ contour plot")
        ax.set_xlabel(r"$\tilde{m}$")
        ax.set_ylabel(r"$\tilde{r}$")
        ax.figure.savefig("entropy_g_contour_kt1_3.pdf", bbox_inches="tight")

    def plot_entropy_3D_mt1(self):

        kt_list = np.linspace(1.e-3, 0.2, 20) 
        rt_list = np.linspace(1.e-3, 1, 20) 
        kt, rt = np.meshgrid(kt_list, rt_list)
        mt = self.mt

        def entropy_divide(kt,rt):
            entropy_g = self.entropy_g(rt, kt, mt)
            entropy_lam = 24* np.pi**2 / self.lam
            return entropy_g / entropy_lam
        
        vfunc = np.vectorize(entropy_divide)
        Z = vfunc(kt, rt)
        Z = np.array(Z, dtype=float)

        # set plot
        fig = plt.figure(figsize=(3.375,2.7))
        ax = fig.add_subplot(111)

        # entropy contour
        level_exp = np.arange(np.floor(np.log10(Z.min())-1),np.ceil(np.log10(Z.max())+1))
        levels = np.power(10, level_exp)
        contour = ax.contourf(kt, rt, Z, levels,cmap=plt.cm.jet, norm=LogNorm())
        cbar = fig.colorbar(contour)
        
        # slope contour
        kt_list = np.linspace(kt_list[0], kt_list[-1] , 200)
        rt_list = np.linspace(rt_list[0], rt_list[-1], 200)
        kt, rt = np.meshgrid(kt_list, rt_list)
        levels = [1/2, 1, 2]
        slope_levels = [N* np.pi/2 for N in levels]
        def fmt(x):
            N = 2/np.pi * x
            if N >= 1:
                N = int(N)
                return rf"R={N:d}"
            else:
                return rf"R={N:.1f}"
        def slope(kt, rt):
            return self.slope(rt, kt, mt)
        slope_vfunc = np.vectorize(slope)
        Z = slope_vfunc(kt, rt)
        Z = np.array(Z, dtype=float)
        CS = ax.contour(kt, rt, Z, levels=slope_levels, colors=['silver', 'grey', 'dimgrey'])
        ax.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=8)
        ax.set_title(r"$S_g/S_{\Lambda}$ contour plot")
        ax.set_xlabel(r"$\tilde{\kappa}$")
        ax.set_ylabel(r"$\tilde{r}$")
        ax.figure.savefig("entropy_g_contour_mt1.pdf", bbox_inches="tight")

    def find_tip_on_contour(self, level=1):

        fig_contour, ax_contour = plt.subplots(1, figsize=(3.375,2.7)) 
        kt_list = np.linspace(0, 0.2, 200)
        rt_list = np.linspace(0, 1, 200) 
        mt = self.mt
        x, y = np.meshgrid(kt_list, rt_list)
        levels = [level]
        slope_levels = [N* np.pi/2 for N in levels]
        def slope(kt, rt):
            return self.slope(rt, kt, mt)
        slope_vfunc = np.vectorize(slope)
        Z = slope_vfunc(x, y)
        Z = np.array(Z, dtype=float)
        # Use contour to find where F(x, y) = level
        contour = ax_contour.contour(x, y, Z, levels=slope_levels)
        # Extract the contour lines (paths)
        contour_paths = contour.collections[0].get_paths()
        # Extract the points from each path
        points = []
        for path in contour_paths:
            points.extend(path.vertices)
        # Convert to NumPy array for easy handling
        points = np.array(points)
        kt_list_contour1 = points[:,0]
        rt_list_contour1 = points[:,1]
        kt_tip = min(kt_list_contour1)
        rt_tip = rt_list_contour1[kt_list_contour1.argmin()]
        print("kt_contour_min = ", min(kt_list_contour1))
        print("corresponding rt = ", rt_list_contour1[kt_list_contour1.argmin()])
        return kt_tip, rt_tip

# Define the Universe
lam = 1
universe = Universe(lam, n=1) ## n=0: kt=1/3 , n=1:mt=1
# universe.plot_entropy_3D_kt1_3()
# universe.plot_entropy_3D_mt1()
# print(universe.transform_kt1_3(0.7, 0.98, 0.9, 1./3)) # omega_lambda, rt, mt, kt
kt_tip, rt_tip = universe.find_tip_on_contour()
print(universe.transform_mt1(0.7, rt_tip, 1, kt_tip)) # omega_lambda, rt, mt, kt

