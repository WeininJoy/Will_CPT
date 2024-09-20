import numpy as np
from mpmath import *
import cmath
from scipy.optimize import root_scalar
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
    def __init__(self, lam, rt, n): 
        
        ### cosmological constants
        self.lam = lam
        self.rt = rt    # dimensionless radiation = r/lam  
        self.n = n
        # self.kt_list_contour1, self.mt_list_contour1 = self.find_kt_mt_on_contour()

    # Define the roots a1, a2, a3, a4
    def roots_a(self, kt, mt):

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
    
    def get_mt_as_kt0(self):
        kt = 0
        def f(mt):
            rt = self.rt
            Pp = -(4./3.)* rt - kt**2
            Qq = -((mt**2) /2) + kt**3 - 4 *kt*rt
            xh = (-Qq + cmath.sqrt(Qq**2 + Pp**3))**(1./3.)
            z0 = 2*kt + xh - Pp/xh
            e0 = cmath.sqrt(z0)
            result = -(-e0)**2 - 2*mt/(-e0) + 6 *kt
            return result.real
        sol = root_scalar(f, bracket=[1, 2])
        return sol.root

    def solve_a0(self, omega_lambda, rt, mt, kt):
        def f(a0):
            return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    def transform(self, omega_lambda, rt, mt, kt):
        a0 = self.solve_a0(omega_lambda, rt, mt, kt)
        print("a0 = ", a0)
        omega_r = omega_lambda / a0**4
        omega_m = mt * omega_lambda**(1/4) * omega_r**(3/4)
        omega_kappa = -3* kt * np.sqrt(omega_lambda* omega_r)
        return omega_lambda, omega_r, omega_m, omega_kappa

    def a(self, eta, lam, rt, mt, kt):
        a1, a2, a3, a4 = self.roots_a(rt, mt, kt)
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        zeta = 1/2 * cmath.sqrt(lam/3 * (a1 - a3) * (a2 - a4))

        x = kt*rt/mt**2
        if 2*kt**3/mt**2 > self.gp(x): # if 2*kt**3/mt**2 > gp(x): turnaround universe
            C = 0  # C = 0 for turnaround universe
        else:
            C = 1.j/2. * ellipk(1-mm) # C = i/2*K(1-mm) for non-turnaround universe
        sn = ellipfun('sn', zeta* eta + C, mm)
        a = ( a2* (a3-a1) - a1*(a3-a2) * sn**2 ) / ( (a3-a1)- (a3-a2)* sn**2 )
        return float(a.real)

    def H_square(a, lam, rt, mt, kt):  # hubble constant H^2
        return 1./3. * lam * ( 1 - 3*kt/a**2 + mt/a**3 + rt/a**4 )
    
    def turningpoint_mt(self):

        def function(kt, mt):
            rt = self.rt
            Pp = -(4./3.)* rt - kt**2
            Qq = -((mt**2) /2) + kt**3 - 4 *kt*rt
            xh = (-Qq + cmath.sqrt(Qq**2 + Pp**3))**(1./3.)
            z0 = 2*kt + xh - Pp/xh
            e0 = cmath.sqrt(z0)
            result_pos = -(+e0)**2 - 2*mt/(+e0) + 6 *kt
            result_neg = -(-e0)**2 - 2*mt/(-e0) + 6 *kt
            return Qq**2 + Pp**3, result_pos, result_neg
        fig, ax = plt.subplots(1, figsize=(3.375,2.7))
        kt_list, mt_list = self.find_kt_mt_on_contour()
        function_list_1 = [function(kt, mt)[0] for kt, mt in zip(kt_list, mt_list)]
        function_list_2 = [function(kt, mt)[1] for kt, mt in zip(kt_list, mt_list)]
        function_list_3 = [function(kt, mt)[2] for kt, mt in zip(kt_list, mt_list)]
        ax.scatter(mt_list, function_list_1, s=[3 for n in range(len(mt_list))], c=['b' for n in range(len(mt_list))], label=r"$Q^2 + P^3$")
        ax.scatter(mt_list, function_list_2, s=[3 for n in range(len(mt_list))], c=['orange' for n in range(len(mt_list))], label=r"$-(e_+)^2 - 2\tilde{m}/e_+ + 6\tilde{\kappa}$")
        ax.scatter(mt_list, function_list_3, s=[3 for n in range(len(mt_list))], c=['g' for n in range(len(mt_list))], label=r"$-(e_-)^2 - 2\tilde{m}/e_- + 6\tilde{\kappa}$")
        absolute_function_1 = list(map(abs, function_list_1))
        absolute_function_2 = list(map(abs, function_list_2))
        absolute_function_3 = list(map(abs, function_list_3))
        root_1 = mt_list[absolute_function_1.index(min(absolute_function_1))]
        root_2 = mt_list[absolute_function_2.index(min(absolute_function_2))]
        root_3 = mt_list[absolute_function_3.index(min(absolute_function_3))]
        ax.vlines(root_1, -3, 6, colors='b', linestyles='dashed', label=r"$\tilde{m} = $"+f"{root_1:.2f}", linewidth=1)
        # ax.vlines(root_2, -3, 6, colors='orange', linestyles='dashed', label=r"$\tilde{m} = $"+f"{root_2:.2f}", linewidth=1)
        ax.vlines(root_3, -3, 6, colors='g', linestyles='dashed', label=r"$\tilde{m} = $"+f"{root_3:.2f}", linewidth=1)
        ax.hlines(0, min(mt_list), max(mt_list), colors='k', linestyles='dashed', linewidth=1)
        ax.set_xlabel(r"$\tilde{m}$")
        # ax.set_ylabel(r"$Q^2 + P^3$")
        ax.set_xlim(0, 2.2)
        ax.set_ylim(-3, 6)
        ax.legend()
        ax.figure.savefig("turning_point_mt.pdf", bbox_inches="tight")
        return 0
    
    def entropy_g(self, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
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
    
    def plot_entropy_2D(self):
        
        def entropy_divide(kt, mt):
            entropy_g = self.entropy_g(kt, mt)
            entropy_lam = 24* np.pi**2 / self.lam
            return entropy_g / entropy_lam
        
        def fmt(x):
            N = x
            if N >= 1:
                N = int(N)
                return rf"R={N:d}"
            else:
                return rf"R={N:.1f}"
        
        fig, ax = plt.subplots(1, figsize=(3.375,2.7))
        levels = [1/2, 1, 2]
        colors = ['silver', 'grey', 'dimgrey']

        for n in range(len(levels)):
            kt_values, mt_values = self.find_kt_mt_on_contour(levels[n])
            entropy_list = [entropy_divide(kt, mt) for kt, mt in zip(kt_values, mt_values)]
            max_idx = entropy_list.index(max(entropy_list))
            ax.vlines(mt_values[max_idx], 8.e-1, max(entropy_list), colors=colors[n], linestyles='dashed', label=fmt(levels[n])+r"$,\tilde{m} = $"+f"{mt_values[max_idx]:.2f}")
            ax.scatter(mt_values, entropy_list, s=[5 for n in range(len(mt_values))], color=colors[n])
        
        ax.set_yscale('log')
        ax.set_xlabel(r"$\tilde{m}$")
        ax.set_ylabel(r"$S_g/S_{\Lambda}$")
        ax.set_xlim(0, 5)
        ax.set_ylim(8.e-1, 8.e2)
        ax.legend()
        ax.figure.savefig("entropy_mt.pdf", bbox_inches="tight")


    def plot_entropy_3D(self):

        kt_list = np.linspace(1.e-3, 2, 20)
        mt_list = np.linspace(0, 1.e3, 20)
        kt, mt = np.meshgrid(kt_list, mt_list)

        def entropy_divide(kt, mt):
            entropy_g = self.entropy_g(kt, mt)
            entropy_lam = 24* np.pi**2 / self.lam
            return entropy_g / entropy_lam
        
        vfunc = np.vectorize(entropy_divide)
        Z = vfunc(kt, mt)
        Z = np.array(Z, dtype=float)

        # set plot
        fig = plt.figure(figsize=(3.375,2.7))
        ax = fig.add_subplot(111)

        # entropy contour
        level_exp = np.arange(np.floor(np.log10(Z.min())-1),np.ceil(np.log10(Z.max())+1))
        levels = np.power(10, level_exp)
        contour = ax.contourf(kt, mt, Z, levels,cmap=plt.cm.jet, norm=LogNorm())
        cbar = fig.colorbar(contour)
        
        # slope contour
        kt_list = np.linspace(kt_list[0], kt_list[-1], 50)
        mt_list = np.linspace(mt_list[0], mt_list[-1], 50)
        kt, mt = np.meshgrid(kt_list, mt_list)
        levels = [1/4,1/3,1/2, 1]
        slope_levels = [N* np.pi/2 for N in levels]
        def fmt(x):
            N = 2/np.pi * x
            if N >= 1:
                N = int(N)
                return rf"R={N:d}"
            else:
                return rf"R=1/{round(1./N):d}"
        slope_vfunc = np.vectorize(self.slope)
        Z = slope_vfunc(kt, mt)
        Z = np.array(Z, dtype=float)
        CS = ax.contour(kt, mt, Z, levels=slope_levels, colors=['silver', 'grey', 'dimgrey'])
        ax.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=8)
        ax.set_title(r"$S_g/S_{\Lambda}$ contour plot")
        ax.set_xlabel(r"$\tilde{\kappa}$")
        ax.set_ylabel(r"$\tilde{m}$")
        ax.figure.savefig("entropy_g_contour_wholeRage.pdf", bbox_inches="tight")

        #####################
        # reproduce Fig. 2 in [2210.01142]
        #####################
        
        # # set plot 
        # fig, ax = plt.subplots(1, figsize=(3.375,2.7))
        # x_list = np.logspace(-2.0, 3.0, num=10)
        # y_list = np.logspace(-2.0, 3.0, num=10)
        # x, y = np.meshgrid(x_list, y_list)
        # kt_list = [np.sqrt(self.rt) / x**(2./3.) for x in x_list]
        # mt_list = [y/x* self.rt**(3./4.) for x, y in zip(x_list, y_list)]
        # kt, mt = np.meshgrid(kt_list, mt_list)

        # def entropy_divide(kt, mt):
        #     entropy_g = self.entropy_g(kt, mt)
        #     entropy_lam = 24* np.pi**2 / self.lam
        #     return entropy_g / entropy_lam
        
        # vfunc = np.vectorize(entropy_divide)
        # Z = vfunc(kt, mt)
        # Z = np.array(Z, dtype=float)
        # levels = np.logspace(-1.0, 3.0, num=5)
        # CS = ax.contour(x, y, Z, levels=levels)
        # ax.clabel(CS, CS.levels, inline=True, fontsize=8)

        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel(r"$\tilde{r}^{3/4}\tilde{\kappa}^{-3/2}$")
        # plt.ylabel(r"$\tilde{\mu}\tilde{\kappa}^{-3/2}$")
        # plt.savefig("entropy_g_contour.pdf", bbox_inches="tight")

    def slope(self, kt, mt):
        a1, a2, a3, a4 = self.roots_a(kt, mt)
        mm = ((a2 - a3) * (a1 - a4))/((a1 - a3) * (a2 - a4))
        rho = 1/2 * cmath.sqrt(self.lam/3 * (a1 - a3) * (a2 - a4))
        ReEta = 2 * ellipk(mm)/rho
        slope = cmath.sqrt(kt)* ReEta/cmath.sqrt(3)
        return slope.real
    
    def plot_slope_contour(self):
        
        # set plot 
        fig, ax = plt.subplots(1, figsize=(3.375,2.7))  
        kt_list = np.linspace(0, 1, 100)
        mt_list = np.linspace(0, 5, 100) 
        x, y = np.meshgrid(kt_list, mt_list)
        levels = [1/2, 1, 2]

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
        Z = slope_vfunc(x, y)
        Z = np.array(Z, dtype=float)
        CS = ax.contour(x, y, Z, linestyles='dashed',levels=slope_levels, colors=['#C0C0C0', '#A0A0A0', '#808080'])
        ax.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=8)
        plt.xlabel(r"$\kappa/\lambda=-\frac{\Omega_\kappa}{3\sqrt{\Omega_\lambda \Omega_r}}$")
        plt.ylabel(r"$m/\lambda=\frac{\Omega_m}{\Omega_\lambda^{1/4} \Omega_r^{3/4}}$")
        plt.savefig("slope_contour.pdf", bbox_inches="tight")
      
    def find_kt_mt_on_contour(self, level=1):

        fig_contour, ax_contour = plt.subplots(1, figsize=(3.375,2.7)) 
        kt_list = np.linspace(0, 5, 100)
        mt_list = np.linspace(0, 1.e3, 100) 
        x, y = np.meshgrid(kt_list, mt_list)
        levels = [level]
        slope_levels = [N* np.pi/2 for N in levels]
        slope_vfunc = np.vectorize(self.slope)
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
        mt_list_contour1 = points[:,1]
        print("kt_contour_min = ", min(kt_list_contour1))
        print("corresponding mt = ", mt_list_contour1[kt_list_contour1.argmin()])
        return kt_list_contour1, mt_list_contour1

    def find_kt_for_mt(self, mt_values, level=1):
        kt_list, mt_list = self.find_kt_mt_on_contour(level)
        func_interpolate = interpolate.interp1d(mt_list, kt_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function
        kt_values = func_interpolate(mt_values)
        return kt_values
    
    def expectation_value_mt(self):
        mt_values = np.linspace(0, 1, 10)
        kt_values = self.find_kt_for_mt(mt_values)
        Int_numerator = 0
        Int_denumerator = 0
        for i in range(len(mt_values)-1):
            mt = mt_values[i]
            kt = kt_values[i]
            entropy_g = self.entropy_g(kt, mt) 
            Int_numerator += mt* cmath.exp(entropy_g) * (mt_values[i+1] - mt_values[i])
            Int_denumerator += cmath.exp(entropy_g) * (mt_values[i+1] - mt_values[i])
        print(Int_numerator)
        print(Int_denumerator)
        # Int_denumerator = quad(integrand_denumerator, 0, 1)
        # expectation_value = Int_numerator[0] / Int_denumerator[0]
        return 0 # expectation_value

# Define the Universe
lam = 1
rt = 1 # dimensionless radiation = r/lam
universe = Universe(lam, rt, 0)
# universe.plot_entropy_3D()

mt = 580
kt = universe.find_kt_for_mt(mt, 1./3)
print(universe.transform(0.7, 1, mt, kt))
# print(universe.transform(0.7, 1, 1.7, 0.01))
# print(universe.transform(0.7, 1, 1.6080402010050252, 0.0617093944077409))
# print(universe.transform(0.7, 1, 6., 0.0617093944077409))
