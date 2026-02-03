import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

kt_list = np.linspace(1.e-4, 1, 10)
mt_list = np.linspace(300, 450, 10)

#####################
# Contour plot of 1/deltaKint
#####################

deltaKint_arr = np.zeros((10, 10)) # 10x10 array of 1/deltaKint
for i in range(10):
    for j in range(10):
        mt = mt_list[i]
        kt = kt_list[j]
        allowedK = np.load(f'./data/allowedK_{10*i+j}.npy')
        deltaK_list = [allowedK[n+1] - allowedK[n] for n in range(len(allowedK)-1)]
        deltaK = sum(deltaK_list[len(deltaK_list)//2:]) / len(deltaK_list[len(deltaK_list)//2:])
        deltaKint = deltaK / np.sqrt(kt)
        print('Delta k_int = ', 1/deltaKint)
        deltaKint_arr[i,j] = 1/deltaKint

# Create the figure and axes object
fig, ax = plt.subplots()

x, y = np.meshgrid(kt_list, mt_list)
Z = deltaKint_arr

levels = [1./N for N in range(5,100)]
levels = levels[::-1]

# slope contour
def fmt(x):
    N = round(1./x)
    return rf"$\Delta kint=${N:d}"

CS = ax.contour(x, y, Z, levels=levels)
ax.clabel(CS, CS.levels[-10:-1], fmt=fmt, inline=True, fontsize=8)

# lev_exp = np.arange(np.floor(np.log10(Z.min())),
#                    np.ceil(np.log10(Z.max())+1),0.3)
# levs = np.power(10, lev_exp)
# CS = ax.contourf(x, y, Z, levs, locator=ticker.LogLocator(), cmap=cm.PuBu_r
# ax.clabel(CS, CS.levels[:6], inline=True, fmt=fmt, fontsize=16)
cbar = fig.colorbar(CS)

plt.xlabel(r'$\tilde \kappa$')
plt.ylabel(r'$\tilde m$')
plt.title(r'$1/\Delta k_{int}$')
plt.savefig("1_kint_contour.pdf")

#####################
# Contour plot of deltaKphys
#####################

# from scipy.optimize import root_scalar

# OmegaLambda = 0.68
# H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
# lam = 1
# rt = 1
# mt_list = np.linspace(300, 450, 10)
# kt_list = np.linspace(1.e-4, 1, 10)


# # calculate present scale factor a0 and energy densities
# def solve_a0(omega_lambda, rt, mt, kt):
#     def f(a0):
#         return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
#     sol = root_scalar(f, bracket=[1, 1.e3])
#     return sol.root

# def transform(omega_lambda, rt, mt, kt):
#     a0 = solve_a0(omega_lambda, rt, mt, kt)
#     s0 = 1/a0
#     omega_r = omega_lambda / a0**4
#     omega_m = mt * omega_lambda**(1/4) * omega_r**(3/4)
#     omega_kappa = -3* kt * np.sqrt(omega_lambda* omega_r)
#     return s0, omega_lambda, omega_r, omega_m, omega_kappa

# s0_list = []
# for i in range(10):
#     for j in range(10):
#         mt = mt_list[i]
#         kt = kt_list[j]
#         s0, OmegaLambda, OmegaR, OmegaM, OmegaK = transform(OmegaLambda, rt, mt, kt)
#         s0_list.append(s0)

# np.save('/home/wnd22/Documents/Research/Will_CPT/python_files/Higher_Order_Finding_U_Matrices/Curve_universes_a0/data/s0', s0_list)

# # mt_list = np.linspace(300, 450, 10)
# # kt_list = np.linspace(1.e-4, 1, 10)
# # s0_list = np.load('/home/wnd22/Documents/Research/Will_CPT/python_files/Higher_Order_Finding_U_Matrices/Curve_universes_a0/data/s0.npy')

# from astropy import units as u
# from astropy.constants import c

# H0 = 66.86 * u.km/u.s/u.Mpc 

# OmegaLambda = 0.68
# Lambda = OmegaLambda * 3 * H0**2 / c**2 
# Lambda = Lambda.si.to(u.Mpc**-2).value

# deltaKphys_arr = np.zeros((10, 10))
# for i in range(10):
#     for j in range(10):
#         mt = mt_list[i]
#         kt = kt_list[j]
#         allowedK = np.load(f'/home/wnd22/Documents/Research/Will_CPT/python_files/Higher_Order_Finding_U_Matrices/Curve_universes_a0/data/allowedK_{10*i+j}.npy')
#         deltaK_list = [allowedK[n+1] - allowedK[n] for n in range(len(allowedK)-1)]
#         deltaK = sum(deltaK_list[len(deltaK_list)//2:]) / len(deltaK_list[len(deltaK_list)//2:])
#         deltaKphys = deltaK * s0_list[10*i+j] * np.sqrt(Lambda)
#         deltaKphys_arr[i,j] = deltaKphys


# # Create the figure and axes object
# fig, ax = plt.subplots()

# x, y = np.meshgrid(kt_list, mt_list)
# Z = deltaKphys_arr

# CS = ax.contour(x, y, Z)

# # lev_exp = np.arange(np.floor(np.log10(Z.min())),
# #                    np.ceil(np.log10(Z.max())+1),0.3)
# # levs = np.power(10, lev_exp)
# # CS = ax.contourf(x, y, Z, levs, locator=ticker.LogLocator(), cmap=cm.PuBu_r
# # ax.clabel(CS, CS.levels[:6], inline=True, fmt=fmt, fontsize=16)
# cbar = fig.colorbar(CS)

# plt.xlabel(r'$\tilde \kappa$')
# plt.ylabel(r'$\tilde m$')
# plt.title(r'$\Delta k_{phys}$')
# plt.savefig("/home/wnd22/Documents/Research/Will_CPT/python_files/Higher_Order_Finding_U_Matrices/Curve_universes_a0/kphys_contour.pdf")