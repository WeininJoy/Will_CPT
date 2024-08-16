import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from astropy import units as u
from astropy.constants import c
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


allowed_kc_list = [0.113284, 0.238231, 0.671658, 0.98607, 0.99969] 
slope_list = [r'$\frac{1}{3}$', r'$\frac{1}{2}$', r'$1$', r'$2$', r'$3$'] 
color_list = ['lightsteelblue', 'cornflowerblue', 'royalblue', 'blue', 'navy', 'darkblue', 'midnightblue']

######################
# Plot kc-k contour
######################

#########
# plot dots for some specific kc

# # make data
# kc_list = np.linspace(0.0, 0.999, 200)
# k_list = np.linspace(1, 7, 100)

# def theta(kc, k):
#     def psi_curved(a, kc, k):
#         return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)
#     try:
#         theta_int  = 2* quad(psi_curved, 0, 1, args=(kc, k))[0]
#     except:
#         theta_int = 0
#     return theta_int

# # find discrete ks
# levels = [n* np.pi/2 for n in range(1,17)]
# kc_sample_list = [0.2, 0.4, 0.6,0.8,0.99]
# discrete_k_list = []
# for kc in kc_sample_list:
#     theta_list = [theta(kc, k) for k in k_list]
#     func_interpolate = interpolate.interp1d(theta_list, k_list, fill_value="extrapolate")
#     k_discrete = func_interpolate(levels)
#     discrete_k_list.append(k_discrete)

# v_func = np.vectorize(theta)

# fig, ax = plt.subplots(1)

# def fmt(x):
#     n = int(2/np.pi * x)
#     return rf"n={n:d}"

# x, y = np.meshgrid(kc_list, k_list)
# Z = v_func(x, y)
# CS = ax.contour(x, y, Z, levels=levels)
# ax.clabel(CS, CS.levels[:6], inline=True, fmt=fmt, fontsize=14)
# for kc in kc_sample_list:
#     discrete_k = discrete_k_list[kc_sample_list.index(kc)]
#     plt.axvline(x=kc, color='k', linestyle='--')
#     plt.plot([kc for i in range(len(discrete_k))], discrete_k, 'r.', markersize=5)
# plt.xlim(0, 0.999)
# plt.ylim(1, 7)
# plt.xlabel(r"$\tilde\kappa$", fontsize=16)
# plt.ylabel(r"$\tilde k$", fontsize=16)
# plt.xticks(kc_sample_list, fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("theta-kc-k_contour.pdf")

###############
# plot dots for some specific k

# # make data
# kc_list = np.linspace(0.0, 0.999, 200)
# k_list = np.linspace(10, 20, 100)

# def theta(kc, k):
#     def psi_curved(a, kc, k):
#         return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)
#     try:
#         theta_int  = 2* quad(psi_curved, 0, 1, args=(kc, k))[0]
#     except:
#         theta_int = 0
#     return theta_int

# # find discrete ks
# levels = [n* np.pi/2 for n in range(10,40)]
# k_sample_list = [n for n in range(int(min(k_list)), int(max(k_list)), 1)]
# discrete_kc_list = []
# for k in k_sample_list:
#     theta_list = [theta(kc, k) for kc in kc_list]
#     func_interpolate = interpolate.interp1d(theta_list, kc_list, fill_value="extrapolate")
#     kc_discrete = func_interpolate(levels)
#     discrete_kc_list.append(kc_discrete)

# v_func = np.vectorize(theta)

# fig, ax = plt.subplots(1)

# def fmt(x):
#     n = int(2/np.pi * x)
#     return rf"n={n:d}"

# x, y = np.meshgrid(kc_list, k_list)
# Z = v_func(x, y)
# CS = ax.contour(x, y, Z, levels=levels)
# ax.clabel(CS, CS.levels[:6], inline=True, fmt=fmt, fontsize=14)
# for k in k_sample_list:
#     discrete_kc = discrete_kc_list[k_sample_list.index(k)]
#     plt.axhline(y=k, color='k', linestyle='--')
#     plt.plot(discrete_kc,[k for i in range(len(discrete_kc))], 'r.', markersize=5)
# plt.xlim(0, 0.999)
# plt.ylim(min(k_list), max(k_list))
# plt.xlabel(r"$\tilde\kappa$", fontsize=16)
# plt.ylabel(r"$\tilde k$", fontsize=16)
# plt.xticks(fontsize=15)
# plt.yticks(k_sample_list, fontsize=15)
# plt.savefig("theta-kc-k_contour.pdf")


######################
# Plot kc-k_int contour
######################

# parameters
n_k = 1000
n_range = 10

# make data
kc_list = np.linspace(0.0, 0.99, 50)
k_int_list = np.linspace(n_k - n_range//2, n_k + n_range//2, n_range)

def theta(kc, k_int):
    trans_constant = np.sqrt(2*kc/3.)
    k = k_int * trans_constant
    def psi_curved(a, kc, k):
        return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)
    try:
        theta_int  = 2* quad(psi_curved, 0, 1, args=(kc, k))[0]
    except:
        theta_int = 0
    return theta_int

v_func = np.vectorize(theta)

levels = [n* np.pi/2 for n in range(500, 1500)]
k_int_sample_list = [n for n in range(int(min(k_int_list)), int(max(k_int_list)), 1)]
discrete_kc_list = []
for k_int in k_int_sample_list:
    theta_list = [theta(kc, k_int) for kc in kc_list]
    func_interpolate = interpolate.interp1d(theta_list, kc_list, fill_value="extrapolate")
    kc_discrete = func_interpolate(levels)
    discrete_kc_list.append(kc_discrete)

fig, ax = plt.subplots(1)

def fmt(x):
    n = int(2/np.pi * x)
    return rf"n={n:d}"

x, y = np.meshgrid(kc_list, k_int_list)
Z = v_func(x, y)
CS = ax.contour(x, y, Z, levels=levels)
ax.clabel(CS, CS.levels[:6], inline=True, fmt=fmt, fontsize=16)
for k_int in k_int_sample_list:
    discrete_kc = discrete_kc_list[k_int_sample_list.index(k_int)]
    # plt.axhline(y=k_int, color='k', linestyle='--')
    plt.plot(discrete_kc,[k_int for i in range(len(discrete_kc))], 'r.', markersize=5)
for n in range(len(allowed_kc_list)):
    plt.axvline(x=allowed_kc_list[n], color='k', linestyle='--', label=rf"slope = "+slope_list[n])
plt.xlabel(r"$\tilde\kappa$", fontsize=16)
plt.ylabel(r"$k_{\text{int}}$", fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0.66,0.68])
# plt.legend(fontsize=15)
plt.savefig("theta-kc-kint_contour.pdf")


    