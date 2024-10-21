# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:15:40 2021

@author: MRose
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
import matplotlib as mpl
import os
from classy import Class

"""
Let's create Cls for k up to 10 Mpc^{-1} with 
k0 = 0.1 * 1e-3 Mpc^{-1}
Delta k = 0.225 * 1e-3 Mpc^{-1}
"""

# ----------------------------
# Params 
# ----------------------------

kmax = 10

params_file = './data/Planck_params/continuous_params.txt'
#allowed_k_file = './class_quantised/allowed_k.txt'
allowed2_k_file = './class_quantised/allowed2_k.txt'
cts_cl_file = 'data/cl_files/continuous_cl.txt'
#cts2_cl_file = 'data/cl_files/continuous2_cl.txt'
quant_cl_file = 'cl.txt'
quant2_cl_file = 'cl2.txt'
plot2_file = 'example2_cl.eps'

# Width of l>30 part relative to l<30
upper_to_lower = 1
T0 = 2.7255  # K
Neff = 2.0328
N_ncdm = 1

# ----------------------------

params = np.loadtxt(params_file, unpack=True)

# Get k values and save to file
"""
nmax = np.ceil((kmax - k0) / Dk)
k = np.arange(nmax + 1) * Dk + k0
"""
"""
sqrtlambda = (2.32e-3)/7.22;
kvals = np.load('allowedK.npy');
deltak = np.load('deltaK.npy');
kvals = kvals*sqrtlambda;
deltak = deltak*sqrtlambda;

nmax = int(np.ceil((kmax - kvals[-1])/deltak));
k = np.ndarray.tolist(kvals);

for i in range(nmax):
    k.append(k[-1]+deltak);

np.savetxt(allowed_k_file, np.array(k));
"""
k2 = [1.79e-5, 2.32e-3];
delk = 2.82e-4;
nmax2 = int(np.ceil((kmax - k2[-1])/delk));
#k2 = np.ndarray.tolist(k2);

for j in range(nmax2):
    k2.append(k2[-1]+delk);

np.savetxt(allowed2_k_file, np.array(k2));

# create instance of the class " Class "
LambdaCDM = Class()

# pass input parameters
LambdaCDM.set({'omega_b':params[0], 
                'omega_cdm':params[1], 
                'omega_ncdm':params[2]*1e-4, 
                'h':params[3], 
                'A_s':params[4]*1e-9, 
                'n_s':params[5], 
                'tau_reio':params[6], 
                'N_ur':Neff, 
                'N_ncdm':N_ncdm})
LambdaCDM.set({'output':'tCl,pCl,lCl,mPk',
                'lensing':'yes',
                'P_k_max_1/Mpc':3.0,
                'l_max_scalars':2508})

# run class
LambdaCDM.compute()

# get all C_l output
cls = LambdaCDM.lensed_cl(2508)

# Don't remove these lines - otherwise uses up too much memory
LambdaCDM.struct_cleanup()
LambdaCDM.empty()

# Save to file
ll = cls['ell'][2:]
clTT = cls['tt'][2:]
clTE = cls['te'][2:]
clEE = cls['ee'][2:]

with open(quant2_cl_file, 'w') as f:
    for j in range(ll.shape[0]):
        print(ll[j], clTT[j], clTE[j], clEE[j], file=f)


# --------------------------------
# Here are some plotting functions
# --------------------------------

# k should have units 10^{-3} Mpc^{-1}
def inverse_limber(k):
    r_star = 144.394
    theta_star = 1.041085 / 100
    return k * r_star / (theta_star * 1000)


def convert_ax_to_l_1d(ax, ax_l_x):

    x1, x2 = ax.get_xlim()
    ax_l_x.set_xlim(inverse_limber(x1 * 1e3), inverse_limber(x2 * 1e3))
    ax_l_x.set_xscale("log")
    ax_l_x.figure.canvas.draw()

    return


def plot_Deltas(axes1, axes2, ll, dl, use_colour):

    change1 = 30
    change_idx1 = np.squeeze(np.where(ll==change1))
    axes1.plot(ll[:change_idx1], dl[:change_idx1], color=use_colour, linewidth=1)
    axes2.plot(ll[change_idx1:], dl[change_idx1:], color=use_colour, linewidth=1)

    return


def compare_data(ll_model, dl_model, data):

    diffs = []
    ll = []
    for l_i, dl_i in zip(data[0, :], data[1, :]):
        i = np.where(ll_model==l_i)
        diffs.append(dl_i - dl_model[i])
        ll.append(l_i)

    return ll, diffs


def plot_data(axes1, axes2, ll_lowl, dl_lowl, ll_highl, dl_highl, data_lowl, data_highl):

    change1 = 30
    colour_data_lowl  = "0.50"
    colour_data_highl = "0.30"
    ll_highl = np.asarray(ll_highl)
    axes1.errorbar(ll_lowl, dl_lowl,
                   yerr = [data_lowl[2, :], data_lowl[3, :]],
                   fmt = ".", color = colour_data_lowl, zorder = -2,
                   capsize = 1, elinewidth = 1)
    axes2.errorbar(ll_highl[:], dl_highl[:],
                   yerr = [data_highl[2, :], data_highl[3, :]],
                   fmt = ".", color = colour_data_highl, zorder = -2,
                   capsize = 1, elinewidth = 1)
    return


# Want figures to use latex
# rc('text', usetex=True)
fs = 9;
mpl.rcParams.update({'font.size': fs})

# Load and prepare data for continuous
ll_cts, clTT_cts, clTE_cts, clEE_cts = np.loadtxt(cts_cl_file, unpack=True)
dlTT_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clTT_cts * 1e12 * T0 ** 2
dlTE_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clTE_cts * 1e12 * T0 ** 2
dlEE_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clEE_cts * 1e12 * T0 ** 2

# Load and prepare data for above calculation
ll_quant, clTT_quant, clTE_quant, clEE_quant = np.loadtxt(quant_cl_file, unpack=True)
dlTT_quant = ll_quant * (ll_quant + 1) / (2 * np.pi) * clTT_quant * 1e12 * T0 ** 2
dlTE_quant = ll_quant * (ll_quant + 1) / (2 * np.pi) * clTE_quant * 1e12 * T0 ** 2
dlEE_quant = ll_quant * (ll_quant + 1) / (2 * np.pi) * clEE_quant * 1e12 * T0 ** 2

# Load and prepare data for above calculation
ll_perf, clTT_perf, clTE_perf, clEE_perf = np.loadtxt(quant2_cl_file, unpack=True)
dlTT_perf = ll_perf * (ll_perf + 1) / (2 * np.pi) * clTT_perf * 1e12 * T0 ** 2
dlTE_perf = ll_perf * (ll_perf + 1) / (2 * np.pi) * clTE_perf * 1e12 * T0 ** 2
dlEE_perf = ll_perf * (ll_perf + 1) / (2 * np.pi) * clEE_perf * 1e12 * T0 ** 2


# Load and prepare Planck data
data_folder = os.path.join(os.path.split(__file__)[0], "data/Planck/")
dlTT_data_lowl  = np.loadtxt(os.path.join(data_folder, "planck_spectrum_TT_lowl.txt"))
dlTT_data_lowl = dlTT_data_lowl.transpose()
dlTT_data_highl = np.loadtxt(os.path.join(data_folder, "planck_spectrum_TT_highl.txt"))
dlTT_data_highl = dlTT_data_highl.transpose()
dlTE_data_lowl  = np.loadtxt(os.path.join(data_folder, "planck_spectrum_TE_lowl.txt"))
dlTE_data_lowl = dlTE_data_lowl.transpose()
dlTE_data_highl = np.loadtxt(os.path.join(data_folder, "planck_spectrum_TE_highl.txt"))
dlTE_data_highl = dlTE_data_highl.transpose()
dlEE_data_lowl  = np.loadtxt(os.path.join(data_folder, "planck_spectrum_EE_lowl.txt"))
dlEE_data_lowl = dlEE_data_lowl.transpose()
dlEE_data_highl = np.loadtxt(os.path.join(data_folder, "planck_spectrum_EE_highl.txt"))
dlEE_data_highl = dlEE_data_highl.transpose()


# Compare to planck data
llTT_data_lowl, delta_dlTT_data_lowl = compare_data(ll_cts, dlTT_cts, dlTT_data_lowl)
llTT_data_highl, delta_dlTT_data_highl = compare_data(ll_cts, dlTT_cts, dlTT_data_highl)
llTE_data_lowl, delta_dlTE_data_lowl = compare_data(ll_cts, dlTE_cts, dlTE_data_lowl)
llTE_data_highl, delta_dlTE_data_highl = compare_data(ll_cts, dlTE_cts, dlTE_data_highl)
llEE_data_lowl, delta_dlEE_data_lowl = compare_data(ll_cts, dlEE_cts, dlEE_data_lowl)
llEE_data_highl, delta_dlEE_data_highl = compare_data(ll_cts, dlEE_cts, dlEE_data_highl)

fig_width_pt = 510;  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27;               # Convert pt to inches
golden_mean = 1.2*(np.sqrt(5)-1.0)/2.0;         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt;  # width in inches
fig_height =fig_width*golden_mean;       # height in inches
fig_size = [fig_width,fig_height];


f, [[ax1a, ax1b], [ax2a, ax2b], [ax3a, ax3b]] = plt.subplots(3, 2, figsize=fig_size, 
                                gridspec_kw={'width_ratios': [1, upper_to_lower]})

continuous_colour = 'k'
quantised_colour = 'r'
perf_colour = 'b'

# Plot difference between quantised and continuous models
plot_Deltas(ax1a, ax1b, ll_cts, dlTT_perf - dlTT_cts, perf_colour)
plot_Deltas(ax2a, ax2b, ll_cts, dlTE_perf - dlTE_cts, perf_colour)
plot_Deltas(ax3a, ax3b, ll_cts, dlEE_perf - dlEE_cts, perf_colour)

# Plot difference between quantised and continuous models
plot_Deltas(ax1a, ax1b, ll_cts, dlTT_quant - dlTT_cts, quantised_colour)
plot_Deltas(ax2a, ax2b, ll_cts, dlTE_quant - dlTE_cts, quantised_colour)
plot_Deltas(ax3a, ax3b, ll_cts, dlEE_quant - dlEE_cts, quantised_colour)

# Plot zero for continuous model
plot_Deltas(ax1a, ax1b, ll_cts, dlTT_cts - dlTT_cts, continuous_colour)
plot_Deltas(ax2a, ax2b, ll_cts, dlTE_cts - dlTE_cts, continuous_colour)
plot_Deltas(ax3a, ax3b, ll_cts, dlEE_cts - dlEE_cts, continuous_colour)


# Plot Planck data
plot_data(ax1a, ax1b, llTT_data_lowl, delta_dlTT_data_lowl, llTT_data_highl, delta_dlTT_data_highl, dlTT_data_lowl, dlTT_data_highl)
plot_data(ax2a, ax2b, llTE_data_lowl, delta_dlTE_data_lowl, llTE_data_highl, delta_dlTE_data_highl, dlTE_data_lowl, dlTE_data_highl)
plot_data(ax3a, ax3b, llEE_data_lowl, delta_dlEE_data_lowl, llEE_data_highl, delta_dlEE_data_highl, dlEE_data_lowl, dlEE_data_highl)


# Legend
custom_lines = [Line2D([0], [0], color=perf_colour, lw=2),
                Line2D([0], [0], color=quantised_colour, lw=2),
                Line2D([0], [0], color=continuous_colour, lw=2)]
label = ['Bartlett Solution',\
         'Higher Order',\
         r'$\Lambda$CDM']
    
ax1a.legend(custom_lines, label, loc=9, bbox_to_anchor=(1, 2), fancybox=True, ncol=3)
plt.setp(ax1a.get_legend().get_texts(), fontsize='9')


# Some limits
ax1a.set_ylim(-1000, 1000)
ax1b.set_ylim(-530, 530)
ax2a.set_ylim(-7, 7)
ax2b.set_ylim(-7, 7)
ax3a.set_ylim(-0.13, 0.13)
ax3b.set_ylim(-1.5, 1.5)
ax1a.set_xlim(1.5, 30)
ax2a.set_xlim(1.5, 30)
ax3a.set_xlim(1.5, 30)
ax1b.set_xlim(30, None)
ax2b.set_xlim(30, None)
ax3b.set_xlim(30, None)
ax1a.set_xscale("log")
ax2a.set_xscale("log")
ax3a.set_xscale("log")

points = [2, 10, 30]
ax3a.set_xticks(points)
ax3a.set_xticklabels(points, minor=False)

# Move ticks to right of plot
ax1c = ax1b.twinx()
ax1b.set_yticks([])
y1, y2 = ax1b.get_ylim()
ax1c.set_ylim(y1, y2)
ax2c = ax2b.twinx()
ax2b.set_yticks([])
y1, y2 = ax2b.get_ylim()
ax2c.set_ylim(y1, y2)
ax3c = ax3b.twinx()
ax3b.set_yticks([])
y1, y2 = ax3b.get_ylim()
ax3c.set_ylim(y1, y2)

ax1a.set_xticks([])
ax2a.set_xticks([])
ax1b.set_xticks([])
ax2b.set_xticks([])

points = [500, 1000, 1500, 2000]
ax3b.set_xticks(points)
ax3b.set_xticklabels(points, minor=False)

labels_fontsize=9
ax1a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{TT}}$ $\left[\mu\mathrm{K}^2\right]$"),
                fontsize=labels_fontsize)
ax2a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{TE}}$ $\left[\mu\mathrm{K}^2\right]$"),
                fontsize=labels_fontsize, labelpad=23)
ax3a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{EE}}$ $\left[\mu\mathrm{K}^2\right]$"),
                fontsize=labels_fontsize, labelpad=13)
ax3a.set_xlabel(r"$\ell$",fontsize=labels_fontsize)
ax3a.xaxis.set_label_coords(1.0, -0.25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

f.subplots_adjust(wspace=0)
f.subplots_adjust(hspace=0)
plt.savefig(plot2_file)
plt.clf()