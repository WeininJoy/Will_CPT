import os
import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import getdist.plots as gplot
import PlanckLogLinearScale

class_type = 'quantised'

if class_type == 'continuous':
    sys.path.insert(1, './class/scripts')
elif class_type == 'quantised':
    sys.path.insert(1, './class_quantised/scripts')
    import generate_spectrum as genk

from classy import Class
from classy import CosmoComputationError

c = 2.99792458 * 10 ** 5  # Speed of light in km/s


def load_all_Planck(spectrum_type):
    
    planck_files = ['./data/COM_PowerSpect_CMB-' + spectrum_type + '-full_R3.01.txt']
    planck_data = []
    for data_file in planck_files:
        planck_data.append(np.loadtxt(data_file))
    roots = ['planck_cl']
    planck_curve = planck_data[0]

    return planck_curve


# Runs the CLASS code with given parameters params in form [omega_b, omega_cdm, 10^4 * omega_ncdm, h, 10^9 * A_s, n_s, tau_reio]
def run_class(params, Neff, N_ncdm):
    
    # create instance of the class " Class "
    LambdaCDM = Class()
    
    # pass input parameters
    LambdaCDM.set({'omega_b':params[0], 'omega_cdm':params[1], 'omega_ncdm':params[2]*1e-4, 'h':params[3], 'A_s':params[4]*1e-9, 'n_s':params[5], 'tau_reio':params[6], 'N_ur':Neff, 'N_ncdm':N_ncdm})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0, 'l_max_scalars':2508})
    
    # run class
    LambdaCDM.compute()
    
    # get all C_l output
    cls = LambdaCDM.lensed_cl(2508)
    
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()
    
    return cls


def save_quantised_spectrum():
    
    file = 'general_lowl.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    best_params = [float(p) for p in results[:-2]]
    lcdm_params = best_params[:7]
    
    # Obtain allowed k values and save to file allowed_k.txt
    genk.general_spectrum(best_params[-2]*1e-3, best_params[-1]*1e-3)
    
    Neff = 2.0328
    N_ncdm = 1
    T0 = 2.7255
    
    # Find corresponding spectra
    cls = run_class(lcdm_params, Neff, N_ncdm)
    
    ll = cls['ell'][2:]
    clTT = cls['tt'][2:]
    clTE = cls['te'][2:]
    clEE = cls['ee'][2:]
    
    filename = 'quantised_best_lowl_cl.txt'
    f = open(filename, 'w')
    for j in range(ll.shape[0]):
        print(ll[j], clTT[j], clTE[j], clEE[j], file=f)
    f.close()

    return
    

def save_fcb_spectrum():

    # Obtain continuous parameters
    opt_file = 'continuous_all_log.txt'
    results = np.loadtxt(opt_file, unpack=True)
    cts_idx = np.argmin(results[-1, :])
    best_params = results[:-1, cts_idx]
    lcdm_params = best_params[:7]

    Neff = 2.0328
    N_ncdm = 1
    T0 = 2.7255
    
    # Get components fron mean_params as needed by generate_spectrum.py
    H0 = best_params[3] * 100
    Omega_m = (best_params[0] + best_params[1] + best_params[2] * 1e-4) / best_params[3] ** 2
    Omega_l = 1 - Omega_m
    components = genk.get_components(T0, H0, Neff, Omega_l, Omega_m)
    
    # Obtain allowed k values and save to file allowed_k.txt
    spectrum_method = 'fcb'
    genk.single_run(components, H0, spectrum_method)

    # Find corresponding spectra
    cls = run_class(lcdm_params, Neff, N_ncdm)

    ll = cls['ell'][2:]
    clTT = cls['tt'][2:]
    clTE = cls['te'][2:]
    clEE = cls['ee'][2:]

    filename = 'LCDM_fcb_with_cts_params_cl.txt'
    f = open(filename, 'w')
    for j in range(ll.shape[0]):
        print(ll[j], clTT[j], clTE[j], clEE[j], file=f)
    f.close()

    return
    

def save_lr_spectrum():

    # Obtain continuous parameters
    opt_file = 'continuous_all_log.txt'
    results = np.loadtxt(opt_file, unpack=True)
    cts_idx = np.argmin(results[-1, :])
    best_params = results[:-1, cts_idx]
    lcdm_params = best_params[:7]

    Neff = 2.0328
    N_ncdm = 1
    T0 = 2.7255

    # Obtain allowed k values and save to file allowed_k.txt
    Omega_m = (best_params[0] + best_params[1] + best_params[2] * 1e-4) / best_params[3] ** 2
    Omega_l = 1 - Omega_m
    genk.generate_lr_spectrum(Omega_l, best_params[3])

    # Find corresponding spectra
    cls = run_class(lcdm_params, Neff, N_ncdm)

    ll = cls['ell'][2:]
    clTT = cls['tt'][2:]
    clTE = cls['te'][2:]
    clEE = cls['ee'][2:]

    filename = 'LR_fcb_with_cts_params_cl.txt'
    f = open(filename, 'w')
    for j in range(ll.shape[0]):
        print(ll[j], clTT[j], clTE[j], clEE[j], file=f)
    f.close()

    return


def save_continuous_spectrum():

    file = 'continuous_lowl.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    best_params = [float(p) for p in results[:-2]]
    lcdm_params = best_params[:7]
    
    Neff = 2.0328
    N_ncdm = 1
    T0 = 2.7255
    
    # Find corresponding spectra
    cls = run_class(lcdm_params, Neff, N_ncdm)
    
    ll = cls['ell'][2:]
    clTT = cls['tt'][2:]
    clTE = cls['te'][2:]
    clEE = cls['ee'][2:]
    
    filename = 'continuous_best_lowl_cl.txt'
    f = open(filename, 'w')
    for j in range(ll.shape[0]):
        print(ll[j], clTT[j], clTE[j], clEE[j], file=f)
    f.close()

    return


def plot_Deltas(axes1, axes2, ll, dl, use_colour):
    
    change1 = 30
    
    change_idx1 = np.squeeze(np.where(ll==change1))
    
    axes1.plot(ll[:change_idx1], dl[:change_idx1], color=use_colour, linewidth=1)
    axes2.plot(ll[change_idx1:], dl[change_idx1:], color=use_colour, linewidth=1)
    
    return


def plot_data(axes1, axes2, ll_lowl, dl_lowl, ll_highl, dl_highl, data_lowl, data_highl):
    
    change1 = 30
    colour_data_lowl  = "0.50"
    colour_data_highl = "0.30"
    
    ll_highl = np.asarray(ll_highl)
    
    axes1.errorbar(ll_lowl, dl_lowl,
                   yerr = [data_lowl[3, :], data_lowl[2, :]],
                   fmt = ".", color = colour_data_lowl, zorder = -2,
                   capsize = 1, elinewidth = 1)
    axes2.errorbar(ll_highl[:], dl_highl[:],
                   yerr = [data_highl[3, :], data_highl[2, :]],
                   fmt = ".", color = colour_data_highl, zorder = -2,
                   capsize = 1, elinewidth = 1)
    return


def compare_data(ll_model, dl_model, data):
    
    diffs = []
    ll = []
    for l_i, dl_i in zip(data[0, :], data[1, :]):
        i = np.where(ll_model==l_i)
        diffs.append(dl_i - dl_model[i])
        ll.append(l_i)
    
    return ll, diffs


def compare_cl():
    
#    high_precision_file = 'continuous_high_precision_cl.txt'
    high_precision_file = 'continuous_high_precision_optimised_cl.txt'
    low_precision_file = 'continuous_low_precision_cl.txt'

#    high_precision_file = 'quantised_high_precision_cl.txt'
#    low_precision_file = './Plotting/k_plane/cl_spectra/cl_A.txt'
    fcb_file = './Plotting/fcb_lcdm_cl.txt'
    
    plot_fcb = False
    units = True
    fractional = False
    
    T0 = 2.7255
    
    ll_low, clTT_low, clTE_low, clEE_low = np.loadtxt(low_precision_file, unpack=True)
    
    if units:
        dlTT_low = ll_low * (ll_low + 1) / (2 * np.pi) * clTT_low * 1e12 * T0 ** 2
        dlTE_low = ll_low * (ll_low + 1) / (2 * np.pi) * clTE_low * 1e12 * T0 ** 2
        dlEE_low = ll_low * (ll_low + 1) / (2 * np.pi) * clEE_low * 1e12 * T0 ** 2
    else:
        dlTT_low = ll_low * (ll_low + 1) / (2 * np.pi) * clTT_low
        dlTE_low = ll_low * (ll_low + 1) / (2 * np.pi) * clTE_low
        dlEE_low = ll_low * (ll_low + 1) / (2 * np.pi) * clEE_low
    
    ll_high, clTT_high, clTE_high, clEE_high = np.loadtxt(high_precision_file, unpack=True)

    if units:
        dlTT_high = ll_high * (ll_high + 1) / (2 * np.pi) * clTT_high * 1e12 * T0 ** 2
        dlTE_high = ll_high * (ll_high + 1) / (2 * np.pi) * clTE_high * 1e12 * T0 ** 2
        dlEE_high = ll_high * (ll_high + 1) / (2 * np.pi) * clEE_high * 1e12 * T0 ** 2
    else:
        dlTT_high = ll_high * (ll_high + 1) / (2 * np.pi) * clTT_high
        dlTE_high = ll_high * (ll_high + 1) / (2 * np.pi) * clTE_high
        dlEE_high = ll_high * (ll_high + 1) / (2 * np.pi) * clEE_high
    
    if plot_fcb:
        # Load and prepare data for FCB
        ll_fcb, clTT_fcb, clTE_fcb, clEE_fcb = np.loadtxt(fcb_file, unpack=True)
        if units:
            dlTT_fcb = ll_fcb * (ll_fcb + 1) / (2 * np.pi) * clTT_fcb * 1e12 * T0 ** 2
            dlTE_fcb = ll_fcb * (ll_fcb + 1) / (2 * np.pi) * clTE_fcb * 1e12 * T0 ** 2
            dlEE_fcb = ll_fcb * (ll_fcb + 1) / (2 * np.pi) * clEE_fcb * 1e12 * T0 ** 2
        else:
            dlTT_fcb = ll_fcb * (ll_fcb + 1) / (2 * np.pi) * clTT_fcb
            dlTE_fcb = ll_fcb * (ll_fcb + 1) / (2 * np.pi) * clTE_fcb
            dlEE_fcb = ll_fcb * (ll_fcb + 1) / (2 * np.pi) * clEE_fcb
    
    # Load and prepare Planck data
    data_folder = os.path.join(os.path.split(__file__)[0], "./Plotting/data")
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
    
    if not units:
        dlTT_data_lowl[:, 1:] = dlTT_data_lowl[:, 1:] / (1e12 * T0 ** 2)
        dlTT_data_highl[:, 1:] = dlTT_data_highl[:, 1:] / (1e12 * T0 ** 2)
        dlTE_data_lowl[:, 1:] = dlTE_data_lowl[:, 1:] / (1e12 * T0 ** 2)
        dlTE_data_highl[:, 1:] = dlTE_data_highl[:, 1:] / (1e12 * T0 ** 2)
        dlEE_data_lowl[:, 1:] = dlEE_data_lowl[:, 1:] / (1e12 * T0 ** 2)
        dlEE_data_highl[:, 1:] = dlEE_data_highl[:, 1:] / (1e12 * T0 ** 2)
    
    # Compare to planck data
    llTT_data_lowl, delta_dlTT_data_lowl = compare_data(ll_low, dlTT_low, dlTT_data_lowl)
    llTT_data_highl, delta_dlTT_data_highl = compare_data(ll_low, dlTT_low, dlTT_data_highl)
    llTE_data_lowl, delta_dlTE_data_lowl = compare_data(ll_low, dlTE_low, dlTE_data_lowl)
    llTE_data_highl, delta_dlTE_data_highl = compare_data(ll_low, dlTE_low, dlTE_data_highl)
    llEE_data_lowl, delta_dlEE_data_lowl = compare_data(ll_low, dlEE_low, dlEE_data_lowl)
    llEE_data_highl, delta_dlEE_data_highl = compare_data(ll_low, dlEE_low, dlEE_data_highl)

    if fractional:
        delta_dlTT_cts =  (dlTT_high - dlTT_low) / dlTT_low
        delta_dlTE_cts = (dlTE_high - dlTE_low) / dlTE_low
        delta_dlEE_cts = (dlEE_high - dlEE_low) / dlEE_low
    else:
        delta_dlTT_cts =  (dlTT_high - dlTT_low)
        delta_dlTE_cts = (dlTE_high - dlTE_low)
        delta_dlEE_cts = (dlEE_high - dlEE_low)

    print(min(delta_dlTT_cts), min(delta_dlTE_cts), min(delta_dlEE_cts))
    print(max(delta_dlTT_cts), max(delta_dlTE_cts), max(delta_dlEE_cts))

    if plot_fcb:
        delta_dlTT_fcb = dlTT_fcb - dlTT_low
        delta_dlTE_fcb = dlTE_fcb - dlTE_low
        delta_dlEE_fcb = dlEE_fcb - dlEE_low
        if fractional:
            delta_dlTT_fcb =  (dlTT_fcb - dlTT_low) / dlTT_low
            delta_dlTE_fcb = (dlTE_fcb - dlTE_low) / dlTE_low
            delta_dlEE_fcb = (dlEE_fcb - dlEE_low) / dlEE_low
        else:
            delta_dlTT_fcb =  (dlTT_fcb - dlTT_low)
            delta_dlTE_fcb = (dlTE_fcb - dlTE_low)
            delta_dlEE_fcb = (dlEE_fcb - dlEE_low)
    
    f, (ax1a, ax2a, ax3a) = plt.subplots(3, 1, sharex=True, figsize=(7, 5.25))
    
    ax1b = ax1a.twinx()
    ax2b = ax2a.twinx()
    ax3b = ax3a.twinx()
    
    continuous_colour = 'k'
    fcb_colour = 'r'
    
    plot_Deltas(ax1a, ax1b, ll_low, delta_dlTT_cts, continuous_colour)
    plot_Deltas(ax2a, ax2b, ll_low, delta_dlTE_cts, continuous_colour)
    plot_Deltas(ax3a, ax3b, ll_low, delta_dlEE_cts, continuous_colour)

    if plot_fcb:
        plot_Deltas(ax1a, ax1b, ll_fcb, delta_dlTT_fcb, fcb_colour)
        plot_Deltas(ax2a, ax2b, ll_fcb, delta_dlTE_fcb, fcb_colour)
        plot_Deltas(ax3a, ax3b, ll_fcb, delta_dlEE_fcb, fcb_colour)

    # Plot Planck data
    if not fractional:
        plot_data(ax1a, ax1b, llTT_data_lowl, delta_dlTT_data_lowl, llTT_data_highl, delta_dlTT_data_highl, dlTT_data_lowl, dlTT_data_highl)
        plot_data(ax2a, ax2b, llTE_data_lowl, delta_dlTE_data_lowl, llTE_data_highl, delta_dlTE_data_highl, dlTE_data_lowl, dlTE_data_highl)
        plot_data(ax3a, ax3b, llEE_data_lowl, delta_dlEE_data_lowl, llEE_data_highl, delta_dlEE_data_highl, dlEE_data_lowl, dlEE_data_highl)

    if plot_fcb:
        custom_lines = [Line2D([0], [0], color=continuous_colour, lw=2),
                        Line2D([0], [0], color=fcb_colour, lw=2)]
        ax3a.legend(custom_lines, [r'High Precision', r'$\Lambda$CDM FCB Quantised'], loc=9, bbox_to_anchor=(0.5, -0.4), fancybox=True, ncol=2)

    if fractional:
        ax1a.set_ylim(-0.001, 0.001)
        ax1b.set_ylim(-0.005, 0.005)
        ax2a.set_ylim(-0.001, 0.001)
        ax2b.set_ylim(-0.05, 0.05)
        ax3a.set_ylim(-0.005, 0.005)
        ax3b.set_ylim(-0.07, 0.07)
    elif units:
#        if class_type == 'continuous':
#            ax1a.set_ylim(-1000, 1000)
#            ax1b.set_ylim(-530, 530)
#            ax2a.set_ylim(-7, 7)
#            ax2b.set_ylim(-7, 7)
#            ax3a.set_ylim(-0.13, 0.13)
#            ax3b.set_ylim(-1.5, 1.5)
#        elif class_type == 'quantised':
#            ax1a.set_ylim(-150, 150)
#            ax1b.set_ylim(-150, 150)
#            ax2a.set_ylim(-2.5, 2.5)
#            ax2b.set_ylim(-2.5, 2.5)
#            ax3a.set_ylim(-0.7, 0.7)
#            ax3b.set_ylim(-0.7, 0.7)
        ax1a.set_ylim(-1000, 1000)
        ax1b.set_ylim(-530, 530)
        ax2a.set_ylim(-7, 7)
        ax2b.set_ylim(-7, 7)
        ax3a.set_ylim(-0.13, 0.13)
        ax3b.set_ylim(-1.5, 1.5)

    ax1a.axvline(x=30, linestyle=":", color="black", linewidth=1)
    ax2a.axvline(x=30, linestyle=":", color="black", linewidth=1)
    ax3a.axvline(x=30, linestyle=":", color="black", linewidth=1)
    
    plt.xscale("planck")

    if fractional:
        labels_fontsize=14
        ax1a.set_ylabel((r"$\frac{\Delta\mathcal{D}_\ell^{\mathrm{TT}}}{\mathcal{D}_\ell^{\mathrm{TT}}}$"),
                        fontsize=labels_fontsize)
        ax2a.set_ylabel((r"$\frac{\Delta\mathcal{D}_\ell^{\mathrm{TE}}}{\mathcal{D}_\ell^{\mathrm{TE}}}$"),
                        fontsize=labels_fontsize)
        ax3a.set_ylabel((r"$\frac{\Delta\mathcal{D}_\ell^{\mathrm{EE}}}{\mathcal{D}_\ell^{\mathrm{EE}}}$"),
                        fontsize=labels_fontsize)
        ax3a.set_xlabel(r"$\ell$",fontsize=labels_fontsize)

    if units:
        labels_fontsize=14
        ax1a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{TT}}$ $\left[\mu\mathrm{K}^2\right]$"),
                        fontsize=labels_fontsize)
        ax2a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{TE}}$ $\left[\mu\mathrm{K}^2\right]$"),
                        fontsize=labels_fontsize, labelpad=23)
        ax3a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{EE}}$ $\left[\mu\mathrm{K}^2\right]$"),
                        fontsize=labels_fontsize, labelpad=13)
        ax3a.set_xlabel(r"$\ell$",fontsize=labels_fontsize)
                    
    plt.tight_layout(rect=[0,0.1,1,1])
                    
#    f.subplots_adjust(hspace=0)
#    plt.savefig('precision.pdf')
#    plt.clf()
    plt.show()

    return


def score_calculated(calculated, full_data, error_above, error_below):
    scored = [0] * len(calculated)
    for idx in range(len(calculated)):
        if calculated[idx] > full_data[idx]:
            scored[idx] = calculated[idx] / error_above[idx]
        elif calculated[idx] < full_data[idx]:
            scored[idx] = calculated[idx] / error_below[idx]
        else:
            scored[idx] = 2 * calculated[idx] / (error_above[idx] + error_below[idx])
    return scored


def score_data(binned_ll, binned_dl, full_ll, full_data, error_above, error_below):
    
    scored = []
    
    binned_dl = np.squeeze(binned_dl)
    binned_ll = np.asarray(binned_ll).astype(int)
    for l_i, dl_i, error_plus, error_minus in zip(binned_ll, binned_dl, error_above, error_below):
        i = np.where(full_ll==l_i)
        
        if dl_i > full_data[i]:
            scored.append(dl_i / error_plus)
        elif dl_i < full_data[i]:
            scored.append(dl_i / error_minus)
        else:
            scored.append(2 * dl_i / (error_plus+ error_minus))

    return scored


def plot_scored_data(axes1, axes2, ll_lowl, dl_lowl, ll_highl, dl_highl):
    
    change1 = 30
    colour_data_lowl  = "0.50"
    colour_data_highl = "0.30"
    
    ll_highl = np.asarray(ll_highl)
    
    axes1.plot(ll_lowl, dl_lowl, ".", color = colour_data_lowl)
    axes2.errorbar(ll_highl[:], dl_highl[:], fmt = ".", color = colour_data_highl, zorder = -2)
    
    return


def compare_scored_cl():
    
#    high_precision_file = 'continuous_high_precision_cl.txt'
    high_precision_file = 'continuous_high_precision_optimised_cl.txt'
    low_precision_file = 'continuous_low_precision_cl.txt'

#    high_precision_file = 'quantised_high_precision_cl.txt'
#    low_precision_file = './Plotting/k_plane/cl_spectra/cl_A.txt'
    fcb_file = './Plotting/fcb_lcdm_cl.txt'
    T0 = 2.7255
        
    ll_low, clTT_low, clTE_low, clEE_low = np.loadtxt(low_precision_file, unpack=True)
    dlTT_low = ll_low * (ll_low + 1) / (2 * np.pi) * clTT_low * 1e12 * T0 ** 2
    dlTE_low = ll_low * (ll_low + 1) / (2 * np.pi) * clTE_low * 1e12 * T0 ** 2
    dlEE_low = ll_low * (ll_low + 1) / (2 * np.pi) * clEE_low * 1e12 * T0 ** 2
    
    ll_high, clTT_high, clTE_high, clEE_high = np.loadtxt(high_precision_file, unpack=True)
    dlTT_high = ll_high * (ll_high + 1) / (2 * np.pi) * clTT_high * 1e12 * T0 ** 2
    dlTE_high = ll_high * (ll_high + 1) / (2 * np.pi) * clTE_high * 1e12 * T0 ** 2
    dlEE_high = ll_high * (ll_high + 1) / (2 * np.pi) * clEE_high * 1e12 * T0 ** 2
    
    # Load and prepare Planck data
    data_folder = os.path.join(os.path.split(__file__)[0], "./Plotting/data")
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
    llTT_data_lowl, delta_dlTT_data_lowl = compare_data(ll_low, dlTT_low, dlTT_data_lowl)
    llTT_data_highl, delta_dlTT_data_highl = compare_data(ll_low, dlTT_low, dlTT_data_highl)
    llTE_data_lowl, delta_dlTE_data_lowl = compare_data(ll_low, dlTE_low, dlTE_data_lowl)
    llTE_data_highl, delta_dlTE_data_highl = compare_data(ll_low, dlTE_low, dlTE_data_highl)
    llEE_data_lowl, delta_dlEE_data_lowl = compare_data(ll_low, dlEE_low, dlEE_data_lowl)
    llEE_data_highl, delta_dlEE_data_highl = compare_data(ll_low, dlEE_low, dlEE_data_highl)
    
    # Obtain full Planck data
    dlTT_data_full = load_all_Planck('TT')
    dlTE_data_full = load_all_Planck('TE')
    dlEE_data_full = load_all_Planck('EE')
    
    # Change lengths
    dlTE_low = dlTE_low[:len(dlTE_data_full[:, 1])]
    dlEE_low = dlEE_low[:len(dlEE_data_full[:, 1])]
    dlTE_high = dlTE_high[:len(dlTE_data_full[:, 1])]
    dlEE_high = dlEE_high[:len(dlEE_data_full[:, 1])]
    
    delta_dlTT_cts = -(dlTT_low - dlTT_high)
    delta_dlTE_cts = -(dlTE_low - dlTE_high)
    delta_dlEE_cts = -(dlEE_low - dlEE_high)
    
    # Obtain deltas for full Planck data
    delta_dlTT_data_full = dlTT_data_full[:, 1] - dlTT_low
    delta_dlTE_data_full = dlTE_data_full[:, 1] - dlTE_low
    delta_dlEE_data_full = dlEE_data_full[:, 1] - dlEE_low
    
    scoredTT_high = score_calculated(delta_dlTT_cts, delta_dlTT_data_full, dlTT_data_full[:, 3], dlTT_data_full[:, 2])
    scoredTE_high = score_calculated(delta_dlTE_cts, delta_dlTE_data_full, dlTE_data_full[:, 3], dlTE_data_full[:, 2])
    scoredEE_high = score_calculated(delta_dlEE_cts, delta_dlEE_data_full, dlEE_data_full[:, 3], dlEE_data_full[:, 2])
    
    # Score the binned data
    scoredTT_data_lowl = score_data(llTT_data_lowl, delta_dlTT_data_lowl, dlTT_data_full[:, 0], delta_dlTT_data_full, dlTT_data_lowl[3, :], dlTT_data_lowl[2, :])
    scoredTT_data_highl = score_data(llTT_data_highl, delta_dlTT_data_highl, dlTT_data_full[:, 0], delta_dlTT_data_full, dlTT_data_highl[3, :], dlTT_data_highl[2, :])
    scoredTE_data_lowl = score_data(llTE_data_lowl, delta_dlTE_data_lowl, dlTE_data_full[:, 0], delta_dlTE_data_full, dlTE_data_lowl[3, :], dlTE_data_lowl[2, :])
    scoredTE_data_highl = score_data(llTT_data_highl, delta_dlTE_data_highl, dlTE_data_full[:, 0], delta_dlTE_data_full, dlTE_data_highl[3, :], dlTE_data_highl[2, :])
    scoredEE_data_lowl = score_data(llEE_data_lowl, delta_dlEE_data_lowl, dlEE_data_full[:, 0], delta_dlEE_data_full, dlEE_data_lowl[3, :], dlEE_data_lowl[2, :])
    scoredEE_data_highl = score_data(llEE_data_highl, delta_dlEE_data_highl, dlEE_data_full[:, 0], delta_dlEE_data_full, dlEE_data_highl[3, :], dlEE_data_highl[2, :])
    
    f, (ax1a, ax2a, ax3a) = plt.subplots(3, 1, sharex=True, figsize=(7, 5.25))
    
    ax1b = ax1a.twinx()
    ax2b = ax2a.twinx()
    ax3b = ax3a.twinx()
    
    plot_colour = 'k'
    
    plot_Deltas(ax1a, ax1b, ll_low, scoredTT_high, plot_colour)
    plot_Deltas(ax2a, ax2b, ll_low[:len(dlTE_data_full[:, 1])], scoredTE_high, plot_colour)
    plot_Deltas(ax3a, ax3b, ll_low[:len(dlEE_data_full[:, 1])], scoredEE_high, plot_colour)
    
    # Plot the scored data
    plot_scored_data(ax1a, ax1b, llTT_data_lowl, scoredTT_data_lowl, llTT_data_highl, scoredTT_data_highl)
    plot_scored_data(ax2a, ax2b, llTE_data_lowl, scoredTE_data_lowl, llTE_data_highl, scoredTE_data_highl)
    plot_scored_data(ax3a, ax3b, llEE_data_lowl, scoredEE_data_lowl, llEE_data_highl, scoredEE_data_highl)
    
    print(max(abs(np.array(scoredTT_high))), max(abs(np.array(scoredTE_high))), max(abs(np.array(scoredEE_high))))
    
    #    custom_lines = [Line2D([0], [0], color=continuous_colour, lw=2),
    #                    Line2D([0], [0], color=fcb_colour, lw=2)]
    #    ax3a.legend(custom_lines, [r'High Precision', r'$\Lambda$CDM FCB Quantised'], loc=9, bbox_to_anchor=(0.5, -0.4), fancybox=True, ncol=2)
    
    ax1a.set_ylim(-3, 3)
    ax1b.set_ylim(-3, 3)
    ax2a.set_ylim(-3, 3)
    ax2b.set_ylim(-3, 3)
    ax3a.set_ylim(-3, 3)
    ax3b.set_ylim(-3, 3)
    
    ax1a.axvline(x=30, linestyle=":", color="black", linewidth=1)
    ax2a.axvline(x=30, linestyle=":", color="black", linewidth=1)
    ax3a.axvline(x=30, linestyle=":", color="black", linewidth=1)
    
    plt.xscale("planck")
    
    labels_fontsize=14
    ax1a.set_ylabel((r"$\frac{\Delta\mathcal{D}_\ell^{\mathrm{TT}}}{\sigma_{\mathrm{TT}}}$"),
                    fontsize=labels_fontsize)
    ax2a.set_ylabel((r"$\frac{\Delta\mathcal{D}_\ell^{\mathrm{TE}}}{\sigma_{\mathrm{TE}}}$"),
                    fontsize=labels_fontsize)
    ax3a.set_ylabel((r"$\frac{\Delta\mathcal{D}_\ell^{\mathrm{EE}}}{\sigma_{\mathrm{EE}}}$"),
                    fontsize=labels_fontsize)
    ax3a.set_xlabel(r"$\ell$",fontsize=labels_fontsize)
                    
    plt.tight_layout(rect=[0,0,1,1])

#    f.subplots_adjust(hspace=0)
#    plt.savefig('scored_precision_quantised.pdf')
#    plt.clf()
    plt.show()

    return
    
    
def plot_single_lowl():

    quantised_cl_file_lowl = '/Users/Deaglan/Desktop/OneDrive - Nexus365/University/Summer_2019/Results/lowl/plus_tau/quantised_fixed_cosmo_nuis_continuous_params_vary_tau_cl.txt'
    continuous_cl_file = 'continuous_cl.txt'
    
    T0 = 2.7255
        
    # Quantised that optimises lowl
    ll, clTT, clTE, clEE = np.loadtxt(quantised_cl_file_lowl, unpack=True)
    dlTT = ll * (ll + 1) / (2 * np.pi) * clTT * 1e12 * T0 ** 2
    dlTE = ll * (ll + 1) / (2 * np.pi) * clTE * 1e12 * T0 ** 2
    dlEE = ll * (ll + 1) / (2 * np.pi) * clEE * 1e12 * T0 ** 2
    
    # Continuous
    ll_cts, clTT_cts, clTE_cts, clEE_cts = np.loadtxt(continuous_cl_file, unpack=True)
    dlTT_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clTT_cts * 1e12 * T0 ** 2
    dlTE_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clTE_cts * 1e12 * T0 ** 2
    dlEE_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clEE_cts * 1e12 * T0 ** 2
    
    delta_dlTT = dlTT - dlTT_cts
    delta_dlTE = dlTE - dlTE_cts
    delta_dlEE = dlEE - dlEE_cts
    
    # Load and prepare Planck data
    data_folder = os.path.join(os.path.split(__file__)[0], "./data")
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

    # Obtain full Planck data
    dlTT_data_full = load_all_Planck('TT')
    delta_dlTT_data_full = dlTT_data_full[:, 1] - dlTT_cts
    
    f, ax = plt.subplots(figsize=(7, 2.5))
    
#    quantised_colour = 'c'
    colour_data_lowl  = "0.50"
    
    ax.plot(ll_cts,  delta_dlTT, linewidth=1)
    ax.errorbar(llTT_data_lowl, delta_dlTT_data_lowl, yerr = [dlTT_data_lowl[3, :], dlTT_data_lowl[2, :]], fmt = ".", color = colour_data_lowl, zorder = -2, capsize = 1, elinewidth = 1)
    
    plt.xscale("planck")
    
    ax.set_xlim(1, 30)
    ax.set_ylim(-1000, 1000)
    
    points = [-1000, -500, 0, 500, 1000]
    ax.set_yticks(points)
    
    labels_fontsize=14
    
    ax.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{TT}} \left[\mu \mathrm{K}^2\right]$"),
                    fontsize=labels_fontsize)
    ax.set_xlabel(r"$\ell$",fontsize=labels_fontsize)
    
    ax.set_title(r'$k_0 = 0.3202 \times 10^{-3} \mathrm{Mpc}^{-1}$, $\Delta k = 0.2312\times 10^{-3} \mathrm{Mpc}^{-1}$')
    
    plt.tight_layout(rect=[0,0,1,1])
    
    plt.savefig('best_lowl.pdf')
    plt.clf()
#    plt.show()


    return
    
    
def plot_lowl():

#    quantised_cl_file = 'quantised_best_lowl_cl.txt'
#    quantised_cl_file_lowl = 'quantised_best_lowl_cl.txt'
#    continuous_cl_file_lowl = 'continuous_best_lowl_cl.txt'

    quantised_cl_file_lowl = '/Users/Deaglan/Desktop/OneDrive - Nexus365/University/Summer_2019/Results/lowl/Plus_As_ns/quantised_fixed_cosmo_nuis_quantised_params_vary_tau_As_ns_cl.txt'
    continuous_cl_file_lowl = '/Users/Deaglan/Desktop/OneDrive - Nexus365/University/Summer_2019/Results/lowl/Plus_As_ns/quantised_fixed_cosmo_nuis_continuous_params_vary_tau_As_ns_cl.txt'
    
    continuous_cl_file = 'continuous_cl.txt'

    T0 = 2.7255
        
    # Quantised that optimises lowl
    ll, clTT, clTE, clEE = np.loadtxt(quantised_cl_file_lowl, unpack=True)
    dlTT = ll * (ll + 1) / (2 * np.pi) * clTT * 1e12 * T0 ** 2
    dlTE = ll * (ll + 1) / (2 * np.pi) * clTE * 1e12 * T0 ** 2
    dlEE = ll * (ll + 1) / (2 * np.pi) * clEE * 1e12 * T0 ** 2
    
    # Continuous that optimises lowl
    ll_cts_lowl, clTT_cts_lowl, clTE_cts_lowl, clEE_cts_lowl = np.loadtxt(continuous_cl_file_lowl, unpack=True)
    dlTT_cts_lowl = ll_cts_lowl * (ll_cts_lowl + 1) / (2 * np.pi) * clTT_cts_lowl * 1e12 * T0 ** 2
    dlTE_cts_lowl = ll_cts_lowl * (ll_cts_lowl + 1) / (2 * np.pi) * clTE_cts_lowl * 1e12 * T0 ** 2
    dlEE_cts_lowl = ll_cts_lowl * (ll_cts_lowl + 1) / (2 * np.pi) * clEE_cts_lowl * 1e12 * T0 ** 2
    
    # Continuous
    ll_cts, clTT_cts, clTE_cts, clEE_cts = np.loadtxt(continuous_cl_file, unpack=True)
    dlTT_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clTT_cts * 1e12 * T0 ** 2
    dlTE_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clTE_cts * 1e12 * T0 ** 2
    dlEE_cts = ll_cts * (ll_cts + 1) / (2 * np.pi) * clEE_cts * 1e12 * T0 ** 2
    
    delta_dlTT = dlTT - dlTT_cts
    delta_dlTE = dlTE - dlTE_cts
    delta_dlEE = dlEE - dlEE_cts
    
    delta_dlTT_cts_lowl = dlTT_cts_lowl - dlTT_cts
    delta_dlTE_cts_lowl = dlTE_cts_lowl - dlTE_cts
    delta_dlEE_cts_lowl = dlEE_cts_lowl - dlEE_cts
    
    # Load and prepare Planck data
    data_folder = os.path.join(os.path.split(__file__)[0], "./data")
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

    # Obtain full Planck data
    dlTT_data_full = load_all_Planck('TT')
    delta_dlTT_data_full = dlTT_data_full[:, 1] - dlTT_cts

    scoredTT = score_calculated(delta_dlTT, delta_dlTT_data_full, dlTT_data_full[:, 3], dlTT_data_full[:, 2])
    scoredTT_cts_lowl = score_calculated(delta_dlTT_cts_lowl, delta_dlTT_data_full, dlTT_data_full[:, 3], dlTT_data_full[:, 2])
    
    # Score the binned data
    scoredTT_data_lowl = score_data(llTT_data_lowl, delta_dlTT_data_lowl, dlTT_data_full[:, 0], delta_dlTT_data_full, dlTT_data_lowl[3, :], dlTT_data_lowl[2, :])
    scoredTT_data_highl = score_data(llTT_data_highl, delta_dlTT_data_highl, dlTT_data_full[:, 0], delta_dlTT_data_full, dlTT_data_highl[3, :], dlTT_data_highl[2, :])

#    f, (ax1a, ax2a) = plt.subplots(2, 1, sharex=True, figsize=(7, 5.25))
    f, ax1a = plt.subplots(figsize=(7, 2.5))

    ax1b = ax1a.twinx()
#    ax2b = ax2a.twinx()
    
    quantised_colour = 'c'
    continuous_colour = 'k'
    
    plot_Deltas(ax1a, ax1b, ll_cts, delta_dlTT, quantised_colour)
    plot_Deltas(ax1a, ax1b, ll_cts, delta_dlTT_cts_lowl, continuous_colour)
    plot_data(ax1a, ax1b, llTT_data_lowl, delta_dlTT_data_lowl, llTT_data_highl, delta_dlTT_data_highl, dlTT_data_lowl, dlTT_data_highl)
    
#    plot_Deltas(ax2a, ax2b, ll_cts, scoredTT, quantised_colour)
#    plot_scored_data(ax2a, ax2a, llTT_data_lowl, scoredTT_data_lowl, llTT_data_highl, scoredTT_data_highl)
    
    custom_lines = [Line2D([0], [0], color=continuous_colour, lw=2),
                    Line2D([0], [0], color=quantised_colour, lw=2)]
#    ax1a.legend(custom_lines, [r'$\Lambda$CDM', r'$\Lambda$CDM + linear quantization'], loc=9, bbox_to_anchor=(0.5, 1.3), fancybox=True, ncol=2)
    ax1a.legend(custom_lines, [r'Continuous Parameters', r'Quantized Parameters'], loc=8, bbox_to_anchor=(0.5, -0.7), fancybox=True, ncol=2)
    
    plt.xscale("planck")

    ax1a.set_xlim(1, 30)
    ax1a.set_ylim(-1000, 1000)
    ax1b.set_ylim(-1000, 1000)
    
    points = [-1000, -500, 0, 500, 1000]
#    labels=[r"\texttt{plik TTTEEE}", r"lowl", r"lowE", r"lensing", r"Total"]
    ax1a.set_yticks(points)
    ax1b.set_yticks(points)
#    ax.set_yticklabels(labels, minor=False)
    
    labels_fontsize=14
    ax1a.set_ylabel((r"$\Delta\mathcal{D}_\ell^{\mathrm{TT}} \left[\mu \mathrm{K}^2\right]$"),
                    fontsize=labels_fontsize)
    ax1a.set_xlabel(r"$\ell$",fontsize=labels_fontsize)
    
    ax1a.set_title(r'Vary $\left( k_0, \Delta k, \tau, A_s, n_s \right)$')
    
#    plt.tight_layout(rect=[0,0,1,0.9])
    plt.tight_layout(rect=[0,0.1,1,1])
    
    plt.savefig('best_lowl_vary_k0_deltak_plus_tau_As_ns.pdf')
    plt.clf()

    return

#save_continuous_spectrum()
#compare_cl()
#compare_scored_cl()
#save_quantised_spectrum()
#plot_lowl()
plot_single_lowl()
#save_fcb_spectrum()
#save_lr_spectrum()
