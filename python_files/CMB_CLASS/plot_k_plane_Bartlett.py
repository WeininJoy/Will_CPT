import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy
from scipy import stats
import getdist.plots as gplot

def get_data(z_param):
    
    param_list = ['k0', 'delta_k', 'plik', 'lowl', 'lowE', 'lensing', 'chi_eff_sq','omega_b', 'omega_cdm', 'omega_ncdm', 'h', 'A_s', 'n_s', 'tau_reio', 'ycal', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217', 'ksz_norm', 'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217', 'calib_100T', 'calib_217T', 'success']
    
    param_num = param_list.index(z_param)
    chisq_num = param_list.index('chi_eff_sq')
    remove_failed = True
    
    # From file data
    
    mypath = './data/cl_files'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    x = []
    y = []
    z = []
    
    cts_chisq = 2767.89
    
    for i in range(len(onlyfiles)):
        
        file = mypath + onlyfiles[i]
        f = open(file, "r")
        results = f.read()
        results = results.split()

        if results[2] != 'Invalid':
            
            if remove_failed:
                if results[-1] == 'True':
                    x.append(float(results[0]) * 1e3)
                    y.append(float(results[1]) * 1e3)
                    if param_num == chisq_num:
                        z.append(float(results[param_num]) - cts_chisq)
                    else:
                        z.append(float(results[param_num]))
            else:
                x.append(float(results[0]) * 1e3)
                y.append(float(results[1]) * 1e3)
                if param_num == chisq_num:
                    z.append(float(results[param_num]) - cts_chisq)
                else:
                    z.append(float(results[param_num]))

    # From optimisation

    param_list = ['omega_b', 'omega_cdm', 'omega_ncdm', 'h', 'A_s', 'n_s', 'tau_reio', 'ycal', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217', 'ksz_norm', 'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217', 'calib_100T', 'calib_217T', 'k0', 'delta_k', 'chi_eff_sq', 'success']
    param_num = param_list.index(z_param)
    chisq_num = param_list.index('chi_eff_sq')
    k0_num = param_list.index('k0')
    delta_k_num = param_list.index('delta_k')
    
    file = 'linear_A.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    x.append(float(results[k0_num]))
    y.append(float(results[delta_k_num]))
    if param_num == chisq_num:
        z.append(float(results[param_num]) - cts_chisq)
    else:
        z.append(float(results[param_num]))
    
    file = 'linear_B.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    x.append(float(results[k0_num]))
    y.append(float(results[delta_k_num]))
    if param_num == chisq_num:
        z.append(float(results[param_num]) - cts_chisq)
    else:
        z.append(float(results[param_num]))

    file = 'linear_C.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    x.append(float(results[k0_num]))
    y.append(float(results[delta_k_num]))
    if param_num == chisq_num:
        z.append(float(results[param_num]) - cts_chisq)
    else:
        z.append(float(results[param_num]))
    
    file = 'linear_D.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    x.append(float(results[k0_num]))
    y.append(float(results[delta_k_num]))
    if param_num == chisq_num:
        z.append(float(results[param_num]) - cts_chisq)
    else:
        z.append(float(results[param_num]))

    file = 'linear_E.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    x.append(float(results[k0_num]))
    y.append(float(results[delta_k_num]))
    if param_num == chisq_num:
        z.append(float(results[param_num]) - cts_chisq)
    else:
        z.append(float(results[param_num]))

    file = 'linear_F.txt'
    f = open(file, "r")
    results = f.read()
    results = results.split()
    x.append(float(results[k0_num]))
    y.append(float(results[delta_k_num]))
    if param_num == chisq_num:
        z.append(float(results[param_num]) - cts_chisq)
    else:
        z.append(float(results[param_num]))

    for chi in z[-6:]:
        print(chi)

    return x, y, z


def get_PLA_value(z_param):
    
    param_list = ['omega_b', 'omega_cdm', 'omega_ncdm', 'h', 'A_s', 'n_s', 'tau_reio', 'ycal', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217', 'ksz_norm', 'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217', 'calib_100T', 'calib_217T']
    PLA_param_list = ['omegabh2', 'omegach2', 'omegancdmh2', 'h', 'A', 'ns', 'tau', 'calPlanck', 'acib217', 'xi', 'asz143', 'aps100', 'aps143', 'aps143217', 'aps217', 'aksz', 'kgal100', 'kgal143', 'kgal143217', 'kgal217', 'galfTE100', 'galfTE100143', 'galfTE100217', 'galfTE143', 'galfTE143217', 'galfTE217', 'cal0', 'cal2']
    
    param_num = param_list.index(z_param)
    PLA_name = PLA_param_list[param_num]
    
    g = gplot.getSinglePlotter(chain_dir=r'../../Planck/')
    roots = ['base_plikHM_TTTEEE_lowl_lowE_lensing']
    
    samples = g.sampleAnalyser.samplesForRoot('base_plikHM_TTTEEE_lowl_lowE_lensing')
    p = samples.getParams()
    
    # Calculate derived parameters
    samples.addDerived(p.H0 / 100, name='h', label=r'h')
    samples.addDerived((p.omegamh2 - p.omegabh2 - p.omegach2) * 1e4, name='omegancdmh2', label=r'\Omega_{\mathrm{ncdm}} \times 10^4')

    return samples.mean(PLA_name), samples.std(PLA_name)
    
def inverse_limber(k):
    r_star = 144.394
    theta_star = 1.041085 / 100
    return k * r_star / (theta_star * 1000)

def convert_ax_to_l_2d(ax, ax_l_x, ax_l_y):
    
    x1, x2 = ax.get_xlim()
    ax_l_x.set_xlim(inverse_limber(x1), inverse_limber(x2))
    ax_l_x.figure.canvas.draw()
    for i in range(int(inverse_limber(x1)), int(inverse_limber(x2)) + 1):
        ax_l_x.axvline(i, color='k', linestyle='--', linewidth=0.5)
    
    y1, y2 = ax.get_ylim()
    ax_l_y.set_ylim(inverse_limber(y1), inverse_limber(y2))
    ax_l_y.figure.canvas.draw()
    for i in range(int(inverse_limber(y1)), int(inverse_limber(y2)) + 1):
        ax_l_y.axhline(i, color='k', linestyle='--', linewidth=0.5)

    return


def convert_ax_to_l_1d(ax, ax_l_x):
    
    x1, x2 = ax.get_xlim()
    ax_l_x.set_xlim(inverse_limber(x1), inverse_limber(x2))
    ax_l_x.figure.canvas.draw()
    for i in range(int(inverse_limber(x1)), int(inverse_limber(x2)) + 1):
        ax_l_x.axvline(i, color='k', linestyle='--', linewidth=0.5)

    return


def remove_points(xmin, xmax, ymin, ymax, x_all, y_all, z_all):
    
    x = []
    y = []
    z = []
    for i in range(len(z_all)):
        if (x_all[i]<xmax) & (x_all[i]>xmin) & (y_all[i]<ymax) & (y_all[i]>ymin):
            x.append(x_all[i])
            y.append(y_all[i])
            z.append(z_all[i])

    return x, y, z


def get_triang(xmin, xmax, ymin, ymax, x, y):
    
    x = np.array(x)
    y = np.array(y)
    x_scaled = (x - xmin) / (xmax - xmin)
    y_scaled = (y - ymin) / (ymax - ymin)
    triang = tri.Triangulation(x_scaled, y_scaled)
    triang.x = triang.x * (xmax - xmin) + xmin
    triang.y = triang.y * (ymax - ymin) + ymin

    return triang


def plot_2d(xmin, xmax, ymin, ymax, savename):
    
    x_all, y_all, z_all = get_data('chi_eff_sq')
    levels = np.linspace(-10, 10, 21)
    
    # Remove points not in desired range
    x, y, z = remove_points(xmin, xmax, ymin, ymax, x_all, y_all, z_all)

    # Scale x and y
    triang = get_triang(xmin, xmax, ymin, ymax, x, y)

    fig, ax = plt.subplots()

    plot = ax.tricontourf(triang, z, levels=levels, cmap='viridis')
    zero_line = ax.tricontour(x, y, z, levels=[0], colors=['k'], linewidths=0.7)
    print(np.amin(z))
    
    ax.plot(x[-1], y[-1], 'b+')
    ax.plot(x[-2], y[-2], 'r+')
    ax.plot(x[-3], y[-3], 'g+')
    ax.plot(x[-4], y[-4], 'c+')
    ax.plot(x[-5], y[-5], 'm+')
    ax.plot(x[-6], y[-6], 'm+')
    
    cbar = fig.colorbar(plot, ax=ax, pad=0.12, format='%.1f', ticks=[-10, -5, 0, 5, 10])
    cbar.set_label('$\Delta \chi^2_{\mathrm{TT}}$')
    cbar.ax.plot([0, 1], [0.5, 0.5], 'k')
    
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax_l_x = ax.twiny()
    ax_l_y = ax.twinx()
    convert_ax_to_l_2d(ax, ax_l_x, ax_l_y)

    ax.set_xlabel('$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    ax.set_ylabel('$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    ax_l_x.set_xlabel('$\ell_0$')
    ax_l_y.set_ylabel('$\Delta \ell$')

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf()
    
    return


def plot_2d_two_scales():
    
    x_all, y_all, z_all = get_data('chi_eff_sq')
    levels = np.linspace(-10, 10, 21)
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5.25))
#    ax2 = ax1.twiny()

    xmin = 0
    xmax = 0.50
    ymin = 0.05
    ymax = 0.235
    ychange = 0.22
    y_top_change = ychange - 0.005
    y_bottom_change = ychange + 0.005

    # Plot lower delta_k region
    x, y, z = remove_points(xmin, xmax, ymin, y_bottom_change, x_all, y_all, z_all)
    triang = get_triang(xmin, xmax, ymin, y_bottom_change, x, y)
    plot = ax2.tricontourf(triang, z, levels=levels, cmap='plasma')
#    zero_line = ax2.tricontour(x, y, z, levels=[0], colors=['k'], linewidths=0.5)
    zero_line = ax2.tricontour(triang, z, levels=[0], colors=['k'], linewidths=0.5)
    ax2.set_ylim(ymin, ychange)
    
    # Plot upper delta_k region
    x, y, z = remove_points(xmin, xmax, y_top_change, ymax, x_all, y_all, z_all)
    triang = get_triang(xmin, xmax, y_top_change, ymax, x, y)
    plot = ax1.tricontourf(triang, z, levels=levels, cmap='plasma')
#    zero_line = ax1.tricontour(x, y, z, levels=[0], colors=['k'], linewidths=0.5)
    zero_line = ax1.tricontour(triang, z, levels=[0], colors=['k'], linewidths=0.5)
    ax1.set_ylim(ychange, ymax)
    ax1.set_xlim(xmin, xmax)
    
    minima_x = x[-6:]
    minima_y = y[-6:]
    minima_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    ax1.plot(minima_x, minima_y, 'ow', mfc='none')
    
    offset = 0.01
    
    for i in range(6):
        ax1.annotate(minima_labels[i], (minima_x[i]+offset, minima_y[i]), color='w')
    
    # Ticks wanted
    lower_ticks = [0.060, 0.100, 0.140, 0.180]
    lower_tick_labels = ['0.060', '0.100', '0.140', '0.180']
    upper_ticks = [0.220, 0.225, 0.230, 0.235]
    ax1.set_yticks(upper_ticks)
    ax2.set_yticks(lower_ticks)
    ax2.set_yticklabels(lower_tick_labels)
    
    cbar = f.colorbar(plot, ax=[ax1,ax2], pad=0.1, format='%.1f', ticks=[-10, -5, 0, 5, 10], anchor=(0.85,0.0))
    cbar.set_label('$\Delta \chi^2_{\mathrm{TT}}$')
    cbar.ax.plot([0, 1], [0.5, 0.5], 'k')
    
    # \ell ticks
    lower_ticks = [1.00, 1.50, 2.00, 2.50, 3.00]
    lower_tick_labels = ['1.00', '1.50', '2.00', '2.50', '3.00']
    
    ax_l_x = ax1.twiny()
    ax_l_y = ax1.twinx()
    convert_ax_to_l_2d(ax1, ax_l_x, ax_l_y)
    ax_l_x.set_xlabel('$\ell_0$')
#    ax_l_y.set_ylabel(' ')

    ax_l_x = ax2.twiny()
    ax_l_y = ax2.twinx()
    ax_l_y.set_yticks(lower_ticks)
    ax_l_y.set_yticklabels(lower_tick_labels)
    ax_l_x.set_xticklabels([])
    convert_ax_to_l_2d(ax2, ax_l_x, ax_l_y)
    
    ax2.set_xlabel('$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
#    ax1.set_ylabel('')
#    ax2.yaxis.set_label_coords(-0.5,-100)
    f.text(0.03, 0.5, '$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$', va='center', rotation='vertical')
    f.text(0.80, 0.5, '$\Delta \ell$', va='center', rotation='vertical')

    plt.tight_layout(rect=[0.05,0,0.8,1])
    f.subplots_adjust(hspace=0)

    plt.savefig('contour_plot_twoscales.pdf')
    plt.clf()

    return


def plot_1d_points():
    
    x, y, z = get_data('chi_eff_sq')
    xmin = 0
    xmax = 0.50
    ymin = 0.20
    ymax = 0.24
    zmin = -10
    zmax = 10
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(7, 5.25))

    ax1.plot(x, z, 'k+')
    ax1.set_ylim(zmin, zmax)
    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel('$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    ax1.set_ylabel('$\Delta \chi^2_{\mathrm{TT}}$')
    ax2.plot(y, z, 'k+')
    ax2.set_ylim(zmin, zmax)
    ax2.set_xlim(ymin, ymax)
    
    ax_l_x = ax1.twiny()
    convert_ax_to_l_1d(ax1, ax_l_x)
    ax_l_x.set_xlabel('$\ell_0$')
    
    ax_l_x = ax2.twiny()
    convert_ax_to_l_1d(ax2, ax_l_x)
    ax_l_x.set_xlabel('$\Delta \ell$')
    
    ax2.set_xlabel('$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    ax2.set_ylabel('$\Delta \chi^2_{\mathrm{TT}}$')
    plt.tight_layout()

    plt.savefig('1d_plot_points.pdf')
    plt.clf()

    return


def plot_1d_bins():
    
    x, y, z = get_data('chi_eff_sq')
    xmin = 0
    xmax = 0.50
    ymin = 0.15
    ymax = 0.25
    zmin = -10
    zmax = 5
    nxbins = 80
    nybins = 600
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(7, 5.25))
    
    zx_min, zx_edges, zx_number  = scipy.stats.binned_statistic(x, z, statistic='min', bins=nxbins, range=None)
    zx_width = (zx_edges[1] - zx_edges[0])
    zx_centres = zx_edges[1:] - zx_width/2
    
    zy_min, zy_edges, zy_number = scipy.stats.binned_statistic(y, z, statistic='min', bins=nybins, range=None)
    zy_width = (zy_edges[1] - zy_edges[0])
    zy_centres = zy_edges[1:] - zy_width/2
    
    ax1.plot(zx_centres, zx_min, 'k')
    ax1.plot(x[-2], z[-2], 'r+')
    ax1.plot(x[-1], z[-1], 'b+')
    ax1.set_ylim(zmin, zmax)
    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel('$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    ax1.set_ylabel('$\Delta \chi^2_{\mathrm{TT}}$')
    
    ax2.plot(zy_centres, zy_min, 'k')
    ax2.plot(y[-2], z[-2], 'r+')
    ax2.plot(y[-1], z[-1], 'b+')
    ax2.set_ylim(zmin, zmax)
    ax2.set_xlim(ymin, ymax)
#
    ax_l_x = ax1.twiny()
    convert_ax_to_l_1d(ax1, ax_l_x)
    ax_l_x.set_xlabel('$\ell_0$')

    ax_l_x = ax2.twiny()
    convert_ax_to_l_1d(ax2, ax_l_x)
    ax_l_x.set_xlabel('$\Delta \ell$')

    ax2.set_xlabel('$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    ax2.set_ylabel('$\Delta \chi^2_{\mathrm{TT}}$')
    plt.tight_layout()

    plt.savefig('1d_plot_bins.pdf')
    plt.clf()

    return


def plot_param_plane():
    
    xmin = 0.0
    xmax = 0.50
    ymin = 0.05
    ymax = 0.25
    
    param_list = ['omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'tau_reio']
    title_list = ['$\Omega_b$', '$\Omega_{\mathrm{cdm}}$', '$h$', '$A_s$', '$n_s$', '$t_{\mathrm{reio}}$']

    fig, axs = plt.subplots(3,2, figsize=(7, 5.25*1.5))

    axs = axs.ravel()

    for i in range(len(param_list)):
        param = param_list[i]
        x, y, z = get_data(param)
        z = np.array(z)
        PLA_z, PLA_sigma = get_PLA_value(param)
        z = (z - PLA_z) / PLA_sigma
#        zmin = int(min(z))
#        zmax = int(max(z))
#        if zmin + zmax > 0:
#            zmin = -zmax
#        else:
#            zmax = -zmin
        zmin = -0.6
        zmax = 0.6
#        levels = np.linspace(zmin, zmax, zmax - zmin + 1)
        levels = np.linspace(zmin, zmax, 10 * (zmax - zmin) + 1)
        axs[i].patch.set_color('k')
        plot = axs[i].tricontourf(x, y, z, levels=levels)
        zero_line = axs[i].tricontour(x, y, z, levels=[0], colors=['w'], linewidths=0.7)
        cbar = fig.colorbar(plot, format='%.1f', ticks=[zmin, zmin / 2, 0, zmax / 2, zmax], pad=0.1, ax=axs[i])
        cbar.ax.plot([0, 1], [0.5, 0.5], 'w')
        cbar.set_label('$\Delta $' + title_list[i] + '$/ \sigma $')
        axs[i].plot(x[-1], y[-1], 'b+')
        axs[i].plot(x[-2], y[-2], 'r+')
        axs[i].set_ylim(ymin, ymax)
        axs[i].set_xlim(xmin, xmax)
        axs[i].set_title(title_list[i])
        plt.close(2)

    axs[0].set_ylabel(r'$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    axs[2].set_ylabel(r'$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    axs[4].set_ylabel(r'$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    axs[4].set_xlabel(r'$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    axs[5].set_xlabel(r'$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')

    plt.tight_layout()
    plt.savefig('param_change.pdf')
    return


def contour_plots():
    
    # Not zoomed plot
    xmin = 0.0; xmax = 0.50
    ymin = 0.05; ymax = 0.25
    savename = 'contour_plot.pdf'
    plot_2d(xmin, xmax, ymin, ymax, savename)
    
    # Zoomed plot
    xmin = 0.0; xmax = 0.50
    ymin = 0.215; ymax = 0.235
    savename = 'contour_plot_zoomed.pdf'
    plot_2d(xmin, xmax, ymin, ymax, savename)

    return



def plot_lowl_2d():
    
    opt_file = '../../lowl_k_plane.txt'
    x_all, y_all, z_all = np.loadtxt(opt_file, unpack=True)

    # Load point from minimisation
    min_file = '../../Results/lowl/Vary_k0_deltak/quantised_fixed_cosmo_nuis_continuous_params.txt'
    f = open(min_file, "r")
    results = f.read()
    results = results.split()
    results = [float(p) for p in results[:-1]]
    x_all = np.append(x_all, results[0])
    y_all = np.append(y_all, results[1])
    z_all = np.append(z_all, results[2])
    print(z_all[-1] - 23.54)
    min_file = '../../Results/lowl/Vary_k0_deltak/lowl_opt_A.txt'
    f = open(min_file, "r")
    results = f.read()
    results = results.split()
    results = [float(p) for p in results[:-1]]
    x_all = np.append(x_all, results[0])
    y_all = np.append(y_all, results[1])
    z_all = np.append(z_all, results[2])
    print(z_all[-1] - 23.54)
    min_file = '../../Results/lowl/Vary_k0_deltak/lowl_opt_B.txt'
    f = open(min_file, "r")
    results = f.read()
    results = results.split()
    results = [float(p) for p in results[:-1]]
    x_all = np.append(x_all, results[0])
    y_all = np.append(y_all, results[1])
    z_all = np.append(z_all, results[2])
    print(z_all[-1] - 23.54)
    min_file = '../../Results/lowl/Vary_k0_deltak/lowl_opt_C.txt'
    f = open(min_file, "r")
    results = f.read()
    results = results.split()
    results = [float(p) for p in results[:-1]]
    x_all = np.append(x_all, results[0])
    y_all = np.append(y_all, results[1])
    z_all = np.append(z_all, results[2])
    print(z_all[-1] - 23.54)
               
    z_all = z_all - 23.54
    
    levels = np.linspace(-10, 10, 21)
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5.25))

    xmin = 0
#    xmax = 0.50
    xmax = 0.60
    ymin = 0.05
#    ymax = 0.235
    ymax = 0.255
    ychange = 0.22
    y_top_change = ychange - 0.005
    y_bottom_change = ychange + 0.005
    
    # Plot lower delta_k region
    x, y, z = remove_points(xmin, xmax, ymin, y_bottom_change, x_all, y_all, z_all)
    triang = get_triang(xmin, xmax, ymin, y_bottom_change, x, y)
    plot = ax2.tricontourf(triang, z, levels=levels, cmap='plasma')
    #    zero_line = ax2.tricontour(x, y, z, levels=[0], colors=['k'], linewidths=0.5)
    zero_line = ax2.tricontour(triang, z, levels=[0], colors=['k'], linewidths=0.5)
    ax2.set_ylim(ymin, ychange)
    
    # Plot upper delta_k region
    x, y, z = remove_points(xmin, xmax, y_top_change, ymax, x_all, y_all, z_all)
    triang = get_triang(xmin, xmax, y_top_change, ymax, x, y)
    plot = ax1.tricontourf(triang, z, levels=levels, cmap='plasma')
    #    zero_line = ax1.tricontour(x, y, z, levels=[0], colors=['k'], linewidths=0.5)
    zero_line = ax1.tricontour(triang, z, levels=[0], colors=['k'], linewidths=0.5)
    ax1.set_ylim(ychange, ymax)
    ax1.set_xlim(xmin, xmax)
    
    # Ticks wanted
    lower_ticks = [0.060, 0.100, 0.140, 0.180]
    lower_tick_labels = ['0.060', '0.100', '0.140', '0.180']
#    upper_ticks = [0.220, 0.225, 0.230, 0.235]
    upper_ticks = [0.220, 0.230, 0.240, 0.250]
    ax1.set_yticks(upper_ticks)
    ax2.set_yticks(lower_ticks)
#    ax2.set_yticklabels(lower_tick_labels)

    cbar = f.colorbar(plot, ax=[ax1,ax2], pad=0.1, format='%.1f', ticks=[-10, -5, 0, 5, 10], anchor=(0.85,0.0))
    
    cbar.ax.plot([0, 1], [0.5, 0.5], 'k')

#    # \ell ticks
#    lower_ticks = [1.00, 1.50, 2.00, 2.50, 3.00]
#    lower_tick_labels = ['1.00', '1.50', '2.00', '2.50', '3.00']

    ax_l_x = ax1.twiny()
    ax_l_y = ax1.twinx()
    convert_ax_to_l_2d(ax1, ax_l_x, ax_l_y)
    ax_l_x.set_xlabel('$\ell_0$')
    #    ax_l_y.set_ylabel(' ')

    ax_l_x = ax2.twiny()
    ax_l_y = ax2.twinx()
#    ax_l_y.set_yticks(lower_ticks)
#    ax_l_y.set_yticklabels(lower_tick_labels)
    ax_l_x.set_xticklabels([])
    convert_ax_to_l_2d(ax2, ax_l_x, ax_l_y)

    ax2.set_xlabel('$k_0 \ / \ 10^{-3} \mathrm{Mpc}^{-1}$')
    #    ax1.set_ylabel('')
    #    ax2.yaxis.set_label_coords(-0.5,-100)
    f.text(0.03, 0.5, '$\Delta k \ / \ 10^{-3} \mathrm{Mpc}^{-1}$', va='center', rotation='vertical')
    f.text(0.80, 0.5, '$\Delta \ell$', va='center', rotation='vertical')
    
    best_idx = np.squeeze(np.where(z == np.amin(z)))
    ax1.plot(x[best_idx], y[best_idx], 'wo', mfc='none')
    
    plt.tight_layout(rect=[0.05,0,0.8,1])
    f.subplots_adjust(hspace=0)
    
    plt.rc('text', usetex=True)
    cbar.set_label(r'$\Delta$lowl')   
    
    plt.savefig('lowl_k_plane_twoscales.pdf')
    plt.clf()


    print(best_idx)
    print(x[best_idx], y[best_idx], z[best_idx])

    return

#contour_plots()
#plot_1d_points()
#plot_1d_bins()
#plot_param_plane()
#plot_2d_two_scales()
plot_lowl_2d()
