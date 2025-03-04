import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy
from scipy import stats
import getdist.plots as gplot
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

mt_list = np.linspace(370, 400, 20)
kt_list = np.linspace(0.4, 1.1, 40)

def get_data(z_param):
    
    # From file data
    x = []
    y = []
    z = []
    
    cts_chisq = 2767.89
    ############
    # input files from cl_files
    ############
    
    # param_list = ['mt', 'kt', 'k0', 'delta_k', 'plik', 'lowl', 'lowE', 'lensing', 'chi_eff_sq','omega_b', 'omega_cdm', 'omega_ncdm', 'h', 'A_s', 'n_s', 'tau_reio', 'ycal', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217', 'ksz_norm', 'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217', 'calib_100T', 'calib_217T', 'success']
    # param_num = param_list.index(z_param)
    # chisq_num = param_list.index('chi_eff_sq')
    # remove_failed = True

    # mypath = './data/cl_files/fix_OmegaR/'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for i in range(len(onlyfiles)):
    #     file = mypath + onlyfiles[i]
    #     f = open(file, "r")
    #     results = f.read()
    #     results = results.split()

    #     if results[4] != 'Invalid':
            
    #         if remove_failed:
    #             if results[-1] == 'True':
    #                 x.append(float(results[1])) # kt
    #                 y.append(float(results[0])) # mt
    #                 if param_num == chisq_num:
    #                     z.append((float(results[param_num]) - cts_chisq))
    #                 else:
    #                     z.append(float(results[param_num]))
    #         else:
    #             x.append(float(results[1]))
    #             y.append(float(results[0]))
    #             if param_num == chisq_num:
    #                 z.append(float(results[param_num]) - cts_chisq)
    #             else:
    #                 z.append(float(results[param_num]))
    
    ############
    # input files from best-fit_H0
    ############

    param_list = ['plik', 'lowl', 'lowE', 'lensing', 'chi_eff_sq', 'h', 'success']
    param_num = param_list.index(z_param)
    chisq_num = param_list.index('chi_eff_sq')

    mypath = "./data/best-fit_H0/"
    for i in range(20):
        for j in range(40):
            input_number = i*100 + j
        
            file_name = f"bestfitH0_{input_number}.txt"
            file = mypath + file_name
            f = open(file, "r")
            results = f.read()
            results = results.split()

            if results[-1] == 'True':
                y.append(mt_list[i]) # mt
                x.append(kt_list[j]) # kt  
                if param_num == chisq_num:
                    z.append((float(results[param_num]) - cts_chisq))
                else:
                    z.append(float(results[param_num]))
    
    return x, y, z

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
    
    # Remove points not in desired range
    # x, y, z = remove_points(xmin, xmax, ymin, ymax, x_all, y_all, z_all)
    x, y, z = x_all, y_all, z_all

    # Scale x and y
    triang = get_triang(xmin, xmax, ymin, ymax, x, y)

    fig, ax = plt.subplots()

    levels = np.linspace(np.amin(z), 1.1*np.amin(z), 21)
    plot = ax.tricontourf(triang, z, levels = levels, cmap='viridis')
    # zero_line = ax.tricontour(x, y, z, levels=[0], colors=['k'], linewidths=0.7)
    print(np.amin(z))
    
    # # cbar = fig.colorbar(plot, ax=ax, pad=0.12, format='%.1f', ticks=[-10, -5, 0, 5, 10])
    cbar = fig.colorbar(plot, ax=ax, pad=0.12, format='%.1f')
    cbar.set_label(r'$\Delta \chi^2_{\mathrm{TT}}$')
    cbar.ax.plot([0, 1], [0.5, 0.5], 'k')
    
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    ax.set_xlabel(r'$\tilde \kappa$')
    ax.set_ylabel(r'$\tilde m$')

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf()
    
    return


def contour_plots():
    # Not zoomed plot
    xmin = 0.4 ; xmax = 1.1
    ymin = 370; ymax = 400
    savename = './figures/chi2_mt_kt_contour_bestH0_wide_test.pdf'
    plot_2d(xmin, xmax, ymin, ymax, savename)
    return 

contour_plots()