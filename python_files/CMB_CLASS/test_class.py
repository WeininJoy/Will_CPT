import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from classy import Class
import time
start_time = time.time()

Neff = 3
N_ncdm = 1
params_file = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_Bartlett/data/Planck_params/continuous_params_omegak.txt'
params = np.loadtxt(params_file, unpack=True)
nu_spacing = 6
Omega_k = -0.04

def run_class(resol_i, params):

    LambdaCDM = Class()
    l_max_scalars = 1000
    # pass input parameters
    LambdaCDM.set({'omega_b':params[0], 
                    'omega_cdm':params[1], 
                    'omega_ncdm':params[2]*1e-4, 
                    'Omega_k': Omega_k,
                    'nu_spacing': nu_spacing,
                    'h':params[3], 
                    'A_s':params[4]*1e-9, 
                    'n_s':params[5], 
                    'tau_reio':params[6], 
                    'N_ur':Neff, 
                    'N_ncdm':N_ncdm})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk',
                    'lensing':'yes',
                    'P_k_max_1/Mpc':3.0,
                    'l_max_scalars':l_max_scalars})

    LambdaCDM.set({'l_linstep': l_linstep_list[resol_i],
                'l_logstep': l_logstep_list[resol_i],
                'q_linstep': q_linstep_list[resol_i],
                'q_logstep_spline': q_logstep_spline_list[resol_i],
                'l_switch_limber' : l_switch_limber_list[resol_i],
                'l_switch_limber_for_nc_local_over_z': l_switch_limber_for_nc_local_over_z_list[resol_i],
                'l_switch_limber_for_nc_los_over_z': l_switch_limber_for_nc_los_over_z_list[resol_i],
                'write warnings' : 'yes'})
    
    # run class
    LambdaCDM.compute()

    # get all C_l output
    cls = LambdaCDM.lensed_cl(l_max_scalars)

    # Don't remove these lines - otherwise uses up too much memory
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()

    return cls


def compare_two_CMB(ell1, cosmo_tt1, ell2, cosmo_tt2): # cls2 must have higher resolution than cls1

    # Interpolate CMB2 to match ell1
    interp_tt2 = interp1d(ell2, cosmo_tt2, kind='linear', fill_value="extrapolate")
    tt2_interp = interp_tt2(ell1)  # Get interpolated tt2 values at ell1 points

    # Compute Mean Squared Error (MSE)
    mse = np.mean((cosmo_tt1 - tt2_interp) ** 2 / cosmo_tt1 ** 2)
    print(f"Mean Squared Error: {mse:.6f}")

    return mse

def smoothness_test(ell, cosmo_tt):
    # Derivative method
    dy_dx = np.gradient(cosmo_tt, ell)
    d2y_dx2 = np.gradient(dy_dx, ell)
    smoothness_score = np.std(d2y_dx2)
    return smoothness_score

# ####################
# # Whole CMB power spectrum
# ####################

def plot_CMB(params):
    for resol_i in range(2, 4):
        start_time = time.time()  # Start timer
        cls = run_class(resol_i, params)
        # print out the time taken for each iteration
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time  # Compute elapsed time
        print(f"Iteration {resol_i+1} took {elapsed_time:.4f} seconds")

        # Plot the CMB power spectrum
        ell = cls['ell'][2:]
        clTT = cls['tt'][2:]
        np.save(f'ell_nu{nu_spacing}_reso{resol_i}.npy', ell)
        np.save(f'clTT_nu{nu_spacing}_reso{resol_i}.npy', clTT)
        cosmo_tt = ell*(ell+1)*clTT * (1e6 * 2.7255)**2 / (2*np.pi)
        plt.plot(ell,cosmo_tt,color = color_list[resol_i] , zorder=4, label=f"resolution={resol_i}")

    plt.xscale('log')
    plt.xlim(1.9,1000)
    # plt.ylim(0,2500)
    plt.ylabel(r'$\mathcal{D}_\ell^{TT}\: [\mu \mathrm{K}^2]$')
    plt.xlabel(r'$\ell$')
    # plt.xticks([2,10,30])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'CMB_OmegaK{Omega_k}_nu{nu_spacing}_resolution_lqstep.pdf')

color_list = ['navy', 'blue', 'cornflowerblue']
l_linstep_list = [40, 10, 2]  
l_logstep_list = [1.26, 1.06, 1.007] 
q_linstep_list = [0.45, 0.2, 0.19] 
q_logstep_spline_list = [170, 10, 2.0] 
l_switch_limber_list = [1000, 2500, 2500]
l_switch_limber_for_nc_local_over_z_list = [1000, 2500, 2500]
l_switch_limber_for_nc_los_over_z_list = [1000, 2500, 2500]

plot_CMB(params)


# ####################
# # Compare the CMB power spectra with different resolutions
# ####################
# ell_list = []
# clTT_list = []
# cosmo_tt_list = []
# for resol_i in range(len(l_linstep_list)):
#     ell_list.append(np.load(f"ell_reso={resol_i}.npy"))
#     clTT_list.append(np.load(f"clTT_reso={resol_i}.npy"))
#     cosmo_tt_list.append(ell_list[resol_i]*(ell_list[resol_i]+1)*clTT_list[resol_i] * (1e6 * 2.7255)**2 / (2*np.pi))

# diff_array = np.zeros((len(l_linstep_list), len(l_linstep_list)))
# for i in range(len(l_linstep_list)):
#     for j in range(i+1, len(l_linstep_list)):
#         diff_array[i,j] = compare_two_CMB(ell_list[i], cosmo_tt_list[i], ell_list[j], cosmo_tt_list[j])
#         diff_array[j,i] = diff_array[i,j]
# print(diff_array)

####################
# check smoothness of the CMB power spectrum
####################

# ell = np.load(f"ell_reso={0}.npy")
# clTT = np.load(f"clTT_reso={0}.npy")
# cosmo_tt = ell*(ell+1)*clTT * (1e6 * 2.7255)**2 / (2*np.pi)
# variation = smoothness_test(ell, cosmo_tt)
# print(f"Resolution {0} has variation {variation:.6f}")

# ell = np.load(f'ell_nu{nu_spacing}_reso{0}.npy')
# clTT = np.load(f'clTT_nu{nu_spacing}_reso{0}.npy')
# cosmo_tt = ell*(ell+1)*clTT * (1e6 * 2.7255)**2 / (2*np.pi)
# variation = smoothness_test(ell, cosmo_tt)
# print(f"Resolution {0} has variation {variation:.6f}")