import clik
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from scipy.interpolate import Rbf
from astropy import units as u
from astropy.constants import c
import time
import classy
from classy import Class
from classy import CosmoComputationError
print(classy.__file__)

start_time = time.time()

data = os.path.join(os.getcwd(),'/home/wnd22/rds/hpc-work/Polychord_solveb/clik_installs/data/planck_2018/baseline/plc_3.0') 

# Parameters
params_file = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/Planck_params/continuous_params_omegak.txt'
params = np.loadtxt(params_file, unpack=True)

lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density 
Neff = 3
N_ncdm = 1

##################
# Set the initial guess
##################
ini_params = [400, 1.0, params[0]/(params[1]+params[0]), 0.63, params[4], params[5], params[6]] # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$, $A_s$, $n_s$, $\tau$

################
# Load data of Delta K for Intepolate
################

# folder_path = '/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/Curve_universes_a0/data/fixOmegaR_SmallRegion/'

# mt_list = np.linspace(435, 450, 20)
# kt_list = np.linspace(0.9, 1.0, 20)
# Omega_gamma_h2 = 2.47e-5 # photon density 
# Neff = 3
# h = 0.63
# Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

# # Calculate deltaK from data
# mt_data = []
# kt_data = [] 
# deltaK_list = []
# for i in range(len(mt_list)):
#     for j in range(len(kt_list)):
#         input_number = i*100 + j
#         mt_data.append(mt_list[i])
#         kt_data.append(kt_list[j])
#         try:
#             allowedK = np.load(folder_path + f'allowedK_{input_number}.npy')
#             deltaK = [allowedK[n+1] - allowedK[n] for n in range(len(allowedK)-1)]
#             deltaKavg = sum(deltaK[len(deltaK)//2:-2]) / len(deltaK[len(deltaK)//2:-2])
#             deltaK_list.append(deltaKavg)
#         except FileNotFoundError:
#             print(f"File {input_number} is not found.")
#             deltaK_list.append(deltaK_list[-1]) # fill the value 

def cosmological_parameters(find_params): 
    mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(Omega_r, rt, mt, kt)
    s0 = 1/a0
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return s0, Omega_lambda, Omega_m, Omega_K


# Runs the CLASS code
def run_class(find_params, nu_spacing=1):

    mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(find_params)
    H0 = h*100 * u.km/u.s/u.Mpc
    Lambda = Omega_lambda * 3 * H0**2 / c**2  # unit: km2 / (Mpc2 m2)
    sqrtlambda = np.sqrt(Lambda.value * 1.e6)

    ################
    ## calculate allowed K directly
    # findind_U_matrices(mt, kt, h)
    # findind_Xrecs(mt, kt, h)
    # solving_Vrinf(mt, kt, h)
    # allowedK = np.load(f'/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS/calculate_allowedK/allowedK.npy')
    # allowedK_phys = [K * s0 * sqrtlambda for K in allowedK]
    # np.savetxt(allowed_k_file, np.array(allowedK_phys))

    ################
    ## get Delta K values by extropolation (based on the data)
    # extropolate_Deltak = Rbf(mt_data, kt_data, deltaK_list, function='linear') # Experiment with different basis functions
    # DeltaK = extropolate_Deltak(mt, kt)
    # DeltaK_phys = DeltaK * s0 * sqrtlambda
    # kmax = 20
    # k0 = 1* DeltaK_phys
    # nmax = np.ceil((kmax - k0) / DeltaK_phys)
    # k = np.arange(nmax + 1) * DeltaK_phys + k0
    # np.savetxt(allowed_k_file, np.array(k))

    ################
    # Contruct CMB power spectrum by "Class"
    # create instance of the class " Class "
    LambdaCDM = Class()

    # # pass input parameters
    LambdaCDM.set({'omega_b': omega_b_ratio*Omega_m*h**2, # omega_b = omega_b/(omega_cdm + omega_b) *omega_m
                    'omega_cdm': (1.-omega_b_ratio) *Omega_m*h**2,
                    'omega_ncdm': params[2]*1e-4, 
                    'Omega_k': Omega_K,
                    'nu_spacing': nu_spacing,
                    'h':h, 
                    'A_s': A_s*1e-9, 
                    'n_s': n_s, 
                    'tau_reio': tau, 
                    'N_ur': Neff,
                    'N_ncdm':N_ncdm})
    LambdaCDM.set({ 'output':'tCl,pCl,lCl,mPk',
                    'lensing':'yes',
                    'P_k_max_1/Mpc':3.0,
                    'l_max_scalars':2508})

    # run class
    LambdaCDM.compute()

    # get all C_l output
    z_list = LambdaCDM.get_thermodynamics()['z']
    xe_list = LambdaCDM.get_thermodynamics()['x_e']
    z_rec = z_list[np.argmin(np.abs(xe_list - 0.1))]

    # Don't remove these lines - otherwise uses up too much memory
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()

    return z_rec


if __name__ == '__main__':

    #####################
    # Calculate the whole 4D z_rec array
    #####################

    # param_names = [r"\tilde m", r"\tilde \kappa", r"\Omega_b/\Omega_m", r"h"]
    # param_ranges = [[350, 500], [0,1.8], [0.15, 0.17+3*0.02/9], [0.5, 0.75]]  # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$
    # grid_num = 10
    # mt_list = np.linspace(param_ranges[0][0],param_ranges[0][1],grid_num)
    # kt_list = np.linspace(param_ranges[1][0],param_ranges[1][1],grid_num)
    # omegab_list = np.linspace(param_ranges[2][0],param_ranges[2][1],grid_num+3)
    # h_list = np.linspace(param_ranges[3][0],param_ranges[3][1],grid_num)

    # zrec_arr = np.zeros((grid_num,grid_num,3,grid_num))
    # for i in range(grid_num): 
    #     for j in range(grid_num):
    #         for k in range(3):
    #             for n in range(grid_num):
    #                 params = [mt_list[i], kt_list[j], omegab_list[k], h_list[n], ini_params[4], ini_params[5], ini_params[6]]
    #                 z_rec = run_class(params)
    #                 zrec_arr[i][j][k][n] = z_rec

    # np.save('zrec_params_extend', zrec_arr)

    ## load the data and find the extremum 
    # zrec_arr_merged = np.load('zrec_arr_merged.npy')

    # # Step 1: Flatten the array
    # flattened = zrec_arr.ravel()

    # # Step 2: Get the indices of the top 10 values
    # top_10_indices_flat = np.argpartition(flattened, -10)[-10:]
    # top_10_indices_flat = top_10_indices_flat[np.argsort(flattened[top_10_indices_flat])[::-1]]  # Sort in descending order

    # # Step 3: Get the values and convert indices back to 4D
    # top_10_values = flattened[top_10_indices_flat]
    # top_10_indices_4d = np.unravel_index(top_10_indices_flat, zrec_arr.shape)

    # # Output the results
    # print("Top 10 values:", top_10_values)
    # print("Top 10 indices (4D):", list(zip(*top_10_indices_4d)))

    # # Step 2: Get the indices of the bottom 10 values
    # bottom_10_indices_flat = np.argpartition(flattened, 10)[:10]
    # bottom_10_indices_flat = bottom_10_indices_flat[np.argsort(flattened[bottom_10_indices_flat])]  # Sort in ascending order

    # # Step 3: Get the values and convert indices back to 4D
    # bottom_10_values = flattened[bottom_10_indices_flat]
    # bottom_10_indices_4d = np.unravel_index(bottom_10_indices_flat, zrec_arr.shape)

    # # Output the results
    # print("Bottom 10 values:", bottom_10_values)
    # # print("Bottom 10 indices (4D):", list(zip(*bottom_10_indices_4d)))


    ####################
    # Go through parameters one-by-one
    ####################

    # zrec_params_list = []
    # for key in parameter_dic.keys():
    #     param_range = parameter_dic[key]
    #     param_list = np.linspace(param_range[0],param_range[1],10)
    #     z_rec_list = []
    #     for n in range(len(param_list)):
    #         ini_params[key] = param_list[n] 
    #         z_rec = run_class(ini_params)
    #         z_rec_list.append(z_rec)

    #     plt.plot(np.linspace(0,1,len(param_list)), z_rec_list, ".", label=param_names[key])
    #     zrec_params_list.append(z_rec_list)

    # np.save('zrec_params', zrec_params_list)

    # zrec_params_list = np.load('zrec_params.npy')
    # for i in range(4):
    #     z_rec_list = zrec_params_list[i]
    #     plt.plot(np.linspace(0,1,10), z_rec_list, ".", label=param_names[i])
    # plt.legend()
    # plt.ylim([1000, 1150])
    # plt.savefig("zrec_params.pdf")