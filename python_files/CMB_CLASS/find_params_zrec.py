##########################
# Goal of this file: find the best-fit 7 parameters 
# mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$, $A_s$, $n_s$, $\tau$, 
# by minimizing chi^2 (difference between prediction and Planck's data)
##########################
import clik
import numpy as np
import sys
import os
import re
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from astropy import units as u
from astropy.constants import c
import pickle
import time
import classy
from classy import Class
from classy import CosmoComputationError
print(classy.__file__)
nu_spacing = int(sys.argv[1])
print("nu_spacing = ", nu_spacing)

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
epsilon = 1e-2 # the accuracy of CMB power spectrum

# resolution lists for CMB power spectrum
l_linstep_list = [40, 10, 2]  
l_logstep_list = [1.26, 1.06, 1.007] 
q_linstep_list = [0.45, 0.2, 0.19] 
q_logstep_spline_list = [170, 10, 2.0] 
l_switch_limber_list = [1000, 2500, 2500]
l_switch_limber_for_nc_local_over_z_list = [1000, 2500, 2500]
l_switch_limber_for_nc_los_over_z_list = [1000, 2500, 2500]

##################
# Set the initial guess
##################
def find_max_tried_num():
    # Define the directory containing the files
    directory = f'/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/best-fit_params_zrec/nu_spacing{nu_spacing}/'
    pattern = re.compile(r"try_params_(\d+)\.txt")
    max_num = 0
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # Extract the number and convert to integer
            max_num = max(max_num, num)
    return max_num

import os.path

bestfit_folder_root = f'/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/best-fit_params_zrec/nu_spacing{nu_spacing}/'
if not os.path.exists(bestfit_folder_root):
    os.makedirs(bestfit_folder_root)
bestfit_file_root = bestfit_folder_root + 'bestfit_params.txt'
if os.path.isfile(bestfit_file_root):
    print("Set initial guess from revious run.")
    try_num = find_max_tried_num() 
    bestfit_params = np.loadtxt(bestfit_file_root, unpack=True)
    start_params = bestfit_params[4:] # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$, $A_s$, $n_s$, $\tau$
    print("start_parameters = ", start_params)
else:
    print("Set initial guess from Planck best-fit data.")
    try_num = 0
    start_params = [400, 1.0, params[0]/(params[1]+params[0]), params[3], params[4], params[5], params[6]] # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$, $A_s$, $n_s$, $\tau$

################
# Load data of Delta K and create Intepolator
################

folder_path = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/"

mt_list = np.linspace(350, 500, 20)
kt_list = np.linspace(0, 1.8, 20)
z_rec_list = np.linspace(1040, 1100, 20)
DeltaK_arr = np.load(folder_path + 'DeltaK_arr.npy')  # 3D array: DeltaK(mt,kt,z_rec)

# create interpolator based on the data
interpolate_Deltak = RegularGridInterpolator((mt_list, kt_list, z_rec_list), DeltaK_arr, bounds_error=False, fill_value=np.nan)

################
# Load data of z_rec and create Intepolator
################
param_ranges = [[350, 500], [0,1.8], [0.15, 0.17+3*0.02/9], [0.5, 0.75]]  # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$
grid_num = 10
mt_list = np.linspace(param_ranges[0][0],param_ranges[0][1],grid_num)
kt_list = np.linspace(param_ranges[1][0],param_ranges[1][1],grid_num)
omegab_list = np.linspace(param_ranges[2][0],param_ranges[2][1],grid_num+3)
h_list = np.linspace(param_ranges[3][0],param_ranges[3][1],grid_num)
zrec_arr = np.load('/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/zrec_params/zrec_params_merged.npy')

# create interpolator based on the data
interpolate_zrec = RegularGridInterpolator((mt_list, kt_list, omegab_list, h_list), zrec_arr, bounds_error=False)

# outliers = (zrec_arr < z_rec_list[0]) | (zrec_arr > z_rec_list[-1]) # Define outlier (z_rec out of range)
# zrec_arr[outliers] = np.nan # Replace outliers with NaN
# zrec_arr[np.isnan(zrec_arr)] = np.nanmean(zrec_arr) # Replace NaN values with the mean of valid data

# # Create the interpolator based on the data without outliers
# interpolate_zrec_physical = RegularGridInterpolator((mt_list, kt_list, omegab_list, h_list), zrec_arr, bounds_error=False, fill_value=np.nan)

################
# Define Planck Likelihood
################

class PlanckLikelihood(object):
    """Baseline Planck Likelihood"""
    # Use TTTEEE, without lensing
    
    def __init__(self):
        self.plik = clik.clik(os.path.join(data, "hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"))
        self.lowl = clik.clik(os.path.join(data, "low_l/commander/commander_dx12_v3_2_29.clik"))
        self.lowE = clik.clik(os.path.join(data, "low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"))
    
    def __call__(self, cls, nuis):
        lkl = []
        for like in [self.plik, self.lowl, self.lowE]:
            #        for like in [self.plik]:
            lmaxes = like.get_lmax()
            dat = []
            order = ['tt','ee','bb','te','tb','eb']
            
            # print(order,len(lmaxes),len(order))
            for spec, lmax in zip(order, lmaxes):
                if lmax>-1:
                    if spec == 'pp':
                        dat += list(cls[spec][:lmax+1])
                    else:
                        dat += list(cls[spec][:lmax+1]* (1e6 * 2.7255)**2 )
        
            for param in like.get_extra_parameter_names():
                dat.append(nuis[param])
    
            lkl.append(like(dat))
    
        return np.array(lkl).flatten()

lkl = PlanckLikelihood()

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


# calculate recombination redshift
def calculate_z_rec(find_params):
    mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(find_params)
    
    ################
    # Contruct CMB power spectrum by "Class"
    LambdaCDM = Class()

    # pass input parameters
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

# Runs the CLASS code
def run_class(resol_i, find_params):

    mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(find_params)

    ################
    # Contruct CMB power spectrum by "Class"
    # create instance of the class " Class "
    LambdaCDM = Class() 

    # # pass input parameters
    LambdaCDM.set({'omega_b': omega_b_ratio*Omega_m*h**2, # omega_b = omega_b/(omega_cdm + omega_b) *omega_m
                   'omega_cdm': (1-omega_b_ratio) *Omega_m*h**2, 
                    'omega_ncdm': params[2]*1e-4, 
                    'Omega_k': Omega_K,
                    'nu_spacing': nu_spacing,
                    'h':h, 
                    'A_s': A_s*1e-9, 
                    'n_s': n_s, 
                    'tau_reio': tau, 
                    'N_ur': Neff,
                    'N_ncdm':N_ncdm})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk',
                    'lensing':'yes',
                    'P_k_max_1/Mpc':3.0,
                    'l_max_scalars':2508})
    LambdaCDM.set({ 'l_linstep': l_linstep_list[resol_i],
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
    cls = LambdaCDM.lensed_cl(2508)

    # Don't remove these lines - otherwise uses up too much memory
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()

    return cls 

def CMB_smoothness(cls):
    ell = cls['ell'][2:]
    clTT = cls['tt'][2:]
    cosmo_tt = ell*(ell+1)*clTT * (1e6 * 2.7255)**2 / (2*np.pi)
    
    # Derivative method
    dy_dx = np.gradient(cosmo_tt, ell)
    d2y_dx2 = np.gradient(dy_dx, ell)
    smoothness_score = np.std(d2y_dx2)
    return smoothness_score

# Global variable for best chi^2
best_chisq = 2e+30 

# Computes TT spectrum and returns chi^2 for given set of parameters params in form [mt, kt, omega_b_ratio, h, A_s, n_s, tau] using linear quantisation
def run_TT(find_params, bestfit_file_root):
            
    try:
        mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
        z_rec = interpolate_zrec([mt, kt, omega_b_ratio, h]) # interpolated recombination redshift
        if z_rec < z_rec_list[0] or z_rec_list[-1] < z_rec: # check whether the recombination redshift is physical or not (1040< z_rec < 1100)
            z_rec = calculate_z_rec(find_params) # if the interpolated z_rec is non-physical, calculate actual one.

        if z_rec_list[0] < z_rec < z_rec_list[-1]: # check whether the recombination redshift is physical (1040< z_rec < 1100)
            # Find corresponding spectra
            resol_i = 0
            cls = run_class(resol_i, find_params)
            # calculate smoothness score. If the smoothness score is too high, increase the resolution
            smoothness_score = CMB_smoothness(cls)
            print("smoothness_score=", smoothness_score)
            if 1 < smoothness_score < 10:
                resol_i = 1
                cls = run_class(resol_i, find_params)
            elif 10 < smoothness_score:
                resol_i = 2
                cls = run_class(resol_i, find_params)
            
            if np.isnan([val for val in cls.values()]).any(): # if there is any Nan in cls -> raise CosmoComputationError
                raise CosmoComputationError
            else:
                nuisance_params = params[7:]

                nuis={
                    'ycal':nuisance_params[0],
                    'A_cib_217':nuisance_params[1],
                    'xi_sz_cib':nuisance_params[2],
                    'A_sz':nuisance_params[3],
                    'ps_A_100_100':nuisance_params[4],
                    'ps_A_143_143':nuisance_params[5],
                    'ps_A_143_217':nuisance_params[6],
                    'ps_A_217_217':nuisance_params[7],
                    'ksz_norm':nuisance_params[8],
                    'gal545_A_100':nuisance_params[9],
                    'gal545_A_143':nuisance_params[10],
                    'gal545_A_143_217':nuisance_params[11],
                    'gal545_A_217':nuisance_params[12],
                    'galf_TE_A_100':nuisance_params[13],
                    'galf_TE_A_100_143':nuisance_params[14],
                    'galf_TE_A_100_217':nuisance_params[15],
                    'galf_TE_A_143':nuisance_params[16],
                    'galf_TE_A_143_217':nuisance_params[17],
                    'galf_TE_A_217':nuisance_params[18],
                    'calib_100T':nuisance_params[19],
                    'calib_217T':nuisance_params[20],
                    'cib_index':-1.3, #no range given in table so assume fixed
                    #-------------------------------------------------------------------
                    # These are all set to 1, so assume that these are fixed -----------
                    'A_cnoise_e2e_100_100_EE':1.,
                    'A_cnoise_e2e_143_143_EE':1.,
                    'A_cnoise_e2e_217_217_EE':1.,
                    'A_sbpx_100_100_TT':1.,
                    'A_sbpx_143_143_TT':1.,
                    'A_sbpx_143_217_TT':1.,
                    'A_sbpx_217_217_TT':1.,
                    'A_sbpx_100_100_EE':1.,
                    'A_sbpx_100_143_EE':1.,
                    'A_sbpx_100_217_EE':1.,
                    'A_sbpx_143_143_EE':1.,
                    'A_sbpx_143_217_EE':1.,
                    'A_sbpx_217_217_EE':1.,
                    'A_pol':1,
                    'A_planck':1.,
                    #-------------------------------------------------------------------
                    # These are fixed from Planck 2018 Likelihood Paper, Table 16 ------
                    'galf_EE_A_100':0.055,
                    'galf_EE_A_100_143':0.040,
                    'galf_EE_A_100_217':0.094,
                    'galf_EE_A_143':0.086,
                    'galf_EE_A_143_217':0.21,
                    'galf_EE_A_217':0.70,
                    'calib_100P':1.021,
                    'calib_143P':0.966,
                    'calib_217P':1.04,
                    #-------------------------------------------------------------------
                    # These are fixed from Planck 2018 Likelihood Paper, pg 39 ---------
                    'galf_EE_index':-2.4,
                    'galf_TE_index':-2.4,
                #-------------------------------------------------------------------
                }

                plik, lowl, lowE = -2 * lkl(cls, nuis)
                chi_eff_sq = plik + lowl + lowE
                print("mt, kt, omega_b_ratio, h, A_s, n_s, tau=", find_params)
                print("chi_eff_sq=", chi_eff_sq)

                # Write the data into files
                global try_num
                print("try_num=", try_num)
                filename = bestfit_folder_root + f'try_params_{try_num}.txt'
                with open(filename, 'w') as f:
                    print(plik, lowl, lowE, chi_eff_sq, *find_params, file=f)
                try_num += 1
                print("--- %s seconds ---" % (time.time() - start_time))

        else: 
            print('zrec is non-physical')
            plik = 2e+30
            lowl = 2e+30
            lowE = 2e+30
            chi_eff_sq = 2e+30

    except CosmoComputationError:
        print('CosmoComputationError')
        plik = 2e+30
        lowl = 2e+30
        lowE = 2e+30
        chi_eff_sq = 2e+30
    except clik.lkl.CError:
        print('CError')
        plik = 2e+30
        lowl = 2e+30
        lowE = 2e+30
        chi_eff_sq = 2e+30

    global best_chisq

    if chi_eff_sq < best_chisq:
        best_chisq = chi_eff_sq
        with open(bestfit_file_root, 'w') as f:
            print(plik, lowl, lowE, chi_eff_sq, *find_params, file=f)

    return plik, lowl, lowE, chi_eff_sq

# constraint Deltak(mt,kt,z_rec) = nu_spacing
def constraint_Deltak(find_params):
    mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
    z_rec = interpolate_zrec([[mt, kt, omega_b_ratio, h]]) # interpolated recombination redshift
    if z_rec < z_rec_list[0] or z_rec_list[-1] < z_rec: # check whether the recombination redshift is physical or not (1040< z_rec < 1100)
        z_rec = calculate_z_rec(find_params) # if the interpolated z_rec is non-physical, calculate actual recombination redshift
    
    if z_rec_list[0] < z_rec < z_rec_list[-1]: # check whether the recombination redshift is physical (1040< z_rec < 1100)
        DeltaK = interpolate_Deltak([[mt, kt, float(z_rec)]])
        DeltaK_int = 1./ np.sqrt(kt) * DeltaK
        print("DeltaK_int=", DeltaK_int)
        return DeltaK_int
    else: 
        return 1e+30


def get_data(bestfit_file_root):

    def to_optimise(find_params):
        plik, lowl, lowE, chi_eff_sq = run_TT(find_params, bestfit_file_root)
        return chi_eff_sq

    from scipy.optimize import NonlinearConstraint

    nlc = NonlinearConstraint(constraint_Deltak, nu_spacing, nu_spacing, keep_feasible=True)
    
    # Define bounds for each variable
    bounds = [
        (mt_list[0], mt_list[-1]),  # mt
        (kt_list[0], kt_list[-1]),  # kt
        (omegab_list[0], omegab_list[-1]),  # omega_b_ratio
        (h_list[0], h_list[-1]),  # h
        (2.0, 2.2),  # A_s
        (0.95, 0.98),  # n_s
        (0.0386, 0.065)  # tau
    ]

    # Perform optimization using COBYLA method 
    res = minimize(to_optimise, start_params, method='trust-constr', bounds=bounds, constraints=nlc, options={'gtol': 1e-8, 'xtol': 1e-8})
    plik, lowl, lowE, chi_eff_sq = run_TT(res.x, bestfit_file_root)

    print(plik, lowl, lowE, chi_eff_sq, res.success)
    print(res.x)

    with open(bestfit_file_root, 'w') as f:
        print(plik, lowl, lowE, chi_eff_sq, *res.x, file=f)


if __name__ == '__main__':

    get_data(bestfit_file_root)
    