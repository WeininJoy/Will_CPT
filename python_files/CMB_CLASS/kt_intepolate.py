import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
from classy import Class
from classy import CosmoComputationError
import sys
import time

nu_spacing = int(sys.argv[1])
print("nu_spacing = ", nu_spacing)
start_time = time.time()

# Parameters
params_file = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/Planck_params/continuous_params_omegak.txt'
params = np.loadtxt(params_file, unpack=True)

lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density 
Neff = 3
N_ncdm = 1
epsilon = 1e-2 # the accuracy of constraint: |Deltak(mt,kt,z_rec)-nu_spacing| < epsilon
A_s, n_s, tau = params[4], params[5], params[6]

################
# Load data of Delta K and create Intepolator
################

folder_path = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/"

mt_list = np.linspace(350, 500, 20)
kt_list = np.linspace(0, 1.8, 20)
z_rec_list = np.linspace(1040, 1100, 20)
DeltaK_arr = np.load(folder_path + 'DeltaK_arr.npy')  # 3D array: DeltaK(mt,kt,z_rec)

# Define integer wave vector k
for i in range(len(mt_list)):
    for j in range(len(kt_list)):
        for k in range(len(z_rec_list)):
            DeltaK_arr[i,j,k] = 1./ np.sqrt(kt_list[j]) * DeltaK_arr[i,j,k]

# create interpolator based on the data
interpolate_Deltak = RegularGridInterpolator((mt_list, kt_list, z_rec_list), DeltaK_arr, bounds_error=False, fill_value=np.nan)

def cosmological_parameters(mt,kt,h): 
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
def calculate_z_rec(mt, kt, omega_b_ratio, h):
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt,kt,h)
    
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
    print(f"z_rec = {z_rec}")
    return z_rec

def find_kt(mt, omega_b_ratio, h):
    # constraint Deltak(mt,kt,z_rec) = nu_spacing
    def constraint_Deltak(kt):
        try: 
            z_rec = calculate_z_rec(mt, kt, omega_b_ratio, h)
            if z_rec_list[0] < z_rec < z_rec_list[-1]:
                return interpolate_Deltak([[mt, kt, z_rec]]) - nu_spacing
            else:
                return np.nan
        except CosmoComputationError:
            return np.nan
    
    # Solve for kt using a root finder
    sol = root_scalar(constraint_Deltak, bracket=[kt_list[0], kt_list[-1]], method='brentq', rtol=1e-2)

    if sol.converged:
        return sol.root
    else:
        return np.nan  # Return NaN if no solution found
    
print("--- %s seconds ---" % (time.time() - start_time))


# Create a new dataset kt(mt, omega_b_ratio, h)
omega_b_ratio_list = np.linspace(0.15, 0.18, 20)
h_list = np.linspace(0.5, 0.75, 20)
kt_solutions = np.array([[[find_kt(mt, omega_b_ratio, h) for h in h_list] for omega_b_ratio in omega_b_ratio_list] for mt in mt_list])
np.save(f'kt_arr_nu_spacing{nu_spacing}.npy', kt_solutions)

# Construct an interpolator for kt(mt, omega_b_ratio, h)
kt_interp = RegularGridInterpolator((mt_list, omega_b_ratio_list, h_list), kt_solutions, bounds_error=False, fill_value=np.nan)

# Test the interpolation function
kt_pred = kt_interp([[400, params[0]/(params[1]+params[0]), params[3]]])

print(f"Interpolated x1({400}, {params[0]/(params[1]+params[0])}, {params[3]}) = {kt_pred}")