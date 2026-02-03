"""
Generate data for allowedK and integerK calculations using cosmological parameters.
"""
import numpy as np
import os
import time
from scipy.optimize import root_scalar
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK
from Higher_Order_Finding_U_Matrices_TimeSeries import compute_U_matrices_timeseries

data_folder = './data/'

nu_spacing = 4
print("nu_spacing = ", nu_spacing)

# Detect number of CPUs
n_processes = 4
print(f"Using {n_processes} parallel processes")

#################
# Parameters
#################
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
N_ncdm = 1
m_ncdm = 0.06
epsilon = 1e-2
nu_spacing4_bestfit = [418.156418,1.472336,0.163104,0.547807,2.030476,0.967708,0.046112]

##################
def cosmological_parameters(mt, kt, h):
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

##################
def calculate_z_rec(params):
    mt, kt, omega_b_ratio, h = params
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt, kt, h)

    CMB_params = {
        'output': 'tCl',
        'h': h,
        'Omega_b': omega_b_ratio*Omega_m,
        'Omega_cdm': (1.-omega_b_ratio) *Omega_m,
        'Omega_k': float(Omega_K),
        'A_s': nu_spacing4_bestfit[4]*1e-9,
        'n_s': nu_spacing4_bestfit[5],
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'tau_reio': nu_spacing4_bestfit[6],
        'lensing': 'no'
    }

    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()

    thermo = cosmo.get_thermodynamics()
    z = thermo['z']
    free_electron_fraction = thermo['x_e']

    z_rec = z[np.argmin(np.abs(free_electron_fraction - 0.1))]
    cosmo.struct_cleanup()
    cosmo.empty()

    return z_rec

def calculate_allowedK(params, z_rec, kvalues, folder_path):
    compute_U_matrices(params, z_rec, kvalues, folder_path, n_processes)
    compute_X_recs(params, z_rec, folder_path)
    compute_allowedK(params, folder_path)
    
def calculate_Umatrices_Xrec(params, z_rec, kvalues, folder_path):
    compute_U_matrices(params, z_rec, kvalues, folder_path, n_processes)
    compute_X_recs(params, z_rec, folder_path)

    
######################################
# Main Execution
######################################
params = nu_spacing4_bestfit[:4] # [mt, kt, omega_b_ratio, h]
discreteK_type = ['integerK', 'allowedK']
folder =  './data/'
z_rec = calculate_z_rec(params)

########### calculate allowedK ##############
kvalues_all_k = np.linspace(1e-5,22,num=200); # data_all_k
kvalues_small_k = np.linspace(1e-5,3,num=100); # data_small_k
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k, folder_path=folder+'data_all_k/')
# calculate_allowedK(params, z_rec, kvalues=kvalues_small_k, folder_path=folder+'data_small_k/')
print("AllowedK calculation completed.")

########### generate discreteK data ##############

## kvalues based on allowedK
small_allowedK = np.load(folder + 'data_small_k/allowedK.npy')
small_allowedK
allowedK_path = folder + 'data_all_k/allowedK.npy'
kvalues_allowedK = np.load(allowedK_path); # only caluclate for the allowed K values
kvalues_allowedK = kvalues_allowedK[kvalues_allowedK > small_allowedK[-1] + 0.5* np.diff(kvalues_allowedK).mean()]  # remove k values that are already in small_allowedK
kvalues_allowedK = np.concatenate((small_allowedK[-3:], kvalues_allowedK))  # add a few small K values for better resolution at low k
print('kvalues_allowedK:', kvalues_allowedK)

## kvalues based on allowedK_integer
small_allowedK_integer = np.load(folder + 'data_small_k/allowedK_integer.npy')
allowedK_integer_path = folder + 'data_all_k/allowedK_integer.npy'
allowedK_integer = np.load(allowedK_integer_path);
allowedK_integer = allowedK_integer[allowedK_integer > small_allowedK_integer[-1] + 0.5* np.diff(allowedK_integer).mean()]  # remove k values that are already in small_allowedK
allowedK_integer = np.concatenate((small_allowedK_integer[-3:], allowedK_integer))  # add a few small K values for better resolution at low k
n_ignore = 8
high_k = np.array(allowedK_integer[n_ignore:])
indices = np.arange(len(high_k))
slope, intercept = np.polyfit(indices, high_k, 1)
ideal_spacing = np.round(slope)
if ideal_spacing != nu_spacing: print("Warning: ideal spacing does not match nu_spacing!")
ideal_intercept = np.mean(high_k - ideal_spacing * indices)
ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)
small_integer_k_sequence = [ideal_k_sequence[0]-nu_spacing*i for i in range(1, n_ignore+1)][::-1]
whole_integer_k_sequence = np.concatenate((small_integer_k_sequence, ideal_k_sequence))
mt, kt, omega_b_ratio, h = params
s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt, kt, h) 
H0 = 1/np.sqrt(3*Omega_lambda); #we are working in units of Lambda=c=1
a0=1; K=-Omega_K * a0**2 * H0**2
kvalues_integerK = [k * np.sqrt(K) for k in whole_integer_k_sequence]
print('kvalues_integerK:', kvalues_integerK)

# calculate U matrices and Xrec for both allowedK and integerK
calculate_Umatrices_Xrec(params, z_rec, kvalues_allowedK, folder_path = folder + 'data_allowedK/')
calculate_Umatrices_Xrec(params, z_rec, kvalues_integerK, folder_path = folder + 'data_integerK/')
print("U matrices and Xrec calculation completed.")


########### generate TimeSeries U matrices ##############
compute_U_matrices_timeseries(params, z_rec, kvalues_allowedK, folder_path=folder + 'data_allowedK_timeseries/')
compute_U_matrices_timeseries(params, z_rec, kvalues_integerK, folder_path=folder + 'data_integerK_timeseries/')
print("TimeSeries U matrices calculation completed.")
print("All data generation completed.")