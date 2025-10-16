import numpy as np
import sys
import os
import re
from scipy.optimize import root_scalar
from scipy.optimize import minimize
import time
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices  # Fixed typo and using correct parallel version
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK
from multiprocessing import cpu_count

nu_spacing = 4
print("nu_spacing = ", nu_spacing)

# Detect number of CPUs (HPC or local)
# n_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
n_processes = 4
print(f"Using {n_processes} parallel processes")

# Print HPC info if available
if 'SLURM_JOB_ID' in os.environ:
    print(f"Running on HPC - Job ID: {os.environ.get('SLURM_JOB_ID')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}")

start_time = time.time()
def find_max_tried_num():
    # Define the directory containing the files
    pattern = re.compile(r"try_params_(\d+)\.txt")
    max_num = 0
    for filename in os.listdir('./data/try/'):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # Extract the number and convert to integer
            max_num = max(max_num, num)
    return max_num

global try_num
try_num = find_max_tried_num() + 1

#################
# Parameters
#################
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density V
Neff = 3.046
N_ncdm = 1  # number of massive neutrino species
m_ncdm = 0.06  # mass of massive neutrino species in e
epsilon = 1e-2 # the accuracy of kt rootfinding
nu_spacing4_bestfit = [401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583, 1.9375648884116028, 0.9787493821596979, 0.019760560255556746] # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$, $A_s$, $n_s$, $\tau$ from nu_spacing=4 best-fit data
kt_list = np.linspace(0, 1.8,10)
z_rec_list = np.linspace(1040, 1100, 20)

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
# calculate recombination redshift
def calculate_z_rec(params):
    mt, kt, omega_b_ratio, h = params
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt, kt, h)
    
    CMB_params = {
        'output': 'tCl',  # Request only thermodynamics outputs
        'h': h,
        'Omega_b': omega_b_ratio*Omega_m,
        'Omega_cdm': (1.-omega_b_ratio) *Omega_m,
        'Omega_k': float(Omega_K),
        'A_s': nu_spacing4_bestfit[4]*1e-9, 
        'n_s': nu_spacing4_bestfit[5],
        'N_ncdm': N_ncdm,      # number of massive neutrino
        'm_ncdm': m_ncdm,      # mass of massive neutrino species in eV
        'N_ur': Neff - N_ncdm, # Effective number of MASSLESS neutrino species, N_eff = N_ncdm + N_ur
        'tau_reio': nu_spacing4_bestfit[6],
        'lensing': 'no' #Turn off lensing.
    }

    # Initialize and compute CLASS
    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()

    # Get thermodynamics data
    thermo = cosmo.get_thermodynamics()
    z = thermo['z']
    free_electron_fraction = thermo['x_e']

    # find z_recombination when xe = 0.1
    z_rec = z[np.argmin(np.abs(free_electron_fraction - 0.1))]
    cosmo.struct_cleanup()
    cosmo.empty()

    return z_rec

def calculate_allowedK(params, folder_path):
    try:
        z_rec = calculate_z_rec(params)
        compute_U_matrices(params, z_rec, folder_path, nu_spacing, n_processes)  # Added n_processes
        compute_X_recs(params, z_rec, folder_path, nu_spacing)
        compute_allowedK(folder_path)
    except CosmoComputationError as e:
        print(f"Computation error for params {params}: {e}")

def calculate_integer_loss(params, folder_path):
    calculate_allowedK(params, folder_path)
    allowedK = np.load(folder_path+'allowedK.npy').tolist()
    
    if not allowedK or len(allowedK) < 9:
        return float('inf')
    
    allowedK = allowedK[8:]  # Ignore the first few values
    allowedK = np.array(allowedK)
    ideal_start = np.round(allowedK[0])
    ideal_series = ideal_start + 4 * np.arange(len(allowedK))
    loss = np.mean((allowedK - ideal_series)**2)
    return loss

def log_posterior_for_gpry(params):
    """
    Returns a single number proportional to the log-posterior.
    GPry will build a surrogate of this function.
    """
    try: 
        global try_num
        folder_path = './data/try/try_'+str(try_num)+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        loss_integer = calculate_integer_loss(params, folder_path) 
        if loss_integer == float('inf'):
            return -1e10  # Return a very low log-posterior value on error
        else:
            filename = folder_path + f'loss_params_{try_num}.txt'
            with open(filename, 'w') as f:
                print(loss_integer, *params, file=f)
            try_num += 1
            print("--- %s seconds ---" % (time.time() - start_time))
        # The log-posterior is proportional to -chi^2 / 2
        return -0.5 * loss_integer

    except (ValueError, CosmoComputationError) as e:
        print(f"Computation error for params {params}: {e}")
        return -1e10  # Return a very low log-posterior value on error


bounds = [[350, 500], [1.2,1.8], [0.15, 0.17], [0.5, 0.62]]

from gpry import Runner

runner = Runner(log_posterior_for_gpry, bounds, checkpoint="./data/output/", load_checkpoint='resume')
runner.run()