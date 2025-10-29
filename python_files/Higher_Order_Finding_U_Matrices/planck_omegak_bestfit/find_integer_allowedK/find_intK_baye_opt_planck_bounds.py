"""
Bayesian optimization to find integer-spaced K values
Using parameter bounds from filtered Planck data (nu_spacing=4, 1-sigma)
"""

import numpy as np
import sys
import os
import re
from scipy.optimize import root_scalar
from scipy.optimize import minimize
import time
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK
from multiprocessing import cpu_count
from bayes_opt import BayesianOptimization

data_folder = './data/try_intK_planck_bounds/'

nu_spacing = 4
print("nu_spacing = ", nu_spacing)

# Detect number of CPUs
n_processes = 4
print(f"Using {n_processes} parallel processes")

# Print HPC info if available
if 'SLURM_JOB_ID' in os.environ:
    print(f"Running on HPC - Job ID: {os.environ.get('SLURM_JOB_ID')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}")

start_time = time.time()

def find_max_tried_num():
    pattern = re.compile(r"try_(\d+)")
    max_num = 0
    if os.path.exists(data_folder):
        for dirname in os.listdir(data_folder):
            match = pattern.match(dirname)
            if match:
                num = int(match.group(1))
                allowedK_file = f'{data_folder}/{dirname}/allowedK_integer.npy'
                if os.path.exists(allowedK_file):
                    max_num = max(max_num, num)
    return max_num

try_num = find_max_tried_num() + 1
print(f"Starting from try_num = {try_num}")

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
nu_spacing4_bestfit = [401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583,
                       1.9375648884116028, 0.9787493821596979, 0.019760560255556746]

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

def calculate_allowedK(params, folder_path):
    try:
        z_rec = calculate_z_rec(params)
        compute_U_matrices(params, z_rec, folder_path, n_processes)
        compute_X_recs(params, z_rec, folder_path)
        allowedK_integer = compute_allowedK(params, folder_path)
        return allowedK_integer
    except CosmoComputationError as e:
        print(f"Computation error for params {params}: {e}")
        return None

def calculate_integer_loss(params, folder_path, n_ignore=8, use_weighting=True):
    """
    Calculate loss using Linear Regression method.
    """
    allowedK_integer = calculate_allowedK(params, folder_path)
    if allowedK_integer is None:
        return float('inf')
    else:
        print("allowedK_integer =", allowedK_integer)

        if not allowedK_integer or len(allowedK_integer) <= n_ignore + 2:
            return float('inf')

        high_k = np.array(allowedK_integer[n_ignore:])
        indices = np.arange(len(high_k))

        slope, intercept = np.polyfit(indices, high_k, 1)

        ideal_spacing = np.round(slope)
        if ideal_spacing == 0:
            return float('inf')

        ideal_intercept = np.mean(high_k - ideal_spacing * indices)
        ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)

        squared_errors = (high_k - ideal_k_sequence)**2

        if use_weighting:
            weights = np.linspace(0.1, 1.0, len(high_k))
            loss = np.mean(weights * squared_errors)
        else:
            loss = np.mean(squared_errors)

        return loss

def log_posterior(mt, kt, Omegab_ratio, h):
    """
    Returns a single number proportional to the log-posterior.
    """
    params = [mt, kt, Omegab_ratio, h]
    try:
        global try_num
        folder_path = f'{data_folder}/try_'+str(try_num)+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        loss_integer = calculate_integer_loss(params, folder_path)
        if loss_integer == float('inf'):
            return -1e10
        else:
            filename = folder_path + f'loss_params_{try_num}.txt'
            with open(filename, 'w') as f:
                print(loss_integer, *params, file=f)
            try_num += 1
            print("--- %s seconds ---" % (time.time() - start_time))
        return -0.5 * loss_integer

    except (ValueError, CosmoComputationError) as e:
        print(f"Computation error for params {params}: {e}")
        return -1e10

############################
# Load parameter bounds from Planck data
############################

def load_planck_bounds(bounds_file='./planck_filtered_pbounds.txt'):
    """
    Load parameter bounds from file.
    If file doesn't exist, use default bounds.
    """
    if os.path.exists(bounds_file):
        print(f"\nLoading parameter bounds from {bounds_file}...")
        pbounds = {}
        with open(bounds_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    param_name = parts[0]
                    lower_bound = float(parts[1])
                    upper_bound = float(parts[2])
                    pbounds[param_name] = (lower_bound, upper_bound)

        print("Loaded bounds:")
        for param, (lower, upper) in pbounds.items():
            print(f"  {param:<15}: [{lower:.6f}, {upper:.6f}]")

        return pbounds
    else:
        print(f"\nWarning: {bounds_file} not found. Using default bounds.")
        print("Run extract_planck_bounds.py first to generate Planck-based bounds.")
        # Default bounds (original)
        return {
            'mt': (350, 500),
            'kt': (1.2, 1.8),
            'Omegab_ratio': (0.15, 0.17),
            'h': (0.5, 0.62)
        }

# Load bounds
pbounds = load_planck_bounds()

############################
# Initialize optimizer
############################

optimizer = BayesianOptimization(
    f=log_posterior,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True,
)

############################
# Load existing data points (if any)
############################

def load_existing_data(data_dir=data_folder):
    """
    Load all existing loss_params files.
    """
    existing_data = []

    if not os.path.exists(data_dir):
        print("No existing data directory found.")
        return existing_data

    for try_dir in os.listdir(data_dir):
        try_path = os.path.join(data_dir, try_dir)
        if os.path.isdir(try_path):
            for filename in os.listdir(try_path):
                if filename.startswith('loss_params_') and filename.endswith('.txt'):
                    filepath = os.path.join(try_path, filename)
                    try:
                        with open(filepath, 'r') as f:
                            line = f.readline().strip()
                            values = [float(x) for x in line.split()]
                            if len(values) == 5:
                                loss_integer, mt, kt, Omegab_ratio, h = values
                                params_dict = {
                                    'mt': mt,
                                    'kt': kt,
                                    'Omegab_ratio': Omegab_ratio,
                                    'h': h
                                }
                                target = -0.5 * loss_integer
                                existing_data.append((params_dict, target))
                    except Exception as e:
                        print(f"Warning: Could not load {filepath}: {e}")

    return existing_data

# Load and register existing data
existing_data = load_existing_data()
if existing_data:
    print(f"\nFound {len(existing_data)} existing data points. Registering them...")
    for params_dict, target in existing_data:
        # Check if parameters are within new bounds
        within_bounds = all(
            pbounds[param][0] <= params_dict[param] <= pbounds[param][1]
            for param in pbounds.keys()
        )
        if within_bounds:
            optimizer.register(params=params_dict, target=target)
        else:
            print(f"Skipping point outside new bounds: {params_dict}")

    print(f"Successfully registered {len(optimizer.space)} points within bounds!")
    if len(optimizer.space) > 0:
        print(f"Current best: {optimizer.max}")
else:
    print("No existing data found. Starting fresh.")

############################
# Run the optimization
############################

print("\n" + "="*80)
print("STARTING BAYESIAN OPTIMIZATION WITH PLANCK-CONSTRAINED BOUNDS")
print("="*80)

optimizer.maximize(
    init_points=0 if len(existing_data) > 20 else 20,
    n_iter=150,
)

# Save the complete optimizer state
STATE_FILE = "./bayes_opt_state_planck_bounds.json"
optimizer.save_state(STATE_FILE)
print(f"\nOptimizer state saved to {STATE_FILE}")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"Best result: {optimizer.max}")
print(f"Total evaluations: {len(optimizer.space)}")
print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
