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

data_folder = './data/try_intK/'

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
    # Define the directory containing the try folders
    pattern = re.compile(r"try_(\d+)")
    max_num = 0
    if os.path.exists(data_folder):
        for dirname in os.listdir(data_folder):
            match = pattern.match(dirname)
            if match:
                num = int(match.group(1))  # Extract the number and convert to integer
                # Only count this try if allowedK_integer.npy exists
                allowedK_file = f'{data_folder}/{dirname}/allowedK_integer.npy'
                if os.path.exists(allowedK_file):
                    max_num = max(max_num, num)
    return max_num

# Initialize try_num from existing files, or start from 1
try_num = find_max_tried_num() + 1
print(f"Starting from try_num = {try_num}")

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
        compute_U_matrices(params, z_rec, folder_path, n_processes)  # Added n_processes
        compute_X_recs(params, z_rec, folder_path)
        allowedK_integer = compute_allowedK(params, folder_path)
        return allowedK_integer
    except CosmoComputationError as e:
        print(f"Computation error for params {params}: {e}")
        return None

def calculate_integer_loss(params, folder_path, n_ignore=8, use_weighting=True):
    """
    Calculate loss using Linear Regression method.
    This fits a line to high-K values and measures deviation from an ideal line
    with integer spacing. More robust for optimization than simple MSE.

    The ideal pattern is: K_i = S*i + C, where S is an INTEGER spacing.

    Args:
        params: Cosmological parameters [mt, kt, omega_b_ratio, h]
        folder_path: Path to save computation results
        n_ignore: Number of initial K values to ignore (focus on high-K modes)
        use_weighting: If True, applies linear weighting to prioritize high-K modes

    Returns:
        float: The calculated loss value (0 = perfect equally spaced integers, larger = worse)
    """
    allowedK_integer = calculate_allowedK(params, folder_path)
    if allowedK_integer is None: 
        return float('inf')
    else:
        print("allowedK_integer =", allowedK_integer)

        if not allowedK_integer or len(allowedK_integer) <= n_ignore + 2:
            return float('inf')

        # Focus on high-K modes
        high_k = np.array(allowedK_integer[n_ignore:])
        indices = np.arange(len(high_k))

        # --- 1. Fit a line to the data: k = slope * i + intercept ---
        slope, intercept = np.polyfit(indices, high_k, 1)

        # --- 2. Determine the ideal integer spacing ---
        # The ideal spacing must be an integer
        ideal_spacing = np.round(slope)
        if ideal_spacing == 0:  # Avoid degenerate case
            return float('inf')

        # --- 3. Construct the ideal sequence ---
        # Find the best intercept for the ideal line with integer spacing
        ideal_intercept = np.mean(high_k - ideal_spacing * indices)
        ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)

        # --- 4. Calculate the loss ---
        # MSE between actual data and reconstructed ideal sequence
        squared_errors = (high_k - ideal_k_sequence)**2

        if use_weighting:
            # Linearly increasing weights: prioritize later (higher-K) values
            weights = np.linspace(0.1, 1.0, len(high_k))
            loss = np.mean(weights * squared_errors)
        else:
            loss = np.mean(squared_errors)

        return loss

def log_posterior(mt, kt, Omegab_ratio, h):
    """
    Returns a single number proportional to the log-posterior.
    GPry will build a surrogate of this function.
    """
    params = [mt, kt, Omegab_ratio, h]
    try: 
        global try_num
        folder_path = f'{data_folder}/try_'+str(try_num)+'/'
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


from bayes_opt import BayesianOptimization

pbounds = {'mt': (350, 500), 'kt': (1.2, 1.8), 'Omegab_ratio': (0.15, 0.17), 'h': (0.5, 0.62)}

optimizer = BayesianOptimization(
    f=log_posterior,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True,  # Allow duplicates when loading existing data
)

############################
# Load existing data points (if any)
############################
def load_existing_data(data_dir=data_folder):
    """
    Load all existing loss_params files and return as a list of (params_dict, target) tuples.

    File format: loss_integer mt kt Omegab_ratio h
    Target for optimizer: -0.5 * loss_integer (log_posterior)
    """
    existing_data = []

    if not os.path.exists(data_dir):
        print("No existing data directory found.")
        return existing_data

    # Find all loss_params files
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
                                # Convert loss to target (log_posterior = -0.5 * loss)
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
        optimizer.register(params=params_dict, target=target)
    print(f"Successfully registered {len(existing_data)} points!")
    print(f"Current best: {optimizer.max}")
else:
    print("No existing data found. Starting fresh.")

# Run the optimization
# If you have existing data, you might want to reduce init_points
optimizer.maximize(
    init_points=0 if len(existing_data) > 20 else 20,
    n_iter=150,
)

# Save the complete optimizer state to a JSON file
STATE_FILE = "./bayes_opt_state.json"
optimizer.save_state(STATE_FILE)
print(f"\nOptimizer state saved to {STATE_FILE}")

############################
# Start from previous run
############################

# # The state file we saved in the previous step
# STATE_FILE = "./bayes_opt_state.json"

# # 1. Initialize a new optimizer instance
# # (must have the SAME objective function and pbounds)
# new_optimizer = BayesianOptimization(
#     f=log_posterior,
#     pbounds=pbounds,
#     random_state=1,
#     allow_duplicate_points=True,
# )

# # 2. Load the previous optimization state
# # This restores all data from the state file, including the GP model.
# new_optimizer.load_state(STATE_FILE)

# # OPTIONAL: You can check how many points have been loaded
# print(f"Optimizer now has {len(new_optimizer.space)} points loaded.")
# print(f"Current best target: {new_optimizer.max}")

# # 3. Continue maximizing for more iterations
# # Since the initial points are already loaded, we only specify new iterations.
# new_optimizer.maximize(
#     init_points=0,  # Set to 0 since we're resuming
#     n_iter=150,     # Run for 150 *additional* iterations
# )

# # 4. Save the updated state
# new_optimizer.save_state(STATE_FILE)
# print(f"\nUpdated optimizer state saved to {STATE_FILE}")