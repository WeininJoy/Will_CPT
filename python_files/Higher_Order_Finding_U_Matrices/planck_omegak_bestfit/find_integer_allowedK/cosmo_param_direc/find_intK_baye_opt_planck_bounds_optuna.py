"""
Optuna Asynchronous Optimization to find integer-spaced K values
Using parameter bounds from filtered Planck data.
Workers are totally independent and synchronize via an SQLite database.
"""
import os

# ==============================================================================
# 1. READ CPUS FOR MULTIPROCESSING POOL
# Read exactly how many CPUs SLURM gave this specific worker (which is 8).
# ==============================================================================
n_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))

# ==============================================================================
# 2. LOCK DOWN C-LIBRARY THREADING
# Now that we know we have 8 CPUs for the pool, force SciPy/CLASS to only use 
# 1 thread per process so they don't fight inside the pool.
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import sys
import time
import classy
from scipy.optimize import root_scalar
from classy import CosmoComputationError
from scipy.spatial import cKDTree
from anesthetic import read_chains
import optuna

# Import your custom modules
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

data_folder = './data/try_intK_planck_optuna/'
os.makedirs(data_folder, exist_ok=True)

nu_spacing = 4

# Set up the shared Optuna database URL (saved in your RDS data folder)
# Using a high timeout prevents "database is locked" errors when many HPC workers write at once.
DB_URL = f"sqlite:///{os.path.join(data_folder, 'planck_optimization.db')}?timeout=60"

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
nu_spacing4_bestfit = [0.456043,-0.038752,0.155504,0.566647,2.095762,0.972835,0.053017]
# [0.4447390978198118, np.float64(-0.03952117480452887),0.16167855679796067, 0.55768703176223,2.095762, 0.972835,0.053017]

##################
def calculate_z_rec(params):
    OmegaM, OmegaK, omega_b_ratio, h = params

    CMB_params = {
        'output': 'tCl',
        'h': h,
        'Omega_b': omega_b_ratio*OmegaM,
        'Omega_cdm': (1.-omega_b_ratio) *OmegaM,
        'Omega_k': float(OmegaK),
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

    derived = cosmo.get_current_derived_parameters(['z_rec'])
    z_rec = derived['z_rec']

    cosmo.struct_cleanup()
    cosmo.empty()

    return z_rec

def calculate_allowedK(params, worker_folder):
    try:
        z_rec = calculate_z_rec(params)
        compute_U_matrices(params, z_rec, worker_folder, n_processes)
        compute_X_recs(params, z_rec, worker_folder)
        allowedK_integer = compute_allowedK(params, worker_folder)
        return allowedK_integer
    except CosmoComputationError as e:
        print(f"Computation error for params {params}: {e}")
        return None

def calculate_integer_loss(allowedK_integer, n_min=12, spacing_tol_factor=3,
                           w_slope=1.0, w_integer=0.0):
                            # w_slope=20.0, w_integer=1.0):
    """
    Loss for K sequences that asymptotically converge as k increases.
    Uses:
      1. A 2-point midpoint filter to kill alternating (+)/(-) wiggles.
      2. A 'plateau' inverted-parabola weighting to equally weight the entire tail,
         improving robustness against noise in the final points.
    """
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    # --- 1. Robust trimming of non-physical jumps ---
    spacings = np.diff(k)
    nu_spacing = 4.0
    median_spacing = np.median(spacings)
    mad = np.median(np.abs(spacings - median_spacing))
    threshold = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(k)
    for i in range(len(spacings) - 1, len(spacings) // 2, -1):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1

    k = k[:end_idx]
    n = len(k)
    if n < n_min:
        return np.inf, np.inf, np.inf

    # --- 2. 2-Point Midpoint Filter (Kills alternating wiggles) ---
    k_smooth = (k[:-1] + k[1:]) / 2.0
    indices_smooth = np.arange(len(k_smooth), dtype=float) + 0.5

    # --- 3. NEW: The User's "Plateau" Weighting (-x^2 style) ---
    # Create z from 0.0 to 1.0 representing position in the smooth array
    z_sp = np.linspace(0, 1, len(k_smooth) - 1)
    sp_weights = 1.0 - (1.0 - z_sp)**2 
    sp_weights /= sp_weights.sum()  # Normalize to sum to 1

    z_int = np.linspace(0, 1, len(k_smooth))
    int_weights = 1.0 - (1.0 - z_int)**2
    int_weights /= int_weights.sum() # Normalize to sum to 1

    # --- 4. Slope Loss ---
    spacings_smooth = np.diff(k_smooth)                         
    tail_slope_loss = np.sum(sp_weights * (spacings_smooth - nu_spacing) ** 2)

    # --- 5. Integer Loss ---
    weighted_mean_intercept = np.sum(int_weights * (k_smooth - nu_spacing * indices_smooth))
    ideal_intercept = np.round(weighted_mean_intercept)
    ideal_sequence  = ideal_intercept + nu_spacing * indices_smooth
    
    tail_integer_loss = np.sum(int_weights * (k_smooth - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss


def calculate_integer_loss_old(allowedK_integer, n_min_stable=12, spacing_tol_factor=8,
                           w_slope=5.0, w_intercept=1.0, w_residual=0): # ignore residuals since the fluctuation of allowed k could be ignored

    if allowedK_integer is None:
        return float('inf'), float('inf'), float('inf')

    all_k = np.array(allowedK_integer)
    if len(all_k) < n_min_stable + 3:
        return float('inf'), float('inf'), float('inf')

    spacings = np.diff(all_k)
    median_spacing = np.median(spacings)
    mad = np.median(np.abs(spacings - median_spacing))
    threshold = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(all_k)
    for i in range(len(spacings) - 1, len(spacings) // 2, -1):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1
        else:
            break

    stable_k = all_k[:end_idx]
    n = len(stable_k)

    if n < n_min_stable:
        return float('inf'), float('inf'), float('inf')

    indices = np.arange(n, dtype=float)
    hk_weights = indices + 1.0
    slope, intercept = np.polyfit(indices, stable_k, 1, w=hk_weights)

    slope_loss = (slope - nu_spacing) ** 2

    # Per-k integer closeness: measure how far each k value is from its target
    # integer, using the nearest integer-spaced grid anchored to the data.
    ideal_intercept_int = np.round(np.mean(stable_k - nu_spacing * indices))
    ideal_sequence = ideal_intercept_int + nu_spacing * indices
    per_k_integer_loss = np.mean((stable_k - ideal_sequence) ** 2)

    fit_values = intercept + slope * indices
    residuals = stable_k - fit_values
    weights_norm = hk_weights / hk_weights.sum()
    weighted_residual_loss = np.sum(weights_norm * residuals ** 2)

    total_loss = ( w_slope * slope_loss + w_intercept * per_k_integer_loss + w_residual * weighted_residual_loss) # 
    return slope_loss, per_k_integer_loss, total_loss

############################
# Load KDTree once globally
############################
print("Loading Planck chains for KDTree interpolation...")
try:
    planck_chain_path = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/Planck/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE'
    global_ns = read_chains(planck_chain_path)
    
    chain_params = np.vstack([
        global_ns['omegam'], 
        global_ns['omegabh2'], 
        global_ns['H0']/100,   
        global_ns['omegak']
    ]).T
    
    chain_chi2 = global_ns['chi2_CMB'].values
    param_stds = np.std(chain_params, axis=0)
    normalized_chain_params = chain_params / param_stds
    mcmc_tree = cKDTree(normalized_chain_params)
    print("KDTree successfully built!")
except Exception as e:
    print(f"Failed to load chains: {e}")
    mcmc_tree = None

############################
# Load parameter bounds 
############################
# def load_planck_bounds():
#     bounds_file = '/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/find_integer_allowedK/planck_filtered_pbounds_3-sigma.txt'
#     if os.path.exists(bounds_file):
#         pbounds = {}
#         with open(bounds_file, 'r') as f:
#             for line in f:
#                 if line.startswith('#') or line.strip() == '': continue
#                 parts = line.strip().split()
#                 if len(parts) >= 3:
#                     pbounds[parts[0]] = (float(parts[1]), float(parts[2]))
#         return pbounds
#     else:
#         return {'mt': (350, 500), 'kt': (1.2, 1.8), 'Omegab_ratio': (0.15, 0.17), 'h': (0.5, 0.62)}

# pbounds = load_planck_bounds()

pbounds = {'OmegaM': (0.42, 0.47), 'OmegaK': (-0.041, -0.038), 'Omegab_ratio': (0.14, 0.17), 'h': (0.55, 0.58)}

############################
# Optuna Objective
############################
def objective(trial):
    # 1. Optuna "suggests" parameters within your bounds
    OmegaM = trial.suggest_float('OmegaM', pbounds['OmegaM'][0], pbounds['OmegaM'][1])
    OmegaK = trial.suggest_float('OmegaK', pbounds['OmegaK'][0], pbounds['OmegaK'][1])
    Omegab_ratio = trial.suggest_float('Omegab_ratio', pbounds['Omegab_ratio'][0], pbounds['Omegab_ratio'][1])
    h = trial.suggest_float('h', pbounds['h'][0], pbounds['h'][1])
    
    params = [OmegaM, OmegaK, Omegab_ratio, h]
    omegabh2 = Omegab_ratio * OmegaM * h**2
    
    # 2. CHECK AGAINST PLANCK DEGENERACY CONTOURS
    if mcmc_tree is not None:
        target_params = np.array([OmegaM, omegabh2, h, OmegaK])
        normalized_target = target_params / param_stds
        distances, indices = mcmc_tree.query(normalized_target, k=5)
        
        # EARLY REJECTION!
        # If it fails, we raise TrialPruned. Optuna immediately marks this as "failed"
        # and grabs the next set of parameters without wasting 15 minutes.
        if distances[0] > 3.0:  
            raise optuna.TrialPruned("Point > 3 sigma away from Planck MCMC")
            
        local_chi2 = np.mean(chain_chi2[indices])
        if local_chi2 > 2820: 
            raise optuna.TrialPruned(f"Poor local chi2: {local_chi2:.2f}")
    else:
        local_chi2 = 0
        
    # 3. HEAVY CALCULATION
    # Give this specific worker its own isolated sub-folder based on its Process ID
    worker_pid = os.getpid()
    worker_folder = os.path.join(data_folder, f'worker_pid_{worker_pid}', '')
    os.makedirs(worker_folder, exist_ok=True)
    
    try:
        allowedK_integer = calculate_allowedK(params, worker_folder)
        slope_loss, per_k_integer_loss, total_loss = calculate_integer_loss(allowedK_integer)
    except (ValueError, CosmoComputationError):
        raise optuna.TrialPruned("CLASS Computation Error")

    if total_loss == float('inf') or np.isnan(total_loss):
        raise optuna.TrialPruned("Loss is inf or NaN")

    # 4. PURE INTEGER LOSS
    # We remove local_chi2 from the loss because the KDTree already guarantees 
    # we are within 3-sigma of Planck. Now Optuna focuses purely on the integer spacing!
    
    # Convert the allowedK_integer list into a comma-separated string (no spaces)
    # e.g., "4.0,8.02,11.98,16.01"
    k_array_str = ",".join(map(str, allowedK_integer))
    
    log_file = os.path.join(data_folder, 'master_log.txt')
    with open(log_file, 'a') as f:
        # Appended k_array_str at the very end of the line
        f.write(f"{total_loss:.6e} {slope_loss:.6e} {per_k_integer_loss:.6e} {local_chi2:.2f} " + " ".join(map(str, params)) + f" {k_array_str}\n")

    return total_loss


############################
# Run the optimization
############################

import time

if __name__ == "__main__":
    
    import random 
    
    # STAGGER STARTUP: Prevent Database locking by making workers wait a random amount of seconds
    stagger_time = random.uniform(1, 15)
    print(f"[Worker PID {os.getpid()}] Sleeping for {stagger_time:.1f} seconds to avoid DB locks...")
    time.sleep(stagger_time)

    try:
        sampler = optuna.samplers.CmaEsSampler()
    except ModuleNotFoundError:
        print("cmaes not installed, falling back to TPESampler")
        sampler = optuna.samplers.TPESampler()

    for attempt in range(10):
        try:
            # MUST USE A NEW STUDY NAME so it doesn't load the wide-ranging data
            study = optuna.create_study(
                study_name="planck_integer_k_SmoothWeightedLoss",
                storage=DB_URL,
                direction="minimize",
                sampler=sampler,
                load_if_exists=True 
            )
            break
        except Exception as e:
            time.sleep(2)
    else:
        # If it fails 10 times in a row, something else is wrong
        raise RuntimeError("Could not connect to Optuna Database after 10 attempts.")

    print(f"\n[Worker PID {os.getpid()}] Starting Optuna Optimization...")
    try:
        best_val = study.best_value
    except ValueError:
        best_val = "None (No completed trials yet)"
        
    print(f"Current best pure loss: {best_val}")

    # Start asking for trials indefinitely (SLURM will cut it off at 36 hours)
    study.optimize(objective, n_trials=None)

    print(f"[Worker PID {os.getpid()}] Finished assigned trials.")
