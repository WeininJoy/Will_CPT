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
nu_spacing4_bestfit = [409.969398,1.459351,0.163514,0.547313,2.095762,0.972835,0.053017]

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

def calculate_integer_loss(params, folder_path, n_min_stable=12, spacing_tol_factor=8,
                           w_slope=2.0, w_intercept=1.0, w_residual=0.2):
    """
    Calculate loss measuring how closely allowedK_integer values form a perfectly
    integer-spaced sequence with spacing exactly equal to nu_spacing (4).

    The loss has three components:

      1. slope_loss         = (fitted_slope - nu_spacing)^2
         Penalises mean spacing != nu_spacing.

      2. per_k_integer_loss = mean( (k_i - ideal_integer_i)^2 )
         Directly measures how far each k value is from its target integer.
         The ideal integer sequence has spacing nu_spacing, anchored at the
         nearest integer to the mean offset of the data.
         Replaces the old intercept_loss = (intercept - round(intercept))^2,
         which only checked the extrapolated k at i=0 and missed the cumulative
         drift across all modes.

      3. weighted_residual_loss = sum(w_i * residual_i^2) / sum(w_i)
         Penalises non-linearity (jitter around the best-fit line).
         w_i = i+1, so high-k modes contribute more to this term.

    Parameters:
    -----------
    params : list
        Cosmological parameters [mt, kt, omega_b_ratio, h]
    folder_path : str
        Path to save/load data
    n_min_stable : int
        Minimum number of stable points required; returns inf if fewer (default: 12)
    spacing_tol_factor : float
        Threshold multiplier on MAD of spacings for boundary trimming (default: 8)
    w_slope : float
        Weight for the slope (spacing) loss term (default: 1)
    w_intercept : float
        Weight for the per-k integer closeness loss term (default: 1)
    w_residual : float
        Weight for the weighted residual loss term (default: 1)
    """
    allowedK_integer = calculate_allowedK(params, folder_path)
    if allowedK_integer is None:
        return float('inf')

    all_k = np.array(allowedK_integer)
    print("allowedK_integer =", all_k)

    if len(all_k) < n_min_stable + 3:
        return float('inf')

    spacings = np.diff(all_k)

    # Trim boundary artifacts from the right end.
    # The median is robust to the few boundary outliers; MAD gives a tight scale.
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
    n_trimmed = len(all_k) - len(stable_k)
    n = len(stable_k)
    print(f"  Trimmed {n_trimmed} boundary point(s); using {n}/{len(all_k)} stable points")

    if n < n_min_stable:
        print(f"  Too few stable points ({n} < {n_min_stable}), returning inf")
        return float('inf')

    indices = np.arange(n, dtype=float)

    # High-k weights: w_i = i+1  (last mode has n× more weight than first)
    hk_weights = indices + 1.0   # shape (n,)

    # ── 1. Weighted linear fit: k_i = intercept + slope * i ─────────────────
    slope, intercept = np.polyfit(indices, stable_k, 1, w=hk_weights)

    # ── 2. Slope loss — penalises mean spacing != nu_spacing ─────────────────
    slope_loss = (slope - nu_spacing) ** 2

    # ── 3. Per-k integer closeness loss ──────────────────────────────────────
    # Construct the nearest ideal integer sequence with spacing = nu_spacing.
    ideal_intercept_int = np.round(np.mean(stable_k - nu_spacing * indices))
    ideal_sequence = ideal_intercept_int + nu_spacing * indices
    per_k_integer_loss = np.mean((stable_k - ideal_sequence) ** 2)

    # ── 4. Weighted residual loss — penalises non-linearity ──────────────────
    fit_values = intercept + slope * indices
    residuals = stable_k - fit_values
    weights_norm = hk_weights / hk_weights.sum()
    weighted_residual_loss = np.sum(weights_norm * residuals ** 2)

    # ── 5. Total loss ─────────────────────────────────────────────────────────
    total_loss = (w_slope     * slope_loss +
                  w_intercept * per_k_integer_loss +
                  w_residual  * weighted_residual_loss)

    print(f"  Fitted slope={slope:.6f} (target {nu_spacing}), "
          f"ideal_intercept={ideal_intercept_int:.0f}")
    print(f"  slope_loss={slope_loss:.3e} (×{w_slope:.0f} → {w_slope*slope_loss:.4f}), "
          f"per_k_integer_loss={per_k_integer_loss:.3e} (×{w_intercept:.0f} → {w_intercept*per_k_integer_loss:.4f}), "
          f"residual_loss={weighted_residual_loss:.3e} (×{w_residual:.0f} → {w_residual*weighted_residual_loss:.4f})")
    print(f"  Total loss={total_loss:.6e}")

    return total_loss
    
def calculate_integer_loss_old(params, folder_path, n_tail=7, slope_weight=2.0):
    """
    Calculate loss focusing on the high-K tail where oscillations stabilize.

    Strategy:
    1. Use broad range (after n_ignore) to determine ideal asymptotic spacing
    2. Focus on last n_tail points where wiggles stabilize
    3. Penalize both deviation from ideal AND the slope of deviations

    Parameters:
    -----------
    params : list
        Cosmological parameters [mt, kt, omega_b_ratio, h]
    folder_path : str
        Path to save/load data
    n_tail : int
        Number of final k-values to use for loss calculation (default: 7)
    slope_weight : float
        Weight for penalizing slope of deviations (default: 10.0)
    """
    allowedK_integer = calculate_allowedK(params, folder_path)
    if allowedK_integer is None:
        return float('inf')
    else:
        print("allowedK_integer =", allowedK_integer)

        # Need enough data points
        if len(allowedK_integer) <= n_tail + 5:
            return float('inf')

        all_k = np.array(allowedK_integer)
        tail_k = all_k[-n_tail:]
        tail_indices = np.arange(len(tail_k))

        slope_tail, _ = np.polyfit(tail_indices, tail_k, 1)
        if slope_tail == 0:
            return float('inf')

        print(f"  Ideal spacing from fit: {slope_tail} (slope: {slope_tail:.6f})")
        ideal_spacing_tail = np.round(slope_tail)

        # Compute ideal sequence for the tail
        ideal_intercept_tail = np.mean(tail_k - ideal_spacing_tail * tail_indices)
        ideal_tail_sequence = ideal_spacing_tail * tail_indices + np.round(ideal_intercept_tail)
        # Step 3: Compute deviations from ideal
        deviations = tail_k - ideal_tail_sequence

        print(f"  Tail deviations: min={np.min(deviations):.6f}, max={np.max(deviations):.6f}, mean={np.mean(deviations):.6f}")

        # Step 4: Loss components
        # (a) Mean squared deviation (penalizes offset from ideal)
        mse_loss = np.mean(deviations**2)

        # (b) Slope of deviations (penalizes trending up or down)
        # We want the deviations to be flat (slope ≈ 0)
        deviation_slope, _ = np.polyfit(tail_indices, deviations, 1)
        slope_loss = deviation_slope**2

        print(f"  MSE loss: {mse_loss:.6e}, Deviation slope: {deviation_slope:.6f}, Slope loss: {slope_loss:.6e}")

        # Step 5: Combined loss
        # The slope term is weighted more heavily because we want flat deviations
        total_loss = mse_loss + slope_weight * slope_loss

        print(f"  Total loss: {total_loss:.6e}")

        return total_loss

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

planck_filtered_pbounds_file = '/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/find_integer_allowedK/planck_filtered_pbounds.txt'
refined_planck_bounds_tight_file = '/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/find_integer_allowedK/refined_planck_bounds_tight.txt'

def load_planck_bounds(bounds_file=refined_planck_bounds_tight_file):
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
