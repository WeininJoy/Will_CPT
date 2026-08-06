"""
Differential Evolution optimization to find integer-spaced K values
Using parameter bounds from filtered Planck data (nu_spacing=4, 1-sigma)
Local test version: 4 CPU cores via multiprocessing (no MPI required)
"""

import numpy as np
import sys
import os
import re
from scipy.optimize import root_scalar, differential_evolution
import multiprocessing
import time
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

data_folder = './data/try_intK_local/'

nu_spacing = 4

# Cannot nest multiprocessing: scipy DE workers are daemon processes and
# cannot spawn children. Pick ONE level of parallelism:
#   - DE runs candidates sequentially (workers=1)
#   - compute_U_matrices uses all 4 cores for the ODE solve (n_processes=4)
N_WORKERS = 1
n_processes = 4

start_time = time.time()

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
    """
    allowedK_integer = calculate_allowedK(params, folder_path)
    if allowedK_integer is None:
        return float('inf')

    all_k = np.array(allowedK_integer)
    print("allowedK_integer =", all_k)

    if len(all_k) < n_min_stable + 3:
        return float('inf')

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
    n_trimmed = len(all_k) - len(stable_k)
    n = len(stable_k)
    print(f"  Trimmed {n_trimmed} boundary point(s); using {n}/{len(all_k)} stable points")

    if n < n_min_stable:
        print(f"  Too few stable points ({n} < {n_min_stable}), returning inf")
        return float('inf')

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

def objective_wrapper(x):
    """
    Wraps calculate_integer_loss for scipy.optimize.differential_evolution.
    Uses the process PID to create an isolated sub-folder per worker.
    """
    mt, kt, Omegab_ratio, h = x
    params = [mt, kt, Omegab_ratio, h]

    pid = os.getpid()
    folder_path = os.path.join(data_folder, f'worker_{pid}', '')
    os.makedirs(folder_path, exist_ok=True)

    try:
        loss_integer = calculate_integer_loss(params, folder_path)
    except (ValueError, CosmoComputationError) as e:
        print(f"[pid {pid}] Computation error for params {params}: {e}")
        loss_integer = float('inf')

    if loss_integer == float('inf') or np.isnan(loss_integer):
        return 1e10

    log_file = os.path.join(folder_path, 'loss_log.txt')
    with open(log_file, 'a') as f:
        print(loss_integer, *params, file=f)

    print(f"[pid {pid}] loss={loss_integer:.6e}  params={params}  "
          f"elapsed={time.time()-start_time:.1f}s")
    return loss_integer

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
        return {
            'mt': (350, 500),
            'kt': (1.2, 1.8),
            'Omegab_ratio': (0.15, 0.17),
            'h': (0.5, 0.62)
        }

############################
# Run the optimization
############################

if __name__ == "__main__":
    pbounds = load_planck_bounds()

    print("nu_spacing = ", nu_spacing)
    print(f"Workers: {N_WORKERS} (multiprocessing), OMP threads per worker: {n_processes}")

    bounds = [
        pbounds['mt'],
        pbounds['kt'],
        pbounds['Omegab_ratio'],
        pbounds['h'],
    ]

    print("\n" + "="*80)
    print("STARTING DIFFERENTIAL EVOLUTION (LOCAL TEST, 4 CORES)")
    print("="*80)

    result = differential_evolution(
        objective_wrapper,
        bounds,
        strategy='best1bin',
        maxiter=10,          # reduced for local testing
        popsize=4,           # population = popsize * len(bounds) = 16 per generation
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.7,
        updating='deferred',
        workers=N_WORKERS,   # 1 = sequential DE (avoids nested daemon pool error)
        polish=True,
        seed=1,
        disp=True,
    )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best parameters: mt={result.x[0]:.6f}, kt={result.x[1]:.6f}, "
          f"Omegab_ratio={result.x[2]:.6f}, h={result.x[3]:.6f}")
    print(f"Minimum loss:    {result.fun:.6e}")
    print(f"Converged:       {result.success}  ({result.message})")
    print(f"Total evaluations: {result.nfev}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
