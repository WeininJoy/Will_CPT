"""
Differential Evolution optimization to find integer-spaced K values
Using parameter bounds from filtered Planck data (nu_spacing=4, 1-sigma)
MPI-parallelised via mpi4py.futures.MPIPoolExecutor
"""

import numpy as np
import sys
import os
import re
from scipy.optimize import root_scalar, differential_evolution
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import time
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

data_folder = './data/try_intK_planck_bounds/'

nu_spacing = 4

# Number of OpenMP threads per MPI worker (set OMP_NUM_THREADS in the job script)
n_processes = int(os.environ.get('OMP_NUM_THREADS', 4))

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

def _compute_loss_from_stable_k(stable_k, w_slope=1.0, w_intercept=1.0, w_residual=1.0,
                                 verbose=True):
    """
    Core loss computation given an already-trimmed array of stable k values.

    Separated from calculate_integer_loss so it can be called directly in tests.

    The loss has three components:

      1. slope_loss         = (fitted_slope - nu_spacing)^2
         Penalises mean spacing != nu_spacing.

      2. per_k_integer_loss = mean( (k_i - ideal_integer_i)^2 )
         Directly measures how far each k value is from its target integer.
         The ideal integer sequence is nu_spacing*i + round(mean(k_i - nu_spacing*i)),
         i.e. the nearest integer-spaced grid to the data.
         This replaces the old intercept_loss = (intercept - round(intercept))^2,
         which only checked the extrapolated k at i=0 and missed the cumulative
         drift of fractional parts across all modes.

      3. weighted_residual_loss = sum(w_i * residual_i^2) / sum(w_i)
         Penalises non-linearity (jitter around the best-fit line).
         w_i = i+1, so high-k modes contribute more.

    Parameters
    ----------
    stable_k : array_like
        Trimmed, monotone increasing k values (no boundary artifacts).
    w_slope, w_intercept, w_residual : float
        Weights for each loss component.
    verbose : bool
        If True, print diagnostics.
    """
    stable_k = np.asarray(stable_k, dtype=float)
    n = len(stable_k)
    indices = np.arange(n, dtype=float)

    # High-k weights: w_i = i+1  (last mode has n× more weight than first)
    hk_weights = indices + 1.0

    # ── 1. Weighted linear fit: k_i = intercept + slope * i ─────────────────
    slope, intercept = np.polyfit(indices, stable_k, 1, w=hk_weights)

    # ── 2. Slope loss — penalises mean spacing != nu_spacing ─────────────────
    slope_loss = (slope - nu_spacing) ** 2

    # ── 3. Per-k integer closeness loss ──────────────────────────────────────
    # Construct the nearest ideal integer sequence with spacing = nu_spacing.
    # The anchor is the integer nearest to the mean offset of the data.
    ideal_intercept_int = np.round(np.mean(stable_k - nu_spacing * indices))
    ideal_sequence = ideal_intercept_int + nu_spacing * indices
    deviations_from_integers = stable_k - ideal_sequence
    per_k_integer_loss = np.mean(deviations_from_integers ** 2)

    # ── 4. Weighted residual loss — penalises non-linearity ──────────────────
    fit_values = intercept + slope * indices
    residuals = stable_k - fit_values
    weights_norm = hk_weights / hk_weights.sum()
    weighted_residual_loss = np.sum(weights_norm * residuals ** 2)

    # ── 5. Total loss ─────────────────────────────────────────────────────────
    total_loss = (w_slope     * slope_loss +
                  w_intercept * per_k_integer_loss +
                  w_residual  * weighted_residual_loss)

    if verbose:
        print(f"  Fitted slope={slope:.6f} (target {nu_spacing}), "
              f"ideal_intercept={ideal_intercept_int:.0f}")
        print(f"  slope_loss={slope_loss:.3e} (×{w_slope:.0f} → {w_slope*slope_loss:.4f}), "
              f"per_k_integer_loss={per_k_integer_loss:.3e} "
              f"(×{w_intercept:.0f} → {w_intercept*per_k_integer_loss:.4f}), "
              f"residual_loss={weighted_residual_loss:.3e} "
              f"(×{w_residual:.0f} → {w_residual*weighted_residual_loss:.4f})")
        print(f"  Total loss={total_loss:.6e}")

    return total_loss


def calculate_integer_loss(params, folder_path, n_min_stable=12, spacing_tol_factor=8,
                           w_slope=2.0, w_intercept=1.0, w_residual=0.2):
    """
    Calculate loss measuring how closely allowedK_integer values form a perfectly
    integer-spaced sequence with spacing exactly equal to nu_spacing (4).

    Delegates core loss computation to _compute_loss_from_stable_k after
    loading data and trimming boundary artifacts.

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
        Weight for the slope (spacing) loss term (default: 500)
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

    return _compute_loss_from_stable_k(stable_k, w_slope=w_slope,
                                        w_intercept=w_intercept,
                                        w_residual=w_residual, verbose=True)
    
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

def objective_wrapper(x):
    """
    Wraps calculate_integer_loss for scipy.optimize.differential_evolution.
    Accepts a 1-D array [mt, kt, Omegab_ratio, h] and returns a scalar loss.
    Each MPI worker writes its results to a worker-specific sub-folder so
    concurrent evaluations do not overwrite each other.
    """
    mt, kt, Omegab_ratio, h = x
    params = [mt, kt, Omegab_ratio, h]

    rank = MPI.COMM_WORLD.Get_rank()
    folder_path = os.path.join(data_folder, f'worker_{rank}', '')
    os.makedirs(folder_path, exist_ok=True)

    try:
        loss_integer = calculate_integer_loss(params, folder_path)
    except (ValueError, CosmoComputationError) as e:
        print(f"[rank {rank}] Computation error for params {params}: {e}")
        loss_integer = float('inf')

    # scipy DE does not handle inf/nan internally; replace with a large sentinel
    if loss_integer == float('inf') or np.isnan(loss_integer):
        return 1e10

    # Log the successful evaluation
    log_file = os.path.join(folder_path, 'loss_log.txt')
    with open(log_file, 'a') as f:
        print(loss_integer, *params, file=f)

    print(f"[rank {rank}] loss={loss_integer:.6e}  params={params}  "
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
        print("Run extract_planck_bounds.py first to generate Planck-based bounds.")
        # Default bounds (original)
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
    # Load bounds
    pbounds = load_planck_bounds()

    # Print run configuration
    print("nu_spacing = ", nu_spacing)
    print(f"Using OMP_NUM_THREADS={n_processes} per MPI worker")
    if 'SLURM_JOB_ID' in os.environ:
        print(f"Running on HPC - Job ID: {os.environ.get('SLURM_JOB_ID')}")
        print(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}")

    # scipy expects a list of (lower, upper) tuples ordered like objective_wrapper unpacks x
    bounds = [
        pbounds['mt'],
        pbounds['kt'],
        pbounds['Omegab_ratio'],
        pbounds['h'],
    ]

    print("\n" + "="*80)
    print("STARTING DIFFERENTIAL EVOLUTION WITH PLANCK-CONSTRAINED BOUNDS")
    print("="*80)

    with MPIPoolExecutor() as executor:
        result = differential_evolution(
            objective_wrapper,
            bounds,
            strategy='best1bin',
            maxiter=50,
            popsize=15,          # population = popsize * len(bounds) = 60 per generation
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.7,
            updating='deferred', # wait for all workers before updating the population
            workers=executor.map,
            polish=True,         # local search at the end to find the exact minimum
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
