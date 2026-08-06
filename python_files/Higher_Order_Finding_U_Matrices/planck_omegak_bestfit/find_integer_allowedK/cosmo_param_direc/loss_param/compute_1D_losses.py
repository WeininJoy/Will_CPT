"""
Compute integer_loss along each cosmological parameter axis, with the other
3 parameters fixed at best-fit values.  Saves one .npz per parameter:

    data/1d_losses/1d_param{i}.npz
        xi          – 1-D array of the varying parameter values
        slope_loss  – 1-D array of slope losses
        integer_loss
        total_loss
        allowedK    – object array of k arrays (or None on failure)
        param_idx   – index of the varying parameter (0–3)
"""

import os
import sys

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import multiprocessing as mp
import multiprocessing.pool

import classy
from classy import CosmoComputationError

custom_path = '/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/find_integer_allowedK/cosmo_param_direc/'
sys.path.insert(0, custom_path)
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

nu_spacing4_bestfit = [0.45531269806747615, -0.0397827163256654, 0.15804679013016237, 0.5647692899377309, 2.068999, 0.977273, 0.051376]
N_ncdm = 1
m_ncdm = 0.06
Neff   = 3.046

data_folder = './data/'
out_folder  = os.path.join(data_folder, '1d_losses')
os.makedirs(out_folder, exist_ok=True)

# ── parameter ranges: same formula as plot script and compute_grid_losses.py ──
_data = []
with open(os.path.join(custom_path, 'data/try_intK_planck_optuna/master_log.txt')) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        _data.append([float(x) for x in parts[:8]])

_arr        = np.array(_data)
_params_all = _arr[:, 4:8]

best_params = np.array(nu_spacing4_bestfit[:4])
param_names = ['OmegaM', 'OmegaK', 'Omegab_ratio', 'h']

p_ranges = []
for i in range(4):
    span = _params_all[:, i].max() - _params_all[:, i].min()
    half = span / 4
    p_ranges.append((best_params[i] - half, best_params[i] + half))


# ── physics (identical to compute_grid_losses.py) ─────────────────────────────
def _calculate_z_rec(params):
    OmegaM, OmegaK, omega_b_ratio, h = params
    CMB_params = {
        'output': 'tCl',
        'h': h,
        'Omega_b': omega_b_ratio * OmegaM,
        'Omega_cdm': (1. - omega_b_ratio) * OmegaM,
        'Omega_k': float(OmegaK),
        'A_s': nu_spacing4_bestfit[4] * 1e-9,
        'n_s': nu_spacing4_bestfit[5],
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'tau_reio': nu_spacing4_bestfit[6],
        'lensing': 'no',
    }
    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()
    z_rec = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
    cosmo.struct_cleanup()
    cosmo.empty()
    return z_rec


def _calculate_allowedK(params, worker_folder):
    try:
        z_rec = _calculate_z_rec(params)
        compute_U_matrices(params, z_rec, worker_folder, n_processes=1)
        compute_X_recs(params, z_rec, worker_folder)
        return compute_allowedK(params, worker_folder)
    except CosmoComputationError as e:
        print(f"  CLASS error for params {params}: {e}")
        return None


def _calculate_integer_loss(allowedK_integer, n_min=12, spacing_tol_factor=3,
                            w_slope=1.0, w_integer=0.0):
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    spacings       = np.diff(k)
    nu_sp          = 4.0
    median_spacing = np.median(spacings)
    mad            = np.median(np.abs(spacings - median_spacing))
    threshold      = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(k)
    for i in range(len(spacings) - 1, len(spacings) // 2, -1):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1

    k = k[:end_idx]
    if len(k) < n_min:
        return np.inf, np.inf, np.inf

    k_smooth       = (k[:-1] + k[1:]) / 2.0
    indices_smooth = np.arange(len(k_smooth), dtype=float) + 0.5

    z_sp       = np.linspace(0, 1, len(k_smooth) - 1)
    sp_weights = 1.0 - (1.0 - z_sp) ** 2
    sp_weights /= sp_weights.sum()

    z_int       = np.linspace(0, 1, len(k_smooth))
    int_weights = 1.0 - (1.0 - z_int) ** 2
    int_weights /= int_weights.sum()

    spacings_smooth  = np.diff(k_smooth)
    tail_slope_loss  = np.sum(sp_weights * (spacings_smooth - nu_sp) ** 2)

    weighted_mean_intercept = np.sum(int_weights * (k_smooth - nu_sp * indices_smooth))
    ideal_intercept         = np.round(weighted_mean_intercept)
    ideal_sequence          = ideal_intercept + nu_sp * indices_smooth
    tail_integer_loss       = np.sum(int_weights * (k_smooth - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss


# ── worker function ───────────────────────────────────────────────────────────
def _compute_1d_point(args):
    """Worker: compute losses for one point on the 1-D scan."""
    param_idx, xi_val, pt_idx = args

    params             = best_params.copy()
    params[param_idx]  = xi_val

    pid           = os.getpid()
    worker_folder = os.path.join(out_folder,
                                 f'tmp_p{param_idx}_pid{pid}_pt{pt_idx}', '')
    os.makedirs(worker_folder, exist_ok=True)

    allowedK = None
    try:
        allowedK                    = _calculate_allowedK(params.tolist(), worker_folder)
        slope_loss, int_loss, tloss = _calculate_integer_loss(allowedK)
    except Exception as e:
        print(f"  Error at param_idx={param_idx} pt={pt_idx}: {e}")
        slope_loss, int_loss, tloss = np.inf, np.inf, np.inf

    print(f"[param={param_names[param_idx]}] pt={pt_idx:3d}  "
          f"{param_names[param_idx]}={xi_val:.6f}  slope_loss={slope_loss:.4e}")
    k_arr = np.array(allowedK) if allowedK is not None else None
    return pt_idx, slope_loss, int_loss, tloss, k_arr


# ── non-daemon pool (allows compute_U_matrices to spawn its own children) ─────
class _NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class _NoDaemonPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return _NoDaemonProcess(*args, **kwargs)


# ── main ──────────────────────────────────────────────────────────────────────
N_1D = 30   # number of points along each parameter axis

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--param-index', type=int, default=None,
                        help='Parameter index 0..3. If omitted, run all parameters.')
    args = parser.parse_args()

    param_indices = range(4) if args.param_index is None else [args.param_index]

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {n_workers} parallel workers  (N_1D={N_1D} per parameter)\n")

    for param_idx in param_indices:
        out_file = os.path.join(out_folder, f'1d_param{param_idx}.npz')
        if os.path.exists(out_file):
            print(f"Skipping param {param_names[param_idx]} — already exists: {out_file}")
            continue

        xi    = np.linspace(*p_ranges[param_idx], N_1D)
        tasks = [(param_idx, xi[i], i) for i in range(N_1D)]

        print(f"=== Varying {param_names[param_idx]} "
              f"(idx={param_idx})  over [{p_ranges[param_idx][0]:.5f}, "
              f"{p_ranges[param_idx][1]:.5f}]  ({N_1D} points) ===")
        print(f"    Other params fixed at best-fit: "
              + ", ".join(f"{param_names[j]}={best_params[j]:.6f}"
                          for j in range(4) if j != param_idx))

        slope_loss_arr   = np.full(N_1D, np.inf)
        integer_loss_arr = np.full(N_1D, np.inf)
        total_loss_arr   = np.full(N_1D, np.inf)
        allowedK_arr     = np.empty(N_1D, dtype=object)

        with _NoDaemonPool(n_workers) as pool:
            for result in pool.imap_unordered(_compute_1d_point, tasks):
                pt_idx, sl, il, tl, k_arr = result
                slope_loss_arr[pt_idx]   = sl
                integer_loss_arr[pt_idx] = il
                total_loss_arr[pt_idx]   = tl
                allowedK_arr[pt_idx]     = k_arr

        np.savez(out_file,
                 xi=xi,
                 slope_loss=slope_loss_arr,
                 integer_loss=integer_loss_arr,
                 total_loss=total_loss_arr,
                 allowedK=allowedK_arr,
                 param_idx=param_idx)
        print(f"--> Saved: {out_file}\n")

    print("All parameters done.")
