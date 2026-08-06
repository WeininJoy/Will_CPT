"""
Generate data for allowedK and integerK calculations using cosmological parameters.
"""
import numpy as np
import os
import time
import shutil
from scipy.optimize import root_scalar
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK
from Higher_Order_Finding_U_Matrices_TimeSeries_parallel import compute_U_matrices_timeseries

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
nu_spacing4_bestfit = [0.456043,-0.038752,0.155504,0.566647,2.095762,0.972835,0.053017] # cosmo_param_direc
cosmo_param_bool = True

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

def calculate_allowedK(params, z_rec, kvalues , folder_path, num_variables=200, cosmo_param_bool=False):
    compute_U_matrices(params, z_rec, kvalues, folder_path,num_variables, n_processes,cosmo_param_bool)
    compute_X_recs(params, z_rec, folder_path, num_variables,cosmo_param_bool)
    compute_allowedK(params, folder_path, num_variables,cosmo_param_bool)
    
def calculate_Umatrices_Xrec(params, z_rec, kvalues, folder_path,
                              num_variables=200, num_variables_save=75, cosmo_param_bool=False):
    compute_U_matrices(params, z_rec, kvalues, folder_path,num_variables, n_processes, num_variables_save, cosmo_param_bool)
    compute_X_recs(params, z_rec, folder_path, num_variables, cosmo_param_bool)


def merge_data_chunks(chunk_folders, output_folder):
    """
    Concatenate static U-matrix data from per-range chunk folders along the
    k-axis (axis 0) into a single output folder.

    All chunks must have been saved with the same num_variables_save so that
    ABCmatrices, DEFmatrices, GHIvectors share the same non-k dimensions.
    X1matrices and X2matrices have shape (N_k, 6, 4) and are always compatible.
    """
    os.makedirs(output_folder, exist_ok=True)
    filenames = ['L70_kvalues', 'L70_ABCmatrices', 'L70_DEFmatrices',
                 'L70_GHIvectors', 'L70_X1matrices', 'L70_X2matrices',
                 'L70_recValues']
    for fname in filenames:
        arrays = [np.load(os.path.join(f, fname + '.npy'))
                  for f in chunk_folders
                  if os.path.exists(os.path.join(f, fname + '.npy'))]
        if arrays:
            np.save(os.path.join(output_folder, fname), np.concatenate(arrays, axis=0))
    print(f"Merged {len(chunk_folders)} static-data chunks → {output_folder}")


def merge_timeseries_chunks(chunk_folders, output_folder):
    """
    Concatenate timeseries data from per-range chunk folders along the k-axis
    (axis 0) into a single output folder.

    All chunks must share the same t_grid — ensured by passing
    k_for_endtime=global_k_min to compute_U_matrices_timeseries for every chunk.
    The t_grid from the first chunk is used as the common grid.
    """
    os.makedirs(output_folder, exist_ok=True)
    t_grid = np.load(os.path.join(chunk_folders[0], 't_grid.npy'))
    np.save(os.path.join(output_folder, 't_grid.npy'), t_grid)

    filenames = ['L70_kvalues', 'L70_ABC_solutions',
                 'L70_DEF_solutions', 'L70_GHI_solutions']
    for fname in filenames:
        arrays = [np.load(os.path.join(f, fname + '.npy'))
                  for f in chunk_folders
                  if os.path.exists(os.path.join(f, fname + '.npy'))]
        if arrays:
            np.save(os.path.join(output_folder, fname), np.concatenate(arrays, axis=0))
    print(f"Merged {len(chunk_folders)} timeseries chunks → {output_folder}")


######################################
# Main Execution
######################################
params = nu_spacing4_bestfit[:4] # [mt, kt, omega_b_ratio, h]
discreteK_type = ['integerK', 'allowedK']
folder =  './cosmo_param_direc/'
z_rec = calculate_z_rec(params)

########### calculate allowedK ##############
kvalues_all_k50_65 = np.linspace(50,65,num=200); # data_all_k50-65, num_variables = 200
kvalues_all_k30_50 = np.linspace(30,50,num=200); # data_all_k30-50, num_variables = 160
kvalues_all_k0_30 = np.linspace(1e-5,30,num=300); # data_all_k0-30, num_variables = 75
kvalues_small_k = np.linspace(1e-5,3,num=100); # data_small_k, num_variables = 75
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k50_65, folder_path=folder+'data_all_k50-65/', num_variables=200, cosmo_param_bool=cosmo_param_bool)
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k30_50, folder_path=folder+'data_all_k30-50/', num_variables=160, cosmo_param_bool=cosmo_param_bool)
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k0_30, folder_path=folder+'data_all_k0-30/', num_variables=75, cosmo_param_bool=cosmo_param_bool)
calculate_allowedK(params, z_rec, kvalues=kvalues_small_k, folder_path=folder+'data_small_k/', num_variables=75, cosmo_param_bool=cosmo_param_bool)
print("AllowedK calculation completed.")

# ########### generate discreteK data ##############

# ## kvalues based on allowedK
# small_allowedK = np.load(folder + 'data_small_k/allowedK.npy')
# allowedK_path = folder + 'data_all_k0-30/allowedK.npy'
# kvalues_allowedK = np.load(allowedK_path); # only caluclate for the allowed K values
# kvalues_allowedK = kvalues_allowedK[kvalues_allowedK > small_allowedK[-1] + 0.5* np.diff(kvalues_allowedK).mean()]  # remove k values that are already in small_allowedK
# kvalues_allowedK = np.concatenate((small_allowedK[-3:], kvalues_allowedK))  # add a few small K values for better resolution at low k
# kvalues_allowedK_k30_50 = np.load(folder + 'data_all_k30-50/allowedK.npy');
# kvalues_allowedK_k50_65 = np.load(folder + 'data_all_k50-65/allowedK.npy');
# kvalues_allowedK = np.concatenate((kvalues_allowedK, kvalues_allowedK_k30_50, kvalues_allowedK_k50_65))
# print('kvalues_allowedK:', kvalues_allowedK)

# ## kvalues based on allowedK_integer
# small_allowedK_integer = np.load(folder + 'data_small_k/allowedK_integer.npy')
# allowedK_integer_path = folder + 'data_all_k0-30/allowedK_integer.npy'
# allowedK_integer = np.load(allowedK_integer_path);
# allowedK_integer = allowedK_integer[allowedK_integer > small_allowedK_integer[-1] + 0.5* np.diff(allowedK_integer).mean()]  # remove k values that are already in small_allowedK
# allowedK_integer = np.concatenate((small_allowedK_integer[-3:], allowedK_integer))  # add a few small K values for better resolution at low k
# allowedK_integer_k30_50 = np.load(folder + 'data_all_k30-50/allowedK_integer.npy');
# allowedK_integer_k50_65 = np.load(folder + 'data_all_k50-65/allowedK_integer.npy');
# allowedK_integer = np.concatenate((allowedK_integer, allowedK_integer_k30_50, allowedK_integer_k50_65))
# print('allowedK_integer:', allowedK_integer)
# n_ignore = 10
# high_k = np.array(allowedK_integer[n_ignore:])
# indices = np.arange(len(high_k))
# slope, intercept = np.polyfit(indices, high_k, 1)
# ideal_spacing = np.round(slope)
# if ideal_spacing != nu_spacing: print("Warning: ideal spacing does not match nu_spacing!")
# ideal_intercept = np.mean(high_k - ideal_spacing * indices)
# ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)
# small_integer_k_sequence = [ideal_k_sequence[0]-nu_spacing*i for i in range(1, n_ignore+1)][::-1]
# whole_integer_k_sequence = np.concatenate((small_integer_k_sequence, ideal_k_sequence))
# print("Ideal integer k sequence (after fitting):", whole_integer_k_sequence)
# mt, kt, omega_b_ratio, h = params
# s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt, kt, h) 
# H0 = 1/np.sqrt(3*Omega_lambda); #we are working in units of Lambda=c=1
# a0=1; K=-Omega_K * a0**2 * H0**2
# kvalues_integerK = [k * np.sqrt(K) for k in whole_integer_k_sequence]
# print('kvalues_integerK:', kvalues_integerK)

# ---------------------------------------------------------------------------
# Per-k-range computation with appropriate num_variables for ODE accuracy.
# All chunks use the same num_variables_save=75 so the saved arrays have
# uniform shape and can be concatenated by the merge helpers below.
#
# num_variables guidelines (increase with k for better multipole convergence):
#   small_k  (k <= 3)   → 75  is sufficient
#   k0_30    (k <= 30)  → 100
#   k30_50   (k <= 50)  → 150
#   k50_65   (k <= 65)  → 200
# ---------------------------------------------------------------------------

# # Split allowedK values by k-range (they are already ordered)
# kvals_allowedK_small = kvalues_allowedK[kvalues_allowedK <= 3]
# kvals_allowedK_0_30  = kvalues_allowedK[(kvalues_allowedK > 3)  & (kvalues_allowedK <= 30)]
# kvals_allowedK_30_50 = kvalues_allowedK[(kvalues_allowedK > 30) & (kvalues_allowedK <= 50)]
# kvals_allowedK_50_65 = kvalues_allowedK[kvalues_allowedK > 50]

# # Split integerK values by k-range
# kvalues_integerK = np.array(kvalues_integerK)
# kvals_integerK_small = kvalues_integerK[kvalues_integerK <= 3]
# kvals_integerK_0_30  = kvalues_integerK[(kvalues_integerK > 3)  & (kvalues_integerK <= 30)]
# kvals_integerK_30_50 = kvalues_integerK[(kvalues_integerK > 30) & (kvalues_integerK <= 50)]
# kvals_integerK_50_65 = kvalues_integerK[kvalues_integerK > 50]

# # Global k_min used to keep t_grid consistent across all timeseries chunks
# k_min_global_allowedK = float(np.min(kvalues_allowedK))
# k_min_global_integerK = float(np.min(kvalues_integerK))

# # (num_variables, num_variables_save) pairs per range
# range_nv = [(75, 75), (100, 75), (150, 75), (200, 75)]

# # --- allowedK: static matrices + Xrec ---
# allowedK_chunk_configs = [
#     (kvals_allowedK_small, folder + 'data_allowedK_chunks/small_k/'),
#     (kvals_allowedK_0_30,  folder + 'data_allowedK_chunks/k0_30/'),
#     (kvals_allowedK_30_50, folder + 'data_allowedK_chunks/k30_50/'),
#     (kvals_allowedK_50_65, folder + 'data_allowedK_chunks/k50_65/'),
# ]
# for (kvals, chunk_folder), (nv, nv_save) in zip(allowedK_chunk_configs, range_nv):
#     if len(kvals) > 0:
#         calculate_Umatrices_Xrec(params, z_rec, kvals, chunk_folder,
#                                   num_variables=nv, num_variables_save=nv_save)

# # --- integerK: static matrices + Xrec ---
# integerK_chunk_configs = [
#     (kvals_integerK_small, folder + 'data_integerK_chunks/small_k/'),
#     (kvals_integerK_0_30,  folder + 'data_integerK_chunks/k0_30/'),
#     (kvals_integerK_30_50, folder + 'data_integerK_chunks/k30_50/'),
#     (kvals_integerK_50_65, folder + 'data_integerK_chunks/k50_65/'),
# ]
# for (kvals, chunk_folder), (nv, nv_save) in zip(integerK_chunk_configs, range_nv):
#     if len(kvals) > 0:
#         calculate_Umatrices_Xrec(params, z_rec, kvals, chunk_folder,
#                                   num_variables=nv, num_variables_save=nv_save)

# # Merge chunks into the final unified folders
# allowedK_static_chunks = [f for (_, f) in allowedK_chunk_configs
#                            if os.path.exists(os.path.join(f, 'L70_kvalues.npy'))]
# integerK_static_chunks = [f for (_, f) in integerK_chunk_configs
#                            if os.path.exists(os.path.join(f, 'L70_kvalues.npy'))]
# merge_data_chunks(allowedK_static_chunks, folder + 'data_allowedK/')
# merge_data_chunks(integerK_static_chunks, folder + 'data_integerK/')
# print("U matrices and Xrec calculation completed.")


# ########### generate TimeSeries U matrices ##############
# # --- allowedK timeseries ---
# allowedK_ts_chunk_configs = [
#     (kvals_allowedK_small, folder + 'data_allowedK_chunks/small_k_timeseries/'),
#     (kvals_allowedK_0_30,  folder + 'data_allowedK_chunks/k0_30_timeseries/'),
#     (kvals_allowedK_30_50, folder + 'data_allowedK_chunks/k30_50_timeseries/'),
#     (kvals_allowedK_50_65, folder + 'data_allowedK_chunks/k50_65_timeseries/'),
# ]
# for (kvals, chunk_folder), (nv, nv_save) in zip(allowedK_ts_chunk_configs, range_nv):
#     if len(kvals) > 0:
#         compute_U_matrices_timeseries(params, z_rec, kvals, chunk_folder,
#                                       num_variables=nv, num_variables_save=nv_save,
#                                       k_for_endtime=k_min_global_allowedK)

# # --- integerK timeseries ---
# integerK_ts_chunk_configs = [
#     (kvals_integerK_small, folder + 'data_integerK_chunks/small_k_timeseries/'),
#     (kvals_integerK_0_30,  folder + 'data_integerK_chunks/k0_30_timeseries/'),
#     (kvals_integerK_30_50, folder + 'data_integerK_chunks/k30_50_timeseries/'),
#     (kvals_integerK_50_65, folder + 'data_integerK_chunks/k50_65_timeseries/'),
# ]
# for (kvals, chunk_folder), (nv, nv_save) in zip(integerK_ts_chunk_configs, range_nv):
#     if len(kvals) > 0:
#         compute_U_matrices_timeseries(params, z_rec, kvals, chunk_folder,
#                                       num_variables=nv, num_variables_save=nv_save,
#                                       k_for_endtime=k_min_global_integerK)

# # Merge timeseries chunks
# allowedK_ts_chunks = [f for (_, f) in allowedK_ts_chunk_configs
#                        if os.path.exists(os.path.join(f, 't_grid.npy'))]
# integerK_ts_chunks = [f for (_, f) in integerK_ts_chunk_configs
#                        if os.path.exists(os.path.join(f, 't_grid.npy'))]
# merge_timeseries_chunks(allowedK_ts_chunks, folder + 'data_allowedK_timeseries/')
# merge_timeseries_chunks(integerK_ts_chunks, folder + 'data_integerK_timeseries/')
# print("TimeSeries U matrices calculation completed.")
# print("All data generation completed.")



# ########### generate data for high-k range (chunked processing) ##############
# # Load existing k values to determine starting point
# highk_folder = folder + 'data_highK_integerK/'
# highk_kvalues_file = os.path.join(highk_folder, 'L70_kvalues.npy')

# if os.path.exists(highk_kvalues_file):
#     existing_kvalues = np.load(highk_kvalues_file)
#     last_k_physical = existing_kvalues[-1]
#     last_k_integer = last_k_physical / np.sqrt(K)
#     print(f"Found existing data. Last k (physical): {last_k_physical:.6f}, Last k (integer space): {last_k_integer:.6f}")
#     start_k_integer = last_k_integer + nu_spacing
# else:
#     start_k_integer = whole_integer_k_sequence[-1] + nu_spacing
#     print(f"No existing data found. Starting from whole_integer_k_sequence[-1]: {whole_integer_k_sequence[-1]}")

# # Generate 3000 k values
# num_kvalues = 3000
# end_k_integer = start_k_integer + nu_spacing * (num_kvalues - 1)
# all_k_integers = np.linspace(start_k_integer, end_k_integer, num=num_kvalues)
# all_kvalues_physical = [k * np.sqrt(K) for k in all_k_integers]
# print(f"Total k range: {all_kvalues_physical[0]:.6f} to {all_kvalues_physical[-1]:.6f}")

# # Process in chunks to avoid memory issues
# chunk_size = 300
# num_chunks = int(np.ceil(num_kvalues / chunk_size))
# print(f"\nProcessing {num_kvalues} k values in {num_chunks} chunks of {chunk_size}")

# for chunk_idx in range(num_chunks):
#     start_idx = chunk_idx * chunk_size
#     end_idx = min((chunk_idx + 1) * chunk_size, num_kvalues)
#     chunk_kvalues = all_kvalues_physical[start_idx:end_idx]

#     print(f"\n{'='*70}")
#     print(f"Processing chunk {chunk_idx+1}/{num_chunks}")
#     print(f"K values {start_idx} to {end_idx-1} (total: {len(chunk_kvalues)})")
#     print(f"K range: {chunk_kvalues[0]:.6f} to {chunk_kvalues[-1]:.6f}")
#     print(f"{'='*70}")

#     # Create chunk-specific folder names
#     chunk_folder = folder + f'data_highK_integerK/chunk_{chunk_idx:02d}/'
#     chunk_timeseries_folder = folder + f'data_highK_integerK_timeseries/chunk_{chunk_idx:02d}/'

#     # Compute and save chunk
#     print(f"\nComputing U matrices and Xrec for chunk {chunk_idx+1}...")
#     calculate_Umatrices_Xrec(params, z_rec, chunk_kvalues, folder_path=chunk_folder)

#     print(f"\nComputing timeseries for chunk {chunk_idx+1}...")
#     compute_U_matrices_timeseries(params, z_rec, chunk_kvalues, folder_path=chunk_timeseries_folder, n_processes=n_processes)

#     print(f"Chunk {chunk_idx+1}/{num_chunks} completed and saved.")

# print(f"\n{'='*70}")
# print(f"All {num_chunks} chunks completed!")
# print(f"Data saved in: {folder}data_highK_integerK/chunk_XX/")
# print(f"Timeseries saved in: {folder}data_highK_integerK_timeseries/chunk_XX/")
# print(f"Use the merge script to combine chunks if needed.")
# print(f"{'='*70}")