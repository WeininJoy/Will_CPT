"""
Generate allowedK data for nu_spacing = [3,4,5,6,7,8,9].
[mt, kt, Omegab_ratio, h] are read from ./planck_p50_diff_nuspacing.txt (p50 per nu_spacing).
A_s, n_s, tau_reio are fixed (they do not affect z_rec significantly).
"""
import sys
import numpy as np
import os
from scipy.optimize import root_scalar
import classy
from classy import CosmoComputationError
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append('../')
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK
from Higher_Order_Finding_U_Matrices_TimeSeries_parallel import compute_U_matrices_timeseries

nu_spacing_list = [3, 4, 5, 6, 7, 8, 9]

# Detect number of CPUs
n_processes = 4
print(f"Using {n_processes} parallel processes")

#################
# Fixed parameters
#################
rt = 1
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
N_ncdm = 1
m_ncdm = 0.06
# Fixed CMB parameters (same for all nu_spacing)
A_s_fixed      = 2.043034e-9
n_s_fixed      = 0.972175
tau_reio_fixed = 0.045755

################
# Load p50 parameters from file
################
p50_file = './planck_p50_diff_nuspacing.txt'
print(f"Loading p50 parameters from: {p50_file}")
p50_data = np.loadtxt(p50_file, comments='#')
# columns: nu_spacing, mt, kt, Omegab_ratio, h
p50_dict = {int(row[0]): row[1:] for row in p50_data}  # {nu_spacing: [mt, kt, Omegab_ratio, h]}

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
        'Omega_cdm': (1.-omega_b_ratio)*Omega_m,
        'Omega_k': float(Omega_K),
        'A_s': A_s_fixed,
        'n_s': n_s_fixed,
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'tau_reio': tau_reio_fixed,
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

######################################
# Main Loop
######################################
base_folder = './data_diff_nu_spacing/'
os.makedirs(base_folder, exist_ok=True)

for nu_spacing in nu_spacing_list:
    print("="*60)
    print(f"Processing nu_spacing = {nu_spacing}")

    if nu_spacing not in p50_dict:
        print(f"  WARNING: nu_spacing={nu_spacing} not found in {p50_file}. Skipping.")
        continue

    mt, kt, Omegab_ratio, h = p50_dict[nu_spacing]
    params = [mt, kt, Omegab_ratio, h]
    print(f"  params: mt={mt:.6f}, kt={kt:.6f}, Omegab_ratio={Omegab_ratio:.6f}, h={h:.6f}")

    folder = base_folder + f'nu_{nu_spacing}/'
    os.makedirs(folder, exist_ok=True)

    z_rec = calculate_z_rec(params)
    print(f"  z_rec = {z_rec:.4f}")

    ########### calculate allowedK ##############
    kvalues_all_k   = np.linspace(1e-5, 22, num=200)  # data_all_k
    kvalues_small_k = np.linspace(1e-5,  3, num=100)  # data_small_k
    calculate_allowedK(params, z_rec, kvalues=kvalues_all_k,   folder_path=folder+'data_all_k/')
    calculate_allowedK(params, z_rec, kvalues=kvalues_small_k, folder_path=folder+'data_small_k/')
    print(f"  AllowedK calculation completed.")

    ########### build kvalues_allowedK ##############
    small_allowedK   = np.load(folder + 'data_small_k/allowedK.npy')
    kvalues_allowedK = np.load(folder + 'data_all_k/allowedK.npy')
    kvalues_allowedK = kvalues_allowedK[kvalues_allowedK > small_allowedK[-1] + 0.5 * np.diff(kvalues_allowedK).mean()]
    kvalues_allowedK = np.concatenate((small_allowedK[-3:], kvalues_allowedK))
    print(f"  kvalues_allowedK: {kvalues_allowedK}")
    np.save(folder + 'kvalues_allowedK.npy', kvalues_allowedK)

print("="*60)
print("All nu_spacing calculations completed.")
