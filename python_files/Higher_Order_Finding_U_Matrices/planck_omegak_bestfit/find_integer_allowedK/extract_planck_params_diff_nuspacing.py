"""
Extract p50 (median) parameter values from filtered Planck MCMC chains
for nu_spacing = [3, 4, 5, 6, 7, 8, 9].

For each nu_spacing, samples satisfying |DeltaK - nu_spacing| <= 0.02 * nu_spacing
are selected and their median (p50) is reported for: mt, kt, Omegab_ratio, h.
"""

import numpy as np
from anesthetic import read_chains
from scipy.interpolate import RegularGridInterpolator

print("="*80)
print("EXTRACTING p50 PARAMETERS FOR DIFFERENT nu_spacing VALUES")
print("="*80)

################
# Load DeltaK interpolator data
################

folder_path_low_kt  = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/low_kt/"
folder_path_high_kt = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/high_kt/"

mt_list    = np.linspace(350, 500, 20)
z_rec_list = np.linspace(1040, 1100, 20)
kt_list_low  = np.linspace(0, 1.8 + 1.8 / 19 * 10, 30)
DeltaK_arr_low_kt  = np.load(folder_path_low_kt  + 'DeltaK_arr.npy')
kt_list_high = np.logspace(np.log2(1.8 + 1.8 / 19 * 11), np.log2(30), num=20, base=2)
DeltaK_arr_high_kt = np.load(folder_path_high_kt + 'DeltaK_arr.npy')
kt_list    = np.concatenate((kt_list_low, kt_list_high))
DeltaK_arr = np.concatenate((DeltaK_arr_low_kt, DeltaK_arr_high_kt), axis=1)

for i in range(len(mt_list)):
    for j in range(len(kt_list)):
        for k in range(len(z_rec_list)):
            DeltaK_arr[i, j, k] = 1. / np.sqrt(kt_list[j]) * DeltaK_arr[i, j, k]

interpolate_Deltak = RegularGridInterpolator(
    (mt_list, kt_list, z_rec_list), DeltaK_arr,
    bounds_error=False, fill_value=np.nan
)

################
# Load Planck MCMC chains
################

Omega_gamma_h2 = 2.47e-5
Neff = 3.046

data_folder = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/'
print("\nLoading Planck MCMC chains...")
ns = read_chains(data_folder + 'Planck/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE')

# Add derived parameters
ns['omegarh2']     = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2
ns['omegar']       = ns['omegarh2'] / ns['H0']**2 * 100**2
ns['omegab']       = ns['omegabh2'] / ns['H0']**2 * 100**2
ns['kappa']        = -ns.omegak / np.sqrt(ns.omegal * ns.omegar) / 3.
ns['m']            = ns.omegam / (ns.omegal)**(1./4.) / (ns.omegar)**(3./4.)
ns['Omegab_ratio'] = ns['omegab'] / ns['omegam']
ns['h']            = ns['H0'] / 100

# Load precomputed DeltaK data
DeltaK_data = np.load(data_folder + 'Planck_anesthetic/DeltaK_data.npy')
ns['DeltaK'] = DeltaK_data

################
# Loop over nu_spacing values and extract p50
################

nu_spacing_list  = [3, 4, 5, 6, 7, 8, 9]
params_to_extract = ['m',  'kappa', 'Omegab_ratio', 'h']
param_labels      = ['mt', 'kt',    'Omegab_ratio', 'h']

results = {}  # results[nu_spacing][label] = p50

print("\n" + "="*80)
print("p50 (MEDIAN) PARAMETER VALUES PER nu_spacing")
print("="*80)
header = f"{'nu_spacing':<12}" + "".join(f"{lbl:<18}" for lbl in param_labels)
print(header)
print("-"*80)

for nu_spacing in nu_spacing_list:
    tolerance   = 0.02 * nu_spacing
    filtered_ns = ns[np.abs(ns['DeltaK'] - nu_spacing) <= tolerance]
    n_samples   = len(filtered_ns)

    if n_samples == 0:
        print(f"nu_spacing={nu_spacing}: NO samples found (tolerance={tolerance:.4f})")
        results[nu_spacing] = {lbl: np.nan for lbl in param_labels}
        continue

    p50_vals = {}
    for param, label in zip(params_to_extract, param_labels):
        p50_vals[label] = filtered_ns[param].quantile(0.50)
    results[nu_spacing] = p50_vals

    row = f"{nu_spacing:<12}" + "".join(f"{p50_vals[lbl]:<18.6f}" for lbl in param_labels)
    print(f"{row}  (N={n_samples})")

print("="*80)

################
# Save results to file
################

output_file = './planck_p50_diff_nuspacing.txt'
with open(output_file, 'w') as f:
    f.write("# p50 (median) parameter values from Planck MCMC chains filtered by nu_spacing\n")
    f.write("# Tolerance: |DeltaK - nu_spacing| <= 0.02 * nu_spacing\n")
    f.write("# Columns: nu_spacing  " + "  ".join(f"{lbl}" for lbl in param_labels) + "\n")
    for nu_spacing in nu_spacing_list:
        vals = results[nu_spacing]
        line = f"{nu_spacing}"
        for lbl in param_labels:
            line += f"  {vals[lbl]:.10f}"
        f.write(line + "\n")

print(f"\nResults saved to: {output_file}")
print("Done!")
