"""
Extract 1-sigma parameter bounds from filtered Planck data (nu_spacing=4)
to use as pbounds in Bayesian optimization.
"""

import numpy as np
from anesthetic import read_chains
from scipy.interpolate import RegularGridInterpolator

print("="*80)
print("EXTRACTING PARAMETER BOUNDS FROM FILTERED PLANCK DATA")
print("="*80)

################
# Load DeltaK interpolator data
################

folder_path_low_kt = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/low_kt/"
folder_path_high_kt = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/high_kt/"

mt_list = np.linspace(350, 500, 20)
z_rec_list = np.linspace(1040, 1100, 20)
kt_list_low = np.linspace(0, 1.8 + 1.8 / 19 * 10, 30)
DeltaK_arr_low_kt = np.load(folder_path_low_kt + 'DeltaK_arr.npy')
kt_list_high = np.logspace(np.log2(1.8 + 1.8 / 19 * 11), np.log2(30), num=20, base=2)
DeltaK_arr_high_kt = np.load(folder_path_high_kt + 'DeltaK_arr.npy')
kt_list = np.concatenate((kt_list_low, kt_list_high))
DeltaK_arr = np.concatenate((DeltaK_arr_low_kt, DeltaK_arr_high_kt), axis=1)

for i in range(len(mt_list)):
    for j in range(len(kt_list)):
        for k in range(len(z_rec_list)):
            DeltaK_arr[i,j,k] = 1./ np.sqrt(kt_list[j]) * DeltaK_arr[i,j,k]

interpolate_Deltak = RegularGridInterpolator((mt_list, kt_list, z_rec_list), DeltaK_arr, bounds_error=False, fill_value=np.nan)

################
# Load Planck MCMC chains
################

Omega_gamma_h2 = 2.47e-5
Neff = 3.046

data_folder = '/home/wnd22/rds/hpc-work/Will_CPT/CMB_CLASS_integer/data/'
print("\nLoading Planck MCMC chains...")
ns = read_chains(data_folder+'Planck/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE')

# Add derived parameters
ns['omegarh2'] = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2
ns['omegar'] = ns['omegarh2'] / ns['H0']**2 * 100**2
ns['omegab'] = ns['omegabh2'] / ns['H0']**2 * 100**2
ns['kappa'] = - ns.omegak / np.sqrt(ns.omegal*ns.omegar) / 3.
ns['m'] = ns.omegam / (ns.omegal)**(1./4.) / (ns.omegar)**(3./4.)

# Load precomputed DeltaK data
DeltaK_data = np.load(data_folder + 'Planck_anesthetic/DeltaK_data.npy')
ns['DeltaK'] = DeltaK_data

################
# Filter for nu_spacing = 4
################

nu_spacing = 4
tolerance = 0.02 * nu_spacing
filtered_ns = ns[np.abs(ns['DeltaK'] - nu_spacing) <= tolerance]
print(f"Filtered Planck data: {len(filtered_ns)} samples satisfy |DeltaK - {nu_spacing}| <= {tolerance}")

if filtered_ns.empty:
    print("ERROR: No samples satisfy the constraint. Cannot extract bounds.")
    exit(1)

################
# Calculate Omegab_ratio for filtered data
################

filtered_ns['Omegab_ratio'] = filtered_ns['omegab'] / filtered_ns['omegam']
filtered_ns['h'] = filtered_ns['H0'] / 100

################
# Extract 2-sigma bounds (2.5th and 97.5th percentiles)
################

params_to_extract = ['m', 'kappa', 'Omegab_ratio', 'h']
param_labels = ['mt', 'kt', 'Omegab_ratio', 'h']

print("\n" + "="*80)
print("2-SIGMA PARAMETER BOUNDS FROM FILTERED PLANCK DATA")
print("="*80)

pbounds = {}

for param, label in zip(params_to_extract, param_labels):
    p2_5 = filtered_ns[param].quantile(0.025)
    p50 = filtered_ns[param].quantile(0.50)
    p97_5 = filtered_ns[param].quantile(0.975)

    lower_bound = p2_5
    upper_bound = p97_5

    pbounds[label] = (lower_bound, upper_bound)

    print(f"{label:<15}: [{lower_bound:.6f}, {upper_bound:.6f}]  (median = {p50:.6f})")

print("="*80)

################
# Also show 1-sigma and 3-sigma bounds for reference
################

print("\nFor reference, 1-sigma bounds (16th - 84th percentiles):")
for param, label in zip(params_to_extract, param_labels):
    p16 = filtered_ns[param].quantile(0.16)
    p84 = filtered_ns[param].quantile(0.84)
    print(f"{label:<15}: [{p16:.6f}, {p84:.6f}]")

print("\nFor reference, 3-sigma bounds (0.15th - 99.85th percentiles):")
for param, label in zip(params_to_extract, param_labels):
    p0_15 = filtered_ns[param].quantile(0.0015)
    p99_85 = filtered_ns[param].quantile(0.9985)
    print(f"{label:<15}: [{p0_15:.6f}, {p99_85:.6f}]")

################
# Generate Python code snippet for pbounds
################

print("\n" + "="*80)
print("PYTHON CODE SNIPPET FOR find_intK_baye_opt.py")
print("="*80)
print("\n# Parameter bounds from filtered Planck data (nu_spacing=4, 2-sigma)")
print("pbounds = {")
for label, (lower, upper) in pbounds.items():
    print(f"    '{label}': ({lower:.6f}, {upper:.6f}),")
print("}")
print("\n" + "="*80)

################
# Save bounds to file
################

output_file = './planck_filtered_pbounds.txt'
with open(output_file, 'w') as f:
    f.write("# Parameter bounds from filtered Planck data (nu_spacing=4, 2-sigma)\n")
    f.write("# Format: parameter lower_bound upper_bound median\n")
    for param, label in zip(params_to_extract, param_labels):
        p2_5 = filtered_ns[param].quantile(0.025)
        p50 = filtered_ns[param].quantile(0.50)
        p97_5 = filtered_ns[param].quantile(0.975)
        f.write(f"{label} {p2_5:.10f} {p97_5:.10f} {p50:.10f}\n")

print(f"\nBounds saved to: {output_file}")
print("\nDone!")
