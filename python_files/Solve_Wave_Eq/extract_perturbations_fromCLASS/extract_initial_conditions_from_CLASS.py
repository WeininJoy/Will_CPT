# -*- coding: utf-8 -*-
"""
Extract perturbation solutions at z=0 from CLASS including higher-order Boltzmann terms.
These will serve as initial conditions for backward integration to the FCB.
"""

from classy import Class
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SETUP
# =============================================================================

k_of_interest = [0.1, 0.01, 0.001]  # in h/Mpc

# Define cosmological parameters matching your model
params = {
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'h': 0.67,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'k_output_values': ','.join(map(str, k_of_interest)),
}

# =============================================================================
# RUN CLASS
# =============================================================================

cosmo = Class()
cosmo.set(params)
cosmo.compute()
print("CLASS computation complete.")

# =============================================================================
# EXTRACT PERTURBATIONS
# =============================================================================

perturbations = cosmo.get_perturbations()
scalar_perturbations = perturbations['scalar']

print(f"\nNumber of k-modes: {len(scalar_perturbations)}")

# =============================================================================
# EXTRACT VALUES AT z=0 (a=1.0) FOR EACH k-MODE
# =============================================================================

print("\n" + "="*80)
print("EXTRACTING INITIAL CONDITIONS AT z=0 FOR BACKWARD INTEGRATION")
print("="*80)

# Storage for all k-modes
initial_conditions = []

for i, k_data in enumerate(scalar_perturbations):
    print(f"\n{'-'*80}")

    # Get k-value - try different possible keys
    k_val = None
    for k_key in ['k (1/Mpc)', 'k [1/Mpc]', 'k', 'k (h/Mpc)', 'k [h/Mpc]']:
        if k_key in k_data:
            k_val = k_data[k_key][0] if hasattr(k_data[k_key], '__len__') else k_data[k_key]
            print(f"k-mode {i}: k = {k_val:.6f} (from key '{k_key}')")
            break

    if k_val is None:
        # If no k-value found, use the input value
        if i < len(k_of_interest):
            k_val = k_of_interest[i] * params['h']  # Convert h/Mpc to 1/Mpc
            print(f"k-mode {i}: k = {k_val:.6f} 1/Mpc = {k_of_interest[i]:.6f} h/Mpc (from input)")
        else:
            print(f"k-mode {i}: k-value not found")
            continue

    # Get all arrays
    tau = k_data['tau [Mpc]']
    a = k_data['a']

    # Find index closest to a=1.0 (z=0)
    idx_z0 = np.argmin(np.abs(a - 1.0))
    a_z0 = a[idx_z0]
    tau_z0 = tau[idx_z0]

    print(f"  Nearest point to z=0: a = {a_z0:.10f}, tau = {tau_z0:.4f} Mpc")
    print(f"  Index: {idx_z0} / {len(a)}")

    # Extract all available perturbation variables at z=0
    ic = {
        'k_1overMpc': k_val,
        'k_hover_Mpc': k_val * params['h'],
        'tau_Mpc': tau_z0,
        'a': a_z0,
        'z': 1.0/a_z0 - 1.0,
    }

    # Metric perturbations
    if 'phi' in k_data:
        ic['phi'] = k_data['phi'][idx_z0]
    if 'psi' in k_data:
        ic['psi'] = k_data['psi'][idx_z0]

    # Photon perturbations
    if 'delta_g' in k_data:
        ic['delta_g'] = k_data['delta_g'][idx_z0]
    if 'theta_g' in k_data:
        ic['theta_g'] = k_data['theta_g'][idx_z0]
    if 'shear_g' in k_data:
        ic['shear_g'] = k_data['shear_g'][idx_z0]  # This is Fr2
        ic['Fr2'] = k_data['shear_g'][idx_z0]      # Alias for clarity

    # Photon polarization (E-mode)
    if 'pol0_g' in k_data:
        ic['pol0_g'] = k_data['pol0_g'][idx_z0]
    if 'pol1_g' in k_data:
        ic['pol1_g'] = k_data['pol1_g'][idx_z0]
    if 'pol2_g' in k_data:
        ic['pol2_g'] = k_data['pol2_g'][idx_z0]

    # Baryon perturbations
    if 'delta_b' in k_data:
        ic['delta_b'] = k_data['delta_b'][idx_z0]
    if 'theta_b' in k_data:
        ic['theta_b'] = k_data['theta_b'][idx_z0]

    # CDM perturbations
    if 'delta_cdm' in k_data:
        ic['delta_cdm'] = k_data['delta_cdm'][idx_z0]
    if 'theta_cdm' in k_data:
        ic['theta_cdm'] = k_data['theta_cdm'][idx_z0]

    # Neutrino perturbations (if present)
    if 'delta_ur' in k_data:
        ic['delta_ur'] = k_data['delta_ur'][idx_z0]
    if 'theta_ur' in k_data:
        ic['theta_ur'] = k_data['theta_ur'][idx_z0]
    if 'shear_ur' in k_data:
        ic['shear_ur'] = k_data['shear_ur'][idx_z0]

    # Print extracted values
    print(f"\n  Extracted values at z=0:")
    print(f"    Metric:")
    print(f"      phi         = {ic.get('phi', np.nan):.10e}")
    print(f"      psi         = {ic.get('psi', np.nan):.10e}")
    print(f"    Photons:")
    print(f"      delta_g     = {ic.get('delta_g', np.nan):.10e}")
    print(f"      theta_g     = {ic.get('theta_g', np.nan):.10e}")
    print(f"      shear_g(Fr2)= {ic.get('Fr2', np.nan):.10e}")
    print(f"      pol0_g      = {ic.get('pol0_g', np.nan):.10e}")
    print(f"      pol1_g      = {ic.get('pol1_g', np.nan):.10e}")
    print(f"      pol2_g      = {ic.get('pol2_g', np.nan):.10e}")
    print(f"    Baryons:")
    print(f"      delta_b     = {ic.get('delta_b', np.nan):.10e}")
    print(f"      theta_b     = {ic.get('theta_b', np.nan):.10e}")
    print(f"    CDM:")
    print(f"      delta_cdm   = {ic.get('delta_cdm', np.nan):.10e}")
    print(f"      theta_cdm   = {ic.get('theta_cdm', np.nan):.10e}")
    if 'delta_ur' in ic:
        print(f"    Neutrinos:")
        print(f"      delta_ur    = {ic.get('delta_ur', np.nan):.10e}")
        print(f"      theta_ur    = {ic.get('theta_ur', np.nan):.10e}")
        print(f"      shear_ur    = {ic.get('shear_ur', np.nan):.10e}")

    initial_conditions.append(ic)

# =============================================================================
# SAVE INITIAL CONDITIONS
# =============================================================================

# Save as numpy file for easy loading
output_file = 'CLASS_initial_conditions_z0.npz'
np.savez(output_file,
         initial_conditions=initial_conditions,
         cosmological_params=params)

print("\n" + "="*80)
print(f"Initial conditions saved to: {output_file}")
print("="*80)

# =============================================================================
# CREATE MAPPING DOCUMENTATION
# =============================================================================

print("\n" + "="*80)
print("VARIABLE MAPPING: CLASS → Your Notation")
print("="*80)
print("""
CLASS Variable       Your Variable    Description
--------------       -------------    -----------
phi                  phi              Newtonian/Bardeen potential
psi                  psi              Curvature perturbation

Photons (radiation):
delta_g              dr               Photon density perturbation
theta_g              vr               Photon velocity perturbation
shear_g              Fr2              Photon shear (l=2 multipole)
pol0_g, pol1_g, ...  -                Polarization (E-mode) multipoles

Baryons:
delta_b              db               Baryon density perturbation
theta_b              vb               Baryon velocity perturbation

CDM:
delta_cdm            dm               CDM density perturbation
theta_cdm            vm               CDM velocity perturbation

Combined matter (for your model):
delta_m = (omega_b * delta_b + omega_cdm * delta_cdm) / (omega_b + omega_cdm)
theta_m = (omega_b * theta_b + omega_cdm * theta_cdm) / (omega_b + omega_cdm)

Notes:
1. CLASS uses conformal Newtonian gauge (phi, psi)
2. Velocity perturbations theta = (k/H) * v in CLASS notation
3. shear_g is the photon quadrupole (Fr2 in Boltzmann hierarchy)
4. Higher multipoles (Fr3, Fr4, ...) may not be directly output by CLASS
   but can be reconstructed if needed
5. For perfect fluid approximation: use delta_m, theta_m
6. For full Boltzmann: use delta_g, theta_g, shear_g (Fr2)
""")

# =============================================================================
# VISUALIZE AVAILABLE MULTIPOLES
# =============================================================================

print("\n" + "="*80)
print("AVAILABLE HIGHER-ORDER TERMS")
print("="*80)

# Check first k-mode for available variables
first_mode = scalar_perturbations[0]
print("\nAll available variables in CLASS output:")
for key in sorted(first_mode.keys()):
    if hasattr(first_mode[key], '__len__'):
        print(f"  {key:30s} : array of length {len(first_mode[key])}")

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

print("\n" + "="*80)
print("USAGE FOR BACKWARD INTEGRATION")
print("="*80)
print("""
To use these initial conditions for backward integration to FCB:

1. Load the saved file:
   data = np.load('CLASS_initial_conditions_z0.npz', allow_pickle=True)
   ics = data['initial_conditions'].item()

2. For each k-mode:
   ic = ics[k_index]  # Get initial condition dict for k-mode

3. Extract values for your backward solver:
   phi_0 = ic['phi']
   psi_0 = ic['psi']
   dr_0 = ic['delta_g']
   vr_0 = ic['theta_g']
   Fr2_0 = ic['Fr2']  # Higher-order term

   # For matter (combine baryons + CDM):
   omega_b = 0.022
   omega_cdm = 0.12
   omega_m = omega_b + omega_cdm
   dm_0 = (omega_b * ic['delta_b'] + omega_cdm * ic['delta_cdm']) / omega_m
   vm_0 = (omega_b * ic['theta_b'] + omega_cdm * ic['theta_cdm']) / omega_m

4. Set up your backward integration:
   - Start at tau = ic['tau_Mpc'], a = ic['a']
   - Integrate backwards in time to Big Bang, OR
   - Integrate forwards in time to FCB (a → ∞)

5. Note: CLASS gives velocities as theta = (k/H) * v
   You may need to convert: v = theta * H / k
""")

# =============================================================================
# CLEANUP
# =============================================================================

cosmo.struct_cleanup()
cosmo.empty()

print("\nDone!")
