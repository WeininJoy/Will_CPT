from classy import Class
import classy
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

nu_spacing = 4
data_folder_path = './data/'
figures_folder_path = './figures/'

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density 
Neff = 3.046
N_ncdm = 1  # number of massive neutrino species
m_ncdm = 0.06  # mass of massive neutrino species in e
k_of_interest = [0.001, 0.01, 0.05, 0.1, 0.2]  # in h/Mpc

def cosmological_parameters(mt, kt, h): 

    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(Omega_r, rt, mt, kt)
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return Omega_lambda, Omega_m, Omega_K

###############################################################################
# Best-fit parameters from nu_spacing=4 (first try)
# mt, kt, Omegab_ratio, h = 401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915 # based on the calculate_z_rec() output
print(f"Recombination redshift z_rec = {z_rec}") 
###############################################################################

# 1. Define your cosmological parameters
# CLASS will automatically integrate perturbations as far as needed
# By default, it evolves to z~0 (today). To see further evolution,
# CLASS limits are set by the background evolution parameters.


# Specify arbitrary k values you want perturbation solutions for
# These do NOT need to satisfy the quantization condition!

# # Output settings
# write parameters = yes
# write background = yes

params = {
    # IMPORTANT: For closed universe with arbitrary k values,
    # request ONLY transfer functions, NOT CMB power spectra (tCl, pCl, etc.)
    # This bypasses the hyperspherical Bessel computation that requires quantized k
    'output': 'dTk, vTk',  # Density and velocity transfer functions
    'modes': 's',           # Scalar perturbations
    'ic': 'ad',            # Adiabatic initial conditions
    'gauge': 'newtonian',  # Newtonian gauge
    'a_final_over_a_today': 100.0,  # Start with a = 100 to test (can increase later)

    # Cosmological parameters
    'h': h,
    'Omega_b': Omegab_ratio*OmegaM,
    'Omega_cdm': (1.-Omegab_ratio)*OmegaM,
    'Omega_k': float(OmegaK),
    'A_s': As*1e-9,
    'n_s': ns,
    'tau_reio': tau,

    # CRITICAL: For closed universe, must specify nu_spacing
    'nu_spacing': nu_spacing,

    # Limit k-range to prevent overflow in primordial k-grid calculation
    'P_k_max_1/Mpc': max(k_of_interest) * 1.5,  # Set max k slightly above our highest k

    # CRITICAL: Precision parameters to prevent too many k-modes when extending to far future
    # These control the k-grid sampling and prevent memory overflow
    # Since we only care about k_output_values, make the internal k-grid VERY sparse
    'k_step_sub': 2.0,           # Much larger step = far fewer k-modes (default: 0.05)
    'k_step_super': 0.1,         # Much larger step = far fewer k-modes (default: 0.002)
    'k_per_decade_for_pk': 2.0,  # Minimal k per decade (default: 10.0)
    'k_per_decade_for_bao': 2.0, # Minimal k in BAO region (default: 70.0)

    # Precision parameters to control multipole hierarchy
    'l_max_g': 50,        # Maximum photon temperature multipole (default: 12)
    'l_max_pol_g': 50,    # Maximum photon polarization multipole (default: 10)

    # Massive neutrinos (optional - uncomment if needed)
    # 'N_ncdm': N_ncdm,      # number of massive neutrino species
    # 'm_ncdm': m_ncdm,      # mass in eV
    # 'N_ur': Neff - N_ncdm, # Effective number of MASSLESS neutrino species

    # Arbitrary k values - no quantization required for perturbations-only output!
    'k_output_values': ','.join(map(str, k_of_interest)),
}

# 3. Create an instance of the CLASS wrapper and set the parameters
cosmo = Class()
cosmo.set(params)

# 4. Run the code
cosmo.compute()
print("CLASS run complete.")

# --- 3. Extract the perturbation data ---
perturbations = cosmo.get_perturbations()

# The perturbations dictionary structure: perturbations['scalar'] is a list
print("\n" + "="*80)
print("EXTRACTING PERTURBATIONS")
print("="*80)
print("Perturbation types:", perturbations.keys())
scalar_perturbations = perturbations['scalar']
print(f"Number of k-modes extracted: {len(scalar_perturbations)}")

# Check that we got the expected number of k modes
if len(scalar_perturbations) != len(k_of_interest):
    print(f"WARNING: Expected {len(k_of_interest)} k modes but got {len(scalar_perturbations)}")

# Display info for each k-mode
print("\nk-modes extracted:")
for i, k_data in enumerate(scalar_perturbations):
    if 'k (1/Mpc)' in k_data:
        k_val = k_data['k (1/Mpc)'][0]
        print(f"  k-mode {i}: k = {k_val:.6e} 1/Mpc (requested: {k_of_interest[i]:.6e})")
    else:
        print(f"  k-mode {i}: (k value not found in output)")

# Get the first k-mode for detailed analysis
one_k_perturbations = scalar_perturbations[0]
k_first = one_k_perturbations['k (1/Mpc)'][0] if 'k (1/Mpc)' in one_k_perturbations else k_of_interest[0]

print(f"\nAnalyzing first k-mode: k = {k_first:.6e} 1/Mpc")
print("Available variables:", list(one_k_perturbations.keys())[:10], "...")

# Extract the variables of interest from first k-mode
tau = one_k_perturbations['tau [Mpc]']
a = one_k_perturbations['a']
phi = one_k_perturbations['phi']
psi = one_k_perturbations['psi']
delta_cdm = one_k_perturbations['delta_cdm']
delta_b = one_k_perturbations['delta_b']
delta_g = one_k_perturbations['delta_g']  # Photon density (l=0)
theta_cdm = one_k_perturbations['theta_cdm']
theta_b = one_k_perturbations['theta_b']
theta_g = one_k_perturbations['theta_g']  # Photon velocity (l=1)
shear_g = one_k_perturbations['shear_g']  # Photon shear (l=2, Fr2)

# Extract higher photon multipoles Fr3, Fr4, Fr5, ...
photon_multipoles = {}
for key in one_k_perturbations.keys():
    if key.startswith('Fr') and key.endswith('_g'):
        # Extract the multipole number from the key (e.g., 'Fr3_g' -> 3)
        l_value = int(key[2:-2])
        photon_multipoles[l_value] = one_k_perturbations[key]

print(f"\nPerturbation evolution (first k-mode):")
print(f"  Number of time steps: {len(tau)}")
print(f"  tau range: [{tau[0]:.4e}, {tau[-1]:.4e}] Mpc")
print(f"  a range: [{a[0]:.4e}, {a[-1]:.4e}]")
print(f"  s=1/a range: [{1/a[-1]:.4e}, {1/a[0]:.4e}]")

# Display available higher multipoles
if photon_multipoles:
    print(f"\nHigher photon multipoles available:")
    for l in sorted(photon_multipoles.keys()):
        print(f"  Fr{l}_g (l={l}): {len(photon_multipoles[l])} time steps")
    print(f"  Total multipoles: l=0 to l={max(photon_multipoles.keys())}")
else:
    print(f"\nNo higher multipoles (Fr3_g, Fr4_g, ...) found in output.")
    print(f"  Make sure CLASS was compiled with the modified code.")

# --- 3b. Save data for all k-modes ---
print("\n" + "="*80)
print("SAVING DATA")
print("="*80)

for i, k_data in enumerate(scalar_perturbations):
    # Extract k value
    if 'k (1/Mpc)' in k_data:
        k_val = k_data['k (1/Mpc)'][0]
        print(f"\nSaving k-mode {i}: k = {k_val:.6e} 1/Mpc")
    else:
        k_val = k_of_interest[i]
        print(f"\nSaving k-mode {i}: k ~ {k_val:.6e} 1/Mpc (estimated)")

    # Save as numpy file
    filename = f'{data_folder_path}perturbations_k{i}_kval{k_val:.6e}.npz'
    np.savez(filename, **k_data)
    print(f"  Saved to: {filename}")
    print(f"  Contains {len(k_data)} variables with {len(k_data['tau [Mpc]'])} time steps each")

# We can also get the background evolution to find the max conformal time
background = cosmo.get_background()
tau_max_computed = background['conf. time [Mpc]'][-1]
a_max = background['(.)rho_crit'][-1]  # Get last point
print(f"\nBackground evolution:")
print(f"  Maximum conformal time computed: {tau_max_computed:.1f} Mpc")
print(f"  Number of background points: {len(background['conf. time [Mpc]'])}")


# --- 4. Plot the results ---
print("\n" + "="*80)
print("PLOTTING")
print("="*80)

# Plot 4a: Evolution for first k-mode
s = 1.0 / a  # Define s = 1/a as requested

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot potentials
ax1.semilogx(a, phi, label=r'$\Phi$ (Newtonian Potential)', linewidth=2)
ax1.semilogx(a, psi, label=r'$\Psi$ (Curvature Perturbation)', linestyle='--', linewidth=2)
ax1.set_ylabel('Potentials', fontsize=12)
ax1.set_title(f'Evolution of Perturbations for k = {k_first:.4e} 1/Mpc', fontsize=14)
ax1.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label='Today (a=1)')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot density contrasts
ax2.semilogx(a, delta_cdm, label=r'$\delta_{cdm}$', linewidth=2)
ax2.semilogx(a, delta_b, label=r'$\delta_{b}$', linewidth=2)
ax2.semilogx(a, delta_g, label=r'$\delta_{g}$', linewidth=2, alpha=0.7)
ax2.set_xlabel('Scale Factor (a)', fontsize=12)
ax2.set_ylabel(r'Density Contrast $\delta$', fontsize=12)
ax2.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label='Today (a=1)')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(figures_folder_path+ 'perturbations_evolution.pdf')
print("Plot saved to: " + figures_folder_path + "perturbations_evolution.pdf")
plt.close()

# Plot 4b: Compare potentials across all k-modes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for i, k_data in enumerate(scalar_perturbations):
    k_val = k_data['k (1/Mpc)'][0] if 'k (1/Mpc)' in k_data else k_of_interest[i]
    a_k = k_data['a']
    phi_k = k_data['phi']
    delta_cdm_k = k_data['delta_cdm']

    ax1.semilogx(a_k, phi_k, label=f'k = {k_val:.4e}', linewidth=1.5)
    ax2.semilogx(a_k, delta_cdm_k, label=f'k = {k_val:.4e}', linewidth=1.5)

ax1.set_ylabel(r'$\Phi$ (Newtonian Potential)', fontsize=12)
ax1.set_title('Perturbation Evolution Across All k-modes', fontsize=14)
ax1.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label='Today')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, ncol=2)

ax2.set_xlabel('Scale Factor (a)', fontsize=12)
ax2.set_ylabel(r'$\delta_{cdm}$', fontsize=12)
ax2.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(figures_folder_path+ 'perturbations_all_k_modes.pdf')
print("Plot saved to: " + figures_folder_path + "perturbations_all_k_modes.pdf")
plt.close()

# Plot 4c: Higher photon multipoles if available
if photon_multipoles:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot multipoles vs scale factor
    ax1 = axes[0]
    # Plot l=0, 1, 2 (standard ones)
    ax1.semilogx(a, delta_g, label=r'$\delta_g$ (l=0)', linewidth=2)
    ax1.semilogx(a, theta_g, label=r'$\theta_g$ (l=1)', linewidth=2)
    ax1.semilogx(a, shear_g, label=r'$F_{r,2}$ (l=2)', linewidth=2)

    # Plot higher multipoles
    for l in sorted(photon_multipoles.keys())[:5]:  # Show first 5 higher multipoles
        ax1.semilogx(a, photon_multipoles[l], label=f'$F_{{r,{l}}}$ (l={l})', alpha=0.7)

    ax1.set_ylabel('Photon Multipoles')
    ax1.set_title(f'Photon Multipole Evolution for k = {k_first:.4e} 1/Mpc', fontsize=14)
    ax1.axvline(x=1.0, color='grey', linestyle=':', label='Today (a=1)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', ncol=2)

    # Plot multipoles vs conformal time (zoom in near recombination)
    ax2 = axes[1]
    # Find recombination index (around z=1100, a~0.0009)
    a_rec = 1.0 / (1.0 + z_rec)
    rec_idx = np.argmin(np.abs(a - a_rec))

    # Plot around recombination
    tau_window = slice(max(0, rec_idx-100), min(len(tau), rec_idx+200))

    ax2.plot(tau[tau_window], delta_g[tau_window], label=r'$\delta_g$ (l=0)', linewidth=2)
    ax2.plot(tau[tau_window], theta_g[tau_window], label=r'$\theta_g$ (l=1)', linewidth=2)
    ax2.plot(tau[tau_window], shear_g[tau_window], label=r'$F_{r,2}$ (l=2)', linewidth=2)

    for l in sorted(photon_multipoles.keys())[:5]:
        ax2.plot(tau[tau_window], photon_multipoles[l][tau_window],
                label=f'$F_{{r,{l}}}$ (l={l})', alpha=0.7)

    ax2.set_xlabel(r'Conformal Time $\tau$ [Mpc]')
    ax2.set_ylabel('Photon Multipoles')
    ax2.set_title('Around Recombination')
    ax2.axvline(x=tau[rec_idx], color='red', linestyle='--', alpha=0.5, label=f'Recombination (z={z_rec:.0f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(figures_folder_path+ 'photon_multipoles_evolution.pdf')
    print("Plot saved to: " + figures_folder_path + "photon_multipoles_evolution.pdf")
    plt.close()

# --- 5. Clean up the CLASS instance ---
print("\n" + "="*80)
print("CLEANUP")
print("="*80)
cosmo.struct_cleanup()
cosmo.empty()
print("CLASS instance cleaned up successfully.")
print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nSummary:")
print(f"  - Extracted perturbations for {len(scalar_perturbations)} k-modes")
print(f"  - Saved data files: perturbations_k*.npz")
print(f"  - Created plots:")
print(f"    * perturbations_evolution.pdf (first k-mode)")
print(f"    * perturbations_all_k_modes.pdf (all k-modes)")
if photon_multipoles:
    print(f"    * photon_multipoles_evolution.pdf (multipoles)")
print(f"\nNOTE: For closed universe (Omega_k = {OmegaK:.4f}):")
print(f"  - These k values work because we requested 'dTk, vTk' output")
print(f"  - NOT 'tCl' which would require quantized k values")
print(f"  - Perturbations are computed WITHOUT hyperspherical Bessel functions")