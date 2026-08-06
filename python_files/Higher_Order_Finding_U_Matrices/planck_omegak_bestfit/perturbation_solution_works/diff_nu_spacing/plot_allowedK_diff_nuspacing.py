"""
Plot kvalues_allowedK for each nu_spacing on one loglog figure,
in both dimensionless and physical (Mpc^{-1}) units.
Also finds and marks the k value where spacing becomes approximately equal.

Conversion:  k_phys [Mpc^{-1}] = k_dimless * H0_phys * sqrt(3 * Omega_Lambda)
  where H0_phys = h*100 / c  [Mpc^{-1}]
(follows generate_CMB.py lines 139-143)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

nu_spacing_list = [3, 4, 5, 6, 7, 8, 9]
base_folder = './data_diff_nu_spacing/'

# ── fixed physical constants ──────────────────────────────────────────────────
rt             = 1
Omega_gamma_h2 = 2.47e-5
Neff           = 3.046
c_kms          = 299792.458   # speed of light [km/s]

# ── load p50 parameters ───────────────────────────────────────────────────────
p50_file = './planck_p50_diff_nuspacing.txt'
p50_data = np.loadtxt(p50_file, comments='#')
p50_dict = {int(row[0]): row[1:] for row in p50_data}  # {nu: [mt, kt, Omegab_ratio, h]}


def k_conversion_factor(mt, kt, h):
    """Return conversion factor so that k_phys [Mpc^{-1}] = k_dimless * factor."""
    Omega_r = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2

    def f(a0):
        return a0**4 - 3*kt*a0**2 + mt*a0 + (rt - 1.0/Omega_r)

    a0           = root_scalar(f, bracket=[1, 1e3]).root
    Omega_lambda = Omega_r * a0**4
    H0_phys      = (h * 100) / c_kms          # [Mpc^{-1}]
    return H0_phys * np.sqrt(3 * Omega_lambda)


def find_transition_k(k, tolerance=0.008, min_sustained=5):
    """
    Find the k where spacing becomes approximately equal.
    Returns the first k[i] from which min_sustained consecutive
    steps dk are all within tolerance of median(dk).
    tolerance=0.008 (±0.8%) calibrated so that k_trans for nu_spacing=4
    matches the PPS cut-off scale (~3e-3 Mpc^{-1}) seen in
    PPS_CMB/figures/PPS_power_spectra_cca_2D.pdf.
    """
    dk = np.diff(k)
    dk_ref = np.median(dk)
    in_band = np.abs(dk - dk_ref) / dk_ref < tolerance
    for i in range(len(dk) - min_sustained + 1):
        if np.all(in_band[i:i + min_sustained]):
            return k[i]
    return np.nan


# ── figure: 2 rows × 2 cols  (left=dimless, right=physical) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
(ax_main_d, ax_main_p), (ax_dk_d, ax_dk_p) = axes

print(f"{'nu':>4}  {'conv':>10}  {'k_trans (dimless)':>18}  {'k_trans (Mpc^-1)':>18}  {'dk_med (Mpc^-1)':>16}")
print("-" * 75)

for nu in nu_spacing_list:
    if nu == 4: 
        k = np.load( '../data/data_allowedK/L70_kvalues.npy')
    else:
        k = np.load(base_folder + f'nu_{nu}/kvalues_allowedK.npy')
    mt, kt, _, h = p50_dict[nu]
    conv   = k_conversion_factor(mt, kt, h)
    k_phys = k * conv

    dk_d   = np.diff(k)
    dk_p   = np.diff(k_phys)
    km_d   = 0.5 * (k[:-1]      + k[1:])
    km_p   = 0.5 * (k_phys[:-1] + k_phys[1:])

    k_trans_d = find_transition_k(k)
    k_trans_p = k_trans_d * conv if not np.isnan(k_trans_d) else np.nan

    idx = np.arange(1, len(k) + 1)

    # --- dimensionless panels ---
    line, = ax_main_d.plot(k, idx, marker='.', markersize=4, linewidth=1,
                           label=f'$\\nu_{{\\rm sp}}={nu}$')
    color = line.get_color()
    if not np.isnan(k_trans_d):
        ax_main_d.axvline(k_trans_d, color=color, lw=0.8, ls='--', alpha=0.7)

    ax_dk_d.plot(km_d, dk_d, marker='.', markersize=3, linewidth=0.8, color=color,
                 label=f'$\\nu_{{\\rm sp}}={nu}$')
    if not np.isnan(k_trans_d):
        ax_dk_d.axvline(k_trans_d, color=color, lw=0.8, ls='--', alpha=0.7)

    # --- physical panels ---
    ax_main_p.plot(k_phys, idx, marker='.', markersize=4, linewidth=1,
                   color=color, label=f'$\\nu_{{\\rm sp}}={nu}$')
    if not np.isnan(k_trans_p):
        ax_main_p.axvline(k_trans_p, color=color, lw=0.8, ls='--', alpha=0.7)

    ax_dk_p.plot(km_p, dk_p, marker='.', markersize=3, linewidth=0.8, color=color,
                 label=f'$\\nu_{{\\rm sp}}={nu}$')
    if not np.isnan(k_trans_p):
        ax_dk_p.axvline(k_trans_p, color=color, lw=0.8, ls='--', alpha=0.7)

    print(f"{nu:>4}  {conv:>10.5f}  {k_trans_d:>18.4f}  {k_trans_p:>18.5f}  {np.median(dk_p):>16.5f}")

# ── panel formatting ──────────────────────────────────────────────────────────
for ax in axes.flat:
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=8, ncol=2)

ax_main_d.set_ylabel('Index', fontsize=12)
ax_main_p.set_ylabel('Index', fontsize=12)
ax_dk_d.set_ylabel(r'$\Delta k$ (dimensionless)', fontsize=12)
ax_dk_p.set_ylabel(r'$\Delta k_{\rm phys}\ [\mathrm{Mpc}^{-1}]$', fontsize=12)

ax_dk_d.set_xlabel(r'$k$ (dimensionless)', fontsize=12)
ax_dk_p.set_xlabel(r'$k_{\rm phys}\ [\mathrm{Mpc}^{-1}]$', fontsize=12)

ax_main_d.set_title(r'Dimensionless $k$', fontsize=12)
ax_main_p.set_title(r'Physical $k\ [\mathrm{Mpc}^{-1}]$', fontsize=12)
ax_dk_d.set_title(r'Step size $\Delta k$ — dashed = transition', fontsize=11)
ax_dk_p.set_title(r'Step size $\Delta k_{\rm phys}$ — dashed = transition', fontsize=11)

plt.tight_layout()
plt.savefig('allowedK_diff_nuspacing.pdf')
print("\nSaved: allowedK_diff_nuspacing.pdf")
plt.show()
