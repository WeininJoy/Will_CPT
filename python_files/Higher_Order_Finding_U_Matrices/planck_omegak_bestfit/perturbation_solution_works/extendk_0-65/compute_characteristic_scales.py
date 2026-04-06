"""
Compute cosmological characteristic scales and compare with the k≈6e-3 Mpc^-1 resonance.

Quantities computed:
  - z_rec, r_s(z_rec): recombination redshift and sound horizon
  - z_drag, r_s(z_drag): baryon drag epoch sound horizon
  - z_eq, k_eq: matter-radiation equality redshift and wavenumber
  - k_s1 = π / r_s(z_rec): first CMB acoustic peak wavenumber
  - l_s1 ≈ π * D_A^comoving(z_rec) / r_s(z_rec): first CMB acoustic peak multipole

Uses cosmological parameters from generate_data.py (nu_spacing4_bestfit).
Also does an analytic integration as a cross-check.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize import root_scalar
import sys, os

# ---------------------------------------------------------------------------
# Make parent dir importable so we can share the parameter block
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------------------------
# Cosmological parameters  (identical to generate_data.py)
# ---------------------------------------------------------------------------
nu_spacing4_bestfit = [426.32052836334015, 1.5217932160492045,
                       0.1555991124300318, 0.5567037724095327,
                       2.030476, 0.967708, 0.046112]

mt, kt, omega_b_ratio, h = nu_spacing4_bestfit[:4]
As_scalar = nu_spacing4_bestfit[4] * 1e-9
ns        = nu_spacing4_bestfit[5]
tau_reio  = nu_spacing4_bestfit[6]

lam = 1
rt  = 1
Omega_gamma_h2 = 2.47e-5
Neff   = 3.046
N_ncdm = 1
m_ncdm = 0.06
N_ur   = Neff - N_ncdm

# ---------------------------------------------------------------------------
# Derived Omega parameters  (same as generate_data.py / generate_PPS.py)
# ---------------------------------------------------------------------------
Omega_r_h2 = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2
Omega_r    = Omega_r_h2 / h**2

def solve_a0(Omega_r, rt, mt, kt):
    def f(a0):
        return a0**4 - 3*kt*a0**2 + mt*a0 + (rt - 1./Omega_r)
    return root_scalar(f, bracket=[1, 1e3]).root

a0           = solve_a0(Omega_r, rt, mt, kt)
Omega_lambda = Omega_r * a0**4
Omega_m      = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
Omega_K      = -3 * kt * np.sqrt(Omega_lambda * Omega_r)

Omega_b   = omega_b_ratio * Omega_m
Omega_cdm = (1 - omega_b_ratio) * Omega_m

print("=" * 60)
print("Derived Omega parameters")
print("=" * 60)
print(f"  h            = {h:.6f}")
print(f"  Omega_r      = {Omega_r:.6e}")
print(f"  Omega_m      = {Omega_m:.6f}")
print(f"  Omega_b      = {Omega_b:.6f}")
print(f"  Omega_cdm    = {Omega_cdm:.6f}")
print(f"  Omega_lambda = {Omega_lambda:.6f}")
print(f"  Omega_K      = {Omega_K:.6f}")
print(f"  a0 (present) = {a0:.4f}")

# ---------------------------------------------------------------------------
# CLASS computation
# ---------------------------------------------------------------------------
try:
    import classy
    cosmo = classy.Class()
    cosmo.set({
        'output':    'tCl',
        'h':         h,
        'Omega_b':   Omega_b,
        'Omega_cdm': Omega_cdm,
        'Omega_k':   float(Omega_K),
        'A_s':       As_scalar,
        'n_s':       ns,
        'N_ncdm':    N_ncdm,
        'm_ncdm':    m_ncdm,
        'N_ur':      N_ur,
        'tau_reio':  tau_reio,
        'lensing':   'no',
    })
    cosmo.compute()

    # --- CLASS derived parameters ---
    derived = cosmo.get_current_derived_parameters([
        'z_rec', 'tau_rec', 'rs_rec',
        'z_d',   'tau_d',   'rs_d',
        'z_eq',  'tau_eq',
    ])

    z_rec_class  = derived['z_rec']
    rs_rec_class = derived['rs_rec']   # Mpc
    z_drag_class = derived['z_d']
    rs_drag_class= derived['rs_d']     # Mpc
    z_eq_class   = derived['z_eq']

    # Comoving angular diameter distance to recombination  (Mpc)
    # CLASS angular_distance() returns physical D_A; comoving = D_A * (1 + z)
    D_ang_phys   = cosmo.angular_distance(z_rec_class)   # Mpc
    D_A_comoving = D_ang_phys * (1 + z_rec_class)

    # First acoustic-peak wavenumber and multipole (flat-sky approx)
    k_s1_class = np.pi / rs_rec_class            # Mpc^-1
    l_s1_class = np.pi * D_A_comoving / rs_rec_class

    # Equality wavenumber  k_eq = a_eq H(a_eq) / c
    # CLASS does not output k_eq directly; compute from z_eq
    a_eq = 1.0 / (1 + z_eq_class)
    def H_over_c(a):
        """H(a)/c in Mpc^-1, using physical H0 = h*100 km/s/Mpc"""
        H0_mpc = h * 100.0 / 299792.458   # Mpc^-1
        Oa = Omega_lambda + Omega_K/a**2 + Omega_m/a**3 + Omega_r/a**4
        return H0_mpc * np.sqrt(Oa)

    k_eq_class = a_eq * H_over_c(a_eq) * 299792.458 / 299792.458  # stays as Mpc^-1
    # Simpler: k_eq = H(a_eq)*a_eq where H is in Mpc^-1 (comoving Hubble rate)
    # = a^2 * dH/da ... actually just use:
    H0_mpc = h * 100.0 / 299792.458
    k_eq_class = H0_mpc * np.sqrt(2 * Omega_m * (1 + z_eq_class))  # standard formula

    cosmo.struct_cleanup()
    cosmo.empty()
    class_ok = True

except Exception as e:
    print(f"\n[Warning] CLASS failed: {e}")
    print("Falling back to analytic computation only.\n")
    z_rec_class = rs_rec_class = z_drag_class = rs_drag_class = None
    z_eq_class = k_s1_class = l_s1_class = k_eq_class = None
    D_A_comoving = None
    class_ok = False

# ---------------------------------------------------------------------------
# Analytic / numerical computation (independent cross-check)
# ---------------------------------------------------------------------------
c_kms = 299792.458   # km/s
H0    = h * 100.0    # km/s/Mpc

# Matter-radiation equality
z_eq_analytic = Omega_m / Omega_r - 1
print(f"\n[Analytic] z_eq = {z_eq_analytic:.1f}")

# Baryon-to-photon ratio at redshift z
Omega_gamma = Omega_gamma_h2 / h**2
def R_baryons(z):
    """R = 3*rho_b / (4*rho_gamma) = (3/4) * (Omega_b/Omega_gamma) / (1+z)"""
    return (3.0 / 4.0) * (Omega_b / Omega_gamma) / (1.0 + z)

def sound_speed(z):
    """c_s = c / sqrt(3*(1+R))"""
    return 1.0 / np.sqrt(3.0 * (1.0 + R_baryons(z)))

def H_z(z):
    """H(z) in km/s/Mpc"""
    a = 1.0 / (1.0 + z)
    return H0 * np.sqrt(Omega_lambda + Omega_K*a**(-2) + Omega_m*a**(-3) + Omega_r*a**(-4))

# Comoving sound horizon: r_s(z*) = integral_z*^infty  c_s / H(z)  dz
def rs_integrand(z):
    return c_kms * sound_speed(z) / H_z(z)   # Mpc

def compute_rs(z_star):
    val, _ = quad(rs_integrand, z_star, 1e6, limit=500)
    return val   # Mpc

# Comoving distance  chi(z) = integral_0^z  c/H(z') dz'
def chi(z):
    val, _ = quad(lambda zp: c_kms / H_z(zp), 0, z, limit=500)
    return val   # Mpc

# Approximate recombination redshift (Hu & Sugiyama 1996 fitting formula)
# z_rec ≈ 1048 * [1 + 0.00124*(Omega_b h²)^-0.738] * [1 + g1*(Omega_m h²)^g2]
Ob_h2 = Omega_b * h**2
Om_h2 = Omega_m * h**2
g1 = 0.0783 * Ob_h2**(-0.238) / (1 + 39.5 * Ob_h2**0.763)
g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
z_rec_fitting = 1048 * (1 + 0.00124 * Ob_h2**(-0.738)) * (1 + g1 * Om_h2**g2)
print(f"[Analytic] z_rec (Hu&Sugiyama fitting formula) = {z_rec_fitting:.1f}")

# Use CLASS z_rec if available, else fitting formula
z_rec_use  = z_rec_class  if class_ok else z_rec_fitting
z_drag_use = z_drag_class if class_ok else z_rec_fitting * 0.98  # approx

rs_rec_analytic  = compute_rs(z_rec_use)
rs_drag_analytic = compute_rs(z_drag_use)

# Comoving angular diameter distance (flat: chi(z_rec); curved: more complex but close)
D_A_analytic = chi(z_rec_use)

k_s1_analytic = np.pi / rs_rec_analytic
l_s1_analytic = np.pi * D_A_analytic / rs_rec_analytic

# Equality wavenumber  k_eq = sqrt(2 Omega_m H0^2 (1+z_eq)) / c  [Mpc^-1]
k_eq_analytic = H0 / c_kms * np.sqrt(2 * Omega_m * (1 + z_eq_analytic))

# ---------------------------------------------------------------------------
# Resonance scale from numerical verification
# ---------------------------------------------------------------------------
k_res = 5.958e-3   # Mpc^-1  (k_index=38, vr^∞ ≈ -3.2e-3)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Characteristic Scales Summary")
print("=" * 60)

if class_ok:
    print(f"\n--- Recombination (CLASS) ---")
    print(f"  z_rec                   = {z_rec_class:.1f}")
    print(f"  r_s(z_rec)              = {rs_rec_class:.2f}  Mpc")
    print(f"  k_s1 = π/r_s(z_rec)    = {k_s1_class:.4e}  Mpc^-1")
    print(f"  D_A comoving(z_rec)     = {D_A_comoving:.1f}  Mpc")
    print(f"  l_1st peak ≈ π*D_A/r_s = {l_s1_class:.1f}")

    print(f"\n--- Drag epoch (CLASS) ---")
    print(f"  z_drag                  = {z_drag_class:.1f}")
    print(f"  r_s(z_drag)             = {rs_drag_class:.2f}  Mpc")
    print(f"  k_drag = π/r_s(z_drag)  = {np.pi/rs_drag_class:.4e}  Mpc^-1")

    print(f"\n--- Matter-radiation equality (CLASS/analytic) ---")
    print(f"  z_eq (CLASS)            = {z_eq_class:.1f}")
    print(f"  z_eq (analytic)         = {z_eq_analytic:.1f}")
    print(f"  k_eq (CLASS formula)    = {k_eq_class:.4e}  Mpc^-1")
    print(f"  k_eq (analytic)         = {k_eq_analytic:.4e}  Mpc^-1")

print(f"\n--- Analytic cross-check ---")
print(f"  z_rec (fitting formula) = {z_rec_fitting:.1f}")
print(f"  r_s(z_rec) [analytic]   = {rs_rec_analytic:.2f}  Mpc")
print(f"  k_s1 [analytic]         = {k_s1_analytic:.4e}  Mpc^-1")
print(f"  D_A comoving [analytic] = {D_A_analytic:.1f}  Mpc")
print(f"  l_1st peak [analytic]   = {l_s1_analytic:.1f}")
print(f"  k_eq [analytic]         = {k_eq_analytic:.4e}  Mpc^-1")

print(f"\n--- Resonance comparison ---")
print(f"  k_res (vr^∞ node)       = {k_res:.4e}  Mpc^-1")
if class_ok:
    print(f"  k_res / k_s1 (CLASS)    = {k_res / k_s1_class:.3f}   (1 = first acoustic peak)")
    print(f"  k_s1 / k_res (CLASS)    = {k_s1_class / k_res:.3f}   (how many k_res fit in k_s1)")
    print(f"  k_res / k_eq (CLASS)    = {k_res / k_eq_class:.3f}   (1 = equality scale)")
    l_res = k_res * D_A_comoving if D_A_comoving else k_res * D_A_analytic
    print(f"  l corresponding to k_res= {l_res:.1f}")
print(f"  k_res / k_s1 [analytic] = {k_res / k_s1_analytic:.3f}")
print(f"  k_res / k_eq [analytic] = {k_res / k_eq_analytic:.3f}")

# ---------------------------------------------------------------------------
# Plot: scales on a k-axis
# ---------------------------------------------------------------------------
os.makedirs('./figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xscale('log')
ax.set_xlim(1e-4, 1e-1)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=13)
ax.set_title('Characteristic scales vs. resonance wavenumber', fontsize=13)
ax.set_yticks([])

# vertical lines
styles = {
    'k_res': (k_res,       'red',    r'$k_\mathrm{res}=5.96\times10^{-3}$ Mpc$^{-1}$ (resonance, verified)'),
    'k_s1':  (k_s1_analytic,'blue',  r'$k_{s1}=\pi/r_s(z_\mathrm{rec})$ (1st acoustic peak)'),
    'k_eq':  (k_eq_analytic,'green', r'$k_\mathrm{eq}$ (matter-radiation equality)'),
}
if class_ok:
    styles['k_s1_class']  = (k_s1_class,  'cornflowerblue', r'$k_{s1}$ CLASS')
    styles['k_eq_class']  = (k_eq_class,  'limegreen',      r'$k_\mathrm{eq}$ CLASS')
    styles['k_drag_class']= (np.pi/rs_drag_class, 'purple', r'$\pi/r_s(z_\mathrm{drag})$ CLASS')

for name, (kval, col, label) in styles.items():
    ax.axvline(kval, color=col, lw=1.8, ls='--', label=f'{label}  ({kval:.3e})')

ax.legend(fontsize=8, loc='upper left', ncol=1)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
out = './figures/characteristic_scales.pdf'
plt.savefig(out, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot: R_b(z) and c_s(z)
# ---------------------------------------------------------------------------
z_arr = np.logspace(0, 4, 500)
R_arr  = R_baryons(z_arr)
cs_arr = sound_speed(z_arr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax1, ax2 = axes

ax1.loglog(z_arr, R_arr)
ax1.axvline(z_rec_use,  color='r', ls='--', label=f'z_rec={z_rec_use:.0f}')
ax1.axvline(z_eq_analytic, color='g', ls='--', label=f'z_eq={z_eq_analytic:.0f}')
ax1.set_xlabel('Redshift $z$');  ax1.set_ylabel(r'$R = 3\rho_b/4\rho_\gamma$')
ax1.set_title('Baryon-to-photon ratio $R(z)$');  ax1.legend();  ax1.grid(True, alpha=0.3)

ax2.semilogx(z_arr, cs_arr)
ax2.axvline(z_rec_use,  color='r', ls='--', label=f'z_rec={z_rec_use:.0f}')
ax2.axvline(z_eq_analytic, color='g', ls='--', label=f'z_eq={z_eq_analytic:.0f}')
ax2.set_xlabel('Redshift $z$');  ax2.set_ylabel(r'$c_s/c$')
ax2.set_title('Sound speed $c_s(z)$');  ax2.legend();  ax2.grid(True, alpha=0.3)

fig.tight_layout()
out2 = './figures/sound_speed_R.pdf'
plt.savefig(out2, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()
