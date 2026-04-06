"""
Plot vr^inf(k) for the integerK basis.

For each integerK mode, solves for x_inf = [dr^inf, dm^inf, vr^inf, vm^inf]
using the same lstsq procedure as compute_allowedK, then plots vr^inf vs k.

If the resonance hypothesis is correct, vr^inf should dip to ~0 near k_index 30-40,
coinciding with the identity block in the CCA heatmap and the PPS peak at ~6e-3 Mpc^-1.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ---------------------------------------------------------------------------
# Cosmological parameters (same as generate_PPS.py)
# ---------------------------------------------------------------------------
nu_spacing4_bestfit = [426.32052836334015, 1.5217932160492045,
                       0.1555991124300318, 0.5567037724095327,
                       2.030476, 0.967708, 0.046112]
mt, kt, omega_b_ratio, h = nu_spacing4_bestfit[:4]
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
num_variables = 75  # must match num_variables_save used when generating data

def cosmological_parameters(mt, kt, h):
    rt = 1
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3)) * Omega_gamma_h2 / h**2
    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt - 1./Omega_r)
        return root_scalar(f, bracket=[1, 1e3]).root
    a0 = solve_a0(Omega_r, rt, mt, kt)
    Omega_lambda = Omega_r * a0**4
    Omega_m  = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K  = -3 * kt * np.sqrt(Omega_lambda * Omega_r)
    return Omega_lambda, Omega_m, Omega_K

OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
H0 = 1 / np.sqrt(3 * OmegaLambda)

# Unit conversion: internal k → Mpc^{-1}
c_kms = 299792.458
H0_phys = (h * 100) / c_kms          # Mpc^{-1}
conv    = H0_phys * np.sqrt(3 * OmegaLambda)

# ---------------------------------------------------------------------------
# Load integerK data
# ---------------------------------------------------------------------------
data_folder = './data/data_integerK/'
kvalues     = np.load(data_folder + 'L70_kvalues.npy')
ABCmatrices = np.load(data_folder + 'L70_ABCmatrices.npy')
DEFmatrices = np.load(data_folder + 'L70_DEFmatrices.npy')
GHIvectors  = np.load(data_folder + 'L70_GHIvectors.npy')
X1matrices  = np.load(data_folder + 'L70_X1matrices.npy')
X2matrices  = np.load(data_folder + 'L70_X2matrices.npy')
recValues   = np.load(data_folder + 'L70_recValues.npy')

print(f"Loaded {len(kvalues)} integerK modes")
print(f"ABCmatrices shape: {ABCmatrices.shape}")

# ---------------------------------------------------------------------------
# Compute x_inf = [dr^inf, dm^inf, vr^inf, vm^inf] for each mode
# (identical procedure to Higher_Order_Solving_for_Vrinf.compute_allowedK)
# ---------------------------------------------------------------------------
vrfcb = []
drfcb = []
dmfcb = []
vmfcb = []

for j in range(len(kvalues)):
    A    = ABCmatrices[j, 0:6,  :]        # (6, 6)
    D    = DEFmatrices[j, 0:6,  :]        # (6, 2)
    G    = GHIvectors[j,  0:6]            # (6,)
    X1   = X1matrices[j]                  # (6, 4)
    X2   = X2matrices[j]                  # (2, 4)
    recs = recValues[j]

    GX3 = np.zeros((6, 4))
    GX3[:, 2] = G

    matrixog = A @ X1 + D @ X2 + GX3     # (6, 4)
    matrix   = matrixog[[2, 3, 4, 5], :] # rows: dr, dm, vr, vm  → (4, 4)
    xrecs    = recs[[2, 3, 4, 5]]

    xinf = np.linalg.lstsq(matrix, xrecs, rcond=None)[0]
    drfcb.append(xinf[0])
    dmfcb.append(xinf[1])
    vrfcb.append(xinf[2])
    vmfcb.append(xinf[3])

vrfcb = np.array(vrfcb)
drfcb = np.array(drfcb)
dmfcb = np.array(dmfcb)
vmfcb = np.array(vmfcb)

k_index    = np.arange(len(kvalues))
k_physical = kvalues * conv

# ---------------------------------------------------------------------------
# Mark where |vr^inf| is smallest (candidate resonant modes)
# ---------------------------------------------------------------------------
abs_vr = np.abs(vrfcb)
resonant_idx = np.where(abs_vr < 0.05 * np.nanmax(abs_vr))[0]
print(f"\nModes with |vr^inf| < 5% of max (resonant candidates):")
for i in resonant_idx:
    print(f"  k_index={i:3d}  k={kvalues[i]:.4f}  "
          f"k_phys={k_physical[i]:.3e} Mpc^-1  vr^inf={vrfcb[i]:.4e}")

# ---------------------------------------------------------------------------
# Plot 1: vr^inf vs k_index
# ---------------------------------------------------------------------------
import os
os.makedirs('./figures', exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

ax = axes[0]
ax.plot(k_index, vrfcb, 'b.-', ms=3, lw=0.8, label=r'$v_r^\infty$')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.scatter(resonant_idx, vrfcb[resonant_idx], color='red', zorder=5,
           label=r'$|v_r^\infty| < 5\%$ of max')
ax.set_xlabel('k index')
ax.set_ylabel(r'$v_r^\infty$')
ax.set_title(r'$v_r^\infty$ for integerK modes (by index)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2: vr^inf vs k_physical (log scale)
# ---------------------------------------------------------------------------
ax2 = axes[1]
ax2.plot(k_physical, vrfcb, 'b.-', ms=3, lw=0.8, label=r'$v_r^\infty$')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.axvline(6e-3, color='orange', lw=1.2, ls=':', label=r'PPS peak $k\approx6\times10^{-3}$ Mpc$^{-1}$')
ax2.scatter(k_physical[resonant_idx], vrfcb[resonant_idx], color='red', zorder=5,
            label=r'$|v_r^\infty| < 5\%$ of max')
ax2.set_xscale('log')
ax2.set_xlabel(r'$k$ [Mpc$^{-1}$]')
ax2.set_ylabel(r'$v_r^\infty$')
ax2.set_title(r'$v_r^\infty$ for integerK modes (physical units)')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', alpha=0.3)

fig.tight_layout()
out = './figures/vrfcb_integerK.pdf'
plt.savefig(out, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3: all four FCB components side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
components = [('dr', drfcb), ('dm', dmfcb), ('vr', vrfcb), ('vm', vmfcb)]
for ax, (name, vals) in zip(axes.flat, components):
    ax.plot(k_physical, vals, '.-', ms=2, lw=0.6)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axvline(6e-3, color='orange', lw=1.2, ls=':',
               label=r'$k\approx6\times10^{-3}$ Mpc$^{-1}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax.set_ylabel(rf'${name}^\infty$')
    ax.set_title(rf'${name}^\infty$ at FCB (integerK)')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.3)
fig.suptitle(r'All FCB components $x^\infty$ for integerK basis', fontsize=12)
fig.tight_layout()
out2 = './figures/xfcb_all_integerK.pdf'
plt.savefig(out2, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()
