"""
Test new convergence-based loss function on existing master_log.txt data.

Physical picture:
  The ideal allowed_K sequence asymptotically converges:
    - spacings start slightly below nu_spacing, approach nu_spacing at large K
    - K values start offset from integers, approach integers at large K
  Loss measures the TAIL of the sequence directly (last tail_frac of points),
  which is where convergence should be most visible.
  No stability filter — use the full sequence.
"""
import numpy as np
import matplotlib.pyplot as plt

nu_spacing = 4

def calculate_integer_loss_smooth(allowedK_integer, n_min=12, spacing_tol_factor=3,
                                       w_slope=20.0, w_integer=1.0):
    """
    Loss for K sequences that asymptotically converge as k increases.
    Uses:
      1. A 2-point midpoint filter to kill alternating (+)/(-) wiggles.
      2. A 'plateau' inverted-parabola weighting to equally weight the entire tail,
         improving robustness against noise in the final points.
    """
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    # --- 1. Robust trimming of non-physical jumps ---
    spacings = np.diff(k)
    nu_spacing = 4.0
    median_spacing = np.median(spacings)
    mad = np.median(np.abs(spacings - median_spacing))
    threshold = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(k)
    for i in range(len(spacings) - 1, len(spacings) // 2, -1):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1

    k = k[:end_idx]
    n = len(k)
    if n < n_min:
        return np.inf, np.inf, np.inf

    # --- 2. 2-Point Midpoint Filter (Kills alternating wiggles) ---
    k_smooth = (k[:-1] + k[1:]) / 2.0
    indices_smooth = np.arange(len(k_smooth), dtype=float) + 0.5

    # --- 3. NEW: The User's "Plateau" Weighting (-x^2 style) ---
    # Create z from 0.0 to 1.0 representing position in the smooth array
    z_sp = np.linspace(0, 1, len(k_smooth) - 1)
    sp_weights = 1.0 - (1.0 - z_sp)**2 
    sp_weights /= sp_weights.sum()  # Normalize to sum to 1

    z_int = np.linspace(0, 1, len(k_smooth))
    int_weights = 1.0 - (1.0 - z_int)**2
    int_weights /= int_weights.sum() # Normalize to sum to 1

    # --- 4. Slope Loss ---
    spacings_smooth = np.diff(k_smooth)                         
    tail_slope_loss = np.sum(sp_weights * (spacings_smooth - nu_spacing) ** 2)

    # --- 5. Integer Loss ---
    weighted_mean_intercept = np.sum(int_weights * (k_smooth - nu_spacing * indices_smooth))
    ideal_intercept = np.round(weighted_mean_intercept)
    ideal_sequence  = ideal_intercept + nu_spacing * indices_smooth
    
    tail_integer_loss = np.sum(int_weights * (k_smooth - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss


def calculate_integer_loss_new(allowedK_integer, n_min=12, spacing_tol_factor=2,
                               w_slope=5.0, w_integer=1.0, alpha=2.0):
    """
    Loss for K sequences that asymptotically converge as k increases:
      - spacings approach nu_spacing
      - K values approach an equally-spaced integer sequence
    """
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    # --- 1. Robust trimming of non-physical jumps at the high-k end ---
    spacings = np.diff(k)
    nu_spacing = 4.0
    median_spacing = np.median(spacings)
    mad = np.median(np.abs(spacings - median_spacing))
    threshold = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(k)
    # Search forwards from the middle: cut off completely at the FIRST unstable jump
    start_search = max(1, len(spacings) // 2)
    for i in range(start_search, len(spacings)):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1  # Keep everything strictly prior to the jump
            break

    k = k[:end_idx]
    n = len(k)
    if n < n_min:
        return np.inf, np.inf, np.inf

    indices = np.arange(n, dtype=float)

    # --- 2. Tail-weighted slope loss ---
    spacings_stable = np.diff(k)                         
    sp_idx     = np.arange(len(spacings_stable), dtype=float)
    sp_weights = (sp_idx + 1.0) ** alpha
    sp_weights /= sp_weights.sum()
    tail_slope_loss = np.sum(sp_weights * (spacings_stable - nu_spacing) ** 2)

    # --- 3. Tail-weighted integer loss ---
    int_weights = (indices + 1.0) ** alpha
    int_weights /= int_weights.sum()
    
    # FIX: Compute the intercept using the EXACT SAME weights used in the loss.
    # 1. This prevents early non-converged points from phase-shifting the grid.
    # 2. This guarantees the loss landscape is completely C^0 continuous for optimizers.
    weighted_mean_intercept = np.sum(int_weights * (k - nu_spacing * indices))
    ideal_intercept = np.round(weighted_mean_intercept)
    ideal_sequence  = ideal_intercept + nu_spacing * indices
    
    tail_integer_loss = np.sum(int_weights * (k - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss


# ── load master_log.txt ───────────────────────────────────────────────────────
log_path = './data/try_intK_planck_optuna/master_log.txt'

rows = []
with open(log_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        old_total = float(parts[0])
        old_slope = float(parts[1])
        old_perk  = float(parts[2])
        chi2      = float(parts[3])
        params    = [float(parts[i]) for i in range(4, 8)]
        k_arr     = np.array([float(x) for x in parts[8].split(',')])
        rows.append((old_total, old_slope, old_perk, chi2, *params, k_arr))

print(f'Loaded {len(rows)} trials from {log_path}')

# ── apply new loss ────────────────────────────────────────────────────────────
results = []
for row in rows:
    k_arr = row[-1]
    tsl, til, tot = calculate_integer_loss_smooth(k_arr)
    results.append((*row[:-1], tsl, til, tot))   # drop k_arr, append new losses

best_indices = sorted(range(len(rows)), key=lambda i: results[i][-1])[:3]
best_k_arrays = [rows[i][-1] for i in best_indices]

results.sort(key=lambda r: r[-1])   # sort by new total_loss

param_names = ['OmegaM', 'OmegaK', 'Omegab_ratio', 'h']

# ── print top 15 ─────────────────────────────────────────────────────────────
print('\nTop 15 by NEW total_loss:')
print(f"{'new_total':>12}  {'tail_slope':>12}  {'tail_intgr':>12}  "
      f"{'OmegaM':>8}  {'OmegaK':>10}  {'Omegab':>8}  {'h':>8}")
for r in results[:15]:
    old_tot, old_sl, old_pk, chi2, OM, OK, Ob, h, tsl, til, tot = r
    print(f"{tot:12.4e}  {tsl:12.4e}  {til:12.4e}  "
          f"{OM:8.5f}  {OK:10.6f}  {Ob:8.5f}  {h:8.5f}")

# ── rank correlation old vs new ───────────────────────────────────────────────
from scipy.stats import spearmanr
old_totals = np.array([r[0] for r in results])
new_totals = np.array([r[-1] for r in results])
rho, pval = spearmanr(old_totals, new_totals)
print(f'\nSpearman rank correlation (old vs new loss): {rho:.4f}  (p={pval:.2e})')

# ── plot 1: loss landscape and decomposition ──────────────────────────────────
tsl_vals = np.array([r[-3] for r in results])
til_vals = np.array([r[-2] for r in results])
OM_vals  = np.array([r[4]  for r in results])
OK_vals  = np.array([r[5]  for r in results])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.scatter(np.log10(old_totals + 1e-12), np.log10(new_totals + 1e-12), s=5, alpha=0.5)
ax.set_xlabel('log10(old total_loss)', fontsize=11)
ax.set_ylabel('log10(new total_loss)', fontsize=11)
ax.set_title(f'Old vs New  (Spearman ρ={rho:.3f})', fontsize=11)

ax = axes[1]
sc = ax.scatter(OM_vals, OK_vals, c=np.log10(new_totals + 1e-12),
                cmap='viridis_r', s=6, alpha=0.7)
plt.colorbar(sc, ax=ax, label='log10(new total_loss)')
ax.set_xlabel(r'$\Omega_M$', fontsize=11)
ax.set_ylabel(r'$\Omega_K$', fontsize=11)
ax.set_title('New total loss in OmegaK-OmegaM space', fontsize=11)

ax = axes[2]
sc2 = ax.scatter(np.log10(tsl_vals + 1e-12), np.log10(til_vals + 1e-12),
                 c=np.log10(new_totals + 1e-12), cmap='viridis_r', s=6, alpha=0.7)
plt.colorbar(sc2, ax=ax, label='log10(new total_loss)')
ax.set_xlabel('log10(tail_slope_loss)', fontsize=11)
ax.set_ylabel('log10(tail_integer_loss)', fontsize=11)
ax.set_title('New loss decomposition', fontsize=11)

plt.suptitle('New tail-based loss on existing 796 trials', fontsize=12)
plt.tight_layout()
out = './data/try_intK_planck_optuna/new_loss_test.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'\nPlot saved: {out}')

# ── plot 2: diagnose best 3 trials — spacing and integer deviation vs index ───
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))

for col, k in enumerate(best_k_arrays):
    r = results[col]
    OM, OK, Ob, h = r[4], r[5], r[6], r[7]
    n = len(k)
    tail_start = int(n * 0.6)

    spacings = np.diff(k)
    frac     = np.abs(k - np.round(k))

    ax = axes2[0, col]
    ax.plot(spacings, 'o-', ms=3, lw=1)
    ax.axhline(nu_spacing, color='red', lw=1.5, ls='--', label=f'nu_spacing={nu_spacing}')
    ax.axvline(tail_start, color='grey', lw=1, ls=':', label='tail start')
    ax.set_xlabel('index', fontsize=10)
    ax.set_ylabel('spacing', fontsize=10)
    ax.set_title(f'Rank {col+1}: spacings\nOK={OK:.5f} OM={OM:.4f}', fontsize=9)
    ax.legend(fontsize=8)

    ax = axes2[1, col]
    ax.plot(frac, 'o-', ms=3, lw=1, color='darkorange')
    ax.axvline(tail_start, color='grey', lw=1, ls=':', label='tail start')
    ax.set_xlabel('index', fontsize=10)
    ax.set_ylabel('|K - round(K)|', fontsize=10)
    ax.set_title(f'Rank {col+1}: deviation from integer\ntail_slope={r[-3]:.2e}  tail_int={r[-2]:.2e}', fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle('Top 3 trials by new loss — full K sequence diagnostics', fontsize=12)
plt.tight_layout()
out2 = './data/try_intK_planck_optuna/new_loss_best_trials.pdf'
plt.savefig(out2, bbox_inches='tight', dpi=150)
print(f'Plot saved: {out2}')
