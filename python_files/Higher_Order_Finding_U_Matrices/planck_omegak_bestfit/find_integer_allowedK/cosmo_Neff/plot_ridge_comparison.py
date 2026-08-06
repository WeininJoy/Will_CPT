"""
Compare the best-slope_loss and best-per_k_loss ridges across all 2D projections.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

data = []
with open('./data/try_intK_planck_optuna/master_log.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        data.append([float(x) for x in parts[:9]])

arr        = np.array(data)
slope_loss = arr[:, 1]
per_k_loss = arr[:, 2]
params     = arr[:, 4:9]
param_names = [r'$\Omega_M$', r'$\Omega_K$', r'$\Omega_b/\Omega_M$', r'$h$', r'$N_\mathrm{eff}$']

pct = 15   # top N% for each loss separately
sl_thresh = np.percentile(slope_loss, pct)
pk_thresh = np.percentile(per_k_loss, pct)

best_sl = slope_loss <= sl_thresh   # low slope_loss ridge
best_pk = per_k_loss <= pk_thresh   # low per_k_loss ridge
both    = best_sl & best_pk

pairs = list(combinations(range(5), 2))   # all 10 pairs
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()

for ax, (i, j) in zip(axes, pairs):
    ax.scatter(params[~best_sl & ~best_pk, j], params[~best_sl & ~best_pk, i],
               s=3, c='lightgrey', alpha=0.4, label='other')
    ax.scatter(params[best_sl, j], params[best_sl, i],
               s=12, c='royalblue', alpha=0.7, label=f'low slope_loss (top {pct}%)')
    ax.scatter(params[best_pk, j], params[best_pk, i],
               s=12, c='tomato', alpha=0.7, label=f'low per_k_loss (top {pct}%)')
    ax.scatter(params[both, j], params[both, i],
               s=40, c='black', marker='*', zorder=5, label='both low')
    ax.set_xlabel(param_names[j], fontsize=11)
    ax.set_ylabel(param_names[i], fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle(f'Blue = low slope_loss ridge,  Red = low per_k_loss ridge\n'
             f'Black stars = both low simultaneously  (top {pct}% each)',
             fontsize=11)
plt.tight_layout(rect=[0, 0.05, 1, 1])

out = './data/try_intK_planck_optuna/ridge_comparison.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved: {out}')
print(f'\nTrials with both losses in top {pct}%: {both.sum()} / {len(arr)}')
if both.sum() > 0:
    print('\nParameter values at overlap points:')
    for name, vals in zip(['OmegaM','OmegaK','Omegab_ratio','h','Neff'],
                           params[both].T):
        print(f'  {name:15s}: [{vals.min():.5f}, {vals.max():.5f}]  median={np.median(vals):.5f}')
