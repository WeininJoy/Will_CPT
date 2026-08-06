"""
Visualise where slope_loss and per_k_integer_loss are simultaneously low
across the 5-parameter space (OmegaM, OmegaK, Omegab_ratio, h, Neff).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── load data ────────────────────────────────────────────────────────────────
data = []
with open('./data/try_intK_planck_optuna/master_log.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        data.append([float(x) for x in parts[:9]])

arr = np.array(data)
total_loss   = arr[:, 0]
slope_loss   = arr[:, 1]
per_k_loss   = arr[:, 2]
params       = arr[:, 4:9]
param_names  = [r'$\Omega_M$', r'$\Omega_K$', r'$\Omega_b/\Omega_M$', r'$h$', r'$N_\mathrm{eff}$']

# log-scale losses for colouring
log_sl = np.log10(slope_loss)
log_pk = np.log10(per_k_loss)

# overlap score: sum of normalised log-losses (lower = both small)
log_sl_norm = (log_sl - log_sl.min()) / (log_sl.max() - log_sl.min())
log_pk_norm = (log_pk - log_pk.min()) / (log_pk.max() - log_pk.min())
overlap_score = log_sl_norm + log_pk_norm   # 0 = both minimal

# ── figure layout: 3 panels each with a 4×4 scatter matrix ──────────────────
n_params = 5
titles   = ['slope_loss  (log scale)', 'per_k_integer_loss  (log scale)',
            'Overlap score  (low = both losses small)']
colors   = [log_sl, log_pk, overlap_score]
cmaps    = ['viridis_r', 'plasma_r', 'RdYlGn_r']

fig, axes = plt.subplots(n_params, n_params * 3,
                         figsize=(30, 10),
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

for panel, (title, c, cmap) in enumerate(zip(titles, colors, cmaps)):
    col_offset = panel * n_params
    vmin, vmax = np.percentile(c, 2), np.percentile(c, 98)

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col_offset + col]
            ax.set_xticks([]); ax.set_yticks([])

            if row == col:
                ax.text(0.5, 0.5, param_names[row],
                        ha='center', va='center', fontsize=11,
                        transform=ax.transAxes)
                ax.set_facecolor('#f5f5f5')
                if row == 0:
                    ax.set_title(title, fontsize=10, pad=6)
                continue

            sc = ax.scatter(params[:, col], params[:, row],
                            c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=4, alpha=0.7, rasterized=True)

            if col == 0:
                ax.set_ylabel(param_names[row], fontsize=8)
            if row == n_params - 1:
                ax.set_xlabel(param_names[col], fontsize=8)

    # one shared colorbar per panel
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([
        0.005 + panel * 0.335 + 0.29,   # left
        0.08,                             # bottom
        0.008,                            # width
        0.85                              # height
    ])
    fig.colorbar(sm, cax=cbar_ax)

plt.suptitle(f'Loss landscape across cosmological parameter space  ({len(arr)} trials)',
             y=1.01, fontsize=13)

out = './data/try_intK_planck_optuna/loss_param_space.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved: {out}')

# ── supplementary: 1-D marginals showing overlap ─────────────────────────────
fig2, axs = plt.subplots(1, 5, figsize=(17, 3.5))
low_overlap = overlap_score < np.percentile(overlap_score, 10)  # best 10%

for i, (ax, name) in enumerate(zip(axs, param_names)):
    ax.hist(params[:, i], bins=30, color='steelblue', alpha=0.5, label='all')
    ax.hist(params[low_overlap, i], bins=20, color='crimson', alpha=0.7,
            label='best 10%\n(both losses low)')
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel('count')
    ax.legend(fontsize=8)

plt.suptitle('Parameter distribution: all trials vs. best overlap region', fontsize=12)
plt.tight_layout()

out2 = './data/try_intK_planck_optuna/loss_param_marginals.pdf'
plt.savefig(out2, bbox_inches='tight', dpi=150)
print(f'Saved: {out2}')
