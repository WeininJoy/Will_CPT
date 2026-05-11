"""
test_loss_function.py

Tests the calculate_integer_loss() function on synthetic k-sequences
with known properties, to find the sweet spot for (w_slope, w_intercept).

Test cases are designed to span all physically relevant failure modes:
  A. Perfect
  B. Constant offset only   (bad intercept, perfect spacing)
  C. Slope drift only       (bad spacing, near-zero intercept error)
  D. Both bad

The "correct" ordering (smaller loss = better) should be:
  Perfect  <  small bad  <  medium bad  <  large bad
  and for same deviation magnitude:
    constant offset  ≈  slope drift that produces same final deviation

Outputs:
  figures/test_loss_ranking.pdf      - loss values under 6 weight combinations
  figures/test_loss_weight_sweep.pdf - slope-loss contribution vs w_slope
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('./figures', exist_ok=True)

nu_spacing = 4
N = 24          # 24 modes, matching the k50-65 range used by the optimizer
k0 = 331        # integer starting point (clean baseline)

# ─────────────────────────────────────────────────────────────────────────────
# Core loss computation (mirrors calculate_integer_loss exactly)
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss_components(k_seq):
    """Return (slope, intercept, slope_loss, intercept_loss, residual_loss)."""
    k = np.asarray(k_seq, dtype=float)
    n = len(k)
    indices = np.arange(n, dtype=float)
    hk_weights = indices + 1.0          # w_i = i+1, high-k emphasis

    slope, intercept = np.polyfit(indices, k, 1, w=hk_weights)

    slope_loss     = (slope - nu_spacing) ** 2
    intercept_loss = (intercept - round(intercept)) ** 2

    fit = intercept + slope * indices
    residuals = k - fit
    wn = hk_weights / hk_weights.sum()
    residual_loss = float(np.sum(wn * residuals**2))

    return slope, intercept, slope_loss, intercept_loss, residual_loss


def total_loss(k_seq, w_slope, w_intercept, w_residual=1.0):
    _, _, sl, il, rl = compute_loss_components(k_seq)
    return w_slope * sl + w_intercept * il + w_residual * rl


# ─────────────────────────────────────────────────────────────────────────────
# Test sequences
# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a sequence with a given intercept frac and per-step slope error
def make_seq(k_start, slope_err=0.0, n=N):
    """k_i = k_start + i * (nu_spacing + slope_err)"""
    return [k_start + i * (nu_spacing + slope_err) for i in range(n)]

# Constant-offset sequences: spacing = 4 exactly, start is non-integer
def make_offset_seq(offset, n=N):
    """k_i = (k0 + offset) + i * 4"""
    return [k0 + offset + i * nu_spacing for i in range(n)]

# Current best-fit (real data, 24 modes)
real_data = np.load(
    '../perturbation_solution_works/extendk_0-65/data/data_all_k50-65/allowedK_integer.npy'
)

test_cases = [
    # (label, description, sequence)
    # ── A: Perfect ───────────────────────────────────────────────────────────
    ('A0  perfect',
     'spacing=4, start=integer\nslope_err=0, offset=0',
     make_seq(k0, slope_err=0.0)),

    # ── B: Constant offset only (spacing exactly 4) ──────────────────────────
    ('B1  offset=0.05',
     'spacing=4, start=k0+0.05\nslope_err=0, offset=0.05',
     make_offset_seq(0.05)),

    ('B2  offset=0.10',
     'spacing=4, start=k0+0.10\nslope_err=0, offset=0.10',
     make_offset_seq(0.10)),

    ('B3  offset=0.25',
     'spacing=4, start=k0+0.25\nslope_err=0, offset=0.25  [worst-ish]',
     make_offset_seq(0.25)),

    ('B4  offset=0.40',
     'spacing=4, start=k0+0.40\nslope_err=0, offset=0.40',
     make_offset_seq(0.40)),

    # ── C: Slope drift only (start exactly integer) ──────────────────────────
    # Final deviation at mode N-1 ≈ slope_err * (N-1)
    # δ=0.001 → final dev ≈ 0.023
    ('C1  δ=0.001/step',
     'spacing=4.001, start=integer\nfinal dev ≈ 0.023',
     make_seq(k0, slope_err=+0.001)),

    # δ=0.002 → final dev ≈ 0.046  (comparable to offset 0.05)
    ('C2  δ=0.002/step',
     'spacing=4.002, start=integer\nfinal dev ≈ 0.046',
     make_seq(k0, slope_err=+0.002)),

    # δ=0.005 → final dev ≈ 0.115  (current best-fit ballpark)
    ('C3  δ=0.005/step',
     'spacing=4.005, start=integer\nfinal dev ≈ 0.115',
     make_seq(k0, slope_err=+0.005)),

    # δ=0.010 → final dev ≈ 0.23
    ('C4  δ=0.010/step',
     'spacing=4.010, start=integer\nfinal dev ≈ 0.230',
     make_seq(k0, slope_err=+0.010)),

    # δ=0.020 → final dev ≈ 0.46
    ('C5  δ=0.020/step',
     'spacing=4.020, start=integer\nfinal dev ≈ 0.460',
     make_seq(k0, slope_err=+0.020)),

    # ── D: Both bad ───────────────────────────────────────────────────────────
    ('D1  offset=0.10 + δ=0.005',
     'spacing=4.005, start=k0+0.10',
     make_seq(k0 + 0.10, slope_err=+0.005)),

    ('D2  offset=0.10 + δ=0.010',
     'spacing=4.010, start=k0+0.10',
     make_seq(k0 + 0.10, slope_err=+0.010)),

    # ── E: Real data ──────────────────────────────────────────────────────────
    ('E0  real data',
     f'current best-fit (k50-65)\nslope≈3.9953, intercept frac≈0.066',
     real_data),
]

# ─────────────────────────────────────────────────────────────────────────────
# Print component table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'Case':<22}  {'slope':>8}  {'intcpt_frac':>11}  "
      f"{'sl_loss':>10}  {'il_loss':>10}  {'rl_loss':>10}")
print("─" * 78)
for label, desc, seq in test_cases:
    slope, intcpt, sl, il, rl = compute_loss_components(seq)
    print(f"{label:<22}  {slope:8.5f}  {intcpt - round(intcpt):+11.5f}  "
          f"{sl:10.3e}  {il:10.3e}  {rl:10.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# Weight combinations to test
# ─────────────────────────────────────────────────────────────────────────────
weight_combos = [
    # (w_slope, w_intercept, label)
    (  50,  1, 'w_s=50,  w_i=1'),
    ( 100,  1, 'w_s=100, w_i=1'),
    ( 200,  1, 'w_s=200, w_i=1'),
    ( 500,  1, 'w_s=500, w_i=1  [current]'),
    (1000,  1, 'w_s=1000,w_i=1'),
    ( 500,  5, 'w_s=500, w_i=5'),
]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Total loss per test case under each weight combo
# ─────────────────────────────────────────────────────────────────────────────
n_cases  = len(test_cases)
n_combos = len(weight_combos)
labels   = [tc[0] for tc in test_cases]
colors   = ['green'] + ['steelblue']*4 + ['darkorange']*5 + ['firebrick']*2 + ['purple']

losses = np.zeros((n_combos, n_cases))
for ci, (ws, wi, _) in enumerate(weight_combos):
    for ti, (_, _, seq) in enumerate(test_cases):
        losses[ci, ti] = total_loss(seq, ws, wi)

fig, axes = plt.subplots(n_combos, 1, figsize=(14, 3.2 * n_combos), sharex=True)
for ci, (ws, wi, wlabel) in enumerate(weight_combos):
    ax = axes[ci]
    bars = ax.bar(range(n_cases), losses[ci], color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('total loss', fontsize=8)
    ax.set_title(f'{wlabel}', fontsize=9)
    ax.set_ylim(1e-8, max(losses[ci]) * 10)
    ax.grid(True, axis='y', alpha=0.3, which='both')
    # Annotate bar values
    for bar, val in zip(bars, losses[ci]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=6, rotation=45)

axes[-1].set_xticks(range(n_cases))
axes[-1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
fig.suptitle(
    'Total loss for synthetic test sequences under different weight combinations\n'
    'Green=perfect, Blue=offset-only, Orange=slope-drift-only, Red=both, Purple=real data\n'
    'Desired ordering: A0 << B1<B2<B3<B4 and C1<C2<C3<C4<C5, '
    'and equal-deviation cases (B vs C) should be comparable',
    fontsize=10)
fig.tight_layout()
plt.savefig('./figures/test_loss_ranking.pdf', bbox_inches='tight')
print(f"\nSaved: ./figures/test_loss_ranking.pdf")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Decomposed loss (slope / intercept contributions) per case
#           under the recommended weights — shows which term drives each case
# ─────────────────────────────────────────────────────────────────────────────
W_S, W_I = 500, 1   # recommended weights

sl_contribs = np.array([W_S * compute_loss_components(seq)[2] for _, _, seq in test_cases])
il_contribs = np.array([W_I * compute_loss_components(seq)[3] for _, _, seq in test_cases])
rl_contribs = np.array([      compute_loss_components(seq)[4] for _, _, seq in test_cases])

x = np.arange(n_cases)
w = 0.55

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x, sl_contribs, w, label=f'w_slope={W_S} × slope_loss',     color='steelblue',  alpha=0.85)
ax.bar(x, il_contribs, w, bottom=sl_contribs,
       label=f'w_intercept={W_I} × intercept_loss', color='firebrick',  alpha=0.85)
ax.bar(x, rl_contribs, w, bottom=sl_contribs + il_contribs,
       label='residual_loss',                        color='gray',        alpha=0.6)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('loss contribution (log scale)')
ax.set_title(f'Decomposed loss contributions  [w_slope={W_S}, w_intercept={W_I}]\n'
             'Blue = slope term, Red = intercept term, Gray = residual term')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3, which='both')
fig.tight_layout()
plt.savefig('./figures/test_loss_decomposed.pdf', bbox_inches='tight')
print(f"Saved: ./figures/test_loss_decomposed.pdf")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: w_slope sweep — how does the B2 / C3 ratio change?
#   B2 = offset 0.10 (pure intercept error)
#   C3 = slope drift 0.005/step (comparable accumulated deviation ~0.115)
#   We want loss(B2) ≈ loss(C3) at the chosen weights,
#   OR loss(B2) < loss(C3) if we consider drift more harmful.
# ─────────────────────────────────────────────────────────────────────────────
w_slope_vals = np.logspace(1, 4, 200)   # 10 to 10000

# Cases of interest
cases_sweep = {
    'B2  offset=0.10':  make_offset_seq(0.10),
    'B3  offset=0.25':  make_offset_seq(0.25),
    'C2  δ=0.002/step': make_seq(k0, slope_err=+0.002),
    'C3  δ=0.005/step': make_seq(k0, slope_err=+0.005),
    'C4  δ=0.010/step': make_seq(k0, slope_err=+0.010),
    'E0  real data':    real_data,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: absolute loss vs w_slope (w_intercept=1 fixed)
ax = axes[0]
for name, seq in cases_sweep.items():
    _, _, sl, il, rl = compute_loss_components(seq)
    y = w_slope_vals * sl + 1.0 * il + rl
    ax.loglog(w_slope_vals, y, lw=1.8, label=name)

ax.axvline(500, color='k', lw=1, ls='--', label='current w_slope=500')
ax.set_xlabel('w_slope  (w_intercept=1 fixed)')
ax.set_ylabel('total loss  (log)')
ax.set_title('Absolute loss vs w_slope')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# Right: ratio loss(B2_offset=0.10) / loss(C3_drift=0.005) vs w_slope
#   ratio=1 means they're equally penalised
#   ratio>1 means offset is penalised more; ratio<1 means drift is penalised more
ax = axes[1]
_, _, sl_B2, il_B2, rl_B2 = compute_loss_components(make_offset_seq(0.10))
_, _, sl_C3, il_C3, rl_C3 = compute_loss_components(make_seq(k0, slope_err=+0.005))
_, _, sl_B3, il_B3, rl_B3 = compute_loss_components(make_offset_seq(0.25))
_, _, sl_C4, il_C4, rl_C4 = compute_loss_components(make_seq(k0, slope_err=+0.010))

pairs = [
    ('B2/C3  offset=0.10 vs δ=0.005', sl_B2, il_B2, rl_B2, sl_C3, il_C3, rl_C3),
    ('B3/C4  offset=0.25 vs δ=0.010', sl_B3, il_B3, rl_B3, sl_C4, il_C4, rl_C4),
]
for name, sl1, il1, rl1, sl2, il2, rl2 in pairs:
    loss1 = w_slope_vals * sl1 + 1.0 * il1 + rl1
    loss2 = w_slope_vals * sl2 + 1.0 * il2 + rl2
    ax.semilogx(w_slope_vals, loss1 / loss2, lw=1.8, label=name)

ax.axhline(1.0, color='k', lw=0.8, ls='--', label='equal penalty (ratio=1)')
ax.axvline(500, color='k', lw=1,   ls=':',  label='current w_slope=500')
ax.set_xlabel('w_slope  (w_intercept=1 fixed)')
ax.set_ylabel('loss ratio  (offset / drift)')
ax.set_title('Ratio of offset-only loss to drift-only loss\n'
             'ratio < 1: drift penalised more\n'
             'ratio > 1: offset penalised more')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('w_slope sweep  (w_intercept=1 throughout)\n'
             'Use this to judge: at w_slope=500, is the relative penalty between offset and drift cases correct?',
             fontsize=10)
fig.tight_layout()
plt.savefig('./figures/test_loss_weight_sweep.pdf', bbox_inches='tight')
print(f"Saved: ./figures/test_loss_weight_sweep.pdf")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table for the recommended and two alternative weight combos
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print(f"{'Case':<22}  {'w_s=500,w_i=1':>15}  {'w_s=200,w_i=1':>15}  "
      f"{'w_s=500,w_i=5':>15}  {'w_s=1000,w_i=1':>16}")
print("─"*100)
for label, desc, seq in test_cases:
    l1 = total_loss(seq,  500, 1)
    l2 = total_loss(seq,  200, 1)
    l3 = total_loss(seq,  500, 5)
    l4 = total_loss(seq, 1000, 1)
    print(f"{label:<22}  {l1:15.4e}  {l2:15.4e}  {l3:15.4e}  {l4:16.4e}")

print("\nKey comparisons (should drift be penalised more than same-magnitude offset?):")
for ws, wi in [(500,1),(200,1),(500,5),(1000,1)]:
    lB2 = total_loss(make_offset_seq(0.10), ws, wi)
    lC3 = total_loss(make_seq(k0, slope_err=+0.005), ws, wi)
    print(f"  w_s={ws:4d}, w_i={wi}:  "
          f"offset=0.10 → {lB2:.4e}   δ=0.005/step → {lC3:.4e}   "
          f"ratio(offset/drift) = {lB2/lC3:.3f}")
