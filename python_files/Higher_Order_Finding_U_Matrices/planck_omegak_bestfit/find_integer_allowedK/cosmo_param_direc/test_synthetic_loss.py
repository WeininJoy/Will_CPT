"""
Synthetic sequence tests for calculate_integer_loss_new.

Generates sequences with known properties (length ~25, k starting ~335)
to verify the loss function ranks them as expected.

Expected ranking (lowest to highest loss):
  perfect_integer < converging < noisy_good < bad_integer < bad_slope < random
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from test_new_loss import calculate_integer_loss_new

nu = 4.0
n  = 25
k0 = 336.0   # nearest integer to dataset mean start, exactly on-grid


def make_perfect_integer(n=n, k0=k0):
    """Exact integer sequence with spacing=4."""
    return k0 + nu * np.arange(n)


def make_converging(n=n, k0=k0, offset_start=1.5, spacing_start=3.99):
    """
    Early elements are off-grid and have wrong spacing;
    tail converges to integer grid with spacing=4.
    This is the 'ideal physical' scenario.
    """
    frac = np.linspace(0, 1, n) ** 2          # quadratic convergence
    spacings = spacing_start + (nu - spacing_start) * frac
    offsets  = offset_start  * (1 - frac)
    k = np.zeros(n)
    k[0] = k0 + offsets[0]
    for i in range(1, n):
        k[i] = k[i-1] + spacings[i]
    return k


def make_noisy_good(n=n, k0=k0, noise=0.3, seed=42):
    """
    Spacing=4, integer-aligned, but with small random jitter.
    Should score well but not perfectly.
    """
    rng = np.random.default_rng(seed)
    base = k0 + nu * np.arange(n)
    return base + rng.normal(0, noise, n)


def make_bad_integer(n=n, k0=k0, half_offset=0.5):
    """
    Spacing is exactly 4 but every element is shifted by 0.5 from integers.
    Slope loss ≈ 0, integer loss should be high.
    """
    return (k0 + half_offset) + nu * np.arange(n)


def make_bad_slope(n=n, k0=k0, wrong_spacing=4.01):
    """
    Integer-aligned start, but spacing is consistently wrong (4.01 instead of 4).
    Integer loss may be moderate, slope loss should be high.
    """
    return k0 + wrong_spacing * np.arange(n)


def make_random(n=n, k0=k0, seed=7):
    """
    Completely random spacings — worst case.
    """
    rng = np.random.default_rng(seed)
    spacings = rng.uniform(1, 8, n - 1)
    k = np.zeros(n)
    k[0] = k0
    for i in range(1, n):
        k[i] = k[i-1] + spacings[i-1]
    return k


def make_unstable_tail(n=n, k0=k0, seed=99):
    """
    Good convergence for the first 2/3, then wild jumps at the tail.
    Tests that the trimming logic handles it.
    """
    rng = np.random.default_rng(seed)
    k = make_converging(n=n, k0=k0)
    cut = int(n * 0.65)
    k[cut:] += rng.uniform(5, 20, n - cut)   # inject large jumps
    return k


# ── evaluate ─────────────────────────────────────────────────────────────────
sequences = {
    'perfect_integer': make_perfect_integer(),
    'converging':      make_converging(),
    'noisy_good':      make_noisy_good(),
    'bad_integer':     make_bad_integer(),
    'bad_slope':       make_bad_slope(),
    'random':          make_random(),
    'unstable_tail':   make_unstable_tail(),
}

print(f"{'name':<18}  {'slope_loss':>12}  {'int_loss':>12}  {'total_loss':>12}")
print('-' * 60)
scored = []
for name, k in sequences.items():
    tsl, til, tot = calculate_integer_loss_new(k)
    scored.append((name, k, tsl, til, tot))
    print(f"{name:<18}  {tsl:12.4e}  {til:12.4e}  {tot:12.4e}")

scored.sort(key=lambda x: x[-1])
print('\nRanked best → worst:')
for rank, (name, _, tsl, til, tot) in enumerate(scored, 1):
    print(f"  {rank}. {name:<18}  total={tot:.4e}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(sequences), figsize=(3.5 * len(sequences), 7))

for col, (name, k, tsl, til, tot) in enumerate(scored):
    tsl, til, tot = calculate_integer_loss_new(k)
    spacings = np.diff(k)
    frac     = np.abs(k - np.round(k))
    n_k      = len(k)
    tail_start = int(n_k * 0.6)

    ax = axes[0, col]
    ax.plot(spacings, 'o-', ms=3, lw=1)
    ax.axhline(nu, color='red', lw=1.5, ls='--', label='nu=4')
    ax.axvline(tail_start, color='grey', lw=1, ls=':')
    ax.set_title(f'{name}\ntot={tot:.2e}', fontsize=8)
    ax.set_xlabel('index', fontsize=8)
    ax.set_ylabel('spacing', fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[1, col]
    ax.plot(frac, 'o-', ms=3, lw=1, color='darkorange')
    ax.axvline(tail_start, color='grey', lw=1, ls=':')
    ax.set_title(f'sl={tsl:.2e}  int={til:.2e}', fontsize=8)
    ax.set_xlabel('index', fontsize=8)
    ax.set_ylabel('|K - round(K)|', fontsize=8)

plt.suptitle('Synthetic sequence tests — sorted best→worst loss', fontsize=11)
plt.tight_layout()
out = './data/try_intK_planck_optuna/synthetic_loss_test.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'\nPlot saved: {out}')
