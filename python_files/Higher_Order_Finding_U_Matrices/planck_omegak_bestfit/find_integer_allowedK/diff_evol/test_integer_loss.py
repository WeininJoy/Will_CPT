"""
Standalone tests for the integer-spacing loss functions.

Tests both the fixed calculate_integer_loss (via _compute_loss_from_stable_k)
and the old calculate_integer_loss_old, on synthetic sequences with known quality.

Also tests weight balance (Section 2): verifies that the weights w_slope,
w_intercept, w_residual give similar loss magnitudes for two "equally bad"
perturbation types of the same size ε:
  Case A: slope=4 exactly, uniform offset ε from integers
  Case B: midpoint anchored at integer, slope error ε (slope = 4+ε)

Expected ranking (lower loss = better):
  perfect  <  good_offset  <  current_bad (slope drift)  <  large_slope_err
  <  half_integer  <  jitter_bad
"""

import numpy as np

# ─── copy the module-level constants needed by the loss functions ──────────
nu_spacing = 4


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone versions of the two loss functions (no I/O, no CLASS calls)
# ══════════════════════════════════════════════════════════════════════════════

def loss_new_components(stable_k, w_slope=500.0, w_intercept=1.0, w_residual=1.0):
    """
    Same as loss_new but also returns the three raw (unweighted) components:
      (slope_loss, per_k_integer_loss, weighted_residual_loss)
    so that the weight-balance test can inspect each term independently.
    """
    stable_k = np.asarray(stable_k, dtype=float)
    n = len(stable_k)
    indices = np.arange(n, dtype=float)
    hk_weights = indices + 1.0

    slope, _ = np.polyfit(indices, stable_k, 1, w=hk_weights)
    slope_loss = (slope - nu_spacing) ** 2

    ideal_intercept_int = np.round(np.mean(stable_k - nu_spacing * indices))
    ideal_sequence = ideal_intercept_int + nu_spacing * indices
    per_k_integer_loss = np.mean((stable_k - ideal_sequence) ** 2)

    _, intercept_fit = np.polyfit(indices, stable_k, 1, w=hk_weights), None
    slope_fit, intercept_fit = np.polyfit(indices, stable_k, 1, w=hk_weights)
    fit_values = intercept_fit + slope_fit * indices
    residuals = stable_k - fit_values
    weights_norm = hk_weights / hk_weights.sum()
    weighted_residual_loss = np.sum(weights_norm * residuals ** 2)

    total = (w_slope * slope_loss +
             w_intercept * per_k_integer_loss +
             w_residual  * weighted_residual_loss)

    return total, slope_loss, per_k_integer_loss, weighted_residual_loss


def loss_new(stable_k, w_slope=500.0, w_intercept=1.0, w_residual=1.0):
    """Fixed loss: per-k integer closeness replaces intercept_loss."""
    stable_k = np.asarray(stable_k, dtype=float)
    n = len(stable_k)
    indices = np.arange(n, dtype=float)
    hk_weights = indices + 1.0

    slope, intercept = np.polyfit(indices, stable_k, 1, w=hk_weights)
    slope_loss = (slope - nu_spacing) ** 2

    ideal_intercept_int = np.round(np.mean(stable_k - nu_spacing * indices))
    ideal_sequence = ideal_intercept_int + nu_spacing * indices
    per_k_integer_loss = np.mean((stable_k - ideal_sequence) ** 2)

    fit_values = intercept + slope * indices
    residuals = stable_k - fit_values
    weights_norm = hk_weights / hk_weights.sum()
    weighted_residual_loss = np.sum(weights_norm * residuals ** 2)

    return (w_slope * slope_loss +
            w_intercept * per_k_integer_loss +
            w_residual  * weighted_residual_loss)


def loss_old(all_k, n_tail=7, slope_weight=2.0):
    """Old loss: focuses on last n_tail points against rounded integer sequence."""
    all_k = np.asarray(all_k, dtype=float)
    if len(all_k) <= n_tail + 5:
        return float('inf')

    tail_k = all_k[-n_tail:]
    tail_indices = np.arange(len(tail_k), dtype=float)

    slope_tail, _ = np.polyfit(tail_indices, tail_k, 1)
    if slope_tail == 0:
        return float('inf')

    ideal_spacing_tail = np.round(slope_tail)
    ideal_intercept_tail = np.mean(tail_k - ideal_spacing_tail * tail_indices)
    ideal_tail_sequence = ideal_spacing_tail * tail_indices + np.round(ideal_intercept_tail)
    deviations = tail_k - ideal_tail_sequence

    mse_loss = np.mean(deviations ** 2)
    deviation_slope, _ = np.polyfit(tail_indices, deviations, 1)
    slope_loss = deviation_slope ** 2

    return mse_loss + slope_weight * slope_loss


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic test sequences  (25 points each, matching the k50-65 run)
# ══════════════════════════════════════════════════════════════════════════════

n = 25
base = 335           # first integer in the sequence
idx = np.arange(n)

sequences = {
    # ── Ideal cases ──────────────────────────────────────────────────────────
    "perfect"       : base + nu_spacing * idx,                    # exact integers
    "good_offset"   : base + 0.01 + nu_spacing * idx,            # uniform +0.01 offset

    # ── The current optimizer result ─────────────────────────────────────────
    # Slope ≈ 3.994, intercept ≈ 335.035; fractional parts drift 0.035 → -0.085
    "current_bad"   : np.array([
        335.03466316, 339.02887881, 343.02382401, 347.01823271,
        351.01337063, 355.00811609, 359.00332084, 362.99813571,
        366.99355529, 370.98856509, 374.98379253, 378.97901936,
        382.97448875, 386.96949705, 390.96489636, 394.95999635,
        398.95539642, 402.95030159, 406.94565956, 410.94049052,
        414.93572527, 418.93042911, 422.92543315, 426.92036762,
        430.91474128,
    ]),

    # ── Other bad cases ───────────────────────────────────────────────────────
    "large_slope"   : base + 4.1 * idx,                           # spacing 4.1, exact integers missed
    "half_integer"  : base + 0.5 + nu_spacing * idx,             # all at x.5, worst integer offset
    "jitter"        : base + nu_spacing * idx + 0.3 * np.sin(idx),  # nonlinear oscillation
    "jitter_bad"    : base + nu_spacing * idx + 0.3 * np.random.default_rng(0).standard_normal(n),

    # ── Edge cases ────────────────────────────────────────────────────────────
    "slope_3.99_int_anchor": 335.0 + 3.99 * idx,   # slope off but starts exactly at integer
    "slope_4.00_offset_0.4": base + 0.4 + nu_spacing * idx,  # perfect slope, large offset
}


# ══════════════════════════════════════════════════════════════════════════════
#  Run and display
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print(f"{'Sequence':<28}  {'loss_new':>12}  {'loss_old':>12}  {'new<old?':>8}")
print("-" * 72)

results = {}
for name, seq in sequences.items():
    ln = loss_new(seq)
    lo = loss_old(seq)
    results[name] = (ln, lo)
    flag = "✓" if ln < lo or abs(ln - lo) < 1e-10 else " "
    print(f"  {name:<26}  {ln:12.6e}  {lo:12.6e}  {flag}")

print("=" * 72)

# ── Ranking check: new loss should rank sequences from best to worst ──────
print("\nNew-loss ranking (ascending = better):")
ranked = sorted(results.items(), key=lambda kv: kv[1][0])
for rank, (name, (ln, _)) in enumerate(ranked, 1):
    print(f"  {rank:2d}. {name:<28}  {ln:.6e}")

# ── Specific assertions ────────────────────────────────────────────────────
print("\nAssertions:")

def check(condition, msg_pass, msg_fail):
    if condition:
        print(f"  PASS  {msg_pass}")
    else:
        print(f"  FAIL  {msg_fail}")

ln = {k: v[0] for k, v in results.items()}

check(ln["perfect"] < 1e-20,
      "perfect sequence → loss ≈ 0",
      f"perfect sequence should have loss≈0, got {ln['perfect']:.3e}")

check(ln["good_offset"] < ln["current_bad"],
      "good_offset (uniform +0.01) beats current_bad (drifting)",
      f"good_offset={ln['good_offset']:.3e} should be < current_bad={ln['current_bad']:.3e}")

check(ln["perfect"] < ln["good_offset"] < ln["current_bad"],
      "ranking: perfect < good_offset < current_bad",
      f"ranking broken: perfect={ln['perfect']:.3e}, "
      f"good_offset={ln['good_offset']:.3e}, current_bad={ln['current_bad']:.3e}")

check(ln["half_integer"] > ln["good_offset"],
      "half_integer offset is worse than small uniform offset",
      f"half_integer={ln['half_integer']:.3e} should be > good_offset={ln['good_offset']:.3e}")

check(ln["perfect"] < ln["jitter"],
      "nonlinear jitter is worse than perfect",
      f"jitter={ln['jitter']:.3e} should be > perfect={ln['perfect']:.3e}")

check(ln["slope_4.00_offset_0.4"] > ln["good_offset"],
      "large fractional offset (0.4) is worse than small offset (0.01)",
      f"offset_0.4={ln['slope_4.00_offset_0.4']:.3e} should be > "
      f"good_offset={ln['good_offset']:.3e}")

# ── Show per-k deviations for current_bad vs good_offset ──────────────────
print("\nPer-k deviations from ideal integer sequence:")
for name in ("perfect", "good_offset", "current_bad"):
    seq = sequences[name]
    ideal = np.round(np.mean(seq - nu_spacing * idx)) + nu_spacing * idx
    devs = seq - ideal
    print(f"  {name:<26}: min={devs.min():.4f}  max={devs.max():.4f}  "
          f"rms={np.sqrt(np.mean(devs**2)):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 — Weight-balance analysis
#
#  For a perturbation of size ε, two "equally bad" error types are:
#    Case A: slope=4 exactly, uniform offset ε  (every k_i is ε from an integer)
#    Case B: midpoint anchored at integer, slope error ε  (slope = 4+ε, so
#            deviations grow linearly: d_i = ε*(i - mid), vanishing at i=mid)
#
#  Good weights → both cases produce similar total loss for the same ε.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("Section 2 — Weight balance analysis")
print("=" * 72)

eps = 0.005          # perturbation magnitude to test
mid = (n - 1) // 2  # index 12 for n=25

# ── Build the two sequences ────────────────────────────────────────────────
# Case A: uniform offset ε, slope = 4
seq_A = base + eps + nu_spacing * idx

# Case B: slope = 4+ε, k at midpoint = exact integer
k_mid_B = base + nu_spacing * mid          # exact integer at midpoint
seq_B = k_mid_B + (nu_spacing + eps) * (idx - mid)

print(f"\nPerturbation ε = {eps}")
print(f"  Case A: slope=4, uniform offset +{eps}  → k_i = {base}+{eps} + 4*i")
print(f"  Case B: slope=4+{eps}={nu_spacing+eps}, midpoint i={mid} is integer "
      f"→ deviations d_i = {eps}*(i-{mid})")

# ── Per-k deviations ──────────────────────────────────────────────────────
for label, seq in [("A", seq_A), ("B", seq_B)]:
    ideal = np.round(np.mean(seq - nu_spacing * idx)) + nu_spacing * idx
    devs = seq - ideal
    print(f"\n  Case {label} deviations from ideal integer sequence:")
    print(f"    range [{devs.min():.4f}, {devs.max():.4f}], "
          f"mean={devs.mean():.4f}, rms={np.sqrt(np.mean(devs**2)):.4f}")

# ── Raw (unweighted) loss components ─────────────────────────────────────
_, sl_A, pk_A, res_A = loss_new_components(seq_A)
_, sl_B, pk_B, res_B = loss_new_components(seq_B)

print(f"\n  Raw (unweighted) loss components:")
print(f"  {'Component':<28} {'Case A':>12} {'Case B':>12} {'ratio B/A':>10}")
print(f"  {'-'*64}")
for comp, a, b in [("slope_loss (δ-4)²",        sl_A,  sl_B),
                   ("per_k_integer_loss",          pk_A,  pk_B),
                   ("weighted_residual_loss",       res_A, res_B)]:
    ratio = b / a if a > 1e-30 else float('inf')
    print(f"  {comp:<28} {a:12.3e} {b:12.3e} {ratio:10.1f}×")

# ── Analytical explanation ────────────────────────────────────────────────
var_factor = np.mean((idx - mid) ** 2)
print(f"\n  Analytical variance factor for n={n}, mid={mid}:")
print(f"    mean((i-mid)²) = {var_factor:.1f}")
print(f"  → per_k_integer_loss(B) / per_k_integer_loss(A) = {var_factor:.1f} "
      f"(regardless of weights)")
print(f"  This ratio is fixed by the sequence length; it cannot be removed by")
print(f"  tuning weights. Setting w_slope=0 minimises the extra imbalance from")
print(f"  the slope_loss term.")

# ── Sweep of w_slope values ───────────────────────────────────────────────
print(f"\n  Total-loss balance for several w_slope choices "
      f"(w_intercept=1, w_residual=1, ε={eps}):")
print(f"  {'w_slope':>10}  {'loss_A':>12}  {'loss_B':>12}  {'ratio B/A':>10}  note")
print(f"  {'-'*66}")

w_slope_candidates = [500.0, 100.0, 10.0, 1.0, 0.0]
for ws in w_slope_candidates:
    la = loss_new(seq_A, w_slope=ws)
    lb = loss_new(seq_B, w_slope=ws)
    ratio = lb / la if la > 1e-30 else float('inf')
    balanced = "✓ within 10×" if ratio < 10 else ("≈ within 100×" if ratio < 100 else "✗ imbalanced")
    print(f"  {ws:>10.1f}  {la:12.3e}  {lb:12.3e}  {ratio:10.1f}×  {balanced}")

# ── Recommendation ────────────────────────────────────────────────────────
print(f"""
  Recommendation
  ──────────────
  With the new per_k_integer_loss, a slope error of ε at n={n} modes already
  causes per_k_integer_loss ≈ ε² × {var_factor:.0f} (drift accumulates over modes),
  while a uniform offset of ε causes per_k_integer_loss = ε².
  This {var_factor:.0f}× ratio is intrinsic and cannot be removed by weights.

  Current w_slope=500 adds an extra ×500 boost to slope errors, making the
  total imbalance ≈ {500 + var_factor:.0f}×.

  Setting w_slope ≈ 1 (= w_intercept) reduces the total imbalance to ≈{var_factor:.0f}×,
  which is the irreducible minimum given that slope errors accumulate over {n} modes.

  If you want stricter balance (ratio < 10×), consider normalising
  per_k_integer_loss by the variance factor:
    normalised_per_k_loss = mean(d_i²) / {var_factor:.0f}
  Then loss_A = ε²/{var_factor:.0f} and loss_B ≈ ε² (ratio ≈ {var_factor:.0f}×/{var_factor:.0f} = 1).
  But note: this de-weights the accumulated drift that IS the real signal.
""")
