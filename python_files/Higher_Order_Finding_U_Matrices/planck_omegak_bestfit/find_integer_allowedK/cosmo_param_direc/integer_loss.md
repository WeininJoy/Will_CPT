# Integer Loss Function — `_old` vs current

Functions compared:
- `calculate_integer_loss_old` — original (`find_intK_baye_opt_planck_bounds_optuna.py:166`)
- `calculate_integer_loss` — current, used in optimisation (`find_intK_baye_opt_planck_bounds_optuna.py:104`)

---

## 1. Trimming

| | `_old` | current |
|---|---|---|
| Scan direction | Backward from end to middle; stops at first *good* spacing (`break`) | Backward from end to middle; no `break` (scans all the way back) |
| `spacing_tol_factor` | 8 (lenient) | 3 (stricter) |

The current version does not break early, so it finds the innermost bad spacing rather than the outermost one, resulting in a shorter but cleaner stable sequence.

---

## 2. Smoothing

| | `_old` | current |
|---|---|---|
| Applied? | No — works on raw k | Yes — 2-point midpoint filter: `k_smooth[i] = (k[i] + k[i+1]) / 2` |

The midpoint filter suppresses alternating-sign wiggles before computing spacings and integer deviations. All subsequent steps in the current version operate on `k_smooth`, not the raw sequence.

---

## 3. Weighting scheme

| | `_old` | current |
|---|---|---|
| Shape | Linear: `w_i = i + 1` | Plateau: `1 − (1−z)²` where `z ∈ [0, 1]` linearly |
| Normalised? | Only where explicitly needed | Yes, normalised to sum to 1 for both slope and integer weights |

The plateau weight rises quickly and saturates near the tail, giving roughly equal emphasis to the upper half of the sequence while still down-weighting early points.

---

## 4. Slope loss

| | `_old` | current |
|---|---|---|
| Method | Global linear fit via `np.polyfit` (weighted); single scalar `(fitted_slope − 4)²` | Weighted sum of per-spacing deviations on smoothed sequence: `Σ wᵢ (Δk_smooth,i − 4)²` |

The old version reduces slope information to one number. The current version measures per-spacing deviation, making it sensitive to local non-convergence even when the global slope looks acceptable.

---

## 5. Intercept estimation for integer loss

| | `_old` | current |
|---|---|---|
| Sequence used | Raw `stable_k` | Smoothed `k_smooth` |
| Method | Unweighted mean: `mean(k − 4·i)` | Unweighted mean: `mean(k_smooth − 4·i)` |

Both versions use an unweighted mean to anchor the ideal integer grid. The only difference is that the current version anchors on the smoothed sequence.

---

## 6. Default hyperparameters

| | `_old` | current |
|---|---|---|
| `w_slope` | 5 | 20 |
| `w_integer` / `w_intercept` | 1 | 1 |
| `w_residual` | 0 (disabled) | — (removed entirely) |
| `spacing_tol_factor` | 8 | 3 |

The residual loss term present in `_old` (measuring scatter around the linear fit) was removed in the current version.
