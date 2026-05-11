# Loss Function Modifications - Focus on Stabilized High-K Tail

## Problem Identified

The previous `calculate_integer_loss()` function used data from index 8 onwards, which includes:
- **Mid-K modes** (indices 8-20): Large oscillations that haven't stabilized yet
- **High-K modes** (indices 20+): Stabilized oscillations but with a trend

Looking at the "Difference between Datasets" plot in `kvalues_comparison.pdf`:
- Early indices show large wiggles (±0.08)
- Mid indices show decreasing oscillations
- **Final ~10-15 points show stabilized behavior but with an upward slope**

The goal is to find parameters where the **final stabilized points** have:
1. **Small deviations** from ideal integer spacing
2. **Zero slope** (flat trend, not trending up or down)

## Old Loss Function Behavior

```python
# OLD: Used all high-k data (after n_ignore=8)
high_k = allowedK_integer[8:]  # Includes unstable mid-K modes
squared_errors = (high_k - ideal_k_sequence)**2
weights = np.linspace(0.1, 1.0, len(high_k))  # Linear weighting
loss = mean(weights * squared_errors)
```

**Problems**:
1. Includes mid-K modes that haven't converged to integer spacing
2. Linear weighting still gives significant weight to unstable regions
3. Doesn't penalize the **slope** of deviations (trending behavior)

## New Loss Function Strategy

### Step 1: Determine Ideal Spacing (Unchanged)
Use indices 8+ to fit and determine the asymptotic spacing:
```python
high_k_for_fit = allowedK_integer[8:]
slope = polyfit(indices, high_k_for_fit, 1)
ideal_spacing = round(slope)  # e.g., 4
```

### Step 2: Focus ONLY on Tail
Select the last `n_tail` points (default: 10) where oscillations have stabilized:
```python
tail_k = allowedK_integer[-n_tail:]  # Last 10 points only
```

### Step 3: Compute Deviations from Ideal
```python
ideal_tail = ideal_spacing * [0, 1, 2, ...] + offset
deviations = tail_k - ideal_tail
```

### Step 4: Two Loss Components

**Component 1: Mean Squared Deviation**
```python
mse_loss = mean(deviations^2)
```
Penalizes how far the points are from ideal spacing.

**Component 2: Slope of Deviations** (NEW!)
```python
deviation_slope = polyfit(deviations)  # Fit line to deviations
slope_loss = deviation_slope^2
```
Penalizes trending behavior. We want the deviations to be **flat** (slope ≈ 0).

### Step 5: Combined Loss
```python
total_loss = mse_loss + slope_weight * slope_loss
```

The `slope_weight` (default: 10.0) emphasizes the importance of having flat deviations.

## Implementation

### Modified Function Signature
```python
def calculate_integer_loss(params, folder_path,
                          n_ignore=8,      # For fitting ideal spacing
                          n_tail=10,       # Focus on last N points
                          slope_weight=10.0)  # Weight for slope penalty
```

### New Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_ignore` | 8 | Number of initial points to skip when determining ideal spacing |
| `n_tail` | 10 | Number of final points to use for loss (where wiggles stabilize) |
| `slope_weight` | 10.0 | Multiplier for slope penalty (higher = more emphasis on flat deviations) |

### Diagnostic Output

The function now prints detailed diagnostics:
```
allowedK_integer = [2.1, 2.5, ...]
  Ideal spacing from fit: 4 (slope: 4.003)
  Tail deviations: min=-0.015, max=0.023, mean=0.004
  MSE loss: 3.2e-04, Deviation slope: 0.0038, Slope loss: 1.4e-05
  Total loss: 4.6e-04
```

## Testing the New Loss Function

Run the test script to visualize the behavior:
```bash
cd find_integer_allowedK
source ../../HO_env/bin/activate  # Or your environment
python test_new_loss_function.py
```

This will:
1. Load your current allowedK data
2. Compare old vs new loss calculations
3. Test different `n_tail` values (8, 10, 12, 15)
4. Generate visualizations showing:
   - allowedK vs ideal sequence
   - Full deviation plot with tail regions highlighted
   - Tail deviations for different n_tail (showing slopes)
   - Loss components comparison

### Expected Output Plot

The plot `loss_function_comparison.pdf` will show:
- **Top left**: Your allowedK data vs ideal integer sequence
- **Top right**: Full deviations with different tail regions marked
- **Bottom left**: Tail deviations with fitted slopes (goal: horizontal line at y=0)
- **Bottom right**: Bar chart comparing loss components for different n_tail values

## Tuning Parameters

### Choosing `n_tail`

- **Too small** (n_tail < 8): Not enough points to reliably detect slope
- **Too large** (n_tail > 15): Includes mid-K oscillations that haven't stabilized
- **Recommended**: 10-12 points

Run `test_new_loss_function.py` to find the optimal value for your data.

### Choosing `slope_weight`

- **Low weight** (1-5): Focuses more on absolute deviation, less on trend
- **Medium weight** (10): Balanced (default)
- **High weight** (20-50): Strongly enforces flat deviations, may tolerate larger offsets

## Expected Improvement

With the new loss function, the Bayesian optimization will:

1. **Ignore noisy mid-K region**: Doesn't try to fit oscillations that haven't converged yet
2. **Minimize tail deviations**: Makes the final points close to ideal integer values
3. **Flatten the trend**: Ensures the deviations don't slope up or down

### Visual Goal

In the "Difference between Datasets" plot, you want the final ~10 points to:
- Cluster tightly around **y = 0** (small deviations)
- Form a **horizontal band** (zero slope)

Currently, they trend from ~-0.02 to ~+0.03 (upward slope). The new loss function will penalize this trend heavily.

## Running the Optimization

The modification is already applied to `find_intK_baye_opt_planck_bounds.py`. Just run:
```bash
python find_intK_baye_opt_planck_bounds.py
```

The optimizer will now search for parameters that minimize both the deviation and slope of the final stabilized points.

## References

- See `kvalues_comparison.pdf` for the current behavior
- See `test_new_loss_function.py` for testing and visualization
- See `find_intK_baye_opt_planck_bounds.py` for the main optimization script
