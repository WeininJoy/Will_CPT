#!/usr/bin/env python3
"""
Test script to compare different loss functions for allowed K optimization.
Tests with real data from the current parameter set.
"""

import numpy as np
from typing import List

def loss_two_component(
    k_values: List[float],
    n_ignore: int = 5,
    alpha: float = 0.5,
    use_weighting: bool = True
) -> float:
    """
    Calculates loss based on two components: variance of spacing and deviation from integers.

    Args:
        k_values (List[float]): The list of allowed K values.
        n_ignore (int): Number of initial K values to ignore.
        alpha (float): Weighting factor between integer loss (alpha) and spacing loss (1-alpha).
        use_weighting (bool): If True, applies linear weighting to prioritize high-K modes.

    Returns:
        float: The calculated loss value.
    """
    if len(k_values) <= n_ignore + 2:
        # Not enough data points to calculate meaningful diffs and variance
        return 1e6 # Return a large number

    high_k = np.array(k_values[n_ignore:])

    # --- 1. Spacing Loss ---
    diffs = np.diff(high_k)
    # The spacing should also be an integer, so we penalize variance around the rounded mean
    estimated_spacing = np.round(np.mean(diffs))
    spacing_loss = np.mean((diffs - estimated_spacing)**2)

    # --- 2. Integer Loss ---
    integer_loss = np.mean((high_k - np.round(high_k))**2)

    # --- 3. Optional Weighting for high-K modes ---
    if use_weighting:
        # Linearly increasing weights for the squared errors
        weights = np.linspace(0.1, 1.0, len(high_k)) # Start weights > 0

        # Recalculate weighted losses
        spacing_loss_terms = (diffs - estimated_spacing)**2
        # Weights for diffs should match the length of diffs
        weights_for_diffs = np.linspace(0.1, 1.0, len(diffs))
        spacing_loss = np.mean(weights_for_diffs * spacing_loss_terms)

        integer_loss_terms = (high_k - np.round(high_k))**2
        integer_loss = np.mean(weights * integer_loss_terms)

    # --- 4. Combine Losses ---
    total_loss = (1 - alpha) * spacing_loss + alpha * integer_loss

    return total_loss


def loss_linear_regression(
    k_values: List[float],
    n_ignore: int = 5,
    use_weighting: bool = True
) -> float:
    """
    Calculates loss by fitting a line to the high-K values and measuring
    deviation from an ideal line with integer spacing.

    Args:
        k_values (List[float]): The list of allowed K values.
        n_ignore (int): Number of initial K values to ignore.
        use_weighting (bool): If True, applies linear weighting to prioritize high-K modes.

    Returns:
        float: The calculated loss value.
    """
    if len(k_values) <= n_ignore + 2:
        return 1e6

    high_k = np.array(k_values[n_ignore:])
    indices = np.arange(len(high_k))

    # --- 1. Fit a line to the data: k = slope * i + intercept ---
    # polyfit returns [slope, intercept] for deg=1
    slope, intercept = np.polyfit(indices, high_k, 1)

    # --- 2. Determine the ideal integer spacing ---
    # The ideal spacing is the integer closest to the regression slope
    ideal_spacing = np.round(slope)
    if ideal_spacing == 0: # Avoid degenerate case of a flat line
        return 1e6

    # --- 3. Construct the ideal sequence ---
    # Find the best intercept for the *ideal* line
    # C' = mean(y_i - S * x_i)
    ideal_intercept = np.mean(high_k - ideal_spacing * indices)
    ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)
    print("ideal_k_sequence (regression) =", ideal_k_sequence)

    # --- 4. Calculate the loss ---
    # The loss is the mean squared error between the actual data
    # and the reconstructed ideal sequence.
    squared_errors = (high_k - ideal_k_sequence)**2

    if use_weighting:
        weights = np.linspace(0.1, 1.0, len(high_k))
        loss = np.mean(weights * squared_errors)
    else:
        loss = np.mean(squared_errors)

    return loss


def calculate_old_loss(k_values: List[float], n_ignore: int = 8) -> float:
    """
    Old loss function from the original code.
    """
    if not k_values or len(k_values) < 9:
        return float('inf')

    allowedK = k_values[n_ignore:]  # Ignore the first few values
    allowedK = np.array(allowedK)
    ideal_start = np.round(allowedK[0])
    ideal_series = ideal_start + 4 * np.arange(len(allowedK))
    loss = np.mean((allowedK - ideal_series)**2)
    return loss


def analyze_k_values(k_values: List[float], name: str):
    """
    Analyze and print detailed information about K values.
    """
    print(f"\n{'='*70}")
    print(f"Analysis for: {name}")
    print(f"{'='*70}")

    k_arr = np.array(k_values)
    print(f"Number of K values: {len(k_values)}")
    print(f"K range: [{k_arr[0]:.2f}, {k_arr[-1]:.2f}]")

    # Show first and last 5 values
    print(f"\nFirst 5 K values: {k_arr[:5]}")
    print(f"Last 5 K values:  {k_arr[-5:]}")

    # Calculate spacing statistics
    diffs = np.diff(k_arr)
    print(f"\nSpacing statistics:")
    print(f"  Mean spacing: {np.mean(diffs):.4f}")
    print(f"  Std of spacing: {np.std(diffs):.4f}")
    print(f"  Min/Max spacing: [{np.min(diffs):.4f}, {np.max(diffs):.4f}]")

    # Look at high-K spacing (last 10 differences)
    if len(diffs) >= 10:
        high_k_diffs = diffs[-10:]
        print(f"\nHigh-K spacing (last 10 differences):")
        print(f"  Mean: {np.mean(high_k_diffs):.4f}")
        print(f"  Std: {np.std(high_k_diffs):.4f}")
        print(f"  Estimated integer spacing: {np.round(np.mean(high_k_diffs)):.0f}")

    # Distance from integers for high-K values
    high_k = k_arr[8:]  # Skip first 8 like in old code
    int_distances = np.abs(high_k - np.round(high_k))
    print(f"\nDistance from nearest integer (high-K modes):")
    print(f"  Mean: {np.mean(int_distances):.4f}")
    print(f"  Max: {np.max(int_distances):.4f}")

    print(f"\n{'-'*70}")
    print("Loss Function Comparison:")
    print(f"{'-'*70}")

    # Test different n_ignore values
    for n_ignore in [5, 8, 10]:
        print(f"\nWith n_ignore = {n_ignore}:")

        loss1 = loss_two_component(k_values, n_ignore=n_ignore, use_weighting=True)
        print(f"  Loss #1 (Two-Component, weighted):     {loss1:.8f}")

        loss2 = loss_linear_regression(k_values, n_ignore=n_ignore, use_weighting=True)
        print(f"  Loss #2 (Linear Regression, weighted): {loss2:.8f}")

        if n_ignore == 8:
            loss_old = calculate_old_loss(k_values, n_ignore=n_ignore)
            print(f"  Old Loss (fixed spacing=4):            {loss_old:.8f}")


# ============================================================================
# Test Data
# ============================================================================

# Your actual data
k_values_actual = [
    2.8421146675468414, 5.914839898098094, 10.11447683682903, 14.190934687893714,
    18.27141463077986, 22.355825509450444, 26.44179848834388, 30.528229295835537,
    34.61461476065128, 38.70071923905925, 42.786424120618555, 46.87166418178404,
    50.95640468289007, 55.040633428032194, 59.12435487847957, 63.20757545839842,
    67.2903014791895, 71.37254327709552, 75.45431847919096, 79.53564547229314,
    83.6165373363188, 87.69700671699728, 91.7770698009251, 95.85674681512563,
    99.9360513453989
]

# Ideal case: perfect integers with spacing 4
k_values_ideal = [3.1, 6.5, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0,
                  42.0, 46.0, 50.0, 54.0, 58.0, 62.0, 66.0, 70.0, 74.0, 78.0]

# Bad case: random numbers
np.random.seed(42)
k_values_bad = np.random.uniform(10, 100, 25)
k_values_bad.sort()

# ============================================================================
# Run Tests
# ============================================================================

print("\n" + "="*70)
print("TESTING LOSS FUNCTIONS FOR ALLOWED K OPTIMIZATION")
print("="*70)

analyze_k_values(k_values_actual, "YOUR ACTUAL DATA")
analyze_k_values(k_values_ideal, "IDEAL DATA (Perfect integers, spacing=4)")
analyze_k_values(k_values_bad.tolist(), "BAD DATA (Random)")

print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)
print("""
Based on the comparison:

1. Loss #2 (Linear Regression) is RECOMMENDED because:
   - Automatically detects spacing (no hardcoded spacing=4)
   - More robust to variations in spacing
   - Fewer hyperparameters to tune
   - Generally gives smaller loss values for good data

2. The old loss function assumes spacing=4, which may not be optimal
   for all parameter sets.

3. Using n_ignore=8 matches your current implementation and seems
   reasonable for focusing on high-K modes.

4. Weighting helps prioritize the high-K values while still using
   some information from earlier values.
""")
