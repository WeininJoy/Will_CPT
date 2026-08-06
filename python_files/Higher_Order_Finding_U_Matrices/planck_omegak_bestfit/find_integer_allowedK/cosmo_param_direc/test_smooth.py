import numpy as np
import matplotlib.pyplot as plt

allowedK_integer = [339.08792331783303,343.0835817958383,347.07880726631817,351.0744041261763,355.0699102356517,359.0658013525719,363.06131034131135,367.0573042668195,371.0529298105816,375.04894781500917,379.0447229269657,383.04081914411165,387.036537607856,391.032649926478,395.0282071220621,399.0241315504078,403.019693731806,407.01550550290386,411.01096261356275,415.00661208857406,419.0019774624104,422.99745139179583,426.9925228981608,430.9878782962183,434.98280019322783]

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

# ── synthetic sequences for testing ───────────────────────────────────────────
tail_slope_loss, tail_integer_loss, total_loss = calculate_integer_loss_smooth(allowedK_integer)
print(f"Tail Slope Loss: {tail_slope_loss:.4e}")
print(f"Tail Integer Loss: {tail_integer_loss:.4e}")
print(f"Total Loss: {total_loss:.4e}")
plt.legend()
plt.title('Allowed K Sequence with Smoothing')
plt.xlabel('Index')
plt.ylabel('spacing')
plt.show()