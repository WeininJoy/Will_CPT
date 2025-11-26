# -*- coding: utf-8 -*-
"""
Evolve perturbations from CLASS output (today, a=1, s=1) forward to FCB (s=0)
Created based on Higher_Order_Finding_U_Matrices.py

Key differences from reference code:
1. Integrates FORWARD from today (s=1) to FCB (s=0) instead of backward
2. Takes initial conditions from CLASS output at z=0
3. Uses dX2_dt equations (s-based) throughout, like the reference code
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from classy import Class
import classy

nu_spacing = 4
l_max_g = 12  # Maximum photon multipole to include (can be changed for convergence tests)
              # Start with 12 for faster testing, increase to 50 for production

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5  # photon density
Neff = 3.046
N_ncdm = 1   # number of massive neutrino species
m_ncdm = 0.06  # mass of massive neutrino species in eV

def cosmological_parameters(mt, kt, h):
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3)) * Omega_gamma_h2/h**2

    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(Omega_r, rt, mt, kt)
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return Omega_lambda, Omega_m, Omega_K

###############################################################################
# Best-fit parameters (matching the reference code)
mt, kt, Omegab_ratio, h, As, ns, tau = 427.161507, 1.532563, 0.155844, 0.543442, 2.108821, 0.965799, 0.052255
OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1061.915  # based on the calculate_z_rec() output
print(f"Cosmological parameters:")
print(f"  OmegaLambda = {OmegaLambda:.6f}")
print(f"  OmegaM = {OmegaM:.6f}")
print(f"  OmegaR = {OmegaR:.6e}")
print(f"  OmegaK = {OmegaK:.6f}")
print(f"  h = {h:.6f}")
###############################################################################

# Set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
# Number of perturbation variables: s(1) + phi,psi(2) + dr,dm,vr,vm(4) + fr2,...,fr_lmax(l_max_g-1 multipoles)
num_variables = 1 + 2 + 4 + (l_max_g - 1)  # = 6 + l_max_g

# Unit conversions to match CLASS
# CLASS uses: H0 = 100*h km/s/Mpc, convert to 1/Mpc by dividing by c
c_light_km_s = 299792.458  # Speed of light in km/s
H0_CLASS = 100.0 * h / c_light_km_s  # H0 in units of 1/Mpc (CLASS units)

# Curvature K in CLASS units (1/Mpc^2)
K_CLASS = -OmegaK * H0_CLASS**2

print(f"  l_max_g = {l_max_g}")
print(f"  num_variables = {num_variables}")
print(f"  H0 (CLASS units) = {H0_CLASS:.6e} Mpc^-1")
print(f"  H0 (km/s/Mpc) = {100.0 * h:.6f}")
print(f"  K (CLASS units) = {K_CLASS:.6e} Mpc^-2")

# For background evolution, we use H0_CLASS
H0 = H0_CLASS

#```````````````````````````````````````````````````````````````````````````````
# BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

def ds_dt(t, s):
    """Inverse scale factor evolution: ds/dt = -H(s)"""
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(s**2) + OmegaM*abs(s**3) + OmegaR*abs(s**4)))

def da_dt(t, a):
    """Scale factor evolution: da/dt = a^2 * H(a)"""
    return a**2 * H0 * np.sqrt((OmegaLambda + OmegaK/abs(a**2) + OmegaM/abs(a**3) + OmegaR/abs(a**4)))

# Find conformal time at FCB by integrating from Big Bang (like reference code)
print('\nFinding FCB conformal time...')

# Estimate typical timescale: t_0 ~ 1/H0
t_scale = 1.0 / H0  # Typical conformal time today
print(f"  Typical timescale t ~ 1/H0 = {t_scale:.1f} Mpc")

# Initial conditions at Big Bang (following reference code)
# Start at much smaller conformal time in CLASS units
t0 = 1e-8 * t_scale  # Scale initial time appropriately
a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda))
a2 = OmegaM/(12*OmegaLambda)
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2))
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2)
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4

swaptime = 0.1 * t_scale  # Switch from a to s at ~10% of t_scale
t_max = 5.0 * t_scale  # Search for FCB up to 5x t_scale

print(f"  Starting at t0 = {t0:.2e} Mpc with a = {a_Bang:.2e}")
print(f"  Switching to s-integration at t = {swaptime:.1f} Mpc")
print(f"  Maximum integration time: {t_max:.1f} Mpc")

def reach_FCB(t, s):
    return s[0]
reach_FCB.terminal = True

# Integrate a from Big Bang to swaptime
sol_a = solve_ivp(da_dt, [t0, swaptime], [a_Bang], max_step=swaptime/1000, method='LSODA', atol=atol, rtol=rtol)

# Then integrate s from swaptime to FCB
s_at_swap = 1.0 / sol_a.y[0][-1]
sol = solve_ivp(ds_dt, [swaptime, t_max], [s_at_swap], max_step=t_max/1000, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)

# Check if t_events[0] is not empty before trying to access its elements
if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"  FCB conformal time: {fcb_time:.6f}")
else:
    print(f"  Warning: Event 'reach_FCB' did not occur.")
    fcb_time = None

if fcb_time is None:
    raise RuntimeError("Could not find FCB time!")

# Find conformal time at today (a=1, s=1) by interpolating
a_today = 1.0
s_today = 1.0
# Combine a and s solutions
t_combined = np.concatenate([sol_a.t, sol.t])
a_combined = np.concatenate([sol_a.y[0], 1.0/sol.y[0]])

# Find index where a is closest to 1
idx_today = np.argmin(np.abs(a_combined - a_today))
t_today = t_combined[idx_today]
a_actual = a_combined[idx_today]
s_actual = 1.0 / a_actual
print(f"  Today: t = {t_today:.6f}, a = {a_actual:.6f}, s = {s_actual:.6f}")
print(f"  Time from today to FCB: Δt = {fcb_time - t_today:.6f}")

#```````````````````````````````````````````````````````````````````````````````
# RUN CLASS TO GET INITIAL CONDITIONS AT TODAY
#```````````````````````````````````````````````````````````````````````````````

print('\nRunning CLASS to get perturbations...')

# k values to process (in h/Mpc, CLASS units)
k_values_CLASS = [0.01]  # h/Mpc  # Start with just one k for testing

params = {
    # IMPORTANT: For closed universe with arbitrary k values,
    # request ONLY transfer functions, NOT CMB power spectra (tCl, pCl, etc.)
    # This bypasses the hyperspherical Bessel computation that requires quantized k
    'output': 'dTk, vTk',  # Density and velocity transfer functions (NOT tCl!)
    'modes': 's',           # Scalar perturbations
    'ic': 'ad',            # Adiabatic initial conditions
    'gauge': 'newtonian',  # Newtonian gauge

    # Cosmological parameters
    # IMPORTANT: CLASS expects omega (lowercase) = physical density = Omega * h^2
    'h': h,
    'omega_b': Omegab_ratio * OmegaM * h**2,      # Physical baryon density
    'omega_cdm': (1.-Omegab_ratio) * OmegaM * h**2,  # Physical CDM density
    'Omega_k': float(OmegaK),
    'A_s': As*1e-9,
    'n_s': ns,
    'tau_reio': tau,

    # Massive neutrinos
    'N_ncdm': N_ncdm,
    'm_ncdm': m_ncdm,
    'N_ur': Neff - N_ncdm,

    # CRITICAL: For closed universe, must specify nu_spacing
    'nu_spacing': nu_spacing,

    # Limit k-range to prevent overflow in primordial k-grid calculation
    'P_k_max_1/Mpc': max(k_values_CLASS) * 1.5,

    # Precision parameters to control multipole hierarchy
    'l_max_g': l_max_g,        # Maximum photon temperature multipole (default: 12)
    'l_max_pol_g': l_max_g,    # Maximum photon polarization multipole (default: 10)

    # Arbitrary k values - no quantization required for perturbations-only output!
    'k_output_values': ','.join(map(str, k_values_CLASS)),
}

cosmo = Class()
cosmo.set(params)
cosmo.compute()
print("  CLASS run complete.")

# Extract perturbations
perturbations = cosmo.get_perturbations()
scalar_perturbations = perturbations['scalar']
print(f"  Number of k-modes: {len(scalar_perturbations)}")

#```````````````````````````````````````````````````````````````````````````````
# UNIT CONVERSIONS
#```````````````````````````````````````````````````````````````````````````````
# CLASS uses:
#   - k in h/Mpc (from k_output_values)
#   - conformal time tau in Mpc
#   - H in km/s/Mpc
#   - H0/c in 1/Mpc where c = 299792.458 km/s
#
# We now use CLASS units throughout to ensure consistency:
#   - k in 1/Mpc (convert from h/Mpc by multiplying by h)
#   - tau in Mpc
#   - H in 1/Mpc (H/c in CLASS conventions)

print('\nExtracting initial conditions from CLASS output...')

def extract_initial_conditions_at_today(k_data, k_requested):
    """
    Extract initial conditions at today (a=1, s=1) from CLASS output

    Parameters:
    -----------
    k_data : dict
        CLASS perturbation data for one k-mode
    k_requested : float
        The k value we requested (from k_values_CLASS), in h/Mpc

    Returns:
    --------
    k_value : float
        Wavenumber in 1/Mpc
    initial_state : array
        [phi, psi, dr, dm, vr, vm, fr2, fr3, fr4, ..., fr_l_max_g] at a=1
    """
    # Get arrays from CLASS
    a_array = k_data['a']

    # Find index closest to today (a=1)
    idx_today = np.argmin(np.abs(a_array - 1.0))
    a_today_actual = a_array[idx_today]
    print(f"    Using data at a = {a_today_actual:.6f} (closest to a=1)")

    # Use the k value we requested (CLASS doesn't include it in 'dTk, vTk' output)
    # Convert from h/Mpc to 1/Mpc
    k_value = k_requested * h  # in 1/Mpc

    # Extract perturbation variables at today
    # Note: 'dTk, vTk' output doesn't include phi/psi, so set them to zero initially
    phi = k_data['phi'][idx_today] if 'phi' in k_data else 0.0
    psi = k_data['psi'][idx_today] if 'psi' in k_data else 0.0

    # Matter perturbations (combine CDM and baryons with proper weighting)
    delta_cdm = k_data['delta_cdm'][idx_today] if 'delta_cdm' in k_data else 0.0
    delta_b = k_data['delta_b'][idx_today] if 'delta_b' in k_data else 0.0
    theta_cdm = k_data['theta_cdm'][idx_today] if 'theta_cdm' in k_data else 0.0
    theta_b = k_data['theta_b'][idx_today] if 'theta_b' in k_data else 0.0

    # Weighted average for total matter
    omega_cdm = (1. - Omegab_ratio) * OmegaM
    omega_b = Omegab_ratio * OmegaM
    dm = (omega_cdm * delta_cdm + omega_b * delta_b) / (omega_cdm + omega_b)
    vm = (omega_cdm * theta_cdm + omega_b * theta_b) / (omega_cdm + omega_b)

    # Radiation perturbations
    dr = k_data['delta_g'][idx_today]
    vr = k_data['theta_g'][idx_today]
    fr2 = k_data['shear_g'][idx_today]

    # Higher multipoles (l=3 to l=l_max_g)
    multipoles = [fr2]  # Start with fr2 (l=2)
    for l in range(3, l_max_g + 1):  # fr3, fr4, ..., fr_l_max_g
        key = f'Fr{l}_g'
        if key in k_data:
            multipoles.append(k_data[key][idx_today])
            if l == 3:  # Print first for debugging
                print(f"      Found multipole {key} = {k_data[key][idx_today]:.6e}")
        else:
            # If not available, set to zero
            multipoles.append(0.0)
            if l == 3:
                print(f"      Warning: multipole {key} not found, using 0")

    # Construct initial state vector (excluding s, which evolves separately)
    initial_state = np.array([phi, psi, dr, dm, vr, vm] + multipoles)

    print(f"      Total variables in initial state: {len(initial_state)}")
    print(f"      Expected num_variables-1 (without s): {num_variables-1}")

    # Verify size matches expected (num_variables - 1 because s is added later)
    expected_size = num_variables - 1  # Don't count s here
    if len(initial_state) != expected_size:
        print(f"      Warning: Size mismatch! Got {len(initial_state)}, expected {expected_size}")
        # Pad or truncate to match
        if len(initial_state) < expected_size:
            initial_state = np.pad(initial_state, (0, expected_size - len(initial_state)), 'constant')
        else:
            initial_state = initial_state[:expected_size]

    return k_value, initial_state

#```````````````````````````````````````````````````````````````````````````````
# PERTURBATION EQUATIONS (FORWARD INTEGRATION using s, matching dX2_dt_local)
#```````````````````````````````````````````````````````````````````````````````

def dX_dt_forward_s(t, X, k):
    """
    Perturbation equations in s = 1/a for FORWARD integration from today (s=1) to FCB (s=0)
    Equations extracted from CLASS perturbations.c (lines 9000-9096) and converted for forward evolution

    State vector X = [s, phi, psi, dr, dm, vr, vm, fr2, fr3, fr4, ..., fr_lmax]
    Indices: s=0, phi=1, psi=2, dr=3, dm=4, vr=5, vm=6, fr2=7, fr3=8, ..., fr_lmax=6+l_max_g-1

    Equations are in Newtonian gauge with conformal time derivatives
    """
    s = X[0]
    a = 1.0 / s if s > 1e-10 else 1e10

    # Extract variables
    phi = X[1]
    psi = X[2]
    dr = X[3]    # delta_g (photon density)
    dm = X[4]    # delta_m (matter density, combined CDM+baryons)
    vr = X[5]    # theta_g (photon velocity)
    vm = X[6]    # theta_m (matter velocity)
    fr2 = X[7]   # shear_g (photon shear, l=2)

    # Higher multipoles fr3, fr4, ..., fr_lmax are in X[8] to X[6+l_max_g-1]

    # Background quantities (in terms of s)
    # H = a'/a^2 = -s' (Hubble parameter in conformal time)
    H = H0 * np.sqrt((OmegaLambda + OmegaK*s**2 + OmegaM*s**3 + OmegaR*s**4))

    # s' = -H
    s_prime = -H

    # s'/s = -H/s
    s_prime_over_s = -H / s

    # a'/a = H/s (since a = 1/s)
    a_prime_over_a = H / s

    # ds/dτ (conformal time evolution)
    sdot = s_prime  # = -H

    # Energy densities
    rho_m = 3 * H0**2 * OmegaM * s**3
    rho_r = 3 * H0**2 * OmegaR * s**4

    # Metric quantities (Newtonian gauge)
    k2 = k**2
    s2_squared = 1.0 - 3.0 * K_CLASS / k2  # For curved space
    metric_euler = k2 * psi

    # Thomson scattering rate (kappa')
    # After recombination (today forward), essentially zero
    kappa_prime = 0.0  # Free-streaming regime

    # Curvature factors s_l for multipole hierarchy (CLASS uses these for curved universes)
    # s_l[l] = sqrt(1 - l*(l+1)*K/k^2) for curved space, = 1 for flat space
    K = K_CLASS  # Curvature in Mpc^-2 (CLASS units)
    s_l = {}
    for l in range(2, l_max_g + 2):
        if k2 > l*(l+1)*abs(K):
            s_l[l] = np.sqrt(1.0 - l*(l+1)*K/k2)
        else:
            s_l[l] = 1.0  # Avoid imaginary numbers

    # Cotangent function for hierarchy truncation (CLASS line 8820)
    # τ ≈ 1/(H*s) in terms of s
    if abs(K) < 1e-10:
        cotKgen = H * s / k  # Flat space: 1/(k*τ) where τ ≈ 1/(H*s)
    elif K < 0:
        sqrt_absK = np.sqrt(abs(K))
        tau = 1.0 / (H * s)  # Approximate conformal time
        cotKgen = sqrt_absK / k / np.tanh(sqrt_absK * tau)
    else:
        sqrt_absK = np.sqrt(K)
        tau = 1.0 / (H * s)
        cotKgen = sqrt_absK / k / np.tan(sqrt_absK * tau)

    # Total stress-energy perturbations
    rho_plus_p_theta = (4.0/3.0) * rho_r * vr + rho_m * vm
    rho_plus_p_shear = (4.0/3.0) * rho_r * fr2

    # Metric perturbations (Einstein equations, CLASS lines 6526-6529)
    # Convert a^2 to 1/s^2 since a = 1/s
    # psi = phi - 4.5 * a^2/k^2 * rho_plus_p_shear = phi - 4.5/(s^2*k^2) * rho_plus_p_shear
    psi_computed = phi - 4.5 / (s**2 * k2) * rho_plus_p_shear

    # phi' (CLASS line 6529)
    # Original: phi' = -a'/a * psi + 1.5 * a^2/k^2 * rho_plus_p_theta
    # With a = 1/s: phi' = -H/s * psi + 1.5/(s^2*k^2) * rho_plus_p_theta
    phi_prime = -a_prime_over_a * psi + 1.5 / (s**2 * k2) * rho_plus_p_theta
    metric_continuity = -3.0 * phi_prime

    phidot = phi_prime
    psidot = phi_prime  # Simplified; psi follows from constraint

    # Photon perturbations (CLASS lines 9001, 9062)
    # delta_g (photon density, l=0)
    drdot = -4.0/3.0 * (vr + metric_continuity)

    # theta_g (photon velocity, l=1)
    vrdot = (k2 * (dr/4.0 - s2_squared*fr2) + metric_euler)
    # Ignore Thomson scattering since kappa_prime ~ 0 after recombination

    # shear_g (photon shear, l=2 = fr2) (CLASS line 9072)
    fr3 = X[8] if len(X) > 8 else 0.0
    fr2dot = 0.5 * (8.0/15.0 * vr - 3.0/5.0 * k * s_l[3]/s_l[2] * fr3)
    # Ignore Thomson scattering and polarization terms

    # Matter perturbations (simplified, combined CDM+baryons)
    # delta_m (matter density)
    dmdot = -(vm + metric_continuity)

    # theta_m (matter velocity)
    # Original: theta_m' = -a'/a * theta_m + k^2*psi
    # With a'/a = H/s: theta_m' = -H/s * theta_m + k^2*psi
    vmdot = -a_prime_over_a * vm + metric_euler

    # Collect derivatives
    derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

    # Higher photon multipole hierarchy (CLASS lines 9080-9096)
    # For l=3 to l=l_max_g, we need to fill in derivatives for X[8] to X[6+l_max_g-1]
    # That's (l_max_g - 2) derivatives for multipoles l=3 to l=l_max_g

    for l in range(3, l_max_g + 1):
        idx = 6 + (l - 2)  # Index in X: fr2 is at 7 (l=2), fr3 is at 8 (l=3), etc.

        if l < l_max_g:
            # Interior multipoles: standard hierarchy
            fr_l_minus_1 = X[idx - 1] if idx - 1 < len(X) else 0.0
            fr_l_plus_1 = X[idx + 1] if idx + 1 < len(X) else 0.0

            if l == 3:
                # Special case for l=3: fr_l_minus_1 is fr2 which needs factor 2*s_l[2]
                fr_l_dot = k/(2.0*l+1.0) * (l*s_l[l]*2.0*s_l[2]*fr_l_minus_1 - (l+1.0)*s_l[l+1]*fr_l_plus_1)
            else:
                fr_l_dot = k/(2.0*l+1.0) * (l*s_l[l]*fr_l_minus_1 - (l+1.0)*s_l[l+1]*fr_l_plus_1)
        else:
            # Last multipole (l=l_max_g): hierarchy truncation
            fr_l_minus_1 = X[idx - 1] if idx - 1 < len(X) else 0.0
            fr_l = X[idx] if idx < len(X) else 0.0
            fr_l_dot = k * (s_l[l]*fr_l_minus_1 - (1.0 + l)*cotKgen*fr_l)

        derivatives.append(fr_l_dot)

    # Check that we have the right number of derivatives
    if len(derivatives) != len(X):
        print(f"ERROR: derivatives has {len(derivatives)} elements but X has {len(X)} elements!")
        print(f"  Expected {len(X)} derivatives")
        # Pad with zeros if needed
        while len(derivatives) < len(X):
            derivatives.append(0.0)

    return np.array(derivatives)

#```````````````````````````````````````````````````````````````````````````````
# PROCESS EACH K-MODE
#```````````````````````````````````````````````````````````````````````````````

# Storage for results
results = []

for i, k_data in enumerate(scalar_perturbations):
    print(f"\nProcessing k-mode {i+1}/{len(scalar_perturbations)}...")

    # Debug: print available keys
    print(f"  Available keys: {list(k_data.keys())[:15]}")

    # Extract initial conditions at today (pass the requested k value)
    k_value, initial_state = extract_initial_conditions_at_today(k_data, k_values_CLASS[i])
    print(f"  k = {k_value:.6e} (our units)")

    # Construct full initial state with s=1 (today)
    X0 = np.concatenate(([s_today], initial_state))

    print(f"  Initial state at s=1 (today):")
    print(f"    phi = {X0[1]:.6e}")
    print(f"    psi = {X0[2]:.6e}")
    print(f"    dr = {X0[3]:.6e}")
    print(f"    dm = {X0[4]:.6e}")
    print(f"    vr = {X0[5]:.6e}")
    print(f"    vm = {X0[6]:.6e}")
    print(f"    fr2 = {X0[7]:.6e}")
    if num_variables > 7:
        print(f"    fr3 = {X0[8]:.6e}")

    # Define event function to stop at FCB (s=0)
    def at_fcb_forward(t, X):
        """Stop when s reaches 0 (FCB)"""
        if X[0] < stol:
            X[0] = 0
        return X[0]

    at_fcb_forward.terminal = True

    # Integrate forward from today (s=1) to FCB (s=0)
    print(f"  Integrating forward from today (s=1, t={t_today:.2f}) to FCB (s=0, t≈{fcb_time:.2f})...")

    sol = solve_ivp(
        lambda t, X: dX_dt_forward_s(t, X, k_value),
        [t_today, fcb_time],  
        X0,
        method='LSODA',
        atol=atol,
        rtol=rtol,
        events=at_fcb_forward,  
        dense_output=False, 
        max_step=0.1  
    )

    if sol.success:
        print(f"  Integration successful!")
        print(f"    Final time: t = {sol.t[-1]:.6f}")
        print(f"    Final s: s = {sol.y[0,-1]:.6e}")
        print(f"    Final a: a = {1.0/sol.y[0,-1] if sol.y[0,-1] > 1e-10 else np.inf:.6f}")
        print(f"    Final phi: {sol.y[1,-1]:.6e}")
        print(f"    Final psi: {sol.y[2,-1]:.6e}")
    else:
        print(f"  Integration failed: {sol.message}")

    # Store results
    results.append({
        'k': k_value,
        'k_CLASS': k_values_CLASS[i],  # Original requested k value in h/Mpc
        'solution': sol,
        'initial_state': X0
    })

#```````````````````````````````````````````````````````````````````````````````
# PLOT RESULTS
#```````````````````````````````````````````````````````````````````````````````

print("\nPlotting results...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for i, result in enumerate(results):
    sol = result['solution']
    k = result['k']
    k_CLASS = result['k_CLASS']

    t = sol.t
    s = sol.y[0]
    a = 1.0 / s
    phi = sol.y[1]
    psi = sol.y[2]
    dr = sol.y[3]
    dm = sol.y[4]
    vr = sol.y[5]
    vm = sol.y[6]

    label = f'k={k_CLASS:.3f} h/Mpc'

    # Plot scale factor
    axes[0,0].semilogy(t, a, label=label)

    # Plot phi
    axes[0,1].plot(t, phi, label=label)

    # Plot psi
    axes[1,0].plot(t, psi, label=label)

    # Plot dr
    axes[1,1].plot(t, dr, label=label)

    # Plot dm
    axes[2,0].plot(t, dm, label=label)

    # Plot vr
    axes[2,1].plot(t, vr, label=label)

axes[0,0].set_ylabel('Scale factor a')
axes[0,0].set_xlabel('Conformal time t')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axvline(x=t_today, color='red', linestyle='--', alpha=0.5, label='Today')

axes[0,1].set_ylabel('Newtonian potential φ')
axes[0,1].set_xlabel('Conformal time t')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].axvline(x=t_today, color='red', linestyle='--', alpha=0.5)

axes[1,0].set_ylabel('Curvature perturbation ψ')
axes[1,0].set_xlabel('Conformal time t')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axvline(x=t_today, color='red', linestyle='--', alpha=0.5)

axes[1,1].set_ylabel('Radiation density contrast δr')
axes[1,1].set_xlabel('Conformal time t')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].axvline(x=t_today, color='red', linestyle='--', alpha=0.5)

axes[2,0].set_ylabel('Matter density contrast δm')
axes[2,0].set_xlabel('Conformal time t')
axes[2,0].legend()
axes[2,0].grid(True, alpha=0.3)
axes[2,0].axvline(x=t_today, color='red', linestyle='--', alpha=0.5)

axes[2,1].set_ylabel('Radiation velocity vr')
axes[2,1].set_xlabel('Conformal time t')
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)
axes[2,1].axvline(x=t_today, color='red', linestyle='--', alpha=0.5)

plt.suptitle('Perturbation Evolution from Today to FCB', fontsize=14)
plt.tight_layout()
plt.savefig('perturbations_forward_to_FCB.pdf')
print("  Plot saved to: perturbations_forward_to_FCB.pdf")
plt.close()

# Clean up CLASS
cosmo.struct_cleanup()
cosmo.empty()

print("\nDone!")
