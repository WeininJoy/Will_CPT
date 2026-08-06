"""
Plot z_rec as a function of Omegab_ratio, with all other parameters fixed at
the best-fit values.  Uses the same CLASS setup as
calculate_integer_loss_func_fix3params_optuna.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import classy
from classy import CosmoComputationError
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ── fixed parameters (from calculate_integer_loss_func_fix3params_optuna.py) ──
nu_spacing4_bestfit = [0.45531269806747615, -0.0397827163256654,
                       0.15804679013016237,  0.5647692899377309,
                       2.068999, 0.977273, 0.051376]

OmegaM, OmegaK, Omegab_ratio_best, h = nu_spacing4_bestfit[:4]
A_s      = nu_spacing4_bestfit[4] * 1e-9
n_s      = nu_spacing4_bestfit[5]
tau_reio = nu_spacing4_bestfit[6]
N_ncdm   = 1
m_ncdm   = 0.06
Neff     = 3.046
Omega_gamma_h2 = 2.47e-5  # photon density
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
OmegaLambda = 1 - OmegaM - OmegaK - OmegaR

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 200  # number of pert variables
swaptime = 2  #set time when we swap from s to sigma
# ADAPTIVE DELTAETA PARAMETERS
deltaeta_max = 6.6e-4  # Original fixed value
k_deltaeta_target = 0.005  # Ensures k*deltaeta < 0.005 for all k
H0 = 1/np.sqrt(3*OmegaLambda)  #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda)

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

t0 = 1e-8;

#set coefficients for initial conditions
smin1 = np.sqrt(3*OmegaLambda/OmegaR);
szero = - OmegaM/(4*OmegaR);
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Check if t_events[0] is not empty before trying to access its elements
if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    # You might want to assign a default value or 'None' to fcb_time here
    fcb_time = None # Or np.nan, or some other indicator

# Rest of your code that uses fcb_time would go here
# For example:
if fcb_time is not None:
    print(f"Further processing with fcb_time = {fcb_time}")
else:
    print(f"No fcb_time available for further processing.")

# ── scan range for Omegab_ratio ───────────────────────────────────────────────
# Use the same p_ranges formula as the optimisation script (EXPAND=1.3, span/4)
_data = []
with open('../data/try_intK_planck_optuna/master_log.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 8:
            _data.append(float(parts[6]))   # column 6 = Omegab_ratio
_omegab_ratio_all = np.array(_data)

EXPAND = 1.3
span   = _omegab_ratio_all.max() - _omegab_ratio_all.min()
half   = EXPAND * span / 4
omegab_ratio_lo = Omegab_ratio_best - half
omegab_ratio_hi = Omegab_ratio_best + half

omegab_ratio_vals = np.linspace(omegab_ratio_lo, omegab_ratio_hi, 10)

# ── compute z_rec via CLASS ───────────────────────────────────────────────────
z_rec_vals = []
for obr in omegab_ratio_vals:
    params = {
        'output':   'tCl',
        'h':        h,
        'Omega_b':  obr * OmegaM,
        'Omega_cdm': (1. - obr) * OmegaM,
        'Omega_k':  float(OmegaK),
        'A_s':      A_s,
        'n_s':      n_s,
        'N_ncdm':   N_ncdm,
        'm_ncdm':   m_ncdm,
        'N_ur':     Neff - N_ncdm,
        'tau_reio': tau_reio,
        'lensing':  'no',
    }
    try:
        cosmo = classy.Class()
        cosmo.set(params)
        cosmo.compute()
        z_rec = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        cosmo.struct_cleanup()
        cosmo.empty()
        z_rec_vals.append(z_rec)
    except CosmoComputationError:
        z_rec_vals.append(np.nan)

z_rec_vals = np.array(z_rec_vals)

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

# Build a continuous s -> t interpolant once (s is monotonically decreasing,
# so reverse arrays so interp1d gets a strictly increasing x).
_s_arr = sol.y[0][::-1]
_t_arr = sol.t[::-1]
_interp_t_from_s = interp1d(_s_arr, _t_arr, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')

recConformalTime_vals = []
for z_rec in z_rec_vals:
    s_rec = 1 + z_rec  # reciprocal scale factor at recombination
    recConformalTime = float(_interp_t_from_s(s_rec))
    recConformalTime_vals.append(recConformalTime)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(omegab_ratio_vals, recConformalTime_vals, color='steelblue', lw=1.5)
ax.axvline(Omegab_ratio_best, color='red', lw=1.2, ls='--', label='best-fit')
ax.set_xlabel(r'$\Omega_b / \Omega_M$', fontsize=12)
# ax.set_ylabel(r'$z_{\rm rec}$', fontsize=12)
ax.set_ylabel(r'$\eta_{\rm rec}$', fontsize=12)
# ax.set_title(r'$z_{\rm rec}$ vs $\Omega_b/\Omega_M$ (other params fixed at best-fit)',
#              fontsize=11)
ax.set_title(r'$\eta_{\rm rec}$ vs $\Omega_b/\Omega_M$ (other params fixed at best-fit)',
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# out = './figures/zrec_vs_omegab_ratio.pdf'
out = './figures/etarec_vs_omegab_ratio.pdf'
plt.savefig(out, dpi=150)


