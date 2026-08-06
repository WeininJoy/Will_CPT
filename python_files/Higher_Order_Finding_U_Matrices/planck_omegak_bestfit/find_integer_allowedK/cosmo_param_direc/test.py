
import os

import numpy as np
import classy
from scipy.optimize import root_scalar

#################
# Parameters
#################
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
N_ncdm = 1
m_ncdm = 0.06
epsilon = 1e-2
nu_spacing4_bestfit = [0.4447390978198118, np.float64(-0.03952117480452887),0.16167855679796067, 0.55768703176223,2.095762, 0.972835,0.053017]

##################
def calculate_z_rec(params):
    OmegaM, OmegaK, omega_b_ratio, h = params

    CMB_params = {
        'output': 'tCl',
        'h': h,
        'Omega_b': omega_b_ratio*OmegaM,
        'Omega_cdm': (1.-omega_b_ratio) *OmegaM,
        'Omega_k': float(OmegaK),
        'A_s': nu_spacing4_bestfit[4]*1e-9,
        'n_s': nu_spacing4_bestfit[5],
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'tau_reio': nu_spacing4_bestfit[6],
        'lensing': 'no'
    }

    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()

    derived = cosmo.get_current_derived_parameters(['z_rec'])
    z_rec = derived['z_rec']
    
    cosmo.struct_cleanup()
    cosmo.empty()

    return z_rec

z_rec = calculate_z_rec(nu_spacing4_bestfit[:4])
print(f"Calculated z_rec: {z_rec:.2f}")