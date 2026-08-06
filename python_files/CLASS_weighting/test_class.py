import numpy as np
import classy
from classy import Class
from classy import CosmoComputationError
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt

lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density 
N_ncdm = 1 # number of massive neutrino species
m_ncdm = 0.06 # mass of massive neutrino species in eV
Neff = 3.046

# resolution lists for CMB power spectrum
l_linstep_list = [40, 10, 2]  
l_logstep_list = [1.26, 1.06, 1.007] 
q_linstep_list = [0.45, 0.2, 0.19] 
q_logstep_spline_list = [170, 10, 2.0] 
l_switch_limber_list = [1000, 2500, 2500]
l_switch_limber_for_nc_local_over_z_list = [1000, 2500, 2500]
l_switch_limber_for_nc_los_over_z_list = [1000, 2500, 2500]

def cosmological_parameters(mt, kt, h): 

    # Define Omega_r by assuming massive neutrinos
    # Omegarh2 = 0.00066984
    # Omega_r = Omegarh2 / (h**2) # photon density

    # # Define Omega_r by assuming massless neutrinos
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(Omega_r, rt, mt, kt)
    s0 = 1/a0
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return s0, Omega_lambda, Omega_m, Omega_K

# Runs the CLASS code
def run_class(resol_i, find_params):

    mt, kt, omega_b_ratio, h, A_s, n_s, tau = find_params
    s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(mt, kt, h)

    ################
    # Construct CMB power spectrum by "Class"
    # create instance of the class " Class "
    LambdaCDM = Class()

    # pass input parameters

    LambdaCDM.set({'Omega_b': omega_b_ratio*Omega_m, # omega_b = omega_b/(omega_cdm + omega_b) *omega_m
                'Omega_cdm': (1-omega_b_ratio) *Omega_m,
                    'Omega_k':-0.01, # Omega_K,
                    'nu_spacing': nu_spacing,
                    'h':h,
                    'A_s': A_s*1e-9,
                    'n_s': n_s,
                    'N_ncdm': N_ncdm,      # number of massive neutrino
                    'm_ncdm': m_ncdm,      # mass of massive neutrino species in eV
                    'N_ur': Neff - N_ncdm, # N_eff = N_ncdm + N_ur
                    'tau_reio': tau})

    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk',
                    'lensing':'yes', 
                    'P_k_max_1/Mpc':3.0,
                    'l_max_scalars':2508})

    LambdaCDM.set({ 'l_linstep': l_linstep_list[resol_i],
                'l_logstep': l_logstep_list[resol_i],
                'q_linstep': q_linstep_list[resol_i],
                'q_logstep_spline': q_logstep_spline_list[resol_i],
                'l_switch_limber' : l_switch_limber_list[resol_i],
                'l_switch_limber_for_nc_local_over_z': l_switch_limber_for_nc_local_over_z_list[resol_i],
                'l_switch_limber_for_nc_los_over_z': l_switch_limber_for_nc_los_over_z_list[resol_i],
                'write warnings' : 'yes'})

    # run class
    LambdaCDM.compute()

    # get all C_l output
    # <--- Change 2: Request unlensed spectra instead of lensed spectra
    # Note: 'lCl' is for the lensing potential, 'pCl' is for polarized, 'tCl' for temperature
    # You need to get the UNLENSED versions of tCl, pCl
    # The unlensed method returns a dictionary similar to lensed_cl
    # cls = LambdaCDM.raw_cl(2508)
    cls = LambdaCDM.lensed_cl(2508)

    # Don't remove these lines - otherwise uses up too much memory
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()

    return cls

# Example use case
resol_i = 0 # choose resolution index from 0, 1, 2 (0 is lowest resolution, 2 is highest resolution)
nu_spacing = 4
find_params = [401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583, 1.9375648884116028, 0.9787493821596979, 0.019760560255556746] # mt, kt, omega_b_ratio, h, A_s, n_s, tau
cls = run_class(resol_i, find_params)
ell = cls['ell'][2:]
clTT = cls['tt'][2:]

# np.save(f'ell_nu{nu_spacing}_reso{resol_i}.npy', ell)
# np.save(f'clTT_nu{nu_spacing}_reso{resol_i}.npy', clTT)
# ell = np.load(f'ell_nu{nu_spacing}_reso{resol_i}.npy')
# clTT = np.load(f'clTT_nu{nu_spacing}_reso{resol_i}.npy')
cosmo_tt = ell*(ell+1)*clTT * (1e6 * 2.7255)**2 / (2*np.pi)
plt.plot(ell,cosmo_tt,color = 'C0' , zorder=4, label=rf"$\Delta k={nu_spacing}$")

plt.xscale('log')
plt.xlim(1.9,1000)
# plt.ylim(0,2500)
plt.ylabel(r'$\mathcal{D}_\ell^{TT}\: [\mu \mathrm{K}^2]$', fontsize=13)
plt.xlabel(r'$\ell$', fontsize=13)
# plt.xticks([2,10,30])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(f'CMB_nu_spacing_{nu_spacing}.pdf')