
import clik
import os
import numpy as np
from scipy.optimize import root_scalar
from astropy import units as u
from astropy.constants import c
import classy
from classy import Class
from classy import CosmoComputationError
print(classy.__file__)

data = os.path.join(os.getcwd(),'/home/wnd22/rds/hpc-work/Polychord_solveb/clik_installs/data/planck_2018/baseline/plc_3.0') 

# Parameters
params_file = '/home/wnd22/rds/hpc-work/Will_CPT/Constrain_Cosmological_Parameters/CMB_CLASS/Metha/data/Planck_params/continuous_params.txt'
allowed_k_file = '/home/wnd22/rds/hpc-work/Will_CPT/Constrain_Cosmological_Parameters/CMB_CLASS/Metha/class_quantised/allowed_k.txt'
params = np.loadtxt(params_file, unpack=True)
Omega_lambda = 0.68
lam = 1
rt = 1
mt_list = np.linspace(300, 450, 10)
kt_list = np.linspace(1.e-4, 1, 10)
kmax = 10
Omega_gamma_h2 = 2.47e-5 # photon density 
# Note: Omega_r = (1 + N_eff*(7/8)*(4/11)**(4/3) ) * Omega_gamma
# Neff = 2.0328
N_ncdm = 1

class PlanckLikelihood(object):
    """Baseline Planck Likelihood"""
    
    def __init__(self):
        self.plik = clik.clik(os.path.join(data, "hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"))
        self.lowl = clik.clik(os.path.join(data, "low_l/commander/commander_dx12_v3_2_29.clik"))
        self.lowE = clik.clik(os.path.join(data, "low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"))
        self.lensing = clik.clik_lensing(os.path.join(data, "lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"))
    
    def __call__(self, cls, nuis):
        lkl = []
        for like in [self.plik, self.lowl, self.lowE, self.lensing]:
            #        for like in [self.plik]:
            lmaxes = like.get_lmax()
            dat = []
            order = ['tt','ee','bb','te','tb','eb']
            if like is self.lensing:
                order = ['pp'] + order
            
            # print(order,len(lmaxes),len(order))
            for spec, lmax in zip(order, lmaxes):
                if lmax>-1:
                    if spec == 'pp':
                        dat += list(cls[spec][:lmax+1])
                    else:
                        dat += list(cls[spec][:lmax+1]* (1e6 * 2.7255)**2 )
        
            for param in like.get_extra_parameter_names():
                dat.append(nuis[param])
    
            lkl.append(like(dat))
    
        return np.array(lkl).flatten()

lkl = PlanckLikelihood()


def cosmological_parameters(input_number):
    mt = mt_list[input_number//10]
    kt = kt_list[input_number%10]

    def solve_a0(mt, kt):
        def f(a0):
            return (1./Omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(mt, kt)
    s0 = 1/a0
    Omega_r = Omega_lambda / a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)

    return s0, Omega_r, Omega_m, Omega_K


# Runs the CLASS code
def run_class(params, input_number):

    s0, Omega_r, Omega_m, Omega_K = cosmological_parameters(input_number)
    h = params[3]
    H0 = h*100 * u.km/u.s/u.Mpc
    Lambda = Omega_lambda * 3 * H0**2 / c**2  # unit: km2 / (Mpc2 m2)
    sqrtlambda = np.sqrt(Lambda.value * 1.e6)
    allowedK = np.load(f'/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/Curve_universes_a0/data/fixOmegaLambda/allowedK_{input_number}.npy')
    deltaK_list = [allowedK[n+1] - allowedK[n] for n in range(len(allowedK)-1)]
    deltaK = sum(deltaK_list[len(deltaK_list)//2:]) / len(deltaK_list[len(deltaK_list)//2:])
    deltaKphys = deltaK * s0 * sqrtlambda

    nmax = int(kmax/deltaKphys)
    k = [i * deltaKphys for i in range(nmax)]

    np.savetxt(allowed_k_file, np.array(k))

    # create instance of the class " Class "
    LambdaCDM = Class()

    # # pass input parameters
    LambdaCDM.set({'omega_b':params[0], 
                    'omega_m' : Omega_m*h**2, # omega_m=Omega_M*h^2
                    'omega_ncdm': params[2]*1e-4, 
                    'Omega_k': Omega_K,
                    'h':h, 
                    'A_s':params[4]*1e-9, 
                    'n_s':params[5], 
                    'tau_reio':params[6], 
                    'N_ur': (Omega_r*h**2/Omega_gamma_h2) *(8/7) *(11/4)**(4/3),  # Neff, Note: Omega_r = (1 + N_eff*(7/8)*(4/11)**(4/3) ) * Omega_gamma
                    'N_ncdm':N_ncdm})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk',
                    'lensing':'yes',
                    'P_k_max_1/Mpc':3.0,
                    'l_max_scalars':2508})

    # run class
    LambdaCDM.compute()

    # get all C_l output
    cls = LambdaCDM.lensed_cl(2508)

    # Don't remove these lines - otherwise uses up too much memory
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()

    return allowedK[0], deltaKphys, cls # k0, delta_k, cls 


# Computes TT spectrum and returns chi^2 for given set of parameters params in form  [logA_SR, N_star, log10f_i, omega_k, H0] using linear quantisation
def run_TT(params, input_number):
        
    success = 'True'
    try:
        # Find corresponding spectra
        k0, delta_k, cls = run_class(params, input_number)

        if np.isnan([val for val in cls.values()]).any(): # if there is any Nan in cls -> raise CosmoComputationError
            raise CosmoComputationError
        else:
            nuisance_params = params[7:]

            nuis={
                'ycal':nuisance_params[0],
                'A_cib_217':nuisance_params[1],
                'xi_sz_cib':nuisance_params[2],
                'A_sz':nuisance_params[3],
                'ps_A_100_100':nuisance_params[4],
                'ps_A_143_143':nuisance_params[5],
                'ps_A_143_217':nuisance_params[6],
                'ps_A_217_217':nuisance_params[7],
                'ksz_norm':nuisance_params[8],
                'gal545_A_100':nuisance_params[9],
                'gal545_A_143':nuisance_params[10],
                'gal545_A_143_217':nuisance_params[11],
                'gal545_A_217':nuisance_params[12],
                'galf_TE_A_100':nuisance_params[13],
                'galf_TE_A_100_143':nuisance_params[14],
                'galf_TE_A_100_217':nuisance_params[15],
                'galf_TE_A_143':nuisance_params[16],
                'galf_TE_A_143_217':nuisance_params[17],
                'galf_TE_A_217':nuisance_params[18],
                'calib_100T':nuisance_params[19],
                'calib_217T':nuisance_params[20],
                'cib_index':-1.3, #no range given in table so assume fixed
                #-------------------------------------------------------------------
                # These are all set to 1, so assume that these are fixed -----------
                'A_cnoise_e2e_100_100_EE':1.,
                'A_cnoise_e2e_143_143_EE':1.,
                'A_cnoise_e2e_217_217_EE':1.,
                'A_sbpx_100_100_TT':1.,
                'A_sbpx_143_143_TT':1.,
                'A_sbpx_143_217_TT':1.,
                'A_sbpx_217_217_TT':1.,
                'A_sbpx_100_100_EE':1.,
                'A_sbpx_100_143_EE':1.,
                'A_sbpx_100_217_EE':1.,
                'A_sbpx_143_143_EE':1.,
                'A_sbpx_143_217_EE':1.,
                'A_sbpx_217_217_EE':1.,
                'A_pol':1,
                'A_planck':1.,
                #-------------------------------------------------------------------
                # These are fixed from Planck 2018 Likelihood Paper, Table 16 ------
                'galf_EE_A_100':0.055,
                'galf_EE_A_100_143':0.040,
                'galf_EE_A_100_217':0.094,
                'galf_EE_A_143':0.086,
                'galf_EE_A_143_217':0.21,
                'galf_EE_A_217':0.70,
                'calib_100P':1.021,
                'calib_143P':0.966,
                'calib_217P':1.04,
                #-------------------------------------------------------------------
                # These are fixed from Planck 2018 Likelihood Paper, pg 39 ---------
                'galf_EE_index':-2.4,
                'galf_TE_index':-2.4,
            #-------------------------------------------------------------------
            }

            plik, lowl, lowE, lensing = -2 * lkl(cls, nuis)
            chi_eff_sq = plik + lowl + lowE + lensing

    except CosmoComputationError:
        print('CosmoComputationError')
        k0 = 0 
        delta_k = 0
        plik = 2e+30
        lowl = 2e+30
        lowE = 2e+30
        lensing = 2e+30
        chi_eff_sq = 2e+30
        success = 'False'
    except clik.lkl.CError:
        print('CError')
        k0 = 0 
        delta_k = 0
        plik = 2e+30
        lowl = 2e+30
        lowE = 2e+30
        lensing = 2e+30
        chi_eff_sq = 2e+30
        success = 'False'

    return k0, delta_k, plik, lowl, lowE, lensing, chi_eff_sq, success

for input_number in range(100):
    k0, delta_k, plik, lowl, lowE, lensing, chi_eff_sq, success = run_TT(params, input_number)
    result = [k0, delta_k, plik, lowl, lowE, lensing, chi_eff_sq] 
    result = np.concatenate((result, params))
    result = np.concatenate((result, [success]))
    file = open(f'/home/wnd22/rds/hpc-work/Will_CPT/Constrain_Cosmological_Parameters/CMB_CLASS/Metha/data/cl_files/cl_{input_number}.txt','w')
    for param in result:
        file.write(param+" ")
    file.close()