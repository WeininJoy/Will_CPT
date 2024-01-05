import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from scipy import interpolate
from scipy.optimize import root_scalar
from scipy import optimize
import pandas as pd
import corner

plt.rcParams['text.usetex'] = True

###############
# Important cosmmological parameters
###############
from astropy import units as u
from astropy.constants import c
from scipy.constants import physical_constants
import numpy as np


class Universe:

    def __init__(self, H0, Omega_lambda, Omega_r_h2): 
        
        ### cosmological constants
        self.H0 = H0   # Hubble constant
        self.h = H0.value / 100.
        self.Omega_lambda = Omega_lambda  # dark energy
        self.Omega_r = Omega_r_h2 / self.h**2
        self.Lambda = self.Omega_lambda * 3 * self.H0**2 / c**2 
        self.a0 = (self.Omega_lambda/ self.Omega_r)**0.25   # suppose a = 1 when the energy density of radiation and the DE are equal.

    ###############
    # Flat universe
    ###############

    def psi_int_flat(self, k):

        def psi_flat(a, k):
            return k * (k**4-4)**0.5 * a**2/(a**4+1)**0.5 / (1+a**2*k**2+a**4)
        
        result, err = quad(psi_flat, 0, float('inf'), args=(k))
        return result

    ###############
    # Curved universe
    ###############

    def psi_int_curved(self, k, kc):

        def psi_curved(a, k, kc):
            return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)
        
        if kc > 1:
            result, err = quad(psi_curved, 0, self.a_tot(kc), args=(k, kc))
        else:
            result, err = quad(psi_curved, 0, float('inf'), args=(k, kc))
        return result

    def psi_int_anal(self, a, k, kc): 
        wolfSession = WolframLanguageSession()
        wolfSession.evaluate(wl.Needs("targetFunctions`"))
        
        result = wolfSession.evaluate(wl.targetFunctions.PsiIntCurved((a, k, kc)))
        wolfSession.terminate()
        return result

    def a_tot(self, kc): # when kc > 1 -> a_tot is finite 
        if kc > 1:
            Lambda = self.Lambda.si.value
            K = kc * 1. / 3. * 2 * Lambda
            R = Lambda
            a_tot_small = math.sqrt( (3* K - math.sqrt(9* K**2 - 4*R*Lambda)) / (2* Lambda) )
            a_tot_large = math.sqrt( (3* K + math.sqrt(9* K**2 - 4*R*Lambda)) / (2* Lambda) )
            return a_tot_small
        else: 
            print("For kc < 1, a_tot is infinite.")
            return float('inf')
    
    def eta_tot(self, kc):

        Lambda = 1
        R = Lambda
        K = kc * 1. / 3. * 2 * Lambda  

        def a_dot(a):
            return math.sqrt( 1./3. * ( R - 3*K* a**2 + Lambda* a**4 ) )

        if kc < 1:
            result = quad(lambda a: 1./a_dot(a), 0, np.inf)
            return result[0]
        else: 
            result = quad(lambda a: 1./a_dot(a), 0, self.a_tot(kc))
            return 2* result[0]

    ###############
    # plot K-theta(K) curves with different kc
    ###############
    
    def plot_k_theta(self, k_list, kc_list):
      
        for kc in kc_list:
            if kc < 1: # Note: for kc<1, a->infinity -> theta=2*psi(a=1)
                a = 1
                plt.plot(k_list, [self.psi_int_curved(k, kc) for k in k_list], label='num, kc='+str(round(kc,1)))
                plt.plot(k_list, [2 * self.psi_int_anal(a, k, kc) for k in k_list], '--',label='anal, kc='+str(round(kc,1)))
            else:
                a = self.a_tot(kc)
                plt.plot(k_list, [self.psi_int_curved(k, kc) for k in k_list],'-.', label='num, kc='+str(round(kc,1)))
                plt.plot(k_list, [self.psi_int_anal(a, k, kc) for k in k_list], '--',label='anal, kc='+str(round(kc,1)))

        plt.xlabel(r"$K$")
        plt.ylabel(r"$\theta(K)$")
        plt.legend()
        plt.savefig("psi_diffK_num_anal.pdf")


    ########################
    # Find K (curvature) with known Lambda (DE) by making slope of theta = pi/2 (Lambda is known)
    # Note: (1) k_int: integer wave vectors, _bar: in K=1 unit
    #       (2) k_com_bar = sqrt( k_int*(k_int+2) - 3*K_bar ) , K_bar = 1
    #       (3) k_com = k* sqrt(Lambda)
    #       (4) k_com_bar/a_bar = k_phys = k_com/a -> k = a_bar/a/sqrt(Lambda)*k_com_bar
    ########################

    def Find_kc(self, n_k, n_range): # n_k > n_range/2
        # note: n_k is the number of wave vector we used for calculating slope (choose larger one, with theta approaching to a straight line))
        # return: kc, K, Omega_K
        
        if n_k > n_range/2 :
            k_int_list = np.linspace(n_k - n_range/2, n_k + n_range/2, 30)

            ##### Calculate slope
            def theta_slope(kc):
                
                K = 2.* self.Lambda / 3. * kc
                Omega_K = - K * c**2 / self.a0**2 / self.H0**2

                if kc > 0: K_bar = 1
                elif kc < 0: K_bar = -1
                else: K_bar = 0

                a0_bar = c * np.sqrt(-K_bar/Omega_K)/self.H0

                if kc > 0:  # close universe 
                    k_com_bar_list = np.real([math.sqrt( k_int*(k_int+2) - 3*K_bar ) for k_int in k_int_list])
                    # k_com_bar_list = np.real([math.sqrt( k_int**2 - K_bar ) for k_int in k_int_list])
                else:       # open or flat universe
                    k_com_bar_list = np.real([math.sqrt( k_int**2 - 3*K_bar ) for k_int in k_int_list])

                k_list = np.real([self.a0/a0_bar.si.value/math.sqrt(self.Lambda.si.value)*k_com_bar for k_com_bar in k_com_bar_list])

                theta_list = [self.psi_int_curved(k, kc) for k in k_list]
                func_interpolate = interpolate.interp1d(k_int_list, theta_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function
                slope = 2./ np.pi * (func_interpolate(n_k) - func_interpolate(n_k-1)) / 1.     # calculate the slope at n_k

                return slope

            ##### Calculate kc by root finder
            sol = root_scalar(lambda kc: theta_slope(kc) - 1., bracket=[0.1 , 0.99], method='brentq') 
            kc = sol.root 
            K = 2.* self.Lambda / 3. * kc
            Omega_K = - K * c**2 / self.a0**2 / self.H0**2

            print("Omega_K=", Omega_K)

            return kc, K, Omega_K

        else:
            raise ValueError("n_k should be larger than n_range/2.")


    ########################
    # Find K (curvature) and Lambda (DE) by making discrete comoving dimensional wave vectors (k_dim) to be integers
    # 1. dimensionless curvature (k_c) = 3./2./Lambda * K
    # 2. dimensionless wave vector (k) = k_dim / sqrt(Lambda) 
    # 3. Constraint1: The slope of theta(k_dim) = np.pi/2 -> 2/np.pi * theta'(k_dim) = 1
    # 4. Constraint2: Discrete k_dim should be integers
    ########################

    # n_k = 110  # the number of wave vector we used for calculating slope and k_dim (choose larger one, with theta approaching to a straight line))

    # def Find_K_Lambda(x):
        
    #     K = x[0]
    #     Lambda = x[1]
        
    #     ##### Calculate slope
    #     kc = 3./2./Lambda * K  ## dimensionless kc
    #     k_dim_list = np.linspace(100, 120, 30)
    #     k_list = [k_dim/(Lambda)**(0.5) for k_dim in k_dim_list]
    #     theta_list = [psi_int_curved(k, kc) for k in k_list]
    #     func_interpolate = interpolate.interp1d(k_dim_list, theta_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function
    #     slope = 2./ np.pi * (func_interpolate(n_k) - func_interpolate(n_k-1)) / 1.     # calculate the slope at n_k
    #     print("slope="+str(slope))

    #     ##### Calculate k_dim by root finder
    #     theta = n_k * np.pi / 2.
    #     sol = root_scalar(lambda k_dim: func_interpolate(k_dim) - theta, bracket=[100, 120], method='brentq') 
    #     sol_k_dim = sol.root   # find the k_dim value corresponding to the theta value we want
    #     print ("sol_k_dim="+str(sol_k_dim))

    #     return [slope - 1., sol_k_dim - int(sol_k_dim)]  # want the slope = 1, and sol_k_dim to be an integer

    # res = optimize.root(Find_K_Lambda, [0.75, 2.], method='hybr') # Use root finder to find the K and Lambda satisfying the two constraints
    # K = res.x[0]
    # Lambda = res.x[1]
    # print('K, Lambda = '+str(K)+', '+str(Lambda))


    ###############################
    # Plot the discrete comoving wave vectors according to the kc value we found
    ###############################


    def find_discrete_k(self, n_k, n_range):

        kc, K, Omega_K = self.Find_kc(110, 20)
        k_int_list = np.linspace(n_k - n_range/2, n_k + n_range/2, 30)

        if kc > 0: K_bar = 1
        elif kc < 0: K_bar = -1
        else: K_bar = 0

        a0_bar = c * np.sqrt(-K_bar/Omega_K)/self.H0

        if kc > 0:  # close universe 
            k_com_bar_list = np.real([math.sqrt( k_int*(k_int+2) - 3*K_bar ) for k_int in k_int_list])
        else:       # open universe
            k_com_bar_list = np.real([math.sqrt( k_int**2 - 3*K_bar ) for k_int in k_int_list])

        k_list = np.real([self.a0/a0_bar.si.value/math.sqrt(self.Lambda.si.value)*k_com_bar for k_com_bar in k_com_bar_list])

        theta_curved_list = [self.psi_int_curved(k, kc) for k in k_list]
        func_interpolate = interpolate.interp1d(k_int_list, theta_curved_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function

        ##### Use root finder to find discrete k 
        discrete_theta_list = [(n + n_k - n_range/2)* np.pi/ 2. for n in range(n_range)]
        discrete_k_int_list = []
        for theta in discrete_theta_list:
            sol_k_int = root_scalar(lambda k_int: func_interpolate(k_int) - theta, bracket=[n_k - n_range/2 -5, n_k + n_range/2+5], method='brentq') 
            discrete_k_int_list.append(sol_k_int.root)

        print(discrete_k_int_list)

        return k_int_list, theta_curved_list, discrete_k_int_list, discrete_theta_list
    

    def k_FCB_closed(self, n_k, n_range):
        
        kc, K, Omega_K = self.Find_kc(110, 20)
        k_int_list = np.linspace(n_k - n_range/2, n_k + n_range/2, n_range+1)

        if kc > 0: K_bar = 1
        elif kc < 0: K_bar = -1
        else: K_bar = 0

        a0_bar = c * np.sqrt(-K_bar/Omega_K)/self.H0

        if kc > 0:  # close universe 
            k_com_bar_list = np.real([math.sqrt( k_int*(k_int+2) - 3*K_bar ) for k_int in k_int_list])
            # k_com_bar_list = np.real([math.sqrt( k_int**2 - K_bar ) for k_int in k_int_list])
        else:       # open universe
            k_com_bar_list = np.real([math.sqrt( k_int**2 - 3*K_bar ) for k_int in k_int_list])

        k_list = np.real([self.a0/a0_bar.si.value/math.sqrt(self.Lambda.si.value)*k_com_bar for k_com_bar in k_com_bar_list])

        theta_curved_list = [self.psi_int_curved(k, kc) for k in k_list]
        func_interpolate = interpolate.interp1d(theta_curved_list, k_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function

        discrete_theta_list = [(n + n_k - n_range/2 -1)* np.pi/ 2. for n in range(n_range+1)]
        discrete_k_list_FCB = [func_interpolate(theta) for theta in discrete_theta_list]

        plt.plot(k_int_list, discrete_k_list_FCB,'.-', label = "k_FCB")
        plt.plot(k_int_list, k_list,'.-', label = "k_closed")
        plt.xlabel(r"$\nu$ (integers)", fontsize=15)
        plt.ylabel(r"$\tilde{k}=k/\sqrt{\lambda}$", fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig("k_discrete_FCB-closed.pdf")


    ##### Make the k-theta plot

    def plot_k_theta_discrete(self, n_k, n_range): 
        
        kc, K, Omega_K = self.Find_kc(110, 20)
        k_int_list, theta_curved_list, discrete_k_int_list, discrete_theta_list = self.find_discrete_k(n_k, n_range)

        ## plot curved case
        plt.plot(k_int_list, theta_curved_list, label = r"curved, $\Omega_{\kappa,0}=$"+str(round(Omega_K.si.value,3)))
        plt.plot(discrete_k_int_list, discrete_theta_list, 'k.')

        # plt.xlim([100, 115])
        # plt.ylim([100*np.pi/2., 114*np.pi/2.])
        plt.vlines(discrete_k_int_list, 0, discrete_theta_list, colors='k', linestyles='dotted')
        plt.hlines(discrete_theta_list, 0, discrete_k_int_list, colors='k', linestyles='dotted')
        # plt.xticks([(n+101) for n in range(15)], [str(n+101) for n in range(15)], fontsize=13)
        # plt.yticks(discrete_theta_list, [str(n+101)+r"$\frac{\pi}{2}$"  for n in range(13)], fontsize=13)
        plt.xlabel(r"$k$ (integer wave vector)", fontsize=15)
        plt.ylabel(r"$\theta(k)$", fontsize=15)
        plt.legend(fontsize=14)
        plt.savefig("theta_k_discrete.pdf")


H0 = 70 * u.km/u.s/u.Mpc
Omega_lambda = 0.73 
Omega_r_h2 = 2.47e-5

universe = Universe(H0, Omega_lambda, Omega_r_h2)
kc, K, Omega_K = universe.Find_kc(n_k=110, n_range=20)
print('kc='+str(kc))

###############################
# Plot the allowed curves in the Omega_K, Omega_Lambda plane
###############################

def plot_OmegaK_OmegaLambda_plane(H0_list, Omega_lambda, Omega_r_h2, n_k, n_range):

    ### Plot Planck data 2018

    # MCMC chain samples
    samples = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE_4.txt')

    # load the column names for the samples
    column_names_raw = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE.paramnames', dtype=str, usecols=[0])
    column_names = [x.replace("b'",'').replace("'",'') for x in column_names_raw]

    # make a data frame with column names and samples
    samples1 = pd.DataFrame(samples[:,2:], columns=column_names) # first two columns are not important

    # define which parameters to use
    use_params = ['H0*', 'omegak']
    ndim = len(use_params)
    # Make the base corner plot
    #figure = corner.corner(samples1[use_params], range=[(00.6, 0.72), (-0.02, 0.0)], bins=20, color='r')
    figure = corner.corner(samples1[use_params],  bins=20, color='r')
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    ### plot my result
    universe_list = [Universe(H0, Omega_lambda, Omega_r_h2) for H0 in H0_list]
    Omega_K_list = [universe.Find_kc(n_k, n_range)[2] for universe in universe_list]

    ax = axes[1, 0]
    ax.plot(H0_list, Omega_K_list, label = r"$\Omega_{\lambda,0}="+str(Omega_lambda)+r", \Omega_{r,0}h^2=$"+'{:.2e}'.format(Omega_r_h2))
    ax.set_xlabel(r"$H_0$", fontsize=10)
    ax.set_ylabel(r"$\Omega_{\kappa,0}$", fontsize=10)
    ax.legend(fontsize=8)
    plt.savefig("H0-OmegaK-withPlanck2018.pdf")


### Set parameters

# H0_list = np.linspace(45, 65, 5) * u.km/u.s/u.Mpc
# Omega_r_h2 = 2.46e-5
# Omega_lambda = 0.56 # np.linspace(0.6, 0.72, 5)

# plot_OmegaK_OmegaLambda_plane(H0_list, Omega_lambda, Omega_r_h2, n_k=110, n_range=20)
