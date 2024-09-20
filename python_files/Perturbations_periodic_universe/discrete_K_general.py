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

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

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

    # calcaulte psi through integral
    def psi_int_flat(self, k):

        def psi_flat(a, k):
            return k * (k**4-4)**0.5 * a**2/(a**4+1)**0.5 / (1+a**2*k**2+a**4)
        
        result, err = quad(psi_flat, 0, float('inf'), args=(k))
        return result

    ###############
    # Curved universe
    ###############

    # calcaulte psi through integral
    def psi_int_curved(self, k, kc):

        def psi_curved(a, k, kc):
            # return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)  # old Phi equation
            return (k**2)**0.5 * ((k**2-2*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2-2*kc)*a**2 +1)  # new Phi equation 
        
        if kc > 1:
            result, err = quad(psi_curved, 0, self.a_tot(kc), args=(k, kc))
        else:
            result, err = quad(psi_curved, 0, 1, args=(k, kc))
        return 2*result

    # analytic solution of psi
    def psi_int_anal(self, a, k, kc): 
        wolfSession = WolframLanguageSession()
        wolfSession.evaluate(wl.Needs("targetFunctions`"))
        
        result = wolfSession.evaluate(wl.targetFunctions.PsiIntCurved((a, k, kc)))
        wolfSession.terminate()
        return 2*result

    # calculate a_tot, which is the maximum value of a for collapsing universe (kc>1)
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
    
    # calculate eta_tot, which is the conformal age of the universe
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

    ###################
    # Plot slope-kc curves
    ###################

    # calculate the slope of theta (dtheta/dk), which is equal to eta_tot/sqrt{3}

    def slope_kc_eta_tot_without_matter(self, kc_list):
        plt.figure(figsize=(3.375,2.7)) 
        slope_list = []
        for kc in kc_list:
            k_trans_const = np.sqrt(2*kc/3.)
            eta_tot = self.eta_tot(kc)
            trans_slope = eta_tot/np.sqrt(3.)* k_trans_const
            slope_list.append(trans_slope)
            
        slope_discrete_list = [1./5.*np.pi/2, 1./4.*np.pi/2, 1./3.*np.pi/2, 1./2.*np.pi/2, 1.*np.pi/2, 2.*np.pi/2, 3.*np.pi/2]  # discrete slope values
        slope_label_list = [r"$\frac{1}{5}\frac{\pi}{2}$",r"$\frac{1}{4}\frac{\pi}{2}$", r"$\frac{1}{3}\frac{\pi}{2}$", r"$\frac{1}{2}\frac{\pi}{2}$", r"$1\frac{\pi}{2}$", r"$2\frac{\pi}{2}$", r"$3\frac{\pi}{2}$"]
        func_interpolate = interpolate.interp1d(slope_list, kc_list, fill_value="extrapolate")
        kc_discrete_list = []
        for n in range(len(slope_discrete_list)):
            slope_discrete = slope_discrete_list[n]
            kc_discrete = func_interpolate(slope_discrete)
            kc_discrete_list.append(kc_discrete)
            plt.vlines(kc_discrete, 0, slope_discrete, colors=color_list[n], linestyles='dotted',linewidth=0.8)
            plt.hlines(slope_discrete, 0, kc_discrete, colors=color_list[n], linestyles='dotted',linewidth=0.8)
        
        plt.plot(kc_list, slope_list, color='r',linewidth=1.4)
        plt.xlabel(r"$\tilde\kappa$")
        plt.ylabel(r"$d\theta/dk_{\mathrm{int}}(\tilde\kappa)$")
        plt.xticks(kc_discrete_list[2:-1], [str(np.round(kc_discrete,3)) for kc_discrete in kc_discrete_list[2:-1]])
        plt.yticks(slope_discrete_list[3:], slope_label_list[3:])
        plt.ylim([0, 3.1*np.pi/2])
        plt.xlim([0, 1.])
        plt.savefig("slope_kc_eta_tot.pdf", bbox_inches="tight")

    # consider matter
    def slope_kc_eta_tot_with_matter(self, kc_list):
        # for kc < 1 case
        slope_list_with_matter = []
        slope_list_without_matter = []
        for kc in kc_list:
            K = 2.* self.Lambda / 3. * kc
            Omega_K = - K * c**2 / self.a0**2 / self.H0**2
            a0_bar = c * np.sqrt(-1/Omega_K)/self.H0
            # k_trans_const = np.real(self.a0 / a0_bar.si.value / math.sqrt(self.Lambda.si.value))
            k_trans_const = np.sqrt(2*kc/3.)

            # without matter
            if kc < 1:
                eta_tot = self.eta_tot(kc)
                trans_slope = 2./np.pi* eta_tot/np.sqrt(3.)* k_trans_const
                slope_list_without_matter.append(trans_slope)
            
            # with matter
            try:
                Omega_m = 1 - Omega_K - self.Omega_lambda - self.Omega_r
                m = Omega_m / self.Omega_lambda  # if Lambda=1 -> H0=1/sqrt(3 Omega_lambda) -> m = 3* H0^2 Omega_m=Omega_m/Omega_lambda 
                def a_dot(a): return math.sqrt( 1./3. * ( 1 + m*a- 2*kc* a**2 + a**4 ) )
                eta_tot = quad(lambda a: 1./a_dot(a), 0, np.inf)[0]
                trans_slope = 2./np.pi* eta_tot/np.sqrt(3.)* k_trans_const
                slope_list_with_matter.append(trans_slope)
            except:
                max_kc = kc
                print("max kc=", kc)
                break

        plt.plot(kc_list[:len(slope_list_without_matter)], slope_list_without_matter, color='b', label = r"without matter")
        plt.axvline(x = 1, color = 'b', linestyle='--')
        plt.plot(kc_list[:len(slope_list_with_matter)], slope_list_with_matter, color='orange', label = r"with matter")
        plt.axvline(x = max_kc, color = 'orange', linestyle='--')
        plt.xlabel(r"$\tilde\kappa$", fontsize=17)
        plt.ylabel(r"$\frac{2}{\pi}$Slope($\tilde\kappa$)", fontsize=17)
        plt.ylim([0, 2.25])
        plt.legend(fontsize=16)
        plt.savefig("slope_kc_eta_tot.pdf")
        


    ###############
    # plot k-theta(k) curves with different kc
    ###############
    
    def plot_k_theta(self, k_list, kc_list):
      
        plt.figure(figsize=(3.375,2.7))
        plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})
        n = 0
        for kc in kc_list:
            K = 2.* self.Lambda / 3. * kc
            Omega_K = - K * c**2 / self.a0**2 / self.H0**2
            if kc < 1:
                # plt.plot(k_list, [2/np.pi*self.psi_int_curved(k, kc) for k in k_list],color=color_list[n] ,label=r'$\Omega_{\kappa,0}=$'+str(round(Omega_K.si.value,3)),linewidth=3.0)
                plt.plot(k_list, [self.psi_int_curved(k, kc) for k in k_list],color=color_list[n],label=r'$\tilde\kappa={%.1f}$'%kc,linewidth=1.4)
            else:
                # plt.plot(k_list, [2/np.pi*self.psi_int_curved(k, kc) for k in k_list], '-.', color=color_list[n], label=r'$\Omega_{\kappa,0}=$'+str(round(Omega_K.si.value,3)),linewidth=3.0)
                plt.plot(k_list, [self.psi_int_curved(k, kc) for k in k_list],color=color_list[n],label=r'$\tilde\kappa={%.1f}$'%kc,linewidth=1.4)

            # plot dots line for discrete theta and k
            theta_list = [self.psi_int_curved(k, kc) for k in k_list]
            func_interpolate = interpolate.interp1d(theta_list, k_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function
            discrete_theta_list = [n for n in range(1, 20)]
            discrete_k_list = []
            for theta in discrete_theta_list: 
                k = func_interpolate(theta)
                if k < 6:
                    discrete_k_list.append(k)
            plt.vlines(discrete_k_list, 0, discrete_theta_list[:len(discrete_k_list)], colors=color_list[n], linestyles='dotted', linewidth=0.8, alpha=0.8)
            plt.hlines(discrete_theta_list[:len(discrete_k_list)], 0, discrete_k_list, colors='k', linestyles='dotted', linewidth=0.8, alpha=0.8)
            n += 1

        # plt.plot([4.3, 5.8], [10.7, 14.2],'--', linewidth=3.0, color='r')
        # t = plt.text(4.9, 12.1, 'Slope', fontsize=8, weight='extra bold',color='r', ha='right', va='bottom')
        # t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
        plt.xlabel(r"$\tilde k$")
        # plt.ylabel(r"$\frac{2}{\pi}\theta(\tilde{k})$")
        # plt.xlabel(r"$k$, should be $n\in \mathbb{N}$")
        plt.ylabel(r"$\theta(\tilde k, a\rightarrow a_{\infty})$, should be $\in \mathbb{N}\frac{\pi}{2}$")
        plt.xlim([1, 6])
        plt.ylim([0, 9])
        plt.xticks([int(ele) for ele in np.linspace(1, 6, 6)])
        plt.yticks([int(ele) for ele in np.linspace(0, 9, 10)], [str(int(ele))+r"$\frac{\pi}{2}$" for ele in np.linspace(0, 9, 10)])
        plt.legend()
        plt.savefig("theta_diffK.pdf", bbox_inches="tight")
        # plt.savefig("theta_diffK_num_anal.pdf")

    ######################
    # Plot theta-k_int with discrete k_int (integer wave vectors)
    ######################

    def plot_kint_theta(self, n_k, n_range, kc):

        # make data
        k_int_list = np.linspace(n_k - n_range//2 +1, n_k + n_range//2, n_range) # k_int are integers

        def theta(kc, k_int):
            trans_constant = np.sqrt(2*kc/3.)
            k_com = np.real(np.sqrt( k_int*(k_int+2)))
            k = k_com * trans_constant
            def psi_curved(a, kc, k):
                # return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)  # old Phi equation
                return (k**2)**0.5 * ((k**2-2*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2-2*kc)*a**2 +1)  # new Phi equation 
            try:
                theta_int  = 2* quad(psi_curved, 0, 1, args=(kc, k))[0]
            except:
                theta_int = 0
            return theta_int

        plt.plot(k_int_list, [2/np.pi * theta(kc, k_int) for k_int in k_int_list])

        # plot the discrete k_int
        theta_discrete_list = [n* np.pi/2 for n in range(round(2/np.pi*theta(kc, k_int_list[0])), round(2/np.pi*theta(kc, k_int_list[-1])))]
        theta_list = [theta(kc, k_int) for k_int in k_int_list]
        func_interpolate = interpolate.interp1d(theta_list, k_int_list, fill_value="extrapolate")
        k_int_discrete_list = []
        for theta_discrete in theta_discrete_list:
            k_int_discrete = func_interpolate(theta_discrete)
            k_int_discrete_list.append(k_int_discrete)
        plt.plot(k_int_discrete_list, [2/np.pi*theta for theta in theta_discrete_list],'r.', markersize=5)

        for n in range(len(k_int_list)):
            plt.axvline(x=k_int_list[n], color='k', linestyle='--')
        plt.xlabel(r"$k_{\text{int}}$", fontsize=16)
        plt.ylabel(r"$\frac{2}{\pi}\theta$", fontsize=16)
        plt.xlim([k_int_list[0], k_int_list[-1]])
        plt.ylim([2/np.pi*theta(kc, k_int_list[0]), 2/np.pi*theta(kc, k_int_list[-1])])
        plt.xticks(k_int_list,fontsize=14)
        n_theta_discrete_list = [n for n in range(round(2/np.pi*theta(kc, k_int_list[0])), round(2/np.pi*theta(kc, k_int_list[-1])))]
        plt.yticks(n_theta_discrete_list, fontsize=14)
        plt.savefig("theta-kint_discretek.pdf")
        

    ######################
    # Plot loglog theta-k_int for allowed kc
    ######################

    def plot_kint_theta_loglog(self):

        plt.figure(figsize=(3.375,2.7)) 
        # make data
        allowed_kc_list = [0.113284, 0.238231, 0.671658, 0.98607, 0.99969] 
        slope_list = [ r"$\frac{1}{3}\frac{\pi}{2}$", r"$\frac{1}{2}\frac{\pi}{2}$", r"$1\frac{\pi}{2}$", r"$2\frac{\pi}{2}$", r"$3\frac{\pi}{2}$"]
        k_int_list = np.logspace(0, 3, 1000) # k_int are more continuous

        def theta(kc, k_int):
            trans_constant = np.sqrt(2*kc/3.)
            k_com = np.real(np.sqrt( k_int*(k_int+2) ))
            k = k_com * trans_constant
            def psi_curved(a, kc, k):
                # return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)  # old Phi equation
                return (k**2)**0.5 * ((k**2-2*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2-2*kc)*a**2 +1)  # new Phi equation 
            try:
                theta_int  = 2* quad(psi_curved, 0, 1, args=(kc, k))[0]
            except:
                theta_int = 0
            return theta_int

        for n in range(len(allowed_kc_list)):
            kc = allowed_kc_list[n]    
            plt.plot(k_int_list, [theta(kc, k_int) for k_int in k_int_list], label=r"$d\theta/dk_{\mathrm{int}}=$ "+slope_list[n], color=color_list[2+n], linewidth=1.2)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$k_{\mathrm{int}}$")
        plt.ylabel(r"$\theta(k_{\mathrm{int}}, a\rightarrow a_{\infty})$")
        plt.xlim([k_int_list[0], k_int_list[-1]])
        plt.ylim([2.e-1, theta(allowed_kc_list[2], k_int_list[-1])])
        # plt.xticks(k_int_list,fontsize=10)
        plt.legend()
        plt.savefig("theta-kint_log.pdf", bbox_inches="tight")

    ########################
    # Find curvature vallue (kc) with known Lambda (DE) by making slope of theta = N^{+-1}pi/2 (Lambda is known)
    # Note: (1) k_int: integer wave vectors, _bar: in K=1 unit
    #       (2) k_com_bar = sqrt( k_int*(k_int+2) ) 
    #       (3) k_com = k* sqrt(Lambda)
    #       (4) k_com_bar/a_bar = k_phys = k_com/a -> k = a_bar/a/sqrt(Lambda)*k_com_bar
    ########################

    def Find_kc(self, N_slope, n_k, n_range): # N_slope: slope=1/N or N, N is integer. n_k > n_range/2
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
                    k_com_bar_list = np.real([math.sqrt( k_int*(k_int+2)  ) for k_int in k_int_list])
                    # k_com_bar_list = np.real([math.sqrt( k_int**2 ) for k_int in k_int_list])
                else:       # open or flat universe
                    k_com_bar_list = np.real([math.sqrt( k_int**2  ) for k_int in k_int_list])

                k_list = np.real([self.a0/a0_bar.si.value/math.sqrt(self.Lambda.si.value)*k_com_bar for k_com_bar in k_com_bar_list])

                theta_list = [self.psi_int_curved(k, kc) for k in k_list]
                func_interpolate = interpolate.interp1d(k_int_list, theta_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function
                slope = 2./ np.pi * (func_interpolate(n_k) - func_interpolate(n_k-1)) / 1.     # calculate the slope at n_k
                print("slope="+str(slope))
                return slope

            ##### Calculate kc by root finder
            sol = root_scalar(lambda kc: theta_slope(kc) - N_slope, bracket=[0.001 , 0.9999999], method='brentq') 
            kc = sol.root 
            K = 2.* self.Lambda / 3. * kc
            Omega_K = - K * c**2 / self.a0**2 / self.H0**2

            print("Omega_K=", Omega_K)

            return kc, K, Omega_K

        else:
            raise ValueError("n_k should be larger than n_range/2.")


    ###############################
    # Plot the discrete comoving wave vectors according to the kc value we found
    ###############################


    def find_discrete_k(self, n_k, n_range):

        kc, K, Omega_K = self.Find_kc(1, 110, 20)
        k_int_list = np.linspace(n_k - n_range/2, n_k + n_range/2, 30)

        if kc > 0: K_bar = 1
        elif kc < 0: K_bar = -1
        else: K_bar = 0

        a0_bar = c * np.sqrt(-K_bar/Omega_K)/self.H0

        if kc > 0:  # close universe 
            k_com_bar_list = np.real([math.sqrt( k_int*(k_int+2) ) for k_int in k_int_list])
        else:       # open universe
            k_com_bar_list = np.real([math.sqrt( k_int**2 ) for k_int in k_int_list])

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
        # Suppose the non-integer behavior comes from k^2 = k_int^2-3K
        
        kc, K, Omega_K = self.Find_kc(1, 110, 20)
        k_int_list = np.linspace(n_k - n_range/2, n_k + n_range/2, n_range+1)

        if kc > 0: K_bar = 1
        elif kc < 0: K_bar = -1
        else: K_bar = 0

        a0_bar = c * np.sqrt(-K_bar/Omega_K)/self.H0

        if kc > 0:  # close universe 
            k_com_bar_list = np.real([math.sqrt( k_int*(k_int+2) ) for k_int in k_int_list])
            # k_com_bar_list = np.real([math.sqrt( k_int**2 ) for k_int in k_int_list])
        else:       # open universe
            k_com_bar_list = np.real([math.sqrt( k_int**2 ) for k_int in k_int_list])

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
        
        kc, K, Omega_K = self.Find_kc(1, 110, 20)
        k_int_list, theta_curved_list, discrete_k_int_list, discrete_theta_list = self.find_discrete_k(n_k, n_range)
        theta_curved_list = [2./np.pi * theta for theta in theta_curved_list]
        discrete_theta_list = [2./np.pi * theta for theta in discrete_theta_list]

        ## plot curved case 
        plt.plot(k_int_list, theta_curved_list, label = r"$\tilde\kappa=$"+str(round(kc,2))+', '+r"$\Omega_{\kappa,0}=$"+str(round(Omega_K.si.value,3)))
        plt.plot(discrete_k_int_list, discrete_theta_list, 'k.')

        # plt.xlim([100, 115])
        # plt.ylim([100*np.pi/2., 114*np.pi/2.])
        plt.vlines(discrete_k_int_list, 0, discrete_theta_list, colors='k', linestyles='dotted')
        plt.hlines(discrete_theta_list, 0, discrete_k_int_list, colors='k', linestyles='dotted')
        plt.xticks([n for n in range(n_k - n_range//2, n_k + n_range//2)], [str(n) for n in range(n_k - n_range//2, n_k + n_range//2)], fontsize=13)
        plt.yticks([n for n in range(n_k - n_range//2, n_k + n_range//2)], [str(n) for n in range(n_k - n_range//2, n_k + n_range//2)], fontsize=13)
        # plt.xticks([(n+101) for n in range(15)], [str(n+101) for n in range(15)], fontsize=13)
        # plt.yticks(discrete_theta_list, [str(n+101)+r"$\frac{\pi}{2}$"  for n in range(13)], fontsize=13)
        plt.xlabel(r"$k_{int}$ (integer wave vector)", fontsize=15)
        plt.ylabel(r"$\frac{2}{\pi}\theta(k)$", fontsize=15)
        plt.xlim([n_k - n_range/2, n_k + n_range/2])
        plt.ylim([n_k - n_range/2, n_k + n_range/2])
        plt.legend(fontsize=14)
        plt.savefig("theta_k_discrete.pdf")


H0 = 66.86 * u.km/u.s/u.Mpc # 70
Omega_lambda = 0.679  # 0.73
Omega_r_h2 = 2.47e-5 
universe = Universe(H0, Omega_lambda, Omega_r_h2)

# universe = Universe(H0, Omega_lambda, Omega_r_h2)
# universe.plot_k_theta_discrete(5, 8)

# k_list = np.linspace(1, 6, 300)
# color_list = ['darkblue', 'blueviolet', 'violet']
# kc_list = [0.9, 0, -0.9]
# universe.plot_k_theta(k_list, kc_list)

color_list = ['aliceblue', 'lightcyan', 'lightblue', 'lightskyblue', 'deepskyblue', 'blue', 'darkblue'] # slope = [1/5, 1/4, 1/3, 1/2, 1, 2, 3] respectively 
kc_list = np.linspace(0, 0.9999, 200)
# universe.slope_kc_eta_tot_without_matter(kc_list)
universe.plot_kint_theta_loglog()

# kc, K, Omega_K = universe.Find_kc(1, n_k=110, n_range=20)
# print('kc='+str(kc))

###############################
# Plot the allowed curves in the Omega_K, Omega_Lambda plane
###############################

def plot_OmegaK_OmegaLambda_plane(H0, Omega_lambda_list, Omega_r_h2, n_k, n_range):

    ### Plot Planck data 2018

    # MCMC chain samples
    samples = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE_4.txt')

    # load the column names for the samples
    column_names_raw = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE.paramnames', dtype=str, usecols=[0])
    column_names = [x.replace("b'",'').replace("'",'') for x in column_names_raw]

    # make a data frame with column names and samples
    samples1 = pd.DataFrame(samples[:,2:], columns=column_names) # first two columns are not important

    # define which parameters to use
    use_params = ['omegal*', 'omegak']
    ndim = len(use_params)
    # Make the base corner plot
    figure = corner.corner(samples1[use_params], range=[(0.6, 0.7), (-0.02, 0.0)], bins=20, color='r')
    # figure = corner.corner(samples1[use_params], range=[(0.48, 0.69), (-0.075, 0.005)], bins=30, color='r')
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    ax = axes[1, 0]
    ### plot my result
    N_slope_list = [1/3, 1/2, 1, 5]
    slope_label = [ r"$\frac{2}{\pi}$ slope $=1/3$", r"$\frac{2}{\pi}$ slope $=1/2$", r"$\frac{2}{\pi}$ slope $=1$", r"$\frac{2}{\pi}$slope $\gg 1$ "]
    for n in range(len(N_slope_list)): 
        N_slope = N_slope_list[n]
        universe_list = [Universe(H0, Omega_lambda, Omega_r_h2) for Omega_lambda in Omega_lambda_list]
        Omega_K_list = [universe.Find_kc(N_slope, n_k, n_range)[2] for universe in universe_list]

        # ax.plot(H0_list, Omega_K_list, label = r"$\Omega_{\lambda,0}="+str(Omega_lambda)+r", \Omega_{r,0}h^2=$"+'{:.2e}'.format(Omega_r_h2))
        ax.plot(Omega_lambda_list, Omega_K_list, label = slope_label[n], color=color_list[n],linewidth=1)
    ax.set_xlabel(r"$\Omega_{\lambda,0}$", fontsize=10)
    ax.set_ylabel(r"$\Omega_{\kappa,0}$", fontsize=10)
    # ax.legend(fontsize=8, loc='lower right')
    plt.savefig("Omegalambda-OmegaK-withPlanck2018_zoom-in.pdf")


### Set parameters

# H0_list = np.linspace(44, 70, 5) * u.km/u.s/u.Mpc
H0 = 66.86 * u.km/u.s/u.Mpc 
Omega_lambda_list = np.linspace(0.48, 0.7, 5)
Omega_r_h2 = 2.47e-5 
# Omega_lambda = 0.535  # 0.73

# plot_OmegaK_OmegaLambda_plane(H0, Omega_lambda_list, Omega_r_h2, n_k=110, n_range=20)

