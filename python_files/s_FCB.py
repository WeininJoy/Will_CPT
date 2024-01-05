import numpy as np
import matplotlib.pyplot as plt
import cplot
import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import cmath
import math
from scipy.special import gamma
from sympy import sympify
from mpmath import ellipfun
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

plt.rcParams['text.usetex'] = True
plt.style.use('seaborn-poster')

class Universe:
    def __init__(self, Lambda, R, k_curved, n): # k_curved = - 1.2  # close:>0, open:<0, flat:0  , abs(value) < 1 : physical 
        
        ### cosmological constants
        self.Lambda = Lambda   # dark energy
        self.R = R    # radiation
        self.k_curved = k_curved
        self.K = k_curved * 1. / 3. * math.sqrt( 4.* Lambda * R )   
        self.n = n

    def a_tot(self): # when k_curved > 1 -> a_tot is finite 
        a_tot_small = math.sqrt( (3*self.K - math.sqrt(9*self.K**2 - 4*self.R*self.Lambda)) / (2* self.Lambda) )
        a_tot_large = math.sqrt( (3*self.K + math.sqrt(9*self.K**2 - 4*self.R*self.Lambda)) / (2* self.Lambda) )
        return a_tot_small

    def s_tot(self): # when k_curved > 1 -> a_tot is finite 
        s_tot_small = math.sqrt( (3*self.K - math.sqrt(9*self.K**2 - 4*self.R*self.Lambda)) / (2* self.R) )
        s_tot_large = math.sqrt( (3*self.K + math.sqrt(9*self.K**2 - 4*self.R*self.Lambda)) / (2* self.R) )
        return s_tot_large

    
    def eta_tot(self):
        if self.k_curved < 1:
            result = quad(lambda a: 1./self.a_dot(a), 0, np.inf)
            return result[0].real
        else: 
            result = quad(lambda a: 1./self.a_dot(a), 0, self.a_tot())
            return 2* result[0].real
    

    #################
    # 1 / scale factor: s = 1 / a(eta)
    #################
    
    def a_dot(self, a): 
        return math.sqrt( 1./3. * ( self.R - 3*self.K* a**2 + self.Lambda* a**4 ) )
    
    def s_dot(self, s): 
        return cmath.sqrt( 1./3. * ( self.R * s**4 - 3*self.K* s**2 + self.Lambda ) )


    def s(self, eta):
        if self.k_curved**2 < 1: # m is complex, choose m has positive imaginary part
            m = 3.*self.K / (2.*self.Lambda*self.R) * (3.*self.K + cmath.sqrt( (3*self.K)**2 - 4*self.Lambda*self.R ) ) - 1
            if m.imag < 0:
                m = 3.*self.K / (2.*self.Lambda*self.R) * (3.*self.K - cmath.sqrt( (3*self.K)**2 - 4*self.Lambda*self.R ) ) - 1
        else: # m is real, choose m < 1
            m = 3.*self.K / (2.*self.Lambda*self.R) * (3.*self.K + math.sqrt( (3*self.K)**2 - 4*self.Lambda*self.R ) ) - 1
            if m > 1:
                m =  3.*self.K / (2.*self.Lambda*self.R) * (3.*self.K - math.sqrt( (3*self.K)**2 - 4*self.Lambda*self.R ) ) - 1
        
        alpha = cmath.sqrt( self.R*(1+m) / (3*self.K) )
        beta = cmath.sqrt( self.K / (1+m) )
        u = beta * eta
        a_eta = alpha * ellipfun('sn', u, m)
        
        return 1. / a_eta.real

    
    def plot_s(self):
        ##### make plot of the real part of analytic solution
        eta_tot = self.eta_tot()
        eta_list = np.linspace(1.e-5, 2*eta_tot*(1.-1.e-5), 100)
        # choose s(eta) positive while eta[0] > 0 
        if eta_list[0] * self.s(eta_list[0]).real > 0:
            s_list = [self.s(eta).real for eta in eta_list]
        else: 
            s_list = [-self.s(eta).real for eta in eta_list]
        plt.plot(eta_list, s_list, color=color_list[self.n], label = 'kc='+str(round(self.k_curved, 2)))
        plt.axvline(x=eta_tot, color=color_list[self.n], ls='--')
    

    #################
    # Phi solution
    #################
    
    # def solve_Phi(self, Phi_i, Phi_dot_i, k_scale):
    #     # phi = Phi/s^2
    #     def phi_ode(s, y):
    #         eta, phi, phi_dot = y
    #         eta_dot = 1. / self.s_dot(s).real
    #         phi_2dot = - ( (2*s**2* (s**2-self.k_curved)) * phi_dot + (-2./s + (8*self.k_curved + k_scale**2)*s - 2*s**3) * phi ) / (s *(1+s**4) - 2*s**3*self.k_curved)
    #         return [eta_dot, phi_dot, phi_2dot]
        
    #     s_i = np.inf
    #     y0 = [0.0, Phi_i/s_i**2, Phi_dot_i/s_i**2 - 2.*Phi_i/s_i**3]
    #     phi_sol = solve_ivp(phi_ode, [float('inf'), float('-inf')], y0, rtol=1e-10, atol=1e-10)

    #     s_list, eta_list, phi_list, phi_dot_list = phi_sol.t, phi_sol.y[0], phi_sol.y[1], phi_sol.y[2]
    #     Phi_list = [s_list[i]**2*phi_list[i] for i in range(len(s_list))]
    #     Phi_dot_list = [s_list[i]**2*phi_dot_list[i] + 2.*Phi_list[i]/s_list[i] for i in range(len(s_list))]
    #     return [s_list, eta_list, Phi_list, Phi_dot_list]

    def solve_Phi(self, Phi_i, Phi_dot_i, k_scale):
        # Phi(s) = s^4*Phi(a)
        def Phi_ode(s, y):
            Phi, Phi_dot, eta = y
            Phi_2dot = - ( (2*(3*s**4 + 2) - 15*s**2*self.K/ self.Lambda) * Phi_dot + s* (4*s**2 + k_scale**2) * Phi ) / (s *(1+s**4) - 3*s**3*self.K/self.Lambda)
            eta_dot = 1./self.s_dot(s).real
            return [Phi_dot, Phi_2dot, eta_dot]
        
        if self.k_curved > 1:
            print("s goes to infinity at FCB.")
            Phi_sol = 0
        else:
            y0 = [0.0, Phi_i, Phi_dot_i]
            Phi_sol = solve_ivp(Phi_ode, [np.inf, -np.inf], y0, rtol=1e-10, atol=1e-10)

        return Phi_sol

    def anal_Phi(self, s, Phi_i, Phi_dot_i, k_scale):
        # Phi(s) = s^4*Phi(a)
        wolfSession = WolframLanguageSession()
        wolfSession.evaluate(wl.Needs("targetFunctions`"))
        
        # result = Phi_i * wolfSession.evaluate(wl.targetFunctions.PhiIntCurved((s, k_scale, self.k_curved))) # integration form
        result = Phi_i * wolfSession.evaluate(wl.targetFunctions.PhiCurved((s, k_scale, self.k_curved))) # Heun function form
        wolfSession.terminate()
        return s**4 * result


    def plot_Phi(self, Phi_i, Phi_dot_i, k_scale):
        
        def plot_Phi_anal():
            # plot analytic solution of Phi(eta)
        
            eta_tot = self.eta_tot()
            eta_list = np.linspace(1.e-5, 2*eta_tot*(1.-1.e-5), 5)
            
            # choose s(eta) positive while eta[0] > 0 
            if eta_list[0] * self.s(eta_list[0]) > 0:
                s_list = [round(float(self.s(eta)), 5) for eta in eta_list] 
            else: 
                s_list = [-round(float(self.s(eta)), 5) for eta in eta_list]
            
            if self.k_curved > 1.0:
                print("s goes to infinity at FCB.")
                Phi_list = [-1 for s in s_list] 
            else:
                Phi_list = [self.anal_Phi(s, Phi_i, Phi_dot_i, k_scale) for s in s_list]
            
            Phi_list = [Phi / Phi_list[0] for Phi in Phi_list] 
            plt.plot(eta_list, Phi_list, label="anal, kc="+str(round(self.k_curved, 2)))

        def plot_Phi_num():
        
            Phi_sol = self.solve_Phi(Phi_i, Phi_dot_i, k_scale)
            s_list, Phi_list, Phi_dot_list, eta_list = Phi_sol.t, Phi_sol.y[0], Phi_sol.y[1], Phi_sol.y[2]
            plt.plot(eta_list, Phi_list, label="kc="+str(round(self.k_curved, 2)))
    
        plot_Phi_anal()


    def plot_V(self, Phi_i, Phi_dot_i, k_scale):
        ### velocity perturbation of radiation
        
        k = k_scale * (self.Lambda)**0.5  # k is dimensional wave vector

        def V_flat(a, Phi, Phi_dot):
            a_dot = self.a_dot(a)
            return 3./2. * (k* (a_dot/a*Phi + Phi_dot) ) / (3*a_dot**2/a**2 + 3*self.K - self.Lambda*a**2)

        def V_curved(a, Phi, Phi_dot):
            a_dot = self.a_dot(a)

            if self.K > 0:
                int_f = np.pi/2. * (1. - math.cos(k_scale/(self.K)**0.5) - k/(self.K)**0.5*math.sin(k_scale/(self.K)**0.5))
            else: 
                int_f = np.pi/2. * (1. - math.exp(-k_scale/(-self.K)**0.5) * (1. + k_scale/(-self.K)**0.5))
            
            return 3./2. * (k* (a_dot/a*Phi + Phi_dot) ) / (3*a_dot**2/a**2 + 3*self.K - self.Lambda*a**2) / (1 + self.K * (4.*np.pi/k_scale**3) * int_f)
            
        Phi_sol = self.solve_Phi_a(Phi_i, Phi_dot_i, k_scale)
        a_list = Phi_sol.t
        Phi_list = Phi_sol.y[0]
        Phi_dot_list = Phi_sol.y[1]        
        
        if self.K == 0:
            plt.plot(a_list, [V_flat(a_list[i], Phi_list[i], Phi_dot_list[i]) for i in range(len(a_list))], label="flat")
        else: 
            plt.plot(a_list, [V_curved(a_list[i], Phi_list[i], Phi_dot_list[i]) for i in range(len(a_list))], label="kc="+str(round(self.k_curved, 2)))
            

    def plot_delta(self, Phi_i, Phi_dot_i, k_scale):
        ### density perturbation of radiation
        
        k = k_scale * (self.Lambda)**0.5  # k is dimensional wave vector

        def delta_flat(a, Phi, Phi_dot):
            a_dot = self.a_dot(a)
            return - 2* (3*a_dot/a*Phi_dot + k**2*Phi + self.Lambda*Phi) / (3*a_dot**2/a**2 + 3*self.K - self.Lambda*a**2) - 2.*Phi
        
        def dPhidk(a): 
            ### dPhidk solution in curved universe
            wolfSession = WolframLanguageSession()
            wolfSession.evaluate(wl.Needs("targetFunctions`"))  
            result = Phi_i * wolfSession.evaluate(wl.targetFunctions.dPhidkCurved((a, k, self.k_curved)))
            wolfSession.terminate()
            return result

        def d2Phidk(a): 
            ### d2Phidt solution in curved universe
            wolfSession = WolframLanguageSession()
            wolfSession.evaluate(wl.Needs("targetFunctions`"))
            result = Phi_i * wolfSession.evaluate(wl.targetFunctions.d2PhidkCurved((a, k, self.k_curved)))
            wolfSession.terminate()
            return result
        
        def delta_curved(a, Phi, Phi_dot):
            a_dot = self.a_dot(a)
            return - 2* (3*a_dot/a*Phi_dot + k**2*Phi + self.Lambda*Phi + self.K* (-9.*Phi + k*dPhidk(a) + k**2*d2Phidk(a))) / (3*a_dot**2/a**2 + 3*self.K - self.Lambda*a**2) - 2.*Phi
            
        Phi_sol = self.solve_Phi_a(Phi_i, Phi_dot_i, k_scale)
        a_list = Phi_sol.t
        Phi_list = Phi_sol.y[0]
        Phi_dot_list = Phi_sol.y[1]

        list_space = 3
        idx_list = []
        for i in range(len(a_list)):
            if i%list_space == 0:
                idx_list.append(i)      
        
        if self.K == 0:
            plt.plot( [a_list[idx] for idx in idx_list], [delta_flat(a_list[idx], Phi_list[idx], Phi_dot_list[idx]) for idx in idx_list], label="flat")
        else: 
            plt.plot( [a_list[idx] for idx in idx_list], [delta_curved(a_list[idx], Phi_list[idx], Phi_dot_list[idx]) for idx in idx_list], label="kc="+str(round(self.k_curved, 2)))


######################
# plot s(eta) 
######################

# k_curved_list = [-1.2, -0.6, 1.e-5, 0.6, 1.2] 
# color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', ]

# for n in range(len(k_curved_list)):
#     universe = Universe(Lambda=1., R=1., k_curved=k_curved_list[n], n=n)
#     universe.plot_s()

# plt.axhline(y=0, color='k', ls='--')
# plt.ylim([-8, 8])
# plt.xlabel(r"$\eta$", fontsize=30)
# plt.ylabel(r"$s(\eta)$", fontsize=30)
# plt.xticks(fontsize=28)
# plt.yticks(fontsize=28)
# plt.legend(fontsize=28)
# plt.savefig("s-eta.pdf")

######################
# plot V(a) or delta(a)
######################

# k_curved_list = [-0.5, -0.01, 0, 0.01, 0.5] 
# Phi_i, Phi_dot_i, k_scale = 1., 0., 10

# for n in range(len(k_curved_list)):
#     universe = Universe(Lambda=1., R=1., k_curved=k_curved_list[n], n=n)
#     universe.plot_delta(Phi_i, Phi_dot_i, k_scale)

# plt.xlabel(r"$a$", fontsize=30)
# plt.ylabel(r"$\delta(a)$", fontsize=30)
# plt.xticks(fontsize=28)
# plt.yticks(fontsize=28)
# plt.legend(fontsize=28)
# plt.savefig("a-delta.pdf")

######################
# plot solution of a 
######################

# k_curved_list = [-0.8, -0.4, 0.001, 0.4, 0.8] 
# color_list = ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple']

# for n in range(len(k_curved_list)):
#     universe = Universe(Lambda=1., R=1., k_curved=k_curved_list[n], n=n)
#     universe.plot_a()

# plt.xlabel(r"$\eta$", fontsize=30)
# plt.ylabel(r"$\log a(\eta)$", fontsize=30)
# plt.xticks(fontsize=28)
# plt.yticks(fontsize=28)
# plt.yscale('log')
# plt.ylim([2.e-2, 5.e1])
# plt.legend(fontsize=28)
# plt.savefig("eta-log_a.pdf")


######################
# plot solution of Phi
######################

k_curved_list = [0.001]        

for n in range(len(k_curved_list)):
    universe = Universe(Lambda=1., R=1., k_curved=k_curved_list[n], n=n)
    Phi_i, Phi_dot_i, k_scale = 1., 0., 3.1167
    universe.plot_Phi(Phi_i, Phi_dot_i, k_scale)

plt.xlabel(r"$\eta$", fontsize=30)
plt.ylabel(r"$\Phi(s)$", fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.ylim([-6., 6.])
plt.legend(fontsize=28)
plt.savefig("eta-Phi.pdf")

