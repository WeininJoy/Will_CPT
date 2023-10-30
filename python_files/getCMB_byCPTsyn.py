import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
import math

from scipy.constants import sigma, c, hbar,G, m_p
from scipy.constants import k as kB
from scipy.constants import physical_constants
from scipy.integrate import solve_ivp

#######################
# Define constants and ODEs
#######################

### constants

T0 = 2.7255
Mpc = 3.086e22  # meter
lp = physical_constants["Planck length"][0]  # meter
rhogamma = np.pi**2*kB**4/(15*hbar**3*c**3) * T0**4/c**2
H0 = 67.4 * 1e3 / Mpc # second
Or = 8*np.pi*G*rhogamma /(3*H0**2) 
Om = 0.3
Ok = -0.01
Ol = 1-Om-Ok-Or

### inflation
m_p = 1
V0 = m_p**4
K = 1

def potential(phi):
    #return V0* phi**(4./3.)
    return V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2

def potential_prime(phi):
    #return V0* 4./3. * phi**(1./3.)
    return 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- math.sqrt(2.0/3.0)* phi/m_p)

def phi_dot(phi): # when start of inflation, V = phi_prime**2
    return - math.sqrt(V0)*( 1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p))

def N_dot(phi, N): # when start of inflation
    return math.sqrt( 1.0/(2.0*m_p**2) * phi_dot(phi)**2 - K* np.exp(-2.0*N) )

# define ODE
def f(t, y):
    phi = y[0]
    N = y[1]
    dphidt = y[2]
    dNdt = y[3]

    # define each ODE
    d2phidt = - 3.0*dNdt*dphidt - potential_prime(phi)
    d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K* np.exp(-2*N)

    return [dphidt, dNdt, d2phidt, d2Ndt]

# inflating event
def inflating(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 - potential(phi)

inflating.terminal = True
inflating.direction = 1


### radiation

def g(t, y):
    N, R, dR = y

    H = (Or * np.exp(-4*N) + Om * np.exp(-3*N)  + Ok * np.exp(-2*N) + Ol)**0.5

    ddR = -3*H*dR - w * k**2*np.exp(-2*N)*R
    
    return H, dR, ddR


def stop(t, y):
    N, Phi, dPhi = y
    return N + np.log(1+zstop)


stop.terminal = True


#######################
# Inflation (R start to oscillate after entering the Horizon)
#######################

### constants
# w = 1/3
# Ni = -18
# ti = np.exp(2*Ni)/Or**0.5/2

# ### get comoving wave vectors
# a0 = np.sqrt(-K/Ok)*c/H0 # meter
# k_phys_list = np.logspace(-4,-1,500) / Mpc  # meter^-1
# k_com_list = [k_phys*a0 for k_phys in k_phys_list]


### plot t-R

# zstop = 1000
# y0=[Ni,1,0]

# for k in k_com_list:

#     sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
#     plt.plot(np.log(sol.t), sol.y[1], label=r"k="+r'{:.2e}'.format(k))

# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.xlabel(r"$t$", fontsize=15)
# plt.ylabel(r"$R$", fontsize=15)
# plt.legend(fontsize=15)
# plt.savefig("t-R_inflation.pdf")


### plot k-Phi^2

# zstop = 1000

# Phi2_list = []
# for k in k_com_list:

#     Phi2 = 0
#     for Ri in np.linspace(-1, 1, 10):
#         y0 = [Ni, Ri, 0]
#         sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
#         Rrec = sol.y_events[0][-1][1]
#         Phi2 += Rrec**2
    
#     Phi2_list.append(Phi2)

# plt.plot(k_phys_list, Phi2_list)
# plt.semilogy()
# plt.ylim([1.e-6, 1.e1])
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.xlabel(r"$k$ (physical)", fontsize=15)
# plt.ylabel(r"$\Phi^2(k)$", fontsize=15)
# plt.savefig("k-Phi2_atCMB_inflation.pdf")


#######################
# CPT (R start to oscillate since ti=0 and ai=0)
#######################

# ### constants
Ni = -20
ti = 0
w = 1/3

# ### get comoving wave vectors
# zstop = 0
# k = 0
# y0 = [Ni, 0, 0]
# sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
# t0 = sol.t_events[0][-1]
# N0 = sol.y_events[0][-1][0]
# a0 = np.exp(N0) * lp  # meter

# k_phys_list = np.logspace(-4,-1,5) / Mpc  # meter^-1
# k_com_list = [k_phys*a0 for k_phys in k_phys_list]
# print(k_com_list)


### plot t-R

# zstop = 1000
# y0=[Ni,1,0]

# for k in np.logspace(5,6,5):
    
#     sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
#     plt.plot(np.log(sol.t), sol.y[1], label=r"k="+r'{:.2e}'.format(k))

# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# # plt.xlim([-33, -9])
# plt.xlabel(r"$t$", fontsize=15)
# plt.ylabel(r"$R$", fontsize=15)
# plt.legend(fontsize=15)
# plt.savefig("t-R_CPT_test_highk.pdf")


### plot k-Phi^2

zstop = 1000

k_list = np.logspace(2,5,500)
Phi2_list = []
for k in k_list:

    Phi2 = 0
    for Ri in np.linspace(-1, 1, 10):
        y0 = [Ni, Ri, 0]
        sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
        Rrec = sol.y_events[0][-1][1]
        Phi2 += Rrec**2
    
    Phi2_list.append(Phi2)

plt.plot(k_list, Phi2_list)

plt.semilogy()
plt.ylim([1.e-6, 1.e1])
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel(r"$k$", fontsize=15)
plt.ylabel(r"$\Phi^2(k)$", fontsize=15)
plt.savefig("k-Phi2_atCMB_CPT_lowk.pdf")