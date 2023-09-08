import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from scipy import interpolate
from scipy.optimize import root_scalar

plt.rcParams['text.usetex'] = True

###############
# Flat universe
###############

def psi_flat(a, k):
    return k * (k**4-4)**0.5 * a**2/(a**4+1)**0.5 / (1+a**2*k**2+a**4)

def psi_int_flat(k):
    result, err = quad(psi_flat, 0, float('inf'), args=(k))
    return result

###############
# Curved universe
###############


def psi_curved(a, k, kc):
    return (k**2+8*kc)**0.5 * ((k**2+6*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2+6*kc)*a**2 +1)

def psi_int_curved(k, kc):
    result, err = quad(psi_curved, 0, float('inf'), args=(k, kc))
    return result

def psi_int_anal(a, k, kc): 
    wolfSession = WolframLanguageSession()
    wolfSession.evaluate(wl.Needs("targetFunctions`"))
    
    result = wolfSession.evaluate(wl.targetFunctions.PsiIntCurved((a, k, kc)))
    wolfSession.terminate()
    return result

###############
# plot K-theta(K)
###############

k_list = np.linspace(0, 6, 50)
# kc_list = np.linspace(1.4, -1.4, 5)
kc_list = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
a = 1

# plt.plot(k_list, [psi_int_flat(k) for k in k_list], label='num, kc=0')
for kc in kc_list:
    if kc < 1:
        plt.plot(k_list, [psi_int_curved(k, kc) for k in k_list], label='num, kc='+str(kc))
    else:
        plt.plot(k_list, [psi_int_curved(k, kc) for k in k_list], '-.',label='num, kc='+str(kc))
    # plt.plot(k_list, [psi_int_anal(a, k, kc) for k in k_list], label='anal, kc='+str(kc))

plt.xlabel(r"$K$")
plt.ylabel(r"$\theta(K)$")
plt.legend()
plt.savefig("psi_diffK_close.pdf")



################################
# Find discrete k (kc is known)
################################

# kc = 1.9210779397786009

# # Use interpolator to find an approximated function (for root finder)
# k_list = np.linspace(0, 20, 50)
# y_list = [psi_int_curved(k, kc) for k in k_list]
# func_interpolate = interpolate.interp1d(k_list,y_list,kind='cubic') 

# # Use root finder to find discrete k 
# discrete_theta_list = [(n+5)*np.pi/2 for n in range(5)]
# discrete_k_list = []
# for theta in discrete_theta_list:
#     sol_k = root_scalar(lambda k: func_interpolate(k) - theta, bracket=[0, 20], method='brentq') 
#     discrete_k_list.append(sol_k.root)

# print(discrete_k_list)

################################
# Find the kc which theta(k) has slope = 1 
################################

# def slope_func(kc):
#     k_list = np.linspace(0, 10, 30)
#     y_list = [psi_int_curved(k, kc) for k in k_list]
#     func_interpolate = interpolate.interp1d(k_list,y_list,kind='cubic') 
#     slope = 2./ np.pi * (func_interpolate(10) - func_interpolate(9)) / 1.
#     return slope - 1. 

# try: 
#     sol_slope = root_scalar(slope_func, bracket=[1.2, 2.2], method='brentq')
#     print('kc='+str(sol_slope.root))
# except:
#     print('Cannot find kc in the range.')

################################
# Find the kc which has integer discrete k 
################################

# achieve this goal by changing dark energy (Lambda_DE)

n_k = 10

def find_Lambda(Lambda_DE):

    def slope_func(kc):
        k_list = np.linspace(0, 20, 30)
        y_list = [psi_int_curved(k, kc) for k in k_list]
        func_interpolate = interpolate.interp1d(k_list,y_list,kind='cubic') 
        slope = 2./ np.pi / Lambda_DE**0.5 * (func_interpolate(10) - func_interpolate(9)) / 1.
        return slope - 1. 

    try: 
        sol_slope = root_scalar(slope_func, bracket=[0, 3], method='brentq')
        print('kc='+str(sol_slope.root))
        kc = sol_slope.root

        k_list = np.linspace(0, 20, 30)
        y_list = [psi_int_curved(k, kc) for k in k_list]
        func_interpolate = interpolate.interp1d(k_list,y_list,kind='cubic') 
        theta = n_k * np.pi/2
        sol_k = root_scalar(lambda k: func_interpolate(k) - theta, bracket=[0, 20], method='brentq') 
        return sol_k.root * Lambda_DE**0.5 - n_k

    except:
        print('Cannot find kc in the range.')
        
try: 
    sol_Lambda = root_scalar(find_Lambda, bracket=[0.5, 1.], method='brentq')
    print('Lambda_DE='+str(sol_Lambda.root)) 
except:
    print('Cannot find Lambda_DE in the range.')