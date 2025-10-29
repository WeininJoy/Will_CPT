import numpy as np
import os
import re
from scipy.optimize import root_scalar

data_folder = './try_intK_planck_bounds/'

# Define the directory containing the try folders
pattern = re.compile(r"try_(\d+)")
min_loss = float('inf')
min_loss_num = 0

if os.path.exists(data_folder):
    for dirname in os.listdir(data_folder):
        match = pattern.match(dirname)
        if match:
            num = int(match.group(1))  # Extract the number and convert to integer
            # Only count this try if loss_params_{num}.txt exists
            loss_params_file = f'{data_folder}/{dirname}/loss_params_{num}.txt'
            if os.path.exists(loss_params_file):
                with open(loss_params_file, 'r') as f:
                    loss = float(f.read().strip().split()[0])
                    if loss < min_loss:
                        min_loss = loss
                        min_loss_num = num

print("min_loss:", min_loss)
print("min_loss_num:", min_loss_num)

allowedK_file = f'{data_folder}/try_{min_loss_num}/allowedK_integer.npy'
allowedK_integer = np.load(allowedK_file)
difference_allowedK = [allowedK_integer[i+1] - allowedK_integer[i] for i in range(len(allowedK_integer)-1)]
print("allowedK_integer:", allowedK_integer)
print("difference_allowedK:", difference_allowedK)

loss_params_file = f'{data_folder}/try_{min_loss_num}/loss_params_{min_loss_num}.txt'
with open(loss_params_file, 'r') as f:
    loss_params = f.read().strip().split()
    parameters_best = [float(x) for x in loss_params[1:]] # mt, kt, omega_b_ratio, h
print("parameters_best:", parameters_best)

#################
# Parameters
#################
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density V
Neff = 3.046
N_ncdm = 1  # number of massive neutrino species
m_ncdm = 0.06  # mass of massive neutrino species in e
epsilon = 1e-2 # the accuracy of kt rootfinding
nu_spacing4_bestfit = [401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583, 1.9375648884116028, 0.9787493821596979, 0.019760560255556746] # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$, $A_s$, $n_s$, $\tau$ from nu_spacing=4 best-fit data

##################
def cosmological_parameters(mt, kt, h): 

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

s0, Omega_lambda, Omega_m, Omega_K = cosmological_parameters(parameters_best[0], parameters_best[1], parameters_best[3])
print("Omega_lambda, Omega_m, Omega_K:", Omega_lambda, Omega_m, Omega_K)