import numpy as np
from scipy.optimize import root_scalar

params_set_1 = [405.83331811777197, 1.4738397441311173, 0.16167855679796067, 0.55768703176223] # 1052
params_set_2 = [410.94310700881147, 1.485771753408034, 0.15849793543489876, 0.5546865247545779] # 582
params_set_3 = [410.6147079906398, 1.485456782881211, 0.15391421508017622, 0.5659898883127065]

# Constants
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5  # photon density
Neff = 3.046

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
    return s0, Omega_lambda, Omega_m, Omega_K, Omega_r

for i in range(3):
    params = [params_set_1, params_set_2, params_set_3][i]
    mt, kt, omega_b_ratio, h = params
    print(cosmological_parameters(mt, kt, h))
