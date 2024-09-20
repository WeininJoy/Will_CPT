import numpy as np
from scipy.optimize import minimize

#####################
# lambda = radiation
#####################
# omega_lambda = 0.679
# omega_rh2 = 2.47e-5
# H0 = 66.88
# h = H0 / 100
# omega_r = omega_rh2 / h**2

# # predicted values
# kt = 0.31
# mt = 3.75
# # kt = -omega_k / 3/ np.sqrt(omega_lambda * omega_r)
# # mt = omega_m / omega_lambda**(1./4.) / omega_r**(3./4.)

# omega_k = -3 * kt * np.sqrt(omega_lambda * omega_r)
# omega_m = mt * omega_lambda**(1./4.) * omega_r**(3./4.)

# print(omega_k, omega_m)

#####################
# lambda = matter
#####################

# omega_lambda = 0.679
# omega_m = 0.315
# H0 = 66.88
# h = H0 / 100
# # omega_rh2 = 2.47e-5
# # omega_r = omega_rh2 / h**2

# # predicted values
# kt = 0.13
# rt = 0.18
# # kt = -omega_k / 3/ omega_m**(2./3) / omega_lambda**(1./3)
# # rt = omega_r / omega_m**(4./3.) / omega_lambda**(-1./3.)

# omega_k = -3 * kt * omega_m**(2./3) * omega_lambda**(1./3)
# omega_r = rt * omega_m**(4./3.) * omega_lambda**(-1./3.)

# print(omega_k, omega_r*h**2)


#####################
# find H0 and omega_lambda, such that the omega_k match
#####################

def function(params):
    H0, omega_lambda = params
    def f1(omega_rh2):
        # define parameters
        h = H0 / 100
        omega_r = omega_rh2 / h**2
        # predicted values
        kt = 0.31
        mt = 3.75
        # calculate omega_k and omega_m
        omega_k = -3 * kt * np.sqrt(omega_lambda * omega_r)
        omega_m = mt * omega_lambda**(1./4.) * omega_r**(3./4.)
        return [omega_k, omega_m]
    
    def f2(omega_m):
        # define parameters
        h = H0 / 100
        # predicted values
        kt = 0.13
        rt = 0.18
        # calculate omega_k and omega_r
        omega_k = -3 * kt * omega_m**(2./3) * omega_lambda**(1./3)
        omega_r = rt * omega_m**(4./3.) * omega_lambda**(-1./3.)
        return [omega_k, omega_r*h**2]
    
    def difference(omega_rh2_initial):
        omega_k, omega_m = f1(omega_rh2_initial)
        omega_k2, omega_rh2 = f2(omega_m)
        print('omega_rh2_calculated=', omega_rh2)
        print('omega_m_calculated=', omega_m)
        print('omega_k_calculated=', omega_k)
        print('omega_k2_calculated=', omega_k2)
        return (omega_k - omega_k2)**2 / omega_k2**2 + (omega_rh2 - omega_rh2_initial)**2 / omega_rh2**2
    
    omega_rh2_initial =0.02 
    bounds = [(0, 0.1)]
    result = minimize(difference, omega_rh2_initial, bounds=bounds, tol=1.e-8)
    print('omega_rh2_min=', result.x)
    error = result.fun
    print('error=', error)
    return error

print(function([67,0.7]))
# H0_initial = 66.88
# omega_lambda_initial = 0.679
# bounds = [(50, 100), (0, 1)]
# result = minimize(function, [H0_initial, omega_lambda_initial], bounds=bounds, tol=1.e-8)
# print('H0_min=', result.x[0])
# print('omega_lambda_min=', result.x[1])
