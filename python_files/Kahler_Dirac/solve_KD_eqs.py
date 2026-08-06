import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coupled_odes(t, y, m, mt, rt, k, lambda_val):
    """
    Defines the system of coupled ODEs.

    Args:
        t: Time variable (not directly used in the equations, but required by solve_ivp).
        y: Array of variables [a, psi, psit, phix].
        m, mt, rt, k, lambda_val: Constant parameters.

    Returns:
        Derivatives [dadt, dpsidt, dpsitdt, dphixdt].
    """
    a, psi, psit, phix = y
    dadt = np.sqrt(lambda_val / 3 * (rt + mt * a + a**4))

    if a == 0: #To avoid division by zero
        dpsidt = 0
        dpsitdt = 0
        dphixdt = 0
    else:

        dpsidt = ( (m**2 * a**2 + k**2) * m * psit - 2 * dadt / a * k**2 * psi) / (m**2 * a**2)
        dpsitdt = -2 * psit * dadt / a - m * a**2 * psi
        dphixdt = -m / a**2 * psi

    return [dadt, dpsidt, dpsitdt, dphixdt]

def solve_coupled_odes(initial_conditions, t_span, m, mt, rt, k, lambda_val):
    """
    Solves the coupled ODEs using solve_ivp.

    Args:
        initial_conditions: List or array of initial values [a0, psi0, psit0, phix0].
        t_span: Tuple (t_start, t_end) defining the time interval.
        m, mt, rt, k, lambda_val: Constant parameters.

    Returns:
        scipy.integrate.OdeSolution object containing the solution.
    """
    solution = solve_ivp(coupled_odes, t_span, initial_conditions, args=(m, mt, rt, k, lambda_val), rtol=1e-3, atol=1e-6, method='RK23')
    return solution

# Example usage:
m = 1.0
mt = 400
rt = 1
k = 2.0
lambda_val = 1

initial_conditions = [1.e-3, 1, 1, 1]  # [a(0), psi(0), psit(0), phix(0)]
t_span = (1.e-3, 1)  # Time interval

solution = solve_coupled_odes(initial_conditions, t_span, m, mt, rt, k, lambda_val)

# Accessing the solution:
t = solution.t
a = solution.y[0]
psi = solution.y[1]
psit = solution.y[2]
phix = solution.y[3]


plt.figure(figsize=(10, 6))
plt.plot(t, a, label='a(t)')
plt.plot(t, psi, label='psi(t)')
plt.plot(t, psit, label='psit(t)')
plt.plot(t, phix, label='phix(t)')
plt.xlabel('t')
plt.ylabel('Variables')
plt.title('Solution of Coupled ODEs')
plt.legend()
plt.grid(True)
plt.savefig(f"Kahler_Dirac_solution_m{m}_k{k}.pdf", bbox_inches='tight')

# #Accessing specific values at a time using solution.sol
# specific_time = 5.0
# a_specific = solution.sol(specific_time)[0]
# psi_specific = solution.sol(specific_time)[1]
# psit_specific = solution.sol(specific_time)[2]
# phix_specific = solution.sol(specific_time)[3]

# print(f"At t={specific_time}:")
# print(f"a({specific_time}) = {a_specific}")
# print(f"psi({specific_time}) = {psi_specific}")
# print(f"psit({specific_time}) = {psit_specific}")
# print(f"phix({specific_time}) = {phix_specific}")