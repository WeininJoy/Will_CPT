import numpy as np
from scipy.integrate import solve_ivp

def simplified_ode(a, y, k, m):
    """
    Defines the simplified second-order ODE.

    Args:
        a: Independent variable (a).
        y: Array of variables [psit, dpsitda].
        k: Constant parameter.

    Returns:
        Derivatives [dpsitda, d2psitda2].
    """
    psit, dpsitda = y

    if a == 0:
        d2psitda2 = 0 #Avoid division by zero
    else:

        d2psitda2 = (-2 * a * (a**6 + (1 + a**4) * k**2/m**2) * dpsitda - (4 * k**2/m**2 - 6 * a**2 + k**2*(4/m**2 +3) * a**4 + (3*m**2-2)*a**6) * psit) / (a**4 * (1 + a**4))

    return [dpsitda, d2psitda2]

def solve_simplified_ode(initial_conditions, a_span, k, m):
    """
    Solves the simplified ODE using solve_ivp.

    Args:
        initial_conditions: List or array of initial values [psit0, dpsitda0].
        a_span: Tuple (a_start, a_end) defining the a interval.
        k, m: Constant parameters.

    Returns:
        scipy.integrate.OdeSolution object containing the solution.
    """
    solution = solve_ivp(simplified_ode, a_span, initial_conditions, args=(k,m), rtol=1e-3, atol=1e-6, method='RK23')
    return solution

# Example usage:
k, m = 1.0, 2
initial_conditions = [0.0, 1.0]  # [psit(a0), psit'(a0)]
a_span = (1.e-3, 1.0)  # a interval (avoiding a=0)

solution = solve_simplified_ode(initial_conditions, a_span, k, m)

# Accessing the solution:
a = solution.t
psit = solution.y[0]
dpsitda = solution.y[1]

# Plotting the results (optional):
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(a, psit, label=r'$\psi_t(a)$')
# plt.yscale('log')
# plt.xscale('log')
# plt.plot(a, dpsitda, label='psit\'(a)')
plt.xlabel('a')
plt.ylabel('Variables')
plt.title('Solution of Simplified ODE')
plt.grid(True)
plt.legend()
plt.savefig(f"psit_a_k{k}_m{m}.pdf")

# #Accessing specific values at an a using solution.sol
# specific_a = 5.0
# psit_specific = solution.sol(specific_a)[0]
# dpsitda_specific = solution.sol(specific_a)[1]

# print(f"At a={specific_a}:")
# print(f"psit({specific_a}) = {psit_specific}")
# print(f"psit'({specific_a}) = {dpsitda_specific}")