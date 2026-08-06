import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def simplified_ode(s, y, k, m):
    """
    Defines the simplified second-order ODE.

    Args:
        a: Independent variable (a).
        y: Array of variables [psit, dpsitda].
        k: Constant parameter.

    Returns:
        Derivatives [dpsitda, d2psitda2].
    """
    psit, dpsitds = y

    if s == 0:
        d2psitds2 = 0 #Avoid division by zero
    else:
        d2psitds2 = - ( (3 * m**4 + m**2 * (-2 + 3 * k**2 * s**2 - 6 * s**4) + 4 * k**2 * s**2 * (1 + s**4)) * psit - 2 * s**2 * (-m**2 * s**3 + k**2 * (s + s**5)) * dpsitds ) /  (m**2 * (1 + s**4) * s**2)
    
    return [dpsitds, d2psitds2]

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
    solution = solve_ivp(simplified_ode, s_span, initial_conditions, args=(k,m), rtol=1e-3, atol=1e-6, method='RK23')
    return solution

# Make the plot
plt.figure(figsize=(10, 6))

k, m = 1.0, 2
initial_conditions = [0.0, 1.0]  # [psit(s=0), psit'(s=0)]

for i in range(2):
    if i ==0: s_span = (1.e-5, 1.0)  # a interval (avoiding a=0)
    else: s_span = (-1.e-5, -1.0)  # a interval (avoiding a=0)

    solution = solve_simplified_ode(initial_conditions, s_span, k, m)

    # Accessing the solution:
    s = solution.t
    psit = solution.y[0]
    dpsitds = solution.y[1]
    plt.plot(s, psit)


# Plotting the results (optional):
import matplotlib.pyplot as plt

plt.xlabel('s')
plt.ylabel(r'$\psi_t(s)$')
plt.title('Solution of Simplified ODE w.r.t. s')
plt.grid(True)
plt.savefig(f"psit_s_k{k}_m{m}.pdf")