import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy import interpolate
from scipy.integrate import quad
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# assume 4piG=1

####################
# Solve dim-0 scalar field equation \box^2 \Phi = C \rho, \rho is density profile
####################

# Define the radial grid
r_initial = 1.e-1 # Avoid singularity at r=0
r_final = 1.e2
r_points = np.logspace(np.log10(r_initial), np.log10(r_final), 100)  # Create an array of r values

# Define the parameters of the density profile
rho0, rs = 1.0, 1.0  
C = 1.0  # Define the constant C = c_chi/N0

def density_profile(r):
    ## NFW profile
    return rho0 / (r/rs) / (1 + r/rs)**2
    ## luminous_profile
    # return rho0 * np.exp(-r/rs)

rho_solution = density_profile(r_points)  # Extract the solution for rho(r)

# Define the system of first-order ODEs
def odes(r, y):
    phi, dphi, d2phi, d3phi = y  # Unpack the variables
    d4phi = C * density_profile(r) - (4/r) * d3phi
    return [dphi, d2phi, d3phi, d4phi]

# Boundary conditions -  You'll need to specify these based on your problem
r_max = 1.e2
r_span = (r_max, 0)
y_initial = [0.0, 0.0, 0.0, 0.0]  # Initial condition at r_max: phi(0)=0, dphi(0)=d2phi(0)=d3phi(0)=0

# Solve the ODE system
sol = integrate.solve_ivp(odes, r_span, y_initial, dense_output=True)

# Evaluate the solution at desired points
phi_solution = sol.sol(r_points)[0]  # Extract the solution for f(r)
phi_interpolate = interpolate.interp1d(r_points, phi_solution, fill_value="extrapolate")

###################
# Analytic solutions phi\propto r, r^2
###################
# def int_alpha(r):
#     return C*density_profile(r)* r**3

# def int_beta(r):
#     return 2*np.pi/3 * C*density_profile(r)* r**4

# def alpha(r):
#     return quad(int_alpha, 0, r)[0]

# def beta(r):
#     return quad(int_beta, 0, r)[0]

# alpha_solution = np.array([alpha(r) for r in r_points])
# beta_solution = np.array([beta(r) for r in r_points])

# def minimize_func(x):
#     A, B, turning_point = x
#     result = 0
#     for i in range(len(r_points)):
#         r = r_points[i]
#         if 1.e-3 < r < 1.e-1:
#            result += (np.log(A*alpha(r)) - np.log(phi_solution[i]))**2
#         elif r > (turning_point*0.5 + r_final*0.5):
#            result += (np.log(B*beta(r)) - np.log(phi_solution[i]))**2
        
#     return result

# x0 = np.array([1.0, 1.0, 5])
# res = minimize(minimize_func, x0, method='nelder-mead')
# A, B, turning_point = res.x
# print(f"A: {A}, B: {B}, turning_point: {turning_point}")

###################
# Analytic solutions with phi(infty)=0, phi'(infty)=0, phi''(infty)=0, phi'''(infty)=0
###################
def int_f1(r): 
    return C*density_profile(r)* r
def int_f2(r):
    return C*density_profile(r)* r**2
def int_f3(r):
    return C*density_profile(r)* r**3
def int_f4(r):
    return C*density_profile(r)* r**4

def f1(r): 
    return -r**2/6 * quad(int_f1, r, np.inf)[0] 
def f2(r): 
    return - r/2 * quad(int_f2, 0, r)[0] 
def f3(r):
    return -1/2 * quad(int_f3, r, np.inf)[0] 
def f4(r):
    return -1/(6*r) * quad(int_f4, 0, r)[0] 

f1_solution = np.array([f1(r) for r in r_points])
f2_solution = np.array([f2(r) for r in r_points])
f3_solution = np.array([f3(r) for r in r_points])
f4_solution = np.array([f4(r) for r in r_points])
f_total = f1_solution + f2_solution + f3_solution + f4_solution
# phi_interpolate = interpolate.interp1d(r_points, f_total, fill_value="extrapolate")

# ###################
# # vaccum solution phi = A*r + B*r^2
# f1_vaccumm = np.array([r for r in r_points])
# f2_vaccumm = np.array([r**2 for r in r_points])
# def minimize_func(x):
#     A, B = x
#     result = 0
#     for i in range(len(r_points)):
#         result += (np.log(A*f1_vaccumm[i] + B*f2_vaccumm[i] + f_total[i]) - np.log(phi_solution[i]))**2
#     return result

# x0 = np.array([1, 1.])
# res = minimize(minimize_func, x0, method='nelder-mead')
# A, B = res.x
# print(f"A: {A}, B: {B}")

# # # Plot the solution
# plt.figure(figsize=(3.375,2.7)) 
# # plt.loglog(r_points, rho_solution, label=r"$\rho(r)$")
# # plt.loglog(r_points, phi_solution, label=r"$\Phi(r)$")
# # plt.loglog(r_points, A*alpha_solution,'--', label=r"$\alpha(r)$")
# # plt.loglog(r_points, B*beta_solution, '--',label=r"$\beta(r)$")
# # plt.plot(r_points, f1_solution, '--',label=r"$f_1(r)$")
# # plt.plot(r_points, f2_solution, '--',label=r"$f_2(r)$")
# # plt.plot(r_points, f3_solution, '--',label=r"$f_3(r)$")
# # plt.plot(r_points, f4_solution, '--',label=r"$f_4(r)$")
# plt.plot(r_points, f_total, '--',label=r"$f_{tot}(r)$")
# # plt.loglog(r_points, A*f1_vaccumm + B*f2_vaccumm + f_total, '--',label="analytic solution")
# # plt.xlim(1.e-3, r_final)
# # plt.ylim(1.e-10, 1.e9)
# plt.xscale('log')
# plt.xlabel(f"$r$")
# plt.title(r"$\phi(r)$ with NFW profile")
# plt.grid(True)
# plt.legend()
# plt.savefig("field_solution_test.pdf", bbox_inches="tight")


######################  
# Solve Poisson equation \nabla^2 \Phi = 4\pi G \rho (1+ C/2*phi)
######################

# Parameters
u_min = np.log(1.e-1)  # ln(r_min)  # Choose appropriately for your problem
u_max = np.log(1.e4)   # ln(r_max)
N = 100
du = (u_max - u_min) / (N - 1)

# Logarithmic grid
u = np.linspace(u_min, u_max, N)
r = np.exp(u)

# Charge density (example:  adjust for log scale)
rho = np.zeros(N)
for i in range(N):
    rho[i] = density_profile(r[i])

# Create the coefficient matrix (A) and RHS vector (b)
A = np.zeros((N, N))
b_rho = np.zeros(N)
b_phi = np.zeros(N)

def box_phi(r):
    return -1/r * quad(int_f2, 0, r)[0] - quad(int_f1, r, np.infty)[0]

def V2(r):
    def int_boxphi2(r):
        return r**2 * box_phi(r)**2
    return 1/r * (3./2.)* quad(int_boxphi2, 0, r)[0]

# Interior points
for i in range(1, N - 1):
    A[i, i-1] = 1 - du/2
    A[i, i] = -2
    A[i, i+1] = 1 + du/2
    b_rho[i] = r[i]**2 * rho[i] * du**2
    b_phi[i] = r[i]**2 * 3/2 * (box_phi(r[i]))**2 * du**2
    # b_phi[i] = r[i]**2 * 3/2*C* phi_interpolate(r[i])* rho[i] * du**2

# Boundary conditions (adapt as needed for your problem)
# Example: Neumann at u_min (dPhi/du = 0), Dirichlet at u_max (Phi=0)
A[0, 0] = -1
A[0, 1] = 1
b_rho[0] = 0
b_phi[0] = 0

A[N-1, N-1] = 1
b_rho[N-1] = 0
b_phi[N-1] = 0

# Solve
Phi_rho = np.linalg.solve(A, b_rho)
Phi_phi = np.linalg.solve(A, b_phi)

# Rotation curve v^2=r dPhi/dr = dPhi/du
v2_rho = np.gradient(Phi_rho, du)
v2_phi = np.gradient(Phi_phi, du)

# ######################
# # Plotting (log scale)
# ######################

# # # Plot Phi(r) 
# # plt.figure(figsize=(3.375,2.7)) 
# # plt.plot(r, Phi_rho, label=r"$\Phi_\rho(r)$")
# # plt.plot(r, Phi_phi, label=r"$\Phi_\phi(r)$")
# # plt.ylabel(r'$\Phi(r)$')
# # plt.title(r'$\Phi(r)$ with $\rho$ and $\phi$')
# # plt.xscale('log')
# # plt.xlabel(r'$r$')
# # plt.grid(True)
# # plt.legend()
# # plt.savefig("Phi_solution.pdf", bbox_inches="tight")

# Plot v^2(r)
plt.figure(figsize=(3.375,2.7)) 
plt.plot(r, v2_rho, label=r"$v^2_\rho(r)$")
plt.plot(r, v2_phi, label=r"$v^2_\phi(r)$, num")
plt.plot(r, [V2(r) for r in r], '--',label=r"$v^2_\phi(r)$, anal")
# plt.xscale('log')
plt.xlabel(r'$r$')
plt.ylabel(r'$v^2(r)=r\frac{d\Phi}{dr}$')
plt.xlim(1.e-4, 2)
plt.ylim(-0.1, 0.8)
plt.grid(True)
plt.legend()
plt.savefig("rotation_curve_test.pdf", bbox_inches="tight")