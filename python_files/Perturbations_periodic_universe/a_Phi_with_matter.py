import numpy as np
import cmath
from mpmath import ellipfun
import matplotlib.pyplot as plt
import cplot

####################
# Define constants #
####################

lam = 1.
k = 1.
m = .5
x = 1
r = (x* m**2)/k
ks = 10. # dimensionless wave number = wave number / sqrt(lam)

####################
# a(eta) solution #
####################

def gp(x):
    return ((1 + 8*x) + (1 + 8/3*x)* cmath.sqrt(1 + 32/3*x))/(1 + 3*cmath.sqrt(1 + 32/3*x))

def gm(x):
    return ((1 + 8*x) - (1 + 8/3*x)* cmath.sqrt(1 + 32/3*x))/(1 - 3*cmath.sqrt(1 + 32/3*x))

kt = k/lam
mt = m/lam
rt = r/lam

def f(a):
    return a**4 - 3*kt*a**2 + mt*a + rt

def V(a):
    return -lam*f(a)

Pp = -(4/3)* rt - kt**2
Qq = -((mt**2) /2) + kt**3 - 4* kt*rt
xh = (-Qq + cmath.sqrt(Qq**2 + Pp**3))**(1/3)
z0 = 2 *kt + xh - Pp/xh
e0 = z0**(1/2)
app = 1/2 *((+e0) + cmath.sqrt(-(+e0)**2 - 2* mt/(+e0) + 6* kt))
apm = 1/2 *((+e0) - cmath.sqrt(-(+e0)**2 - 2* mt/(+e0) + 6* kt))
amp = 1/2 *((-e0) + cmath.sqrt(-(-e0)**2 - 2* mt/(-e0) + 6* kt))
amm = 1/2 *((-e0) - cmath.sqrt(-(-e0)**2 - 2* mt/(-e0) + 6* kt))

a1 = app
a2 = apm
a3 = amp
a4 = amm

mm = ((a2 - a3)* (a1 - a4))/((a1 - a3)* (a2 - a4))
rho = 1/2 *cmath.sqrt(lam/3 *(a1 - a3)* (a2 - a4))


def a(eta):
    np_ellipfun = np.frompyfunc(ellipfun, 3, 1)
    Jacobi_sn = np_ellipfun('sn', rho* eta, mm)
    result = (a2*(a3 - a1) - a1*(a3 - a2)* Jacobi_sn**2)/((a3 - a1) - (a3 - a2)* Jacobi_sn**2)
    return result


# ## Plot a(eta) solution

# N = 100
# lim = 10
# x = np.linspace(-lim,lim,N)
# y = np.linspace(-lim,lim,N)

# a_mesh = np.zeros((N,N), dtype=complex)

# for i in range(N):
#     for j in range(N):
#         a_mesh[i,j] = a(x[i] + 1j*y[j])

# plt.contourf(x, y, a_mesh.real, levels=20, cmap='viridis')
# plt.colorbar()
# plt.xlabel('Re')
# plt.ylabel('Im')
# plt.title(r'$a(\eta)$ with mass')
# plt.savefig("a_eta_with_mass.pdf")

##########################
# Phi(a) solution with mass #
##########################

from scipy.integrate import solve_ivp

# define ODEs
def odes(a, y):
    phi = y[0]
    dphidt = y[1]
    d2phidt = ( - (4/3*rt/a + 4*mt/3 - 5*kt*a + 2*a**3) * dphidt - 1/3*(ks**2 + 4*a**2) * phi ) / (rt/3 + mt*a/3 - kt*a**2 + a**4/3)
    return [dphidt, d2phidt]

y0 = [1, 0]
a_list = np.linspace(1.e-3, 1, 1000)
sol_inf = solve_ivp(odes, [1.e-3, 10], y0, t_eval = a_list, method='RK45')
phi_sol = sol_inf.y[0]

## Plot Phi(eta) solution
plt.plot(sol_inf.t, phi_sol)
plt.xlabel(r'$a$')
plt.ylabel(r'$\Phi(a)$')
plt.title(r'$\Phi(a)$ with mass')
plt.savefig("Phi_with_mass.pdf")