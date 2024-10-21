import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.optimize import root_scalar


plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

class Universe:
    def __init__(self, lam, rt, kt, mt):
        
        ### cosmological constants
        self.lam = lam   # dark energy
        self.rt = rt    # radiation
        self.kt = kt
        self.mt = mt

    def eta_tot(self):
        result = quad(lambda a: 1./self.a_dot(a), 0, np.inf)
        return result[0]
        

    #################
    # scale factor a(eta)
    #################
    
    def a_dot(self, a): 
        return np.sqrt( 1./3. * self.lam * ( self.rt + self.mt*a - 3*self.kt*a**2 + a**4 ) )
    
    def plot_a(self):
        def ode_a(y, eta):
            a = y
            a_dot = self.a_dot(a)
            return a_dot
        
        eta_tot = self.eta_tot()
        eta_list = np.linspace(0, eta_tot, 100)
        sol = odeint(ode_a, 0, eta_list)
        plt.plot(eta_list, np.log(sol[:, 0]))
        plt.axvline(x=eta_tot, ls='--')
        plt.show()

    def solve_a0(self, omega_lambda):
        rt, kt, mt = self.rt, self.kt, self.mt
        def f(a0):
            return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    def transform(self, omega_lambda):
        a0 = self.solve_a0(omega_lambda)
        omega_r = omega_lambda / a0**4
        omega_m = mt * omega_lambda**(1/4) * omega_r**(3/4)
        omega_kappa = -3* kt * np.sqrt(omega_lambda* omega_r)
        return a0, omega_lambda, omega_r, omega_m, omega_kappa


# OmegaLambda = 0.68 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
# OmegaM = 0.367 # in Metha's code, OmegaM = 0.321
# OmegaR = 9.21e-5
# OmegaK = 1 - OmegaLambda - OmegaM - OmegaR
# H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1

# lam = rt = 1
# mt = OmegaM / (OmegaLambda**(1./4.) * OmegaR**(3./4.))
# a0 = (OmegaLambda/OmegaR)**(1./4.)
# kt = 1./3. * ( mt/a0 + rt/a0**2 - (1./OmegaLambda -1)*a0**2 ) 
# print('kt=', kt)
# print('mt=', mt)

omega_lambda = 0.68
lam, rt, kt, mt = 1, 1, 0.5, 400
universe = Universe(lam, rt, kt, mt)
# universe.plot_a()
eta_tot_value = universe.eta_tot()
a0, OmegaLambda, OmegaR, OmegaM, OmegaK = universe.transform(omega_lambda)
print("eta_tot=", eta_tot_value)
print('dimensionless physical Delta k=', np.sqrt(3)*np.pi/eta_tot_value/a0)
print(" OmegaLambda, OmegaR, OmegaM, OmegaK=", OmegaLambda, OmegaR, OmegaM, OmegaK)

####################
# Calculate fcb time by Omega_i
####################

OmegaLambda = 0.68
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
lam = 1
rt = 1
mt = 400
kt = 0.5

def solve_a0(omega_lambda, rt, mt, kt):
    def f(a0):
        return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
    sol = root_scalar(f, bracket=[1, 1.e3])
    return sol.root

def transform(omega_lambda, rt, mt, kt):
    a0 = solve_a0(omega_lambda, rt, mt, kt)
    omega_r = omega_lambda / a0**4
    omega_m = mt * omega_lambda**(1/4) * omega_r**(3/4)
    omega_kappa = -3* kt * np.sqrt(omega_lambda* omega_r)
    return a0, omega_lambda, omega_r, omega_m, omega_kappa

a0, OmegaLambda, OmegaR, OmegaM, OmegaK = transform(OmegaLambda, rt, mt, kt)
print('a0, OmegaLambda, OmegaR, OmegaM, OmegaK=', a0, OmegaLambda, OmegaR, OmegaM, OmegaK)


def a_dot(a): 
    return a**2 * H0*np.sqrt( OmegaR * (a0/a)**4 + OmegaM*(a0/a)**3 + OmegaK*(a0/a)**2 + OmegaLambda )

def eta_tot():
    result = quad(lambda a: 1./a_dot(a), 0, np.inf)
    return result[0]

eta_tot_value = float(eta_tot())
print('eta_tot_a=', eta_tot_value)
print('dimensionless conformal Delta k = ', np.sqrt(3)*np.pi/eta_tot_value)  # use Deaglan's definition of Delta k in [2104.01938]
print('dimensionless physical Delta k = ', np.sqrt(3)*np.pi/eta_tot_value/a0)  # use Deaglan's definition of Delta k in [2104.01938]

from astropy import units as u
from astropy.constants import c

H0 = 66.86 * u.km/u.s/u.Mpc 

Lambda = OmegaLambda * 3 * H0**2 / c**2 
Lambda = Lambda.si.to(u.Mpc**-2).value

print('Lambda=', Lambda)
print('dimensional Delta k in Mpc^-1 = ', np.sqrt(3)*np.pi/eta_tot_value/a0 * np.sqrt(Lambda))