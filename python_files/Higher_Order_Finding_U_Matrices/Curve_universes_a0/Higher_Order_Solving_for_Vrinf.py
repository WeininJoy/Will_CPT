# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:22:53 2021

@author: MRose
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import matplotlib as mpl
import sys

input_number = int(sys.argv[1])


num_variables = 75  # number of pert variables, 75 for original code

kvalues = np.load(f'L70_kvalues_{input_number}.npy')
ABCmatrices = np.load(f'L70_ABCmatrices_{input_number}.npy')
DEFmatrices = np.load(f'L70_DEFmatrices_{input_number}.npy')
# GHIvectors = np.load(f'L70_GHIvectors_{input_number}.npy')
X1matrices = np.load(f'L70_X1matrices_{input_number}.npy')
X2matrices = np.load(f'L70_X2matrices_{input_number}.npy')
recValues = np.load(f'L70_recValues_{input_number}.npy')

#first extract A and D matrices from results

Amatrices = []
Bmatrices = []
Cmatrices = []
Dmatrices = []
Ematrices = []
Fmatrices = []
# GX3matrices = []
# HX3matrices = []
# IX3matrices = []

for i in range(len(kvalues)):
    
    ABC = ABCmatrices[i]
    DEF = DEFmatrices[i]
    # GHI = GHIvectors[i]
    
    A = ABC[0:6, 0:6]
    B = ABC[6:8, 0:6]
    C = ABC[8:num_variables, 0:6]
    D = DEF[0:6, 0:2]
    E = DEF[6:8, 0:2]
    F = DEF[8:num_variables, 0:2]
    # G = GHI[0:6]
    # H = GHI[6:8]
    # I = GHI[8:num_variables]
    
    # #create zero arrays for G,H and I matrices
    # GX3mat = np.zeros(shape=(6,4))
    # GX3mat[:,2] = G
    # HX3mat = np.zeros(shape=(2,4))
    # HX3mat[:,2] = H
    # IX3mat = np.zeros(shape=(num_variables-8, 4))
    # IX3mat[:,2] = I
    
    Amatrices.append(A)
    Bmatrices.append(B)
    Cmatrices.append(C)
    Dmatrices.append(D)
    Ematrices.append(E)
    Fmatrices.append(F)
    # GX3matrices.append(GX3mat)
    # HX3matrices.append(HX3mat)
    # IX3matrices.append(IX3mat)
    
#now set up matrix equations to solve for xinf
vrfcb = []
drfcb = []
dmfcb = []
vmdotfcb = []
yrecs = []

for j in range(1,len(kvalues)):
    
    A = Amatrices[j]
    D = Dmatrices[j]
    # GX3 = GX3matrices[j]
    B = Bmatrices[j]
    E = Ematrices[j]
    # HX3 = HX3matrices[j]
    C = Cmatrices[j]
    F = Fmatrices[j]
    # IX3 = IX3matrices[j]
    X1 = X1matrices[j]
    X2 = X2matrices[j]
    recs = recValues[j]
    
    #calculate full matrix but then remove top two rows
    AX1 = A.reshape(6,6) @ X1.reshape(6,4)
    DX2 = D.reshape(6,2) @ X2.reshape(2,4)
    matrixog = AX1 + DX2 # + GX3
    matrix = matrixog[[2,3,4,5], :]
    
    xrecs = [recs[2], recs[3], recs[4], recs[5]]
    
    xinf = np.linalg.solve(matrix, xrecs)
    vrfcb.append(xinf[2])
    drfcb.append(xinf[0])
    dmfcb.append(xinf[1])
    vmdotfcb.append(xinf[3])
    
    #mat1 = B.reshape(2,6) @ X1.reshape(6,4);
    #mat2 = E.reshape(2,2) @ X2.reshape(2,4);
    #yrec = (mat1 + mat2).reshape(2,4) @ xinf.reshape(4,1);
    
    #yrecs.append(yrec);

np.save(f'L70_vrfcb_{input_number}', vrfcb);

#set cosmological parameters
# OmegaLambda = 0.68
# H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
# lam = 1
# rt = 1
# mt_list = np.linspace(300, 450, 10)
# kt_list = np.linspace(1.e-4, 1, 10)
# mt = mt_list[input_number//10]
# kt = kt_list[input_number%10]

OmegaLambda = 0.679 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.321 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 0
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
s0 = 1

# lam = rt = 1
# a0 = (OmegaLambda/OmegaR)**(1./4.)
# s0 = 1/a0
# mt = OmegaM / (OmegaLambda**(1./4.) * OmegaR**(3./4.))
# kt = - OmegaK / np.sqrt(OmegaLambda* OmegaR) / 3

# calculate present scale factor a0 and energy densities
def solve_a0(omega_lambda, rt, mt, kt):
    def f(a0):
        return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt
    sol = root_scalar(f, bracket=[1, 1.e3])
    return sol.root

def transform(omega_lambda, rt, mt, kt):
    a0 = solve_a0(omega_lambda, rt, mt, kt)
    s0 = 1/a0
    omega_r = omega_lambda / a0**4
    omega_m = mt * omega_lambda**(1/4) * omega_r**(3/4)
    omega_kappa = -3* kt * np.sqrt(omega_lambda* omega_r)
    return s0, omega_lambda, omega_r, omega_m, omega_kappa

# s0, OmegaLambda, OmegaR, OmegaM, OmegaK = transform(OmegaLambda, rt, mt, kt)
print('s0, OmegaLambda, OmegaR, OmegaM, OmegaK=', s0, OmegaLambda, OmegaR, OmegaM, OmegaK)

idxzeros = np.where(np.diff(np.sign(vrfcb)) != 0)[0]
allowedK = []
for idx in idxzeros:
    k1 = kvalues[idx]
    k2 = kvalues[idx+1]
    vrfcb1 = vrfcb[idx]
    vrfcb2 = vrfcb[idx+1]
    allowK = k1 - vrfcb1 * (k2 - k1) / (vrfcb2 - vrfcb1)
    allowedK.append(allowK)
np.save(f'allowedK_{input_number}', allowedK)
deltaK_list = [allowedK[i+1] - allowedK[i] for i in range(len(allowedK)-1)]
print('Delta K list = ', deltaK_list)
deltaK = sum(deltaK_list[len(deltaK_list)//2:-2]) / len(deltaK_list[len(deltaK_list)//2:-2])
print('Delta K = ', deltaK)

# from astropy import units as u
# from astropy.constants import c

# H0 = 66.88 * u.km/u.s/u.Mpc  # 66.86 km/s/Mpc

# Lambda = OmegaLambda * 3 * H0**2 / c**2 
# Lambda = Lambda.si.to(u.Mpc**-2).value

# print('dimensional Delta k in Mpc^-1 = ', deltaK * s0 * np.sqrt(Lambda))

plt.plot(kvalues[1:] , vrfcb)
plt.plot(allowedK, np.zeros_like(allowedK), 'ro')
plt.ylim(-2, 4)
plt.xlim(0, 20)
plt.savefig(f'L70_vrfcb_{input_number}.pdf')
