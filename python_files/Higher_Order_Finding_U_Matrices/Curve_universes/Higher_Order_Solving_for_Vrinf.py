# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:22:53 2021

@author: MRose
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

num_variables = 75  # number of pert variables, 75 for original code

kvalues = np.load('L70_kvalues.npy')
ABCmatrices = np.load('L70_ABCmatrices.npy')
DEFmatrices = np.load('L70_DEFmatrices.npy')
GHIvectors = np.load('L70_GHIvectors.npy')
X1matrices = np.load('L70_X1matrices.npy')
X2matrices = np.load('L70_X2matrices.npy')
recValues = np.load('L70_recValues.npy')

#first extract A and D matrices from results

Amatrices = []
Bmatrices = []
Cmatrices = []
Dmatrices = []
Ematrices = []
Fmatrices = []
GX3matrices = []
HX3matrices = []
IX3matrices = []

for i in range(len(kvalues)):
    
    ABC = ABCmatrices[i]
    DEF = DEFmatrices[i]
    GHI = GHIvectors[i]
    
    A = ABC[0:6, 0:6]
    B = ABC[6:8, 0:6]
    C = ABC[8:num_variables, 0:6]
    D = DEF[0:6, 0:2]
    E = DEF[6:8, 0:2]
    F = DEF[8:num_variables, 0:2]
    G = GHI[0:6]
    H = GHI[6:8]
    I = GHI[8:num_variables]
    
    #create zero arrays for G,H and I matrices
    GX3mat = np.zeros(shape=(6,4))
    GX3mat[:,2] = G
    HX3mat = np.zeros(shape=(2,4))
    HX3mat[:,2] = H
    IX3mat = np.zeros(shape=(num_variables-8, 4))
    IX3mat[:,2] = I
    
    Amatrices.append(A)
    Bmatrices.append(B)
    Cmatrices.append(C)
    Dmatrices.append(D)
    Ematrices.append(E)
    Fmatrices.append(F)
    GX3matrices.append(GX3mat)
    HX3matrices.append(HX3mat)
    IX3matrices.append(IX3mat)
    
#now set up matrix equations to solve for xinf
vrfcb = []
drfcb = []
dmfcb = []
vmdotfcb = []
yrecs = []

for j in range(1,len(kvalues)):
    
    A = Amatrices[j]
    D = Dmatrices[j]
    GX3 = GX3matrices[j]
    B = Bmatrices[j]
    E = Ematrices[j]
    HX3 = HX3matrices[j]
    C = Cmatrices[j]
    F = Fmatrices[j]
    IX3 = IX3matrices[j]
    X1 = X1matrices[j]
    X2 = X2matrices[j]
    recs = recValues[j]
    
    #calculate full matrix but then remove top two rows
    AX1 = A.reshape(6,6) @ X1.reshape(6,4)
    DX2 = D.reshape(6,2) @ X2.reshape(2,4)
    matrixog = AX1 + DX2 + GX3
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

#np.save('L70_vrfcb', vrfcb);

idxzeros = np.where(np.diff(np.sign(vrfcb)) != 0)[0]
allowedK = kvalues[idxzeros]
print(allowedK)
np.save('allowedK', allowedK)

plt.plot(kvalues[1:], vrfcb)
plt.plot(allowedK, np.zeros_like(allowedK), 'ro')
plt.savefig('L70_vrfcb.pdf')
