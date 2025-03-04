########### PPS For FIC 1st Order Model ###############

#Dependencies
import numpy as np
import scipy.special as scp
import sys
import math as mt

#Get Command Line PPS Parameters
nv = float(sys.argv[1])
As =  float(sys.argv[2])
Dk = float(sys.argv[3])


#Pivot Scale in Mpc^-1
k0 = 0.05

#Range Parameters
klow_p = -7.0
khigh_p = 2.0
khigh = 68.0

#Accuracy Paramters
LCDM_knum = 10000
FIC_points_per_osc = 50
FIC_knum = mt.ceil(khigh*FIC_points_per_osc/Dk)



#Full FIC regime

#k values to calculate PPS at
ks = np.concatenate((np.linspace(np.power(10,klow_p),Dk*0.9995,mt.ceil(Dk/np.power(10,klow_p))),  np.linspace(Dk,khigh,FIC_knum)))

#Calcaulate Power Spectrum
kDks = (np.pi*ks)/(3.0*Dk)
prefac = ((np.pi**2.0)/3.0)*As*(k0/Dk)
Ps = prefac * np.power(ks/k0,nv)*(scp.j0(kDks)*np.sin(2*kDks) + scp.j1(kDks)*(np.cos(2*kDks) - np.sin(2*kDks)/(2*kDks)))**2

for k, P in zip(ks, Ps):
    print( "{:.8e}".format(k) + "\t" + "{:.5e}".format(P))


