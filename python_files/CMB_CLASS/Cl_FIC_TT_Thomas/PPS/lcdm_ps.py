########### PPS For BD 1st Order Model ###############

#Dependencies
import numpy as np
import scipy.special as scp
import sys

#Get Arguments
ns = float(sys.argv[1])
As =  float(sys.argv[2])

#Set Parameters for generating spectrum
k0 = 0.05
klow = -7.0
khigh = 2.0
knum = 100000
ks = np.logspace(klow,khigh,knum)

#Power spectrum functions

def pps(k):
    return As*np.power(k/k0,ns-1) 

#Print out in required format
Ps = pps(ks)

for k, P in zip(ks, Ps):
    print( "{:.8e}".format(k) + "\t" + "{:.6e}".format(P))


