import numpy as np
import pandas as pd
import corner 
import matplotlib.pyplot as plt

# MCMC chain samples
samples = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE_1.txt')

# load the column names for the samples
column_names_raw = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE.paramnames', dtype=str, usecols=[0])
column_names = [x.replace("b'",'').replace("'",'') for x in column_names_raw]

# make a data frame with column names and samples
samples1 = pd.DataFrame(samples[:,2:], columns=column_names) # first two columns are not important


# define which parameters to use
use_params = ['H0*', 'omegak']


sigma1 = 1. - np.exp(-(1./1.)**2/2.)
sigma2 = 1. - np.exp(-(2./1.)**2/2.)
# figure = corner.corner(samples1[use_params])
figure = corner.corner(samples1[use_params], bins=20, levels=(sigma1, sigma2), color='r')

plt.show()