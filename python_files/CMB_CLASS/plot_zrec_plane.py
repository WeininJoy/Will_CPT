import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy import units as u
from astropy.constants import c
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

param_ranges = [[350, 500], [0,1.8], [0.15, 0.17], [0.5, 0.75]]  # mt, kt, $omega_b/(omega_cdm + omega_b)$, $h$
grid_num = 10
mt_list = np.linspace(param_ranges[0][0],param_ranges[0][1],grid_num)
kt_list = np.linspace(param_ranges[1][0],param_ranges[1][1],grid_num)
omegab_list = np.linspace(param_ranges[2][0],param_ranges[2][1],grid_num)
h_list = np.linspace(param_ranges[3][0],param_ranges[3][1],grid_num)
zrec_arr = np.load('zrec_params.npy')

# make contour plot based on the data

plt.figure(figsize=(6, 4.5))
contour = plt.contourf( kt_list, omegab_list, zrec_arr[5,:,:,5].T, levels=20, cmap='viridis')
plt.colorbar(contour)  # Add color bar
plt.xlabel(r"$\tilde \kappa$", fontsize=18)
plt.ylabel(r"$\Omega_b/\Omega_m$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(r"Contour Plot of $z_{rec}$ ", fontsize=20)
plt.tight_layout()
plt.savefig("zrec_mt5_h5.pdf")