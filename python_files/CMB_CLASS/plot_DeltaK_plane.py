import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy import units as u
from astropy.constants import c
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

folder_path = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data_CurvedUniverse/"

mt_list = np.linspace(350, 500, 20)
kt_list = np.linspace(0, 1.8, 20)
z_rec_list = np.linspace(1040, 1100, 20)
DeltaK_arr = np.load(folder_path + 'DeltaK_arr.npy')  # 3D array: DeltaK(mt,kt,z_rec)

# Define integer wave vector k
for i in range(len(mt_list)):
    for j in range(len(kt_list)):
        for k in range(len(z_rec_list)):
            DeltaK_arr[i,j,k] = 1./ np.sqrt(kt_list[j]) * DeltaK_arr[i,j,k]


# make contour plot based on the data

plt.figure(figsize=(6, 4.5))
contour = plt.contourf(mt_list, z_rec_list, DeltaK_arr[:,10,:].T, levels=20, cmap='viridis')
plt.colorbar(contour)  # Add color bar
plt.xlabel(r"$\tilde m$", fontsize=18)
plt.ylabel(r"$z_{rec}$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(r"Contour Plot of $\Delta k_{int}$ ", fontsize=20)
plt.tight_layout()
plt.savefig("DeltaK_kt10.pdf")