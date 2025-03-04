import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator

folder_path = "/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/generate_data/data/"

mt_list = np.linspace(350, 500, 20)
kt_list = np.linspace(0, 1.8, 20)
z_rec_list = np.linspace(1040, 1100, 20)
DeltaK_arr = np.zeros((len(mt_list), len(kt_list), len(z_rec_list)))  # 3D array: DeltaK(mt,kt,z_rec)

for i in range(len(mt_list)):
    for j in range(len(kt_list)):
        input_number = int(i*100 + j)
        try: 
            with open(folder_path + f'allowedK_{input_number}', 'rb') as f:
                allowedK_save = pickle.load(f)

            for k in range(len(z_rec_list)):
                allowedK = allowedK_save[k]
                deltaK_list = [allowedK[n+1] - allowedK[n] for n in range(len(allowedK)-1)]
                deltaK = sum(deltaK_list[len(deltaK_list)//2:-2]) / len(deltaK_list[len(deltaK_list)//2:-2])
                DeltaK_arr[i,j,k] = deltaK
        except: 
            print(f"file allowedK_{input_number} doen't exit. DeltaK set to be its neighbor.")
            DeltaK_arr[i,j,:] = DeltaK_arr[i-1,j-1,:]

# Get interpolator (based on the data)
interpolate_Deltak = RegularGridInterpolator((mt_list, kt_list, z_rec_list), DeltaK_arr)
DeltaK = interpolate_Deltak([442.2608532617835, 0.9534314943641817, 1060.4225])
print("DeltaK=",DeltaK)