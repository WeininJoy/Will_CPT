import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_list = [5,10,15,20,25,30]
N_t = 500
N_x = 500
kappa = 1.5
T = np.pi 
L = 2. * np.pi
eigenvalues_threshold = 1.e-1
N_plot = 6


eigenvalues_list = []
for N in N_list:
    eigenvalues_list.append(np.loadtxt(f"./data/linear_combination/all_eigenvalues1_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}.txt", dtype=np.complex_))

for i in range(len(eigenvalues_list[0])-1): 
    plt.plot(N_list, [np.abs(eigenvalues_list[j][i+1]) for j in range(len(N_list))],'.', label=f"eigenvalue{i+1:d}")

plt.xlabel("N")
plt.legend()
plt.savefig(f"eigenvalue_convergence_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}.pdf")