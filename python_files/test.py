import numpy as np
import matplotlib.pyplot as plt

def inner_product(f, g):
    """
    Compute the inner product of two functions f(t,x), g(t,x).
    """
    return np.sum(f*g) / (2*N)

def qr_decomposition(basis):
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix)

# Parameters
N = 10
N_t = 100
N_x = 100
kappa = 1.5
T = np.pi 
L = 2. * np.pi
eigenvalues_threshold = 1.e-1
N_plot = 6

# Interval t=[0, T], x=[0, L]
t_list = np.linspace(0, T, N_t)
x_list = np.linspace(0, L, N_x)
t, x = np.meshgrid(t_list, x_list)

# List of basis to apply the Gram-Schmidt process
basis_1 = []
basis_2 = []
for k in range(1, N+1):
    basis_1.append(np.cos(np.sqrt(k**2 + kappa**2) * t) * np.cos(k*x))
    basis_1.append(np.cos(np.sqrt(k**2 + kappa**2) * t) * np.sin(k*x))

if (np.pi/T)**2 - kappa**2 < 0: # make sure wave vector k is real
    start_n = round(np.sqrt(abs((np.pi/T)**2 - kappa**2))) + 1
else: start_n = 1
for n in range(start_n, N+start_n):
    basis_2.append(np.cos(n*np.pi/T * t) * np.cos(np.sqrt((n*np.pi/T)**2 - kappa**2)*x))
    basis_2.append(np.cos(n*np.pi/T * t) * np.sin(np.sqrt((n*np.pi/T)**2 - kappa**2)*x))

orthonormal_functions_1, transformation_matrix_1 = qr_decomposition(np.array(basis_1).T)
orthonormal_functions_2, transformation_matrix_2 = qr_decomposition(np.array(basis_2).T)


for i in range(N):
    plt.plot(t_list, orthonormal_functions_1.T[i, :, N_t//2], label=f"i={i}") 
plt.legend()
plt.savefig("orthonormal_functions_1_QR.pdf")  
