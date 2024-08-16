# %%
%load_ext line_profiler
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA
import time

start_time = time.time()

# functions for gram schmidt procedure

def inner_product(f, g):
    """
    Compute the inner product of two functions f(t,x), g(t,x).
    """
    return np.sum(np.vdot(f, g)) 

def gram_schmidt(functions):
    # Generate orthogonal functions
    orthogonal_functions = [functions[0]]
    for i in range(1, len(functions)):
        new_function = functions[i]
        for j in range(i):
            functions_to_sbtract = inner_product(orthogonal_functions[j], functions[i])/inner_product(orthogonal_functions[j], orthogonal_functions[j]) * orthogonal_functions[j]
            new_function = new_function - functions_to_sbtract
        orthogonal_functions.append(new_function)
    
    # calculate transformation matrix
    transformation_matrix = np.zeros((len(functions), len(functions)), dtype=complex)
    transformation_matrix[0,0] = 1
    for i in range(1, len(functions)):
        transformation_matrix[i,i] = 1  
        for m in range(i):
            for j in range(m, i):
                transformation_matrix[i, m] -= inner_product(orthogonal_functions[j], functions[i]) / inner_product(orthogonal_functions[j], orthogonal_functions[j]) * transformation_matrix[j, m]

    return orthogonal_functions, transformation_matrix.real


def normalization(orthogonal_functions, transformation_matrix):
    
    orthonormal_functions = [f / np.sqrt(inner_product(f, f)) for f in orthogonal_functions]
    for i in range(N):
        for j in range(N):
            transformation_matrix[i, j] = transformation_matrix[i, j] / np.sqrt(inner_product(orthogonal_functions[i], orthogonal_functions[i]))

    return orthonormal_functions, transformation_matrix


def compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2):
    
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            M[i, j] = inner_product(orthonormal_functions_1[i], orthonormal_functions_2[j])
    M = np.asmatrix(M)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M, M.getH() ))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M.getH(), M))

    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2

def choose_eigenvalues(eigenvalues, eigenvalues_threshold):
    eigen_valid_idx = []
    for i in range(N):
        if np.abs(eigenvalues[i] - 1.) < eigenvalues_threshold:
            eigen_valid_idx.append(i)
    return eigen_valid_idx

# Parameters
N = 15
N_t = 500
N_x = 500
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
basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j*k*x) for k in range(1, N+1)]
if (np.pi/T)**2 - kappa**2 < 0: # make sure wave vector k is real
    start_n = round(np.sqrt(abs((np.pi/T)**2 - kappa**2))) + 1
else: start_n = 1
basis_2 = [np.cos(n*np.pi/T * t) * np.exp(1j*np.sqrt((n*np.pi/T)**2 - kappa**2)*x) for n in range(start_n, N+start_n)]

# Apply Gram-Schmidt process to get orthogonal basis
orthogonal_functions_1, transformation_matrix_1 = gram_schmidt(basis_1)
orthogonal_functions_2, transformation_matrix_2 = gram_schmidt(basis_2)

# Normalize the orthogonal basis
orthonormal_functions_1, transformation_matrix_1 = normalization(orthogonal_functions_1, transformation_matrix_1)
orthonormal_functions_2, transformation_matrix_2 = normalization(orthogonal_functions_2, transformation_matrix_2)

# Comute coefficient matrix A and its eigenvalues and eigenvectors
eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2)

# Choose the egienvectors with eigenvalues close to 1
eigen_valid_idx_1 = choose_eigenvalues(eigenvalues_1, eigenvalues_threshold)
eigenvalues_valid_1 = eigenvalues_1[eigen_valid_idx_1]
eigenvectors_valid_1 = eigenvectors_1[:, eigen_valid_idx_1]

eigen_valid_idx_2 = choose_eigenvalues(eigenvalues_2, eigenvalues_threshold)
eigenvalues_valid_2 = eigenvalues_2[eigen_valid_idx_2]
eigenvectors_valid_2 = eigenvectors_2[:, eigen_valid_idx_2]

# Compute coefficients
coefficients_1 = np.dot(eigenvectors_valid_1.T, transformation_matrix_1)
coefficients_2 = np.dot(eigenvectors_valid_2.T, transformation_matrix_2)

# Check the efficiency of the functions
def functions():
    gram_schmidt(basis_1)
    normalization(orthogonal_functions_1, transformation_matrix_1)
    compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2)
   
%lprun -f functions functions()

# %%

###############
# Plot the solutions
###############

# Sort the solutions
max_indices = np.argmax(np.array(coefficients_1), axis=1) # Find the index of the maximum value in each row (coefficients)
sorted_index = np.argsort(max_indices)  # Sort the coefficients based on the dominant k mode (from small to large)

# Construct the solutions
solution_1_list = []
solution_2_list = []
for i in range(len(eigenvalues_valid_1)):
    solution_1 = np.zeros_like(basis_1[0])
    solution_2 = np.zeros_like(basis_2[0])
    for j in range(N):
        solution_1 += coefficients_1[i, j]*basis_1[j]
        solution_2 += coefficients_2[i, j]*basis_2[j]  
    solution_1_list.append(solution_1)
    solution_2_list.append(solution_2)

diff_array = np.zeros((len(eigenvalues_valid_1), len(eigenvalues_valid_2)))
small_diff_index = []
for i in range(len(eigenvalues_valid_1)):
    solution_1 = solution_1_list[i]
    for j in range(len(eigenvalues_valid_2)):
        solution_2 = solution_2_list[j]
        diff_array[i, j] += np.linalg.norm(solution_1 - solution_2) / np.linalg.norm(solution_1)
    small_diff_index.append(np.argmin(diff_array[i, :]))

# Plot the solution Phi(t) at x=L/2
fig, axs = plt.subplots(len(eigenvalues_valid_1), figsize=(6,0.8*len(eigenvalues_valid_1)))
fig.suptitle(r"$\Phi(t,x=L/2)$")
for i in range(len(eigenvalues_valid_1)):
    idx = sorted_index[i]
    solution_1 = solution_1_list[idx]
    solution_2 = solution_2_list[small_diff_index[idx]]
    axs[i].plot(t_list, solution_1[N_x//2, :].real, color='r', label=r"basis: $\phi$")
    axs[i].plot(t_list, solution_2[N_x//2, :].real, color='g', linestyle='dashed', label=r"basis: $\tilde\phi$")
plt.legend()
plt.xlabel(r"$t$")
plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_xpi_real.pdf")


# # Plot the solution Phi(x) at t=T/2
# if N_plot > len(eigenvalues_valid_1):
#     N_plot = len(eigenvalues_valid_1)
# fig, axs = plt.subplots(N_plot)
# fig.suptitle(r"$\Phi(x,t=T/2)$")
# for i in range(N_plot):
#     idx = sorted_index[i]
#     solution_1 = solution_1_list[idx]
#     solution_2 = solution_2_list[small_diff_index[idx]]
#     axs[i].plot(t_list, solution_1[:, N_t//2].real, color='r', label=r"basis: $\phi$")
#     axs[i].plot(t_list, solution_2[:, N_t//2].real, color='g', linestyle='dashed', label=r"basis: $\tilde\phi$")  
# plt.legend()
# plt.xlabel(r"$x$")
# plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_t0.5pi.pdf")


# ## 3D plot of the solution Phi(t,x) 
# # Contour plot
# fig, axs = plt.subplots(len(eigenvalues_valid_1)//3, 3, figsize=(7,7))
# cmap = plt.get_cmap('RdGy')
# for i in range(len(eigenvalues_valid_1)//3):
#     for j in range(3):
#         idx = sorted_index[3*i+j]
#         solution = solution_1_list[idx].real
#         axs[i, j].contourf(x, t, solution, 20, cmap=cmap)
#         axs[i, j].set_title(f"dominate n: {max_indices[idx]:d}", size=8)
# for ax in axs.flat:
#     ax.set_xlabel(r'$x$', fontsize=8)
#     ax.set_ylabel(r'$t$', fontsize=8)
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# fig.tight_layout()
# plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_contour_sequences.pdf")

# # Cylinder Plot 
# y = np.cos(x)
# z = np.sin(x)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# ax.set_axis_off()

# facecolors = cmap((solution-solution.min())/(solution.max()-solution.min()))
# plot = ax.plot_surface(y, z, t, rstride=1, cstride=1, facecolors=facecolors, linewidth=0, antialiased=False, alpha=0.9)
# plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_cylinder.pdf")



print("--- %s seconds ---" % (time.time() - start_time))

# %%
