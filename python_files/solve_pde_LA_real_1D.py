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
    Compute the inner product of two functions over the interval [a, b].
    """
    return np.dot(f, g)

def qr_decomposition(basis):
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix.T)

def compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2):
    
    M = np.dot(orthonormal_functions_1.T, orthonormal_functions_2)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M, M.T))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M.T, M))

    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2

def choose_eigenvalues(eigenvalues, eigenvectors, eigenvalues_threshold):
    eigenvalues_valid = []
    eigenvectors_valid = []
    for i in range(N):
        if np.abs(eigenvalues[i] - 1.) < eigenvalues_threshold:
            eigenvalues_valid.append(eigenvalues[i])
            eigenvectors_valid.append(eigenvectors[:, i])
    return eigenvalues_valid, eigenvectors_valid

def compute_coefficients(eigenvalues, eigenvectors, transformation_matrix):
    coefficients = np.zeros((len(eigenvalues), N))
    for i in range(len(eigenvalues)):
        coefficients[i, :] = np.dot(np.array(eigenvectors[i]), transformation_matrix)
    return coefficients

if __name__=="__main__":

    # Parameters
    N = 15
    N_t = 500
    kappa = 1.5
    T = np.pi
    eigenvalues_threshold = 1.e-1
    N_plot = 6

    t = np.linspace(0, T, N_t)

    # List of basis to apply the Gram-Schmidt process
    basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) for k in range(1, N+1)]
    basis_2 = [np.cos(n*np.pi/T * t) for n in range(2, N+2)]

    # Apply Gram-Schmidt process (by qr decomposition) to get orthonormal basis 
    orthonormal_functions_1, transformation_matrix_1 = qr_decomposition(np.array(basis_1).T)
    orthonormal_functions_2, transformation_matrix_2 = qr_decomposition(np.array(basis_2).T)
   
    # Comute coefficient matrix A and its eigenvalues and eigenvectors
    eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2)

    # Choose the egienvectors with eigenvalues close to 1
    eigenvalues_valid_1, eigenvectors_valid_1 = choose_eigenvalues(eigenvalues_1, eigenvectors_1, eigenvalues_threshold)
    eigenvalues_valid_2, eigenvectors_valid_2 = choose_eigenvalues(eigenvalues_2, eigenvectors_2, eigenvalues_threshold)

    # Compute coefficients
    coefficients_1 = compute_coefficients(eigenvalues_valid_1, eigenvectors_valid_1, transformation_matrix_1)
    coefficients_2 = compute_coefficients(eigenvalues_valid_2, eigenvectors_valid_2, transformation_matrix_2)

    # Plot the solution
    if N_plot > len(eigenvalues_valid_1):
        N_plot = len(eigenvalues_valid_1)
    for i in range(N_plot):
        plt.subplot(N_plot, 1, i+1)
        solution_1 = np.zeros_like(t, dtype=complex)
        solution_2 = np.zeros_like(t, dtype=complex)
        for j in range(N):
            solution_1 += coefficients_1[i, j]*basis_1[j]
            solution_2 += coefficients_2[i, j]*basis_2[j]
        if solution_1.real[0] < 0:
            solution_1 = -solution_1
        if solution_2.real[0] < 0:
            solution_2 = -solution_2
        plt.plot(t, solution_1.real, 'b--', label="basis k")
        plt.plot(t, solution_2.real, 'r:', label="basis n")
        plt.ylabel(r"$\Phi(t)$")
        
    plt.legend()
    plt.xlabel(r"$t$")
    plt.savefig(f"eigenvalue_1d_N{N:d}_Nt{N_t:d}_test.pdf")

    def all():
        qr_decomposition(np.array(basis_1).T)
        compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2)
        choose_eigenvalues(eigenvalues_1, eigenvectors_1, eigenvalues_threshold)
        compute_coefficients(eigenvalues_valid_1, eigenvectors_valid_1, transformation_matrix_1)
    
    %lprun -f all all()
    

print("--- %s seconds ---" % (time.time() - start_time))

# %%
