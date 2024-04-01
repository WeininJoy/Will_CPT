import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

start_time = time.time()

# functions for gram schmidt procedure

def inner_product(f, g):
    """
    Compute the inner product of two functions over the interval [a, b].
    """
    dt = T / (N_t-1)
    return np.dot(np.conj(f), g)*dt / T

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

    return orthogonal_functions, transformation_matrix

def normalization(orthogonal_functions, transformation_matrix):
    
    orthonormal_functions = [f / np.sqrt(inner_product(f, f)) for f in orthogonal_functions]
    for i in range(N):
        for j in range(N):
            transformation_matrix[i, j] = transformation_matrix[i, j] / np.sqrt(inner_product(orthogonal_functions[i], orthogonal_functions[i]))

    return orthonormal_functions, transformation_matrix


def compute_A_matrix(N, orthonormal_functions_1, orthonormal_functions_2):
    
    A_1 = np.zeros((N, N), dtype=complex)
    A_2 = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for m in range(N):
                A_1[i, j] += inner_product(orthonormal_functions_1[i], orthonormal_functions_2[m]) * inner_product(orthonormal_functions_2[m], orthonormal_functions_1[j])
                A_2[i, j] += inner_product(orthonormal_functions_2[i], orthonormal_functions_1[m]) * inner_product(orthonormal_functions_1[m], orthonormal_functions_2[j])
    # Find eigenvalues and eigenvectors
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(A_1)
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(A_2)

    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2

if __name__=="__main__":

    # Parameters
    N = 20
    N_t = 1000
    kappa = 1.5
    T = 1.
    eigenvalues_threshold = 1.e-1
    N_plot = 6

    t = np.linspace(0, T, N_t)

    # List of basis to apply the Gram-Schmidt process
    basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) for k in range(1, N+1)]
    basis_2 = [np.cos(n*np.pi/T * t) for n in range(1, N+1)]

    # Apply Gram-Schmidt process to get orthogonal basis
    orthogonal_functions_1, transformation_matrix_1 = gram_schmidt(basis_1)
    orthogonal_functions_2, transformation_matrix_2 = gram_schmidt(basis_2)

    # Normalize the orthogonal basis
    orthonormal_functions_1, transformation_matrix_1 = normalization(orthogonal_functions_1, transformation_matrix_1)
    orthonormal_functions_2, transformation_matrix_2 = normalization(orthogonal_functions_2, transformation_matrix_2)
    
    # Comute coefficient matrix A and its eigenvalues and eigenvectors
    eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(N, orthonormal_functions_1, orthonormal_functions_2)
    
    # Choose the egienvectors with eigenvalues close to 1
    eigenvalues_valid_1 = []
    eigenvectors_valid_1 = []
    eigenvalues_valid_2 = []
    eigenvectors_valid_2 = []
    for i in range(N):
        if np.abs(eigenvalues_1[i] - 1.) < eigenvalues_threshold:
            eigenvalues_valid_1.append(eigenvalues_1[i])
            eigenvectors_valid_1.append(eigenvectors_1[:, i])
        if np.abs(eigenvalues_2[i] - 1.) < eigenvalues_threshold:
            eigenvalues_valid_2.append(eigenvalues_2[i])
            eigenvectors_valid_2.append(eigenvectors_2[:, i])

    # Compute coefficients
    coefficients_1 = np.zeros((len(eigenvalues_valid_1), N), dtype=complex)
    coefficients_2 = np.zeros((len(eigenvalues_valid_2), N), dtype=complex)
    for i in range(len(eigenvalues_valid_1)):
        # coefficients_1[i, :] = np.dot(np.linalg.inv(R_1), eigenvectors_valid_1[i])
        coefficients_1[i, :] = np.dot(np.array(eigenvectors_valid_1[i]), transformation_matrix_1)
    for i in range(len(eigenvalues_valid_2)):
        # coefficients_2[i, :] = np.dot(np.linalg.inv(R_2), eigenvectors_valid_2[i])
        coefficients_2[i, :] = np.dot(np.array(eigenvectors_valid_2[i]), transformation_matrix_2)

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
        plt.plot(t, solution_1.real, 'b--', label="basis k")
        plt.plot(t, solution_2.real, 'r:', label="basis n")
        plt.ylabel(r"$\Phi(t)$")
        
    plt.legend()
    plt.xlabel(r"$t$")
    plt.savefig(f"eigenvalue_1d_N{N:d}_Nt{N_t:d}.pdf")

print("--- %s seconds ---" % (time.time() - start_time))
