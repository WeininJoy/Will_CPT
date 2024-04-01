import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import sys

start_time = time.time()
mp.set_start_method('fork')

# function for calculation

def multiplied_function(z, f):
    return lambda t,x: z * f(t,x)

def divide_function(f, z):
    return lambda t,x: f(t,x) / z

def sum_functions(f, g):
    return lambda t,x: f(t,x) + g(t,x)

def subtract_functions(f, g):
    return lambda t,x: f(t,x) - g(t,x)

def multiply_functions(f, g):
    return lambda t,x: f(t,x) * g(t,x)

"""
Compute the inner product of two functions over the interval [a, b].
"""
def inner_product(f, g, a, b, c, d):
    def y(t,x): return np.conj(f(t,x)) * g(t,x)
    # Perform the double integration
    result_real, error_real = dblquad(lambda t, x: y(t, x).real, c, d, a, b)
    result_imag, error_imag = dblquad(lambda t, x: y(t, x).imag, c, d, a, b)

    return result_real + 1j * result_imag

"""
Generate orthogonal functions by applying the Gram-Schmidt process to the given list of functions.
"""
### Function for computing transformation matrix from original functions to orthogonal functions (for parallel processing)
def compute_transformation_element(i, m, orthogonal_functions, functions, a, b, c, d, transformation_matrix, result_queue):
    for j in range(m, i):
        transformation_matrix[i, m] -= inner_product(orthogonal_functions[j], functions[i], a, b, c, d) / inner_product(orthogonal_functions[j], orthogonal_functions[j], a, b, c, d) * transformation_matrix[j, m]
    result_queue.put((i, m, transformation_matrix[i, m]))

### Main function for generating orthogonal functions and computing transformation matrix
def gram_schmidt(functions, a, b, c, d):
    # Generate orthogonal functions
    orthogonal_functions = [functions[0]]
    for i in range(1, len(functions)):
        new_function = functions[i]
        for j in range(i):
            functions_to_sbtract = multiplied_function( inner_product(orthogonal_functions[j], functions[i], a, b, c, d)/inner_product(orthogonal_functions[j], orthogonal_functions[j], a, b, c, d), orthogonal_functions[j] )
            new_function = subtract_functions( new_function, functions_to_sbtract)
        orthogonal_functions.append(new_function)
    
    # Calculate transformation matrix
    transformation_matrix = np.zeros((len(functions), len(functions)), dtype=complex)
    transformation_matrix[0,0] = 1
    for i in range(1, len(functions)):
        transformation_matrix[i,i] = 1  
        processes = []
        result_queue = mp.Queue()
        for m in range(i):
            args = (i, m, orthogonal_functions, functions, a, b, c, d, transformation_matrix, result_queue)
            p = mp.Process(target=compute_transformation_element, args=args)
            processes.append(p)
            p.start() 
        for p in processes:
            p.join()
        while not result_queue.empty():
            result = result_queue.get() # (i, m, transformation_matrix[i, m])
            transformation_matrix[result[0], result[1]] = result[2]

    return orthogonal_functions, transformation_matrix


"""
Generate orthonormal functions by normalizing the orthogonal functions.
"""
### Function for computing transformation matrix from orthogonal functions to orthonormal functions (for parallel processing)
def compute_orthonormal_matrix(i, j, transformation_matrix, orthogonal_functions, result_queue, a, b, c, d):
    transformation_matrix[i, j] = transformation_matrix[i, j] / np.sqrt(inner_product(orthogonal_functions[i], orthogonal_functions[i], a, b, c, d))
    result_queue.put((i, j, transformation_matrix[i, j]))

### Main function for generating orthonormal functions and computing transformation matrix
def normalization(orthogonal_functions, transformation_matrix, a, b, c, d):
    
    # Generate orthogonal functions
    orthonormal_functions = [divide_function(f, np.sqrt(inner_product(f, f, a, b, c, d))) for f in orthogonal_functions]
    
    # Calculate transformation matrix
    result_queue = mp.Queue()
    processes = []
    for i in range(N):
        for j in range(N):
            args = (i, j, transformation_matrix, orthogonal_functions, result_queue, a, b, c, d)
            p = mp.Process(target=compute_orthonormal_matrix, args=args)
            processes.append(p)
            p.start()
    for p in processes:
        p.join()
    while not result_queue.empty():
        i, j, value = result_queue.get()
        transformation_matrix[i, j] = value

    return orthonormal_functions, transformation_matrix

"""
Construct the coefficient matrix and find its eigenvalues and eigenvectors.
"""
def compute_A_1_element(i, j, N, orthonormal_functions_1, orthonormal_functions_2, a, b, c, d, result_queue):
    A_1_ij = 0
    for m in range(N):
        A_1_ij += inner_product(orthonormal_functions_1[i], orthonormal_functions_2[m], a, b, c, d) * inner_product(orthonormal_functions_2[m], orthonormal_functions_1[j], a, b, c, d)
    result_queue.put((i, j, A_1_ij))

def compute_A_2_element(i, j, N, orthonormal_functions_1, orthonormal_functions_2, a, b, c, d, result_queue):
    A_2_ij = 0
    for m in range(N):
        A_2_ij += inner_product(orthonormal_functions_2[i], orthonormal_functions_1[m], a, b, c, d) * inner_product(orthonormal_functions_1[m], orthonormal_functions_2[j], a, b, c, d)
    result_queue.put((i, j, A_2_ij))

def compute_A_matrix(N, orthonormal_functions_1, orthonormal_functions_2, a, b, c, d, num_processes=4):

    # Create queues to collect results
    result_queue_A_1 = mp.Queue()
    result_queue_A_2 = mp.Queue()
    # Split the computation of A_1 across multiple processes
    processes_A_1 = []
    for i in range(N):
        for j in range(N):
            p = mp.Process(target=compute_A_1_element, args=(i, j, N, orthonormal_functions_1, orthonormal_functions_2, a, b, c, d, result_queue_A_1))
            processes_A_1.append(p)
            p.start()
    # Split the computation of A_2 across multiple processes
    processes_A_2 = []
    for i in range(N):
        for j in range(N):
            p = mp.Process(target=compute_A_2_element, args=(i, j, N, orthonormal_functions_1, orthonormal_functions_2, a, b, c, d, result_queue_A_2))
            processes_A_2.append(p)
            p.start()

    # Wait for all processes computing A_1 to complete
    for p in processes_A_1:
        p.join()
    # Wait for all processes computing A_2 to complete
    for p in processes_A_2:
        p.join()

    # Retrieve results from the queue and update the matrices A_1 and A_2
    A_1 = np.zeros((N, N), dtype=complex)
    while not result_queue_A_1.empty():
        i, j, value = result_queue_A_1.get()
        A_1[i, j] = value

    A_2 = np.zeros((N, N), dtype=complex)
    while not result_queue_A_2.empty():
        i, j, value = result_queue_A_2.get()
        A_2[i, j] = value

    # Find eigenvalues and eigenvectors
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(A_1)
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(A_2)

    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2

"""
Find the eigenvectors with eigenvalues close to 1. -> coefficients for linear combination.
"""
def check_eigenvalues(eigenvalues, eigenvectors, threshold, result_queue, idx_start, idx_end):
    eigenvalues_valid = []
    eigenvectors_valid = []
    for i in range(idx_start, idx_end):
        if np.abs(eigenvalues[i].real - 1.) < threshold:
            eigenvalues_valid.append(eigenvalues[i])
            eigenvectors_valid.append(eigenvectors[:, i])
    result_queue.put((eigenvalues_valid, eigenvectors_valid))

def find_valid_eigenvectors(eigenvalues, eigenvectors, threshold, num_processes=4):
        # Create queues to collect results
    result_queue = mp.Queue()
    # Split the computation across multiple processes for eigenvalues_1
    processes = []
    chunk_size = N // num_processes
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else N
        p = mp.Process(target=check_eigenvalues, args=(eigenvalues, eigenvectors, threshold, result_queue, start_idx, end_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    eigenvalues_valid = []
    eigenvectors_valid = []
    while not result_queue.empty():
        eigenvalues, eigenvectors = result_queue.get()
        eigenvalues_valid.extend(eigenvalues)
        eigenvectors_valid.extend(eigenvectors)

    return eigenvalues_valid, eigenvectors_valid

"""
Compute coefficients corresponding to original functions by multiplying transformation matrix.
"""
def compute_coefficient_elements(m, N, eigenvector_valid, transformation_matrix, result_queue):
    c_list = eigenvector_valid
    coefficient_row = np.zeros(N, dtype=complex)
    for i in range(N):
        for j in range(N):
            coefficient_row[i] += c_list[j] * transformation_matrix[j, i]
    result_queue.put((m, coefficient_row))

def compute_coefficients(N, eigenvalues_valid, eigenvectors_valid, transformation_matrix):
    result_queue = mp.Queue()
    # Split the computation across multiple processes
    processes = []
    for m in range(len(eigenvalues_valid)):
        args = (m, N, eigenvectors_valid[m], transformation_matrix, result_queue)
        p = mp.Process(target=compute_coefficient_elements, args=args)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    coefficient_array = np.zeros((len(eigenvalues_valid), N), dtype=complex)
    while not result_queue.empty():
        m, coefficient_row = result_queue.get()
        coefficient_array[m, :] = coefficient_row
    return coefficient_array

if __name__=="__main__":

    # Parameters
    N = int(sys.argv[1])
    eigenvalues_threshold = float(sys.argv[2])
    num_processes = int(sys.argv[3])
    kappa = 1.5
    T = 1.

    # Interval t=[a, b], x=[c, d]
    a, b = 0, T
    c, d = 0, 2*np.pi
    x_plot = 0.5

    # List of basis to apply the Gram-Schmidt process
    basis_1 = [lambda t, x, k=k: np.exp(1j*k*x) * np.exp(1j * np.sqrt(k**2 + kappa**2) * t) for k in range(1, N+1)]
    basis_2 = [lambda t, x, n=n: np.exp(1j * n*np.pi/T * t) * np.exp(1j*np.sqrt((n*np.pi/T)**2 - kappa**2)*x) for n in range(1, N+1)]
    
    # Apply Gram-Schmidt process to get orthogonal basis
    orthogonal_functions_1, transformation_matrix_1 = gram_schmidt(basis_1, a, b, c, d)
    orthogonal_functions_2, transformation_matrix_2 = gram_schmidt(basis_2, a, b, c, d)

    # Normalize the orthogonal basis
    orthonormal_functions_1, transformation_matrix_1 = normalization(orthogonal_functions_1, transformation_matrix_1, a, b, c, d)
    orthonormal_functions_2, transformation_matrix_2 = normalization(orthogonal_functions_2, transformation_matrix_2, a, b, c, d)

    eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(N, orthonormal_functions_1, orthonormal_functions_2, a, b, c, d, num_processes)

    # Choose the egienvectors with eigenvalues close to 1
    eigenvalues_valid_1, eigenvectors_valid_1 = find_valid_eigenvectors(eigenvalues_1, eigenvectors_1, eigenvalues_threshold, num_processes)
    eigenvalues_valid_2, eigenvectors_valid_2 = find_valid_eigenvectors(eigenvalues_2, eigenvectors_2, eigenvalues_threshold, num_processes)
     
    # Calculate the coefficients of the solution in the original basis 1
    coefficient_array_1 = compute_coefficients(N, eigenvalues_valid_1, eigenvectors_valid_1, transformation_matrix_1)
    coefficient_array_2 = compute_coefficients(N, eigenvalues_valid_2, eigenvectors_valid_2, transformation_matrix_2)
    print("coefficient_array_1", coefficient_array_1)
    print("coefficient_array_2", coefficient_array_2)

    # Construct the solution by linear combination of the original basis 1
    for i in range(len(coefficient_array_1)):
        linear_combined_function_1 = lambda t, x: 0
        for j in range(N):
            linear_combined_function_1 = sum_functions(linear_combined_function_1, multiplied_function(coefficient_array_1[i, j], basis_1[j]))
        
        # Plot 1D linear combined solution
        t_list = np.linspace(a, b, 500)
        if i == 0: # plot the first solution with label
            plt.plot(t_list, [linear_combined_function_1(t, x_plot).real for t in t_list], 'b--', label="basis k")
        else:
            plt.plot(t_list, [linear_combined_function_1(t, x_plot).real for t in t_list], 'b--')

    # Construct the solution by linear combination of the original basis 2
    for i in range(len(coefficient_array_2)):
        linear_combined_function_2 = lambda t, x: 0
        for j in range(N):
            linear_combined_function_2 = sum_functions(linear_combined_function_2, multiplied_function(coefficient_array_2[i, j], basis_2[j]))
        
        # Plot 1D linear combined solution
        t_list = np.linspace(a, b, 500)
        if i == 0: # plot the first solution with label
            plt.plot(t_list, [linear_combined_function_2(t, x_plot).real for t in t_list], 'r:', label="basis n")
        else:
            plt.plot(t_list, [linear_combined_function_2(t, x_plot).real for t in t_list], 'r:')

    plt.legend()
    plt.savefig(f"eigenvalue_2d_N{N:d}_x={x_plot:.1f}.pdf")


print("--- %s seconds ---" % (time.time() - start_time))
