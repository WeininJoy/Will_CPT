import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

start_time = time.time()

# function calculation

def multiplied_function(z, f):
    return lambda t: z * f(t)

def divide_function(f, z):
    return lambda t: f(t) / z

def sum_functions(f, g):
    return lambda t: f(t) + g(t)

def subtract_functions(f, g):
    return lambda t: f(t) - g(t)

def multiply_functions(f, g):
    return lambda t: f(t) * g(t)


# functions for gram schmidt procedure

def inner_product(f, g, a, b):
    """
    Compute the inner product of two functions over the interval [a, b].
    """
    def y(t): return np.conj(f(t)) * g(t)
    result, error = quad(lambda t: y(t), a, b)
    return result

def gram_schmidt(functions, a, b):
    """
    Apply the Gram-Schmidt process to a list of functions over the interval [a, b].
    """
    # define list to store orthogonal functions
    orthogonal_functions = [functions[0]]
    # define array to store transformation_matrix
    transformation_matrix = np.zeros((len(functions), len(functions)), dtype=complex)
    transformation_matrix[0,0] = 1
    for i in range(1, len(functions)):
        # generate orthogonal functions
        new_function = functions[i]
        for j in range(i):
            new_function = subtract_functions( new_function, multiplied_function( inner_product(orthogonal_functions[j], functions[i], a, b)/inner_product(orthogonal_functions[j], orthogonal_functions[j], a, b), orthogonal_functions[j] ) )
        orthogonal_functions.append(new_function)
        # calculate transformation matrix
        transformation_matrix[i,i] = 1
        for m in range(i):
            for j in range(m, i):
                transformation_matrix[i,m] -= inner_product(orthogonal_functions[j], functions[i], a, b)/inner_product(orthogonal_functions[j], orthogonal_functions[j], a, b) * transformation_matrix[j,m]
    return orthogonal_functions, transformation_matrix

# Parameters
N = 6
kappa = 1.5
T = 1.
n = 3

# Interval t=[a, b], x=[c, d]
a, b = 0, T
c, d = 0, 2*np.pi


# List of basis to apply the Gram-Schmidt process
basis_1 = [lambda t, k=k: np.exp(1j * np.sqrt(k**2 + kappa**2) * t) for k in range(N)]
basis_2 = [lambda t, n=n: np.exp(1j * n*np.pi/T * t) for n in range(N)]

# Apply Gram-Schmidt process
orthogonal_functions_1, transformation_matrix_1 = gram_schmidt(basis_1, a, b)
orthogonal_functions_2, transformation_matrix_2 = gram_schmidt(basis_2, a, b)

# Normalize the orthogonal basis
orthonormal_functions_1 = [divide_function(f, np.sqrt(inner_product(f, f, a, b))) for f in orthogonal_functions_1]
orthonormal_functions_2 = [divide_function(f, np.sqrt(inner_product(f, f, a, b))) for f in orthogonal_functions_2]
for i in range(N):
    for j in range(N):
        transformation_matrix_1[i,j] = transformation_matrix_1[i,j] / np.sqrt(inner_product(orthogonal_functions_1[i], orthogonal_functions_1[i], a, b))
        transformation_matrix_2[i,j] = transformation_matrix_2[i,j] / np.sqrt(inner_product(orthogonal_functions_2[i], orthogonal_functions_2[i], a, b))

# Find coefficients c_k using fixed-point iteration
def fixed_point_iteration(calculate_coefficients, ck_list_0, cn_list_0, tol=1.e-3, max_iter=200):
    """
    Perform fixed-point iteration to find a solution to g(x) = x.

    Parameters:
        calculate_coefficients (function): The function for which we want to find coefficients.
        ck_list_0, cn_list_0 (list): Initial guess for the coefficients.
        tol (float): Tolerance for convergence (default: 1e-3).
        max_iter (int): Maximum number of iterations (default: 200).

    Returns:
        float: Approximation of the fixed point.
    """
    ck_list = ck_list_0
    cn_list = cn_list_0
    for i in range(max_iter):
        ck_list_new, cn_list_new = calculate_coefficients(ck_list, cn_list)
        diff = 0.0
        for j in range(N):
            diff += abs(ck_list_new[j] - ck_list[j]) + abs(cn_list_new[j] - cn_list[j])
        if diff < tol:
            return ck_list_new, cn_list_new
        ck_list = ck_list_new
        cn_list = cn_list_new
    raise RuntimeError("Fixed-point iteration did not converge.")

# Define the function for calculating the next iteration of c_k
def calculate_coefficients(ck_list, cn_list):
    ck_list_new = np.zeros(N, dtype=complex)
    cn_list_new = np.zeros(N, dtype=complex)
    # for i in range(N):
    #     for j in range(N):
    #         ck_list_new[i] += cn_list[j] * inner_product(orthonormal_functions_1[i], orthonormal_functions_2[j], a, b)
    #         cn_list_new[i] += ck_list[j] * inner_product(orthonormal_functions_2[i], orthonormal_functions_1[j], a, b)
    for i in range(N):
        for j in range(N):
            for m in range(N):
                ck_list_new[i] += ck_list[j] * inner_product(orthonormal_functions_1[i], orthonormal_functions_2[m], a, b) * inner_product(orthonormal_functions_2[m], orthonormal_functions_1[j], a, b)
                cn_list_new[i] += cn_list[j] * inner_product(orthonormal_functions_2[i], orthonormal_functions_1[m], a, b) * inner_product(orthonormal_functions_1[m], orthonormal_functions_2[j], a, b)
    return ck_list_new, cn_list_new

# Initial guess (normal distribution with mean 3 and standard deviation 1)
ck_list_0 = np.random.normal(loc=n, scale=1, size=N)
cn_list_0 = np.random.normal(loc=n, scale=1, size=N)

# Find the fixed point using fixed-point iteration
# ck_list, cn_list = fixed_point_iteration(calculate_coefficients, ck_list_0, cn_list_0)
ck_list, cn_list = fixed_point_iteration(calculate_coefficients, ck_list_0, cn_list_0)
print("c_k:", ck_list)
print("c_n:", cn_list)

# Calculate the coefficients for constructing the solution by original basis
dk_list = np.zeros(N, dtype=complex)
dn_list = np.zeros(N, dtype=complex)
for i in range(N):
    for j in range(N):
        dk_list[i] += ck_list[j] * transformation_matrix_1[j,i]
        dn_list[i] += cn_list[j] * transformation_matrix_2[j,i]
print("d_k:", dk_list)
print("d_n:", dn_list)

# Construct the solution by linear combination of the original basis in 1D (t) space
linear_combined_function_1 = lambda t: 0
linear_combined_function_2 = lambda t: 0
for i in range(N):
    linear_combined_function_1 = sum_functions(linear_combined_function_1, multiplied_function(dk_list[i], basis_1[i]))
    linear_combined_function_2 = sum_functions(linear_combined_function_2, multiplied_function(dn_list[i], basis_2[i]))

# Plot 1D linear combined solution
t_list = np.linspace(a, b, 500)
# Plot the linear combined solution
plt.plot(t_list, [linear_combined_function_1(t).real for t in t_list], label="basis k")
plt.plot(t_list, [linear_combined_function_2(t).real for t in t_list],'--', label="basis n")
plt.legend()
plt.savefig(f"iteration_1d_n{n:d}_N{N:d}.pdf")

print("--- %s seconds ---" % (time.time() - start_time))