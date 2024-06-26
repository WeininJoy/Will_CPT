# %%
import pandas as pd
import numpy as np
import spca
import utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
# %%

def qr_decomposition(basis):
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix.T)

# Note: implementation of np.cov()
def cov_matrix(X):  
    X -= X.mean(axis=1)[:, None]
    N = X.shape[1]  
    return np.dot(X, X.T.conj())/float(N-1)

def eigenvectors(X):
    cov_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    return sorted_eigenvalue, sorted_eigenvectors

# this would give as the same result as eigenvectors()
def eigenvectors_PCA(X):
    pca = PCA(n_components=N)
    pca.fit(X)
    sorted_eigenvalues = pca.explained_variance_
    sorted_eigenvectors = pca.components_.T
    return sorted_eigenvalues, sorted_eigenvectors

def eigenvectors_sparsePCA(X, num_sol):
    cov_matrix = np.dot(X, X.T.conj())
    lambda2 = 0.1* np.ones(num_sol)
    Y, eigenvalues, eigenvectors, v_h = spca.sparcepca(X=cov_matrix, lambda2=lambda2, lambda1=0.1, k=num_sol, max_iteration=int(1e3), threshold=1e-4, type="cov")
    return Y, eigenvalues, eigenvectors

def valid_eigenvalues(eigenvalues):
    eigenvalues_valid_idx = []
    for i in range(1, len(eigenvalues)):
        if np.abs(eigenvalues[0] - eigenvalues[i]) < eigenvalues[0]*eigenvalues_threshold:
            eigenvalues_valid_idx.append(i)
        else: 
            break
    return eigenvalues_valid_idx

# Parameters
N = 15
N_t = 300
kappa = 1.5
T = np.pi
eigenvalues_threshold = 0.5
diff_threshold = 0.1
N_plot = N

t = np.linspace(0, T, N_t)

# List of basis to apply the Gram-Schmidt process
basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) for k in range(1, N+1)]
basis_2 = [np.cos(n*np.pi/T * t) for n in range(2, N+2)]

M = np.dot(np.array(basis_1), np.array(basis_2).T)

X_1 = M.T
X_2 = M

eigenvalues_1, eigenvectors_1 = eigenvectors_PCA(X_1)
eigenvalues_2, eigenvectors_2 = eigenvectors_PCA(X_2)

eigenvalues_valid_idx_1 = valid_eigenvalues(eigenvalues_1)
eigenvalues_valid_idx_2 = valid_eigenvalues(eigenvalues_2)
eigenvectors_valid_1 = eigenvectors_1[:, eigenvalues_valid_idx_1]
eigenvectors_valid_2 = eigenvectors_2[:, eigenvalues_valid_idx_2]

# Compute coefficients
coefficients_1 = eigenvectors_valid_1.T
coefficients_2 = eigenvectors_valid_2.T

# Construct the solutions
solution_1_list = []
solution_2_list = []
for i in range(len(eigenvalues_valid_idx_1)):
    solution_1 = np.zeros_like(basis_1[0])
    solution_2 = np.zeros_like(basis_2[0])
    for j in range(N):
        solution_1 += coefficients_1[i, j]*basis_1[j]
        solution_2 += coefficients_2[i, j]*basis_2[j] 
    if solution_1.real[0] < 0:
        solution_1 = -solution_1
    if solution_2.real[N_t//2] * solution_1.real[N_t//2] < 0:
        solution_2 = -solution_2 
    solution_1_list.append(solution_1)
    solution_2_list.append(solution_2)

diff_array = np.zeros((len(eigenvalues_valid_idx_1), len(eigenvalues_valid_idx_2)))
small_diff_index = []
min_diff = []
num_sol = len(eigenvalues_valid_idx_1)  # number of solutions, which has small difference
for i in range(len(eigenvalues_valid_idx_1)):
    solution_1 = solution_1_list[i]
    for j in range(len(eigenvalues_valid_idx_2)):
        solution_2 = solution_2_list[j]
        diff_array[i, j] += np.linalg.norm(solution_1 - solution_2) / np.linalg.norm(solution_1)
    small_diff_index.append(np.argmin(diff_array[i, :]))
    min_diff.append(np.min(diff_array[i, :]))
    if np.min(diff_array[i, :]) > diff_threshold:
        num_sol -= 1
sorted_index = np.argsort(min_diff)


Y, eigenvalues, eigenvectors = eigenvectors_sparsePCA(X_1, num_sol)
print(Y)
print(eigenvalues)
print(eigenvectors)
