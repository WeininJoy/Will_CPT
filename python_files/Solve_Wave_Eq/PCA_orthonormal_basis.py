import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA

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

def valid_eigenvalues(eigenvalues):
    eigenvalues_valid_idx = []
    for i in range(1, len(eigenvalues)):
        if np.abs(eigenvalues[0] - eigenvalues[i]) < eigenvalues[0]*eigenvalues_threshold:
            eigenvalues_valid_idx.append(i)
        else: 
            break
    return eigenvalues_valid_idx

def sparse_coefficients(coefficients, n_sparse):
    transformer = SparsePCA(n_components=n_sparse)
    transformer.fit(coefficients)
    transformed_coefficients = transformer.components_
    return transformed_coefficients

# Parameters
N = 15
N_t = 300
kappa = 1.5
T = np.pi
eigenvalues_threshold = 1.e-2
N_plot = N

t = np.linspace(0, T, N_t)

# List of basis to apply the Gram-Schmidt process
basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) for k in range(1, N+1)]
basis_2 = [np.cos(n*np.pi/T * t) for n in range(2, N+2)]

# Apply Gram-Schmidt process (by qr decomposition) to get orthonormal basis 
orthonormal_functions_1, transformation_matrix_1 = qr_decomposition(np.array(basis_1).T)
orthonormal_functions_2, transformation_matrix_2 = qr_decomposition(np.array(basis_2).T)

M = np.dot(orthonormal_functions_1.T, orthonormal_functions_2)

X_1 = M.T
X_2 = M

eigenvalues_1, eigenvectors_1 = eigenvectors_PCA(X_1)
eigenvalues_2, eigenvectors_2 = eigenvectors_PCA(X_2)

eigenvalues_valid_idx_1 = valid_eigenvalues(eigenvalues_1)
eigenvalues_valid_idx_2 = valid_eigenvalues(eigenvalues_2)
eigenvalues_valid_1 = eigenvalues_1[eigenvalues_valid_idx_1]
eigenvalues_valid_2 = eigenvalues_2[eigenvalues_valid_idx_2]
eigenvectors_valid_1 = eigenvectors_1[:, eigenvalues_valid_idx_1]
eigenvectors_valid_2 = eigenvectors_2[:, eigenvalues_valid_idx_2]

# Compute coefficients
coefficients_1 = np.dot(eigenvectors_valid_1.T, transformation_matrix_1)
coefficients_2 = np.dot(eigenvectors_valid_2.T, transformation_matrix_2)

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
for i in range(len(eigenvalues_valid_idx_1)):
    solution_1 = solution_1_list[i]
    for j in range(len(eigenvalues_valid_idx_2)):
        solution_2 = solution_2_list[j]
        diff_array[i, j] += np.linalg.norm(solution_1 - solution_2) / np.linalg.norm(solution_1)
    small_diff_index.append(np.argmin(diff_array[i, :]))
    min_diff.append(np.min(diff_array[i, :]))
sorted_index = np.argsort(min_diff)


# Plot the solutions
if N_plot > len(eigenvalues_valid_idx_1):
    N_plot = len(eigenvalues_valid_idx_1)

fig, axs = plt.subplots(N_plot, figsize=(6,0.8*N_plot))
fig.suptitle(r"$\Phi(t)$")
for i in range(N_plot):
    idx = sorted_index[i]
    solution_1 = solution_1_list[idx]
    solution_2 = solution_2_list[small_diff_index[idx]]
    solution_1_sparse = solution_1_sparse_list[small_diff_index_sparse[idx]]
    axs[i].plot(t, solution_1.real, color='r', label=r"basis: $\phi$")
    axs[i].plot(t, solution_2.real, color='g', linestyle='dashed', label=r"basis: $\tilde\phi$")
    # axs[i].plot(t, solution_1_sparse.real, color='b', linestyle='dotted', label=r"basis: $\phi_{sparse}$")
plt.legend()
plt.xlabel(r"$t$")
plt.savefig(f"eigenvalue_1d_N{N:d}_Nt{N_t:d}_sparsePCA.pdf")

