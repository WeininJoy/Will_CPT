import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA

if __name__=="__main__":
    
    # Parameters
    N = 15
    N_t = 300
    kappa = 1.5
    T = np.pi 
    L = 2. * np.pi
    eigenvalues_threshold = 1.e-10
    N_plot = 6

    # Interval t=[0, T]
    t = np.linspace(0, T, N_t)

    # List of basis to apply the Gram-Schmidt process
    basis_1 = []
    for k in range(1, N+1):
        basis_1.append(np.cos(np.sqrt(k**2 + kappa**2) * t)  )
    basis_1 = np.array(basis_1).T

    if (np.pi/T)**2 - kappa**2 < 0: # make sure wave vector k is real
        start_n = round(np.sqrt(abs((np.pi/T)**2 - kappa**2))) + 1
    else: start_n = 1
    basis_2 = []
    for n in range(start_n, N+start_n):
        basis_2.append(np.cos(n*np.pi/T * t) )
    basis_2 = np.array(basis_2).T
    
    # get orthonormal basis by QR decomposition
    orthonormal_functions_1, transformation_matrix_1 = np.linalg.qr(basis_1)
    orthonormal_functions_2, transformation_matrix_2 = np.linalg.qr(basis_2)

    for i in range(N):
        dt = T / N_t
        phi_divide = dt/T * np.ones(orthonormal_functions_1.shape[0])
        if orthonormal_functions_1[0, i] < 0:
            orthonormal_functions_1[:, i] = -orthonormal_functions_1[:, i]
            transformation_matrix_1[i, :] = -transformation_matrix_1[i, :]
        if orthonormal_functions_2[0, i] < 0:
            orthonormal_functions_2[:, i] = -orthonormal_functions_2[:, i]
            transformation_matrix_2[i, :] = -transformation_matrix_2[i, :]
    
    # Compute mixing matrix M, and find the eigenvalues and eigenvectors of MM^T and M^T M
    M = np.dot(orthonormal_functions_1.T, orthonormal_functions_2)
    X = M
    transformer = SparsePCA(n_components=5, random_state=1)
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    print(X_transformed)
    # most values in the components_ are zero (sparsity)
    print(np.mean(transformer.components_ == 0))