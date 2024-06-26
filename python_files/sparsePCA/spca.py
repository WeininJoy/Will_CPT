from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
import elasticnet

def pre_estimation(X, type):
    """
    Function used in sparcepca() for calculating the eigenvalues and eigenvectors of X, and 
    calculating Y for estimation using elasticnet. Requires a matrix X as input, and returns the 
    vectors Y and eigenvalues, and the matrix eigenvectors
    
    :param X: input data
    :param type: 'data' (original dataset) or 'cov' (covariance matrix) 
    :return: vector Y, eigenvalues, eigenvectors
    """
    if(type == "data"):
        n, p = X.shape
        U, D, vh = np.linalg.svd(X)
        if(n < p):
            D = np.hstack((D, np.zeros(p-n)))
        Y = (vh.T * D) @ vh
        print("Pre-estimation terminated succesfully")
        return Y, vh
    else:
        eigenvalues, eigenvectors = np.linalg.eig(X)
        eigenvalues = eigenvalues[eigenvalues.argsort()[::-1]]
        eigenvectors = eigenvectors[eigenvalues.argsort()[::-1]]
        Y = (eigenvectors * eigenvalues ** 0.5) @ eigenvectors.T
        return Y, eigenvalues, eigenvectors

def estimation_output(A, B, D, vh, fault=0):
    """
    Used for formatting and outputing the generated coefficients to the user. Requires the
    parameter vectors A (weights) and B (loadings) or a integer "fault" as input, returns either
    a string with an error message or the vectors A and B in correct format as output.
    
    :param A: weights matrix
    :param B: loadings matrix
    :param D: D matrix from SVD results
    :param vh: vh matrix from SVD results
    :param fault: 0 (noncorvergence) or 1 (convergence) 
    :return: a string with an error message or the vectors A and B in correct format as output.
    """
    if fault == 0:
        normalized_loadings = (B / np.linalg.norm(B)).T
        return A, normalized_loadings, D, vh
    elif fault == 1:
        return "Optimization terminated because maximum iteration is reached, thus estimation did \
                not converge."

def sparcepca(X, lambda2, lambda1, k, max_iteration, threshold, type, optimizer="sklearn"):
    """
    Main function used for performing the SPCA algorithm proposed by Zou, Hastie, and 
    Tibshirani (2006). Requires a matrix X, lambda1 and lambda2 as l-norms, k as # of
    PCs, max_iteration as maximum number of iterations, threshold as threshold for difference,
    and an optional parameter "optimizer" for choosing between the built-in optimizer or the
    self-written optimizer. Returns two matrices in case of correct estimation; A being the matrix 
    of weights and B being the loadings matrix. In case of estimation error, returns a string 
    containing the issue.
    
    :param X: input data matrix
    :param lambda2: L1-regulization (list of values)
    :param lambda1: L2-regulization (float)
    :param k: number of components (int)
    :param max_iteration: maximum number of iterations (int)
    :param treshold: optimization treshold (float)
    :param type: data type (cov/data)
    :param optimizer: sklearn or elasticnet
    :return: two matrices in case of correct estimation; A being the matrix 
    of weights and B being the loadings matrix.
    """
    if(type == "data"):
        Y, eigenvalues = pre_estimation(X, type)
        A_temp = eigenvalues[:k].T
    else:
        Y, eigenvalues, eigenvectors = pre_estimation(X, type)
        A_temp = eigenvectors[:, :k]
    i, difference_a, difference_b = 0, 1, 1
    B_temp = np.zeros((k, eigenvalues.shape[0]))
    B = np.zeros(((max_iteration*100), eigenvalues.shape[0])) 

    while difference_a > threshold and difference_b > threshold and i < max_iteration:
        for j in range(0,k):
            if(optimizer == "sklearn"):
                elastic_net_solver = ElasticNet(alpha=lambda2[j], l1_ratio=lambda1, 
                fit_intercept=False, max_iter=max_iteration).fit(Y, Y @ A_temp[:, j])
                B_temp[j] = np.array(elastic_net_solver.coef_)
                
            else:
                B_temp[j] = np.array(elasticnet.solver(l2=lambda2[j], l1=lambda1, 
                X=(Y @ A_temp[:, j]), Y=Y, A=A_temp[:, j]))
        B[(i*k):((i*k)+k)] = B_temp
        U, D, vh = np.linalg.svd((X.T @ X) @ np.transpose(B[(i*k):((i*k)+k)]))
        U =  U[:, :k]
        vh_out = vh
        vh = vh[:k]
        difference_a = np.linalg.norm((U @ np.transpose(vh))-A_temp)
        difference_b = 1 if i<7 else np.linalg.norm((B[(i*k):((i*k)+k)])-(B[((i*k)-6):((i*k)+k-6)]))
        A_temp = (U @ np.transpose(vh))
        i += 1
    if i >= max_iteration:
        return estimation_output(A_temp, B[((i-1)*k):(((i-1)*k)+k)], D, vh_out, fault=1)
    else:
        return estimation_output(A_temp, B[((i-1)*k):(((i-1)*k)+k)], D, vh_out)