import numpy as np
import pandas as pd

def nonzeroloadings(B):
    """
    This function calculates the number of nonzero loadings. Requires the loadings matrix B, and
    returns the number of nonzero loadings per column.
    
    :param B: loading matrix
    :return: number of nonzero loadings (int)
    """
    nonzero_count = (B != 0).astype(int).sum(axis=0)
    return nonzero_count

def variance(X, V):
    """
    This function returns the variance (not the adjusted variance) based on the formulas provided
    by Zou, Hastie, and Tibshirani (2006). The function requires the matrices X and V as input,
    and gives the variance as output.
    :param X: input data
    :param V: weight data
    :return: variance, diagonal of the covariance matrix
    """
    k = V.shape[0]
    X = X.iloc[:, :k]
    sigma = X.T @ X
    Z = V.T @ sigma @ V
    print(Z.shape)
    variance = np.trace(Z.T @ Z)
    Z_array = Z.to_numpy()
    diagonal = Z_array.T @ Z_array
    diagonal = diagonal.diagonal()
    return variance, diagonal
    
def tex_output(tables):
    """
    This function can be used to generate LaTeX output from multiple tables. Requires one or more
    tables as input, and yields the LaTeX source code as output.
    """
    width = tables[0].shape[0]
    for table in tables:
        if(table.shape[0] != width):
            return "The provided tables do not have the same dimensions."
        else:
            pass
    
