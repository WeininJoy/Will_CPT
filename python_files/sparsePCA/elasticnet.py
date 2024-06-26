from scipy.optimize import minimize
import numpy as np
import pandas as pd

def objective(B, l2, l1, X, Y, A):
    """
    Objective function for the elasticnet minimalization problem. Requires the l1 and l2 norms, X,
    Y, A and B as arguments. Returns a OptimizeResult object.
    """
    obj = np.linalg.norm(Y - X @ B)**2 + l1 * np.linalg.norm(B)+ l2 * np.linalg.norm(B)**2
    return obj

def solver(l2, l1, X, Y, A):
    """
    Minimizer function for the elasticnet problem, creates the starting values for B and then runs
    the SciPy minimizer. Requires the l1 and l2 norms, X, Y and A as arguments. Returns the matrix
    B containing the parameter estimates.
    """
    B_start = np.full((1, 13), 0.45)
    B_new = minimize(objective, x0=B_start, args=(l2, l1, X, Y, A), method="BFGS", 
            options={"maxiter":1e4})
    return B_new.x
