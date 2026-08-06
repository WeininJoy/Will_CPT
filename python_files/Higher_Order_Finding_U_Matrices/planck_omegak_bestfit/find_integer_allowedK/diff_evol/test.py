import numpy as np
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

initial_population = np.load("./data/initial_population.npy")
print(initial_population)

allowedK_integer = np.load("./data/param_set_1/data_all_k50-65/allowedK_integer.npy")
print(allowedK_integer)