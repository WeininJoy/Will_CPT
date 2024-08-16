import numpy as np
from sklearn.decomposition import NMF

# Define the known matrix C
C = np.array([[1, 2, 3, -4, 6],
              [0, 1, 6, 3, 4],
              [0, 3, -1, 2, 8],
              [6, 3, 6, 1, 2],
              [7, 4, 9, 1, 1]])


# Add a small positive constant to ensure non-negativity
offset = np.abs(np.min(C)) + 1e-9
C_offset = C + offset

# Define the NMF model with sparsity constraints
nmf_model = NMF(n_components=C.shape[0], init='random', solver='mu', beta_loss='frobenius', max_iter=1000, random_state=0)

# Fit the model to the data (C_offset)
nmf_model.fit(C_offset)

# Get the factor matrices M and D
M = nmf_model.components_
D = nmf_model.transform(C_offset)

# D will have the most components equal to 0 among all possible solutions
print(D)
print(M @ D)