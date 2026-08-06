Our current work flow is: 

Generate perturbations solutions for all integerK (basis_1) and allowedK (basis_2)

Use QR decomposition to find orthonormal basis_1 and basis_2, and their transformation matrix (depends on perturbation types) w.r.t the original ones.

Calculate valid solutions (eigenvectors with egienvalue close to 1) based on orthonormal solutions.

Use Varimax to find sparse valid solutions.

Practical Problem: while calculating the QR decomposition, we already mix all bases → the index of orthonromal solution doesn’t represent the one in original basis. As a result the sparse solution (w.r.t the index) we found would not be sparse in the original integerK and allowedK basis.Fundamental Problem: after we did every steps and finally get the orthonormal sparse solution, we found that the number of valid modes reduce from N=335 to N=155. It don’t make sense, since modes larger then K=25 approach to each other. So we should have at least 335-25=310 valid solutions. I think the  problem is because the bases in basis_1 and basis_2 are degenerate (ex: 6 vectors in 3D space). So after we normalize them, we can only find 155 orthonormal basis, the others are squeezed to be nearly zero. As a result, after we calculate the eigenvector and eigenvalues, those degenerate bases are ignored. Lefting only 155 sparse and orthonormal valid solutions.

Solution 1

An althernative method is calculating coefficients without normalization, such as PCA. Eigenvectors with larger eigenvalues means better aligned. However, since we don’t normalized, the eigenvalues for valid solutions would no longer be 1. We should find another way to tell whether a solution is valid or not.

Another problem is there are several types of perturbations. Our current method is sum up inner product of all types to get final inner product between two basis $\big<\Phi_1|\Phi_2\big>$. 

As for PCA, we might be able to perform the similar tasl by at least normalize all perturbation solutions (only change the amplitude without mixing bases), and then sum them together before performing PCA.