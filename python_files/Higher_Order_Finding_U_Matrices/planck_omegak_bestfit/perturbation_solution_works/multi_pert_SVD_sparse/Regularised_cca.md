


This is an excellent observation. What you are experiencing is a classic phenomenon in spectral methods known as **eigenspace degeneracy**, combined with **ill-conditioning**. 

Here is exactly what is happening mathematically, why it ruins your sparse modes, and how **Regularised CCA (RCCA)** rigorously solves it.

---

### 1. The Mathematical Origin of the High-$k$ Degeneracy

At high wavenumbers ($k$), the boundary conditions (e.g., closed universe vs. flat universe approximations) matter less. The `integerK` and `allowedK` bases become practically identical. 

Mathematically, in the high-$k$ subspace, the Gram matrices become nearly equal:
$$ G_1 \approx G_2 \approx G_{12} $$
When you compute the whitened cross-Gram matrix in standard CCA:
$$ W = G_1^{-1/2} G_{12} G_2^{-1/2} $$
In this high-$k$ subspace, this evaluates to:
$$ W \approx G_1^{-1/2} G_1 G_1^{-1/2} = I $$

**The problem:** The identity matrix $I$ has repeated singular values of exactly $1$. 
If an entire block of high-$k$ modes all yield a canonical correlation of $\rho = 1$, the SVD solver sees a "flat plateau." Because any linear combination of eigenvectors with the same eigenvalue is also a valid eigenvector, **the SVD solver will pick arbitrary, dense, mixed combinations of your high-$k$ modes.** 

It no longer cares about keeping the modes isolated (sparse); it just mixes them randomly because they all look equally perfect ($\rho=1$) to the standard CCA objective. Furthermore, numerical noise gets amplified by $G^{-1/2}$ for modes with low power.

---

### 2. Regularised CCA (RCCA) Formulation

To fix this, we introduce **Regularised CCA (also called Ridge CCA)**. We add an $\ell_2$ (Tikhonov) penalty to the magnitudes of the coefficient vectors $\mathbf{a}$ and $\mathbf{b}$. This forces the algorithm to prefer simpler, smaller coefficients and fundamentally breaks the $\rho=1$ degeneracy.

#### The Standard CCA Objective
Standard CCA maximizes the correlation:
$$ \max_{\mathbf{a}, \mathbf{b}} \mathbf{a}^\top G_{12} \mathbf{b} $$
Subject to the constraints that the variance in each basis is exactly 1:
$$ \mathbf{a}^\top G_1 \mathbf{a} = 1, \quad \mathbf{b}^\top G_2 \mathbf{b} = 1 $$

#### The Regularised CCA Objective
We relax the constraints by adding a penalty $\gamma > 0$ on the squared norm of the vectors $\mathbf{a}^\top \mathbf{a}$ and $\mathbf{b}^\top \mathbf{b}$:
$$ \mathbf{a}^\top G_1 \mathbf{a} + \gamma \mathbf{a}^\top \mathbf{a} = 1 $$
$$ \mathbf{b}^\top G_2 \mathbf{b} + \gamma \mathbf{b}^\top \mathbf{b} = 1 $$

We can rewrite this elegantly by defining regularised Gram matrices:
$$ \tilde{G}_1 = G_1 + \gamma I $$
$$ \tilde{G}_2 = G_2 + \gamma I $$

The Lagrangian for this new optimization problem is:
$$ \mathcal{L}(\mathbf{a}, \mathbf{b}, \rho_1, \rho_2) = \mathbf{a}^\top G_{12} \mathbf{b} - \frac{\rho_1}{2} (\mathbf{a}^\top \tilde{G}_1 \mathbf{a} - 1) - \frac{\rho_2}{2} (\mathbf{b}^\top \tilde{G}_2 \mathbf{b} - 1) $$

#### Taking the Derivatives
Taking the derivative with respect to $\mathbf{a}$ and setting to zero:
$$ \frac{\partial \mathcal{L}}{\partial \mathbf{a}} = G_{12} \mathbf{b} - \rho_1 \tilde{G}_1 \mathbf{a} = 0 \implies G_{12} \mathbf{b} = \rho_1 \tilde{G}_1 \mathbf{a} $$
Taking the derivative with respect to $\mathbf{b}$:
$$ \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = G_{12}^\top \mathbf{a} - \rho_2 \tilde{G}_2 \mathbf{b} = 0 \implies G_{12}^\top \mathbf{a} = \rho_2 \tilde{G}_2 \mathbf{b} $$

Multiplying the first equation by $\mathbf{a}^\top$ and the second by $\mathbf{b}^\top$ reveals that $\rho_1 = \rho_2 = \rho$, where $\rho$ is the regularised canonical correlation.

Substituting $\mathbf{b}$ into the first equation yields the generalized eigenvalue problem:
$$ \tilde{G}_1^{-1} G_{12} \tilde{G}_2^{-1} G_{12}^\top \mathbf{a} = \rho^2 \mathbf{a} $$

Which is mathematically solved by finding the SVD of the new whitened matrix:
$$ \tilde{W} = (G_1 + \gamma I)^{-1/2} G_{12} (G_2 + \gamma I)^{-1/2} = U \Sigma V^\top $$

---

### 3. Why this breaks the degeneracy (The Physical Magic)

Let's look at what happens in the problematic high-$k$ subspace where $G_1 \approx G_2 \approx G_{12}$. 
Let the natural power (eigenvalue) of a specific high-$k$ mode be $\lambda_i$. 

In **Standard CCA**, the correlation for this mode is:
$$ \rho_i = \frac{\lambda_i}{\sqrt{\lambda_i \cdot \lambda_i}} = \frac{\lambda_i}{\lambda_i} = 1 $$
*(Every mode equals 1, creating the flat plateau).*

In **Regularised CCA**, the correlation becomes:
$$ \rho_i = \frac{\lambda_i}{\sqrt{(\lambda_i + \gamma)(\lambda_i + \gamma)}} = \frac{\lambda_i}{\lambda_i + \gamma} $$

Because the natural power $\lambda_i$ of the cosmological perturbations generally decreases or oscillates as $k$ increases, **each high-$k$ mode will have a slightly different $\lambda_i$**. 
Therefore, $\frac{\lambda_i}{\lambda_i + \gamma}$ will be slightly different for every single mode! 

By adding $\gamma$, you tilt the flat plateau. The singular values are no longer perfectly identical. The SVD is mathematically forced to isolate the individual, distinct $k$-modes rather than mixing them randomly. **Sparsity is automatically restored.**

---

### 4. How to implement this in your Python code

It requires a tiny but powerful modification to your `compute_gram_matrices` or `matrix_sqrt_inv` function.

In `cca_sparse_analysis.py`, simply modify the `matrix_sqrt_inv` function to accept a `gamma` parameter:

```python
def matrix_sqrt_inv(G, ev_threshold=1e-10, gamma=1e-5):
    """
    Compute G^{-1/2} via eigendecomposition with Ridge regularisation.
    
        G_reg = G + gamma * I
        G_reg = Q Lambda Q^T  =>  G_reg^{-1/2} = Q Lambda^{-1/2} Q^T
    """
    # 1. Apply Regularisation (Ridge penalty)
    N_k = G.shape[0]
    G_reg = G + gamma * np.eye(N_k)
    
    # 2. Symmetrise
    G_reg = 0.5 * (G_reg + G_reg.T)
    
    # 3. Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(G_reg)

    # 4. Truncate and invert
    cutoff = ev_threshold * eigvals[-1]
    keep   = eigvals > cutoff
    rank   = int(np.sum(keep))

    lam      = eigvals[keep]
    Q        = eigvecs[:, keep]
    Ginvhalf = Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T
    
    return Ginvhalf, rank
```

**How to choose $\gamma$:**
* Start with a small value relative to the maximum eigenvalue of your Gram matrices, e.g., $\gamma = 10^{-5}$ or $10^{-4}$.
* If $\gamma$ is too small, the SVD will still mix the modes (degeneracy remains).
* If $\gamma$ is too large, you artificially suppress the cross-correlation $\rho$ too much, and the CCA might drop valid modes because they fall below your `rho_min = 0.99` threshold.