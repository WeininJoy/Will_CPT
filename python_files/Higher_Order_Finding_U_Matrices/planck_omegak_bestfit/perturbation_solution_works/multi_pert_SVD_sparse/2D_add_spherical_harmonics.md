



Yes, this is a **brilliant and highly rigorous idea**. 

Currently, your CCA algorithm matches modes purely based on how well their **time evolutions** ($\tau$) align. However, if Basis 1 (`integerK`) and Basis 2 (`allowedK`) use different $k$-grids, two modes might accidentally have similar time evolutions but represent completely different physical spatial scales. 

By adding the spatial part (making it a 2D space-time match), you force the algorithm to only correlate modes that are physically compatible in real space. This will drastically reduce spurious correlations and make the resulting joint eigenfunctions much more reliable.

Here is the physical and mathematical breakdown of what the spatial part is in a closed universe, and exactly how you can implement it in your pipeline.

---

### 1. What is the spatial part in a closed universe?

In a flat universe, the spatial eigenfunctions of the Laplacian $\nabla^2$ are plane waves $e^{i \vec{k} \cdot \vec{x}}$. 

In a **closed universe** (a 3-sphere, $S^3$), the eigenfunctions are **Hyperspherical Harmonics**, commonly denoted as $Y_{\beta \ell m}(\chi, \theta, \phi)$ or $Q_{\beta \ell m}(\chi, \theta, \phi)$. 

Because the background universe is homogeneous and isotropic, you don't need the full 3D spatial geometry; the angular parts ($\theta, \phi$) perfectly integrate out due to spherical symmetry. You only need to match the **radial part**, which depends on the comoving distance $\chi$.

For scalar cosmological perturbations (like $\delta_m, \delta_r$), the radial spatial eigenfunction is:
$$ \Pi_\beta(\chi) = \frac{\sin(\beta \chi)}{\beta \sin \chi} $$
Where:
* $\chi$ is the comoving radial distance, with domain $\chi \in [0, \pi]$.
* $\beta$ is the dimensionless wavenumber (often called $\nu$ in CAMB/CLASS). For a closed universe, $\beta$ takes **integer values** ($\beta = 3, 4, 5, \dots$) — which is exactly where your `integerK` basis comes from!

---

### 2. How this makes the solution more reliable

If you construct the full 2D inner product, you integrate over both time $\tau$ and space $\chi$. The volume element for a closed universe is $dV \propto \sin^2\chi \, d\chi$.

The spatial overlap between a mode with wavenumber $\beta_1$ (from Basis 1) and a mode with wavenumber $\beta_2$ (from Basis 2) is:
$$ S(\beta_1, \beta_2) = \int_0^\pi \Pi_{\beta_1}(\chi) \, \Pi_{\beta_2}(\chi) \sin^2\chi \, d\chi $$

Substitute $\Pi_\beta(\chi)$:
$$ S(\beta_1, \beta_2) = \int_0^\pi \frac{\sin(\beta_1 \chi)}{\beta_1 \sin \chi} \frac{\sin(\beta_2 \chi)}{\beta_2 \sin \chi} \sin^2\chi \, d\chi = \frac{1}{\beta_1 \beta_2} \int_0^\pi \sin(\beta_1 \chi) \sin(\beta_2 \chi) \, d\chi $$

**Why this is mathematically powerful for your CCA:**
* If both $\beta_1$ and $\beta_2$ are integers, this integral evaluates to $\frac{\pi}{2 \beta_1^2}$ if $\beta_1 = \beta_2$, and exactly **$0$** if $\beta_1 \neq \beta_2$.
* If Basis 2 (`allowedK`) contains non-integers, $S(\beta_1, \beta_2)$ acts as a tight mathematical filter. It will naturally be near zero unless $\beta_1 \approx \beta_2$.

By including this, you explicitly tell the CCA: *"Do not mix modes that have zero spatial overlap, no matter how similar their time evolutions look."*

---

### 3. How to implement this in your CCA formulation

The beauty of this idea is that you do **not** need to rewrite your data structures into huge 2D arrays. Because space and time are separable in the background universe, the 2D space-time Gram matrix is simply the **element-wise (Hadamard) product** of the time-Gram matrix and the space-Gram matrix.

In your mathematical formulation (Section 3.1 of your `.tex` file), you currently have:
$$ G_{12} = \sum_p w_p \, (\bar{X}_1^p)^\top \bar{X}_2^p $$

To upgrade this to 2D Space-Time CCA, you calculate a spatial overlap matrix $S$ where $S_{ij}$ is the integral $S(\beta_{1,i}, \beta_{2,j})$ evaluated for the $i$-th mode of Basis 1 and the $j$-th mode of Basis 2.

Your new Gram matrices simply become:
$$ G_1^{(2D)} = G_1 \odot S_{11} $$
$$ G_2^{(2D)} = G_2 \odot S_{22} $$
$$ G_{12}^{(2D)} = G_{12} \odot S_{12} $$
*(where $\odot$ denotes element-wise multiplication in Python: `G12 * S12`)*.

#### Code modifications (`cca_sparse_analysis.py`):

You would add a small function to compute the spatial overlap:

```python
def compute_spatial_overlap(k_array_1, k_array_2):
    """
    Compute the spatial overlap matrix S_{ij} for a closed universe.
    k_array_1 and k_array_2 are the beta wavenumbers for basis 1 and 2.
    """
    N1 = len(k_array_1)
    N2 = len(k_array_2)
    S = np.zeros((N1, N2))
    
    for i, b1 in enumerate(k_array_1):
        for j, b2 in enumerate(k_array_2):
            # Analytic evaluation of the integral: int_0^pi sin(b1*x) sin(b2*x) dx
            if np.isclose(b1, b2):
                integral = np.pi / 2.0
            else:
                integral = (np.sin((b1 - b2)*np.pi) / (2 * (b1 - b2)) 
                          - np.sin((b1 + b2)*np.pi) / (2 * (b1 + b2)))
            
            S[i, j] = integral / (b1 * b2)
            
    return S
```

Then, in Step 3 (`compute_gram_matrices`), you multiply the time-Gram matrix by the spatial overlap matrix:

```python
# Assuming you pass the k_arrays (beta values) into the function
S11 = compute_spatial_overlap(k1, k1)
S22 = compute_spatial_overlap(k2, k2)
S12 = compute_spatial_overlap(k1, k2)

# Element-wise multiplication merges the time and space overlaps
G1  = G1_time * S11
G2  = G2_time * S22
G12 = G12_time * S12
```

### Summary
Yes, adding the spatial part is exactly the right intuition. It grounds the CCA in the physical geometry of the universe ($S^3$), acting as a stringent geometric prior that will dramatically naturally sparsify $G_{12}$ even before the Promax rotation is applied.