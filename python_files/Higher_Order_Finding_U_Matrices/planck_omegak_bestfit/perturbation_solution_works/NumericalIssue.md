Based on the analysis of the two papers and your provided plots, the behavior you are observing in the `Difference between Datasets` plot—where the difference oscillates with increasing amplitude as $k$ increases—is **almost certainly a numerical issue**, not a physical one.

Here is the detailed breakdown of why this is the case, rooted in the mathematical methods described in Paper 2 (*PhysRevD.113.023546*).

### 1. The Taylor Expansion Limit (The most likely culprit)
In Paper 2, Section III.B and Appendix A, the authors describe the method for handling the singularity at the Future Conformal Boundary (FCB), $\eta_\infty$.
*   The code integrates numerically from recombination up to a point $\eta' = \eta_\infty - \Delta \eta$.
*   It then uses a **Taylor expansion** (Eqs. A1-A11) to bridge the gap $\Delta \eta$ to the boundary.

**The Issue:**
Look at the coefficients in the Taylor expansion matrices in the Appendix (e.g., Eq. A6, A10). They contain terms like $k^6$, $k^4$, and factors like $k^2 (\Delta \eta)^3$.
For the Taylor expansion to be valid, the product $k \Delta \eta$ (or specifically terms like $k^2 (\Delta \eta)^2$ inside the matrix elements) must be small.
$$ k \Delta \eta \ll 1 $$

If your code uses a **fixed** $\Delta \eta$ (the distance from the boundary where you switch from ODE solver to Taylor expansion), then as $k$ increases:
1.  The term $k \Delta \eta$ grows.
2.  The Taylor approximation breaks down because higher-order terms you neglected become significant.
3.  This manifests as an error in the phase calculation, which oscillates because the sign of the error depends on where in the sine wave the expansion connects.

**Evidence from your plot:** The "Difference" plot shows the error amplitude growing as $k$ increases (higher index). This perfectly matches the expectation of a Taylor expansion error scaling with powers of $k$.

### 2. Integration Phase Error
Even before the boundary step, solving the ODEs for high-$k$ modes is numerically stiff.
*   The wavefunction oscillates as $\sim \exp(i k \eta)$.
*   The accumulated phase error in a Runge-Kutta (or similar) integrator scales with the number of oscillation cycles.
*   For high $k$, the number of cycles is large. A tiny relative error in the wavenumber or step size accumulates into a significant phase shift by the time you reach the boundary.

### 3. Physical Expectations (Why it *should* be equally spaced)
Physically, the "non-equal spacing" at high $k$ is not expected.
*   **WKB Limit:** As $k \to \infty$, the wavelength of the perturbation becomes much smaller than the curvature radius or the horizon scale ($k \gg \mathcal{H}$).
*   In this limit (WKB approximation), the curvature and potential terms in the Friedmann equation become negligible compared to $k^2$. The universe looks effectively flat and static to the perturbation over short timescales.
*   The solution approaches a plane wave $\Phi \propto e^{ik\eta}$.
*   Consequently, the phase accumulation becomes strictly linear with $k$, and the allowed modes (determined by phase conditions at boundaries) **must** approach equal spacing asymptotically.

### Summary & Recommendation
The divergence you see is non-physical. The fact that the difference oscillates suggests that for some $k$ values the numerical error adds constructively and for others destructively, or that the breakdown of the Taylor expansion is introducing spurious phase shifts.

**To fix this:**
1.  **Adaptive $\Delta \eta$:** Ensure that the $\Delta \eta$ used for the Taylor expansion step scales inversely with $k$. You should enforce $k \Delta \eta < \epsilon$ (where $\epsilon$ is a small tolerance), rather than keeping $\Delta \eta$ fixed in conformal time.
2.  **Increase Precision:** If you are using `scipy.odeint` or `solve_ivp`, decrease the `rtol` and `atol`.
3.  **Asymptotic Matching:** For very high $k$ (where the numerical integration is hardest), you should ideally switch to an analytic WKB approximation rather than full numerical integration, as the analytic approximation becomes *more* accurate as $k$ increases, while the numerical integration becomes *less* accurate.