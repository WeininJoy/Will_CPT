# Possible Q&A Questions — KCL TPPC Seminar
## CPT-Symmetric Universe: From Quantized Curvature to Kähler-Dirac Fermions

---

## Questions by Audience Member

### Prof. Nick Mavromatos
*Expertise: CPT violation, quantum decoherence, spacetime foam, quadratic gravity*

**Questions:**
1. "Your entire framework hinges on CPT being an *exact* symmetry, imposed as a boundary condition. In string-inspired models with D-branes or spacetime foam, CPT can be violated spontaneously by the vacuum. What mechanism in your model protects this exact CPT symmetry from quantum gravitational corrections, which one would expect to be rampant near the Planckian singularity you invoke?"
Ans: It depends on what quantum gravity you choose. If string theory (have branes), have lots of extra dof. Then the CPT-violated phenomenon might be unavoidable. However, we currently interested in a new string theory picture. Instead of living in 26D for boson string or 10D in supersymmetry, we consider two-sheeted of 4D spacetime. In our KD work, we found that in which we can also cancel chiral anomaly in the case. And hopelly  the ghost introduced in gauge symmetry of string theory can also be canneled in the case. Go back to the original question, if string theory indeed live in two-sheeted 4D spacetime, we won't have CPT-violation (??? is this correct?).

2. "You appeal to quadratic gravity to generate the scale-invariant spectrum. The ghost associated with the $C^2$ term is a well-known problem. You mention resolving it with PT-reversal, but this often leads to non-unitarity. Can you be more explicit about how the two-sheeted structure and PT-reversal on the backward sheet render the combined theory unitary and ghost-free, especially concerning the spin-2 ghost?"
Ans: based on John Donogues work [2112.01974]
(1) The "Merlin Mode" Interpretation (Time-Reversed Causality)
Instead of viewing the massive spin-2 ghost as a negative-energy particle, they interpret it as a positive-energy particle that propagates backward in time.
(2) They coin the term "Merlin modes" (after the wizard who ages backward) to describe these states.
The propagator for this mode has a minus sign in the numerator (standard ghost signature) but also has an opposite sign for the imaginary part in the denominator (the decay width term).
(3) The combination of these two sign flips means the particle decays exponentially rather than growing exponentially, preserving stability.

**Key point of challenge:** His career is built on CPT *violation* — will immediately compare CPT-symmetric universe (exact CPT) to his foam-based CPT violation.

**Backup slide: "Protection of CPT & Unitarity in Quadratic Gravity"**
- CPT is a *geometric* reflection symmetry of the spacetime manifold itself — quantum fluctuations are perturbations *on* this manifold and must respect its global topology; CPT violation would require tearing the manifold.
- The ghost pole in the propagator at mass $M_g \sim M_P/\sqrt{\beta}$ splits into a complex conjugate pair in a PT-symmetric theory. The "ghost" propagates on the backward sheet (negative energy) and the normal particle on the forward sheet. Total energy is conserved and positive, and S-matrix unitarity is preserved for the combined system.

---

### Prof. Jean Alexandre
*Expertise: Non-Hermitian QFT, PT symmetry (co-authors with Bender), dynamical mass generation, Lorentz violation*

**Questions:**
1. "For a PT-symmetric theory to be physically viable, one must construct a new inner product (the CPT inner product) under which time evolution is unitary. Can you explicitly write down this inner product for the full 16-component KD field and demonstrate how the S-matrix for the combined two-sheet system is unitary?"
Ans: Yes, it can be written down. In our work we write it as PT flip way (observer in our universe). However, equivalently you can write down by flipping i <-> -i, then you will get psitive norm and energy states.

2. "Given that the wrong-sign states are reinterpreted as living on a separate spacetime sheet, how does this manifest in a practical lattice simulation? Does it require a 'doubled' lattice, and how are the links between the two sheets implemented in the lattice action to ensure the correct continuum limit?"
Ans: ?? How does it related to dual KD space? How does it related to the formulation in our KD work?


**Key point of challenge:** He is an expert in the machinery you invoke to solve the KD wrong-sign problem — will press on mathematical consistency and rigour.

**Backup slide: "The Kähler-Dirac CPT Inner Product"**
- Standard indefinite inner product: $\langle\Psi|\Psi\rangle_{std} = \int d^3x\,(\psi_F^\dagger\psi_F - \psi_B^\dagger\psi_B)$
- CPT inner product: $\langle\Psi|\Psi\rangle_{CPT} = \int d^3x\,\Psi^\dagger\mathcal{C}\Psi$, which is positive definite
- The Hamiltonian $H_{total} = H_F \oplus (-H_B)$ is Hermitian under the CPT inner product → unitary time evolution

---

### Prof. Mairi Sakellariadou
*Expertise: Loop quantum cosmology, non-commutative spectral geometry, cosmic strings*

**Questions:**
1. "In loop quantum cosmology, the need for a boundary condition at the Big Bang is obviated by the bounce. Your model instead imposes Neumann boundary conditions directly at a singularity. From what more fundamental principle are these BCs derived? Do they emerge from an underlying theory of quantum gravity, or are they an ad-hoc assumption to fit the CMB data?"
Ans: good question. So far the theory only assume classical spacetime and quantize fields on it. We haven't consider quantum gravity yet. However, as mention in the talk, if quadratic gravity or string theory on two-sheeted 4D spacetime is correct, might give us hint on it.

2. "How precisely does matching the discrete temporal wavenumbers $k_n$ to the integer eigenvalues of the Laplacian on the spatial 3-sphere uniquely fix the curvature radius? It seems sensitive to the exact evolution of $a(\eta)$ near the bang."

**Key point of challenge:** LQC resolves the singularity via a bounce — your model embraces it; she will question the physical basis for the BCs.

**Backup slide: "Quantized Curvature: Matching Temporal & Spatial Spectra"**
- Temporal Mukhanov-Sasaki equation with Neumann BC: $v_k'(0)=0$ gives discrete set $\{k_n\}$, where $k_n \approx n\pi/\eta_{max}$ for radiation domination
- Spatial Laplacian on $S^3$ of radius $R_c = 1/\sqrt{-K}$: eigenvalues $k^2 = l(l+2)/R_c^2$
- Matching condition $k_n^2 \approx k_{spatial}^2$ requires $R_c$ to take a specific value relative to the comoving particle horizon $\eta_{max}$, fixing $\Omega_K$

---

### Dr. Lucien Heurtier
*Expertise: Baryogenesis, leptogenesis, DM from right-handed neutrinos*

**Questions:**
1. "A 5×10⁸ GeV right-handed neutrino is far too heavy for thermal freeze-out. How is the correct relic abundance produced in your cosmology? Standard non-thermal mechanisms like gravitational production are sensitive to the unknown physics of reheating, which your no-inflation model lacks. What is the precise production mechanism at the Big Bang?"
2. "The existence of such a heavy Majorana neutrino with Yukawa couplings to the SM suggests lepton-number violating interactions that could wash out any pre-existing asymmetry. How do you avoid this washout problem? And how is the local baryon asymmetry generated within our universe?"
Ans: The most Massive Right-handed neutrinos, which act as DM is stable, however, the other two are not. Before electroweak transition, they would decay into Higgs boson+leptons or anti-Higss+anti-leptons. By CP-violation, the two process would be asymmetry. Then the leptons turned into baryons through Sphaleron (non-perturbative instanton) process. By which, the asymmetry between leptons and anti-leptons become baryonic asymmetry. 

**Key point of challenge:** Will scrutinise the phenomenology of the ν_R DM candidate — production mechanism and matter asymmetry.

**Backup slide: "Phenomenology of ν_R Dark Matter"**
- **Production:** Gravitational particle production — the rapidly changing metric near $t=0$ non-adiabatically excites quantum fields; rate $\propto e^{-M/H}$, and for $M \sim M_P$ this is significant
- **Relic abundance:** $\Omega_{DM}h^2 \propto (M/M_P)^2$; plugging in $M \approx 5\times10^8$ GeV gives $\Omega_{DM} \sim 0.1$
- **Stability:** ν_R is the lightest particle on the sterile KD sheet → stable by the $\mathbb{Z}_2$ symmetry
- **Washout:** Global B-L = 0 by construction (universe + anti-universe); no net asymmetry to wash out

---

### Prof. Malcolm Fairbairn
*Expertise: Dark matter, inflation, early universe cosmology*

**Questions:**
1. "Inflation elegantly solves the horizon problem by superluminal expansion. Your model uses an analytic Friedmann solution but no inflation. How do you establish causal contact across the entire observable CMB sky at the time of last scattering without inflation?"
Ans: thermodynamics
2. "What is the dynamical mechanism for 'turning off' the influence of the dim-0 scalars to allow the universe to transition into a standard radiation-dominated era? What plays the role of reheating?"
Ans: we don't know. There is a possibility that the universe 

**Key point of challenge:** Will view the proposal through the lens of standard inflationary cosmology and question whether it solves the same problems.

**Backup slide: "CPT Universe vs. Inflation"**

| Problem | Inflation | CPT Universe |
|---|---|---|
| Horizon | Superluminal expansion | Neumann BC synchronises perturbations across all scales |
| Flatness | Drives $\Omega_K \to 0$ | Predicts specific small negative $\Omega_K$ |
| PPS | Quantum fluctuations of inflaton | Conformal invariance of dim-0 scalars |
| Reheating/Exit | Inflaton decay | Dim-0 scalars are spectators; radiation domination from decay of heavy quadratic-gravity states |

---

### Prof. John Ellis
*Expertise: BSM, SUSY, DM, Higgs physics*

**Expected questions:**
- Will push on the composite Higgs model: "What specific model do you have in mind? How does it avoid the fine-tuning problem and current LHC bounds ($\kappa_V, \kappa_F$ within 5% of SM)?"
- May ask about the anomaly-cancellation argument: "Is the $n_0 = 0$ condition robust against higher-loop corrections?"

---

### Dr. Tevong You
*Expertise: Higgs phenomenology, SM EFT*

**Expected questions:**
- "How do the 36 dim-0 scalars collectively produce a composite Higgs with the right quantum numbers? What is the symmetry-breaking pattern?"
- "What is the predicted scale of compositeness, and is it compatible with current EFT bounds from Higgs coupling measurements?"

---

### Dr. Sebastian Ellis / Dr. David Marsh
*Expertise: Axion/ultralight DM, fuzzy DM*

**Expected questions:**
- "The dim-0 scalars are conformally coupled, massless, and produce a scale-invariant spectrum — this sounds like an axiverse. Is there an overlap with axion phenomenology? Could they be detected as ultralight DM?"
Ans: 
While both are scalar (spin-0) fields, they are fundamentally different objects in Quantum Field Theory.
| Feature | **Axions** | **FT Scalars (Dimension-0)** |
| :--- | :--- | :--- |
| **Kinetic Term** | Standard 2-derivative ($\partial_\mu a \partial^\mu a$) | **4-derivative** ($\phi \square^2 \phi$) |
| **Scaling Dimension** | Dimension 1 (standard boson) | **Dimension 0** |
| **Symmetry** | **Shift Symmetry** ($a \to a + c$). They are Goldstone bosons arising from broken symmetry (PQ symmetry). | **Weyl Symmetry** ($\phi \to \Omega^0 \phi$). They are defined by their invariance under conformal rescaling. |
| **Interactions** | Typically couple to gauge fields via $a F \tilde{F}$ (CP violation). | In this paper, they couple conformally to the metric (gravity). |
| **Mass** | Technically massless at classical level, but gain a small mass via instantons (QCD effects). | **Strictly massless** at the UV fixed point to maintain conformal invariance. |
| **Physical Role** | Solves the Strong CP problem; Dark Matter candidate. | **Cancels Gravitational Vacuum Energy**; stabilizes the UV behavior of gravity. |

---

### Prof. Ruth Gregory / Dr. Lionel London
*Expertise: Black holes, GR, QNMs*

**Expected questions:**
- Gregory: "Does the black mirror proposal modify the BH thermodynamics — entropy, temperature? Is the Penrose diagram consistent with a regular horizon?"
- London: "Does the two-sheeted KD structure modify quasi-normal modes? Could there be 'mirror QNMs' from the backward spacetime sheet detectable in gravitational wave ringdown?"

---

## General Backup Slides (Cross-Audience)

### 1. KD Formalism in a Nutshell
*(For: Alexandre, Sarkar, Ellis)*
- KD field $\Phi = \sum_p \phi^{(p)}$ is a polyform with 16 complex components in 4D
- Action: $S = \int \overline{\Phi}(d - d^\dagger - m)\Phi$; decomposes into 4 copies of the Dirac equation
- Lorentzian problem: kinetic term $\bar\psi_1\partial\psi_1 + \bar\psi_2\partial\psi_2 - \bar\psi_3\partial\psi_3 - \bar\psi_4\partial\psi_4$
- Solution: $(\psi_3, \psi_4)$ live on the PT-reversed backward sheet

### 2. Analytic Friedmann Solution
*(For: Sakellariadou, Lim, Fairbairn)*
- Jacobi elliptic function solution $a(\eta) \propto \text{sn}(\eta | m)$
- Plot showing doubly-periodic nature and mirror symmetry around the Big Bang
- Imaginary period → gravitational entropy $S_g$; thermodynamics selects most probable universe

### 3. Why Dim-0 Scalars Give a Scale-Invariant Spectrum
*(For: Marsh, S. Ellis, Fairbairn)*
- Fradkin-Tseytlin scalar: conformally invariant by construction
- Two-point function in de Sitter: $\langle\phi(x)\phi(y)\rangle \propto \log d(x,y)$ → flat power spectrum $P(k) \propto k^0$
- Small deviation $n_s \approx 0.96$ from running of gauge couplings

### 4. Current Observational Status of $\Omega_K$
*(For: everyone in Part 2 Q&A)*
- Planck 2018 + lensing: $\Omega_K = -0.044^{+0.018}_{-0.015}$ ($2\sigma$ preference for closed)
- Di Valentino, Melchiorri, Silk (2019): combining datasets gives >3σ preference for closed
- Handley reanalysis: ~3σ Bayesian tension for $\Omega_K < 0$
- Mark the CPT-symmetric universe prediction on the Planck plot

### 5. Black Mirror: Penrose Diagram
*(For: Gregory, London)*
- Left: Standard Schwarzschild BH — infalling observer hits future singularity
- Right: Black mirror — interior replaced by gateway to anti-universe sheet
- Infalling matter meets anti-matter partner from anti-universe's Big Crunch; annihilate at junction
- Consequences: no information paradox; possibly modified QNM spectrum ("mirror QNMs")

### 6. CPT-Symmetric vs. CPT-Violating Approaches
*(For: Mavromatos — pre-empt his comparison)*

| Approach | Key idea | Observational handle |
|---|---|---|
| Boyle-Turok CPT universe | Exact CPT at Big Bang; $\mathbb{Z}_2$ anti-universe | $\Omega_K < 0$, massless lightest $\nu$, scale-inv. PPS |
| Mavromatos D-foam | CPT violated by spacetime foam; quantum decoherence | Neutrino anomalies, neutral meson EPR correlations |
| Standard inflation | No CPT requirement | $r < 0.036$ (BICEP/Keck), $n_s \approx 0.965$ |
