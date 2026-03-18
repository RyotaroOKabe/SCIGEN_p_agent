# Technical Terms - scigenp_overview_MA

> Reading notes for: `/pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview_MA/material_dpo.tex`
> Last updated: 2026-03-16

This document is the mathematical/algorithmic companion to overview.tex, detailing the DPO formulation for crystal diffusion models with SCIGEN constraints.

---

## Core Acronyms

| Acronym | Full Form | Meaning |
|---------|-----------|---------|
| DPO | Direct Preference Optimization | Training approach that directly optimizes model from pairwise preferences without explicit reward model |
| SCIGEN+ | SCIGEN Plus | Extension of SCIGEN with human preference learning via DPO |
| DiffCSP++ | Diffusion for Crystal Structure Prediction Plus | Variant of DiffCSP with Wyckoff position constraints removed for greater flexibility |
| H-DPO | Hölder-DPO | Robust variant of DPO using Hölder divergence instead of KL divergence |
| DDPM | Denoising Diffusion Probabilistic Model | Core diffusion model architecture |
| RLHF | Reinforcement Learning from Human Feedback | Traditional approach requiring explicit reward model (DPO avoids this) |

---

## Mathematical Notation

### Crystal Structure Representation

| Symbol | Meaning | Equation Reference |
|--------|---------|-------------------|
| $\bs{x}_0$ | Clean crystal structure | $\bs{x}_0 = (\bs{L}, \bs{F}, \bs{A})$ |
| $\bs{L}$ | Lattice matrix (3×3) | |
| $\bs{F}$ | Fractional coordinates | $\bs{F} \in [0,1)^{N \times 3}$, lives on torus |
| $\bs{A}$ | Atom types (one-hot) | $\bs{A} \in \mathbb{R}^{h \times N}$ |
| $\bs{k}$ | Lattice coefficient vector | O(3)-invariant, $\bs{k} = (k_1, \ldots, k_6)$ |
| $\bs{M}_t$ | Noisy state tuple | $(\bs{k}_t, \bs{F}_t, \bs{A}_t)$ |
| $\bs{m}$ | Space group mask | $\bs{m} \in \{0,1\}^6$ for lattice bases |

### Diffusion Process

| Symbol | Meaning | Equation Reference |
|--------|---------|-------------------|
| $q(\bs{x}_t \mid \bs{x}_0)$ | Forward diffusion | Eq. 2.1, 2.3, 2.5 |
| $p_\theta(\bs{x}_{t-1} \mid \bs{x}_t)$ | Reverse denoising | |
| $\boldsymbol{\varepsilon}_\theta$ | Predicted noise | |
| $\alpha_t, \bar{\alpha}_t$ | Noise schedule | $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ |
| $\sigma_t$ | Wrapped Gaussian std | For fractional coords |
| $\mathcal{N}_w$ | Wrapped Normal | Gaussian on torus $[0,1)^{N\times3}$ |
| $T$ | Total timesteps | Typically 1000 |

### Preference Learning

| Symbol | Meaning | Equation Reference |
|--------|---------|-------------------|
| $\bs{x}^w$ | Preferred (winner) | Better flat band |
| $\bs{x}^\ell$ | Rejected (loser) | Worse flat band |
| $c_A$ | Phase A context | (chem family, N-bucket, symmetry) |
| $c_B$ | Phase B context | $(C, N, \mathcal{A}^c)$ |
| $\kappa$ | Confidence score | $\kappa \in \{1,2,3,4,5\}$ |
| $C$ | Structural constraint | kagome, Lieb, honeycomb, etc. |

### DPO Margin

| Symbol | Meaning | Equation Reference |
|--------|---------|-------------------|
| $I_\theta^{(z)}(\bs{x}, t)$ | Improvement score (channel $z$) | Eq. 3.5, 3.7, 3.9 |
| $I_\theta(\bs{x}, t)$ | Total improvement | $\sum_{z \in \{L,F,A\}} I_\theta^{(z)}$ |
| $g_\theta(t)$ | DPO margin | $I_\theta(\bs{x}^w, t) - I_\theta(\bs{x}^\ell, t)$ |
| $d_\theta^{(z)}$ | Denoising error | Channel-specific |
| $\omega_t^{(z)}$ | Channel weight | Noise-schedule dependent |
| $\beta$ | Preference sharpness | Bradley-Terry temperature |
| $\gamma$ | Hölder exponent | Robustness param, default 2.0 |

### SCIGEN Constraints

| Symbol | Meaning | Equation Reference |
|--------|---------|-------------------|
| $\bs{C}^{(z)}$ | Constraint mask | Binary, 1 = constrained |
| $\bar{\bs{C}}^{(z)}$ | Free mask | $1 - \bs{C}^{(z)}$ |
| $\Pi_C(\cdot; C)$ | SCIGEN projection | Eq. 3.20 |
| $\tilde{p}_{\theta,C}$ | Executed constrained kernel | Eq. 3.19 |
| $\Lambda_{\theta,C}(\tau, t)$ | Per-step log-ratio (bridge) | Eq. 3.25 |
| $\Delta_{\theta,C}(\bs{x}_0)$ | Endpoint log-ratio | Eq. 3.22 |

### Robustness

| Symbol | Meaning | Equation Reference |
|--------|---------|-------------------|
| $\ell_\gamma(x)$ | Hölder loss | Eq. 3.15 |
| $\iota_\gamma(x)$ | Influence weight | Eq. 3.31 |
| $\epsilon$ | Outlier proportion | Eq. 3.48 |
| $\xi^\star$ | Clean proportion | $1 - \epsilon$ |

---

## Key Technical Terms

### Denoising Improvement Score
- **Definition**: Difference in denoising error between reference and trainable model
- **Context in paper**: Section 3.2.2, Equations 3.5-3.9
- **Formula**: $I_\theta^{(z)}(\bs{x}, t) = \omega_t^{(z)}(d_{\text{ref}}^{(z)} - d_\theta^{(z)})$
- **Related concepts**: Log-likelihood ratio proxy for diffusion models
- **Usage**: Tractable alternative to intractable $\log p_\theta(\bs{x})$

### Wrapped Difference
- **Definition**: Signed difference on torus respecting periodicity
- **Context in paper**: Equation 2.15
- **Formula**: $\text{wrap}_{\pm}(u) = u - \lfloor u + 1/2 \rfloor \in [-1/2, 1/2)$
- **Related concepts**: Wrapped Normal distribution, periodic boundary conditions
- **Why needed**: Fractional coordinates live on $[0,1)^{N\times3}$ torus

### Hölder Divergence
- **Definition**: Robust alternative to KL divergence
- **Context in paper**: Section 3.2.3, Equation 3.15
- **Property**: Redescending - outlier influence vanishes as disagreement increases
- **Related concepts**: M-estimators, robust statistics, influence functions
- **References**: Fujisawa et al. (2025), NeurIPS

### Constraint Cancellation Lemma
- **Definition**: Under shared SCIGEN constraint, constrained DOF cancel from log-ratio
- **Context in paper**: Lemma 3.1, Equation 3.21
- **Formula**: $\log \frac{\tilde{p}_{\theta,C}}{\tilde{p}_{\text{ref},C}} = \log \frac{p_\theta(\text{free})}{p_{\text{ref}}(\text{free})}$
- **Implication**: Only compute loss on unconstrained components
- **Caveat**: Bridge distribution still depends on SCIGEN dynamics

### Endpoint-to-Bridge Identity
- **Definition**: Endpoint log-ratio as log-expectation of trajectory log-ratios
- **Context in paper**: Proposition 3.2, Equation 3.24
- **Problem**: Exact posterior bridge unavailable (trajectories not stored)
- **Solution**: Reconstruct pseudo-bridge online (Section 3.3)
- **Related concepts**: Path integral, importance sampling

### Redescending Property
- **Definition**: Influence function bounds vanish for extreme outliers
- **Context in paper**: Proposition 3.5, Equation 3.49
- **Formula**: $\lim_{u \to -\infty} ||\text{IF}(s)|| = 0$
- **Interpretation**: Mislabeled pairs have bounded effect on training
- **Benefit**: Robust to crowdsourced noisy labels

---

## Channel-Specific Details

### Lattice Channel (L)
- **Representation**: O(3)-invariant $\bs{k} = (k_1, \ldots, k_6)$
- **Forward**: Standard DDPM (Eq. 2.1)
- **Space group**: Some $k_i$ fixed by symmetry
- **Example**: Hexagonal: $k_1 = -\log(3/4)$, $k_2=k_3=k_4=0$
- **Mask**: $\bs{m}$ indicates free components

### Fractional Coordinate Channel (F)
- **Domain**: Torus $[0,1)^{N \times 3}$
- **Forward**: Wrapped Normal (Eq. 2.3)
- **Reverse proxy**: Wrapped Gaussian (Eq. 2.16)
- **Error**: $||\Delta(\bs{F}_{t-1}, \bs{\mu}_\theta^{(F)})||^2$ with wrapped diff
- **Weight**: $\omega_t^{(F)} = 1/(2\tilde{\sigma}_t^2)$

### Atom Type Channel (A)
- **Representation**: One-hot in $\mathbb{R}^{h \times N}$
- **Forward**: Standard DDPM (Eq. 2.5)
- **Reverse**: Gaussian with predicted mean
- **Discretization**: Argmax at $t=0$

---

## Three-Phase Training

### Phase A: Offline DPO
- **Data**: MP-20 with DFT band structures
- **Context**: Broad buckets (chemistry, size, symmetry)
- **Scale**: 800-1500 materials, 3500-5000 pairs
- **Objective**: General flat band preferences
- **Loss**: Unconstrained H-DPO (Eq. 3.14)

### Phase B: Motif-Focused
- **Data**: SCIGEN-generated + human preferences
- **Context**: Tight $(C, N, \mathcal{A}^c)$
- **Scale**: 300-800 per motif, 1500-4200 pairs
- **Objective**: Within-constraint ranking
- **Loss**: Bridge H-DPO (Eq. 3.43)
- **LR**: Lower ($10^{-5}$ vs $10^{-4}$)

### Phase C: Active Learning
- **Strategy**: Query uncertain/high-value pairs
- **Acquisition**: Uncertainty + proxy disagreement + novelty
- **Scale**: 50-100/round × 5 rounds = 250-500 pairs

---

## Implementation Notes

### Confidence Score
- **NOT used for gradient weighting** (Hölder already robust)
- **Used for**: Diagnostics, outlier enrichment, tuning $\gamma$

### Bridge Reconstruction
1. Sample $b \sim \rho(\cdot)$
2. Forward noise to $\bs{x}_b$
3. Run frozen ref + SCIGEN for $b$ steps
4. Pseudo-bridge $\hat{\tau}$
5. Compute improvement on $\hat{\tau}$ states

### Normalized Errors
- **Why**: Different channel dimensionality
- **Formula**: $d^{(z)} = ||r^{(z)}||^2 / (||\bar{\bs{C}}^{(z)}||_1 + \epsilon_0)$

---

## Notes

This paper provides the mathematical foundation for combining:
1. Multi-channel crystal diffusion (DiffCSP++)
2. Human preference learning (DPO)
3. Structural constraints (SCIGEN)
4. Robust training (Hölder divergence)
5. Endpoint-only data (bridge reconstruction)

Key innovation: Phase B formulation that correctly handles SCIGEN constraints with endpoint-conditioned bridge sampling.
