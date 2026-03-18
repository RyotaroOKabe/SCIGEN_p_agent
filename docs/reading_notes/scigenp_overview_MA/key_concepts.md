# Key Concepts - scigenp_overview_MA

> Reading notes for: `/pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview_MA/material_dpo.tex`
> Last updated: 2026-03-16

This document captures the main ideas and methodology from the mathematical formulation of SCIGEN+ with DPO.

---

## Main Research Question

**How do we train a crystal diffusion model to generate materials with high-quality flat bands using human preference data, while respecting geometric structural constraints (SCIGEN)?**

Key challenges:
1. Flat band quality is subjective and multidimensional (flatness + proximity to E_F + isolation)
2. Diffusion models have intractable likelihoods
3. SCIGEN constraints modify the generation process
4. Preference data may be noisy (crowdsourced annotations)
5. Phase B: Only endpoint crystals stored, not full SCIGEN trajectories

---

## Core Methodology: Hölder-Diffusion-DPO

### Overview

Adapt Direct Preference Optimization (DPO) from language models to crystal diffusion models with three key extensions:

1. **Multi-channel formulation**: Handle lattice (DDPM), fractional coords (wrapped Gaussian), atom types (DDPM)
2. **SCIGEN integration**: Respect structural constraints via masked denoising errors
3. **Hölder robustness**: Handle noisy crowdsourced labels via redescending loss

### Why DPO Instead of RLHF?

Traditional RLHF:
```
Preference data → Train reward model → RL to maximize reward
```

DPO:
```
Preference data → Directly optimize policy
```

**Benefits**:
- No separate reward model training
- More stable (no RL instabilities)
- Direct log-likelihood ratio optimization
- Mathematical equivalence under KL-regularized RL

---

## Section 2: Per-Channel Crystal Diffusion

### Key Insight

Crystals have heterogeneous structure: $\bs{x}_0 = (\bs{L}, \bs{F}, \bs{A})$

Each channel needs different diffusion process:
- **Lattice**: Euclidean space with symmetry constraints
- **Fractional coords**: Torus (periodic boundaries)
- **Atom types**: Discrete → continuous embedding

### Lattice Channel: O(3)-Invariant Representation

**Problem**: Lattice matrix $\bs{L}$ is not rotationally invariant

**Solution**: Use $\bs{L}^\top\bs{L} = \exp(2\sum_i k_i \bs{B}_i)$ decomposition
- Coefficients $\bs{k} = (k_1, \ldots, k_6)$ are O(3)-invariant
- Space groups fix certain $k_i$ (mask $\bs{m}$)
- Diffuse only free components

**Example** (Hexagonal):
- $k_1 = -\log(3/4)$ (fixed)
- $k_2 = k_3 = k_4 = 0$ (fixed)
- $k_5, k_6$ free

### Fractional Coordinate Channel: Wrapped Normal

**Problem**: $\bs{F} \in [0,1)^{N\times3}$ lives on torus, not Euclidean space

**Solution**: Wrapped Normal distribution
- Forward: $\bs{F}_t = w(\bs{F}_0 + \sigma_t \boldsymbol{\varepsilon})$ where $w(\cdot)$ wraps to $[0,1)$
- Distance: Use wrapped difference $\Delta(\bs{F}, \bs{G})$ not naive subtraction
- Score matching instead of noise prediction

**Why it matters**: Near boundaries (e.g., 0.01 vs 0.99), Euclidean distance is wrong

---

## Section 3: Diffusion-DPO for Crystals

### Phase A: Unconstrained Forward-Proxy DPO

**Setting**: No SCIGEN constraints during training, broad context buckets

**Key Idea**: Use denoising improvement as likelihood proxy

For DDPM channels (lattice, atom types):
```
log p_θ(x_{t-1}|x_t) / p_ref(x_{t-1}|x_t)
≈ -ω_t [||ε - ε_θ||² - ||ε - ε_ref||²]
```

Define improvement:
```
I_θ(x, t) = ω_t (d_ref - d_θ)
```

Aggregate across channels:
```
I_θ(x, t) = I_θ^(L)(x,t) + I_θ^(F)(x,t) + I_θ^(A)(x,t)
```

DPO margin:
```
g_θ(t) = I_θ(x^w, t) - I_θ(x^ℓ, t)
```

Standard Diffusion-DPO loss:
```
L = -E[log σ(β·T·g_θ(t))]
```

**Hölder robustness**:
```
L_H-DPO = E[ℓ_γ(β·T·g_θ(t))]
where ℓ_γ(x) = -(1+γ)σ(x)^γ + γσ(x)^(1+γ)
```

### Phase B: Constrained Reverse-Policy DPO (Bridge)

**Challenge**: With SCIGEN, the executed sampling process is:
```
propose u_{t-1} ~ p_θ(·|x_t)
execute x_{t-1} = Π_C(u_{t-1})  ← SCIGEN overwrite
```

The aligned object is $\tilde{p}_{\theta,C}$ (executed policy), not just $p_\theta$

**Constraint Cancellation Lemma** (3.1):
- Constrained DOF cancel from ratio: $\log(\tilde{p}_{\theta,C} / \tilde{p}_{ref,C}) = \log(p_\theta(free) / p_{ref}(free))$
- **But**: The bridge distribution $q_C^\star(\tau|\bs{x}_0)$ still depends on SCIGEN

**Problem**: Phase B dataset has only endpoints $(\bs{x}_0^w, \bs{x}_0^\ell)$, not trajectories

**Endpoint-to-Bridge Identity** (Prop 3.2):
```
Δ_{θ,C}(x_0) = log E_{τ~q*_C}[exp(Σ_t Λ_{θ,C}(τ,t))]
```

where $\Lambda$ is per-step log-ratio and $q_C^\star$ is exact posterior bridge

**Solution**: Reconstruct pseudo-bridge online
1. Sample bridge level $b$
2. Forward noise: $\bs{x}_b \sim q(\bs{x}_b | \bs{x}_0)$
3. Reverse rollout: Run frozen ref + SCIGEN for $b$ steps → $\hat{\tau}$
4. Compute improvement on $\hat{\tau}$ states

**Masked denoising errors**:
```
d_θ^{(z),BR} = ||C̄^{(z)} ⊙ (x̂_{t-1} - μ_θ)||² / (||C̄^{(z)}||_1 + ε_0)
```

Only compute loss on free components $\bar{\bs{C}}^{(z)} = 1 - \bs{C}^{(z)}$

---

## Robustness: Hölder Divergence

### Why Not Standard DPO?

Standard DPO loss (based on KL divergence) is **non-robust**:
- Single mislabeled pair can arbitrarily distort the solution
- Outlier influence doesn't vanish

### Hölder Loss Properties

**Redescending** (Prop 3.5):
```
lim_{u→-∞} ||IF(s)|| = 0
```

When model strongly disagrees with a label (u → -∞), influence vanishes

**Influence weight**:
```
ι_γ(x) = σ(x)^γ (1-σ(x))²
```

- Clean pairs (model agrees): high influence
- Outliers (model disagrees): low influence

### Outlier Detection

**Extended model** with scaling parameter $\xi$:
```
m_η(s) = ξ · p_θ(s)
```

Optimal $\xi^\star$ estimates clean proportion:
```
ξ^\star ≈ 1 - ε
```

where $\ε$ is outlier proportion

**Normalized estimator** (Prop 3.6):
```
ξ̂ = (1/N) · (Σ p̄_i^γ) / (Σ p̄_i^{1+γ})
```

where $\bar{p}_i = p_i / \sum_j p_j$

**Outlier set**:
```
Ô_flip = bottom ⌈ε̂·N⌉ samples by p_θ(s)
```

Flags potentially mislabeled pairs for review

---

## Three-Phase Training Pipeline

### Phase A: Offline DPO on MP-20

**Objective**: Learn general flat band preferences from existing database

**Data construction**:
1. Query MP-20 for materials with band structures (~45k → subset with DFT bands)
2. Compute flatness score: $s(\bs{x}) = \lambda_1 s_{\text{flat}} + \lambda_2 s_{\text{iso}} + \lambda_3 s_{E_F}$
3. Within context buckets, create pairs with sufficient score gap
4. Supplement with human labels for ambiguous cases

**Context**: $(c_A)$ = (chemistry family, N-bucket, symmetry bucket)
- Broad grouping to learn transferable preferences
- Winner/loser share same $c_A$

**Scale**: 800-1500 materials, 3500-5000 total pairs

**Training**: Unconstrained H-DPO (no SCIGEN masks)

**Expected outcome**: Model biased toward flatter, better-isolated bands near E_F

### Phase B: Motif-Focused Preference Adaptation

**Objective**: Specialize for specific motifs with human preferences on generated structures

**Data construction**:
1. Generate with Phase-A model + SCIGEN (various $C, N, \mathcal{A}^c$ settings)
2. Screen: SMACT + CHGNet/M3GNet → retain 80-90% stable + 10-20% borderline
3. Compute band structures (DFT or surrogate)
4. Within $c_B$ buckets, select pairs and collect human judgments

**Context**: $(c_B) = (C, N, \mathcal{A}^c)$
- Tight alignment with SCIGEN control variables
- Winner/loser share same constraint $C$

**Scale per motif**: 300-800 candidates, 1500-4200 pairs

**Training**: Bridge H-DPO with SCIGEN-masked errors
- Reconstruct pseudo-bridges online
- Compute loss only on free DOF
- Lower learning rate ($10^{-5}$ vs $10^{-4}$)

**Expected outcome**: Model learns which unconstrained completions yield better flat bands within constraint

### Phase C: Active Learning Loop

**Objective**: Iteratively improve via strategic expert queries

**Acquisition strategy**:
```
acquire(x_i, x_j) = α·uncertainty + β·proxy_disagreement + γ·novelty
```

where:
- **Uncertainty**: $|I_\theta(\bs{x}_i, t) - I_\theta(\bs{x}_j, t)|$ small
- **Proxy disagreement**: Flatness score and model disagree on winner
- **Novelty**: Under-explored composition/structure space
- **Anomaly**: High Hölder anomaly score (potential Level-3 discoveries)

**Workflow**:
1. Generate batch with Phase-B model + SCIGEN
2. Screen for stability, compute bands + scores
3. Select high-value pairs for annotation
4. (Optional) Async expert query with "hot pool"
5. Retrain and iterate

**Scale**: 5 rounds × 50-100 pairs/round = 250-500 total

**Expected outcome**: Efficient expert time use, improvement where uncertainty highest

---

## Integration: Two Options

### Option A: Offline DPO, Constraints at Inference Only

- Train on unconstrained pairs: $(c, \bs{x}^w, \bs{x}^\ell, \kappa)$
- SCIGEN applied only at generation time
- **Learns**: General flat-band prior that transfers via inference masking

### Option B: Constrained DPO (Phase B approach)

- Train on constrained pairs: $(c, C, \bs{x}^w, \bs{x}^\ell, \kappa)$
- Both winner/loser share constraint $C$
- Masked improvement scores during training
- **Learns**: Motif-specific preferences for unconstrained completions

**Chosen approach**: Hybrid
- Phase A uses Option A (broad prior)
- Phase B uses Option B (motif-specific)

---

## Key Algorithmic Innovations

### 1. Multi-Channel Aggregation
- Respect heterogeneous crystal structure
- Channel-specific noise schedules and error metrics
- Wrapped difference for periodic coordinates

### 2. Constraint Cancellation
- Fixed DOF cancel from log-ratio
- Compute loss only on free components
- Reduces computational cost + avoids spurious gradients

### 3. Pseudo-Bridge Reconstruction
- Handle endpoint-only data
- Online rollout via frozen reference
- Bridge level sampling for cost/quality tradeoff

### 4. Normalized Masked Errors
- Account for varying free DOF counts
- Prevents domination by high-dimensional channels
- Numerical stability ($\epsilon_0$ regularization)

### 5. Confidence-Free Hölder Training
- Don't weight by $\kappa$ during training
- Use $\kappa$ only for diagnostics and $\gamma$ tuning
- Avoid introducing annotator-confidence bias

---

## Comparison with Related Work

### vs. Standard Diffusion-DPO (Wallace et al. 2024)
- **Theirs**: Single-channel images, unconstrained
- **Ours**: Multi-channel crystals, SCIGEN constraints, bridge formulation

### vs. RLHF for Diffusion
- **RLHF**: Train reward model → RL optimization (unstable)
- **DPO**: Direct policy optimization (stable, no reward model)

### vs. Score-Based Fine-Tuning
- **Score matching**: Requires ground-truth score function
- **DPO**: Only needs pairwise preferences (weaker signal, easier to collect)

### vs. Original SCIGEN
- **SCIGEN**: Constraint-guided generation, no preference learning
- **SCIGEN+**: Adds human feedback loop for electronic properties

---

## Expected Outcomes & Evaluation

### Structural Validity
- SMACT charge balance pass rate
- CHGNet/M3GNet stability scores
- Formation energy distribution

### Flat Band Yield
- **Flatness**: Max bandwidth $W$ within $E_F \pm 0.5$ eV
- **Isolation**: Min gap to adjacent bands
- **Combined metric**: Weighted combination

### Novelty & Diversity
- Embed in structure latent space
- Cluster analysis vs. known databases (2DMatPedia, MP-20)
- Novel = beyond threshold distance from all known entries

### Hypotheses
1. **Constraints necessary but not sufficient**: SCIGEN alone improves yield, but not enough
2. **Preference learning enhances yield**: DPO-SCIGEN+ achieves highest yield
3. **Novel families discovered**: Under-explored compositions satisfying both geometry + electronics

---

## Open Questions & Future Directions

### From Discussion Section (Section 5)

**Strategic questions**:
- Does constraint + preference approach generalize to other properties?
- What alternative evaluation strategies beyond pairwise?

**Target extensions**:
- Dirac/Weyl semimetals (linearity, symmetry protection)
- Topological insulators (band inversions, orbital character)
- Van Hove singularities (saddle points, nesting)
- Breathing kagome for topological flat bands

**Success criteria**:
- Higher flat band yield vs. baselines?
- At least one novel structural family?
- Candidate compelling for experimental synthesis?

### Technical questions
- Optimal bridge level distribution $\rho(b)$?
- Multi-objective acquisition for active learning?
- Human-AI collaborative workflows?
- Integration with large language models for natural language specification?

---

## Summary: What Makes This Work Novel?

1. **First DPO for crystal structures** with tractable multi-channel formulation
2. **Bridge formulation** for endpoint-only preference data under constraints
3. **Hölder robustness** for crowdsourced noisy labels
4. **Three-phase curriculum**: Broad → motif-specific → active learning
5. **Mathematical rigor**: Lemmas + propositions proving correctness of approximations

**Core contribution**: Enables human expert judgment to guide generative models toward scientifically interesting materials without requiring explicit, hand-coded evaluation functions.
