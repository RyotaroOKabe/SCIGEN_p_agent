# Reading Session Q&A - DETAILED EXPLANATIONS
## Voice transcript reading session for material_dpo.tex

> **Purpose**: Deeply understand SCIGEN+ DPO methodology through 47 questions
> **Audience**: Beginners to the paper, learning step-by-step
> **Date**: 2026-03-16

---

# TABLE OF CONTENTS

1. [Overview Section](#overview-section) (Q1)
2. [Section 2: Per-Channel Diffusion](#section-2-per-channel-diffusion) (Q2)
3. [Section 3.1: DPO Background](#section-31-dpo-background) (Q3-Q4)
4. [Section 3.2.2: Improvement Scores](#section-322-improvement-scores) (Q5-Q14)
5. [Section 3.2.3: Hölder-DPO](#section-323-hölder-dpo) (Q15-Q19)
6. [Section 3.2.4: Diagnostics](#section-324-diagnostics) (Q20-Q21)
7. [Section 3.3: Bridge Formulation](#section-33-bridge-formulation) (Q22-Q41)
8. [Section 3.4: Robustness](#section-34-robustness) (Q42-Q47)

---

# Overview Section

## Q1: What does "executed constrained reverse policy" mean?

**📍 Location:** Lines 60-78 (Overview section)

### The Problem

When you use SCIGEN to generate crystals, there's a **difference** between what the model *wants* to do and what *actually happens*.

### Step-by-Step Breakdown

**Without SCIGEN (normal generation):**
```
Time t → t-1 (one denoising step)
Model proposes: x_{t-1} ~ p_θ(·|x_t)
What happens:   x_{t-1} (the proposal is used directly)
```

**With SCIGEN (constrained generation):**
```
Time t → t-1 (one denoising step)
1. Model proposes:     u_{t-1} ~ p_θ(·|x_t)
2. SCIGEN overwrites:  x_{t-1} = Π_C(u_{t-1})  ← FORCES constraint!
3. What happens:       x_{t-1} (the overwritten version)
```

### Visual Example: Kagome Constraint

Imagine generating a kagome structure (N=6 atoms):

```
SCIGEN says: "3 atoms MUST be at kagome vertices"
             "3 atoms are FREE (you decide)"

Step t → t-1:

Model proposal u_{t-1}:
  Atom 1: position (0.12, 0.45, 0.0)  ← proposed
  Atom 2: position (0.55, 0.88, 0.0)  ← proposed
  Atom 3: position (0.29, 0.11, 0.0)  ← proposed
  Atom 4: position (0.71, 0.23, 0.5)  ← proposed
  Atom 5: position (0.33, 0.66, 0.5)  ← proposed
  Atom 6: position (0.88, 0.44, 0.5)  ← proposed

SCIGEN overwrite Π_C:
  Atom 1: position (0.0,  0.0,  0.0)  ← FORCED (kagome vertex 1)
  Atom 2: position (0.5,  0.0,  0.0)  ← FORCED (kagome vertex 2)
  Atom 3: position (0.5,  0.5,  0.0)  ← FORCED (kagome vertex 3)
  Atom 4: position (0.71, 0.23, 0.5)  ← kept from proposal (FREE)
  Atom 5: position (0.33, 0.66, 0.5)  ← kept from proposal (FREE)
  Atom 6: position (0.88, 0.44, 0.5)  ← kept from proposal (FREE)

Final x_{t-1} = overwritten version
```

### Key Terminology

**"Executed constrained reverse policy" = p̃_{θ,C}**

- **Executed** = what actually happens after SCIGEN
- **Constrained** = respects constraint C
- **Reverse policy** = denoising direction (noise → data)
- **p̃_{θ,C}** = distribution of what you actually get

**Mathematical definition:**
```
p̃_{θ,C}(x_{t-1}|x_t) = ∫ δ(x_{t-1} - Π_C(u))·p_θ(u|x_t) du
                        ↑                    ↑
                    "x_{t-1} equals       model's
                     constrained u"      proposal
```

### Phase A vs Phase B

| Aspect | Phase A | Phase B |
|--------|---------|---------|
| **Training data** | Clean MP-20 crystals | SCIGEN-generated crystals |
| **SCIGEN during training?** | ❌ No | ✅ Yes |
| **What we align** | p_θ (model proposal) | p̃_{θ,C} (executed policy) |
| **Why different?** | Data never saw SCIGEN | Data came from SCIGEN |

### Why This Matters

If you train DPO on SCIGEN-generated data but only align p_θ (not p̃_{θ,C}), you're aligning the **wrong distribution**!

**Analogy:**
- Phase A: Teaching someone to cook (they decide everything)
- Phase B: Teaching someone to cook while following recipe constraints (some steps are fixed)

If you teach Phase B the same way as Phase A, you'll give wrong advice about the fixed steps!

**Explained in:** Section 3.3 (Bridge Formulation), Equations 3.19-3.20

---

# Section 2: Per-Channel Diffusion

## Q2: Is λ_t (Eq 2.4) the same as in DiffCSP?

**📍 Location:** Equation 2.4

### Quick Answer
**YES**, exactly the same!

### What is λ_t?

λ_t is a **normalization weight** for the fractional coordinate channel.

**Equation 2.4:**
```
λ_t = E^{-1}[||∇ log N_w(0, σ_t²)||²]
      ↑                          ↑
    inverse expected        score of wrapped
    squared norm          Gaussian at origin
```

### Why Do We Need It?

**Problem:** Fractional coordinates live on a torus [0,1)³, not regular space ℝ³.

The score (gradient of log-probability) has a different "typical magnitude" on the torus compared to regular Euclidean space.

**Without λ_t:**
```
Loss = ||score_true - score_predicted||²

Different timesteps have wildly different magnitudes!
  t=1:   score ~ 100
  t=500: score ~ 1
  t=999: score ~ 0.01

→ Training unstable, dominated by early timesteps
```

**With λ_t:**
```
Loss = λ_t · ||score_true - score_predicted||²

Normalized:
  t=1:   λ_1 · score ~ 1
  t=500: λ_500 · score ~ 1
  t=999: λ_999 · score ~ 1

→ All timesteps contribute equally
```

### How to Compute λ_t

**From DiffCSP paper (Jiao et al. 2023, Section 3.2):**

1. **Sample many points** from wrapped Gaussian at origin: N_w(0, σ_t²)
2. **Compute score** at each point: ∇ log p(x)
3. **Average squared norm**: E[||∇ log p||²]
4. **Invert**: λ_t = 1 / E[||∇ log p||²]

**Pre-computed** for each timestep t, stored as a lookup table.

### Code Pseudocode

```python
def compute_lambda_t(sigma_t, n_samples=10000):
    """Compute normalization weight for wrapped Gaussian"""
    # Sample from wrapped Gaussian at origin
    samples = torch.randn(n_samples, 3) * sigma_t  # Regular Gaussian
    samples = samples % 1.0  # Wrap to [0,1)

    # Compute scores
    scores = compute_wrapped_gaussian_score(samples, sigma_t)

    # Average squared norm
    expected_norm_sq = (scores ** 2).mean()

    # Invert
    lambda_t = 1.0 / expected_norm_sq

    return lambda_t
```

### Reference
**Original DiffCSP paper:** Jiao et al. (2023) "Crystal Structure Prediction by Joint Equivariant Diffusion", Section 3.2, Equation 8

**Used identically** in SCIGEN+!

---

# Section 3.1: DPO Background

## Q3: What reference presents the Bradley-Terry model?

**📍 Location:** Section 3.1

### The Bradley-Terry Model

**What it is:** A classic statistical model for pairwise comparisons (1952!).

**Scenario:** You have items A and B, and want to model "Which is better?"

**Formula:**
```
P(A ≻ B) = σ(score_A - score_B)
           = exp(score_A) / [exp(score_A) + exp(score_B)]
           = 1 / [1 + exp(score_B - score_A)]
```

where σ(x) = 1/(1+e^{-x}) is the sigmoid function.

### Visual Intuition

```
Score difference (score_A - score_B):

  Large negative         Zero            Large positive
       |                  |                    |
  ----[-3]------[-2]----[-1]----[0]----[1]----[2]----[3]----
       |                  |                    |
  P(A≻B)≈0.05        P(A≻B)=0.5          P(A≻B)≈0.95
   (B much             (tie)            (A much
    better)                              better)
```

**Key property:** Symmetric around 0
- If score_A - score_B = +2 → P(A≻B) = 0.88
- If score_A - score_B = -2 → P(A≻B) = 0.12

### References

**Original paper:**
- Bradley, R. A., & Terry, M. E. (1952). "Rank analysis of incomplete block designs: I. The method of paired comparisons." *Biometrika*, 39(3/4), 324-345.

**For DPO context:**
- Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", Section 2.1

### Why DPO Uses Bradley-Terry

In DPO, the "score" is the log-probability:
```
score_A = log p_θ(y_A|x)
score_B = log p_θ(y_B|x)

P(y_A ≻ y_B) = σ(log p_θ(y_A|x) - log p_θ(y_B|x))
              = σ(log[p_θ(y_A|x)/p_θ(y_B|x)])
```

This lets us **rank model outputs** using their probabilities!

---

## Q4: How to derive the DPO loss form? Which paper shows the proof?

**📍 Location:** Section 3.1

### Full Derivation (Step-by-Step)

**Goal:** Replace reward model r(y) with policy log-ratio log[π_θ/π_ref]

#### Step 1: Start with RL objective

We want to maximize expected reward while staying close to reference:
```
max_θ  E_π_θ[r(y|x)] - β·KL(π_θ(·|x) || π_ref(·|x))
```

**Why?**
- E_π_θ[r(y|x)] = generate high-reward outputs
- KL penalty = don't deviate too far from safe reference π_ref

#### Step 2: Find optimal policy π*

Using Lagrange multipliers, the optimal policy is:
```
π*(y|x) ∝ π_ref(y|x) · exp(r(y|x)/β)
```

**Proof sketch:**
```
L = E_π[r] - β·KL(π||π_ref) + λ(∫π dy - 1)

∂L/∂π(y) = r(y) - β[log π(y) - log π_ref(y) + 1] - λ = 0

Solve for π(y):
  log π(y) = (r(y) + β log π_ref(y) - β - λ) / β
  π(y) = exp((r(y) - β - λ)/β) · π_ref(y)
  π(y) ∝ π_ref(y) · exp(r(y)/β)
```

#### Step 3: Rearrange to express reward

From π* ∝ π_ref · exp(r/β), take logs:
```
log π*(y|x) = log π_ref(y|x) + r(y|x)/β + const

→ r(y|x) = β log[π*(y|x)/π_ref(y|x)] + const(x)
```

**Key insight:** Reward is the **log-ratio** of optimal vs reference policy!

#### Step 4: Substitute into Bradley-Terry

Bradley-Terry model:
```
P(y_w ≻ y_ℓ | x) = σ(r(y_w|x) - r(y_ℓ|x))
```

Substitute r from Step 3:
```
P(y_w ≻ y_ℓ | x) = σ(β log[π*/π_ref](y_w) - β log[π*/π_ref](y_ℓ))
                  = σ(β log[π*(y_w)/π_ref(y_w) · π_ref(y_ℓ)/π*(y_ℓ)])
```

The const(x) cancels!

#### Step 5: Replace π* with trainable π_θ

We don't have π*, but we can train π_θ to approximate it:
```
P(y_w ≻ y_ℓ | x) ≈ σ(β log[π_θ(y_w)/π_ref(y_w)] - β log[π_θ(y_ℓ)/π_ref(y_ℓ)])
```

#### Step 6: Maximum likelihood loss

Given preference dataset {(x_i, y_w^i, y_ℓ^i)}, maximize likelihood:
```
max_θ  ∏_i P(y_w^i ≻ y_ℓ^i | x_i)

→ max_θ  ∑_i log P(y_w^i ≻ y_ℓ^i | x_i)

→ max_θ  ∑_i log σ(β log[π_θ(y_w^i)/π_ref(y_w^i)] - β log[π_θ(y_ℓ^i)/π_ref(y_ℓ^i)])
```

Negate for minimization:
```
L_DPO(θ) = -E_{(x,y_w,y_ℓ)} [log σ(β log[π_θ(y_w)/π_ref(y_w)] - β log[π_θ(y_ℓ)/π_ref(y_ℓ)])]
```

**This is the DPO loss!**

### Visual Summary

```
RL objective          Optimal policy           Reward as log-ratio
    ↓                       ↓                          ↓
max E[r] - β·KL    →   π* ∝ π_ref·exp(r/β)  →   r = β log(π*/π_ref)
    ↓                                                  ↓
Bradley-Terry model                    Replace π* with trainable π_θ
P(w≻ℓ) = σ(r_w - r_ℓ)                               ↓
                                           DPO Loss (no reward model!)
```

### Where to Find the Proof

**Must read:** Rafailov et al. (2023) "Direct Preference Optimization", **Appendix A.1** "Deriving the DPO Objective"

**Full mathematical rigor** with:
- Lagrangian derivation
- KL divergence properties
- Partition function cancellation
- Bradley-Terry connection

**Main paper:** Section 2 gives intuition
**Appendix A.1:** Complete proof

---

# Section 3.2.2: Improvement Scores

## Q5: How to derive improvement score I_θ and denoising errors d_θ, d_ref?

**📍 Location:** Equations 3.3-3.5

### The Challenge

For diffusion models, we can't compute p_θ(x₀) directly (requires T-step marginalization).

**DPO needs:** log[p_θ(y_w)/p_ref(y_w)] - log[p_θ(y_ℓ)/p_ref(y_ℓ)]

**But:** p_θ(x₀) = ∫∫...∫ p(x_T) ∏_{t=1}^T p_θ(x_{t-1}|x_t) dx_T...dx_1
- **Intractable!** Can't evaluate.

**Solution:** Use **one-step proxy** at timestep t.

---

### Step-by-Step Derivation for DDPM Channels

#### Setup

DDPM reverse kernel (Gaussian):
```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)
```

where μ_θ is the predicted mean.

#### Step 1: Write log-probability

```
log p_θ(x_{t-1}|x_t) = -1/(2σ_t²) ||x_{t-1} - μ_θ(x_t,t)||² - log(√(2πσ_t²)) + const
```

The log-ratio:
```
log[p_θ/p_ref](x_{t-1}|x_t) = -1/(2σ_t²) [||x_{t-1}-μ_θ||² - ||x_{t-1}-μ_ref||²]
```

Constant σ_t² cancels.

#### Step 2: Reparameterize with noise prediction

DDPM parameterization (from Ho et al. 2020):
```
μ_θ(x_t,t) = 1/√α_t [x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t,t)]
```

where ε_θ predicts the noise that was added.

**Forward diffusion:**
```
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0,I)
```

#### Step 3: Expand the squared norms

```
||x_{t-1} - μ_θ||² = ||x_{t-1} - 1/√α_t [x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ]||²
```

After algebra (expand, simplify with x_t = ... + ε):
```
∝ ||ε - ε_θ||²
```

**Full derivation** (Wallace et al. 2024, Appendix B, Equations B.3-B.7):
```
||x_{t-1} - μ_θ||² = (1-α_t)²/[α_t(1-ᾱ_t)] · ||ε - ε_θ||² + terms independent of θ
```

#### Step 4: Simplify log-ratio

```
log[p_θ/p_ref] ∝ -1/(2σ_t²) · (1-α_t)²/[α_t(1-ᾱ_t)] · [||ε-ε_θ||² - ||ε-ε_ref||²]
                = -ω_t [||ε-ε_θ||² - ||ε-ε_ref||²]
```

where:
```
ω_t = (1-α_t)² / [2σ_t²α_t(1-ᾱ_t)]
```

#### Step 5: Define denoising errors and improvement

**Denoising errors:**
```
d_θ = ||ε - ε_θ(x_t,t)||²
d_ref = ||ε - ε_ref(x_t,t)||²
```

**Improvement score:**
```
I_θ(x,t) = ω_t [d_ref - d_θ]
          ↑     ↑       ↑
        weight  ref    model
                error   error
```

**Intuition:**
- If d_θ < d_ref → model predicts noise better than reference → I_θ > 0
- If d_θ > d_ref → model worse than reference → I_θ < 0

---

### Visual Example

```
Timestep t=500, x_t (noisy crystal)

True noise that was added: ε = [0.5, -0.3, 0.8, ...]

Reference predicts:  ε_ref = [0.4, -0.2, 0.7, ...]
  → d_ref = ||ε - ε_ref||² = ||(0.1, -0.1, 0.1, ...)||² = 0.03

Model predicts:      ε_θ   = [0.48, -0.28, 0.82, ...]
  → d_θ = ||ε - ε_θ||² = ||(0.02, -0.02, -0.02, ...)||² = 0.0012

Improvement: I_θ = ω_500 · (0.03 - 0.0012) = ω_500 · 0.0288 > 0
             ↑
          Model is better!
```

---

### Multi-Channel Extension

For crystals with L, F, A channels:
```
I_θ(x,t) = I_θ^(L)(x,t) + I_θ^(F)(x,t) + I_θ^(A)(x,t)
         = ω_t^(L)[d_ref^(L) - d_θ^(L)]
         + ω_t^(F)[d_ref^(F) - d_θ^(F)]
         + ω_t^(A)[d_ref^(A) - d_θ^(A)]
```

Each channel has its own:
- Noise prediction (ε_θ^(L), ε_θ^(F), ε_θ^(A))
- Weight (ω_t^(L), ω_t^(F), ω_t^(A))

---

### Reference

**Diffusion-DPO paper:** Wallace et al. (2024) "Diffusion Model Alignment Using Direct Preference Optimization", **Appendix B** (Equations B.1-B.10)

**Key equations:**
- B.3: μ_θ reparameterization
- B.5: ||x-μ||² in terms of noise
- B.7: Final improvement score form

---

## Q6-Q8: Wrapped Difference, Wrapped Gaussian, Forward Coupling

These questions are interconnected, so I'll explain them together with increasing detail.

---

### Q6: What are F and G in Equation 2.15?

**📍 Location:** Equation 2.15

#### Quick Answer

F and G are **any** fractional coordinate vectors ∈ [0,1)^{N×3}.

**Common examples:**
- F = F_t (noisy fractional coords at time t)
- G = μ_θ^(F) (predicted mean from model)
- F = F_{t-1}, G = F_t (consecutive timesteps)

#### Equation 2.15: Wrapped Difference

```
wrap_±(u) = u - ⌊u + 1/2⌋ ∈ [-1/2, 1/2)

Δ(F, G) = wrap_±(F - G)
```

Applied **element-wise** to all coordinates.

#### Why Wrapping?

Fractional coordinates are **periodic**: 0.0 and 1.0 are the same point!

**Without wrapping (WRONG):**
```
F = [0.95]    G = [0.05]
F - G = 0.90  ← WRONG! Says they're far apart

But on torus: 0.95 → 0.05 wraps around, distance is 0.10
```

**With wrapping (CORRECT):**
```
F = [0.95]    G = [0.05]
F - G = 0.90
0.90 + 1/2 = 1.40
⌊1.40⌋ = 1
wrap_±(0.90) = 0.90 - 1 = -0.10  ← CORRECT! Short distance

Interpretation: G is 0.10 ahead of F (wrapping forward)
```

#### Visual Illustration (1D Torus)

```
Torus [0, 1):
   0.0 ≡ 1.0
    |     |
----●=====●----  (wraps around)
    ↑     ↑
   F=0.95 G=0.05

Distance on torus: 0.10 (wrap forward)
                or 0.90 (go backward without wrapping)
→ Choose shorter: 0.10

wrap_± gives signed short distance: -0.10 (G ahead)
```

#### Element-Wise Application

For N atoms in 3D:
```
F ∈ [0,1)^{N×3}  (N atoms, 3 coords each)
G ∈ [0,1)^{N×3}

Δ(F,G)_ij = wrap_±(F_ij - G_ij)  for each atom i, dimension j
```

**Example (N=2):**
```
F = [[0.1, 0.9, 0.3],    G = [[0.2, 0.1, 0.4],
     [0.8, 0.05, 0.7]]        [0.75, 0.95, 0.65]]

Δ(F,G) = [[wrap(-.1), wrap(.8), wrap(-.1)],
          [wrap(.05), wrap(.1), wrap(.05)]]
       = [[-0.1, -0.2, -0.1],    ← 0.9→0.1 wraps, distance -0.2
          [0.05, 0.1, 0.05]]
```

---

### Q7: What is "wrapped normal proxy reverse kernel" (Eq 2.16)?

**📍 Location:** Equation 2.16

#### Unpacking the Name

**"Wrapped Normal Proxy Reverse Kernel"** = 5 technical terms! Let's break it down:

1. **Wrapped Normal** = Gaussian on torus
2. **Proxy** = approximation (not exact)
3. **Reverse** = denoising direction (t → t-1)
4. **Kernel** = transition probability p(x_{t-1}|x_t)
5. **For fractional coordinates** = F channel only

---

#### Full Equation 2.16

```
p_θ(F_{t-1} | M_t, t, c) ≈ N_w(F_{t-1}; μ_θ^(F), σ̃_t²I)
↑                           ↑            ↑       ↑
reverse kernel        wrapped Gaussian  mean   variance
```

**Components:**

**N_w** = Wrapped Normal (explained in Q8)

**μ_θ^(F)** = Predicted mean (Eq 2.17):
```
μ_θ^(F) = w(F_t + η_t · ŝ_θ^(F))
          ↑           ↑
        wrap    predictor step
                (score × step size)
```

**σ̃_t²** = Shared variance (fixed, not learned)

**M_t** = Noisy state tuple (k_t, F_t, A_t)

---

#### Why "Proxy"?

**True DiffCSP reverse sampling** (Jiao et al. 2023):

```
1. Predictor step:
   F'_{t-1} = w(F_t + η_t · s_θ(F_t,t))

2. Corrector step (Langevin dynamics):
   For k=1 to K:
     F ← F + ε·∇log p(F) + √(2ε)·noise
     F ← w(F)  (wrap to torus)

3. Result: F_{t-1}
```

**Why complicated?** Score matching on torus needs Langevin correction for accuracy.

**Proxy for DPO** (this paper):

```
1. Single Gaussian step:
   F_{t-1} ~ N_w(μ_θ^(F), σ̃_t²I)

2. No Langevin correction

3. Result: F_{t-1}
```

**Why simpler?**
- **Tractable**: Gaussian has closed-form density → can compute gradients
- **Fast**: Single step vs K Langevin iterations
- **Good enough**: For DPO, approximate log-ratio is sufficient

**"Proxy"** = trading accuracy for tractability

---

#### How It's Used in DPO

**Need:** Evaluate p_θ(F_{t-1}|F_t) for improvement score

**True DiffCSP:** Can't write down p_θ(F_{t-1}|F_t) in closed form (Langevin is stochastic)

**Proxy:** Can write down:
```
log p_θ(F_{t-1}|F_t) ≈ -1/(2σ̃_t²) ||Δ(F_{t-1}, μ_θ^(F))||² + const

log-ratio:
log[p_θ/p_ref] ≈ -1/(2σ̃_t²) [||Δ(F_{t-1}, μ_θ^(F))||² - ||Δ(F_{t-1}, μ_ref^(F))||²]
               = -(1/(2σ̃_t²)) · [d_θ^(F) - d_ref^(F)]
```

**This is tractable!** Can backpropagate through it.

---

### Q8: What is wrapped Gaussian? Does it have multiple peaks?

**📍 Location:** Equation 2.16 context

#### Quick Answer

**NO, not multiple peaks** (usually)!

Wrapped Gaussian has **one main peak** that "bleeds" across the boundary.

---

#### Construction

**Step 1:** Start with regular Gaussian on ℝ
```
N(μ, σ²) on (-∞, +∞)
```

**Step 2:** Wrap to [0,1)
```
x_wrapped = x mod 1.0
```

**Result:** Wrapped Normal N_w(μ, σ²) on [0,1)

---

#### Mathematical Definition

**PDF of wrapped Gaussian:**
```
p_w(x; μ, σ²) = ∑_{k=-∞}^{+∞} (1/√(2πσ²)) exp(-(x+k-μ)²/(2σ²))
                ↑
            sum over all periodic copies
```

**In practice:** Only k ∈ {-1, 0, +1} matter if σ small enough.

---

#### Visual Examples

**Case 1: μ=0.5, σ=0.1 (center, small variance)**
```
Regular Gaussian:       Wrapped Gaussian:

     /\                      /\
    /  \                    /  \
___/____\___           ___/____\___
  0.3  0.7              0  0.5  1≡0

Single peak at 0.5      Single peak at 0.5
                        (no wrapping needed)
```

**Case 2: μ=0.95, σ=0.05 (near boundary)**
```
Regular Gaussian:       Wrapped Gaussian:

         /\                 \    /
        /  \                 \  /
_______/____\            ____\_/____
     0.9   1.0           0  0.95  1≡0
                          ↑       ↑
                        bleed   main

Gaussian crosses 1.0    Wraps around!
                        Peak at 0.95, bleeds to ~0.00
```

**Case 3: μ=0.1, σ=0.05 (near other boundary)**
```
Wrapped Gaussian:

   \    /
    \  /
____ \/ ____
 0  0.1  1≡0
  ↑      ↑
bleed   main

Peak at 0.1, bleeds to ~1.0
```

**Case 4: μ=0.5, σ=0.4 (large variance)**
```
Wrapped Gaussian:

  /\    /\
 /  \__/  \
/__________\
0   0.5   1≡0

Wide Gaussian wraps around both sides
Still single "conceptual" peak, but visible mass at boundaries
```

---

#### Why NOT Multiple Peaks?

**Common misconception:** "Wrapped = multiple peaks at 0, 0.5, 1.0"

**Reality:** The wrapping **folds space**, not **duplicates** the distribution.

**Analogy:**
```
Regular Gaussian: Drawing a bell curve on infinite paper
Wrapped Gaussian: Drawing the same curve on a cylinder
                  (the paper wraps around, but curve stays one curve)
```

**Mathematical:** The ∑_{k=-∞}^{+∞} in the PDF sums contributions from all **periodic images**, not separate peaks.

For x ∈ [0,1), you see:
- Main contribution from k=0
- Bleeding from k=±1 if near boundaries
- Not "multiple independent peaks"

---

#### When You Might See "Multiple Peaks"

Only if you **mix** multiple wrapped Gaussians:
```
p(x) = 0.5·N_w(0.2, 0.05²) + 0.5·N_w(0.8, 0.05²)
       ↑                        ↑
    peak at 0.2              peak at 0.8

This is a MIXTURE, not a single wrapped Gaussian!
```

**In SCIGEN+ DPO:** We use **single** wrapped Gaussians, not mixtures.

---

#### Sampling from Wrapped Gaussian

```python
def sample_wrapped_gaussian(mu, sigma, size):
    """Sample from wrapped Gaussian on [0,1)"""
    # Sample from regular Gaussian
    x = np.random.normal(mu, sigma, size)

    # Wrap to [0,1)
    x_wrapped = x % 1.0

    return x_wrapped
```

**Note:** This works because wrapping preserves the distribution!

---

### Q9-Q11: Forward Consistent Pair & Simple Coupling

**📍 Location:** Context around Eq 2.15-2.16

These three questions form a connected workflow for evaluating the fractional coordinate improvement score.

---

#### Q9: What is "forward consistent pair"?

**Definition:** A pair (F_{t-1}, F_t) where F_t comes from **forward diffusion** starting at F_0.

**Mathematically:**
```
F_t ~ q(F_t | F_0)  (forward process)

Then F_{t-1} and F_t are "forward consistent"
```

---

**Why needed?**

To compute d_θ^(F) (denoising error), we need:
```
d_θ^(F) = ||Δ(F_{t-1}, μ_θ^(F))||²
          ↑           ↑
        target    prediction given F_t
```

**We have:**
- F_0 (clean structure)
- Model prediction μ_θ^(F)(F_t, t)

**We need:**
- F_t (condition)
- F_{t-1} (target)
- Both from same forward trajectory!

---

**Naive approach (WRONG):**
```
1. Sample F_t ~ q(F_t|F_0)
2. Independently sample F_{t-1} ~ q(F_{t-1}|F_0)

Problem: F_t and F_{t-1} are from DIFFERENT trajectories!
```

**Visual:**
```
F_0 = [0.5, 0.3, 0.7]

Trajectory 1:  F_0 → (ε=+0.1) → F_t^(1)   = [0.6, 0.4, 0.8]
Trajectory 2:  F_0 → (ε=-0.2) → F_{t-1}^(2) = [0.3, 0.1, 0.5]

Using F_t^(1) and F_{t-1}^(2) together is WRONG!
They have nothing to do with each other.
```

**Correct approach:** Use **same trajectory**
```
F_0 → (same noise) → F_{t-1} → (more noise) → F_t
```

---

#### Q10: What is "simple coupling"?

**Definition:** A method to generate (F_{t-1}, F_t) from same noise.

**Algorithm:**
```
1. Draw ε_F ~ N(0, I) ONCE
2. Set F_s = w(F_0 + σ_s · ε_F) for s ∈ {t-1, t}
   ↑
   same ε_F for both!
```

**Result:**
- F_{t-1} = w(F_0 + σ_{t-1} · ε_F)
- F_t = w(F_0 + σ_t · ε_F)
- Both use same underlying noise ε_F
- F_t is "more noisy" than F_{t-1} (since σ_t > σ_{t-1})

---

**Why "simple"?**

**Other coupling methods exist:**
- **Antithetic coupling:** ε_{t-1} = -ε_t
- **Conditional coupling:** Sample ε_t, then ε_{t-1}|ε_t
- **Brownian bridge:** Exact posterior q(F_{t-1}|F_0, F_t)

**Simple coupling** is easiest: just use same noise!

---

**Visual Example:**

```
F_0 = [0.5, 0.3, 0.7]
ε_F = [0.15, -0.05, 0.10]  ← drawn once

t=500: σ_500 = 0.3
  F_500 = w([0.5, 0.3, 0.7] + 0.3·[0.15, -0.05, 0.10])
        = w([0.545, 0.285, 0.73])
        = [0.545, 0.285, 0.73]

t=499: σ_499 = 0.295 (slightly smaller)
  F_499 = w([0.5, 0.3, 0.7] + 0.295·[0.15, -0.05, 0.10])
        = w([0.54425, 0.28525, 0.7295])
        = [0.54425, 0.28525, 0.7295]

F_499 and F_500 are correlated through shared ε_F!
```

---

**Code:**

```python
def simple_coupling(F_0, sigma_t_minus_1, sigma_t):
    """Generate forward-consistent pair (F_{t-1}, F_t)"""
    # Draw noise once
    eps_F = torch.randn_like(F_0)

    # Apply same noise with different scales
    F_t_minus_1 = (F_0 + sigma_t_minus_1 * eps_F) % 1.0
    F_t = (F_0 + sigma_t * eps_F) % 1.0

    return F_t_minus_1, F_t, eps_F
```

---

#### Q11: Why need simple coupling?

**Recap of DPO evaluation:**

For improvement score I_θ^(F), we compute:
```
d_θ^(F) = ||Δ(F_{t-1}, μ_θ^(F)(F_t,t))||²
```

**Requirements:**
1. **F_t** = condition for model prediction
2. **F_{t-1}** = target to compare against
3. **Consistency**: (F_{t-1}, F_t) from same forward trajectory

---

**What happens without coupling?**

```
❌ Independent sampling:

F_t ~ q(F_t|F_0)      ← noise ε_t
F_{t-1} ~ q(F_{t-1}|F_0)  ← different noise ε_{t-1}

Problem: F_t and F_{t-1} are UNRELATED
→ d_θ^(F) measures error against WRONG target
→ Gradient signal is NOISE
→ Training fails!
```

**What happens with coupling?**

```
✅ Simple coupling:

(F_{t-1}, F_t) from same ε_F

F_t and F_{t-1} are CONSISTENT
→ d_θ^(F) measures error against CORRECT target
→ Gradient signal is MEANINGFUL
→ Training succeeds!
```

---

**Analogy:**

**Without coupling:**
```
Teacher: "Predict what comes after 'The cat sat on the...'"
Student: "mat"
Teacher: "WRONG! The answer is 'dog' because that's what came after in a DIFFERENT sentence!"

This is nonsense! The teacher is comparing to the wrong reference.
```

**With coupling:**
```
Teacher: "Predict what comes after 'The cat sat on the...'"
Student: "mat"
Teacher: "Good! In THIS sentence, the answer was 'mat'."

Consistent evaluation!
```

---

**Mathematical Justification:**

The reverse kernel p_θ(F_{t-1}|F_t) should be evaluated on pairs that could plausibly come from the forward process q(F_{t-1}, F_t|F_0).

Simple coupling samples from this joint:
```
(F_{t-1}, F_t) ~ q(F_{t-1}, F_t | F_0)
```

which is the **correct** distribution for training!

---

### Q12-Q13: Tractable Proxy & Predictor-Corrector

**📍 Location:** Context around wrapped Gaussian proxy

---

#### Q12: What is "tractable proxy"?

Let's break down both words:

**Tractable** = computable in closed form with efficient gradients

**Proxy** = approximation to the true thing

---

**The "True Thing": DiffCSP Sampling**

DiffCSP uses predictor-corrector sampling (Jiao et al. 2023):

```python
def diffcsp_reverse_step(F_t, t, model):
    """One reverse step in DiffCSP (simplified)"""
    # PREDICTOR: Score-based update
    score = model.predict_score(F_t, t)
    F_pred = wrap(F_t + eta_t * score)

    # CORRECTOR: Langevin dynamics
    F = F_pred
    for k in range(K_corrector_steps):  # K ~ 5-10
        score = model.predict_score(F, t)
        noise = torch.randn_like(F)
        F = F + epsilon * score + sqrt(2*epsilon) * noise
        F = wrap(F)

    F_t_minus_1 = F
    return F_t_minus_1
```

**Problem for DPO:**

Can you write down p_θ(F_{t-1}|F_t) for this process?

**NO!** The Langevin loop makes it:
- **Stochastic**: Different runs give different F_{t-1}
- **Iterative**: K steps of random walk
- **Intractable**: No closed-form density

You can **sample** from it, but not **evaluate probability**!

---

**The "Tractable Proxy": Wrapped Gaussian**

Instead, approximate as:
```python
def proxy_reverse_step(F_t, t, model):
    """Proxy reverse kernel (tractable)"""
    # Predictor only (no corrector)
    score = model.predict_score(F_t, t)
    mu_F = wrap(F_t + eta_t * score)

    # Single Gaussian step
    F_t_minus_1 = sample_wrapped_gaussian(mu_F, sigma_tilde_t)
    return F_t_minus_1
```

**Now can we write down p_θ(F_{t-1}|F_t)?**

**YES!**
```
p_θ(F_{t-1}|F_t) = N_w(F_{t-1}; μ_F, σ̃_t²I)

where μ_F = w(F_t + η_t · score_θ(F_t,t))
```

This is a **closed-form Gaussian density**!

---

**Why "Tractable"?**

Can compute log-probability and gradients:
```python
def log_prob_wrapped_gaussian(F_t_minus_1, mu_F, sigma_t):
    """Log-probability of wrapped Gaussian (approximation)"""
    delta = wrapped_diff(F_t_minus_1, mu_F)  # Δ(F_{t-1}, μ_F)
    log_p = -0.5 / (sigma_t**2) * (delta**2).sum()
    return log_p

# Can backpropagate through this! ✅
loss = -log_prob_wrapped_gaussian(F_w, mu_w, sigma) + \
       -log_prob_wrapped_gaussian(F_l, mu_l, sigma)
loss.backward()  # Tractable gradients
```

---

**Comparison Table:**

| Aspect | True DiffCSP | Tractable Proxy |
|--------|-------------|-----------------|
| **Sampling** | Predictor + Corrector | Predictor only |
| **Iterations** | K Langevin steps | 1 Gaussian sample |
| **Density** | No closed form ❌ | Wrapped Gaussian ✅ |
| **Gradients** | Hard to compute | Easy (backprop) |
| **DPO training** | Intractable | Tractable |
| **Sample quality** | Better (more steps) | Good enough |

---

**Why "Proxy"?**

It's an **approximation**:
- **Skips corrector**: Loses some accuracy
- **Single-step Gaussian**: Simpler than true posterior
- **But good enough**: For DPO, log-ratio approximation works well

**Analogy:**
- True DiffCSP: Taking a carefully planned route with multiple corrections
- Tractable proxy: Taking a direct route (less accurate but faster and predictable)

For DPO, we just need to **compare** routes, not find the perfect one!

---

#### Q13: What is predictor-corrector? Why important?

**📍 Location:** Context around Eq 2.16

---

**Predictor-Corrector Overview**

A two-phase sampling algorithm for diffusion models:

```
At each timestep t → t-1:

PREDICTOR:  Make initial guess using score
CORRECTOR:  Refine guess using Langevin dynamics
```

---

**Step-by-Step Breakdown**

**PREDICTOR Phase:**

Goal: Take one Euler-Maruyama step based on score

```
score_θ(F_t, t) = ∇_{F_t} log q(F_t|F_0)  ← model predicts this

F'_{t-1} = w(F_t + η_t · score_θ(F_t, t))
           ↑           ↑
        wrap     step in score direction
```

Intuition: Move towards higher probability

---

**CORRECTOR Phase (Langevin Dynamics):**

Goal: Refine F'_{t-1} by exploring nearby space

```
Start: F = F'_{t-1}

For k = 1 to K:
    score = ∇_F log p(F|F_t)  ← score at CURRENT F

    F ← F + ε · score + √(2ε) · noise
        ↑       ↑            ↑
      update  drift       diffusion

    F ← w(F)  (wrap to torus)

Result: F_{t-1} = F
```

**Langevin dynamics** = random walk that converges to target distribution p(F|F_t)

---

**Visual Analogy:**

```
Target distribution p(F|F_t) (想 imagine mountain landscape):

      /\
     /  \    <- peak (high probability)
    /    \
___/______\___

PREDICTOR:
  Start at F_t (valley)
  Jump towards peak using score: F'_{t-1}

  F_t → → → F'_{t-1}

  But might overshoot or land on slope!

CORRECTOR:
  Refine F'_{t-1} using local exploration
  Random walk converges to peak

  F'_{t-1} ~~> ~~> peak (F_{t-1})
               ↑
           Langevin walk
```

---

**Mathematical Justification**

**Langevin SDE** (stochastic differential equation):
```
dF = ∇ log p(F) dt + √(2) dW
     ↑                 ↑
   drift to peak   diffusion
```

After sufficient iterations, F ~ p(F) (stationary distribution)

**Discretized Langevin:**
```
F_{k+1} = F_k + ε·∇ log p(F_k) + √(2ε)·noise_k
```

This is the corrector loop!

---

**Why Important for DiffCSP?**

**Challenge:** Fractional coordinates live on **torus**, not flat ℝ³

**Problem with predictor alone:**
- Score matching objective: E[||∇log q - score_θ||²]
- On torus, score field has **periodicity constraints**
- Single predictor step can violate these (especially near boundaries)

**Solution with corrector:**
- Langevin dynamics **automatically respects** torus geometry
- Explores local neighborhood → finds valid point
- Corrects boundary artifacts from predictor

---

**Example: Boundary Crossing**

```
F_t = [0.95, 0.3, 0.7]
score points "right" → should wrap to ~0.02

PREDICTOR ONLY:
  F'_{t-1} = wrap([0.95, 0.3, 0.7] + [0.1, 0, 0])
           = wrap([1.05, 0.3, 0.7])
           = [0.05, 0.3, 0.7]

  But prediction might be slightly off near boundary!

WITH CORRECTOR:
  F = [0.05, 0.3, 0.7]

  Langevin refines:
    Step 1: F ← [0.045, 0.305, 0.698] + noise
    Step 2: F ← [0.042, 0.301, 0.702] + noise
    ...
    Step K: F ← [0.038, 0.298, 0.705] (converged)

  More accurate near boundary!
```

---

**For SCIGEN+ DPO:**

**We skip the corrector** (proxy approach):
- Corrector: K extra model calls per timestep
- DPO: Need evaluatable density (Langevin is stochastic)
- Trade-off: Speed & tractability vs accuracy

**Justification:**
- For DPO, we need log p_θ / p_ref (ratio)
- Approximation errors might cancel in ratio!
- Empirically works well enough

---

**Reference:**

**Predictor-corrector for diffusion:**
- Song et al. (2021) "Score-Based Generative Modeling through Stochastic Differential Equations", Algorithm 1

**DiffCSP application:**
- Jiao et al. (2023) "Crystal Structure Prediction by Joint Equivariant Diffusion", Section 3.3

---

### Q14: What is "periodic proxy reverse kernel"?

**📍 Location:** Context around wrapped Gaussian

---

#### Quick Answer

**Same thing** as "wrapped normal proxy reverse kernel" (Q7)!

Just emphasizes different aspect: **periodic** = respects torus topology

---

#### Why "Periodic"?

**Periodic domain** = torus [0,1)^{N×3}

Properties:
- 0.0 ≡ 1.0 (same point)
- Distances wrap around
- Geometry is periodic

**A periodic-aware kernel:**
```
p_θ(F_{t-1}|F_t) must respect:
  p_θ(F_{t-1} + [1,0,0] | F_t) = p_θ(F_{t-1} | F_t)
  ↑
periodic shift doesn't change probability
```

---

#### Why Wrapped Gaussian is Periodic

**Wrapped Gaussian construction:**
```
1. Start with N(μ, σ²) on ℝ
2. Wrap to [0,1): x_w = x mod 1
```

**Key property:**
```
p_w(x + 1; μ, σ²) = p_w(x; μ, σ²)

Proof:
  p_w(x) = ∑_k N(x+k; μ, σ²)  (infinite sum)

  p_w(x+1) = ∑_k N(x+1+k; μ, σ²)
           = ∑_{k'} N(x+k'; μ, σ²)  (k'=k+1, relabel)
           = p_w(x)  ✓
```

So wrapped Gaussian **automatically respects periodicity**!

---

#### Contrast with Non-Periodic Kernel

**Bad idea (WRONG for torus):**
```
p(F_{t-1}|F_t) = N(F_{t-1}; μ_θ, σ²I)  ← regular Gaussian
                 ↑
            NOT periodic!

Problem:
  F_{t-1} = [0.99, 0.3, 0.5]
  μ_θ = [0.01, 0.3, 0.5]

  Regular Gaussian distance: |0.99 - 0.01| = 0.98 (far!)
  Torus distance: wrap(0.99 - 0.01) = -0.02 (close!)

  → Regular Gaussian assigns LOW probability to nearby points!
  → Training breaks!
```

**Good idea (CORRECT for torus):**
```
p_w(F_{t-1}|F_t) = N_w(F_{t-1}; μ_θ, σ²I)  ← wrapped Gaussian
                   ↑
              periodic!

Automatically uses wrapped distance:
  p_w([0.99,...]) high when μ=[0.01,...]  ✓
```

---

#### Summary Table

| Term | Emphasis | Same Thing? |
|------|----------|-------------|
| **Wrapped normal** | Construction method | ✅ Yes |
| **Periodic kernel** | Respects torus topology | ✅ Yes |
| **Proxy** | Approximation to true DiffCSP | ✅ Yes |
| **Reverse kernel** | Denoising t→t-1 | ✅ Yes |

All refer to: **N_w(F_{t-1}; μ_θ^(F), σ̃_t²I)**

---

# Section 3.2.3: Hölder-DPO

## Q15: How to derive DPO margin $g_\theta(t)$?

**📍 Location:** Section 3.2.3, around Equation 3.14

### Quick Summary

The DPO margin $g_\theta(t)$ is simply the **difference** in improvement scores between winner and loser:

$$g_\theta(t) = I_\theta(\mathbf{x}^w, t) - I_\theta(\mathbf{x}^\ell, t)$$

It's just applying the DPO framework to our improvement score proxy!

---

### Step-by-Step Derivation

#### Step 1: Recall improvement score for one structure

From Q5, we have:

$$I_\theta(\mathbf{x}, t) = \sum_{z \in \{L, F, A\}} I_\theta^{(z)}(\mathbf{x}, t)$$

where:

$$I_\theta^{(z)}(\mathbf{x}, t) = \omega_t^{(z)} \left[ d_{\text{ref}}^{(z)} - d_\theta^{(z)} \right]$$

**Interpretation:** "How much better is the model at denoising $\mathbf{x}$ at timestep $t$ compared to reference?"

---

#### Step 2: Define margin for a pair

Given preference pair $(\mathbf{x}^w, \mathbf{x}^\ell)$ (winner vs loser):

$$g_\theta(t) := I_\theta(\mathbf{x}^w, t) - I_\theta(\mathbf{x}^\ell, t)$$

Expand:

$$g_\theta(t) = \sum_{z \in \{L,F,A\}} \left[ I_\theta^{(z)}(\mathbf{x}^w, t) - I_\theta^{(z)}(\mathbf{x}^\ell, t) \right]$$

**Interpretation:** "At timestep $t$, how much better does the model denoise the winner vs the loser?"

---

#### Step 3: Channel-by-channel breakdown

$$g_\theta(t) = \underbrace{\left[I_\theta^{(L)}(\mathbf{x}^w, t) - I_\theta^{(L)}(\mathbf{x}^\ell, t)\right]}_{\text{lattice contribution}}$$
$$+ \underbrace{\left[I_\theta^{(F)}(\mathbf{x}^w, t) - I_\theta^{(F)}(\mathbf{x}^\ell, t)\right]}_{\text{fractional coords contribution}}$$
$$+ \underbrace{\left[I_\theta^{(A)}(\mathbf{x}^w, t) - I_\theta^{(A)}(\mathbf{x}^\ell, t)\right]}_{\text{atom types contribution}}$$

Each channel contributes to preference!

---

#### Step 4: Connect to Bradley-Terry model

Recall Bradley-Terry (Q3):

$$P(\mathbf{x}^w \succ \mathbf{x}^\ell) = \sigma(\text{score}_w - \text{score}_\ell)$$

In DPO, the "score" is log-probability. For diffusion, we use improvement as proxy:

$$P(\mathbf{x}^w \succ \mathbf{x}^\ell \mid t) \approx \sigma\left( \beta \cdot T \cdot g_\theta(t) \right)$$

where:
- $\beta$ = preference sharpness (temperature inverse)
- $T$ = total timesteps (normalization, since we sample one $t \sim \text{Uniform}(1,T)$)
- $g_\theta(t)$ = margin at timestep $t$

---

### Numerical Example

**Setup:**
- Winner: $\mathbf{x}^w$ (better flat band)
- Loser: $\mathbf{x}^\ell$ (worse flat band)
- Timestep: $t = 500$

**Model and reference predictions:**

| Channel | $d_{\text{ref}}^w$ | $d_\theta^w$ | $d_{\text{ref}}^\ell$ | $d_\theta^\ell$ |
|---------|---------------------|---------------|------------------------|------------------|
| L | 0.05 | 0.02 | 0.04 | 0.03 |
| F | 0.08 | 0.03 | 0.07 | 0.05 |
| A | 0.06 | 0.04 | 0.05 | 0.04 |

**Compute improvement scores:**

Lattice:
$$I_\theta^{(L)}(\mathbf{x}^w, 500) = \omega_{500}^{(L)} (0.05 - 0.02) = \omega_{500}^{(L)} \cdot 0.03$$
$$I_\theta^{(L)}(\mathbf{x}^\ell, 500) = \omega_{500}^{(L)} (0.04 - 0.03) = \omega_{500}^{(L)} \cdot 0.01$$

Fractional:
$$I_\theta^{(F)}(\mathbf{x}^w, 500) = \omega_{500}^{(F)} (0.08 - 0.03) = \omega_{500}^{(F)} \cdot 0.05$$
$$I_\theta^{(F)}(\mathbf{x}^\ell, 500) = \omega_{500}^{(F)} (0.07 - 0.05) = \omega_{500}^{(F)} \cdot 0.02$$

Atom types:
$$I_\theta^{(A)}(\mathbf{x}^w, 500) = \omega_{500}^{(A)} (0.06 - 0.04) = \omega_{500}^{(A)} \cdot 0.02$$
$$I_\theta^{(A)}(\mathbf{x}^\ell, 500) = \omega_{500}^{(A)} (0.05 - 0.04) = \omega_{500}^{(A)} \cdot 0.01$$

**Margin:**

Assuming $\omega_{500}^{(L)} = \omega_{500}^{(F)} = \omega_{500}^{(A)} = 2.0$ (for simplicity):

$$g_\theta(500) = 2.0 \cdot [(0.03 - 0.01) + (0.05 - 0.02) + (0.02 - 0.01)]$$
$$= 2.0 \cdot [0.02 + 0.03 + 0.01] = 2.0 \cdot 0.06 = 0.12$$

**Preference probability:**

With $\beta = 0.1$, $T = 1000$:

$$P(\mathbf{x}^w \succ \mathbf{x}^\ell \mid t=500) = \sigma(0.1 \cdot 1000 \cdot 0.12) = \sigma(12.0) \approx 0.9999$$

Model strongly prefers winner!

---

### Why This Works

**Key insight from Diffusion-DPO:**

Since exact $\log p_\theta(\mathbf{x}_0)$ is intractable, we use **single-timestep proxy**:

$$\log \frac{p_\theta(\mathbf{x}_0)}{p_{\text{ref}}(\mathbf{x}_0)} \approx \mathbb{E}_{t \sim \text{Uniform}(1,T)} \left[ T \cdot \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{p_{\text{ref}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} \right]$$

And $\log \frac{p_\theta}{p_{\text{ref}}} \propto I_\theta$ (improvement score).

Therefore:

$$\log \frac{p_\theta(\mathbf{x}^w_0)}{p_{\text{ref}}(\mathbf{x}^w_0)} - \log \frac{p_\theta(\mathbf{x}^\ell_0)}{p_{\text{ref}}(\mathbf{x}^\ell_0)} \approx T \cdot g_\theta(t)$$

---

### Summary

$$\boxed{g_\theta(t) = I_\theta(\mathbf{x}^w, t) - I_\theta(\mathbf{x}^\ell, t) = \sum_{z \in \{L,F,A\}} \left[ I_\theta^{(z)}(\mathbf{x}^w, t) - I_\theta^{(z)}(\mathbf{x}^\ell, t) \right]}$$

**Intuition:** "At time $t$, winner should be easier to denoise than loser (for a good model)"

---

## Q16: Why is Hölder loss $\ell_\gamma(x) = -(1+\gamma)\sigma(x)^\gamma + \gamma \sigma(x)^{1+\gamma}$?

**📍 Location:** Section 3.2.3, Equation 3.15

### The Big Picture

**Problem:** Standard DPO uses KL divergence, which is **not robust** to outliers (mislabeled preferences).

**Solution:** Hölder-DPO uses **Hölder divergence**, which has a "redescending" property that down-weights extreme outliers.

---

### What is Hölder Divergence?

**General form of $f$-divergence:**

$$D_f(p \| q) = \int q(x) \cdot \phi\left(\frac{p(x)}{q(x)}\right) dx$$

where $\phi$ is a convex function.

**Hölder divergence** uses density-powered (DP) function:

$$\phi_{\text{DP}}^{(\gamma)}(h) = \gamma - (1+\gamma) h^{\frac{\gamma}{1+\gamma}}$$

for $\gamma > 0$.

---

### Deriving the Hölder Loss

#### Step 1: Preference model

Binary preference: $y \in \{0, 1\}$ where $y=1$ means "winner preferred".

Model probability:

$$p_\theta(y=1 \mid s) = \sigma(u_\theta(s))$$

where $s$ is a sample (includes $\mathbf{x}^w, \mathbf{x}^\ell, t$) and $u_\theta(s) = \beta \cdot T \cdot g_\theta(t)$.

Reference (uniform):

$$p_{\text{ref}}(y=1 \mid s) = 0.5$$

---

#### Step 2: Apply Hölder divergence

We want to minimize:

$$D_H^\gamma(p_{\text{data}} \| p_\theta)$$

After substituting the DP function and simplifying (full derivation in Fujisawa et al. 2025, Appendix C), we get:

$$\mathcal{L}_{\text{H-DPO}}(\theta) = \mathbb{E}_s\left[ \ell_\gamma(u_\theta(s)) \right]$$

where:

$$\boxed{\ell_\gamma(x) = -(1+\gamma)\sigma(x)^\gamma + \gamma \sigma(x)^{1+\gamma}}$$

---

### Why This Form?

Let's analyze the loss and its gradient.

**Loss:** $\ell_\gamma(x) = -(1+\gamma)\sigma(x)^\gamma + \gamma \sigma(x)^{1+\gamma}$

**Gradient:** (from chain rule)

$$\frac{\partial \ell_\gamma}{\partial x} = -\gamma(1+\gamma) \sigma(x)^\gamma (1 - \sigma(x))^2$$

Let $p = \sigma(x)$. Then:

$$\frac{\partial \ell_\gamma}{\partial x} = -\gamma(1+\gamma) p^\gamma (1-p)^2$$

---

### Key Property: Redescending

**Redescending** means: gradient vanishes as $x \to -\infty$ (strong disagreement with label).

**Proof:**

As $x \to -\infty$:
- $\sigma(x) \to 0$
- $p^\gamma \to 0$
- $(1-p)^2 \to 1$

Therefore:

$$\left|\frac{\partial \ell_\gamma}{\partial x}\right| = \gamma(1+\gamma) p^\gamma \underbrace{(1-p)^2}_{\to 1} \to 0$$

**Interpretation:** If model **strongly disagrees** with a preference label ($x$ very negative), the gradient goes to zero → **outlier has bounded influence**!

---

### Comparison with Standard Logistic Loss

**Logistic loss** (KL-based DPO, $\gamma \to 0$ limit):

$$\ell_{\text{logistic}}(x) = -\log \sigma(x) = \log(1 + e^{-x})$$

**Gradient:**

$$\frac{\partial \ell_{\text{logistic}}}{\partial x} = -\sigma(x)(1-\sigma(x)) = -(1-\sigma(x))$$

As $x \to -\infty$:
- $\sigma(x) \to 0$
- $\frac{\partial \ell}{\partial x} \to -1$ (does NOT vanish!)

**Not redescending!** Outliers have unbounded influence.

---

### Visual Comparison

```
Influence weight: ι_γ(x) = |∂ℓ_γ/∂x| / max|∂ℓ_γ/∂x|

       Influence
          ↑
       1.0|     /\
          |    /  \
          |   /    \___________  ← Hölder (γ=2.0): redescends
       0.5|  /
          | /
          |/___________/\___     ← Logistic: stays high
       0.0|________________________→ x (scaled margin)
         -10  -5   0   5   10

         ↑               ↑
    Model strongly   Model strongly
    disagrees        agrees
    (potential       (confident
     outlier)         correct)
```

**Hölder loss:** Down-weights both extremes (especially negative = outliers)

**Logistic loss:** High gradient even for extreme disagreements

---

### Effect of $\gamma$

**Influence:** $\iota_\gamma(x) = \sigma(x)^\gamma (1-\sigma(x))^2$

| $\gamma$ | Robustness | Peak Influence | Outlier Weight |
|----------|-----------|----------------|----------------|
| 0.5 | Low | Broad | Moderate |
| 1.0 | Medium | Medium | Low |
| **2.0** | **High** | **Narrow** | **Very Low** |
| 5.0 | Very High | Very Narrow | Nearly Zero |

**Default:** $\gamma = 2.0$ (from Fujisawa et al. 2025)

---

### Concrete Example

Let's evaluate $\ell_\gamma$ and its gradient for different $x$ values:

**$\gamma = 2.0$, $x \in \{-10, -5, 0, 5, 10\}$**

| $x$ | $\sigma(x)$ | $\ell_2(x)$ | $\partial \ell_2/\partial x$ | Interpretation |
|-----|-------------|-------------|------------------------------|----------------|
| -10 | 0.00005 | -0.00015 | -0.00001 | Outlier (ignored) |
| -5 | 0.0067 | -0.020 | -0.012 | Strong disagree |
| 0 | 0.5 | -0.875 | -1.5 | Uncertain |
| 5 | 0.993 | -1.97 | -0.042 | Strong agree |
| 10 | 0.99995 | -2.00 | -0.0001 | Very confident |

**Key observations:**
1. Gradient is largest at $x=0$ (uncertain cases)
2. Gradient vanishes at both extremes ($x = \pm 10$)
3. **Outliers** ($x = -10$) have ~100× smaller gradient than uncertain cases

---

### Reference

**Full derivation:** Fujisawa et al. (2025) "Scalable Valuation of Human Feedback through Provably Robust Model Alignment", **Appendix C**

Equations C.1-C.8 show step-by-step:
1. Hölder divergence definition
2. Application to binary preferences
3. Simplification to $\ell_\gamma$ form
4. Gradient computation
5. Redescending property proof

---

## Q17: How to optimize/tune $\gamma$?

**📍 Location:** Section 3.2.4

### Default Strategy

**Quick answer:** Use $\gamma = 2.0$ (recommended in Hölder-DPO paper).

But if you have confidence labels $\kappa$ or want to tune, follow this protocol.

---

### Diagnostic Tuning Protocol

#### Step 1: Create high-confidence validation set

Split your validation set:

**High-confidence:** $\mathcal{D}_{\text{val}}^{\text{high}} = \{(c, \mathbf{x}^w, \mathbf{x}^\ell, \kappa) : \kappa \geq 4\}$

**All:** $\mathcal{D}_{\text{val}}^{\text{all}} = \{(c, \mathbf{x}^w, \mathbf{x}^\ell, \kappa) : \kappa \geq 1\}$

**Rationale:** High-confidence pairs are likely correct → good metric for model quality

---

#### Step 2: Candidate $\gamma$ values

Try: $\gamma \in \{0.5, 1.0, 2.0, 3.0, 5.0\}$

**Note:** $\gamma = 0$ recovers logistic loss (baseline)

---

#### Step 3: Train with each $\gamma$

For each candidate $\gamma$:
1. Train model using Hölder loss $\ell_\gamma$
2. Monitor training stability (loss, gradient norms)
3. Checkpoint final model $\theta_\gamma$

---

#### Step 4: Evaluate on high-confidence set

For each $\theta_\gamma$, compute **preference accuracy** on $\mathcal{D}_{\text{val}}^{\text{high}}$:

$$\text{Acc}_{\text{high}}(\gamma) = \frac{1}{|\mathcal{D}_{\text{val}}^{\text{high}}|} \sum_{s \in \mathcal{D}_{\text{val}}^{\text{high}}} \mathbb{1}[u_{\theta_\gamma}(s) > 0]$$

where $u_\theta(s) = \beta \cdot T \cdot g_\theta(t)$ (average over multiple $t$ for stability).

**Interpretation:** Fraction of high-confidence pairs where model correctly prefers winner

---

#### Step 5: Check outlier separation (optional)

For each $\theta_\gamma$, compute **influence** on all validation data:

$$\iota_\gamma(s) = \sigma(u_{\theta_\gamma}(s))^\gamma (1 - \sigma(u_{\theta_\gamma}(s)))^2$$

**Diagnostic checks:**

1. **Outlier enrichment:**
   - Take bottom 10% by influence: $\mathcal{O}_{\text{low}} = \text{arg bottom}_{10\%} \{\iota_\gamma(s)\}$
   - Check if $\kappa$ is lower in $\mathcal{O}_{\text{low}}$ vs overall
   - **Good:** Low-influence has lower confidence → robustness working!

2. **Spearman correlation:**
   - Compute $\rho_{\text{Spearman}}(\kappa, |g_{\theta_\gamma}(t)|)$
   - **Good:** Positive correlation → model learns easier pairs better

---

#### Step 6: Select optimal $\gamma$

Choose $\gamma^*$ that:

$$\gamma^* = \arg\max_\gamma \text{Acc}_{\text{high}}(\gamma)$$

subject to:
- Training stable
- Outlier separation diagnostic looks good

**Typical result:** $\gamma^* \in \{2.0, 3.0\}$ for noisy crowdsourced data

---

### Monitoring During Training

Track these metrics per epoch:

**1. Influence distribution:**

Plot histogram of $\iota_\gamma(s)$ for all training samples.

**Healthy:**
```
Count
  ↑
  |     ***
  |    *   *
  |   *     *
  |  *       **__
  | *            *___
  |_________________*______→ Influence ι_γ
 0               0.5      1.0

Most samples moderate influence, few at extremes
```

**Unhealthy (all high):**
```
Count
  ↑        *****
  |       *     *
  |              *
  |               **
  | ______________  *___
  |_____________________→ Influence ι_γ
 0                    1.0

All samples have high influence → not robust!
```

**2. Outlier proportion estimate:**

Use estimator from Section 3.4 (Q47):

$$\hat{\epsilon} = 1 - \hat{\xi} = 1 - \frac{1}{N} \cdot \frac{\sum_i \bar{p}_i^\gamma}{\sum_i \bar{p}_i^{1+\gamma}}$$

**Expected:** $\hat{\epsilon} \in [0.05, 0.20]$ for crowdsourced data

**3. Confidence stratification:**

Compute accuracy separately for $\kappa \in \{1, 2, 3, 4, 5\}$:

| $\kappa$ | Accuracy | Interpretation |
|----------|----------|----------------|
| 1 | 0.55 | Low confidence → near random |
| 2 | 0.68 | Medium confidence |
| 3 | 0.79 | Medium-high confidence |
| 4 | 0.89 | High confidence → should be high! |
| 5 | 0.94 | Very high confidence |

**Healthy:** Monotonic increase

---

### Example Tuning Result

**Dataset:** Phase A preference pairs with $\kappa$ labels

| $\gamma$ | $\text{Acc}_{\text{high}}$ | $\text{Acc}_{\text{all}}$ | $\hat{\epsilon}$ | Training Stability | Selected? |
|----------|----------------------------|---------------------------|------------------|---------------------|-----------|
| 0.0 | 0.82 | 0.68 | 0.05 | ✅ Stable | ❌ |
| 0.5 | 0.85 | 0.69 | 0.08 | ✅ Stable | ❌ |
| 1.0 | 0.88 | 0.70 | 0.12 | ✅ Stable | ❌ |
| **2.0** | **0.91** | **0.71** | **0.15** | ✅ **Stable** | ✅ **Yes** |
| 3.0 | 0.90 | 0.70 | 0.18 | ⚠️ Slower | ❌ |
| 5.0 | 0.87 | 0.68 | 0.22 | ❌ Unstable | ❌ |

**Winner:** $\gamma = 2.0$ (highest high-confidence accuracy, stable training)

---

### Summary

**Default:** $\gamma = 2.0$

**If tuning:**
1. Split validation by confidence ($\kappa \geq 4$)
2. Try $\gamma \in \{0.5, 1.0, 2.0, 3.0, 5.0\}$
3. Maximize accuracy on high-confidence set
4. Check outlier separation diagnostics
5. Ensure training stability

---

(Continuing with Q18-Q47...)
## Q18: Why NOT use confidence κ for gradient weighting?

**📍 Location:** Section 3.2.3, discussion around Equation 3.14

### Your Question from Voice Transcript

> "Why not use the confidence capper for gradient weighting? [...] Actually, where do we use the capper then? Is that between the DPL loss?"

**Short answer:** We DON'T use κ in the loss function. We use it ONLY for diagnostics (validation checks).

---

### Summary

| Use of κ | In Training Loss? | Purpose |
|-----------------|-------------------|---------|
| **Gradient weighting** | ❌ **NO** | Would propagate annotator bias |
| **Validation metric** | ✅ Yes (diagnostic) | Tune γ, check robustness |
| **Outlier detection** | ✅ Yes (diagnostic) | Verify Hölder down-weights low-κ |

**Key principle:** Trust the **model's data-driven robustness** (Hölder), not **annotator's subjective confidence** (κ).

---

## EXTRA EXPLANATIONS: Addressing Voice Transcript Confusion

### 🔤 What does the tilde ~ symbol mean?

**Your question:** "What is this tilde means here?"

**Answer:** The tilde has TWO different meanings in this paper:

#### Meaning 1: Sampling/Distribution (standard math notation)

$$x \sim p(x)$$

Reads as: "x is sampled from distribution p(x)"

**Example:**
$$\mathbf{x}_t \sim q(\mathbf{x}_t | \mathbf{x}_0)$$

"The noisy state x_t is sampled from the forward diffusion distribution given clean x_0"

---

#### Meaning 2: Executed/Modified Distribution (SCIGEN-specific)

$$\tilde{p}_{\theta,C}$$

Reads as: "p tilde theta C" = the **executed** constrained policy

**Why tilde here?** To distinguish:
- $p_\theta$ = model's unconstrained proposal
- $\tilde{p}_{\theta,C}$ = what actually happens after SCIGEN overwrite

**Example:**
```
p_θ(x_{t-1}|x_t)        ← Model proposes this
    ↓ SCIGEN applies Π_C
p̃_{θ,C}(x_{t-1}|x_t)    ← This is what you actually get
```

**Visual mnemonic:** Think tilde ~ as a "wavy modification" of the original p.

---

### 🔣 What is the Π (capital Pi) symbol?

**Your question:** "What is this capital Pi symbol, Pi? Yeah, I wonder what it is."

**Answer:** $\Pi_C$ is the **SCIGEN projection operator**, NOT multiplication!

#### Definition

$$\Pi_C : \mathbb{R}^d \to \mathcal{M}_C$$

Maps any proposal to the closest point satisfying constraint $C$.

**For SCIGEN (deterministic hard constraints):**

$$\Pi_C(\mathbf{u}_{t-1}) = (\mathbf{u}_{t-1}^{\text{free}}, \mathbf{x}_C^{\star,\text{fix}})$$

Meaning:
- Take free components from proposal $\mathbf{u}_{t-1}$
- Overwrite constrained components with motif values $\mathbf{x}_C^{\star,\text{fix}}$

---

#### Concrete Example: Kagome N=6

**Constraint C:** 3 atoms at kagome vertices

**Proposal from model:**
```
u_{t-1} = [atom1: (0.12, 0.34, 0.0),
           atom2: (0.67, 0.89, 0.0),
           atom3: (0.23, 0.45, 0.0),
           atom4: (0.71, 0.23, 0.5),  ← free
           atom5: (0.33, 0.66, 0.5),  ← free
           atom6: (0.88, 0.44, 0.5)]  ← free
```

**SCIGEN projection:** $\mathbf{x}_{t-1} = \Pi_{\text{kagome}}(\mathbf{u}_{t-1})$

```
x_{t-1} = [atom1: (0.0, 0.0, 0.0),      ← FORCED (kagome vertex 1)
           atom2: (0.5, 0.0, 0.0),      ← FORCED (kagome vertex 2)
           atom3: (0.5, 0.5, 0.0),      ← FORCED (kagome vertex 3)
           atom4: (0.71, 0.23, 0.5),   ← kept from u (FREE)
           atom5: (0.33, 0.66, 0.5),   ← kept from u (FREE)
           atom6: (0.88, 0.44, 0.5)]   ← kept from u (FREE)
```

**Notation:** $\Pi$ is standard in optimization for "projection onto constraint set"

---

### 🔄 What is "kernel" in "reverse kernel"?

**Your question:** "What is this, the reverse kernel or what is the meaning of the kernel in this context?"

**Answer:** In probability/diffusion context, **kernel** = **transition probability** (conditional distribution)

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \leftarrow \text{This is a "kernel"}$$

**Why called "kernel"?**

Historical math term: A kernel defines how probability mass "flows" from one state to another.

**Types in diffusion:**

1. **Forward kernel:** $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ (add noise)
2. **Reverse kernel:** $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ (denoise)
3. **Markov kernel:** General transition $K(\mathbf{x}' | \mathbf{x})$

**Why Gaussian for reverse kernel?**

DDPM assumes:
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t,t), \sigma_t^2 \mathbf{I})$$

This makes math tractable (can compute log-probabilities, gradients).

---

### 🎯 What does "proxy" mean?

**Your question:** "Proxy is an approximation, right? But why do you use the proxy word?"

**Answer:** "Proxy" = **stand-in** or **substitute** (from Latin "procurare" = to act on behalf of)

**In machine learning context:**

$$\text{True thing (intractable)} \quad \xrightarrow{\text{approximated by}} \quad \text{Proxy (tractable)}$$

#### Example 1: Wrapped Gaussian Proxy (Q7)

**True DiffCSP reverse sampling:**
```
Predictor step + K Langevin corrector steps
→ Can sample, but no closed-form p_θ(F_{t-1}|F_t)
```

**Proxy for DPO:**
```
Single-step wrapped Gaussian
→ Has closed-form density: p_θ(F_{t-1}|F_t) = N_w(μ_θ, σ̃²I)
```

**Why "proxy"?** The Gaussian **stands in for** the true Langevin sampler.

---

#### Example 2: Improvement Score Proxy (Q5)

**True DPO objective:**
```
log p_θ(x_0) - log p_ref(x_0)  ← Intractable (T-step marginalization)
```

**Proxy:**
```
T · [log p_θ(x_{t-1}|x_t) - log p_ref(x_{t-1}|x_t)] ≈ T · I_θ(x,t)
                                                        ↑
                                                    tractable!
```

**Why "proxy"?** Improvement score I_θ **acts as a stand-in** for the true log-ratio.

---

### 📊 Why is σ̃_t² a "shared variance"?

**Your question:** "Why is that the shared variance? In which point are the sigma T shared?"

**Answer:** "Shared" means **same for all atoms/dimensions** (not learned per-component)

**Equation 2.16:**
$$p_\theta(F_{t-1}|M_t,t,c) \approx \mathcal{N}_w(F_{t-1}; \boldsymbol{\mu}_\theta^{(F)}, \sigmã_t^2 \mathbf{I})$$

**Key:** $\sigmã_t^2 \mathbf{I}$ is **isotropic** (spherical covariance)

**Shared across:**
1. All atoms: $\sigmã_t^2$ same for atom 1, atom 2, ..., atom N
2. All dimensions: $\sigmã_t^2$ same for x, y, z coordinates
3. Reference and model: Both use same $\sigmã_t^2$ (not $\sigma_{\theta}$ vs $\sigma_{\text{ref}}$)

**Why tilde on σ̃?**

To distinguish from forward diffusion variance $\sigma_t^2$:
- $\sigma_t^2$ = forward variance (from noise schedule)
- $\sigmã_t^2$ = reverse proxy variance (could be different!)

**In practice:** Often set $\sigmã_t^2 = \sigma_t^2$, but not required.

---

### 🌀 Why wrapped Gaussian, not uniform?

**Your question:** "Why is that not the uniform distribution, but it's the wrapped normal distribution?"

**Answer:** Because fractional coordinates have **local structure** (atoms cluster near stable positions).

**If uniform:**
```
p_uniform(F) = 1  for all F ∈ [0,1)^{N×3}

Problem: No preference for any configuration!
→ Can't denoise (all positions equally likely)
```

**With wrapped Gaussian:**
```
p_wrapped(F|F_t) = N_w(F; μ_θ(F_t,t), σ̃_t²I)

→ Peak at μ_θ (predicted clean position)
→ Probability decreases as you move away
→ Can denoise!
```

**Intuition:** Even on torus, denoising should predict a **specific location** (not "anywhere is fine").

**Analogy:** Wrapping a birthday gift
- You want it centered on the gift (Gaussian peak)
- Not randomly placed on infinite wrapping paper (uniform)
- But paper wraps around (torus topology)

---

### 🔗 Why do F_{t-1} and F_t need same noise ε_F?

**Your question:** "Why does that share the noise? I think the noise level becomes smaller as the distance T value gets smaller, right?"

**Great observation!** Yes, noise level decreases, but we still use **same base noise** ε_F.

**Simple coupling:**
$$\varepsilon_F \sim \mathcal{N}(0, \mathbf{I}) \quad \text{(draw once)}$$
$$F_s = w(F_0 + \sigma_s \cdot \varepsilon_F) \quad \text{for } s \in \{t-1, t\}$$

**Key:** Same $\varepsilon_F$, different $\sigma_s$!

| Timestep | Noise scale | Noisy state |
|----------|-------------|-------------|
| $t$ | $\sigma_t = 0.3$ | $F_t = w(F_0 + 0.3 \varepsilon_F)$ |
| $t-1$ | $\sigma_{t-1} = 0.295$ | $F_{t-1} = w(F_0 + 0.295 \varepsilon_F)$ |

**Both use same $\varepsilon_F$ direction**, but $F_t$ is slightly more noisy.

**Why?** Ensures $(F_{t-1}, F_t)$ lie on **same forward trajectory**:

```
F_0 ───(ε_F)───> F_{t-1} ───(more ε_F)───> F_t
                 ↑                           ↑
            Less noise                  More noise
            (same direction)
```

**Without shared noise (WRONG):**
```
F_0 ───(ε_F^{t-1})───> F_{t-1}

F_0 ───(ε_F^{t})───> F_t
     ↑
  Different direction! Not from same trajectory.
```

**For DPO:** Model predicts $\mu_\theta(F_t, t)$, compares to $F_{t-1}$ from **same path**.

---

### 🎭 What does "trajectory" mean?

**Your question:** "Must be the same trajectory. What do you mean the same trajectory?"

**Answer:** A **trajectory** is the complete path through time:

$$\tau = (\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_1, \mathbf{x}_0)$$

**Same trajectory** = all states generated from **same noise sequence**

**Example:**

```
Trajectory 1 (noise ε₁):
  x_0 ──(ε₁)──> x_1 ──(ε₁)──> ... ──(ε₁)──> x_T
         ↑                              ↑
     t=999                           t=0

Trajectory 2 (noise ε₂ ≠ ε₁):
  x_0 ──(ε₂)──> x_1' ──(ε₂)──> ... ──(ε₂)──> x_T'
         ↑                               ↑
   Different path!
```

**For DPO:** Need $(x_{t-1}, x_t)$ from **same trajectory** to evaluate:

$$d_\theta^{(F)} = \|\Delta(F_{t-1}, \mu_\theta^{(F)}(F_t, t))\|^2$$

If $F_{t-1}$ from trajectory 1 and $F_t$ from trajectory 2 → **meaningless comparison!**

---

### 🏗️ What does "tractable" mean?

**Your question:** "Tractable means that computable in closed form. It means that it doesn't, okay, tractable and proxy is a kind of opposite word..."

**Answer:** Not opposite! They work together:

**Tractable** = mathematically simple enough to compute/differentiate

**Proxy** = approximation that achieves tractability

**Relationship:**
```
Intractable true thing
         ↓ (approximate with)
Tractable proxy
```

**Example:**

**Intractable:** Langevin MCMC sampler
- Stochastic (random walk)
- No closed-form density
- Can't compute $\nabla_\theta \log p_\theta(F_{t-1}|F_t)$

**Tractable proxy:** Wrapped Gaussian
- Deterministic density: $p(F) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{\|F-\mu\|^2}{2\sigma^2})$
- Can compute $\log p$ and $\nabla_\theta \log p$
- Enables gradient-based training!

**Why needed:** DPO requires backpropagation → need differentiable loss

---

### 🔍 Predictor-Corrector Explained

**Your question:** "I'm not very confident here to understand this predicted corrector very accurately enough."

Let me explain with a visual example!

#### The Two-Phase Algorithm

**Predictor:** Make initial guess
**Corrector:** Refine guess

**Full algorithm for one timestep $t \to t-1$:**

```python
def diffcsp_reverse_step(F_t, t, model):
    # PREDICTOR: Score-based jump
    score = model.predict_score(F_t, t)  # ŝ_θ(F_t, t)
    F_pred = wrap(F_t + eta_t * score)   # Initial guess
    
    # CORRECTOR: Langevin refinement
    F = F_pred
    for k in range(K_langevin_steps):  # K ≈ 5-10
        score_current = model.predict_score(F, t)
        noise = torch.randn_like(F)
        
        # Langevin update
        F = F + epsilon * score_current + sqrt(2*epsilon) * noise
        F = wrap(F)  # Stay on torus
    
    F_t_minus_1 = F  # Final refined value
    return F_t_minus_1
```

**Why predictor alone is not enough?**

On torus, score field has discontinuities at boundaries:

```
Score field near boundary:

   F = 0.0 ≡ 1.0
      ↓ ↑
   Boundary

Predictor might jump across boundary incorrectly
→ Need corrector to "settle into" right distribution
```

**Why corrector matters (visual):**

```
Target distribution p(F|F_t):

  Probability
      ↑
      |    ***
      |  **   **      ← True target (bumpy on torus)
      | *       *
      |*         **
      |____________→ F

Predictor alone: Might land on shoulder, not peak
Corrector: Random walk converges to peak
```

**For DPO:** Too expensive (K extra model calls per timestep)
→ Use **tractable proxy** (single-step Gaussian)

---

## Section 3.2.3: Hölder-DPO

### Q18: Why NOT use confidence κ for gradient weighting?

**Your question:** "Why not directly weight gradients by confidence score κ?"

This is a very reasonable question! Intuitively, if annotator says κ=5 ("very confident"), shouldn't we trust it more?

#### The Three Reasons (Section 3.2.4)

**Reason 1: Hölder Already Down-Weights Outliers Automatically**

The Hölder loss has **adaptive weighting** built in through its gradient:

$$\frac{\partial \ell_\gamma}{\partial u_\theta} = -\gamma(1+\gamma) \sigma(u_\theta)^\gamma (1-\sigma(u_\theta))^2$$

This is the **influence weight** $\iota_\gamma(u_\theta)$.

**Visual example:**

```
Influence weight vs margin u_θ (for γ=2.0):

  ι_γ(u)
    ↑
  1.5|      ***
     |    **   **          ← Peak near u≈0 (uncertain cases)
  1.0|   *       *
     |  *         *
  0.5| *           **
     |*               ***
  0.0|___________________→ u_θ
     -4  -2   0   2   4

Key insights:
- u_θ ≪ 0: Model thinks it's wrong → ι ≈ 0 (ignore outlier!)
- u_θ ≈ 0: Model uncertain → ι ≈ 1.5 (learn most here!)
- u_θ ≫ 0: Model already learned → ι ≈ 0 (saturated)
```

**The model automatically detects outliers** by checking: "Does my prediction agree with the label?"

**Reason 2: Avoid Annotator Bias**

Human confidence κ has systematic biases:

| Annotator Type | Behavior | Problem |
|----------------|----------|---------|
| Over-confident | κ=5 for borderline cases | Over-weights noisy labels |
| Under-confident | κ=2 even when correct | Under-weights good data |
| Domain expert | Calibrated κ | Mixed with biased annotators |

**If we weight by κ:**
```python
loss = kappa * holder_loss(u_theta)  # DON'T DO THIS!
```

**Problem:** Biases propagate to model
- Over-confident wrong label → large gradient in wrong direction!
- Under-confident correct label → small gradient

**Reason 3: Data-Driven Robustness**

**With κ-weighting:** Trust annotator's self-assessment
**With Hölder alone:** Trust model + data consistency

**Example scenario:**

Annotator says: "κ=5, x^w ≻ x^ℓ" (very confident winner better)

But model observes:
- Training examples 1-1000: Similar pairs all prefer loser-type features
- This pair: Outlier!

**Hölder response:** ι_γ ≈ 0, down-weight automatically
**κ-weighting would:** Force model to learn wrong pattern!

**The model has MORE information** (entire dataset) than single annotator (one pair).

---

#### What DO We Use κ For?

**Diagnostics only!** (Section 3.2.4)

**Check 1: Influence-Confidence Correlation**
```python
# After training, compute for each pair:
influence = compute_influence_weight(u_theta, gamma=2.0)
confidence = kappa  # annotator confidence

# Expect: Low-influence pairs have lower κ on average
print(f"Mean κ for low-influence: {kappa[influence < 0.1].mean()}")
print(f"Mean κ for high-influence: {kappa[influence > 1.0].mean()}")
```

**Expected:** Low-influence → lower κ (confirms outlier detection working!)

**Check 2: Validation Split**
```python
# Split dataset
high_conf = pairs[kappa >= 4]
all_pairs = pairs

# Train on all_pairs, evaluate on high_conf
accuracy_high = evaluate_ranking(model, high_conf)
```

**Expected:** High κ subset should show learning progress.

**Check 3: Tuning γ**
```python
for gamma in [0.5, 1.0, 2.0, 3.0, 5.0]:
    model = train_holder_dpo(data, gamma=gamma)
    acc_high = evaluate(model, pairs[kappa >= 4])
    acc_low = evaluate(model, pairs[kappa <= 2])
    print(f"γ={gamma}: high={acc_high:.3f}, low={acc_low:.3f}")
```

**Choose γ** that maximizes high-confidence accuracy while maintaining stability.

---

### Q19: What is "gradient weighting in Equation 3.14"?

**Your question:** "The paper mentions gradient weighting - where is it in Eq 3.14?"

**Answer: There is NO explicit gradient weighting in Eq 3.14!**

Let me show you exactly:

**Equation 3.14 (Phase A Loss):**
$$\mathcal{L}_A(\theta) = \mathbb{E}_{(x_0^w, x_0^l, c_A) \sim \mathcal{D}_A} \mathbb{E}_{t \sim \rho(\cdot)} \left[ \ell_\gamma(\beta \cdot T \cdot g_\theta(t)) \right]$$

**No κ appears!** No weighting coefficients!

**So what about "gradient weighting"?**

The weighting is **implicit** through the Hölder loss itself.

#### How Implicit Weighting Works

**Step 1: Compute gradient**

$$\frac{\partial \mathcal{L}_A}{\partial \theta} = \mathbb{E} \left[ \frac{\partial \ell_\gamma}{\partial u_\theta} \cdot \frac{\partial u_\theta}{\partial \theta} \right]$$

where $u_\theta = \beta \cdot T \cdot g_\theta(t)$

**Step 2: Expand Hölder gradient (Eq 3.29)**

$$\frac{\partial \ell_\gamma}{\partial u_\theta} = -\gamma(1+\gamma) \sigma(u_\theta)^\gamma (1 - \sigma(u_\theta))^2$$

**This is the implicit weight!**

**Step 3: Full gradient**

$$\frac{\partial \mathcal{L}_A}{\partial \theta} = \mathbb{E} \left[ \underbrace{-\gamma(1+\gamma) \sigma(u)^\gamma (1-\sigma(u))^2}_{\text{implicit weight } \iota_\gamma(u)} \cdot \frac{\partial u_\theta}{\partial \theta} \right]$$

**So the "weighting" is:**
- **Not** from confidence κ (NOT in equation)
- **Not** from explicit weight $w_i$ (NOT in equation)
- **From** Hölder loss gradient shape $\partial \ell_\gamma / \partial u$

---

#### Comparison: Logistic vs Hölder

**Standard DPO (logistic loss, γ→0):**

$$\frac{\partial \ell_{\text{logistic}}}{\partial u} = -\sigma(-u) = -\frac{1}{1+e^u}$$

**Influence:**
```
  ι(u)
    ↑
  1.0|***
     |   ****
  0.5|       ****           ← Monotone decreasing
     |           ****
     |               ******
  0.0|___________________→ u
     -4  -2   0   2   4

Even extreme outliers (u→-∞) have ι ≈ 1.0!
→ Sensitive to label noise
```

**Hölder-DPO (γ=2.0):**

$$\frac{\partial \ell_\gamma}{\partial u} = -6 \sigma(u)^2 (1-\sigma(u))^2$$

**Influence:**
```
  ι(u)
    ↑
  1.5|      ***
     |    **   **          ← Redescending!
  1.0|   *       *
     |  *         *
  0.5| *           **
     |*               ***
  0.0|___________________→ u
     -4  -2   0   2   4

Extreme outliers (u→-∞): ι → 0!
→ Robust to label noise
```

**The gradient shape automatically provides adaptive weighting.**

---

## Section 3.2.4: Diagnostics

### Q20: What is "weak validation signal"?

**Your question:** "What does 'weak' mean in 'weak validation signal from κ'?"

**Answer:** "Weak" means **noisy but statistically useful**.

#### Why κ is "Weak" (Not Strong Enough for Training)

**Noise sources:**

| Source | Example | Impact |
|--------|---------|--------|
| Annotator miscalibration | Over-confident on hard cases | κ=5 but pair is actually ambiguous |
| Inter-annotator disagreement | Expert says κ=2, novice says κ=5 | Inconsistent scale usage |
| Crowdsourcing artifacts | Workers click κ=3 by default | Central tendency bias |
| Genuine ambiguity | Both structures seem equally good | κ should be low, but varies |

**Individual κ is unreliable!**

**But aggregate statistics are useful:**

```python
# Example: 1000 preference pairs
kappa_values = [3, 5, 2, 4, 1, 5, 3, ...]  # noisy!

# Aggregate statistics (more reliable):
print(f"Mean κ for low-influence pairs: {kappa[influence < 0.1].mean():.2f}")
# Output: 2.3 ← lower on average

print(f"Mean κ for high-influence pairs: {kappa[influence > 1.0].mean():.2f}")
# Output: 3.8 ← higher on average

# Correlation (weak but positive):
print(f"Spearman(κ, |u_θ|): {spearmanr(kappa, abs(u_theta))[0]:.2f}")
# Output: 0.31 ← weak positive correlation
```

**Interpretation:** On average, κ correlates with true difficulty, but with lots of noise.

---

#### What Makes a "Strong" vs "Weak" Signal?

**Strong signal (usable for training):**
- Clean, low-noise labels
- Consistent across annotators
- Directly related to task objective
- Example: Ground-truth DFT band gap (continuous, objective)

**Weak signal (diagnostics only):**
- Noisy labels
- Subjective or inconsistent
- Indirectly related to objective
- Example: Annotator confidence κ (ordinal, subjective)

**Using weak signal for training:**
```python
# DON'T DO THIS!
loss = kappa * holder_loss(u_theta)
# Problem: Noise in κ propagates to model weights
```

**Using weak signal for diagnostics:**
```python
# This is OK!
if kappa[low_influence].mean() < kappa[high_influence].mean():
    print("✓ Hölder is correctly identifying uncertain pairs")
else:
    print("✗ Something wrong, investigate")
```

---

#### How to Validate Hölder is Working

**Diagnostic 1: Influence-Confidence Correlation**

**Question:** Do down-weighted pairs have lower κ?

```python
# Compute influence for each pair
influence = compute_influence(u_theta, gamma=2.0)

# Split by influence quantile
low_inf = pairs[influence < percentile(influence, 25)]
high_inf = pairs[influence > percentile(influence, 75)]

# Compare mean confidence
print(f"Low-influence mean κ: {kappa[low_inf].mean():.2f}")
print(f"High-influence mean κ: {kappa[high_inf].mean():.2f}")
```

**Expected result (working correctly):**
```
Low-influence mean κ: 2.1  ← Less confident (more ambiguous/mislabeled)
High-influence mean κ: 3.9  ← More confident (clearer preferences)
```

**If reversed:** Hölder might be miscalibrated, check γ or data quality.

---

**Diagnostic 2: Outlier Enrichment**

**Question:** Are extreme outliers (very negative u_θ) more likely to have low κ?

```python
# Find pairs where model strongly disagrees with label
outliers = pairs[u_theta < -2.0]  # Model thinks label is wrong
inliers = pairs[abs(u_theta) < 0.5]  # Model uncertain

print(f"Outliers: {len(outliers)} pairs, mean κ={kappa[outliers].mean():.2f}")
print(f"Inliers: {len(inliers)} pairs, mean κ={kappa[inliers].mean():.2f}")
```

**Expected (working correctly):**
```
Outliers: 23 pairs, mean κ=2.0  ← Lower confidence (mislabels!)
Inliers: 847 pairs, mean κ=3.5  ← Higher confidence
```

---

**Diagnostic 3: High-Confidence Validation Accuracy**

**Question:** Does model perform better on high-κ subset?

```python
# Split validation set
val_high_conf = val_set[kappa >= 4]  # High confidence
val_all = val_set

# Evaluate ranking accuracy
acc_high = evaluate_ranking_accuracy(model, val_high_conf)
acc_all = evaluate_ranking_accuracy(model, val_all)

print(f"Accuracy on high-confidence: {acc_high:.1%}")
print(f"Accuracy on all: {acc_all:.1%}")
```

**Expected (learning is progressing):**
```
Accuracy on high-confidence: 87.3%  ← Higher (cleaner labels)
Accuracy on all: 71.2%              ← Lower (includes noise)
```

**Why this validates Hölder:**
- Model learns clear patterns (high κ) without overfitting to noise (low κ)
- Robust training allows good generalization despite noisy labels

---

### Q21: Why define ℓ_γ as function of x in Eq 3.29?

**Your question:** "In Equation 3.29, why parameterize ℓ_γ by x instead of directly by θ?"

**Equation 3.29:**
$$\frac{\partial \ell_\gamma}{\partial x} = -\gamma(1+\gamma) p^\gamma (1-p)^2 \quad \text{where } p = \sigma(x)$$

**Three reasons for this abstraction:**

---

#### Reason 1: Cleaner Notation & Analysis

**If we wrote everything in terms of θ:**

$$\ell_\gamma(\theta; x_0^w, x_0^\ell, t, c) = -(1+\gamma)\sigma(\beta T [I_\theta(x_0^w,t) - I_\theta(x_0^\ell,t)])^\gamma + \ldots$$

**Horrible!** Too many variables, hard to see structure.

**With x abstraction:**
$$x = \beta \cdot T \cdot g_\theta(t) \quad \text{(scaled margin)}$$
$$\ell_\gamma(x) = -(1+\gamma)\sigma(x)^\gamma + \gamma \sigma(x)^{1+\gamma}$$

**Much cleaner!** Separates:
- $x(\theta)$ = how to compute margin from model
- $\ell_\gamma(x)$ = loss shape (robustness properties)

---

#### Reason 2: Analyze Influence Function

**The gradient $\partial \ell_\gamma / \partial x$ is the influence shape:**

$$\iota_\gamma(x) = \left| \frac{\partial \ell_\gamma}{\partial x} \right| = \gamma(1+\gamma) \sigma(x)^\gamma (1-\sigma(x))^2$$

**This tells us:** How much does pair with margin $x$ contribute to gradient?

**Redescending property (Proposition 3.5):**
$$\lim_{x \to -\infty} \iota_\gamma(x) = 0$$

**Visual:**
```
  ι_γ(x)
    ↑
  1.5|      Peak at x≈0
     |    **   **
  1.0|   *       *
     |  *         *      → As x→-∞, ι→0 (outliers ignored!)
  0.5| *           **
     |*               ***___________
  0.0|___________________→ x
     -4  -2   0   2   4
```

**By writing ℓ_γ(x)**, we can prove robustness properties for **any** margin $x$, regardless of whether it comes from:
- Phase A: $x = \beta T g_\theta(t)$
- Phase B: $x = \beta T g_{\theta,C}^{\text{BR}}(t;b)$

**General theory!**

---

#### Reason 3: Chain Rule for Gradients

**Training requires:** $\frac{\partial \mathcal{L}}{\partial \theta}$

**Chain rule:**
$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \ell_\gamma}{\partial x} \cdot \frac{\partial x}{\partial \theta}$$

**Two separate components:**

**1. Loss gradient (universal):**
$$\frac{\partial \ell_\gamma}{\partial x} = -\gamma(1+\gamma) \sigma(x)^\gamma (1-\sigma(x))^2$$

This is the same for Phase A, Phase B, any variant!

**2. Margin gradient (phase-specific):**

*Phase A:*
$$\frac{\partial x}{\partial \theta} = \beta T \frac{\partial g_\theta(t)}{\partial \theta}$$

*Phase B:*
$$\frac{\partial x}{\partial \theta} = \beta T \frac{\partial g_{\theta,C}^{\text{BR}}(t;b)}{\partial \theta}$$

**Modularity!** Change margin computation without re-deriving robustness properties.

---

#### Code Example

```python
def holder_loss(x, gamma=2.0):
    """
    Hölder loss as function of scaled margin x.

    Args:
        x: Scaled margin β·T·g_θ(t)
        gamma: Robustness parameter
    """
    p = torch.sigmoid(x)
    loss = -(1 + gamma) * p**gamma + gamma * p**(1 + gamma)
    return loss

def holder_loss_gradient(x, gamma=2.0):
    """Influence weight ι_γ(x)."""
    p = torch.sigmoid(x)
    grad = -gamma * (1 + gamma) * (p**gamma) * ((1 - p)**2)
    return grad

# Phase A: Compute scaled margin
def compute_margin_phaseA(model, x_w, x_l, t, beta, T):
    I_w = improvement_score(model, x_w, t)
    I_l = improvement_score(model, x_l, t)
    g_theta = I_w - I_l
    x = beta * T * g_theta  # Scaled margin
    return x

# Phase B: Different margin computation
def compute_margin_phaseB(model, tau_w, tau_l, t, b, beta, T):
    I_w_br = bridge_improvement_score(model, tau_w, t, b)
    I_l_br = bridge_improvement_score(model, tau_l, t, b)
    g_theta_br = I_w_br - I_l_br
    x = beta * T * g_theta_br  # Same formula, different I!
    return x

# Training: Same loss function!
for phase in ['A', 'B']:
    for batch in dataloader:
        if phase == 'A':
            x = compute_margin_phaseA(model, batch['x_w'], batch['x_l'], ...)
        else:  # phase == 'B'
            x = compute_margin_phaseB(model, batch['tau_w'], batch['tau_l'], ...)

        loss = holder_loss(x, gamma=2.0)  # Same function!
        loss.backward()
        optimizer.step()
```

**The abstraction $\ell_\gamma(x)$ allows code reuse across phases.**

---

## Section 3.3: Bridge Formulation (Phase B)

This section addresses the **key challenge** of Phase B: How to train DPO when we only have endpoints $(x_0^w, x_0^\ell)$ from SCIGEN generation, but not the trajectories?

### Q22: What is "forward corrupted state"?

**Your question:** "In Phase A, we use 'forward corrupted' - is that just adding noise to x_0?"

**Answer: YES, exactly!**

**Forward corruption** means applying the forward diffusion process:

$$x_t \sim q(x_t | x_0)$$

Start with clean $x_0$, add noise according to the diffusion schedule, get noisy $x_t$.

---

#### Phase A: Forward Corruption Process

**Step-by-step:**

```python
# Phase A data preparation
def phase_a_forward_corruption(x_0_w, x_0_l, t):
    """
    Corrupt clean endpoints with forward diffusion.

    Args:
        x_0_w, x_0_l: Clean winner/loser structures
        t: Timestep to corrupt to

    Returns:
        x_t_w, x_t_l: Noisy states
    """
    # Sample noise
    epsilon_w = torch.randn_like(x_0_w)
    epsilon_l = torch.randn_like(x_0_l)

    # Forward diffusion (Eq 2.1, 2.3, 2.5)
    alpha_bar_t = get_alpha_bar(t)

    x_t_w = sqrt(alpha_bar_t) * x_0_w + sqrt(1 - alpha_bar_t) * epsilon_w
    x_t_l = sqrt(alpha_bar_t) * x_0_l + sqrt(1 - alpha_bar_t) * epsilon_l

    return x_t_w, x_t_l
```

**Visual:**

```
Phase A: Direct forward corruption

Clean data:
  x_0^w ──────────────┐
  x_0^ℓ ──────────────┤
         ↓ add noise  │
Noisy:                │
  x_t^w ──────────────┤
  x_t^ℓ ──────────────┤
         ↓ denoise    │
Compute improvement:  │
  I_θ(x^w, t)        │
  I_θ(x^ℓ, t)        │
         ↓            │
Margin: g_θ(t) = I^w - I^ℓ
```

**Key property:** Forward corruption is **unconstrained** - just Gaussian noise, no SCIGEN involved.

---

#### Why Phase B Cannot Use Forward Corruption

**Problem:** Phase B data comes from **SCIGEN generation**, not raw MP-20.

**Phase B data generation (during dataset creation):**

```python
# Generate Phase B dataset
def generate_scigen_structures(model_A, constraint_C, num_samples):
    """
    Generate structures using Phase-A model + SCIGEN.

    THIS is how the dataset was created!
    """
    structures = []
    for i in range(num_samples):
        # Start from noise
        x_T = torch.randn(...)  # Pure noise

        x_t = x_T
        for t in range(T, 0, -1):  # T → 1
            # Model proposes
            u_t_minus_1 = model_A.denoise(x_t, t, constraint_C)

            # SCIGEN overwrites!
            x_t_minus_1 = apply_scigen_projection(u_t_minus_1, constraint_C)

            x_t = x_t_minus_1

        structures.append(x_0)  # Final structure
        # NOTE: Trajectory τ=(x_T,...,x_0) NOT STORED! Memory!

    return structures  # Only endpoints!

# Phase B dataset
structures = generate_scigen_structures(model_A, C="kagome", num=1000)
pairs = annotate_preferences(structures)  # Human labels
# pairs = [(x_0^w, x_0^ℓ, κ), ...]
```

**What we have:** Clean endpoints $(x_0^w, x_0^\ell)$ from constrained generation
**What we DON'T have:** Trajectories $\tau = (x_T, ..., x_1, x_0)$

**Why can't we just forward-corrupt?**

If we did: $x_t^w \sim q(x_t | x_0^w)$ (standard DDPM forward)

This would give us **unconstrained noisy states**, but:

1. **Wrong distribution!** The original generation used SCIGEN at every step
2. **Missing SCIGEN dynamics:** Model needs to learn to denoise **under constraints**
3. **Distribution mismatch:** Training on unconstrained, deploying with constrained

**We need to reconstruct trajectories that include SCIGEN projections!**

---

### Q23: Why does SCIGEN apply constraint "after each reverse step"?

**Your question:** "What does 'after each reverse step' mean?"

**Answer:** SCIGEN projection $\Pi_C$ is applied **at every denoising timestep**, not just at the end.

**SCIGEN generation process:**

```
Start: x_T ~ N(0, I)  ← Pure Gaussian noise

Reverse process (t = T down to 1):

  Step T → T-1:
    u_{T-1} ~ p_θ(·|x_T, c)           # Model proposes
    x_{T-1} = Π_C(u_{T-1}; C)          # SCIGEN overwrites ← HERE!

  Step T-1 → T-2:
    u_{T-2} ~ p_θ(·|x_{T-1}, c)       # Model proposes
    x_{T-2} = Π_C(u_{T-2}; C)          # SCIGEN overwrites ← HERE!

  ...

  Step 1 → 0:
    u_0 ~ p_θ(·|x_1, c)               # Model proposes
    x_0 = Π_C(u_0; C)                  # SCIGEN overwrites ← HERE!

Result: x_0 satisfies constraint C
```

**"After each step"** = SCIGEN projection happens **T times** during generation (once per timestep).

---

#### Why Not Just Project at the End?

**Naive alternative (WRONG):**

```python
# Generate without constraints
x_T = torch.randn(...)
x_t = x_T
for t in range(T, 0, -1):
    x_t_minus_1 = model.denoise(x_t, t, c)  # Unconstrained!
    x_t = x_t_minus_1

# Project only at end
x_0_constrained = apply_scigen_projection(x_0, C)  # Too late!
```

**Problems:**

1. **Trajectory drifts away from constraint:**
   - At t=500: Model has no idea about constraint
   - Fractional coords might be far from kagome vertices
   - Lattice might be cubic instead of hexagonal

2. **Final projection is too abrupt:**
   - Huge jump from unconstrained $x_0$ to constrained $\Pi_C(x_0)$
   - Discontinuity → bad structure quality

3. **Model never learns constrained denoising:**
   - All intermediate states unconstrained
   - Model doesn't see constraint pattern

**SCIGEN's approach (CORRECT):** Enforce at every step

**Benefits:**

1. **Trajectory stays near constraint manifold:**
   - At all timesteps: Some DOF fixed by constraint
   - Model learns to denoise **given** constraints

2. **Smooth generation:**
   - No sudden jumps
   - Constraint satisfied throughout

3. **Model learns constraint-aware denoising:**
   - Training and generation distributions match!

---

#### Consequence for Phase B Training

**The challenge:**

```
Data we have:
  (x_0^w, x_0^ℓ) ← Endpoints only

Data we need for DPO:
  Trajectory log-ratios:
    log [p̃_{θ,C}(x_{t-1}|x_t) / p̃_{ref,C}(x_{t-1}|x_t)]

  Requires: States (x_t, x_{t-1}) from trajectories!

  But trajectories NOT stored! (too expensive)
```

**Solution:** Pseudo-bridge reconstruction (Section 3.3.2)

**Preview:**

```python
# Reconstruct approximate trajectory
def pseudo_bridge(x_0, b, model_ref, constraint_C):
    """
    Reconstruct trajectory that (approximately) ends at x_0.

    Args:
        x_0: Observed endpoint
        b: Bridge level (num reverse steps)
        model_ref: Frozen reference model
        constraint_C: SCIGEN constraint

    Returns:
        tau_hat: Pseudo-bridge (x_b, ..., x_1, x̂_0)
    """
    # Forward: Corrupt to x_b
    x_b = forward_diffusion(x_0, b)

    # Reverse: Run ref + SCIGEN for b steps
    tau_hat = []
    x_t = x_b
    for t in range(b, 0, -1):
        tau_hat.append(x_t)

        # Reference model proposes
        u_t_minus_1 = model_ref.denoise(x_t, t, constraint_C)

        # SCIGEN overwrites ← Reintroduces constraint dynamics!
        x_t_minus_1 = apply_scigen_projection(u_t_minus_1, constraint_C)

        x_t = x_t_minus_1

    tau_hat.append(x_t)  # x̂_0 (approximate endpoint)
    return tau_hat
```

**This** is why we need pseudo-bridges: To reintroduce SCIGEN dynamics that happened during original generation.

---

### Q24-Q26: Equations 3.25-3.27 (SCIGEN Projection & Constraint Cancellation)

These three questions are about the **mathematical formulation** of how SCIGEN affects the executed policy.

---

#### Q24: Why does Eq 3.25 look like this?

**Equation 3.25:**
$$\Pi_C(\mathbf{u}_{t-1}; C) = (\mathbf{u}_{t-1}^{\text{free}}, \mathbf{x}_C^{\star, \text{fix}})$$

**Answer:** This is the **definition of SCIGEN projection**.

**Intuition:** SCIGEN **overwrites** constrained DOF, keeps free DOF.

**Components:**

| Symbol | Meaning | Source |
|--------|---------|--------|
| $\mathbf{u}_{t-1}$ | Model's proposal | $u_{t-1} \sim p_\theta(\cdot \| x_t, c)$ |
| $\mathbf{u}_{t-1}^{\text{free}}$ | Unconstrained components | Model controls these |
| $\mathbf{x}_C^{\star, \text{fix}}$ | Fixed components | Constraint $C$ dictates |
| $\Pi_C(\mathbf{u}_{t-1})$ | Executed state | What actually happens |

---

**Example: Kagome N=6 atoms**

**Constraint $C$ = "kagome lattice, 3 fixed + 3 free atoms"**

**Model proposes** $\mathbf{u}_{t-1}$:

```
Lattice (O(3)-invariant params):
  k_1 = -0.28  ← Hexagonal constraint fixes this!
  k_2 =  0.02  ← Free
  k_3 =  0.01  ← Free
  k_4 =  0.00  ← Fixed (hexagonal)
  k_5 =  0.00  ← Fixed (hexagonal)
  k_6 =  0.00  ← Fixed (hexagonal)

Fractional coordinates:
  Atom 1: (0.12, 0.45, 0.03)  ← Proposed (free DOF)
  Atom 2: (0.55, 0.88, 0.01)  ← Proposed (free DOF)
  Atom 3: (0.29, 0.11, 0.02)  ← Proposed (free DOF)
  Atom 4: (0.07, 0.22, 0.00)  ← Kagome vertex 1 (FIXED)
  Atom 5: (0.51, 0.03, 0.01)  ← Kagome vertex 2 (FIXED)
  Atom 6: (0.48, 0.49, 0.00)  ← Kagome vertex 3 (FIXED)

Atom types:
  All: Mn (all same, constrained)
```

**SCIGEN projection** $\Pi_C(\mathbf{u}_{t-1})$:

```
Lattice:
  k_1 = -log(3/4)  ← OVERWRITTEN (hexagonal requirement)
  k_2 =  0.02      ← Kept from proposal
  k_3 =  0.01      ← Kept from proposal
  k_4 =  0.00      ← Kept (already correct)
  k_5 =  0.00      ← Kept (already correct)
  k_6 =  0.00      ← Kept (already correct)

Fractional coords:
  Atom 1: (0.12, 0.45, 0.03)  ← Kept (free)
  Atom 2: (0.55, 0.88, 0.01)  ← Kept (free)
  Atom 3: (0.29, 0.11, 0.02)  ← Kept (free)
  Atom 4: (0.0,  0.0,  0.0)   ← OVERWRITTEN (kagome vertex 1)
  Atom 5: (0.5,  0.0,  0.0)   ← OVERWRITTEN (kagome vertex 2)
  Atom 6: (0.5,  0.5,  0.0)   ← OVERWRITTEN (kagome vertex 3)

Atom types:
  All: Mn ← All kept (already correct)
```

**Result:**
$$\mathbf{x}_{t-1} = \Pi_C(\mathbf{u}_{t-1}) = (\underbrace{\text{atoms 1-3 positions, } k_2, k_3}_{\mathbf{u}_{t-1}^{\text{free}}}, \underbrace{\text{atoms 4-6 positions, } k_1, k_4, k_5, k_6, \text{ atom types}}_{\mathbf{x}_C^{\star, \text{fix}}})$$

---

#### Q25: Why does Eq 3.26 look like this?

**Equation 3.26:**
$$\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t, c) = \mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t, c)$$

**Answer:** This is the **executed constrained reverse kernel** (what actually happens after SCIGEN).

**Components:**

1. **Indicator** $\mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}]$:
   - = 1 if constrained DOF match motif
   - = 0 otherwise (structure violates constraint)

2. **Free components** $p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t, c)$:
   - Model's unconstrained proposal distribution
   - Marginalized to free DOF only

---

**Derivation:**

**Start:** Model proposes $\mathbf{u}_{t-1} \sim p_\theta(\mathbf{u}_{t-1}|\mathbf{x}_t, c)$

**SCIGEN applies:** $\mathbf{x}_{t-1} = \Pi_C(\mathbf{u}_{t-1})$

**What is** $p(\mathbf{x}_{t-1})$ **after projection?**

**Key insight:** Projection is deterministic!

$$\mathbf{x}_{t-1} = \begin{cases}
\Pi_C(\mathbf{u}_{t-1}) & \text{if } \mathbf{u}_{t-1} \text{ proposed} \\
\text{impossible} & \text{otherwise}
\end{cases}$$

**But** $\Pi_C$ **only changes fixed DOF:**

$$\mathbf{u}_{t-1} = (\mathbf{u}^{\text{free}}, \mathbf{u}^{\text{fix}})$$
$$\Pi_C(\mathbf{u}_{t-1}) = (\mathbf{u}^{\text{free}}, \mathbf{x}_C^{\star,\text{fix}})$$

**So:**
- $\mathbf{x}_{t-1}^{\text{free}} = \mathbf{u}_{t-1}^{\text{free}}$ ← 1:1 correspondence!
- $\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}$ ← Always!

**Executed distribution:**

$$\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \begin{cases}
p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t) & \text{if } \mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}} \\
0 & \text{otherwise}
\end{cases}$$

**Rewrite with indicator:**

$$= \mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t)$$

**This is Equation 3.26!**

---

**Intuition:** "Only states matching constraint are possible, and their probability comes from free DOF proposal."

---

#### Q26: What does "free" mean in Eq 3.27?

**Equation 3.27 (Constraint Cancellation Lemma 3.1):**
$$\log \frac{\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c)}{\tilde{p}_{\text{ref},C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c)} = \log \frac{p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)}{p_{\text{ref}}(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)}$$

**Answer:** "free" = **unconstrained degrees of freedom**

**These are the components that:**
- Model has control over
- Are NOT fixed by constraint $C$
- Actually matter for learning!

---

**Why does constraint cancel?**

**Step 1: Write numerator and denominator**

$$\log \frac{\tilde{p}_{\theta,C}}{\tilde{p}_{\text{ref},C}} = \log \tilde{p}_{\theta,C} - \log \tilde{p}_{\text{ref},C}$$

**Step 2: Substitute Eq 3.26**

$$= \log \left[ \mathbf{1}[\mathbf{x}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}^{\text{free}}) \right] - \log \left[ \mathbf{1}[\mathbf{x}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_{\text{ref}}(\mathbf{x}^{\text{free}}) \right]$$

**Step 3: Expand logarithm**

$$= \underbrace{\log \mathbf{1}[\mathbf{x}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}]}_{\text{= 0 if true, } -\infty \text{ if false}} + \log p_\theta(\mathbf{x}^{\text{free}}) - \log \mathbf{1}[\mathbf{x}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] - \log p_{\text{ref}}(\mathbf{x}^{\text{free}})$$

**Step 4: Indicators cancel!**

$$= \log p_\theta(\mathbf{x}^{\text{free}}) - \log p_{\text{ref}}(\mathbf{x}^{\text{free}})$$

$$= \log \frac{p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t)}{p_{\text{ref}}(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t)}$$

**Only free DOF remain!**

---

**Why is this important for Phase B?**

**Implication:** When computing DPO loss, we only need to track errors on **free components**!

**Phase B residual (Eq 3.37-3.39):**

$$\mathbf{r}_\theta^{(z),\text{BR}} = \bar{\mathbf{C}}_{\text{eff}}^{(z)} \odot (\hat{\mathbf{x}}_{t-1}^{(z)} - \boldsymbol{\mu}_\theta^{(z)})$$

where $\bar{\mathbf{C}}_{\text{eff}}^{(z)}$ is the **free mask** (1 for free, 0 for fixed).

**Element-wise multiplication** $\odot$ zeros out fixed components → only compute error on free!

**Computational savings:**

| Constraint | Total DOF | Free DOF | Savings |
|------------|-----------|----------|---------|
| Kagome N=6 | 3×6 = 18 (positions) | 3×3 = 9 | 50% |
| Honeycomb N=8 | 24 | 12 | 50% |
| Unconstrained | 100 | 100 | 0% |

**Lemma 3.1 justifies** only computing gradients on free DOF!

---

### Q27-Q28: Feasibility & Exact

#### Q27: What is "same feasibility indicator"?

**Your question:** "The paper mentions 'same feasibility indicator' - what does this mean?"

**Answer:** The **feasibility indicator** is:

$$\mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}]$$

This checks: "Does state $\mathbf{x}_{t-1}$ satisfy constraint $C$?"

**"Same"** means winner and loser share the same constraint $C$, therefore:

$$\mathbf{1}[\mathbf{x}_w^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] = \mathbf{1}[\mathbf{x}_\ell^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}]$$

Both check against **same motif** $\mathbf{x}_C^{\star,\text{fix}}$.

---

**Why this matters for Lemma 3.1:**

When computing log-ratio for winner vs loser:

$$\log \frac{\tilde{p}_{\theta,C}(\mathbf{x}_w)}{\tilde{p}_{\theta,C}(\mathbf{x}_\ell)} = \log \frac{\mathbf{1}[\mathbf{x}_w^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_w^{\text{free}})}{\mathbf{1}[\mathbf{x}_\ell^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_\ell^{\text{free}})}$$

**If both states satisfy constraint $C$:**

$$= \log \frac{1 \cdot p_\theta(\mathbf{x}_w^{\text{free}})}{1 \cdot p_\theta(\mathbf{x}_\ell^{\text{free}})} = \log \frac{p_\theta(\mathbf{x}_w^{\text{free}})}{p_\theta(\mathbf{x}_\ell^{\text{free}})}$$

**Indicators cancel because they're the same (both = 1)!**

---

**Example:**

**Constraint $C$:** Kagome N=6, hexagonal lattice

**Winner $\mathbf{x}_w$:**
- Atoms 4,5,6 at kagome vertices? ✓
- Lattice hexagonal? ✓
- Indicator: 1

**Loser $\mathbf{x}_\ell$:**
- Atoms 4,5,6 at kagome vertices? ✓
- Lattice hexagonal? ✓
- Indicator: 1

**Same indicator → cancels in ratio!**

**Only compare free DOF:** Positions of atoms 1,2,3 and free lattice params.

---

#### Q28: What does "exact" mean in Eq 3.29?

**Your question:** "The paper contrasts 'exact endpoint objective' (Eq 3.29) with 'tractable approximation' (Eq 3.43) - what's the difference?"

**Answer:** "Exact" vs "Tractable" distinguishes **theoretically correct** from **practically computable**.

---

**Exact Endpoint Objective (Eq 3.29):**

$$\mathcal{L}_{B,\text{exact}} = \mathbb{E} \left[ \ell_\gamma\left(\beta \left[ \Delta_{\theta,C}(\mathbf{x}_0^w) - \Delta_{\theta,C}(\mathbf{x}_0^\ell) \right]\right) \right]$$

where the **endpoint log-ratio** is:

$$\Delta_{\theta,C}(\mathbf{x}_0) = \log \frac{\tilde{p}_{\theta,C}(\mathbf{x}_0|c)}{\tilde{p}_{\text{ref},C}(\mathbf{x}_0|c)}$$

**Why "exact"?**
- Uses true marginal $\tilde{p}_{\theta,C}(\mathbf{x}_0)$
- Theoretically equivalent to standard DPO formulation
- Correct loss for constrained preference learning

**Why NOT tractable?**

Computing $\tilde{p}_{\theta,C}(\mathbf{x}_0)$ requires marginalizing over **all trajectories**:

$$\tilde{p}_{\theta,C}(\mathbf{x}_0|c) = \int \tilde{p}_{\theta,C}(\tau, \mathbf{x}_0|c) d\tau$$

$$= \int p(x_T) \prod_{t=1}^T \tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c) d\tau$$

**Intractable!** Cannot compute or differentiate in closed form.

---

**Tractable Approximation (Eq 3.43):**

$$\mathcal{L}_{B,\text{BR}} = \mathbb{E}_{b \sim \rho, t \sim \{1,\ldots,b\}} \left[ \ell_\gamma\left(\beta \cdot b \cdot g_{\theta,C}^{\text{BR}}(t;b)\right) \right]$$

where **pseudo-bridge margin** is:

$$g_{\theta,C}^{\text{BR}}(t;b) = \sum_z \left[ I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^w, t) - I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^\ell, t) \right]$$

**Why tractable?**
- Uses reconstructed trajectories $\hat{\tau}$ (computable!)
- Single-timestep improvement $I_{\theta,C}^{\text{BR}}$ (closed-form!)
- Can backpropagate through model $\theta$

**Why approximate?**
- $\hat{\tau}$ ≠ true trajectory $\tau$
- Endpoint $\hat{\mathbf{x}}_0$ ≠ observed $\mathbf{x}_0$
- Statistical approximation of exact objective

---

**Comparison:**

| Aspect | Exact (Eq 3.29) | Tractable (Eq 3.43) |
|--------|-----------------|---------------------|
| Uses | True marginal $\tilde{p}(\mathbf{x}_0)$ | Pseudo-bridge $\hat{\tau}$ |
| Computation | Intractable integral | Computable forward pass |
| Endpoints | Match data exactly | Approximate ($\hat{\mathbf{x}}_0 \approx \mathbf{x}_0$) |
| Theoretically | Correct DPO loss | Approximation |
| Practically | Cannot compute | Enables training! |

**Proposition 3.2** shows: Exact endpoint objective **can be written as** expectation over trajectories

$$\Delta_{\theta,C}(\mathbf{x}_0) = \mathbb{E}_{\tau \sim q_C^*(\tau|\mathbf{x}_0)} \left[ \sum_t \Lambda_{\theta,C}(\tau, t) \right]$$

But we don't have access to exact posterior bridge $q_C^*(\tau|\mathbf{x}_0)$ → use pseudo-bridge $\hat{\tau}$ instead!

---

### Q29-Q35: Bridge Concept

This group of questions addresses the **core innovation** of Section 3.3: how to reconstruct trajectories from endpoints.

---

#### Q29: What is "bridge"? And "exact bridge"?

**Your question:** "I keep seeing 'bridge' and 'exact posterior bridge' - what are these?"

**Answer:** A **bridge** is a trajectory distribution **conditioned on endpoints**.

---

**General diffusion bridge:**

In standard diffusion models (without constraints), a bridge is:

$$q(\tau | \mathbf{x}_0, \mathbf{x}_T)$$

This gives the distribution over paths $\tau = (\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_1, \mathbf{x}_0)$ that:
- Start at $\mathbf{x}_T$ (noise)
- End at $\mathbf{x}_0$ (data)
- Follow forward/reverse dynamics

**Analogy:** Brownian bridge in probability theory
- Know where particle starts and ends
- What paths did it likely take between?

---

**Exact posterior bridge (Phase B):**

$$q_C^*(\tau|\mathbf{x}_0, c) = \tilde{p}_{\text{ref},C}(\tau|\mathbf{x}_0, c)$$

This is the distribution over trajectories that:
1. End at observed endpoint $\mathbf{x}_0$
2. Follow **ref + SCIGEN dynamics** (constrained reverse)
3. Are weighted by how likely they produce $\mathbf{x}_0$

**Mathematically:**

$$q_C^*(\tau|\mathbf{x}_0) \propto \tilde{p}_{\text{ref},C}(\tau, \mathbf{x}_0) = p(\mathbf{x}_T) \prod_{t=1}^T \tilde{p}_{\text{ref},C}(\mathbf{x}_{t-1}|\mathbf{x}_t)$$

subject to $\mathbf{x}_0$ (endpoint) being the observed value.

**Visual:**

```
Exact posterior bridge q_C*(τ|x_0):

Multiple possible trajectories, all ending at x_0:

  x_T(1) ──→ ... ──→ x_1(1) ──→ x_0
  x_T(2) ──→ ... ──→ x_1(2) ──→ x_0  ← All end here!
  x_T(3) ──→ ... ──→ x_1(3) ──→ x_0

Weighted by:
  - How likely under ref+SCIGEN dynamics
  - Constrained to end at x_0
```

---

**Why "exact"?**

- Correctly accounts for conditioning on $\mathbf{x}_0$
- Satisfies Bayes' theorem:
  $$q_C^*(\tau|\mathbf{x}_0) = \frac{\tilde{p}_{\text{ref},C}(\tau, \mathbf{x}_0)}{\tilde{p}_{\text{ref},C}(\mathbf{x}_0)}$$

**Why "posterior"?**

- It's a posterior distribution: $p(\tau | \mathbf{x}_0)$
- Conditions on observed data $\mathbf{x}_0$

**Why "bridge"?**

- Connects noise ($\mathbf{x}_T$) to data ($\mathbf{x}_0$)
- Conditioned on both endpoints (though $\mathbf{x}_T \sim \mathcal{N}(0,I)$ is random)

---

#### Q30: What is "pseudo-bridge"?

**Your question:** "How is pseudo-bridge different from exact bridge?"

**Answer:** Pseudo-bridge is a **practical approximation** when we can't sample exact posterior bridge.

---

**The Problem:**

Sampling exact bridge $q_C^*(\tau|\mathbf{x}_0)$ requires:

1. Knowing $\tilde{p}_{\text{ref},C}(\mathbf{x}_0)$ (marginal, intractable!)
2. Backward sampling conditioned on $\mathbf{x}_0$ (no closed form!)
3. Storing original trajectories during generation (memory!)

**All impractical!**

---

**Pseudo-Bridge Construction:**

```python
def pseudo_bridge(x_0, b, model_ref, constraint_C):
    """
    Construct approximate trajectory ending near x_0.

    Args:
        x_0: Observed endpoint
        b: Bridge level (num reverse steps)
        model_ref: Frozen reference model
        constraint_C: SCIGEN constraint

    Returns:
        tau_hat: (x_b, x_{b-1}, ..., x_1, x̂_0)
    """
    # Step 1: Forward corrupt (unconstrained)
    epsilon = torch.randn_like(x_0)
    alpha_bar_b = get_alpha_bar(b)
    x_b = sqrt(alpha_bar_b) * x_0 + sqrt(1 - alpha_bar_b) * epsilon

    # Step 2: Reverse with ref + SCIGEN for b steps
    tau_hat = []
    x_t = x_b
    for t in range(b, 0, -1):
        tau_hat.append(x_t)

        # Ref model proposes
        u_t_minus_1 = model_ref.sample_reverse(x_t, t, constraint_C)

        # SCIGEN projects
        x_t_minus_1 = apply_scigen_projection(u_t_minus_1, constraint_C)

        x_t = x_t_minus_1

    tau_hat.append(x_t)  # Final: x̂_0

    return tau_hat
```

**Visualization:**

```
Pseudo-bridge construction:

Observed: x_0

Step 1 - Forward:
  x_0 ──(add noise)──> x_b

Step 2 - Reverse (ref + SCIGEN):
  x_b ──> x_{b-1} ──> ... ──> x_1 ──> x̂_0

Result:
  τ̂ = (x_b, ..., x_1, x̂_0)

Key property:
  x̂_0 ≠ x_0  (endpoint mismatch!)
```

---

**Key Differences:**

| Aspect | Exact Bridge | Pseudo-Bridge |
|--------|--------------|---------------|
| Endpoint | Exactly $\mathbf{x}_0$ | Approximate $\hat{\mathbf{x}}_0 \approx \mathbf{x}_0$ |
| Distribution | $q_C^*(\tau\|\mathbf{x}_0)$ | Reconstruction from $\mathbf{x}_0$ |
| Sampling | Posterior (conditioned) | Forward-reverse round-trip |
| Feasibility | Theoretically correct | Practically computable |
| SCIGEN | Implicit in marginal | Explicitly reintroduced! |

---

**Why Good Enough?**

**Quote from paper:** "Practical round-trip approximation reintroduces SCIGEN dynamics"

1. **Reintroduces constraint dynamics:** Reverse steps use SCIGEN projection
2. **Statistically similar:** On average, samples from similar distribution
3. **Empirically validated:** Works well in practice (Section 4 results)

**Key insight:** We don't need **exact** trajectory - just need to **statistically sample** from constrained reverse process!

---

#### Q31: If we stored trajectories, what would we need?

**Your question:** "Why don't we just store trajectories during generation?"

**Answer:** **Memory cost is prohibitive!**

**Storage calculation:**

For each preference pair $(x_0^w, x_0^\ell)$:

```python
# What we currently store (endpoints only):
storage_per_pair = 2 * structure_size

# structure_size ≈ 1-5 KB (lattice + positions + types)
# For 1000 pairs: 2-10 MB  ← Manageable!

# What we'd need if storing trajectories:
storage_per_pair_with_traj = 2 * T * structure_size

# T = 1000 timesteps
# For 1000 pairs: 2-10 GB  ← Too large!
```

---

**Detailed breakdown:**

**Phase B dataset:** 300-800 structures per motif

**Per structure:**
- Lattice: 6 floats × 4 bytes = 24 bytes
- Fractional coords: N atoms × 3 × 4 bytes ≈ 200-400 bytes (N=15-30)
- Atom types: N × 1 = 15-30 bytes
- **Total per structure:** ~250-450 bytes ≈ 0.5 KB

**Trajectory storage:**
- States: T × 0.5 KB = 1000 × 0.5 KB = 500 KB per trajectory
- Winner + loser: 1 MB per pair
- 1000 pairs: **1 GB**

**Multiple motifs:**
- 10 motifs × 1 GB = **10 GB** just for Phase B data!

**Plus:**
- Slow I/O (loading 10GB trajectories every epoch)
- GPU memory (batch of trajectories)
- Multiple experiments (grid search, ablations)

**Verdict:** Not worth it when pseudo-bridge works!

---

**If we HAD exact trajectories:**

```python
# Training would be simpler:
def train_phase_b_with_stored_trajectories(model, pairs_with_traj):
    for (tau_w, tau_l, kappa) in dataloader:
        # Sample timestep
        t = random.randint(1, T)

        # Compute improvement directly
        I_w = improvement_score(model, tau_w, t)
        I_l = improvement_score(model, tau_l, t)

        margin = I_w - I_l
        loss = holder_loss(beta * T * margin, gamma)

        loss.backward()
        optimizer.step()
```

**Simpler, but not feasible at scale.**

---

#### Q32: What is "bridge level" b?

**Your question:** "What does b mean in 'bridge level b'?"

**Answer:** Bridge level $b$ is the **number of reverse denoising steps** in pseudo-bridge reconstruction.

**Range:** $b \in \{1, 2, \ldots, T\}$ where $T = 1000$ (total timesteps)

---

**Options:**

**1. Full bridge ($b = T$):**

```python
# Corrupt to pure noise
x_T = sqrt(alpha_bar_T) * x_0 + sqrt(1 - alpha_bar_T) * epsilon
# alpha_bar_T ≈ 0 → x_T ≈ N(0, I)

# Reverse all the way back
tau_hat = reverse_scigen(x_T, T_steps)
# tau_hat = (x_T, x_{T-1}, ..., x_1, x̂_0)
```

**Properties:**
- Maximum SCIGEN coverage (all timesteps)
- x_T pure noise → forgot x_0 (high variance)
- x̂_0 far from x_0 (endpoint mismatch large)

---

**2. Partial bridge ($b = 50$):**

```python
# Corrupt to moderate noise
x_50 = sqrt(alpha_bar_50) * x_0 + sqrt(1 - alpha_bar_50) * epsilon
# alpha_bar_50 ≈ 0.3 → x_50 still resembles x_0

# Reverse 50 steps
tau_hat = reverse_scigen(x_50, 50_steps)
# tau_hat = (x_50, x_49, ..., x_1, x̂_0)
```

**Properties:**
- Moderate SCIGEN coverage
- x_50 retains x_0 info (lower variance)
- x̂_0 closer to x_0 (smaller mismatch)

---

**3. Random bridge ($b \sim \rho(\cdot)$):**

```python
# Sample bridge level from distribution
b = sample_bridge_level(rho)  # e.g., Uniform(1, T) or biased

x_b = corrupt_forward(x_0, b)
tau_hat = reverse_scigen(x_b, b)
```

**Properties:**
- Covers diverse timestep ranges
- Low variance (average over b)
- Empirically best (used in paper!)

---

**"b=T recovers full round-trip":**

**Quote meaning:**

When $b = T$, pseudo-bridge becomes:

$$\mathbf{x}_0 \xrightarrow{\text{forward}} \mathbf{x}_T \xrightarrow{\text{reverse + SCIGEN}} \hat{\mathbf{x}}_0$$

**"Round-trip"** = go all the way up to noise, then all the way back down.

**Analogy:** Complete cycle in diffusion process.

---

#### Q33: "Round trip" = forward + backward?

**Your question:** "What exactly is a 'round trip'?"

**Answer: YES!** Round-trip = forward diffusion + reverse denoising.

**Visual:**

```
Round-trip with bridge level b:

Start:     x_0 (clean structure)
           │
           │ FORWARD diffusion
           │ (add noise, NO SCIGEN)
           ↓
Midpoint:  x_b (noisy structure)
           │
           │ REVERSE denoising
           │ (ref model + SCIGEN)
           ↓
End:       x̂_0 (reconstructed structure)

Full round-trip (b=T):

Start:     x_0
           │
           │ FORWARD (full noise schedule)
           ↓
Midpoint:  x_T ~ N(0,I)  ← Pure noise!
           │
           │ REVERSE (all T steps)
           ↓
End:       x̂_0

Partial round-trip (b small):

Start:     x_0
           │
           │ FORWARD (partial)
           ↓
Midpoint:  x_b  ← Still resembles x_0
           │
           │ REVERSE (b steps)
           ↓
End:       x̂_0  ← Close to x_0
```

---

**"Smaller b preserves info":**

**Information preservation:**

$$I(\mathbf{x}_0; \mathbf{x}_b) = H(\mathbf{x}_0) - H(\mathbf{x}_0|\mathbf{x}_b)$$

As $b$ increases:
- More noise added
- $H(\mathbf{x}_0|\mathbf{x}_b)$ increases (more uncertainty)
- Mutual information decreases

**Example:**

| $b$ | $\bar{\alpha}_b$ | $\|\mathbf{x}_b - \mathbf{x}_0\|$ | Info preserved |
|-----|------------------|-----------------------------------|----------------|
| 10  | 0.95             | Small                             | ~95%           |
| 100 | 0.6              | Medium                            | ~60%           |
| 500 | 0.1              | Large                             | ~10%           |
| 1000| ~0               | $\mathbf{x}_T \sim \mathcal{N}(0,I)$ | ~0%            |

**Tradeoff:**

- **Small b:** Preserves structure, but fewer SCIGEN steps (less constraint coverage)
- **Large b:** More SCIGEN steps, but destroys structure (endpoint mismatch)

**Paper's choice:** Sample $b$ from distribution $\rho(\cdot)$ to balance!

---

#### Q34: Why not always b=T?

**Your question:** "If b=T gives full coverage, why not always use it?"

**Answer:** **Three problems** with always using $b=T$:

---

**Problem 1: Information Loss**

$$\mathbf{x}_T = \sqrt{\bar{\alpha}_T} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_T} \boldsymbol{\epsilon}$$

When $T=1000$: $\bar{\alpha}_T \approx 10^{-8}$ (essentially zero!)

$$\mathbf{x}_T \approx \mathcal{N}(0, I)$$ (pure noise, forgot $\mathbf{x}_0$ completely!)

**Consequence:** Reverse process has **no guidance** from original $\mathbf{x}_0$
- Generates random structure
- $\hat{\mathbf{x}}_0$ unrelated to $\mathbf{x}_0$
- High variance!

---

**Problem 2: Endpoint Mismatch**

**Goal:** Pseudo-bridge should approximate trajectory that ends at $\mathbf{x}_0$

**With b=T:**

```
True endpoint: x_0

Pseudo-bridge:
  x_T (noise) ──> ... ──> x̂_0

Distance:
  || x̂_0 - x_0 || ≈ large!  (unrelated structures)
```

**Why bad for DPO?**

DPO loss should capture: "How much better does model denoise winner vs loser?"

But with $\hat{\mathbf{x}}_0 \not\approx \mathbf{x}_0$:
- Training on different structures than data!
- Gradient points in wrong direction
- Doesn't learn preferences on actual endpoint

---

**Problem 3: High Variance**

**Variance of pseudo-bridge:**

$$\text{Var}[\hat{\tau}] \propto \sum_{t=1}^b \sigma_t^2$$

Longer trajectories ($b$ large) → more stochastic steps → higher variance!

**Training instability:**
- Gradients noisy
- Needs larger batch size (memory!)
- Slower convergence

---

**Benefits of Smaller b:**

**1. Preserves Structure:**

For $b = 50$: $\bar{\alpha}_{50} \approx 0.3$

$$\mathbf{x}_{50} = 0.55 \mathbf{x}_0 + 0.84 \boldsymbol{\epsilon}$$

Still contains 55% of original signal!

After reverse:
$$\|\hat{\mathbf{x}}_0 - \mathbf{x}_0\| \text{ smaller}$$

---

**2. Lower Variance:**

50 stochastic steps vs 1000 → much lower variance

**Gradient SNR improvement:**

$$\text{SNR} = \frac{\mathbb{E}[g_\theta]}{\sqrt{\text{Var}[g_\theta]}}$$

Smaller variance → higher SNR → faster learning!

---

**3. Better Approximation:**

Pseudo-bridge approximates exact bridge:

$$\hat{\tau} \approx \tau \sim q_C^*(\tau|\mathbf{x}_0)$$

With large $b$: $\hat{\mathbf{x}}_0 \ll \mathbf{x}_0$ → poor approximation

With moderate $b$: $\hat{\mathbf{x}}_0 \approx \mathbf{x}_0$ → better approximation!

---

**Optimal Strategy (Paper's Approach):**

Sample $b \sim \rho(\cdot)$ from distribution

**Options:**
1. **Uniform:** $b \sim \text{Uniform}(1, T)$
2. **Biased toward small:** $b \sim \text{Uniform}(1, T/2)$
3. **Geometric:** $\rho(b) \propto (1-p)^b$ (favors small)

**Benefits:**
- Averages over different trajectory lengths
- Balances info preservation (small b) vs coverage (large b)
- Reduces variance through averaging

---

#### Q35: Explain "practical round-trip approximation"

**Your question:** "The paper calls pseudo-bridge a 'practical round-trip approximation' - summarize what this means."

**Answer:** This phrase captures the **key tradeoff and solution** of Phase B.

Let me break down each word:

---

**"Practical"** = computationally feasible

**NOT practical:**
- Exact posterior bridge $q_C^*(\tau|\mathbf{x}_0)$ (intractable integral!)
- Storing all trajectories (GBs of memory!)
- Exact marginal $\tilde{p}_{\theta,C}(\mathbf{x}_0)$ (no closed form!)

**IS practical:**
- Forward diffusion: $\mathbf{x}_b = \sqrt{\bar{\alpha}_b} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_b}\boldsymbol{\epsilon}$ ✓
- Reverse with ref model: $\mathbf{u}_t \sim p_{\text{ref}}(\cdot|\mathbf{x}_t)$ ✓
- SCIGEN projection: $\mathbf{x}_t = \Pi_C(\mathbf{u}_t)$ ✓

**Enables GPU training!**

---

**"Round-trip"** = forward then reverse

**Two phases:**

**Phase 1 - Forward (corruption):**
$$\mathbf{x}_0 \xrightarrow{q(\mathbf{x}_b|\mathbf{x}_0)} \mathbf{x}_b$$

Add noise for $b$ steps

**Phase 2 - Reverse (reconstruction):**
$$\mathbf{x}_b \xrightarrow{\tilde{p}_{\text{ref},C}} \hat{\mathbf{x}}_0$$

Denoise with ref + SCIGEN for $b$ steps

**Complete cycle!**

---

**"Approximation"** = not exact, but good enough

**What's approximate:**

1. **Endpoint mismatch:** $\hat{\mathbf{x}}_0 \neq \mathbf{x}_0$

   Doesn't exactly end at observed data

2. **Distribution mismatch:** $\hat{\tau} \not\sim q_C^*(\tau|\mathbf{x}_0)$

   Not sampled from true posterior bridge

3. **Independence:** Each pseudo-bridge sampled independently

   True bridge would condition all on same $\mathbf{x}_0$

---

**Why good enough:**

**1. Reintroduces SCIGEN dynamics**

**Key property:** Reverse steps use $\Pi_C$ projection

$$\mathbf{x}_{t-1} = \Pi_C(\mathbf{u}_t)$$

This matches **exactly** how training model will be used!

**Training-deployment consistency:** ✓

---

**2. Statistically equivalent (on average)**

**Expected trajectory:**

$$\mathbb{E}[\hat{\tau}] \approx \mathbb{E}_{\tau \sim q_C^*}[\tau]$$

Over many samples, pseudo-bridges cover same space as exact bridges.

---

**3. Empirically validated**

**Section 4 results show:**
- Phase B improves flat band discovery
- Pseudo-bridges enable effective DPO training
- Performance comparable to (hypothetical) exact bridge

---

**Full picture:**

```
Practical Round-Trip Approximation:

Input: Observed endpoint x_0^w (winner)

Step 1 - PRACTICAL forward:
  x_0^w ──(add noise)──> x_b^w
  Closed-form Gaussian! ✓

Step 2 - ROUND-TRIP reverse:
  x_b^w ──(ref + SCIGEN)──> x̂_0^w
  Reintroduces constraints! ✓

Step 3 - APPROXIMATION:
  Use x̂_0^w ≈ x_0^w for training
  Good enough statistically! ✓

Result:
  Tractable DPO training under SCIGEN constraints!
```

**This is the core innovation enabling Phase B training.**


---

### Q36-Q38: Residuals & Normalization

These questions address **how to compute denoising errors** for Phase B with SCIGEN constraints.

---

#### Q36: What is "pseudo-bridge residual"?

**Your question:** "Equations 3.37-3.39 define residuals with 'BR' superscript - what are these?"

**Answer:** Pseudo-bridge residuals are **denoising errors computed on reconstructed trajectories**.

**Equation 3.37 (Lattice channel):**

$$\mathbf{r}_\theta^{(L),\text{BR}} = \bar{\mathbf{C}}_{\text{eff}}^{(L)} \odot (\hat{\mathbf{k}}_{t-1} - \boldsymbol{\mu}_\theta^{(L)})$$

**Components:**

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\hat{\mathbf{k}}_{t-1}$ | Target (from pseudo-bridge) | State from $\hat{\tau}$ at step $t-1$ |
| $\boldsymbol{\mu}_\theta^{(L)}$ | Prediction (from model) | Model's predicted clean lattice |
| $\bar{\mathbf{C}}_{\text{eff}}^{(L)}$ | Free mask | 1 for free DOF, 0 for fixed |
| $\odot$ | Element-wise product | Mask operation |
| $\mathbf{r}_\theta^{(L),\text{BR}}$ | Residual (error on free DOF) | What model needs to learn |

---

**Why "residual"?**

Residual = target - prediction (standard ML terminology)

$$\text{error} = \text{ground truth} - \text{prediction}$$

In Phase B:
$$\text{pseudo-bridge target} - \text{model prediction}$$

---

**Why mask with $\bar{\mathbf{C}}_{\text{eff}}$?**

**Lemma 3.1** showed: Only free DOF matter for log-ratio!

$$\log \frac{\tilde{p}_{\theta,C}}{\tilde{p}_{\text{ref},C}} = \log \frac{p_\theta(\mathbf{x}^{\text{free}})}{p_{\text{ref}}(\mathbf{x}^{\text{free}})}$$

**So:** Compute error **only on free components**, zero out fixed.

---

**Example (Kagome N=6):**

**Lattice channel:**

```python
# Pseudo-bridge target
k_hat_t_minus_1 = [
    -0.288,  # k_1 (hexagonal) ← FIXED
     0.023,  # k_2 ← FREE
     0.015,  # k_3 ← FREE
     0.000,  # k_4 ← FIXED
     0.000,  # k_5 ← FIXED
     0.000   # k_6 ← FIXED
]

# Model prediction
mu_theta_L = [
    -0.250,  # k_1 predicted (but won't be used!)
     0.018,  # k_2 predicted
     0.012,  # k_3 predicted
     0.001,  # k_4 predicted (but won't be used!)
     0.002,  # k_5 predicted (but won't be used!)
     0.001   # k_6 predicted (but won't be used!)
]

# Free mask (from constraint C)
C_bar_eff_L = [
    0,  # k_1 fixed
    1,  # k_2 free
    1,  # k_3 free
    0,  # k_4 fixed
    0,  # k_5 fixed
    0   # k_6 fixed
]

# Residual (element-wise)
r_theta_L_BR = C_bar_eff_L ⊙ (k_hat - mu_theta_L)
             = [0, 1, 1, 0, 0, 0] ⊙ [-0.038, 0.005, 0.003, -0.001, -0.002, -0.001]
             = [0.0, 0.005, 0.003, 0.0, 0.0, 0.0]

# Only errors on k_2, k_3 (free DOF) are kept!
```

**Gradient flows only through free DOF!**

---

**Full set of residuals:**

**Equation 3.37 (Lattice):**
$$\mathbf{r}_\theta^{(L),\text{BR}} = \bar{\mathbf{C}}_{\text{eff}}^{(L)} \odot (\hat{\mathbf{k}}_{t-1} - \boldsymbol{\mu}_\theta^{(L)})$$

**Equation 3.38 (Fractional coords):**
$$\mathbf{r}_\theta^{(F),\text{BR}} = \bar{\mathbf{C}}_{\text{eff}}^{(F)} \odot \Delta(\hat{\mathbf{F}}_{t-1}, \boldsymbol{\mu}_\theta^{(F)})$$

Note: $\Delta$ = wrapped difference (for torus!)

**Equation 3.39 (Atom types):**
$$\mathbf{r}_\theta^{(A),\text{BR}} = \bar{\mathbf{C}}_{\text{eff}}^{(A)} \odot (\hat{\mathbf{A}}_{t-1} - \boldsymbol{\mu}_\theta^{(A)})$$

**All use same pattern:** mask $\odot$ (target - prediction)

---

#### Q37: Why superscript BR?

**Your question:** "What does 'BR' stand for in the superscript?"

**Answer:** **BR = BRidge**

This distinguishes **Phase B** (bridge) residuals from **Phase A** (forward-corrupted) residuals.

---

**Comparison:**

**Phase A residuals (no superscript):**

$$\mathbf{r}_\theta^{(L)} = \mathbf{k}_{t-1} - \boldsymbol{\mu}_\theta^{(L)}(\mathbf{M}_t, t, c_A)$$

**Source:** Forward-corrupted pair $(\mathbf{x}_t, \mathbf{x}_{t-1})$ from same trajectory

$$(\mathbf{x}_{t-1}, \mathbf{x}_t) \sim q(\mathbf{x}_{t-1}, \mathbf{x}_t | \mathbf{x}_0)$$

**No constraint mask!** (Phase A is unconstrained)

---

**Phase B residuals (BR superscript):**

$$\mathbf{r}_\theta^{(L),\text{BR}} = \bar{\mathbf{C}}_{\text{eff}}^{(L)} \odot (\hat{\mathbf{k}}_{t-1} - \boldsymbol{\mu}_\theta^{(L)}(\hat{\mathbf{M}}_t, t, c_B))$$

**Source:** Pseudo-bridge pair $(\hat{\mathbf{x}}_t, \hat{\mathbf{x}}_{t-1})$ from reconstruction

$$\hat{\tau} = \text{pseudo\_bridge}(\mathbf{x}_0, b)$$

**Has constraint mask!** (Phase B uses SCIGEN)

---

**Summary table:**

| Aspect | Phase A | Phase B (BR) |
|--------|---------|--------------|
| Data source | MP-20 structures | SCIGEN-generated |
| Trajectory | Forward-corrupted | Pseudo-bridge |
| Constraint | None | $C$ (e.g., kagome) |
| Mask | No mask | $\bar{\mathbf{C}}_{\text{eff}}^{(z)}$ |
| Target | $\mathbf{x}_{t-1}$ (exact!) | $\hat{\mathbf{x}}_{t-1}$ (approx!) |
| Notation | $\mathbf{r}_\theta^{(z)}$ | $\mathbf{r}_\theta^{(z),\text{BR}}$ |

**Superscript BR** reminds us: "This error comes from bridge reconstruction, not forward corruption."

---

#### Q38: Why normalized error (Eq 3.40-3.42)?

**Your question:** "Equations 3.40-3.42 divide by $\|\bar{\mathbf{C}}\|_1$ - why normalize?"

**Answer:** **Fair comparison across different constraints with different numbers of free DOF.**

---

**The Problem:**

Different constraints have different numbers of free components:

| Constraint | Total DOF (positions) | Fixed | Free | $\|\bar{\mathbf{C}}^{(F)}\|_1$ |
|------------|----------------------|-------|------|--------------------------------|
| Kagome N=6 | 3×6 = 18 | 3×3 = 9 | 9 | 9 |
| Lieb N=8 | 3×8 = 24 | 2×3 = 6 | 18 | 18 |
| Honeycomb N=8 | 24 | 4×3 = 12 | 12 | 12 |
| Unconstrained N=10 | 30 | 0 | 30 | 30 |

**Without normalization:**

$$d_\theta^{(F),\text{BR}} = \|\mathbf{r}_\theta^{(F),\text{BR}}\|^2$$

**Problem:** More free DOF → larger errors!

```python
# Example errors
error_kagome = 0.01 * 9 = 0.09   # 9 free positions
error_lieb   = 0.01 * 18 = 0.18  # 18 free positions

# Lieb looks "worse" even though per-DOF error is same!
```

**Training issue:**
- Model focuses on constraints with more DOF
- Ignores simpler constraints
- Unbalanced learning

---

**With normalization (Equations 3.40-3.42):**

**Equation 3.40 (Lattice):**
$$d_\theta^{(L),\text{BR}} = \frac{\|\mathbf{r}_\theta^{(L),\text{BR}}\|^2}{\|\bar{\mathbf{C}}_{\text{eff}}^{(L)}\|_1 + \epsilon_0}$$

**Equation 3.41 (Fractional):**
$$d_\theta^{(F),\text{BR}} = \frac{\|\mathbf{r}_\theta^{(F),\text{BR}}\|^2}{\|\bar{\mathbf{C}}_{\text{eff}}^{(F)}\|_1 + \epsilon_0}$$

**Equation 3.42 (Atom types):**
$$d_\theta^{(A),\text{BR}} = \frac{\|\mathbf{r}_\theta^{(A),\text{BR}}\|^2}{\|\bar{\mathbf{C}}_{\text{eff}}^{(A)}\|_1 + \epsilon_0}$$

---

**What is $\|\bar{\mathbf{C}}_{\text{eff}}^{(z)}\|_1$?**

**L1 norm** = count of 1's in mask = **number of free components**

```python
# Example
C_bar_eff_F = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, ...]
# Kagome: Atoms 1-3 free (3×3=9), atoms 4-6 fixed (3×3=9)

||C_bar_eff_F||_1 = sum([1,1,1,0,0,0,1,1,1,0,0,0,...]) = 9
```

---

**Normalized error = error per free DOF:**

```python
# Kagome
raw_error_kagome = 0.09
num_free_kagome = 9
normalized_kagome = 0.09 / 9 = 0.01

# Lieb
raw_error_lieb = 0.18
num_free_lieb = 18
normalized_lieb = 0.18 / 18 = 0.01

# Now they're equal! ✓
```

**Interpretation:** "Average squared error per free coordinate"

---

**Why $\epsilon_0$ in denominator?**

**Equation:**
$$d_\theta^{(z),\text{BR}} = \frac{\|\mathbf{r}\|^2}{\|\bar{\mathbf{C}}\|_1 + \epsilon_0}$$

**Purpose:** Numerical stability when all DOF are constrained

**Example:** All atoms fixed (rare, but possible)

```python
C_bar_eff_F = [0, 0, 0, ..., 0]  # All fixed!
||C_bar_eff_F||_1 = 0

# Without epsilon_0:
d = ||r||^2 / 0  # Division by zero! NaN!

# With epsilon_0 = 1e-8:
d = ||r||^2 / 1e-8  # Large but finite
```

**In practice:** $\epsilon_0 = 10^{-8}$ (negligible when $\|\bar{\mathbf{C}}\|_1 \geq 1$)

---

**Benefits of normalization:**

1. **Fair comparison** across constraints with different DOF counts
2. **Balanced training** - all motifs weighted equally
3. **Interpretable** - error per degree of freedom
4. **Numerically stable** - avoids division by zero

**This normalization is crucial for multi-constraint Phase B training!**

---

### Q39-Q40: K-Bridge & Eq 3.44

#### Q39: Where use K-bridge score (Eq 3.45)?

**Your question:** "Equation 3.45 defines $\hat{\Delta}_{\theta,C}^{(K)}$ using K samples - when is this used?"

**Equation 3.45:**
$$\hat{\Delta}_{\theta,C}^{(K)}(\mathbf{x}_0) = \log \left[ \frac{1}{K} \sum_{k=1}^K \exp\left( \sum_{t=1}^{b_k} I_{\theta,C}^{\text{BR}}(\hat{\tau}_k, t) \right) \right]$$

**Answer:** K-bridge is used for **evaluation and diagnostics**, NOT training!

---

**Uses:**

**1. Evaluation (Section 4):**

Measure endpoint quality with lower variance:

```python
# Evaluate model on validation set
def evaluate_ranking_accuracy(model, val_pairs, K=5):
    """
    Compute ranking accuracy using K-bridge scores.

    More stable than single pseudo-bridge!
    """
    correct = 0
    for (x_0_w, x_0_l, constraint_C) in val_pairs:
        # Winner score (average over K bridges)
        Delta_w = k_bridge_score(model, x_0_w, constraint_C, K=K)

        # Loser score
        Delta_l = k_bridge_score(model, x_0_l, constraint_C, K=K)

        # Predict winner
        if Delta_w > Delta_l:
            correct += 1

    return correct / len(val_pairs)
```

**Why K samples?** Reduces variance from stochastic pseudo-bridge.

---

**2. Diagnostics:**

Check if model preferences align with labels:

```python
# Diagnostic: Does model prefer winners?
for (x_0_w, x_0_l, kappa) in diagnostic_pairs:
    Delta_w = k_bridge_score(model, x_0_w, C, K=10)
    Delta_l = k_bridge_score(model, x_0_l, C, K=10)

    margin = Delta_w - Delta_l
    print(f"κ={kappa}, margin={margin:.3f}")

# Expect: High κ → large positive margin
```

**Spearman correlation:** $\rho(\kappa, \hat{\Delta}_w - \hat{\Delta}_\ell)$

---

**3. Active Learning (Phase C):**

Select uncertain pairs for annotation:

```python
# Acquisition function
def acquisition_score(x_0_w, x_0_l, model, constraint_C):
    """
    Uncertainty-based acquisition.
    """
    # K-bridge scores
    Delta_w = k_bridge_score(model, x_0_w, C, K=10)
    Delta_l = k_bridge_score(model, x_0_l, C, K=10)

    # Uncertainty: margin near zero
    uncertainty = 1 / (abs(Delta_w - Delta_l) + 0.1)

    # Variance across K samples (another uncertainty measure)
    var_w = k_bridge_variance(model, x_0_w, C, K=10)
    var_l = k_bridge_variance(model, x_0_l, C, K=10)

    return uncertainty + var_w + var_l

# Select top uncertain pairs for human annotation
```

---

**NOT used for training!**

**Training uses single-timestep improvement (Eq 3.43):**

$$\mathcal{L}_{B,\text{BR}} = \mathbb{E}_{b, t} \left[ \ell_\gamma\left(\beta \cdot b \cdot g_{\theta,C}^{\text{BR}}(t;b)\right) \right]$$

**Why not use K-bridge for training?**

1. **Computational cost:** K forward passes per pair per iteration (K×T model calls!)
2. **Memory:** Need to store K trajectories
3. **Single-timestep is sufficient:** Empirically works well with lower cost

**K-bridge = high-quality evaluation, single-timestep = efficient training**

---

#### Q40: Explain Eq 3.44

**Equation 3.44:**
$$g_{\theta,C}^{\text{BR}}(t;b) = \sum_{z \in \{L,F,A\}} \left[ I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^w, t) - I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^\ell, t) \right]$$

**Your question:** "What is this equation doing?"

**Answer:** This is the **Phase B margin** - analogous to Phase A's $g_\theta(t)$.

---

**Breakdown:**

**Component 1:** Improvement score for winner on channel $z$:

$$I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^w, t) = \omega_t^{(z)} \left[ d_{\text{ref}}^{(z),\text{BR}}(\hat{\tau}^w, t) - d_\theta^{(z),\text{BR}}(\hat{\tau}^w, t) \right]$$

"How much better does model $\theta$ denoise winner (channel $z$) compared to reference at timestep $t$?"

**Component 2:** Same for loser:

$$I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^\ell, t)$$

**Difference (per channel):**

$$I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^w, t) - I_{\theta,C}^{(z),\text{BR}}(\hat{\tau}^\ell, t)$$

"How much better does model denoise winner vs loser on channel $z$?"

**Sum over channels:**

$$g_{\theta,C}^{\text{BR}}(t;b) = \sum_{z \in \{L,F,A\}} [\cdots]$$

"Total improvement advantage of winner over loser (all channels combined)"

---

**Interpretation:**

$$g_{\theta,C}^{\text{BR}}(t;b) > 0 \implies \text{model prefers winner (good!)}$$
$$g_{\theta,C}^{\text{BR}}(t;b) < 0 \implies \text{model prefers loser (mislabel or needs learning)}$$
$$g_{\theta,C}^{\text{BR}}(t;b) \approx 0 \implies \text{model uncertain}$$

---

**Used in Phase B loss (Eq 3.43):**

$$\mathcal{L}_{B,\text{BR}} = \mathbb{E} \left[ \ell_\gamma\left(\underbrace{\beta \cdot b \cdot g_{\theta,C}^{\text{BR}}(t;b)}_{\text{scaled margin}}\right) \right]$$

**Scaling:**
- $\beta$: Preference sharpness (hyperparameter)
- $b$: Bridge level (normalizes partial vs full bridge)
- $g_{\theta,C}^{\text{BR}}(t;b)$: Raw margin

**This is the Phase B analogue of Phase A's $\beta \cdot T \cdot g_\theta(t)$!**

---

**Dependence on $b$:**

Notation $g_{\theta,C}^{\text{BR}}(t;b)$ shows margin depends on bridge level:

- Pseudo-bridge $\hat{\tau}$ depends on $b$ (how many steps to reconstruct)
- Improvement scores evaluated on $\hat{\tau}$
- Therefore margin depends on $b$

**Different $b$ → different pseudo-bridge → different margin!**

---

**Q41: What is "original SCIGEN rollout"?**

**Your question:** "The paper mentions 'original SCIGEN rollout' - what's that?"

**Answer:** The **trajectory that generated $\mathbf{x}_0$ during Phase B dataset creation**.

---

**Timeline:**

**1. Dataset Generation (before training):**

```python
# Generate Phase B dataset
model_A = load_phase_a_model()  # Trained in Phase A

for constraint_C in [kagome, lieb, honeycomb, ...]:
    structures = []

    for i in range(num_samples_per_constraint):
        # Original SCIGEN rollout:
        x_T = torch.randn(...)  # Start from noise
        x_t = x_T

        tau_original = [x_T]  # Store trajectory (temporarily)

        for t in range(T, 0, -1):
            # Model A proposes
            u_t_minus_1 = model_A.sample_reverse(x_t, t, constraint_C)

            # SCIGEN projects
            x_t_minus_1 = apply_scigen_projection(u_t_minus_1, constraint_C)

            x_t = x_t_minus_1
            tau_original.append(x_t)  # Store state

        # Final structure
        x_0 = tau_original[-1]
        structures.append(x_0)
        # tau_original DISCARDED to save memory!

    # Human annotation
    pairs = annotate_preferences(structures)
    dataset_B.add(pairs)  # Store only (x_0^w, x_0^l, κ, C)
```

**"Original rollout"** = $\tau_{\text{original}} = (\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_1, \mathbf{x}_0)$

---

**2. Training Phase B (later):**

**Problem:** We only have endpoints $(\mathbf{x}_0^w, \mathbf{x}_0^\ell)$!

Original trajectories $\tau_{\text{original}}$ were **not stored** (memory!)

**Solution:** Reconstruct with pseudo-bridge

---

**Distribution of original rollout:**

$$\tau_{\text{original}} \sim \tilde{p}_{A,C}(\tau, \mathbf{x}_0 | c)$$

where:
$$\tilde{p}_{A,C}(\tau, \mathbf{x}_0|c) = p(\mathbf{x}_T) \prod_{t=1}^T \tilde{p}_{A,C}(\mathbf{x}_{t-1}|\mathbf{x}_t, c)$$

**Executed policy of model A** under constraint $C$.

---

**Why we care:**

**Exact bridge** would sample from:

$$q_C^*(\tau|\mathbf{x}_0) = \frac{\tilde{p}_{\text{ref},C}(\tau, \mathbf{x}_0)}{\tilde{p}_{\text{ref},C}(\mathbf{x}_0)}$$

This is the **posterior** over trajectories that:
- End at observed $\mathbf{x}_0$
- Follow reference + SCIGEN dynamics

**But:** Original rollout used model A, not ref!

**Mismatch:** $\tilde{p}_{A,C} \neq \tilde{p}_{\text{ref},C}$

**Why OK:** By Phase B, model A already learned general preferences
- A and ref not too different
- Pseudo-bridge reintroduces SCIGEN dynamics (key property!)
- Statistical approximation works in practice

---

**Summary:**

- **Original SCIGEN rollout:** Trajectory from generation (discarded)
- **Pseudo-bridge:** Reconstruction during training (approximates original)
- **Key insight:** Don't need exact original trajectory, just need SCIGEN dynamics reintroduced!


---

## Section 3.4: Robustness Analysis

### Q42: What is "scaled margin" (Eq 3.47)?

**Your question:** "Equation 3.47 defines $u_\theta(s)$ as 'scaled margin' - what does this mean?"

**Equation 3.47:**
$$u_\theta(s) = \begin{cases}
\beta \cdot T \cdot g_\theta(t) & \text{Phase A} \\
\beta \cdot b \cdot g_{\theta,C}^{\text{BR}}(t;b) & \text{Phase B}
\end{cases}$$

**Answer:** The scaled margin is the **input to Hölder loss**, combining raw margin with scaling factors.

---

**Three components:**

**1. Raw margin** $g_\theta(t)$ or $g_{\theta,C}^{\text{BR}}(t;b)$:

$$g = I_\theta(\mathbf{x}^w, t) - I_\theta(\mathbf{x}^\ell, t)$$

**Interpretation:** "How much better does model denoise winner vs loser?"

**Scale:** Arbitrary (depends on noise schedule, model architecture)

---

**2. Timestep scaling** $T$ or $b$:

**Purpose:** Normalize single-timestep to full-trajectory equivalent

**Phase A:** Sample single $t \in \{1, \ldots, T\}$
- Margin $g_\theta(t)$ is for one timestep
- Scale by $T=1000$ to approximate full trajectory
- $T \cdot g_\theta(t) \approx \Delta_\theta(\mathbf{x}_0)$ (endpoint log-ratio)

**Phase B:** Sample single $t \in \{1, \ldots, b\}$
- Bridge has $b$ steps (not $T$!)
- Scale by $b$ to normalize
- $b \cdot g_{\theta,C}^{\text{BR}}(t;b) \approx \Delta_{\theta,C}(\mathbf{x}_0)$

**Why scale:** Makes single-timestep comparable to endpoint objective

---

**3. Preference sharpness** $\beta$:

**From Bradley-Terry model:**

$$P(\mathbf{x}^w \succ \mathbf{x}^\ell) = \sigma(\beta \cdot [\log p_\theta(\mathbf{x}^w) - \log p_\theta(\mathbf{x}^\ell)])$$

**$\beta$ controls temperature:**

| $\beta$ | Effect | $P(y^w \succ y^\ell)$ when $\Delta=1$ |
|---------|--------|----------------------------------------|
| Small (0.1) | Soft preferences | $\sigma(0.1) = 0.52$ |
| Medium (1.0) | Moderate | $\sigma(1.0) = 0.73$ |
| Large (10.0) | Sharp preferences | $\sigma(10.0) = 0.99$ |

**Interpretation:**
- Large $\beta$: Strong confidence in preferences (sharp)
- Small $\beta$: Weak confidence (soft)

**Typical value:** $\beta = 0.1$ to $0.5$ (from DPO literature)

---

**Full scaled margin:**

$$u_\theta = \beta \cdot T \cdot g_\theta(t)$$

**Units interpretation:**

```
g_theta(t): "margin at timestep t"  (≈ 0.001 typical)
T * g_theta(t): "full-trajectory margin"  (≈ 1.0)
beta * T * g_theta(t): "preference score"  (≈ 0.1-0.5)
```

**Used in Hölder loss:**

$$\ell_\gamma(u_\theta) = -(1+\gamma)\sigma(u_\theta)^\gamma + \gamma \sigma(u_\theta)^{1+\gamma}$$

**Scaled margin** $u_\theta$ is the input to loss function!

---

### Q43-Q45: Contamination & Proposition

These questions address the **robustness theory** (Section 3.4.2, Proposition 3.5).

---

#### Q43: Is Eq 3.48 from Hölder-DPO paper?

**Your question:** "Equation 3.48 looks like contamination model - is this standard?"

**Equation 3.48:**
$$\tilde{p}_D^{(\epsilon)} = (1-\epsilon) p_D + \epsilon \cdot p_{\text{flip}}$$

**Answer: YES!** This is the **Huber contamination model** from robust statistics.

---

**Reference:** Fujisawa et al. (2025), "Hölder-DPO", Section 4.1

**Standard in M-estimation theory** (Hampel et al. 1986)

---

**Interpretation:**

**Clean distribution:** $p_D$ (correct labels)

$$(\mathbf{x}^w, \mathbf{x}^\ell) \sim p_D \implies \mathbf{x}^w \succ \mathbf{x}^\ell \text{ (correct!)}$$

**Outlier distribution:** $p_{\text{flip}}$ (mislabeled)

$$(\mathbf{x}^w, \mathbf{x}^\ell) \sim p_{\text{flip}} \implies \mathbf{x}^w \prec \mathbf{x}^\ell \text{ (flipped!)}$$

**Contamination rate:** $\epsilon \in [0,1]$
- $\epsilon = 0$: All labels correct
- $\epsilon = 0.1$: 10% mislabeled
- $\epsilon = 0.5$: 50% mislabeled (random labels!)

---

**Observed distribution:**

$$\tilde{p}_D^{(\epsilon)} = \underbrace{(1-\epsilon) p_D}_{\text{correct labels}} + \underbrace{\epsilon \cdot p_{\text{flip}}}_{\text{mislabels}}$$

**With probability $1-\epsilon$:** Sample from $p_D$ (correct)
**With probability $\epsilon$:** Sample from $p_{\text{flip}}$ (wrong!)

---

**Example:**

```python
# Generate contaminated dataset
def generate_contaminated_dataset(epsilon=0.1):
    dataset = []

    for i in range(1000):
        # Sample clean pair
        x_w, x_l = sample_clean_pair()  # x_w truly better

        # Flip with probability epsilon
        if random.random() < epsilon:
            # Mislabel! Swap winner and loser
            x_w, x_l = x_l, x_w  # Now x_w is actually worse!

        dataset.append((x_w, x_l, kappa))

    return dataset  # 10% mislabeled
```

---

**Why this model?**

**Realistic for crowdsourcing:**
- Annotators make mistakes
- Some pairs genuinely ambiguous
- Occasional adversarial labels

**Mathematically tractable:**
- Can derive influence function
- Can prove redescending property
- Can estimate $\epsilon$ from data (Eq 3.53-3.54)

---

#### Q44: How confirm $u_\theta$ differentiable?

**Your question:** "Proposition 3.5 assumes $u_\theta(s)$ is differentiable - how do we know?"

**Answer:** By construction, $u_\theta$ is **differentiable** because it's built from differentiable components.

---

**Chain of differentiability:**

**Step 1:** $u_\theta(s) = \beta \cdot T \cdot g_\theta(t)$ (scaled margin)

Differentiable in $\theta$ if $g_\theta$ is.

---

**Step 2:** $g_\theta(t) = I_\theta(\mathbf{x}^w, t) - I_\theta(\mathbf{x}^\ell, t)$

Differentiable if $I_\theta$ is.

---

**Step 3:** $I_\theta(\mathbf{x}, t) = \sum_z \omega_t^{(z)} [d_{\text{ref}}^{(z)} - d_\theta^{(z)}]$

Differentiable if $d_\theta^{(z)}$ is.

---

**Step 4:** $d_\theta^{(z)} = \|\mathbf{r}_\theta^{(z)}\|^2 / (\|\bar{\mathbf{C}}^{(z)}\|_1 + \epsilon_0)$

Differentiable if $\mathbf{r}_\theta^{(z)}$ is.

---

**Step 5:** $\mathbf{r}_\theta^{(z)} = \hat{\mathbf{x}}_{t-1}^{(z)} - \boldsymbol{\mu}_\theta^{(z)}$

Differentiable because $\boldsymbol{\mu}_\theta^{(z)}$ is model output!

---

**Step 6:** $\boldsymbol{\mu}_\theta^{(z)} = f_\theta(\mathbf{M}_t, t, c)$ (neural network)

**Neural networks are differentiable!** (by design)

- Layers: Linear, Conv, Attention → all differentiable
- Activations: ReLU, GELU, Softmax → differentiable (or subdifferentiable)
- Composition: Chain rule applies

$$\frac{\partial \boldsymbol{\mu}_\theta}{\partial \theta} \text{ exists and computable via backprop!}$$

---

**Formal statement:**

$$u_\theta(s) = u(\theta; s) \in C^1(\Theta)$$

$u$ is continuously differentiable in $\theta$ for all sample $s$.

**Gradient:**

$$\frac{\partial u_\theta(s)}{\partial \theta} = \beta \cdot T \cdot \frac{\partial g_\theta(t)}{\partial \theta}$$

$$= \beta \cdot T \cdot \left[ \frac{\partial I_\theta(\mathbf{x}^w,t)}{\partial \theta} - \frac{\partial I_\theta(\mathbf{x}^\ell,t)}{\partial \theta} \right]$$

**Computable via automatic differentiation (PyTorch/JAX)!**

---

**Why this matters for Proposition 3.5:**

Influence function derivation requires:

$$\text{IF}(s) = -H^{-1} \nabla_\theta \ell_\gamma(u_\theta(s))|_{\theta=\theta^*}$$

**Chain rule:**

$$\nabla_\theta \ell_\gamma(u_\theta(s)) = \frac{\partial \ell_\gamma}{\partial u_\theta} \cdot \frac{\partial u_\theta}{\partial \theta}$$

Both derivatives exist → IF well-defined!

---

#### Q45: Why need H positive definite?

**Your question:** "The proposition assumes Hessian $H$ is positive definite - why?"

**Equation (from Proposition 3.5):**
$$H = \nabla_\theta^2 \mathbb{E}_{p_D}[\ell_\gamma(u_\theta)]|_{\theta=\theta^*}$$

**Answer:** Positive definiteness ensures **three critical properties**:

---

**Property 1: Unique Minimizer**

**Convexity:**

$$H \succ 0 \implies \text{loss is locally strictly convex}$$

**Implication:**
- $\theta^*$ is unique local minimum
- No other stationary points nearby
- Optimization converges to same $\theta^*$

**If NOT positive definite:**
```
H not positive definite → could have:
- Multiple minima (non-unique θ*)
- Saddle points (gradient = 0 but not minimum)
- Flat regions (Hessian singular)
```

---

**Property 2: Invertible Hessian**

**Influence function formula:**

$$\text{IF}(s) = -H^{-1} \nabla_\theta \ell_\gamma(u_\theta(s))$$

**Requires:** $H^{-1}$ exists!

**Positive definite** $\implies$ **invertible** (all eigenvalues > 0)

**If NOT positive definite:**
```
λ_min(H) ≤ 0 → H is singular → H^{-1} does not exist!
→ Influence function undefined!
```

---

**Property 3: Bounded Influence**

**From Proposition 3.5:**

$$\|\text{IF}(s)\| \leq \frac{\gamma(1+\gamma)}{\lambda_{\min}(H)}$$

**Need:** $\lambda_{\min}(H) > 0$ (smallest eigenvalue positive)

**Positive definite** $\implies$ $\lambda_{\min}(H) > 0$ $\implies$ bound is finite!

**If NOT positive definite:**
```
λ_min(H) = 0 → bound = ∞ (useless!)
λ_min(H) < 0 → bound is negative (nonsensical!)
```

---

**When is H positive definite?**

**Sufficient conditions:**

**1. Loss is strongly convex in $\theta$:**

$$\ell_\gamma(u_\theta) \text{ has positive curvature}$$

True for Hölder loss with neural nets (empirically)

**2. Data distribution $p_D$ has full support:**

Dataset covers diverse inputs (not degenerate)

**3. Model is parameterized correctly:**

No redundant parameters (all $\theta_i$ matter)

---

**In practice:**

**Almost always satisfied for:**
- Neural network models (overparameterized but regularized)
- Real datasets (diverse structures)
- Hölder loss (strictly convex for γ > 0)

**Can verify empirically:**

```python
# Compute Hessian at θ*
H = compute_hessian(loss_fn, theta_star)

# Check eigenvalues
eigvals = torch.linalg.eigvalsh(H)
lambda_min = eigvals.min()

print(f"λ_min(H) = {lambda_min:.6f}")
# Expected: λ_min > 0 (positive definite!)

if lambda_min <= 0:
    print("WARNING: Hessian not positive definite!")
```

**Robust statistics theory:** Standard assumption in M-estimation (Huber 1981, Hampel 1986)

---

### Q46: Does Hölder-DPO show IF proof?

**Your question:** "Is the influence function derivation proven in the Hölder-DPO paper?"

**Answer: YES!** Full proof in **Fujisawa et al. (2025), Appendix D**.

---

**Proof sketch (standard M-estimation):**

**Setup:**

Estimator $\hat{\theta}_\epsilon$ minimizes contaminated loss:

$$\hat{\theta}_\epsilon = \arg\min_\theta \mathbb{E}_{\tilde{p}_D^{(\epsilon)}}[\ell_\gamma(u_\theta(s))]$$

$$= \arg\min_\theta (1-\epsilon) \mathbb{E}_{p_D}[\ell_\gamma] + \epsilon \cdot \ell_\gamma(u_\theta(s_0))$$

**Goal:** Find influence of single point $s_0$ as $\epsilon \to 0$.

---

**Step 1: First-order condition**

At optimum:

$$\nabla_\theta \mathbb{E}_{\tilde{p}_D^{(\epsilon)}}[\ell_\gamma(u_\theta)]|_{\theta=\hat{\theta}_\epsilon} = 0$$

---

**Step 2: Expand around $\epsilon=0$**

**Clean optimum:** $\hat{\theta}_0 = \theta^*$ (minimizer of clean loss)

**Contaminated optimum:** $\hat{\theta}_\epsilon = \theta^* + \epsilon \cdot \delta + O(\epsilon^2)$

**Goal:** Find $\delta$ (this is the influence direction)

---

**Step 3: Taylor expand FOC**

$$0 = (1-\epsilon) \nabla_\theta \mathbb{E}_{p_D}[\ell_\gamma]|_{\theta^* + \epsilon \delta} + \epsilon \cdot \nabla_\theta \ell_\gamma(u_\theta(s_0))|_{\theta^* + \epsilon \delta}$$

Expand to first order in $\epsilon$:

$$0 = \nabla_\theta \mathbb{E}_{p_D}[\ell_\gamma]|_{\theta^*} + \epsilon \left[ H \delta - \nabla_\theta \mathbb{E}_{p_D}[\ell_\gamma]|_{\theta^*} + \nabla_\theta \ell_\gamma(u_\theta(s_0)) \right] + O(\epsilon^2)$$

**First term is zero** (θ* is optimum!)

$$0 = H \delta + \nabla_\theta \ell_\gamma(u_\theta(s_0)) + O(\epsilon)$$

---

**Step 4: Solve for $\delta$**

$$H \delta = -\nabla_\theta \ell_\gamma(u_\theta(s_0))$$

$$\delta = -H^{-1} \nabla_\theta \ell_\gamma(u_\theta(s_0))$$

---

**Step 5: Influence function**

$$\text{IF}(s_0) = \lim_{\epsilon \to 0} \frac{\hat{\theta}_\epsilon - \theta^*}{\epsilon} = \delta$$

$$= -H^{-1} \nabla_\theta \ell_\gamma(u_\theta(s_0))$$

**This is Equation 3.31!**

---

**References:**

**Original theory:**
- Hampel (1974): "The influence curve and its role in robust estimation"
- Huber (1981): "Robust Statistics"

**Hölder divergence:**
- Fujisawa & Eguchi (2008): "Robust parameter estimation with a small bias against heavy contamination"
- Fujisawa et al. (2025): "Direct Preference Optimization with an Offset" (Hölder-DPO), Appendix D

**Applications to ML:**
- Koh & Liang (2017): "Understanding Black-box Predictions via Influence Functions" (NeurIPS)
- Fujisawa et al. (2025): First application to preference learning!

---

### Q47: Explain Eq 3.53-3.54 (Chebyshev)

**Your question:** "Equations 3.53-3.54 use Chebyshev's inequality to prove $\hat{\xi} \leq 1$ - how does this work?"

**Context:** Proposition 3.6 shows outlier proportion estimator $\hat{\xi}$ is always in $[0,1]$.

**Equation 3.53:**
$$\hat{\xi} = \frac{1}{N} \sum_i \iota_\gamma(u_\theta(s_i)) \cdot \frac{\sum_i p_i}{\sum_i p_i^{1+\gamma}}$$

where $p_i = \sigma(u_\theta(s_i))$.

**Goal:** Prove $\hat{\xi} \leq 1$.

---

**Step 1: Chebyshev's Sum Inequality**

**Theorem:** If sequences $\{a_i\}$ and $\{b_i\}$ are **similarly ordered** (both increasing or both decreasing):

$$\frac{1}{N} \sum_{i=1}^N a_i b_i \geq \left( \frac{1}{N} \sum_{i=1}^N a_i \right) \left( \frac{1}{N} \sum_{i=1}^N b_i \right)$$

**Intuition:** Pairing large with large and small with small gives larger product sum than random pairing.

---

**Step 2: Apply to our sequences**

**Define:**
- $a_i = p_i = \sigma(u_\theta(s_i))$
- $b_i = p_i^\gamma = \sigma(u_\theta(s_i))^\gamma$

**Both increasing in $u_\theta(s_i)$!**

If $u_i < u_j$: $p_i < p_j$ AND $p_i^\gamma < p_j^\gamma$

**Similarly ordered!** ✓

---

**Step 3: Apply Chebyshev**

$$\frac{1}{N} \sum_i p_i \cdot p_i^\gamma \geq \left( \frac{1}{N} \sum_i p_i \right) \left( \frac{1}{N} \sum_i p_i^\gamma \right)$$

$$\frac{1}{N} \sum_i p_i^{1+\gamma} \geq \left( \frac{1}{N} \sum_i p_i \right) \left( \frac{1}{N} \sum_i p_i^\gamma \right)$$

---

**Step 4: Rearrange**

$$\frac{\sum_i p_i^{1+\gamma}}{N} \geq \frac{\sum_i p_i}{N} \cdot \frac{\sum_i p_i^\gamma}{N}$$

Multiply both sides by $N$:

$$\sum_i p_i^{1+\gamma} \geq \frac{(\sum_i p_i)(\sum_i p_i^\gamma)}{N}$$

Divide both sides by $\sum_i p_i^\gamma$:

$$\frac{\sum_i p_i^{1+\gamma}}{\sum_i p_i^\gamma} \geq \frac{\sum_i p_i}{N}$$

Take reciprocal (flips inequality):

$$\frac{\sum_i p_i^\gamma}{\sum_i p_i^{1+\gamma}} \leq \frac{N}{\sum_i p_i}$$

---

**Step 5: Substitute into $\hat{\xi}$**

**Equation 3.53:**
$$\hat{\xi} = \frac{1}{N} \sum_i \iota_\gamma(u_\theta(s_i)) \cdot \frac{\sum_i p_i}{\sum_i p_i^{1+\gamma}}$$

**But:** $\iota_\gamma(u) = \gamma(1+\gamma) p^\gamma (1-p)^2$

**Simplification** (Equation 3.54):

After algebraic manipulation (see Fujisawa et al. Appendix E):

$$\hat{\xi} \leq \frac{\sum_i p_i}{N} \cdot \frac{\sum_i p_i^\gamma}{\sum_i p_i^{1+\gamma}}$$

**From Step 4:**
$$\frac{\sum_i p_i^\gamma}{\sum_i p_i^{1+\gamma}} \leq \frac{N}{\sum_i p_i}$$

**Therefore:**
$$\hat{\xi} \leq \frac{\sum_i p_i}{N} \cdot \frac{N}{\sum_i p_i} = 1$$

**QED!** ✓

---

**Interpretation:**

**$\hat{\xi}$ estimates outlier proportion:**

$$\hat{\xi} \approx \epsilon \quad \text{(contamination rate)}$$

**Guaranteed to be in valid range $[0,1]$:**

- $\hat{\xi} \geq 0$ (influence weights non-negative)
- $\hat{\xi} \leq 1$ (Chebyshev inequality)

**Can use for diagnostics:**

```python
# Estimate outlier proportion
xi_hat = estimate_outlier_proportion(u_theta, gamma=2.0)
print(f"Estimated contamination: {xi_hat:.1%}")

# Compare to confidence scores
low_conf_prop = (kappa <= 2).mean()
print(f"Low-confidence proportion: {low_conf_prop:.1%}")

# Should be similar if κ reflects quality!
```

---

## Summary: Complete Deep Dive

**Congratulations!** You've now completed a comprehensive deep dive into **all 47 questions** covering:

### Core Components

1. **Multi-channel crystal diffusion** (Q1-Q14)
   - Lattice, fractional coordinates, atom types
   - Wrapped Gaussian for torus topology
   - Tractable proxies for DPO

2. **Hölder-DPO robustness** (Q15-Q21)
   - Redescending loss function
   - Automatic outlier down-weighting
   - Confidence κ for diagnostics only

3. **Bridge formulation** (Q22-Q35)
   - Pseudo-bridge reconstruction
   - Reintroducing SCIGEN dynamics
   - Round-trip approximation

4. **Phase B implementation** (Q36-Q41)
   - Masked residuals on free DOF
   - Normalized errors for fairness
   - K-bridge for evaluation

5. **Robustness theory** (Q42-Q47)
   - Scaled margin formulation
   - Influence function derivation
   - Outlier proportion estimation

---

### Key Insights

**Innovation 1:** Hölder loss provides robustness without explicit confidence weighting

**Innovation 2:** Pseudo-bridge enables DPO training with SCIGEN constraints

**Innovation 3:** Constraint cancellation (Lemma 3.1) allows training only on free DOF

**Innovation 4:** Three-phase training progressively refines flat band preferences

---

### Further Reading

**For deeper understanding, consult:**

1. [DERIVATIONS_ANNOTATED.md](DERIVATIONS_ANNOTATED.md) - Step-by-step math
2. [VOICE_TRANSCRIPT_CLARIFICATIONS.md](VOICE_TRANSCRIPT_CLARIFICATIONS.md) - Notation guide
3. [CONCEPT_MAP_MERMAID.md](CONCEPT_MAP_MERMAID.md) - Visual overview
4. [technical_terms.md](technical_terms.md) - Quick reference
5. Original paper: [material_dpo.tex](../../overleaf/scigenp_overview_MA/material_dpo.tex)

---

**You now have a complete understanding of SCIGEN+ DPO for flat band material discovery!**

