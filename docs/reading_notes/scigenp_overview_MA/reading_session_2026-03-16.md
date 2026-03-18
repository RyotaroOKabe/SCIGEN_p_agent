# Reading Session Q&A - 2026-03-16

> Voice transcript reading session for material_dpo.tex
> 47 questions extracted and answered

---

## Overview Section

### Q1: What does "executed constrained reverse policy" mean?
**Location:** Lines ~60-78

**Answer:**
When SCIGEN is active, generation has 2 steps:
1. Model proposes: u_{t-1} ~ p_θ(·|x_t)
2. SCIGEN overwrites: x_{t-1} = Π_C(u_{t-1})

The **executed policy** = what actually happens after SCIGEN overwrite, not just model proposal.

**Phase A vs Phase B:**
- Phase A: No constraints during training → align p_θ
- Phase B: Constraints during training → align p̃_{θ,C} (executed)

**Explained in:** Section 3.3, Equations 3.19-3.20

---

## Section 2: Per-Channel Diffusion

### Q2: Is λ_t (Eq 2.4) same as DiffCSP?
**Location:** Equation 2.4

**Answer:** YES, exactly the same!

λ_t = E⁻¹[||∇ log N_w(0, σ_t²)||²]

Score-matching normalization weight from original DiffCSP paper.

**Reference:** Jiao et al. (2023) "Crystal Structure Prediction by Joint Equivariant Diffusion", Section 3.2

---

## Section 3.1: DPO Background

### Q3: What reference presents Bradley-Terry model?
**Location:** Section 3.1

**Answer:**
**Original:** Bradley & Terry (1952) "Rank analysis of incomplete block designs"

**For DPO context:** Rafailov et al. (2023) "Direct Preference Optimization" Section 2.1

**The model:** P(A ≻ B) = σ(score_A - score_B)

---

### Q4: How to derive DPO loss form? Which paper shows proof?
**Location:** Section 3.1

**Answer:**
**Derivation in:** Rafailov et al. (2023) DPO paper, Appendix A

**Key steps:**
1. KL-regularized RL: max E[r(y)] - β·KL(π_θ || π_ref)
2. Optimal policy: π*(y|x) ∝ π_ref(y|x) · exp(r(y)/β)
3. Rearrange: r(y) = β log(π*/π_ref) + const
4. Substitute into Bradley-Terry
5. Result: P(y_w ≻ y_l) = σ(β log[π_θ(y_w)/π_ref(y_w)] - β log[π_θ(y_l)/π_ref(y_l)])

**Must read:** DPO Appendix A.1 "Deriving the DPO Objective"

---

## Section 3.2.2: Improvement Scores

### Q5: How to derive improvement score I_θ and d_θ, d_ref?
**Location:** Equations 3.3-3.5

**Answer:**
For DDPM reverse kernel (Gaussian):
```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)
```

Log-probability:
```
log p_θ = -1/(2σ_t²) ||x_{t-1} - μ_θ||² + const
```

Log-ratio:
```
log[p_θ/p_ref] = -1/(2σ_t²)[||x_{t-1}-μ_θ||² - ||x_{t-1}-μ_ref||²]
```

Reparameterize with noise prediction (DDPM):
```
μ_θ = (x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ)/√α_t
```

Simplify → error difference form:
```
log-ratio ∝ ω_t[||ε-ε_ref||² - ||ε-ε_θ||²] = ω_t[d_ref - d_θ]
```

where ω_t = (1-α_t)²/[2σ_t²(1-ᾱ_t)]

**Reference:** Wallace et al. (2024) "Diffusion Model Alignment Using DPO", Appendix B

---

### Q6-Q8: Wrapped Difference

**Q6: What are F and G in Equation 2.15?**

**Answer:**
- F, G: Any fractional coordinate vectors ∈ [0,1)^{N×3}
- Example: F could be F_t, G could be μ_θ^(F)

Function: wrap_±(u) = u - ⌊u + 1/2⌋ ∈ [-1/2, 1/2)

Applied element-wise: Δ(F, G) = wrap_±(F - G)

---

**Q7: What is "wrapped normal proxy reverse kernel" (Eq 2.16)?**

**Answer:**
**"Proxy"** = approximation (not exact DiffCSP sampler)

**"Wrapped Normal"** = Gaussian on torus [0,1)^{N×3}

**Equation 2.16:**
```
p_θ(F_{t-1}|M_t,t,c) ≈ N_w(F_{t-1}; μ_θ^(F), σ̃_t²I)
```

Components:
- N_w = wrapped Gaussian
- μ_θ^(F) = w(F_t + η_t·ŝ_θ^(F)) (predicted mean)
- σ̃_t² = shared variance

**Why "proxy"?**
- True DiffCSP: predictor-corrector (Langevin)
- For DPO: single-step Gaussian (tractable)

---

**Q8: What is wrapped Gaussian? Multiple peaks?**

**Answer:** NO, not multiple peaks!

**Wrapped Gaussian** = standard Gaussian "wrapped" onto torus

**Construction:**
1. Gaussian N(μ, σ²) on ℝ
2. Wrap to [0,1): x_wrapped = x mod 1

**Properties:**
- Single peak (usually)
- Wraps around at boundaries
- If μ=0.99, σ small → peak near 0.99, bleeds to ~0.0

**Visual:**
```
Regular:     ___/\___
            -∞      +∞

Wrapped:    /\___|___/\
           0    0.5    1
           (wraps around)
```

---

### Q9-Q11: Forward Consistent Pair

**Q9: What is "forward consistent pair"?**

**Answer:**
Pair (F_{t-1}, F_t) where F_t from forward diffusion:
```
F_t ~ q(F_t|F_0)
```

**Consistent** = from same trajectory

**Why needed?** To compute d_θ^(F), need F_{t-1} and prediction given F_t

---

**Q10: What is "simple coupling"?**

**Answer:**
**Coupling** = method for correlated random variables

**Simple coupling:**
```
1. Draw ε_F ~ N(0,I) ONCE
2. Set F_s = w(F_0 + σ_s·ε_F) for s ∈ {t-1,t}
```

Result: (F_{t-1}, F_t) share same noise ε_F

---

**Q11: Why need simple coupling?**

**Answer:**
For DPO evaluation of reverse kernel p_θ(F_{t-1}|F_t):
- Need F_t (condition) and F_{t-1} (target)
- Must be from same trajectory
- Simple coupling ensures this efficiently

---

### Q12-Q13: Tractable Proxy

**Q12: What is "tractable proxy"?**

**Answer:**
**Tractable** = computable in closed form (fast gradient)

**Proxy** = approximation

**Context:**
- True DiffCSP: Langevin MCMC (predictor-corrector) - expensive
- Tractable proxy: Single-step wrapped Gaussian - fast for DPO

---

**Q13: What is predictor-corrector? Why important?**

**Answer:**
**Predictor-corrector** sampling algorithm:

**Predictor:**
```
F_{t-1}^pred = w(F_t + η_t·ŝ_θ(F_t,t))
```

**Corrector (Langevin):**
```
For k=1 to K:
  F ← F + ε·∇log p(F) + √(2ε)·z_k
```

**Why important?**
- Score matching on torus needs correction
- DiffCSP uses for high-quality samples

**For DPO:** Too expensive, use proxy

**Reference:** Song et al. (2021) "Score-Based Generative Modeling through SDEs"

---

### Q14: What is "periodic proxy reverse kernel"?

**Answer:**
**Periodic** = respects torus topology

**Proxy** = approximation (Eq 2.16)

**Reverse kernel** = p_θ(F_{t-1}|F_t,c)

Same as "wrapped normal proxy reverse kernel" - emphasizes periodic domain.

**Non-periodic alternative:** Regular Gaussian (wrong for torus!)

---

## Section 3.2.3: Hölder-DPO

### Q15: How to derive DPO margin g_θ(t)?

**Answer:**
1. For structure x: improvement I_θ(x,t)
2. For pair (x^w, x^ℓ): g_θ(t) = I_θ(x^w,t) - I_θ(x^ℓ,t)
3. Interpretation: "How much better does model denoise winner vs loser?"
4. Bradley-Terry: P(x^w ≻ x^ℓ) = σ(β·T·g_θ(t))

Just applying DPO framework to improvement proxy!

---

### Q16: Why is Hölder loss ℓ_γ(x) = -(1+γ)σ(x)^γ + γσ(x)^{1+γ}?

**Answer:**
**Derived from Hölder divergence:**
```
D_H^γ(p||q) = ∫ φ(p/q) q dx
where φ(h) = γ - (1+γ)h^{γ/(1+γ)}
```

Apply to preferences with p(y=1|s) = σ(u_θ(s)) → get ℓ_γ form

**Why this form?**
1. Redescending: ∂ℓ_γ/∂x → 0 as x → -∞
2. Hölder-continuous gradient (robustness)
3. Recovers logistic loss as γ → 0

**Reference:** Fujisawa et al. (2025) Appendix C

---

### Q17: How to optimize/tune γ?

**Answer:**
**Recommended (Section 3.2.4):**

1. **Default:** γ = 2.0 (from Hölder-DPO paper)

2. **Diagnostic tuning:**
   - Split: high-confidence (κ≥4) vs all
   - Try: γ ∈ {0.5, 1.0, 2.0, 3.0, 5.0}
   - Measure: accuracy on κ≥4 subset
   - Check: training stability

3. **Choose γ that:**
   - Maximizes κ≥4 accuracy
   - Maintains stable training
   - Shows outlier separation

4. **Monitor:**
   - Influence distribution ι_γ(x)
   - Outlier enrichment
   - Spearman(κ, |g_θ|)

---

### Q18: Why not use confidence κ for gradient weighting?

**Answer:**
**Three reasons:**

1. **Hölder already down-weights outliers**
   - Influence ∝ σ(x)^γ(1-σ(x))²
   - Auto-detects outliers

2. **Avoid annotator bias**
   - Some over-confident, some under-confident
   - Weighting propagates these biases

3. **Data-driven robustness**
   - Model decides outliers
   - Based on consistency with learned preference

**Use κ for:** Diagnostics only

---

### Q19: What is "gradient weighting in Equation 3.14"?

**Answer:**
**NO explicit gradient weighting in Eq 3.14!**

Confidence NOT used for weighting.

**Implicit weighting** through Hölder loss:
```
∂L/∂θ ∝ -γ(1+γ)σ(u)^γ(1-σ(u))²·∂u/∂θ
         \_________________/
           implicit weight
```

Depends on model belief σ(u), not κ.

---

## Section 3.2.4: Diagnostics

### Q20: What is "weak validation signal"?

**Answer:**
**Weak** = noisy, imperfect, but useful

**Context:**
- κ might be wrong (annotator inconsistent)
- But on average, κ correlates with difficulty
- Can validate Hölder is working

**Validation:**
- If Hölder down-weights low-κ → good!
- If low-influence has lower κ → confirms robustness
- If high-κ has higher |g_θ| → learned well

Not strong enough for training, good for diagnostics.

---

### Q21: Why define ℓ_γ as function of x in Eq 3.29?

**Answer:**
x = β·T·g_θ(t) = **scaled margin**

**Why parameterize by x?**
1. Cleaner notation
2. Analyze gradient: ∂ℓ_γ/∂x is influence shape
3. General form for Phase A or B

Then: ∂L/∂θ = (∂ℓ_γ/∂x)·(∂x/∂θ)

Eq 3.29: ∂ℓ_γ/∂x = -γ(1+γ)p^γ(1-p)² where p=σ(x)

---

## Section 3.3: Bridge Formulation

### Q22: What is "forward corrupted state"?

**Answer:** YES, exactly!

**Forward corrupted** = x_t ~ q(x_t|x_0)

Clean x_0 → add noise → x_t

**Phase A:**
1. Take clean x_0^w, x_0^ℓ
2. Add noise: x_t^w ~ q(·|x_0^w)
3. Compute improvement on x_t^w, x_t^ℓ

**Phase B different:** Uses backward rollout (pseudo-bridge)

---

### Q23: Why does SCIGEN apply constraint "after each reverse step"?

**Answer:**
**SCIGEN generation:**
```
Start: x_T ~ N(0,I)
For t=T down to 1:
  u_{t-1} ~ p_θ(·|x_t,c)      # propose
  x_{t-1} = Π_C(u_{t-1})      # overwrite
Result: x_0
```

**"After each step"** = after every denoising t → t-1

**Why?** Constraints enforced throughout trajectory, not just end

**Consequence for Phase B:**
- Data = endpoints (x_0^w, x_0^ℓ) from SCIGEN
- Trajectories NOT stored!
- Must reconstruct for DPO training

---

### Q24-Q26: Equations 3.25-3.27

**Q24: Why does Eq 3.25 look like this?**

Eq 3.25: Π_C(u_{t-1}; C) = (u_{t-1}^free, x_C^⋆,fix)

**Answer:**
**Definition of SCIGEN projection**

- u_{t-1}^free: Unconstrained DOF (model controls)
- x_C^⋆,fix: Fixed by constraint C (e.g., kagome vertices)

**Example (kagome N=6):**
- 3 positions fixed → x_C^⋆,fix
- 3 positions free → u_{t-1}^free
- Lattice: some params fixed, some free

---

**Q25: Why does Eq 3.26 look like this?**

Eq 3.26: p̃_{θ,C}(x_{t-1}|x_t,c) = 1[x_{t-1}^fix = x_C^⋆,fix]·p_θ(x_{t-1}^free|x_t,c)

**Answer:**
**Executed kernel** after SCIGEN overwrite

**Indicator 1[·]:**
- = 1 if constrained parts match motif
- = 0 otherwise

**Free components:** Follow p_θ

**Intuition:** "Keep only proposals consistent with C"

---

**Q26: What does "free" mean in Eq 3.27?**

**Answer:**
**"free"** = unconstrained degrees of freedom

Eq 3.27 (Constraint Cancellation):
```
log[p̃_{θ,C}/p̃_{ref,C}] = log[p_θ(x^free)/p_ref(x^free)]
```

**Why cancellation?**
- Fixed: x^fix = x_C^⋆,fix (same for θ and ref)
- Indicator cancels in ratio
- Only free components remain

**Key insight of Lemma 3.1!**

---

### Q27-Q28: Feasibility & Exact

**Q27: What is "same feasibility indicator"?**

**Answer:**
**Feasibility indicator** = 1[x_{t-1}^fix = x_C^⋆,fix]

**"Same":**
- Winner and loser share constraint C
- Same motif requirements
- Identical indicator function

**In Lemma 3.1 proof:**
- Numerator: 1[·]·p_θ(x_w^free)
- Denominator: 1[·]·p_ref(x_w^free)
- Cancels!

---

**Q28: What does "exact" mean in Eq 3.29?**

**Answer:**
**Exact endpoint objective** (Eq 3.29):
```
L_B,exact = E[ℓ_γ(β[Δ_{θ,C}(x_0^w) - Δ_{θ,C}(x_0^ℓ)])]
```

**"Exact"** = theoretically correct, uses true Δ_{θ,C}

**Problem:** Δ_{θ,C} = log[p̃_{θ,C}(x_0)/p̃_{ref,C}(x_0)] intractable!

**Opposite:** Pseudo-bridge approximation (Eq 3.43) - tractable but approximate

**Section 3.3 derives the tractable version!**

---

### Q29-Q35: Bridge Concept

**Q29: What is "bridge"? And "exact bridge"?**

**Answer:**
**Bridge** = conditional trajectory distribution

**Exact posterior bridge:**
```
q_C⋆(τ|x_0,c) = p̃_{ref,C}(τ|x_0,c)
```

Distribution over τ=(x_T,...,x_1) that:
1. Start from x_T ~ N(0,I)
2. End at observed x_0
3. Follow ref + SCIGEN

**Analogy:** Brownian bridge
- Know start and end
- What paths between?

**"Bridge"** connects noise to data

---

**Q30: What is "pseudo-bridge"?**

**Answer:**
**Pseudo-bridge** = practical approximation

**Construction:**
```
1. Sample b ~ ρ(·)
2. Forward: x_b ~ q(x_b|x_0)
3. Reverse: Run ref+SCIGEN for b steps
   x_b → ... → x̂_0
4. Result: τ̂ = (x_b,...,x̂_0)
```

**Difference:** x̂_0 ≠ x_0

**Still useful:** Reintroduces SCIGEN dynamics

---

**Q31: If we stored trajectories, what would we need?**

**Answer:**
```
For each (x_0^w, x_0^ℓ):
  Store: τ_w = (x_T^w,...,x_0^w)
         τ_ℓ = (x_T^ℓ,...,x_0^ℓ)
```

**Storage:** T × structure_size × dataset_size
- T=1000, structure~KB, dataset~1000s
- **Total: ~GB!**

**With exact bridge:**
- Evaluate Λ_{θ,C}(τ,t) directly
- No pseudo-bridge needed
- More accurate, expensive

---

**Q32: What is "bridge level" b?**

**Answer:**
**Bridge level** = number of reverse steps

**Options:**
- b=T: Full rollout (noise→data)
- b=50: Partial rollout
- b~Uniform: Random length

**Tradeoff:**
- Large b: More info, expensive
- Small b: Cheaper, less coverage
- Random b: Balanced

**"b=T recovers full round-trip":**
- Forward: x_0 → x_T
- Reverse: x_T → x̂_0
- Complete cycle

---

**Q33: "Round trip" = forward + backward?**

**Answer:** YES!

```
x_0 --[forward]--> x_T --[reverse]--> x̂_0
    add noise         denoise with
                      ref+SCIGEN
```

**When b<T:**
```
x_0 --> x_b --> x̂_0
      (partial)
```

**"Smaller b preserves info":**
- b=T: x_T pure noise (forgot x_0)
- b small: x_b close to x_0
- Tradeoff: coverage vs fidelity

---

**Q34: Why not always b=T?**

**Answer:**
**Problems:**
1. **Info loss:** x_T pure noise
2. **High variance:** Long rollout varies
3. **Endpoint mismatch:** x̂_0 far from x_0

**Benefits of smaller b:**
1. **Preserves structure:** x_b retains x_0 info
2. **Lower variance:** Less noise
3. **Better approx:** x̂_0 closer to x_0

**Optimal:** Uniform or biased toward smaller

---

**Q35: Explain "practical round-trip approximation"**

**Answer:**
Quote after Eq 3.36:

**Not exact:** x̂_0 ≠ x_0

**Practical:** Computationally feasible

**Round-trip:** Forward (x_0→x_b) + reverse (x_b→x̂_0)

**Reintroduces SCIGEN:** Captures constraint dynamics

**Why good enough?**
- Statistically samples ref+SCIGEN distribution
- Covers same state space on average
- Empirically validated

---

### Q36-Q38: Residuals & Normalization

**Q36: What is "pseudo-bridge residual"?**

**Answer:**
**Residual** = target - prediction

Eq 3.37: r_θ^{(L),BR} = C̄_eff^{(L)} ⊙ (x̂_{t-1} - μ_θ^{(L)})

**Components:**
- x̂_{t-1}: Target (pseudo-bridge)
- μ_θ^{(L)}: Prediction (model mean)
- C̄_eff^{(L)}: Free mask
- ⊙: Element-wise product

**Purpose:** Error only on controlled DOF

---

**Q37: Why superscript BR?**

**Answer:** BR = BRidge

**Distinguishes:**
- Phase A: Forward-corrupted (no superscript)
- Phase B: Pseudo-bridge (BR superscript)

**Different sampling:**
- Phase A: x_t ~ q(x_t|x_0)
- Phase B: x̂_t from rollout

---

**Q38: Why normalized error (Eq 3.40-3.42)?**

**Answer:**
```
d_θ^{(L),BR} = ||r||² / (||C̄||_1 + ε_0)
```

**Why:**

1. **Different free DOF counts**
   - Kagome: 3 free, 3 fixed
   - Lieb: 6 free, 2 fixed
   - Without norm: More DOF → larger errors

2. **Fair comparison**
   - Divide by # free components
   - Error per free DOF

3. **Numerical stability**
   - ε_0 prevents /0
   - When all constrained: C̄=0

**||C̄||_1:** Count of free components

---

### Q39-Q40: K-Bridge & Eq 3.44

**Q39: Where use K-bridge score (Eq 3.45)?**

**Answer:**
```
Δ̂_{θ,C}^{(K)}(x_0) = log[(1/K)Σ exp(Σ_t I)]
```

**Uses:**
1. **Evaluation:** Lower-variance endpoint quality
2. **Diagnostics:** Check if prefers winners
3. **Active learning:** Select uncertain pairs

**NOT for training!** Training uses single-timestep (Eq 3.43)

**Why K samples?** Reduce variance

---

**Q40: Explain Eq 3.44**

Eq 3.44: g_{θ,C}^{BR}(t;b) = Σ_z [I_{θ,C}^{(z),BR}(τ̂^w,t) - I_{θ,C}^{(z),BR}(τ̂^ℓ,t)]

**Answer:**
**Phase B margin** (analogous to Phase A)

**Components:**
- τ̂^w, τ̂^ℓ: Pseudo-bridges for winner/loser
- I_{θ,C}^{(z),BR}: Improvement on bridge, channel z
- Σ_z: Sum over L, F, A
- t: Sampled ∈ {1,...,b}
- b: Bridge level

**Interpretation:** "At step t, how much better does model denoise winner vs loser (free DOF only)?"

**Used in Eq 3.43:** L = E[ℓ_γ(β·b·g_{θ,C}^{BR})]

---

**Q41: What is "original SCIGEN rollout"?**

**Answer:**
**Original rollout** = trajectory that generated x_0 during data collection

**Generation (Phase B dataset creation):**
```
1. x_T ~ N(0,I)
2. For t=T to 1:
     u_{t-1} ~ p_θ_A(·|x_t)  [Phase-A model]
     x_{t-1} = Π_C(u_{t-1})  [SCIGEN]
3. Result: x_0, τ_original=(x_T,...,x_0)
4. Annotate: x_0^w vs x_0^ℓ
5. Store: ONLY endpoints ← τ NOT stored!
```

**Distribution:** p̃_{θ_A,C}(τ,x_0|c)

**Problem:** Only have marginal p̃(x_0|c)

**Solution:** Reconstruct with pseudo-bridge

---

## Section 3.4: Robustness

### Q42: What is "scaled margin" (Eq 3.47)?

**Answer:** YES, u_θ is scaled margin

Eq 3.47: u_θ(s) = β·T·g_θ(t) [Phase A] or β·T·g_{θ,C}^{RR}(t) [Phase B]

**"Scaled":**
- β: Preference sharpness
- T: Timesteps (normalizes single→full)
- g_θ(t): Raw margin

**Purpose:** Input to Hölder loss ℓ_γ(u_θ)

---

### Q43-Q45: Contamination & Proposition

**Q43: Is Eq 3.48 from Hölder-DPO paper?**

**Answer:** YES!

Eq 3.48: p̃_D^{(ε)} = (1-ε)p_D + ε·p_flip

**From:** Fujisawa et al. (2025), Section 4.1

**Interpretation:**
- p_D: Clean (correct)
- p_flip: Outliers (incorrect)
- ε: Contamination rate

**Standard in robust statistics**

---

**Q44: How confirm u_θ differentiable?**

**Answer:**
**By construction:**
- u_θ = β·T·g_θ(t)
- g_θ = I_θ(x^w) - I_θ(x^ℓ)
- I_θ depends on ε_θ (neural net)
- Neural nets differentiable!

**Formally:** ∂u_θ/∂θ = β·T·[∂I_θ(x^w)/∂θ - ∂I_θ(x^ℓ)/∂θ]

---

**Q45: Why need H positive definite?**

**Answer:**
H = ∇_θ² E_{p_D}[ℓ_γ(u_θ)]

**Need for:**
1. **Optimization:** Unique minimizer θ*
2. **Influence function:** IF = -H⁻¹·∇ℓ requires invertible
3. **Robustness theory:** Standard M-estimation assumption

**If not positive definite:**
- Non-unique solutions
- IF undefined
- Guarantees fail

**In practice:** Almost always satisfied

---

**Q46: Does Hölder-DPO show IF proof?**

**Answer:** YES!

**Reference:** Fujisawa et al. (2025), Appendix D

**Proof sketch:**
1. Contaminated: θ̂_ε minimizes (1-ε)R(θ;p_D) + ε·R(θ;δ_s)
2. Implicit function theorem: ∂θ̂_ε/∂ε at ε=0
3. Result: IF(s) = -H⁻¹·∇_θ ℓ(θ*;s)

**Standard robust stats** (Hampel et al. 1986)

---

**Q47: Explain Eq 3.53-3.54 (Chebyshev)**

**Answer:**
**Context:** Proving ξ̂ ≤ 1

**Chebyshev's Sum Inequality:**
If {a_i}, {b_i} similarly ordered:
```
(1/N)Σ a_i·b_i ≥ [(1/N)Σ a_i]·[(1/N)Σ b_i]
```

**Application:**
- a_i = p_i
- b_i = p_i^γ
- Both increasing

**Steps:**
```
1. Chebyshev: (1/N)Σp_i^{1+γ} ≥ [(1/N)Σp_i]·[(1/N)Σp_i^γ]
2. Rearrange: [(1/N)Σp_i]·[Σp_i^γ/Σp_i^{1+γ}] ≤ 1
3. Note: p̄_i = p_i/Σp_j
4. Substitute: ξ̂ ≤ 1
```

**Conclusion:** Estimator always in [0,1] ✓

---

## Must-Read References

1. **DPO derivation:** Rafailov et al. (2023) Appendix A
2. **Diffusion-DPO:** Wallace et al. (2024) Appendix B
3. **Hölder robustness:** Fujisawa et al. (2025) Sections 4-5, Appendix D
4. **Bradley-Terry:** Original or DPO Section 2.1
5. **Score-based models:** Song et al. (2021) - predictor-corrector

---

## Next Steps

**Completed:** Sections 1-3 (Overview, Diffusion, DPO, Bridge, Robustness)

**Next:** Section 4 (Three-Phase Training) - more concrete!

**Revisit:** Section 3.3 bridge formulation with fresh perspective

**Look at:** Algorithm 1 pseudocode (clearer than equations)
