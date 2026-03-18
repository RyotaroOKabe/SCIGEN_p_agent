# Voice Transcript Clarifications
## Additional Detailed Explanations for Unclear Points

> **Purpose:** Address specific confusion points from voice transcript reading
> **Date:** 2026-03-16

---

## 🔤 NOTATION CLARIFICATIONS

### What does tilde ~ mean?

**Your question:** "What is this tilde means here?"

The tilde has **TWO different meanings** in this paper:

#### Meaning 1: Sampling (standard notation)

$$x \sim p(x)$$

Reads: "$x$ is sampled from distribution $p(x)$"

**Examples:**
- $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)$ = "Sample noisy state from forward diffusion"
- $t \sim \text{Uniform}(1,T)$ = "Sample timestep uniformly"

---

#### Meaning 2: Executed/Modified Distribution (SCIGEN-specific)

$$\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t)$$

Reads: "$p$ tilde" = **executed** constrained distribution

**Why tilde?** Distinguishes:
- $p_\theta$ = model's proposal (before SCIGEN)
- $\tilde{p}_{\theta,C}$ = actual distribution (after SCIGEN overwrite)

**Example:**
```
Step t → t-1:
1. Model proposes: u ~ p_θ(·|x_t)
2. SCIGEN applies:  x = Π_C(u)
3. What you get:    x ~ p̃_{θ,C}(·|x_t)
                         ↑
                      tilde!
```

**Visual:** Think tilde as "wavy modification" of $p$

---

### What is Π (capital Pi)?

**Your question:** "What is this capital Pi symbol, Pi? [...] This is product?"

**Answer:** NO! $\Pi_C$ is **SCIGEN projection operator**, NOT multiplication!

#### Definition

$$\Pi_C : \mathbb{R}^d \to \mathcal{M}_C$$

**Reads:** "Pi sub C" = projection onto constraint set $\mathcal{M}_C$

**For SCIGEN:**
$$\Pi_C(\mathbf{u}) = (\mathbf{u}^{\text{free}}, \mathbf{x}_C^{\star,\text{fix}})$$

**Means:**
- Keep unconstrained components from $\mathbf{u}$
- Overwrite constrained components with motif values

---

#### Concrete Example: Kagome N=6

**Proposal $\mathbf{u}_{t-1}$ from model:**
```
atom 1: (0.12, 0.34, 0.0)  ← will be FORCED
atom 2: (0.67, 0.89, 0.0)  ← will be FORCED
atom 3: (0.23, 0.45, 0.0)  ← will be FORCED
atom 4: (0.71, 0.23, 0.5)  ← keep (FREE)
atom 5: (0.33, 0.66, 0.5)  ← keep (FREE)
atom 6: (0.88, 0.44, 0.5)  ← keep (FREE)
```

**After projection:** $\mathbf{x}_{t-1} = \Pi_{\text{kagome}}(\mathbf{u}_{t-1})$

```
atom 1: (0.0, 0.0, 0.0)    ← FIXED (kagome vertex 1)
atom 2: (0.5, 0.0, 0.0)    ← FIXED (kagome vertex 2)
atom 3: (0.5, 0.5, 0.0)    ← FIXED (kagome vertex 3)
atom 4: (0.71, 0.23, 0.5)  ← kept from u
atom 5: (0.33, 0.66, 0.5)  ← kept from u
atom 6: (0.88, 0.44, 0.5)  ← kept from u
```

**Mathematical notation:**
- $\Pi$ = standard symbol for "projection"
- Subscript $C$ = which constraint
- NOT related to product $\prod$!

---

### What does "kernel" mean?

**Your question:** "What is the reverse kernel or what is the meaning of the kernel in this context?"

**Answer:** In probability/diffusion, **kernel** = **transition probability** (conditional distribution)

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) \quad \leftarrow \text{ This is a "kernel"}$$

#### Why called "kernel"?

Historical term from **Markov process theory**:

A kernel $K(\mathbf{x}'|\mathbf{x})$ tells you how to **transition** from state $\mathbf{x}$ to $\mathbf{x}'$

**In diffusion:**

| Name | Formula | Meaning |
|------|---------|---------|
| **Forward kernel** | $q(\mathbf{x}_t\|\mathbf{x}_{t-1})$ | Add noise |
| **Reverse kernel** | $p_\theta(\mathbf{x}_{t-1}\|\mathbf{x}_t)$ | Denoise |
| **Executed kernel** | $\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}\|\mathbf{x}_t)$ | Denoise + SCIGEN |

**Why Gaussian?**

DDPM assumes reverse kernel is Gaussian:
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \sigma_t^2\mathbf{I})$$

This makes it **tractable** (can compute log-probability, gradients).

---

### What does "proxy" mean?

**Your question:** "Proxy is an approximation, right? But why do you use the proxy word?"

**Answer:** "Proxy" = **stand-in** / **substitute** (legal/business term)

**Origin:** Latin *procurare* = "to act on behalf of"

**In ML:** Something that **acts on behalf of** an intractable quantity

$$\underbrace{\text{True thing}}_{\text{intractable}} \quad \xrightarrow{\text{approximated by}} \quad \underbrace{\text{Proxy}}_{\text{tractable}}$$

#### Example 1: Wrapped Gaussian Proxy

**True DiffCSP:**
- Predictor + Langevin corrector
- Can sample, but **no closed-form** $p_\theta(F_{t-1}|F_t)$
- Can't compute gradients!

**Proxy:**
- Single-step wrapped Gaussian
- **Has closed-form:** $p_\theta(F_{t-1}|F_t) = \mathcal{N}_w(\mu_\theta, \sigmã_t^2\mathbf{I})$
- Can backpropagate!

**"Proxy" because:** Gaussian **stands in for** true Langevin sampler

---

#### Example 2: Improvement Score Proxy

**True DPO objective:**
$$\log p_\theta(\mathbf{x}_0) - \log p_{\text{ref}}(\mathbf{x}_0)$$

**Problem:** Requires marginalizing over $T$ timesteps → **intractable**!

**Proxy:**
$$T \cdot I_\theta(\mathbf{x},t) = T \cdot \sum_z \omega_t^{(z)} [d_{\text{ref}}^{(z)} - d_\theta^{(z)}]$$

**Can compute at single timestep $t$** → **tractable**!

---

### Why σ̃_t² is "shared"?

**Your question:** "Why is that the shared variance? In which point are the sigma T shared?"

**Answer:** "Shared" means **same for all components** (not learned separately)

**Equation 2.16:**
$$p_\theta(F_{t-1}|M_t,t,c) \approx \mathcal{N}_w(F_{t-1}; \boldsymbol{\mu}_\theta^{(F)}, \sigmã_t^2 \mathbf{I})$$

**Key:** $\sigmã_t^2 \mathbf{I}$ is **isotropic** (identity matrix)

**Shared across:**
1. **All atoms:** Same $\sigmã_t^2$ for atom 1, atom 2, ..., atom $N$
2. **All dimensions:** Same $\sigmã_t^2$ for $x$, $y$, $z$ coordinates
3. **Model & reference:** Both use same $\sigmã_t^2$

**Why tilde (~) on σ?**

Distinguishes from forward diffusion:
- $\sigma_t^2$ = forward noise variance (from schedule)
- $\sigmã_t^2$ = reverse proxy variance (might be different!)

**In practice:** Often set $\sigmã_t^2 = \sigma_t^2$, but not required

---

### Why wrapped Gaussian, not uniform?

**Your question:** "Why is that not the uniform distribution, but it's the wrapped normal distribution?"

**Answer:** Fractional coordinates have **local structure** (atoms cluster near stable positions)

**If uniform:**
$$p_{\text{uniform}}(F) = 1 \quad \forall F \in [0,1)^{N \times 3}$$

**Problem:** All positions equally likely → **can't denoise!**

```
Uniform on torus:
  All positions equally probable
  No preference for any location
  → Can't predict where atom should be
```

**With wrapped Gaussian:**
$$p(F|F_t) = \mathcal{N}_w(F; \mu_\theta(F_t,t), \sigmã_t^2\mathbf{I})$$

**Benefits:**
- **Peak at $\mu_\theta$** (predicted clean position)
- Probability **decreases away from peak**
- Can denoise!

```
Wrapped Gaussian on torus:
      /\
     /  \      ← Peak at μ_θ (model's prediction)
  __/    \__
 0        1≡0
```

**Intuition:** Even on torus, we want to predict a **specific location**, not "anywhere is fine"

---

## 🔄 TRAJECTORY & COUPLING

### What is a "trajectory"?

**Your question:** "Must be the same trajectory. What do you mean the same trajectory?"

**Answer:** Complete path through time from noise to data (or vice versa)

$$\tau = (\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_1, \mathbf{x}_0)$$

**Concrete example:**

```
Trajectory 1 (noise ε₁):
  x_0 ──(ε₁)──> x_1 ──(ε₁)──> ... ──(ε₁)──> x_T
        ↑                                    ↑
     t=T-1                                  t=0

Trajectory 2 (different noise ε₂):
  x_0 ──(ε₂)──> x_1' ──(ε₂)──> ... ──(ε₂)──> x_T'
        ↑
   Different path!
```

**Same trajectory** = all states generated from **same noise sequence**

**Why needed for DPO:** To evaluate model prediction at timestep $t$, need:
- $\mathbf{x}_t$ (condition)
- $\mathbf{x}_{t-1}$ (target from same path!)

If from different trajectories → meaningless comparison

---

### Why shared noise in simple coupling?

**Your question:** "Why does that share the noise? I think the noise level becomes smaller as the distance T value gets smaller, right?"

**Excellent observation!** Yes, noise **level** decreases, but we use **same base noise**

**Simple coupling:**
$$\varepsilon_F \sim \mathcal{N}(0,\mathbf{I}) \quad \text{(draw ONCE)}$$
$$F_s = w(F_0 + \sigma_s \cdot \varepsilon_F) \quad \text{for } s \in \{t-1,t\}$$

**Key insight:** Same $\varepsilon_F$ direction, different magnitudes!

| Timestep | Noise scale | Noisy state |
|----------|-------------|-------------|
| $t$ | $\sigma_t = 0.300$ | $F_t = w(F_0 + 0.300 \cdot \varepsilon_F)$ |
| $t-1$ | $\sigma_{t-1} = 0.295$ | $F_{t-1} = w(F_0 + 0.295 \cdot \varepsilon_F)$ |

**Visual:**
```
F_0 = [0.5, 0.3]  (clean)
ε_F = [0.8, -0.4]  (drawn once)

t=500:   F_500 = w([0.5, 0.3] + 0.300·[0.8, -0.4])
               = w([0.74, 0.18])

t=499:   F_499 = w([0.5, 0.3] + 0.295·[0.8, -0.4])
               = w([0.736, 0.182])

Both on SAME path (same ε_F direction)!
```

**Why needed?** For DPO, model predicts $\mu_\theta(F_t,t)$ and compares to $F_{t-1}$ from **same trajectory**

**Without coupling (WRONG):**
```
F_t from ε₁:   [0.74, 0.18]
F_{t-1} from ε₂: [0.23, 0.91]  ← Different path!

Comparison meaningless!
```

---

## 🧮 TRACTABILITY

### What does "tractable" mean?

**Your question:** "Tractable means that computable in closed form..."

**Answer:** **Tractable** = can compute efficiently with closed-form formulas

**Intractable** = requires expensive approximation (MCMC, integration, etc.)

**Examples:**

| Operation | Tractable? | Why |
|-----------|------------|-----|
| Gaussian log-prob | ✅ Yes | $\log p(x) = -\frac{1}{2\sigma^2}(x-\mu)^2 + \text{const}$ |
| Gaussian gradient | ✅ Yes | $\nabla_\theta \log p(x)$ has closed form |
| Langevin sampling | ❌ No | Requires $K$ MCMC iterations |
| Langevin log-prob | ❌ No | Stochastic, no density formula |
| $p_\theta(\mathbf{x}_0)$ | ❌ No | Requires $T$-step marginalization |

**For DPO:** Need to **backpropagate** through loss
→ Requires **tractable** (differentiable) log-probabilities

---

### Relationship: Tractable vs Proxy

**Your confusion:** "Tractable and proxy is a kind of opposite word..."

**Clarification:** They're NOT opposites! They work together:

```
Intractable true thing
        ↓
   (approximate with)
        ↓
  Tractable proxy
```

**Example:**

**Intractable:** True DiffCSP Langevin sampler
- Stochastic random walk
- No closed-form density
- Can't compute $\nabla_\theta \log p_\theta(F_{t-1}|F_t)$

**Tractable proxy:** Wrapped Gaussian
- Deterministic density
- Closed form: $\log p = -\frac{1}{2\sigma^2}\|\Delta(F,\mu)\|^2 + \text{const}$
- Can backpropagate!

**Relationship:**
- **Proxy** = the approximation we use
- **Tractable** = property that makes proxy useful

---

## 🎭 PREDICTOR-CORRECTOR

**Your question:** "I'm not very confident here to understand this predicted corrector very accurately enough."

Let me give a super detailed explanation!

### The Algorithm

**Goal:** Sample $F_{t-1} \sim p_\theta(F_{t-1}|F_t,c)$ on torus $[0,1)^{N \times 3}$

**Two phases:**
1. **Predictor:** Initial jump using score
2. **Corrector:** Refine via Langevin dynamics

---

### Predictor Step

**Formula:**
$$F_{\text{pred}} = w\left(F_t + \eta_t \cdot \hat{s}_\theta(F_t,t,c)\right)$$

where:
- $\hat{s}_\theta$ = predicted score (direction to move)
- $\eta_t$ = step size
- $w(\cdot)$ = wrap to $[0,1)$

**Visual:**
```
Current state F_t:

    F_t
     ↓
   (0.7, 0.3)

Score says: "Move toward (0.9, 0.1)"

Predictor jump:
   F_pred = w((0.7, 0.3) + 0.1·(+0.2, -0.2))
          = w((0.72, 0.28))
          = (0.72, 0.28)
```

**Problem:** Predictor can overshoot or land on wrong side of peak (especially near boundaries)

---

### Corrector Step (Langevin Dynamics)

**Formula:** For $k = 1, \ldots, K$:
$$F^{(k+1)} = F^{(k)} + \varepsilon \cdot \nabla_F \log p(F^{(k)}|F_t,c) + \sqrt{2\varepsilon} \cdot \mathbf{z}_k$$

where:
- $\varepsilon$ = Langevin step size (small, e.g., 0.001)
- $\mathbf{z}_k \sim \mathcal{N}(0,\mathbf{I})$ = noise
- $\nabla_F \log p$ = score (refines position)

**Purpose:** Random walk that **converges to $p(F|F_t,c)$** (target distribution)

**Visual:**
```
Target distribution p(F|F_t):

  Probability
      ↑
      |     /\
      |    /  \      ← Peak (true clean position)
      |   /    \
      |  /      \
      |_/________\___→ F
     0.6  0.8  1.0

Predictor: Lands at F_pred = 0.7 (on shoulder)

Corrector: Random walk
  k=1: 0.7 → 0.75  (drift + noise)
  k=2: 0.75 → 0.78
  k=3: 0.78 → 0.81  ← Converging to peak!
  ...
  k=K: 0.79 ← Near peak!
```

---

### Why Both Steps?

**Predictor alone:** Fast but inaccurate
- Single large jump
- Might overshoot or miss peak

**Corrector alone:** Accurate but slow
- Many small steps from scratch
- Expensive (many model calls)

**Together:** Best of both
- Predictor: Get close quickly
- Corrector: Refine accurately

---

### Why DPO Uses Proxy Instead

**Problem:** Langevin corrector requires:
- $K$ extra model evaluations per timestep (e.g., $K=10$)
- $T$ timesteps per sample (e.g., $T=1000$)
- Total: $10 \times 1000 = 10000$ model calls per sample!

**For DPO training:** Need to evaluate **many pairs** → too expensive!

**Solution:** Use **tractable proxy** (wrapped Gaussian)
- Single-step (no corrector)
- Closed-form density
- Can backpropagate

**Trade-off:** Slight accuracy loss, but **huge speedup**

---

(To be continued with Bridge formulation...)