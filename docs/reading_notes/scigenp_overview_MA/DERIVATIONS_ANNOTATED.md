# Annotated Derivations with Step-by-Step Comments
## SCIGEN+ DPO Paper - Key Mathematical Derivations

> **Purpose:** Show every algebraic step with explanatory comments
> **Format:** Each line has `# explanation` comment
> **Audience:** For deep understanding of mathematical transformations

---

## 📐 DERIVATION 1: DPO Loss (Q4)

### Goal
Replace reward model $r(y)$ with policy log-ratio $\log[\pi_\theta/\pi_{\text{ref}}]$

---

### Step 1: KL-Regularized RL Objective

**Start with:**
$$\max_\theta \quad \mathbb{E}_{\pi_\theta(y|x)}[r(y|x)] - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

where:
- First term = expected reward (higher is better)
- Second term = KL penalty (stay close to reference)
- $\beta$ = trade-off parameter

---

### Step 2: Find Optimal Policy via Lagrangian

**Lagrangian:**
$$\mathcal{L} = \mathbb{E}_{\pi}[r(y|x)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}}) + \lambda \left( \int \pi(y|x) dy - 1 \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Constraint: probabilities sum to 1}$$

**Expand KL term:**
$$\mathcal{L} = \int \pi(y|x) r(y|x) dy - \beta \int \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} dy + \lambda \left( \int \pi(y|x) dy - 1 \right)$$

$$= \int \pi(y|x) \left[ r(y|x) - \beta \log \pi(y|x) + \beta \log \pi_{\text{ref}}(y|x) \right] dy + \lambda \left( \int \pi(y|x) dy - 1 \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Distribute } \pi \text{ inside integral}$$

---

**Take derivative w.r.t. $\pi(y)$ and set to zero:**
$$\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(y|x) - \beta \log \pi(y|x) - \beta + \beta \log \pi_{\text{ref}}(y|x) + \lambda = 0$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Chain rule: } \frac{\partial}{\partial \pi}[\pi \log \pi] = \log \pi + 1$$

**Solve for $\log \pi(y|x)$:**
$$\beta \log \pi(y|x) = r(y|x) + \beta \log \pi_{\text{ref}}(y|x) - \beta + \lambda$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Collect } \log \pi \text{ terms on left}$$

$$\log \pi(y|x) = \frac{r(y|x)}{\beta} + \log \pi_{\text{ref}}(y|x) + \frac{\lambda - \beta}{\beta}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Divide both sides by } \beta$$

**Exponentiate both sides:**
$$\pi^*(y|x) = \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(y|x)}{\beta}\right) \cdot \exp\left(\frac{\lambda - \beta}{\beta}\right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Apply } \exp \text{ to both sides}$$

$$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(y|x)}{\beta}\right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Constant } \exp(\frac{\lambda-\beta}{\beta}) \text{ absorbed into normalization}$$

---

### Step 3: Rearrange to Express Reward

**From optimal policy:**
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(y|x)}{\beta}\right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# } Z(x) = \text{partition function (normalization)}$$

**Take logarithm:**
$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{r(y|x)}{\beta} - \log Z(x)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Apply } \log \text{ to both sides}$$

**Solve for reward:**
$$r(y|x) = \beta \log \pi^*(y|x) - \beta \log \pi_{\text{ref}}(y|x) + \beta \log Z(x)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Multiply by } \beta \text{ and rearrange}$$

$$r(y|x) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Combine logs: } \log a - \log b = \log(a/b)$$

$$r(y|x) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + C(x)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Define } C(x) := \beta \log Z(x) \text{ (depends only on } x \text{, not } y \text{)}$$

---

### Step 4: Substitute into Bradley-Terry

**Bradley-Terry preference model:**
$$P(y^w \succ y^\ell | x) = \sigma(r(y^w|x) - r(y^\ell|x))$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Sigmoid of reward difference}$$

**Substitute $r$ from Step 3:**
$$P(y^w \succ y^\ell | x) = \sigma\left( \left[\beta \log \frac{\pi^*(y^w|x)}{\pi_{\text{ref}}(y^w|x)} + C(x)\right] - \left[\beta \log \frac{\pi^*(y^\ell|x)}{\pi_{\text{ref}}(y^\ell|x)} + C(x)\right] \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Plug in reward formula}$$

$$= \sigma\left( \beta \log \frac{\pi^*(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \beta \log \frac{\pi^*(y^\ell|x)}{\pi_{\text{ref}}(y^\ell|x)} \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# } C(x) \text{ cancels!}$$

$$= \sigma\left( \beta \left[ \log \frac{\pi^*(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \log \frac{\pi^*(y^\ell|x)}{\pi_{\text{ref}}(y^\ell|x)} \right] \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Factor out } \beta$$

$$= \sigma\left( \beta \log \frac{\pi^*(y^w|x) / \pi_{\text{ref}}(y^w|x)}{\pi^*(y^\ell|x) / \pi_{\text{ref}}(y^\ell|x)} \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Combine logs: } \log a - \log b = \log(a/b)$$

---

### Step 5: Replace $\pi^*$ with Trainable $\pi_\theta$

**We don't have optimal policy $\pi^*$, so approximate with $\pi_\theta$:**
$$P(y^w \succ y^\ell | x) \approx \sigma\left( \beta \log \frac{\pi_\theta(y^w|x) / \pi_{\text{ref}}(y^w|x)}{\pi_\theta(y^\ell|x) / \pi_{\text{ref}}(y^\ell|x)} \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Replace } \pi^* \text{ with learnable } \pi_\theta$$

$$= \sigma\left( \beta \left[ \log \frac{\pi_\theta(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \log \frac{\pi_\theta(y^\ell|x)}{\pi_{\text{ref}}(y^\ell|x)} \right] \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Expand division in log}$$

---

### Step 6: Maximum Likelihood Loss

**Given dataset $\mathcal{D} = \{(x_i, y^w_i, y^\ell_i)\}_{i=1}^N$, maximize log-likelihood:**
$$\max_\theta \quad \sum_{i=1}^N \log P(y^w_i \succ y^\ell_i | x_i)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Log-likelihood of observed preferences}$$

$$= \max_\theta \quad \sum_{i=1}^N \log \sigma\left( \beta \left[ \log \frac{\pi_\theta(y^w_i|x_i)}{\pi_{\text{ref}}(y^w_i|x_i)} - \log \frac{\pi_\theta(y^\ell_i|x_i)}{\pi_{\text{ref}}(y^\ell_i|x_i)} \right] \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Substitute preference probability}$$

**Convert to minimization (negate):**
$$\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y^w,y^\ell) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \beta \log \frac{\pi_\theta(y^\ell|x)}{\pi_{\text{ref}}(y^\ell|x)} \right) \right]}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Expectation form of DPO loss}$$

---

## 📐 DERIVATION 2: Improvement Score for DDPM (Q5)

### Goal
Derive tractable proxy $I_\theta(\mathbf{x},t) \propto \log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) - \log p_{\text{ref}}(\mathbf{x}_{t-1}|\mathbf{x}_t)$

---

### Step 1: DDPM Reverse Kernel

**Gaussian reverse transition:**
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t,t), \sigma_t^2 \mathbf{I})$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# DDPM assumes Gaussian reverse kernel}$$

**Log-probability:**
$$\log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = -\frac{1}{2\sigma_t^2} \|\mathbf{x}_{t-1} - \boldsymbol{\mu}_\theta(\mathbf{x}_t,t)\|^2 - \frac{D}{2}\log(2\pi\sigma_t^2)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Gaussian PDF: } \mathcal{N}(\mathbf{x};\boldsymbol{\mu},\sigma^2\mathbf{I}) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp(-\frac{\|\mathbf{x}-\boldsymbol{\mu}\|^2}{2\sigma^2})$$

---

### Step 2: Log-Ratio of Model vs Reference

**Model log-prob:**
$$\log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = -\frac{1}{2\sigma_t^2} \|\mathbf{x}_{t-1} - \boldsymbol{\mu}_\theta\|^2 + \text{const}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Drop constant terms (same for model and ref)}$$

**Reference log-prob:**
$$\log p_{\text{ref}}(\mathbf{x}_{t-1}|\mathbf{x}_t) = -\frac{1}{2\sigma_t^2} \|\mathbf{x}_{t-1} - \boldsymbol{\mu}_{\text{ref}}\|^2 + \text{const}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Same variance } \sigma_t^2 \text{ for both}$$

**Log-ratio:**
$$\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{p_{\text{ref}}(\mathbf{x}_{t-1}|\mathbf{x}_t)} = -\frac{1}{2\sigma_t^2} \left[ \|\mathbf{x}_{t-1} - \boldsymbol{\mu}_\theta\|^2 - \|\mathbf{x}_{t-1} - \boldsymbol{\mu}_{\text{ref}}\|^2 \right]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Subtract log-probs; constants cancel}$$

---

### Step 3: Reparameterize with Noise Prediction

**DDPM mean formula (Ho et al. 2020):**
$$\boldsymbol{\mu}_\theta(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t) \right)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Reparameterize mean using predicted noise } \boldsymbol{\varepsilon}_\theta$$

where:
- $\alpha_t$ = noise schedule parameter at timestep $t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ = cumulative product
- $\boldsymbol{\varepsilon}_\theta$ = neural network predicting the noise

**Forward diffusion (for context):**
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0,\mathbf{I})$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Noisy state } \mathbf{x}_t \text{ as function of clean } \mathbf{x}_0 \text{ and noise } \boldsymbol{\varepsilon}$$

---

### Step 4: Expand Squared Norms

**Expand $\|\mathbf{x}_{t-1} - \boldsymbol{\mu}_\theta\|^2$:**

Substitute DDPM mean:
$$\|\mathbf{x}_{t-1} - \boldsymbol{\mu}_\theta\|^2 = \left\| \mathbf{x}_{t-1} - \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\varepsilon}_\theta \right) \right\|^2$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Plug in } \boldsymbol{\mu}_\theta \text{ formula}$$

*[After lengthy algebra (see Wallace et al. 2024, Appendix B.3-B.7), this simplifies to:]*

$$\|\mathbf{x}_{t-1} - \boldsymbol{\mu}_\theta\|^2 = \frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)} \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\|^2 + \text{terms independent of } \theta$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Key result: squared norm reduces to noise prediction error}$$

Similarly for reference:
$$\|\mathbf{x}_{t-1} - \boldsymbol{\mu}_{\text{ref}}\|^2 = \frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)} \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\text{ref}}\|^2 + \text{terms independent of } \theta$$

---

### Step 5: Substitute Back into Log-Ratio

$$\log \frac{p_\theta}{p_{\text{ref}}} = -\frac{1}{2\sigma_t^2} \cdot \frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)} \left[ \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\|^2 - \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\text{ref}}\|^2 \right]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Substitute squared norm expressions}$$

**Define weight:**
$$\omega_t := \frac{(1-\alpha_t)^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Collect constants into weight factor}$$

$$\log \frac{p_\theta}{p_{\text{ref}}} = -\omega_t \left[ \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\|^2 - \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\text{ref}}\|^2 \right]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Simplified form}$$

---

### Step 6: Define Denoising Errors and Improvement Score

**Denoising errors:**
$$d_\theta := \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)\|^2$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Model's noise prediction error}$$

$$d_{\text{ref}} := \|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\text{ref}}(\mathbf{x}_t,t)\|^2$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Reference's noise prediction error}$$

**Improvement score:**
$$I_\theta(\mathbf{x},t) := \omega_t [d_{\text{ref}} - d_\theta]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# How much better model denoises than reference}$$

**Proportional to log-ratio:**
$$\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{p_{\text{ref}}(\mathbf{x}_{t-1}|\mathbf{x}_t)} \propto I_\theta(\mathbf{x},t)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Tractable proxy for log-ratio!}$$

---

## 📐 DERIVATION 3: Hölder Loss Gradient (Q16)

### Goal
Show that Hölder loss has redescending gradient: $\frac{\partial \ell_\gamma}{\partial x} \to 0$ as $x \to -\infty$

---

### Step 1: Hölder Loss Function

$$\ell_\gamma(x) = -(1+\gamma)\sigma(x)^\gamma + \gamma \sigma(x)^{1+\gamma}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Hölder loss with parameter } \gamma > 0$$

where $\sigma(x) = \frac{1}{1+e^{-x}}$ is the sigmoid function.

---

### Step 2: Compute Derivative

**Let $p := \sigma(x)$ for convenience. Note:**
$$\frac{dp}{dx} = \frac{d}{dx} \left[ \frac{1}{1+e^{-x}} \right] = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)(1-\sigma(x)) = p(1-p)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Standard sigmoid derivative formula}$$

**Apply chain rule to $\ell_\gamma$:**
$$\frac{\partial \ell_\gamma}{\partial x} = \frac{\partial \ell_\gamma}{\partial p} \cdot \frac{dp}{dx}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Chain rule: } \frac{d}{dx}[f(g(x))] = f'(g) \cdot g'(x)$$

**Compute $\frac{\partial \ell_\gamma}{\partial p}$:**
$$\frac{\partial \ell_\gamma}{\partial p} = \frac{\partial}{\partial p} \left[ -(1+\gamma)p^\gamma + \gamma p^{1+\gamma} \right]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Take derivative w.r.t. } p$$

$$= -(1+\gamma) \cdot \gamma p^{\gamma-1} + \gamma \cdot (1+\gamma) p^{\gamma}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Power rule: } \frac{d}{dp}[p^n] = n p^{n-1}$$

$$= \gamma(1+\gamma) \left[ p^\gamma - p^{\gamma-1} \right]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Factor out } \gamma(1+\gamma)$$

$$= \gamma(1+\gamma) p^{\gamma-1} \left[ p - 1 \right]$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Factor out } p^{\gamma-1}$$

$$= -\gamma(1+\gamma) p^{\gamma-1} (1-p)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Note: } p-1 = -(1-p)$$

**Combine with $\frac{dp}{dx} = p(1-p)$:**
$$\frac{\partial \ell_\gamma}{\partial x} = -\gamma(1+\gamma) p^{\gamma-1} (1-p) \cdot p(1-p)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Multiply } \frac{\partial \ell_\gamma}{\partial p} \text{ and } \frac{dp}{dx}$$

$$\boxed{\frac{\partial \ell_\gamma}{\partial x} = -\gamma(1+\gamma) p^\gamma (1-p)^2}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Simplify: } p^{\gamma-1} \cdot p = p^\gamma$$

---

### Step 3: Analyze Redescending Property

**As $x \to -\infty$:**

$$\sigma(x) = \frac{1}{1+e^{-x}} \to \frac{1}{1+e^{+\infty}} \to 0$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Sigmoid approaches 0 for large negative } x$$

Therefore $p = \sigma(x) \to 0$.

**Analyze gradient terms:**
- $p^\gamma \to 0$ (since $p \to 0$ and $\gamma > 0$)
- $(1-p)^2 \to (1-0)^2 = 1$ (bounded)

**Gradient vanishes:**
$$\left| \frac{\partial \ell_\gamma}{\partial x} \right| = \gamma(1+\gamma) \cdot \underbrace{p^\gamma}_{\to 0} \cdot \underbrace{(1-p)^2}_{\to 1} \to 0$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Product of vanishing term and bounded term → 0}$$

**Conclusion:** $\boxed{\lim_{x \to -\infty} \left| \frac{\partial \ell_\gamma}{\partial x} \right| = 0}$ ✓

**Interpretation:** Outliers (model strongly disagrees: $x \ll 0$) have **vanishing influence** on training!

---

## 📐 DERIVATION 4: Constraint Cancellation (Q26, Lemma 3.1)

### Goal
Show that constrained degrees of freedom cancel in the executed policy log-ratio.

---

### Step 1: Executed Kernel with SCIGEN

**SCIGEN projection (deterministic hard constraint):**
$$\Pi_C(\mathbf{u}_{t-1}) = (\mathbf{u}_{t-1}^{\text{free}}, \mathbf{x}_C^{\star,\text{fix}})$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Keep free DOF from proposal, overwrite fixed DOF with motif}$$

**Executed kernel:**
$$\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c) = \int \delta(\mathbf{x}_{t-1} - \Pi_C(\mathbf{u}_{t-1})) \cdot p_\theta(\mathbf{u}_{t-1}|\mathbf{x}_t,c) d\mathbf{u}_{t-1}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Marginalize over proposals, keep only those that project to } \mathbf{x}_{t-1}$$

For deterministic projection:
$$\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c) = \mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Indicator: fixed parts must match motif; free parts follow model}$$

---

### Step 2: Compute Log-Ratio

**Model's executed kernel:**
$$\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c) = \mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)$$

**Reference's executed kernel:**
$$\tilde{p}_{\text{ref},C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c) = \mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_{\text{ref}}(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Same motif } \mathbf{x}_C^{\star,\text{fix}} \text{ for both!}$$

**Log-ratio:**
$$\log \frac{\tilde{p}_{\theta,C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c)}{\tilde{p}_{\text{ref},C}(\mathbf{x}_{t-1}|\mathbf{x}_t,c)} = \log \frac{\mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)}{\mathbf{1}[\mathbf{x}_{t-1}^{\text{fix}} = \mathbf{x}_C^{\star,\text{fix}}] \cdot p_{\text{ref}}(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Substitute executed kernel formulas}$$

$$= \log \frac{p_\theta(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)}{p_{\text{ref}}(\mathbf{x}_{t-1}^{\text{free}}|\mathbf{x}_t,c)}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Indicator cancels! (same in numerator and denominator)}$$

---

### Step 3: Conclusion

$$\boxed{\log \frac{\tilde{p}_{\theta,C}}{\tilde{p}_{\text{ref},C}} = \log \frac{p_\theta(\mathbf{x}^{\text{free}})}{p_{\text{ref}}(\mathbf{x}^{\text{free}})}}$$
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\text{\# Only free DOF contribute to log-ratio!}$$

**Key insight:** Constrained components $\mathbf{x}^{\text{fix}}$ cancel from the ratio, so **only train on free DOF**!

**However:** The distribution $\tilde{p}_{\theta,C}$ still depends on SCIGEN dynamics (via reverse trajectory), so Phase B needs bridge reconstruction.

---

## 📚 Summary Table

| Derivation | Key Result | Technique |
|------------|------------|-----------|
| **DPO Loss** | $r(y) = \beta \log[\pi^*/\pi_{\text{ref}}]$ | Lagrangian + log-exp manipulations |
| **Improvement Score** | $I_\theta \propto d_{\text{ref}} - d_\theta$ | DDPM reparameterization + algebra |
| **Hölder Gradient** | $\frac{\partial \ell_\gamma}{\partial x} \to 0$ as $x \to -\infty$ | Chain rule + limit analysis |
| **Constraint Cancellation** | Only free DOF in log-ratio | Indicator function cancellation |

Each step includes `# comment` explaining the transformation!
