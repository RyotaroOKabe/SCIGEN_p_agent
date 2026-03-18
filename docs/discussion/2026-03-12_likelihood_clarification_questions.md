# Clarification Questions After Reviewing the Mar 11 Discussion Report

**Date:** 2026-03-12
**Context:** RO read through `2026-03-11_Discussion_Report.md` and identified gaps in understanding. This document lists unresolved questions, provides explanations where possible, and suggests learning resources and next communication steps with MA.

---

## 1. What Is "Likelihood" in This Context?

### 1.1 The Basic Idea

**Likelihood** = the probability that a model assigns to a specific data point.

If you have a model $p_\theta$ and a data point $x$, the likelihood is simply $p_\theta(x)$ — "how probable does the model think this particular sample is?"

**Everyday analogy:** Imagine a language model trained on English text. The sentence "The cat sat on the mat" has high likelihood (common pattern). The sentence "Zxqw plf mat the on" has near-zero likelihood (never seen anything like it). The likelihood tells you how "normal" something looks to the model.

**For crystal structures:** A DiffCSP model trained on Materials Project assigns:
- High likelihood to perovskites (very common in training data)
- Medium likelihood to spinels, pyrites (moderately common)
- Low likelihood to exotic kagome flat-band materials (rare in training data)

### 1.2 Likelihood in Standard ML vs. Diffusion Models

| Setting | How likelihood is computed | Difficulty |
|---------|--------------------------|-----------|
| **Coin flip** | $p(\text{heads}) = 0.5$ | Trivial |
| **Gaussian model** | $p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/2\sigma^2}$ | Easy — closed-form |
| **Autoregressive LLM** | $p(y) = \prod_i p(y_i \mid y_{<i})$ — product of next-token probabilities | Easy — just multiply |
| **Diffusion model** | $p(x_0) = \int p(x_T) \prod_{t} p(x_{t-1} \mid x_t) \, dx_{1:T}$ — integral over all denoising trajectories | **Hard** — intractable integral |

For diffusion models, we can't compute the exact likelihood. Instead we approximate it using the **Evidence Lower Bound (ELBO)**: go forward (add noise), go backward (denoise), and sum up the per-step reconstruction quality. This gives a lower bound on the true log-likelihood.

### 1.3 Per-Step Likelihood in Diffusion

At each denoising step $t$, the model predicts the noise $\hat{\varepsilon}_\theta(x_t, t)$. The closer this prediction is to the true noise $\varepsilon$, the higher the per-step likelihood:

$$\log p_\theta(x_{t-1} | x_t) \propto -\|\varepsilon - \hat{\varepsilon}_\theta(x_t, t)\|^2$$

So: **lower denoising error = higher likelihood**. The total likelihood sums these across all timesteps.

### 1.4 How to Compute Likelihood for a Crystal Structure (Conceptual Steps)

Given a pre-trained DiffCSP model and a crystal structure $x_0 = (L, F, A)$:

1. **Forward pass**: Add noise progressively: $x_0 \to x_1 \to \cdots \to x_T$ (pure noise)
2. **At each step $t$**: Compute the model's noise prediction $\hat{\varepsilon}_\theta(x_t, t)$ and compare it to the actual noise $\varepsilon$ that was added
3. **Sum up**: The total log-likelihood ≈ $\sum_t \omega_t \cdot (-\|\varepsilon - \hat{\varepsilon}_\theta(x_t, t)\|^2)$ where $\omega_t$ are schedule-dependent weights
4. **Per-channel**: Do this separately for $L$ (lattice), $F$ (coordinates), $A$ (atom types), then combine

**What MA will provide**: The specific implementation of this for DiffCSP — which functions to call, how to handle the three channels, and how to aggregate.

---

## 2. Why Is the Fractional Coordinate Channel (F) Problematic?

### 2.1 The Core Issue: Periodic Boundary

Fractional coordinates $F \in [0, 1)^{N \times 3}$ live on a **torus** (periodic: 0.99 and 0.01 are neighbors, not far apart). Standard Gaussian noise doesn't respect this periodicity.

**Example**: If an atom is at position $f = 0.98$ and you add Gaussian noise $\varepsilon = 0.05$, you get $0.98 + 0.05 = 1.03$, which wraps to $0.03$. A standard Gaussian likelihood would say $0.03$ is very far from $0.98$ (distance = 0.95), but the true periodic distance is only $0.05$.

### 2.2 What DiffCSP Does (Score Matching)

DiffCSP handles this by using **Wrapped Normal distribution** and **score matching** instead of DDPM:
- The forward process wraps: $F_t = w(F_0 + \sigma_t \varepsilon)$ where $w(\cdot)$ takes the fractional part
- The training objective matches the **score** $\nabla_F \log q(F_t | F_0)$ — the gradient of the log-probability
- This is a valid training objective that converges to the correct distribution

### 2.3 Why Score Matching ≠ Likelihood

The score $\nabla \log p(x)$ tells you the **direction** of increasing probability, but not the probability itself. Analogy: a weather vane tells you which way the wind blows, but not how strong it is. You can reconstruct the full distribution from the score (in theory, with infinite data), but a finite-sample score estimate is not the same as a likelihood estimate.

**For DPO, we need likelihood ratios** (how much more probable is sample A than sample B?). The score-matching training objective doesn't directly give us this.

### 2.4 MA's Solution: Wrapped-Normal Proxy Reverse Kernel

MA's fix (in the revised `material_dpo.tex`):

1. **Approximate** the reverse step for $F$ as a Wrapped Normal: $p_\theta(F_{t-1} | M_t) \approx \mathcal{N}_w(F_{t-1} | \mu_\theta^{(F)}, \tilde{\sigma}_t^2 I)$
2. The mean $\mu_\theta^{(F)}$ comes from one predictor step of the DiffCSP score network
3. Under this proxy kernel, the per-step log-ratio becomes proportional to the **wrapped squared error**: $\|\Delta(F_{t-1}, \mu_\theta^{(F)})\|^2$ where $\Delta$ is the periodic distance
4. This gives something structurally similar to the DDPM log-ratio (difference of squared errors) but respecting periodicity

**Why "proxy" and not "exact"**: DiffCSP's actual sampler uses a predictor-corrector procedure (multiple Langevin steps), not a single-step Gaussian reverse. The proxy kernel simplifies this to one step for tractability. It's an approximation, but it's the best available one that gives a log-ratio-like quantity.

**Bottom line**: For $L$ and $A$ (DDPM channels), the likelihood computation is standard. For $F$, we use an approximate proxy — this is the "hard part" MA flagged.

---

## 3. Glossary of Key Terms

| Term | Meaning | In our context |
|------|---------|---------------|
| **Likelihood** ($p_\theta(x)$) | Probability the model assigns to sample $x$ | How "normal" a crystal looks to DiffCSP |
| **Log-likelihood** | $\log p_\theta(x)$; easier to work with numerically | What we actually compute and compare |
| **Policy model** ($\pi_\theta$) | The model that generates samples | DiffCSP / SCIGEN (the crystal generator) |
| **Reward model** ($r(x)$) | A model that scores how "good" a sample is | In DPO: not needed separately — likelihood IS the reward |
| **RLHF** | Reinforcement Learning from Human Feedback — train policy + reward jointly | The hard way (unstable, expensive) |
| **DPO** | Direct Preference Optimization — use likelihood as reward, skip the reward model | The easy way (just MLE with preference data) |
| **Bradley-Terry model** | $P(A \succ B) = \sigma(r(A) - r(B))$ — probability A is preferred = sigmoid of reward difference | How pairwise preferences relate to likelihoods |
| **DDPM** | Denoising Diffusion Probabilistic Model — Gaussian forward/reverse | Used for lattice $L$ and atom types $A$ |
| **Score matching** | Train by matching $\nabla \log p$ (gradient of log-density) | Used for fractional coordinates $F$ (Wrapped Normal) |
| **Wrapped Normal** ($\mathcal{N}_w$) | Gaussian distribution wrapped onto $[0,1)$ (periodic) | The noise model for fractional coordinates |
| **ELBO** | Evidence Lower Bound — a tractable lower bound on log-likelihood | How we approximate diffusion model likelihood |
| **Denoising error** | $\|\varepsilon - \hat{\varepsilon}_\theta\|^2$ — how well the model predicts noise | Lower error = higher likelihood at that step |
| **Improvement score** ($I_\theta$) | ref error - model error: how much better $\theta$ denoises than reference | The per-step DPO signal |
| **Hölder-DPO** | Robust variant of DPO using Hölder divergence | Handles noisy/mislabeled preference pairs |
| **Heuristics** (in "diffusion models are mysterious") | Tricks/techniques that work empirically but lack formal justification | E.g., specific noise schedules, classifier-free guidance, EMA |

---

## 4. What Are "Heuristics" in Diffusion Models?

MA said "diffusion models have many heuristics that work in practice but lack theoretical justification." Examples:

- **Noise schedule choice**: Cosine vs. linear vs. sigmoid schedules — people use what works, but there's no theorem saying which is optimal
- **Classifier-free guidance**: Mixing conditional and unconditional generation improves quality dramatically, but the theory for why is incomplete
- **EMA (Exponential Moving Average)**: Using an averaged version of model weights for generation works better, but it's empirical
- **Timestep weighting**: How to weight the loss at different noise levels — many recipes, no consensus on which is best
- **The likelihood-quality gap**: In image generation, maximizing likelihood doesn't produce the best-looking images (the "dimension curse" MA mentioned — the most likely image is the blurry average, not a sharp realistic one)

For our project, the relevant heuristic question is: **does likelihood rank crystal structures correctly?** In images, it doesn't correlate well with perceptual quality. In crystals, we don't know yet — that's exactly what Step 0 (the sanity check) will test.

---

## 5. Toy Examples for Building Intuition

### 5.1 1D Gaussian: Simplest Likelihood

Model: $p_\theta(x) = \mathcal{N}(x; \mu, \sigma^2)$. Likelihood of a point $x_0$:

$$p_\theta(x_0) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x_0 - \mu)^2}{2\sigma^2}\right)$$

Points near the mean $\mu$ have high likelihood; points far away have low likelihood. DPO would adjust $\mu$ and $\sigma$ so that preferred samples have higher $p_\theta$.

### 5.2 1D Diffusion DPO (MA's Toy Experiment)

From the Diffusion-DPO reference paper:
- Two conditions $c \in \{A, B\}$ with target means $\mu_A = -4, \mu_B = 4$
- Data: 2-component Gaussian mixtures
- Preference: sample closer to target mean wins
- After DPO fine-tuning: the diffusion model's distribution shifts toward the preferred mode

This is the simplest possible test case. If DPO works here (it does), it validates the framework before moving to crystals.

### 5.3 Crystal Structure Likelihood (Our Task)

For DiffCSP with a crystal $(L, F, A)$:

$$\log p_\theta(x_0) \approx \sum_{t=1}^{T} \left[\omega_t^{(L)} \cdot (-\|e_L - \hat{e}_L\|^2) + \omega_t^{(F)} \cdot (-\text{score error}_F) + \omega_t^{(A)} \cdot (-\|e_A - \hat{e}_A\|^2)\right]$$

Each term is the per-channel, per-timestep contribution. The sum gives the total log-likelihood.

---

## 6. Learning Resources

### Textbooks / Courses
- **Understanding Deep Learning** (Simon Prince, 2023) — Chapter 18 covers diffusion models with clear math. Free PDF: [udlbook.github.io](https://udlbook.github.io/udlbook/)
- **Lilian Weng's blog**: "What are Diffusion Models?" — excellent visual walkthrough. [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- **DPO original paper** (Rafailov et al., 2023) — Sections 1-3 are very readable, even without RL background

### For the Likelihood Specifically
- **Yang Song's blog**: "Generative Modeling by Estimating Gradients of the Data Distribution" — explains score matching vs. likelihood clearly
- **Kingma et al., "Variational Diffusion Models" (2021)** — derives the ELBO for diffusion models step by step

### For Hands-On Intuition
- The **1D toy experiment** in the Diffusion-DPO paper (Section 2) — reproduce this to see DPO shift a distribution
- **Denoising Score Matching tutorial notebooks** — many exist on GitHub; search "score matching tutorial pytorch"

---

## 7. CRITICAL UPDATE: MA's Correction on Likelihood (Mar 12, 2026)

> **MA's message (Mar 12):**
> "I just realized the likelihood estimation for diffusion is super complicated than I thought. I was wrong in explanation. Current Diffusion-DPO considers only forward process; i.e., crystal to noise $x \to \varepsilon$ and we compute likelihood proxy in noise space PDF. Thus, there is no difference between SCIGEN and its base model (DiffCSP) in Diffusion-DPO formulation. So, I will consider more on how to incorporate reverse process, i.e., noise to crystal $\varepsilon \to x$."

### 7.1 What This Changes

This is a significant correction to the meeting discussion. Key implications:

**1. Diffusion-DPO uses the forward process only.**
The DPO improvement score (Eq. 5 in Wallace et al.) works by:
- Taking a clean sample $x_0$ (winner or loser)
- Adding noise to get $x_t$ (forward process: $x_0 \to x_t$)
- Having both the model $\theta$ and reference $\theta_{\text{ref}}$ predict the noise from $x_t$
- Comparing their denoising errors

This is entirely in the **forward/noising direction**. The model never actually generates (reverse-denoises) during DPO training — it only evaluates how well it can denoise noised versions of existing samples.

**2. SCIGEN constraints don't affect the DPO loss.**
SCIGEN operates during the **reverse process** (generation): at each denoising step, it overwrites constrained components with the motif template. But since Diffusion-DPO never runs the reverse process, SCIGEN masking has no effect on the loss computation.

This means:
- The "SCIGEN-masked variant" in `material_dpo.tex` Section 3.5 may be conceptually wrong — or at least not applicable to the standard Diffusion-DPO formulation
- There is **no difference** between computing the DPO loss for a DiffCSP model vs. a SCIGEN model — they share the same forward process and the same denoising network
- The Phase A vs. Phase B distinction (unconstrained vs. constrained DPO) needs rethinking

**3. The "likelihood" discussed in the meeting was actually a "likelihood proxy in noise space."**
During the meeting, MA described computing likelihood by going forward (add noise) then backward (denoise). But Diffusion-DPO actually stays in the forward direction — it computes a proxy for the per-step log-ratio using denoising errors on the noised samples, without ever running the full reverse chain.

### 7.2 What This Does NOT Change

- The per-channel decomposition ($L$, $F$, $A$) is still correct — each channel has its own denoising error
- The Hölder-DPO robustness layer is still applicable
- The preference data collection strategy (Phase A/B/C) is still valid
- The F-channel complication (Wrapped Normal ≠ Gaussian) still applies to how the denoising error is computed

### 7.3 Open Questions After This Correction

1. **How to incorporate SCIGEN into DPO?** If the standard Diffusion-DPO formulation ignores the reverse process, how can we make the DPO loss aware of structural constraints? MA is working on this.

2. **Does the forward-only likelihood proxy still rank materials correctly?** The sanity check (Step 0) is still important, but now we know it measures "how well does the model denoise this crystal?" rather than "how likely is this crystal under the full generative process?"

3. **Is the forward-only proxy sufficient?** Perhaps for Phase A (no constraints), the forward-only DPO is fine. The constraint-aware formulation (Phase B) is the part that needs a new approach.

4. **Could we incorporate SCIGEN by modifying what we noise?** One idea: instead of noising the raw crystal $x_0$, noise the SCIGEN-mixed state (apply constraints to $x_0$ first, then noise). This would at least make the denoising evaluation happen on constraint-consistent states.

### 7.4 Revised Communication Plan with MA

**Immediate (wait for MA's follow-up):**
- MA is reconsidering how to incorporate the reverse process
- Do NOT proceed with implementing the SCIGEN-masked loss from `material_dpo.tex` until MA clarifies

**Questions to ask MA when he follows up:**

1. "For the forward-only DPO (no SCIGEN), can we still run the sanity check? I.e., compute the denoising improvement score for existing materials and check if the ranking makes sense?"

2. "When you say 'likelihood proxy in noise space PDF' — is this the denoising MSE difference $d_{\text{ref}}^{(z)} - d_\theta^{(z)}$ that we already have in `material_dpo.tex`? Or something different?"

3. "For incorporating SCIGEN into the reverse process: are you thinking of (a) running actual reverse-chain generation during DPO training, or (b) a different mathematical trick that avoids full generation?"

4. "Should I still prepare the sanity-check materials (perovskites, kagome candidates) while you work on the formulation? Or should I wait?"

**What RO can do in the meantime:**
- [ ] Prepare the sanity-check dataset (perovskites, kagome, preference pairs) — this is useful regardless
- [ ] Ensure the pre-trained DiffCSP checkpoint loads and can compute denoising predictions
- [ ] Read the Diffusion-DPO paper (Wallace et al.) Section 3 carefully — understand the forward-only formulation
- [ ] Review `material_dpo.tex` Eqs. 6-15 with this new understanding: which parts are forward-only (still valid) vs. which assume reverse-process access (may need revision)?
