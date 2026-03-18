# Discussion Report: DPO Likelihood, SCIGEN Constraints, and Next Steps (Mar 11, 2026)

**Participants:** Ryotaro Okabe (RO), Masaki Adachi (MA), Shoki (SH)
**Location:** Lattice Lab Meeting (Zoom)
**Date:** March 11, 2026
**Recording:** [260311-Lattice-lab.m4a](https://www.dropbox.com/scl/fi/9megpfdt25u173xtdt0tu/260311-Lattice-lab.m4a?rlkey=tcu8i9l9th8t7xtl9sbjyat6w&st=exv03af2&dl=0)
**Zoom Transcript/Summary:** [Zoom AI Companion Notes](https://docs.zoom.us/doc/B6g8-D7YSbyowjg2Vxn_Jw)
**Key Document:** `material_dpo.tex` in [scigenp_overview_MA](file:///pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview_MA/)

---

## 1. Executive Summary

This meeting focused on building foundational understanding of how Diffusion-DPO works for crystal structure generation, and defining the immediate next steps before any implementation. MA walked through the conceptual framework of DPO — from reinforcement learning to preference learning to likelihood-based evaluation — and identified the **critical first experiment**: computing the likelihood of existing crystal structures under the pre-trained DiffCSP/SCIGEN model to see whether the model's internal ranking aligns with human intuition.

The core insight from the meeting: **DPO's entire power rests on whether the generative model's likelihood correctly ranks materials.** Before any fine-tuning, we must verify this assumption by computing likelihoods for known materials (perovskites, kagome candidates, SCIGEN-generated structures) and checking whether the ranking makes physical sense.

The meeting also covered: why the fractional coordinate channel is theoretically tricky (Wrapped Normal ≠ Gaussian, so likelihood is only approximate); whether band structure data needs to be in the training loop (answer: no, but it would improve data efficiency); how SCIGEN constraints change the likelihood distribution; and a concrete action plan (MA provides likelihood computation method → RO computes and visualizes → sanity check → then fine-tune).

---

## 2. DPO Theory for Crystal Structures: Key Concepts Explained

### 2.1 Why DPO Works — The Likelihood-as-Reward Insight

MA gave an extended tutorial on DPO's conceptual foundations, building up from reinforcement learning:

1. **Standard RL**: You have a **policy model** (generator, e.g., DiffCSP) and a **reward model** (evaluator). Training the policy to maximize the reward is reinforcement learning. For preference learning (RLHF), the reward is unknown and must be learned from pairwise comparisons — making it doubly hard.

2. **DPO's breakthrough** (Rafailov et al., NeurIPS 2023 Best Paper): The policy model and the reward model are actually the same. A generative model implicitly encodes a ranking through its **likelihood** — samples the model considers more probable are "better" according to its internal distribution. Therefore:
   - No explicit reward model is needed
   - The likelihood difference between two samples serves as the reward difference
   - Fine-tuning reduces to maximum likelihood estimation with preference-weighted data

3. **Why this worked for LLMs**: In language, helpful/polite responses are much more common in web training data than harmful ones. So the pre-trained LLM's likelihood already roughly correlates with "helpfulness." DPO just sharpens this existing signal.

4. **The open question for crystals**: Does this hold for crystal structures? In Materials Project, common structures (perovskites, etc.) have high likelihood. Flat band materials are rare — their likelihood is probably low. DPO fine-tuning would need to **reshape** the distribution to elevate these rare but desirable structures.

**MA's key quote**: "Likelihood (yuudo/尤度) is everything. If the likelihood correctly ranks materials, then DPO can learn from it. If it doesn't, nothing works."

### 2.2 The Likelihood Computation Challenge

For diffusion models, likelihood is not directly computable (unlike autoregressive LLMs where $p(y|c) = \prod_i p(y_i|y_{<i}, c)$). The process is:

1. Start with a crystal structure $x_0$
2. Add noise forward: $x_0 \to x_1 \to \cdots \to x_T$ (noise)
3. Denoise backward: $x_T \to x_{T-1} \to \cdots \to x_0$ (reconstruction)
4. The reconstruction quality at each step gives a per-step likelihood contribution

For **DDPM channels** (lattice $L$, atom types $A$): Gaussian forward/reverse → per-step likelihood is tractable (difference of denoising MSEs).

For **fractional coordinates** $F$: Uses Wrapped Normal / score matching (not DDPM). The score-matching objective is a valid training objective that converges to the true distribution — but it is **not a likelihood**. MA emphasized: "This is a beautiful mathematical approximation, and it's correct for training, but it's not the likelihood. We can't directly use it for DPO."

This is why MA rewrote the F-channel section in `material_dpo.tex` — replacing the score-MSE surrogate with a wrapped-normal proxy reverse kernel that gives something closer to a per-step log-ratio (see [2026-03-11_masaki_feedback_review.md](../progress/2026-03-11_masaki_feedback_review.md), Point 1).

### 2.3 Hölder-DPO: Robustness to Label Noise

MA explained the motivation for Hölder-DPO:
- DPO assumes labels are always correct (100% accurate human annotations)
- This is unrealistic — annotators make mistakes, click wrong buttons, or disagree
- Standard DPO is sensitive to "label flips" (marking the wrong sample as winner)
- Hölder-DPO uses **robust divergence** (not KL divergence) that automatically down-weights outlier pairs
- Key property: tail noise (rare, extreme mislabeled pairs) is suppressed without needing to identify which pairs are wrong
- **Secondary use**: The same mechanism that identifies outlier labels can identify **anomalous materials** — structures that the majority rejected but that might actually be interesting (Level-3 discovery from the Feb 25 meeting)

MA also clarified terminology: use "Hölder-divergence" or "robust divergence," not "Hölder robustness."

---

## 3. SCIGEN Constraints and Their Effect on Likelihood

### 3.1 Constraints Reshape the Distribution

A major discussion topic was how SCIGEN constraints affect the likelihood distribution. Key points:

- **Without constraints** (pure DiffCSP): The model's likelihood reflects Materials Project's distribution. Common structures (perovskites, spinels) have high likelihood; rare structures (kagome flat band materials) have low likelihood.

- **With SCIGEN constraints**: The constrained generation restricts the sample space dramatically. This changes the likelihood landscape:
  - Structures that are common in MP but violate the constraint (e.g., non-kagome structures) get likelihood = 0
  - Within the constrained subspace, the relative ranking may be very different from the unconstrained ranking
  - MA: "The difference between what SCIGEN considers common and what DiffCSP considers common is probably huge. The ranking will change a lot."

- **RO noted**: SCIGEN-generated structures include many unstable candidates (DFT doesn't converge). This is noise in the generated distribution, and it's unclear how this affects the likelihood computation.

### 3.2 Should SCIGEN Be in the Training Loss?

This connects to Point 5 of MA's earlier written feedback. The discussion reached a pragmatic conclusion:

- MA originally thought SCIGEN could be disentangled from the DPO objective
- But since the preference dataset is constructed from SCIGEN-generated structures, strictly speaking SCIGEN should be incorporated
- **Practical plan**: Compare SCIGEN-free and SCIGEN-aware losses experimentally
- For the likelihood experiment (step 0): compute likelihoods both with and without SCIGEN constraints, and compare

### 3.3 RO's Distribution Diagram

RO drew a diagram (referenced multiple times in the transcript) showing:
- A large blob = Materials Project distribution (what DiffCSP learned)
- A smaller, partially overlapping blob = SCIGEN-constrained distribution
- The SCIGEN blob extends partly outside MP (generates novel structures not in the database)
- The "tail" of the SCIGEN distribution contains the rare, potentially interesting flat band candidates

MA agreed this captures the situation well, and noted that the SCIGEN constraint region might be "extremely narrow" compared to the MP distribution — much narrower than the diagram suggests.

---

## 4. Band Structure Data in Training: Not Required, But Helpful

A question from RO that had been lingering since earlier sessions:

**RO**: "The DPO training uses crystal structure $x$ as input and preference (winner/loser) as the label. But the preference is derived from band structure, which is derived from crystal structure. The band structure itself never enters the training data. Is this a problem?"

**MA's answer**: No, it's not a fundamental problem. By first-principles physics, the crystal structure determines the electronic structure — band structure is an intermediate product, not a hidden confounder. The necessary and sufficient information is in the crystal structure.

**However**: Having band structure data would improve **data efficiency**. The model would need fewer preference pairs to learn the same signal if band structure features were included as conditioning. But if you can collect enough preference data, it works without band structure in the loop.

**RO's interpretation**: This means we can start with a pure structure → preference pipeline (no band structure in training), and later consider adding band structure features if data efficiency becomes a bottleneck.

---

## 5. Concrete Action Plan

### Step 0: Likelihood Computation (MA leads)

**Goal**: Compute the likelihood of existing crystal structures under the pre-trained DiffCSP/SCIGEN model.

**Method** (MA to provide details):
1. Take a crystal structure $x_0$ from the dataset
2. Forward-noise it to $x_T$ (pure noise)
3. Denoise back to $\hat{x}_0$ using the pre-trained model
4. The reconstruction quality (aggregated per-step log-probability) gives the likelihood

**Samples to evaluate**:
- Common MP materials (perovskites, spinels) — expected high likelihood under DiffCSP
- SCIGEN-generated kagome/Lieb candidates — expected lower likelihood under DiffCSP
- Existing preference data (~600 pairs) — check if the model's likelihood ranking correlates with human preference labels

**Expected outcome**:
- Perovskites: high likelihood (common in training data)
- Flat band materials: low likelihood (rare)
- If this basic sanity check passes → proceed to fine-tuning

**Complication**: The F-channel likelihood is approximate (wrapped normal, not exact Gaussian). The lattice and atom-type channels should be straightforward.

### Step 1: Sanity Check

After computing likelihoods:
- Plot the likelihood distribution for different material categories
- Check: does the model's ranking match physical intuition?
- Compare likelihoods with and without SCIGEN constraints
- If the results are sensible → proceed. If not → debug (wrong math? wrong code? or genuine high-dimensional weirdness?)

**MA's comment**: "It's probably fine — the model generates reasonable structures, so the likelihood is probably correct. But we should confirm."

**Interesting side question** (MA): What is the correlation between DiffCSP's likelihood and DFT formation energy? If there's a strong correlation, that itself could be a paper-worthy observation. (RO agreed this would be surprising and interesting.)

### Step 2: Fine-Tune with Preference Data

Using the existing ~600 preference pairs:
- Fine-tune with Hölder-Diffusion-DPO
- Compare likelihood distributions before and after fine-tuning
- Check: does the fine-tuned model assign higher likelihood to human-preferred structures?
- Check: is the distribution shift in the right direction (toward flat band materials)?

### Step 3: Evaluate Generation Quality

After fine-tuning:
- Generate structures from the fine-tuned model
- Check if flat band structures are generated more frequently
- Check for mode collapse (does it only generate one type of structure?)
- Use ML-based band structure prediction (not DFT) for rapid evaluation
- If results look promising → run DFT on the best candidates

### Step 4 (Future): Scale Up

- If the ML-based band predictor works → use it for both preference labeling and evaluation
- Potentially automate the entire pipeline (generate → ML band prediction → preference labeling → fine-tune)
- Consider having VLMs do the A/B testing automatically (MA mentioned this possibility)

---

## 6. Discussion: Research Methodology

MA shared his general approach to ML research:

1. **Verify assumptions** — Check that the premises are correct before implementing
2. **Identify unsolved problems** — What specifically is the gap?
3. **Define evaluation** — How will you know if it worked?
4. **Implement** — Only after the above are clear

He cautioned against jumping to implementation too early: "If you implement first, you'll see results go up and down and get confused. Better to know what you're looking for."

RO acknowledged this is different from his instinct ("implement first, think later") and expressed interest in adopting MA's approach.

---

## 7. Side Discussions

### 7.1 Diffusion Models Are Mysterious

MA noted that diffusion models have many heuristics that work in practice but lack theoretical justification. The likelihood-based evaluation is one such area — in image generation, likelihood doesn't correlate well with perceptual quality (e.g., the "dimension curse" where the most average-looking sample maximizes likelihood but looks unnatural). Whether this transfers to crystal structures is an open and interesting question.

### 7.2 Potential for a "Likelihood Paper"

If it turns out that DiffCSP's likelihood correlates with formation energy or stability, this observation alone could be a standalone paper. MA: "Nobody really understands how diffusion model likelihood relates to physical properties — if we find a clean relationship, that's interesting."

### 7.3 Agentic Automation

Brief discussion about automating the pipeline with LLM agents (prompt → generate → evaluate → fine-tune). MA: possible in principle, but should only be done after the fundamentals are understood. Reference to El Agente Sólido (Aspuru-Guzik, UofT) and Terence Tao's use of AI for mathematical proofs — both are powerful but limited to problems where sufficient information exists in the literature.

### 7.4 Academia vs Industry

Both agreed that the "ranking" component (defining what makes a material interesting) is where academia can contribute uniquely. Industry has the compute and models, but the scientific judgment of what's interesting requires domain expertise that lives in academic labs.

---

## 8. Data Availability Summary

| Data | Status | Notes |
|------|--------|-------|
| MP-20 crystal structures | Available | ~45k structures, pre-training data for DiffCSP |
| MP-20 band structures | Partially available | Not all entries have band structure; need to query API |
| SCIGEN (v1) generated structures | Available | Multiple motifs (kagome, Lieb, etc.) |
| SCIGEN (v1) DFT stability | Available | Subset screened with SMACT + DFT |
| SCIGEN (v1) Lieb band structures | Available | Only Lieb lattice; limited set |
| SCIGEN+ generated structures | Not yet | New model, needs new generation runs |
| Preference data (A/B test) | Available | ~600 pairwise comparisons via band_eval_web |
| ML band structure predictor | Available | Less accurate than DFT, but fast; suitable for rapid prototyping |

---

## 9. Action Items

### Masaki Adachi (MA)
- [ ] Provide the likelihood computation method/code for DiffCSP (how to compute per-sample likelihood given a pre-trained model) — aim for today/this week
- [ ] Clarify how to handle the F-channel (wrapped normal) in the likelihood computation
- [ ] Review RO's likelihood results when available

### Ryotaro Okabe (RO)
- [ ] Implement likelihood computation following MA's instructions
- [ ] Compute likelihoods for: (a) common MP materials (perovskites), (b) SCIGEN-generated kagome/Lieb candidates, (c) existing preference data pairs
- [ ] Visualize the likelihood distributions and check for physical consistency
- [ ] Compare likelihoods with and without SCIGEN constraints
- [ ] Share results with MA for joint analysis
- [ ] (Later) Implement Hölder-Diffusion-DPO fine-tuning using the loss from `material_dpo.tex`

### Scheduling
- Next meeting: **not next week** (RO at APS conference, MA traveling)
- Resume meetings after results from the likelihood experiment are available
- Async communication via Slack in the meantime

---

## 10. Key Takeaways

1. **Likelihood is the foundation**: Everything in DPO reduces to whether the model's likelihood correctly ranks materials. Verify this first.

2. **F-channel is the hard part**: Lattice and atom types use DDPM (Gaussian, tractable). Fractional coordinates use Wrapped Normal / score matching — likelihood is only a proxy.

3. **Band structure not needed in training**: The crystal structure contains sufficient information. Band structure would improve data efficiency but isn't required.

4. **SCIGEN dramatically changes the distribution**: The constrained and unconstrained likelihood landscapes are probably very different. Must compare both.

5. **Don't implement prematurely**: Verify assumptions → identify gaps → define evaluation → then implement.

6. **Hölder-DPO's dual use**: Not just noise removal — the outlier detection mechanism can surface novel materials.

---

## ADDENDUM (Mar 12, 2026): MA's Correction on Likelihood and SCIGEN

**After the meeting, MA sent a critical correction:**

> "I just realized the likelihood estimation for diffusion is super complicated than I thought. I was wrong in explanation. Current Diffusion-DPO considers only forward process; i.e., crystal to noise $x \to \varepsilon$ and we compute likelihood proxy in noise space PDF. Thus, there is no difference between SCIGEN and its base model (DiffCSP) in Diffusion-DPO formulation. So, I will consider more on how to incorporate reverse process, i.e., noise to crystal $\varepsilon \to x$."

**Impact on this report:**

- **Section 2.1** described likelihood computation as "go forward, go backward." This is incorrect for Diffusion-DPO — it uses the **forward process only** (noise a clean sample, evaluate denoising quality). The model never generates during DPO training.

- **Section 3** discussed how SCIGEN constraints reshape the likelihood distribution. This is now in question — since SCIGEN operates during the **reverse** (generation) process, and Diffusion-DPO only uses the **forward** process, **SCIGEN has no effect on the standard DPO loss**. There is no difference between DiffCSP and SCIGEN from the DPO formulation's perspective.

- **Section 5 (Action Plan)**: Step 0 (likelihood sanity check) is still valid but the interpretation changes — we are checking the forward-process denoising quality, not the full generative likelihood. The SCIGEN-aware vs. SCIGEN-free comparison may be moot under the current formulation.

- **Takeaway 4** ("SCIGEN dramatically changes the distribution") needs revision. Under standard Diffusion-DPO, SCIGEN doesn't change anything. MA is now working on how to incorporate the reverse process to make the DPO loss SCIGEN-aware.

See `2026-03-12_likelihood_clarification_questions.md` Section 7 for detailed analysis of this correction and revised communication plan.
