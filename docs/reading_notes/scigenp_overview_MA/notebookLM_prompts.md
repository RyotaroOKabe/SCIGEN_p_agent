# NotebookLM Prompts for SCIGEN+ Material DPO

> Prompts to create study materials from material_dpo.tex + reading session Q&A

---

## 📋 Setup Instructions

### Step 1: Upload Sources to NotebookLM

Upload these files:
1. **Primary source:** `material_dpo.tex` (or compiled PDF)
2. **Q&A supplement:** `reading_session_2026-03-16.md`
3. **Optional:** `technical_terms.md`, `key_concepts.md`

### Step 2: Wait for Processing

NotebookLM will index all sources (~2-3 minutes)

---

## 🎯 Prompts for Different Use Cases

### **Prompt 1: Comprehensive Slide Deck (Overview)**

```
Create a comprehensive slide deck (20-25 slides) explaining the SCIGEN+ DPO framework
from material_dpo.tex, incorporating clarifications from reading_session_2026-03-16.md.

Structure:
1. Title slide: "SCIGEN+: Preference-Guided Flat Band Discovery"
2. Motivation (2-3 slides):
   - Why flat band materials matter
   - Limitations of geometric constraints alone
   - Need for human preference learning

3. Background (4-5 slides):
   - Per-channel crystal diffusion (Lattice, Fractional coords, Atom types)
   - DPO basics (Bradley-Terry model, why not RLHF)
   - SCIGEN constraint mechanism

4. Core Method (6-8 slides):
   - Phase A: Unconstrained forward-proxy DPO
   - Phase B: Constrained reverse-policy DPO (bridge formulation)
   - Hölder robustness for noisy labels
   - Constraint cancellation lemma

5. Three-Phase Training (3-4 slides):
   - Phase A: Offline DPO on MP-20
   - Phase B: Motif-focused adaptation
   - Phase C: Active learning loop

6. Key Innovations (2-3 slides):
   - Multi-channel aggregation
   - Pseudo-bridge reconstruction
   - Normalized masked errors
   - Confidence-free Hölder training

7. Next Steps & Discussion (1-2 slides)

For each slide:
- Include key equations from material_dpo.tex
- Add "Key Point" callouts from Q&A answers
- Reference specific Q numbers (e.g., "See Q7 for wrapped normal explanation")
- Use concrete examples where available in Q&A
```

---

### **Prompt 2: Technical Deep-Dive Slides (Equations Focus)**

```
Create a technical slide deck (15-20 slides) focusing on the mathematical formulation,
using material_dpo.tex equations and Q&A clarifications from reading_session_2026-03-16.md.

For each major section, create slides with:

**Slide format:**
- Title: [Section name + Equation range]
- Left column: Key equations from material_dpo.tex
- Right column:
  * "What it means" (intuitive explanation from Q&A)
  * "Why it matters" (from Q&A answers)
  * Common confusions addressed (from Q&A questions)

**Sections to cover:**
1. Per-Channel Diffusion (Equations 2.1-2.7)
   - Reference Q2, Q6-Q8, Q13-Q14 from Q&A

2. Denoising Improvement Scores (Equations 3.3-3.11)
   - Reference Q5, Q15 from Q&A

3. Hölder-DPO Loss (Equations 3.14-3.15, 3.29-3.31)
   - Reference Q16-Q21 from Q&A

4. Bridge Formulation (Equations 3.19-3.27, 3.32)
   - Reference Q22-Q35 from Q&A

5. Pseudo-Bridge Reconstruction (Equations 3.36-3.45)
   - Reference Q29-Q41 from Q&A

6. Robustness Guarantees (Equations 3.47-3.54)
   - Reference Q42-Q47 from Q&A

For equations that had Q&A clarification, add a colored box with "Common Question"
and the simplified explanation.
```

---

### **Prompt 3: Concept Map / Flowchart Slides**

```
Create visual flowchart slides (8-10 slides) showing the relationships between concepts
in material_dpo.tex, using clarifications from reading_session_2026-03-16.md.

**Slide 1: Overall Architecture**
Flowchart showing:
- Input: Crystal structure x₀ = (L, F, A)
- Three channels (L, F, A) with different diffusion processes
- Preference pairs (x^w, x^ℓ)
- Improvement scores → Margin → Hölder loss
- Output: Fine-tuned model

**Slide 2: Phase A vs Phase B Decision Tree**
Decision tree:
- Q: "Do we have SCIGEN constraints during training?"
  - NO → Phase A (forward-proxy, Eq 3.14)
    * Context: c_A = (chemistry, N-bucket, symmetry)
    * Data: MP-20 materials
    * See Q22 for "forward corrupted state"
  - YES → Phase B (bridge, Eq 3.43)
    * Context: c_B = (C, N, A^c)
    * Data: SCIGEN-generated + human labels
    * See Q23, Q29-Q35 for bridge explanation

**Slide 3: Improvement Score Computation Flow**
For each channel z ∈ {L, F, A}:
1. Sample timestep t
2. Add noise: x₀ → x_t (or x₀ → x_b → rollout for Phase B)
3. Predict: ε_θ(x_t, t), ε_ref(x_t, t)
4. Compute error: d_θ, d_ref (with masking in Phase B)
5. Improvement: I_θ^(z) = ω_t(d_ref - d_θ)
6. Aggregate: I_θ = Σ_z I_θ^(z)
See Q5, Q36-Q38 for derivation and normalization

**Slide 4: Bridge Reconstruction Process** (Reference Q29-Q35)
Step-by-step flowchart:
1. Start: Observed endpoint x₀
2. Sample bridge level b ~ ρ(·)
3. Forward noise: x₀ → x_b
4. Reverse rollout (b steps):
   - For t = b down to 1:
     * Propose: u_{t-1} ~ p_ref(·|x_t)
     * SCIGEN overwrite: x_{t-1} = Π_C(u_{t-1})
5. Result: Pseudo-bridge τ̂ = (x_b, ..., x̂₀)
6. Compute improvement on τ̂
Note: x̂₀ ≠ x₀ (but good enough!)

**Slide 5: Constraint Cancellation Visual** (Reference Q24-Q27)
Side-by-side comparison:
- Left: Full structure (fixed + free DOF)
- Middle: Constraint mask C
- Right: After cancellation (only free DOF)
Show equation 3.27: Constrained parts cancel in log-ratio

**Slide 6: Hölder Influence Function** (Reference Q16, Q19)
Plot showing:
- X-axis: Margin u = β·T·g_θ(t)
- Y-axis: Influence weight ι_γ(u) = σ(u)^γ(1-σ(u))²
- Show curves for different γ values
- Annotate regions:
  * u >> 0: Model agrees, high influence
  * u ≈ 0: Uncertain, high influence
  * u << 0: Model disagrees (outlier), LOW influence (redescending!)

**Slide 7: Three-Phase Data Flow**
Three parallel lanes:
- Phase A: MP-20 → score-based pairs → unconstrained H-DPO → θ_A
- Phase B: θ_A + SCIGEN → generate → human labels → bridge H-DPO → θ_B
- Phase C: θ_B → generate → active acquisition → human labels → retrain (loop)

**Slide 8: Why Each Component Matters**
Table or mind map:
- Wrapped difference → Periodic boundaries (Q6-Q8)
- Simple coupling → Trajectory consistency (Q9-Q11)
- Tractable proxy → Computational efficiency (Q12-Q13)
- Constraint cancellation → Reduce computation (Q26-Q27)
- Pseudo-bridge → Handle endpoint-only data (Q29-Q35)
- Normalized errors → Fair cross-channel comparison (Q38)
- Hölder loss → Robust to noisy labels (Q16, Q18)
```

---

### **Prompt 4: Q&A-Based FAQ Slides**

```
Create FAQ slides (10-15 slides) directly from the questions in reading_session_2026-03-16.md,
organized by difficulty level.

**Beginner Level (3-4 slides):**
Q&A from the session that are fundamental:
- Q1: What is "executed constrained reverse policy"?
- Q3: What is Bradley-Terry model?
- Q22: What is "forward corrupted state"?
- Q32: What is "bridge level" b?

**Intermediate Level (4-5 slides):**
Q&A requiring some background:
- Q7: What is wrapped normal proxy reverse kernel?
- Q15: How to derive DPO margin?
- Q23: Why does SCIGEN apply constraints after each step?
- Q26: What does "free" mean in constraint cancellation?

**Advanced Level (3-4 slides):**
Q&A on complex topics:
- Q29: What is "bridge" in diffusion? Exact vs pseudo-bridge?
- Q35: Why is pseudo-bridge a "practical round-trip approximation"?
- Q38: Why normalized error form?
- Q47: Chebyshev's sum inequality application?

**For each slide:**
- Title: The question verbatim
- Answer: Simplified explanation from Q&A
- Connection: Reference to material_dpo.tex equation
- Visual: Diagram or equation when helpful
```

---

### **Prompt 5: Before-Meeting Preparation Slides**

```
Create a focused slide deck (10 slides) for preparing to discuss implementation details
with Masaki, based on material_dpo.tex and reading_session_2026-03-16.md.

**Slide 1: Reading Progress Summary**
- Sections covered: Overview, Sections 2-3.4 ✓
- Sections remaining: Section 4 (Three-Phase Training)
- Key breakthroughs: 5 main insights
- Open questions: 4 items

**Slides 2-5: Implementation Questions** (from changes_log.md)
Each slide = 1 question with:
- Question
- Why it matters (context from material_dpo.tex)
- What I understand so far (from Q&A)
- What I need clarification on

Questions:
1. Bridge level distribution ρ(b)? (Reference Q32-Q34)
2. K-bridge evaluation: How many samples? (Reference Q39)
3. Optimal γ value for our domain? (Reference Q17)
4. Predictor-corrector vs proxy implementation? (Reference Q12-Q13)

**Slides 6-9: Code Correspondence Questions**
Each slide = 1 code question with:
- Where in paper (equation/section)
- Expected location in SCIGEN_p_agent
- What to look for

Topics:
1. Improvement scores I_θ computation (Eq 3.5-3.9)
2. SCIGEN constraint masks C^(z) (Eq 3.25-3.26)
3. Bridge reconstruction (Eq 3.36-3.43)
4. Confidence κ diagnostics (Section 3.2.4)

**Slide 10: Equations to Discuss**
List of equations that need clarification for implementation:
- Eq 3.16 (Wrapped normal proxy reverse kernel)
- Eq 3.43 (Bridge H-DPO loss)
- Eq 3.41 (Normalized masked errors)
Include: Page numbers, current understanding, specific confusion
```

---

### **Prompt 6: Comparison Slides (vs Related Work)**

```
Create comparison slides (6-8 slides) showing how SCIGEN+ DPO differs from related methods,
using material_dpo.tex and insights from reading_session_2026-03-16.md.

**Slide 1: SCIGEN+ vs Standard DPO**
Side-by-side table:
| Aspect | Standard DPO | SCIGEN+ DPO |
|--------|--------------|-------------|
| Domain | Language (discrete tokens) | Crystals (continuous + discrete) |
| Channels | Single | Multi-channel (L, F, A) |
| Likelihood | Tractable (autoregressive) | Intractable (diffusion) |
| Proxy | Log-ratio | Denoising improvement |
| Constraints | None | SCIGEN masks (Phase B) |

**Slide 2: Phase A vs Phase B**
Comparison table with Q&A references:
| Aspect | Phase A | Phase B |
|--------|---------|---------|
| Constraints? | No | Yes (SCIGEN) |
| Data | Forward-corrupted (Q22) | Pseudo-bridge (Q29-Q35) |
| Context | Broad (c_A) | Tight (c_B) |
| Loss | Eq 3.14 | Eq 3.43 |
| Masks | None | C^(z) (Q26) |
| Learning | General preferences | Motif-specific |

**Slide 3: Hölder-DPO vs Standard DPO**
| Aspect | Standard DPO | Hölder-DPO |
|--------|--------------|------------|
| Loss | -log σ(·) | ℓ_γ(·) (Q16) |
| Robustness | No | Yes (redescending) |
| Outliers | Full influence | Vanishing influence (Q19) |
| Confidence use | Often weighted | Diagnostics only (Q18) |
| γ parameter | N/A | 2.0 default (Q17) |

**Slide 4: Exact Bridge vs Pseudo-Bridge** (Q29-Q35)
| Aspect | Exact Bridge | Pseudo-Bridge |
|--------|--------------|---------------|
| Data needed | Stored trajectories | Endpoints only |
| Storage | ~GB | ~MB |
| Accuracy | Perfect | Approximate |
| Endpoint match | x̂₀ = x₀ | x̂₀ ≠ x₀ |
| Practical? | No (expensive) | Yes (used in practice) |

**Additional slides:**
- SCIGEN+ vs RLHF for diffusion
- Forward-proxy (Phase A) vs Reverse-policy (Phase B)
- Wrapped Normal vs Regular Gaussian (Q8)
```

---

### **Prompt 7: Study Guide / Cheat Sheet Slides**

```
Create compact reference slides (5-7 slides) as a "cheat sheet" for key concepts,
using material_dpo.tex equations and Q&A from reading_session_2026-03-16.md.

**Slide 1: Notation Cheat Sheet**
Two columns:
- Left: Symbol | Meaning | Where defined
- Right: Common confusions from Q&A

Key symbols:
- x₀ = (L, F, A) | Clean crystal | Q6
- x_t | Noised state | Q22
- τ | Trajectory | Q29
- I_θ(x,t) | Improvement score | Q5
- g_θ(t) | DPO margin | Q15
- C^(z) | Constraint mask | Q26
- b | Bridge level | Q32
- γ | Hölder exponent | Q16, Q17

**Slide 2: Key Equations Reference**
Essential equations with one-line explanations:
- Eq 2.15: Wrapped difference (Q6-Q7)
- Eq 3.5-3.9: Improvement scores (Q5)
- Eq 3.14: Phase A loss (unconstrained)
- Eq 3.27: Constraint cancellation (Q26)
- Eq 3.43: Phase B loss (bridge)
- Eq 3.29: Hölder influence gradient (Q21)

**Slide 3: Decision Trees**
When to use what:
- Lattice vs Fractional coords vs Atom types → Different diffusion (Q2)
- Phase A vs Phase B → Constraints active? (Q1, Q23)
- Forward-corrupted vs Pseudo-bridge → Which phase? (Q22, Q29)
- Exact vs Pseudo bridge → Trajectories stored? (Q31)

**Slide 4: Common Pitfalls** (from Q&A)
- ❌ Don't use Euclidean distance for fractional coords → Use wrapped (Q6-Q8)
- ❌ Don't skip simple coupling → Trajectory consistency breaks (Q9-Q11)
- ❌ Don't weight by confidence κ → Hölder handles it (Q18-Q19)
- ❌ Don't always use b=T → Smaller b often better (Q33-Q34)
- ❌ Don't forget normalization in Phase B → Unfair comparison (Q38)

**Slide 5: Derivation Roadmap**
Flow showing how to derive key results:
1. DDPM reverse kernel → Log-ratio (Eq 3.2)
2. Reparameterize with noise pred → Improvement score (Q5)
3. Aggregate channels → Total improvement (Eq 3.11)
4. Winner vs loser → Margin (Q15)
5. Bradley-Terry + KL-regularized RL → DPO (Q4)
6. Robust divergence → Hölder-DPO (Q16)

**Slide 6: Quick Reference - Three Phases**
Compact table:
| | Phase A | Phase B | Phase C |
|-|---------|---------|---------|
| Data | MP-20 DFT | SCIGEN-gen | Uncertain pairs |
| Size | 3.5-5k pairs | 1.5-4.2k/motif | 250-500 |
| Goal | General prefs | Motif-specific | Refine |
| Context | c_A (broad) | c_B (tight) | c_B |
| Loss | Eq 3.14 | Eq 3.43 | Eq 3.43 |

**Slide 7: Must-Read Papers** (from Q&A)
For each paper:
- Citation
- What to read (specific sections)
- Why it matters for understanding

List:
1. Rafailov et al. (2023) - DPO derivation (Q3-Q4)
2. Wallace et al. (2024) - Diffusion-DPO (Q5)
3. Fujisawa et al. (2025) - Hölder robustness (Q16, Q43-Q47)
4. Jiao et al. (2023) - DiffCSP details (Q2, Q13)
5. Song et al. (2021) - Score-based models (Q12-Q13)
```

---

## 🎤 Bonus: Audio Podcast Prompts

If you want NotebookLM to generate an audio discussion:

### **Podcast Prompt 1: Technical Walkthrough**
```
Generate a podcast episode (15-20 min) explaining the SCIGEN+ DPO framework
from material_dpo.tex, using Q&A from reading_session_2026-03-16.md to
address common confusions.

Hosts: 2 researchers discussing the paper

Structure:
1. Introduction: What is SCIGEN+? Why combine constraints + preferences?
2. Deep dive: Phase A vs Phase B (reference Q22, Q23, Q29)
3. Tricky concepts: Wrapped normal, pseudo-bridge (reference Q7, Q8, Q29-Q35)
4. Hölder robustness: Why it matters (reference Q16-Q19)
5. Open questions: What's still unclear (reference "Still Complex" from changes_log)

Make sure hosts address the 47 questions that came up during reading.
Use concrete examples where the Q&A provided them.
```

### **Podcast Prompt 2: Before-Meeting Prep**
```
Generate a podcast episode (10-12 min) reviewing key concepts to discuss
with Masaki, based on material_dpo.tex and reading_session_2026-03-16.md.

Hosts: Student (you) and advisor (reviewing together)

Topics:
1. What I understood well (breakthroughs from changes_log)
2. Implementation questions (4 questions from changes_log)
3. Code correspondence tasks (where to look in SCIGEN_p_agent)
4. Equations needing clarification (Eq 3.16, 3.41, 3.43)

Use this as practice for the actual meeting.
```

---

## 💾 Save These Prompts

**Usage:**
1. Copy desired prompt
2. Paste into NotebookLM chat
3. Review generated content
4. Iterate with follow-ups like:
   - "Add more visual diagrams to slide 3"
   - "Expand the explanation on slide 7 using Q&A answer from Q16"
   - "Create a speaker notes version for each slide"

**Export options:**
- Download slides as PDF
- Copy to Google Slides for editing
- Generate speaker notes
- Create study guide version

---

## 🔄 Iterative Refinement Prompts

After initial generation, use these follow-ups:

```
"For each slide mentioning an equation, add a 'Common Question' callout
box if there was a related question in reading_session_2026-03-16.md"
```

```
"Add speaker notes to each slide with:
- Key points to emphasize
- Potential questions from audience
- References to Q&A answers"
```

```
"Create a condensed 10-slide version for a 15-minute presentation,
keeping only the most essential concepts"
```

```
"Generate quiz questions for each major section to test understanding"
```

---

## ✅ Recommended Workflow

1. **Start with Prompt 1** (Comprehensive slides) - Get overview
2. **Use Prompt 4** (Q&A FAQ) - Study specific confusions
3. **Use Prompt 5** (Before-meeting prep) - Prepare for Masaki
4. **Export and refine** - Download, add your notes
5. **Generate audio** - Listen while commuting/exercising

Good luck with your presentation!
