# Change Log - scigenp_overview_MA

> Track updates to the overleaf paper and their implications

**Overleaf location**: `/pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview_MA/`

---

## Template Entry (Delete this section when you add your first entry)

```markdown
## YYYY-MM-DD: [Brief Description]

### Source
- **Commits**: [commit hash] - [commit message]
- **Author**: [author name/collaborator]
- **Files changed**: [list of files]

### Summary of Changes
[Brief summary of what was modified]

### User Comments
[Paste comments/messages from collaborators here]

### Impact on Implementation
- [ ] Requires code changes
- [ ] Requires new experiments
- [ ] Documentation only
- [ ] No action needed

### Related Tasks
- Link to implementation_plan.md tasks if applicable

---
```

## Change History

<!-- Add new entries below, most recent first -->

## 2026-03-16: Voice Transcript Reading Session 1

### Sections Covered
- ✅ Overview (lines ~60-78)
- ✅ Section 2: Per-Channel Crystal Diffusion (lines ~81-155)
- ✅ Section 3.1: DPO Background (lines ~160-175)
- ✅ Section 3.2: Diffusion-DPO (lines ~176-360)
- ✅ Section 3.3: Bridge Formulation (lines ~361-680)
- ✅ Section 3.4: Robustness (lines ~681-967)
- ⏸️ Section 4: Three-Phase Training - not started

### Method
Voice transcript reading with real-time question extraction

### Questions Asked
**47 questions total** extracted from voice transcript

**Key questions:**
- Q4: How to derive DPO loss form?
- Q7: What is wrapped normal proxy reverse kernel?
- Q16: Why is Hölder loss defined this way?
- Q23: Why does SCIGEN apply constraints after each step?
- Q29: What is "bridge" in diffusion models?
- Q38: Why normalized error form?

### Key Breakthroughs
- ✅ **Wrapped difference** is for periodic torus topology (fractional coords live on [0,1)³)
- ✅ **Pseudo-bridge** reconstructs trajectories online (storage-efficient alternative to storing full rollouts)
- ✅ **Constraint cancellation** (Lemma 3.1) reduces compute - only train on free DOF
- ✅ **Hölder robustness** auto-detects outliers via influence function (no manual confidence weighting needed)
- ✅ **Bridge formulation** solves endpoint-only data problem in Phase B

### Still Complex (Normal!)
- [ ] Bridge formulation math (Proposition 3.2 - exact posterior bridge vs pseudo-bridge)
- [ ] Robustness proofs (influence functions, Chebyshev inequality application)
- [ ] K-bridge endpoint score usage in practice
- [ ] Optimal bridge level distribution ρ(b)

### Clarifications Documented
All 47 Q&A pairs saved to: [reading_session_2026-03-16.md](./reading_session_2026-03-16.md)

Topics covered:
- Executed constrained reverse policy (Q1)
- Bradley-Terry model derivation (Q3-Q4)
- Improvement score derivation (Q5)
- Wrapped difference & wrapped Gaussian (Q6-Q8, Q13-Q14)
- Forward consistent pair & simple coupling (Q9-Q11)
- Tractable proxy & predictor-corrector (Q12-Q13)
- Hölder loss form & γ tuning (Q15-Q17, Q20-Q21)
- Confidence score usage (Q18-Q19)
- Bridge concept & reconstruction (Q22-Q35, Q39-Q41)
- Pseudo-bridge residuals & normalization (Q36-Q38)
- Robustness guarantees (Q42-Q47)

### Must-Read References Identified
1. **DPO derivation:** Rafailov et al. (2023) "Direct Preference Optimization", Appendix A
2. **Diffusion-DPO:** Wallace et al. (2024) "Diffusion Model Alignment Using DPO", Appendix B
3. **Hölder robustness:** Fujisawa et al. (2025) "Scalable Valuation of Human Feedback", Sections 4-5, Appendix D
4. **Bradley-Terry:** Original (1952) or DPO Section 2.1
5. **Score-based models:** Song et al. (2021) - for predictor-corrector understanding
6. **DiffCSP:** Jiao et al. (2023) - for wrapped normal on coordinates

### Next Session Plan
1. **Read:** Section 4 (Three-Phase Training) - more concrete/practical
2. **Revisit:** Section 3.3 bridge formulation with fresh perspective after break
3. **Study:** Algorithm 1 pseudocode (often clearer than dense equations)
4. **Prepare:** Questions for Masaki about implementation details

### Code Correspondence Tasks
- [ ] Find where improvement scores I_θ are computed
- [ ] Locate SCIGEN constraint masks C^(z)
- [ ] Understand bridge reconstruction implementation
- [ ] Check how confidence κ is used in diagnostics

### Implementation Questions for Masaki
- [ ] What bridge level distribution ρ(b) do we use in practice?
- [ ] How many K samples for K-bridge evaluation?
- [ ] What γ value works best for our domain?
- [ ] Do we implement full predictor-corrector or just proxy?

---
