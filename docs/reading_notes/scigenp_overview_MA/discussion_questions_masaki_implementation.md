# Discussion Questions for Masaki - SCIGEN+ DPO Implementation

**Date:** 2026-03-17
**Topic:** Implementation plan for SCIGEN+ DPO (Hölder-DPO with bridge formulation)
**Paper:** scigenp_overview_MA (SCIGEN+ with Direct Preference Optimization)

---

## 📋 Overview Questions

### 1. Project Scope & Timeline

**Q1.1:** What is the target timeline for implementing SCIGEN+ DPO? Are we aiming for:
- Proof-of-concept (minimal working version)?
- Full three-phase training pipeline?
- Research-grade reproducible implementation?

**Q1.2:** Which phases should we prioritize?
- Phase A only (offline DPO on MP-20)?
- Phase A + Phase B (motif-focused bridge training)?
- Complete three-phase pipeline (A + B + C with active learning)?

**Q1.3:** What is the expected scope for validation/experiments?
- Single motif (e.g., kagome only)?
- Multiple motifs (kagome, lieb, honeycomb)?
- Full paper reproduction with all evaluation metrics?

**Q1.4:** Are there any hard deadlines or milestones we need to hit?

---

## 🏗️ Architecture & Codebase

### 2. Existing Infrastructure

**Q2.1:** Which existing codebase should we build on?
- Original DiffCSP++ repository?
- Your modified version of DiffCSP++?
- Start fresh with new implementation?

**Q2.2:** Do we have a working DiffCSP++ baseline already?
- Pre-trained model available?
- Training pipeline working?
- Generation working with current code?

**Q2.3:** Is SCIGEN projection already implemented?
- If yes, where in the codebase?
- Does it support all constraint types (kagome, lieb, honeycomb)?
- Does it handle partial constraints correctly?

**Q2.4:** What is the current state of the preference annotation tool (`band_eval_web`)?
- Is it functional and deployed?
- Do we have existing preference data?
- How many pairs have been annotated so far?

---

## 🎯 Phase A Implementation

### 3. Offline DPO on MP-20

**Q3.1:** Do we have the MP-20 dataset preprocessed and ready?
- Clean structures available?
- Forward corruption pairs prepared?
- How many preference pairs do we need to generate?

**Q3.2:** How should we generate preference pairs from MP-20?
- DFT band structure calculations (expensive!)?
- Proxy metrics (e.g., CHGNet screening)?
- Synthetic preferences based on heuristics?

**Q3.3:** For the improvement score $I_\theta(x, t)$:
- Should we implement for all three channels (L, F, A)?
- Can we simplify to fewer channels initially?
- What about the weighted sum $\omega_t^{(z)}$?

**Q3.4:** Hölder loss implementation:
- Confirm $\gamma = 2.0$ as the robustness parameter?
- Should we implement γ-tuning diagnostics from the start?
- Do we need the influence function calculation for debugging?

**Q3.5:** Training hyperparameters for Phase A:
- Learning rate and schedule?
- Batch size (how many pairs per batch)?
- Number of timesteps $T$ (paper uses 1000)?
- Preference sharpness $\beta$ (paper uses 0.1)?
- How many training epochs/iterations?

---

## 🌉 Phase B Implementation

### 4. Bridge Formulation (Most Complex!)

**Q4.1:** Should we implement Phase B at all in the first iteration?
- If yes, how much of it?
- If not, what's the plan for adding it later?

**Q4.2:** Pseudo-bridge reconstruction:
- Bridge level $b$ - use fixed value or sample from distribution?
- Which distribution $\rho(b)$: Uniform(1, T), Uniform(1, T/2), or geometric?
- Should we implement K-bridge ($K$ samples) for evaluation?

**Q4.3:** SCIGEN-generated dataset for Phase B:
- Do we need to generate this from scratch?
- How many structures per motif (paper uses 300-800)?
- Which motifs to focus on (kagome only, or multiple)?
- Can we use existing generated structures if available?

**Q4.4:** Constraint mask implementation:
- How to represent $\bar{\mathbf{C}}_{\text{eff}}^{(z)}$ (free mask)?
- Should it be hard-coded per motif or computed dynamically?
- Does it integrate with existing SCIGEN code?

**Q4.5:** Normalized error calculation (Eq 3.40-3.42):
- Confirm we divide by $\|\bar{\mathbf{C}}\|_1$ (number of free DOF)?
- What is $\epsilon_0$ for numerical stability (paper uses $10^{-8}$)?

**Q4.6:** For multi-channel residuals (L, F, A):
- Do we implement all three channels or start with subset?
- Wrapped distance $\Delta(F_{t-1}, \mu_\theta^{(F)})$ for torus - is this already implemented?

**Q4.7:** Training hyperparameters for Phase B:
- Lower learning rate than Phase A (how much lower)?
- Separate learning rates per channel?
- Fine-tuning from Phase A model or train from scratch?
- How long to train Phase B?

---

## 🔄 Phase C Implementation

### 5. Active Learning (Optional)

**Q5.1:** Should we implement Phase C active learning?
- If yes, when (after A+B working)?
- If not initially, is the architecture extensible to add it later?

**Q5.2:** If implementing Phase C:
- Uncertainty sampling: which uncertainty metric?
- Novelty detection: how to measure?
- How many pairs per active learning round (paper uses 50-100)?
- How many rounds to run?

**Q5.3:** Expert annotation workflow:
- Who will be the "expert" annotators?
- Integration with `band_eval_web` tool?
- Turnaround time per annotation round?

---

## 🧮 Mathematical & Algorithmic Details

### 6. Multi-Channel Diffusion

**Q6.1:** Wrapped Gaussian for fractional coordinates:
- Is this already implemented in DiffCSP++?
- If not, should we implement or use approximation?
- Wrapped distance $\Delta(F, G)$ calculation?

**Q6.2:** Simple coupling (same noise $\varepsilon_F$ for forward-consistent pairs):
- Is this critical for Phase A or can we skip initially?
- How to implement efficiently?

**Q6.3:** Predictor-corrector sampling:
- Does DiffCSP++ use this already?
- For DPO, paper uses "tractable proxy" instead - confirm we use single-step Gaussian?

**Q6.4:** Lattice parameterization:
- O(3)-invariant parameters $\mathbf{k}$ (6D)?
- Or direct lattice matrix (9D)?
- Which is easier to implement and maintain?

---

### 7. DPO-Specific Implementation

**Q7.1:** Reference model $p_{\text{ref}}$:
- Use the pre-trained DiffCSP++ as reference and freeze?
- Or train a separate reference model?
- When to "snapshot" the reference (before Phase A training)?

**Q7.2:** Improvement score $I_\theta(x, t)$ computation:
- Requires forward pass through model θ and ref
- How to efficiently compute in batch?
- Should we cache ref model outputs?

**Q7.3:** Margin computation $g_\theta(t) = I_\theta(x^w, t) - I_\theta(x^\ell, t)$:
- Computed on-the-fly during training?
- Pre-computed and cached?

**Q7.4:** Timestep sampling:
- Uniform $t \sim \text{Uniform}(1, T)$?
- Or importance sampling (weight by $\omega_t$)?

**Q7.5:** Gradient computation:
- Manual implementation of Hölder gradient or use autograd?
- Any numerical stability concerns (e.g., sigmoid saturation)?

---

## 📊 Data & Evaluation

### 8. Datasets

**Q8.1:** MP-20 dataset:
- Do we have access to the full MP-20 dataset?
- Pre-processed and filtered?
- How to load and batch?

**Q8.2:** Preference annotations:
- Format: (x_w, x_l, κ, constraint_C)?
- Storage: JSON, pickle, HDF5?
- How to handle confidence scores κ (use for diagnostics only, not training)?

**Q8.3:** SCIGEN-generated structures (Phase B):
- Where to store generated structures?
- How to associate with constraints and preferences?
- Checkpoint and versioning strategy?

---

### 9. Evaluation Metrics

**Q9.1:** What metrics should we track during training?
- Ranking accuracy on validation set?
- High-confidence accuracy (κ ≥ 4)?
- Spearman correlation (κ vs margin)?
- Influence weight distribution?

**Q9.2:** Generation quality metrics:
- Validity (how to check valid crystals)?
- Stability (CHGNet energy, DFT relaxation)?
- Flat band presence (DFT band structure)?
- Diversity (how to measure)?

**Q9.3:** Constraint satisfaction:
- How to verify generated structures satisfy motif constraints?
- Tolerance for floating-point errors?

**Q9.4:** Comparison baselines:
- DiffCSP++ without DPO?
- Phase A only vs Phase A+B?
- Standard DPO (logistic loss) vs Hölder-DPO?

---

## 💻 Implementation Details

### 10. Software Engineering

**Q10.1:** Programming framework:
- PyTorch or JAX?
- Which version (compatibility with existing DiffCSP++ code)?

**Q10.2:** Code organization:
- Separate repo or branch of existing repo?
- Module structure (separate files for Phase A, B, C)?
- Where to put Hölder loss, bridge reconstruction, etc.?

**Q10.3:** Configuration management:
- YAML configs for hyperparameters?
- Hydra for experiment management?
- How to track different runs?

**Q10.4:** Logging and monitoring:
- Weights & Biases (wandb)?
- TensorBoard?
- What to log (loss, accuracy, influence, etc.)?

**Q10.5:** Checkpointing:
- How often to save checkpoints?
- What to save (model, optimizer, training state)?
- Checkpoint format and naming convention?

**Q10.6:** Reproducibility:
- Random seeds?
- Deterministic operations?
- Version control for data and code?

---

## 🖥️ Computational Resources

### 11. Hardware & Compute

**Q11.1:** What compute resources do we have access to?
- GPU type and quantity (A100, V100, etc.)?
- CPU cores and RAM?
- Storage capacity?

**Q11.2:** Training time estimates:
- Expected wall-clock time for Phase A?
- Expected time for Phase B (if implemented)?
- Acceptable training time budget?

**Q11.3:** Memory constraints:
- Batch size limitations?
- Model size (number of parameters)?
- Can we fit full model + ref model in GPU memory?

**Q11.4:** Parallelization:
- Multi-GPU training (DDP, FSDP)?
- Data parallel or model parallel?
- Any distributed training experience/infrastructure?

---

## 🧪 Validation & Debugging

### 12. Testing & Verification

**Q12.1:** How to verify correctness before large-scale training?
- Unit tests for key components (Hölder loss, improvement score)?
- Integration tests (end-to-end forward pass)?
- Overfitting sanity check (train on tiny dataset)?

**Q12.2:** Debugging strategy:
- What to check if training doesn't work?
- Gradient norms, loss curves, accuracy?
- Ablations to isolate issues?

**Q12.3:** Intermediate milestones:
- Phase A: Ranking accuracy > random (50%)?
- Phase A: Ranking accuracy > 70%?
- Phase B: Generated structures satisfy constraints?

---

## 📝 Documentation & Collaboration

### 13. Knowledge Transfer

**Q13.1:** Code documentation:
- Docstrings for all functions?
- README with setup instructions?
- Implementation notes (linking to paper sections)?

**Q13.2:** Experiment documentation:
- How to record hyperparameters and results?
- Shared experiment log or notebook?

**Q13.3:** Collaboration workflow:
- Code review process?
- Branch naming and PR workflow?
- Communication channel (Slack, email, meetings)?

---

## 🚧 Risk & Challenges

### 14. Potential Issues

**Q14.1:** What are the biggest implementation risks?
- Phase B bridge reconstruction (most complex)?
- Memory constraints for large models?
- Insufficient preference data?
- Computational cost of DFT validation?

**Q14.2:** Contingency plans:
- If Phase B is too hard, can we get useful results with Phase A only?
- If DFT is too expensive, can we use proxy metrics?
- If full three-phase is too ambitious, what's the MVP?

**Q14.3:** Known difficulties from paper:
- Wrapped Gaussian on torus (special handling)?
- Constraint cancellation (correct masking)?
- Bridge level selection (which distribution $\rho(b)$)?
- Balancing learning rates across phases?

---

## 🎯 Priority & Decision-Making

### 15. What to Implement First?

**Q15.1:** Minimal viable product (MVP):
- What's the smallest implementation that's scientifically meaningful?
- Phase A with single motif?
- Phase A without SCIGEN (unconstrained only)?

**Q15.2:** Incremental milestones:
- Milestone 1: Hölder-DPO on simple synthetic data?
- Milestone 2: Phase A on MP-20 subset?
- Milestone 3: Full Phase A on MP-20?
- Milestone 4: Phase B with one motif?

**Q15.3:** What can we defer or simplify?
- Skip predictor-corrector (use tractable proxy only)?
- Skip wrapped Gaussian (use standard Gaussian approximation)?
- Skip Phase C active learning initially?
- Skip K-bridge evaluation (use single pseudo-bridge)?

**Q15.4:** What is absolutely essential?
- Hölder loss (γ = 2.0)?
- Improvement score calculation?
- Reference model snapshot?
- Multi-channel structure (or can we merge channels)?

---

## 🤝 Collaboration & Expertise

### 16. Division of Labor

**Q16.1:** Who implements what?
- Core DPO training loop?
- Hölder loss implementation?
- Bridge reconstruction (if Phase B)?
- Data preprocessing and loading?
- Evaluation metrics and plotting?

**Q16.2:** External help or existing code:
- Can we adapt DPO code from other repos (e.g., Diffusion-DPO)?
- Are there Hölder-DPO reference implementations?
- SCIGEN code already exists?

**Q16.3:** Your expertise and preferences:
- What parts are you most comfortable implementing?
- What parts would you like help with?
- What parts should I focus on?

---

## 📚 Reference Implementation & Prior Work

### 17. Code Availability

**Q17.1:** Does the paper have code released?
- Official implementation available?
- If yes, should we use it directly or adapt?
- If no, are there related implementations we can reference?

**Q17.2:** Related codebases to leverage:
- Original DPO (Rafailov et al.)?
- Diffusion-DPO (Wallace et al.)?
- Hölder-DPO (Fujisawa et al.)?
- DiffCSP++ (original paper)?

**Q17.3:** Existing tools:
- Pymatgen for structure handling?
- ASE for crystal operations?
- CHGNet for screening?
- Custom tools for DFT band structure?

---

## 🔮 Long-Term Vision

### 18. Beyond Initial Implementation

**Q18.1:** What's the research goal beyond implementation?
- Paper submission?
- New motif discovery?
- Method improvement?
- Application to specific materials problem?

**Q18.2:** Extensibility:
- Should the code be general (support other constraints)?
- Or optimized for flat band materials only?
- Support for future preference learning methods?

**Q18.3:** Open-sourcing:
- Plan to release code publicly?
- License considerations?
- Documentation level needed?

---

## ❓ Clarification & Understanding

### 19. Paper Interpretation

**Q19.1:** Are there any parts of the paper you found unclear or ambiguous?
- Bridge formulation details?
- Hyperparameter choices?
- Evaluation setup?

**Q19.2:** Have you contacted the authors?
- Did they provide any implementation details?
- Clarifications on specific equations?
- Access to code or data?

**Q19.3:** Assumptions we need to make:
- Where paper is underspecified, what should we assume?
- Conservative or aggressive design choices?

---

## 📋 Action Items & Next Steps

### 20. Immediate Next Steps

**Q20.1:** What should we focus on in the next week?
- Read and understand specific paper sections?
- Set up codebase and environment?
- Implement specific components?
- Run baseline experiments?

**Q20.2:** What do you need from me?
- Code review?
- Implementation of specific components?
- Literature review on specific topics?
- Debugging help?

**Q20.3:** When is our next sync?
- Regular meeting schedule?
- Ad-hoc when stuck?
- Milestone-based check-ins?

**Q20.4:** How to track progress?
- GitHub issues and PRs?
- Shared document?
- Weekly status updates?

---

## 📎 Additional Context

### 21. Related Materials

**Q21.1:** Should I prepare anything before the discussion?
- Specific paper sections to review in detail?
- Code examples to prototype?
- Literature on specific topics?

**Q21.2:** Are there specific experiments you want to replicate?
- Table X from the paper?
- Figure Y showing results?
- Ablation study Z?

**Q21.3:** Other relevant papers or resources:
- Prerequisite papers to understand?
- Related methods to compare against?
- Tutorial resources for unfamiliar topics?

---

## 🎓 Learning & Development

### 22. Skill Gaps & Training

**Q22.1:** Are there concepts I should study in depth?
- Diffusion models (DDPM, score-based)?
- Preference learning (DPO, RLHF)?
- Robust statistics (Hölder divergence)?
- Crystal structure representation?

**Q22.2:** Tools or frameworks to learn:
- Specific PyTorch features (e.g., DDP)?
- Experiment tracking tools (wandb, MLflow)?
- Materials science tools (pymatgen, ASE)?

---

## 📊 Summary & Prioritization

### Top 10 Most Critical Questions to Ask:

1. **Q1.1-1.2**: Timeline and scope (Phase A only? A+B? Full pipeline?)
2. **Q2.1-2.3**: Which codebase to build on? SCIGEN already implemented?
3. **Q3.2**: How to generate preference pairs for Phase A (DFT vs proxy)?
4. **Q4.1**: Should we implement Phase B at all initially?
5. **Q7.1**: Reference model strategy (freeze pretrained DiffCSP++?)
6. **Q10.1-10.2**: Framework (PyTorch/JAX?) and code organization
7. **Q11.1-11.2**: Compute resources and training time budget
8. **Q15.1-15.4**: MVP definition and what's essential vs deferrable
9. **Q16.1**: Division of labor (who does what?)
10. **Q20.1**: Immediate next steps for this week

---

## 📝 Notes Section

*Use this space during the meeting to record Masaki's answers and decisions:*

**Key Decisions:**
-
-
-

**Action Items:**
- [ ]
- [ ]
- [ ]

**Open Questions:**
-
-

**Follow-up Discussion Needed:**
-
-

---

**Document prepared by:** Claude Code
**Based on:** Complete reading notes for SCIGEN+ DPO paper (47 questions, 8,600+ lines)
**Related docs:**
- [reading_session_2026-03-16_DETAILED.md](reading_session_2026-03-16_DETAILED.md)
- [CONCEPT_MAP_MERMAID.md](CONCEPT_MAP_MERMAID.md)
- [DERIVATIONS_ANNOTATED.md](DERIVATIONS_ANNOTATED.md)
