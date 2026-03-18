---
description: Discuss implementation planning for concepts from overleaf papers
---

# Task: Implementation Planning from Overleaf

Plan implementation of concepts from overleaf papers, considering current resources and codebase.

## Arguments

- **concept** (required): The concept/method/algorithm to implement
- **paper_name** (optional): Specific paper, if not specified will search across all

## Workflow

### Step 1: Understand the Concept

1. **Locate in paper**:
   - Which section/equation/algorithm?
   - What's the theoretical foundation?
   - What are the key parameters?

2. **Read existing notes**:
   ```bash
   grep -r "{concept}" /pscratch/sd/r/ryotaro/data/generative/overleaf_notes/
   ```

3. **Extract requirements**:
   - Input data format
   - Computational requirements
   - Dependencies (libraries, models, etc.)

### Step 2: Assess Current Resources

Check what we have:

1. **Code**:
   ```bash
   cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
   grep -r "{concept}" scigen/ script/
   ```
   - Is anything already implemented?
   - What can be reused?

2. **Data**:
   ```bash
   ls data/
   ```
   - Do we have the required datasets?
   - What format are they in?

3. **Compute**:
   - Available resources: SLURM queue, GPU availability
   - Estimated compute needs

4. **Dependencies**:
   - Check conda env: `scigen_py312`
   - Missing packages?

### Step 3: Create Implementation Plan

Break down into phases:

#### Phase 1: Immediate (What we can do now)
- Tasks that use existing resources
- Preparatory work
- Code scaffolding

#### Phase 2: Short-term (What needs setup)
- Data preparation
- Dependency installation
- Initial experiments

#### Phase 3: Long-term (What needs new resources)
- Large-scale experiments
- New data collection
- Major refactoring

### Step 4: Document the Plan

Update `/pscratch/sd/r/ryotaro/data/generative/overleaf_notes/{paper_name}/implementation_plan.md`:

```markdown
## [Concept Name] Implementation

**Paper reference**: Section X.Y, Equation (Z)
**Complexity**: [Low/Medium/High]
**Priority**: [P0/P1/P2]

### Current Status
- [ ] Not started
- Existing code: [file.py:line] (if any)
- Available data: [dataset_name]

### Requirements
- Input: [format/structure]
- Output: [format/structure]
- Compute: [X GPU hours estimated]
- Dependencies: [libraries]

### Implementation Phases

#### Phase 1: Immediate
- [ ] Task 1: [description]
  - File: [where to implement]
  - Dependencies: [what's needed]
  - Effort: [hours/days]

#### Phase 2: Short-term
- [ ] Task 2: [description]

#### Phase 3: Long-term
- [ ] Task 3: [description]

### Design Decisions
- **Choice 1**: [decision] because [rationale]
- **Choice 2**: [decision] because [rationale]

### Risks & Challenges
- [Challenge 1]: Potential solutions...
- [Challenge 2]: Potential solutions...

### Success Criteria
- [ ] Unit tests pass
- [ ] Results match paper
- [ ] Documentation complete
```

### Step 5: Code Correspondence

Update `/pscratch/sd/r/ryotaro/data/generative/overleaf_notes/{paper_name}/code_correspondence.md`:

Map the concept to specific code locations:
```markdown
### [{Concept Name}]

**Paper**: Section X.Y, Algorithm Z
**Code**:
- Main implementation: `scigen/module/file.py:line`
- Helper functions: `script/utils.py:line`
- Config: `conf/model/model_name.yaml`

**Differences from paper**:
- [Difference 1]: [explanation]
- [Difference 2]: [explanation]

**Parameters**:
| Paper Symbol | Code Variable | Default Value |
|--------------|---------------|---------------|
| α            | alpha         | 0.1           |
```

### Step 6: Discuss with User

Present the plan:
1. Summary of the concept
2. What we can do now vs. later
3. Estimated effort and timeline
4. Open questions or decisions needed

Ask for feedback:
- Is this the right direction?
- What's the priority?
- Any constraints or preferences?

## Example Usage

User: `/overleaf-plan "DPO loss function"`

Agent:
```
📋 Implementation Planning: DPO Loss Function

Found in: overleaf/Diffusion_DPO_archive/main.tex, Section 3.2

### Concept Summary
Direct Preference Optimization loss that directly optimizes a diffusion model
using pairwise preferences without requiring a reward model.

Equation (7): L_DPO = -log σ(β log π_θ(x_w|c) - β log π_θ(x_l|c))

### Current Resources Assessment

✓ We have:
- Base diffusion model: scigen/pl_modules/diffusion_module.py
- Training loop: scigen/run.py
- SLURM setup for multi-GPU

✗ We need:
- Pairwise preference data (x_w > x_l)
- Preference collection mechanism
- Modified loss function

### Implementation Plan

#### Phase 1: Immediate (Can do now)
- [x] Understand the math (done in this session)
- [ ] Add DPO loss function to scigen/pl_modules/loss.py
- [ ] Create config for DPO training: conf/train/dpo.yaml

Effort: ~4 hours
Files: loss.py, conf/train/dpo.yaml

#### Phase 2: Short-term (Need data)
- [ ] Design preference data format
- [ ] Create preference dataset from existing generations
  - Use stability/energy as proxy for preference
- [ ] Implement data loader for preference pairs

Effort: ~2 days
Dependencies: Need to generate and rank structures

#### Phase 3: Long-term (Full experiment)
- [ ] Run full DPO training on alex_2d
- [ ] Compare with standard diffusion
- [ ] Ablation studies

Effort: ~1 week + compute time
Resources: 2 GPU × 24h

### Design Decisions

**Q1**: How to get preference data?
Options:
  A) Use existing GNN screening scores (stability/magnetism)
  B) Generate pairs and use human preference
  C) Use DFT results as ground truth

Recommendation: Start with (A) - we have screening pipeline

**Q2**: Where to implement the loss?
Options:
  A) Extend existing diffusion_module.py
  B) Create new dpo_module.py

Recommendation: (A) - minimize code duplication

### Blockers

⚠️ Need decision on preference data source before Phase 2

### Next Steps

What would you like to do?
1. Start with Phase 1 (loss function implementation)
2. Discuss preference data strategy first
3. Review the plan and adjust priorities
```

## See Also

- [OVERLEAF.md](../OVERLEAF.md) - Overleaf handling rules
- [overleaf-read.md](./overleaf-read.md) - Reading sessions
- [implementation_plan.md template](../../overleaf_notes/README.md) - Template format
