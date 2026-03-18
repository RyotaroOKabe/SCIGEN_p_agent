---
description: Start an interactive reading session for an overleaf paper
---

# Task: Start Overleaf Reading Session

Start an interactive reading session for the specified overleaf paper. This will:
1. Sync the overleaf repo (if needed)
2. Load existing reading notes
3. Prepare to answer questions iteratively

## Arguments

- **paper_name** (required): Name of the overleaf paper
  - Options: `scigenp_overview`, `scigenp_overview_MA`, `Diffusion_DPO_archive`, `Diffusion_finetune_archive`, `SCIGEN_archive`

## Workflow

### Step 1: Sync Repository

First, run the sync protocol from `/overleaf-sync {paper_name}`:
- Check for local modifications
- Pull latest changes
- Log any updates

### Step 2: Load Context

Load existing reading notes:
```bash
ls /pscratch/sd/r/ryotaro/data/generative/overleaf_notes/{paper_name}/
```

Check what's already documented:
- `technical_terms.md` - Known terms
- `key_concepts.md` - Understood concepts
- `code_correspondence.md` - Code mappings
- `implementation_plan.md` - Plans
- `changes_log.md` - Recent changes

### Step 3: Report Status

Report to user:
```
📖 Reading session started for: {paper_name}

Repository status:
- ✓ Synced (no conflicts)
- Last update: [date]
- Location: /pscratch/sd/r/ryotaro/data/generative/overleaf/{paper_name}

Existing notes:
- Technical terms: [X terms documented]
- Key concepts: [X sections covered]
- Code mappings: [Y files mapped]
- Implementation plan: [Z tasks defined]

Reading notes: /pscratch/sd/r/ryotaro/data/generative/overleaf_notes/{paper_name}/

Ready for questions!
```

### Step 4: Interactive Q&A Mode

For each user question:

1. **Read relevant sections** from the tex files
2. **Check existing notes** for context
3. **Search online if needed** (for challenging concepts)
4. **Answer the question** with:
   - Direct quotes/references from paper
   - Explanations and clarifications
   - Links to related concepts
   - Online resources if used
5. **Update reading notes** with new insights

### Step 5: Update Notes

After answering, update the appropriate note file:

- New technical term → `technical_terms.md`
- New concept/insight → `key_concepts.md`
- Code implementation idea → `code_correspondence.md` or `implementation_plan.md`

## Example Usage

User: `/overleaf-read scigenp_overview`

```
📖 Reading session started for: scigenp_overview

Repository status:
- ✓ Synced (no conflicts)
- Last update: 2026-03-15
- Location: /pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview

Existing notes:
- Technical terms: 12 terms documented
- Key concepts: 3 sections covered
- Code mappings: 5 files mapped
- Implementation plan: 8 tasks defined

Reading notes: /pscratch/sd/r/ryotaro/data/generative/overleaf_notes/scigenp_overview/

Ready for questions! Ask me anything about the paper.
```

User: "What is the DPO loss function?"

Agent:
1. Reads `main.tex` Section 3.2
2. Checks `technical_terms.md` for "DPO"
3. Provides explanation with equation reference
4. Searches online for additional clarification if needed
5. Updates `technical_terms.md` with the definition

## Using Online Resources

When encountering challenging concepts:

1. First try to understand from the paper itself
2. If still unclear:
   - Use WebSearch for tutorials, explanations, related papers
   - Look for code examples or implementations
   - Find prerequisite concepts
3. Synthesize findings and relate back to the paper
4. Document everything in reading notes

## Cross-Referencing

While reading, cross-reference:
- Other overleaf papers in `/pscratch/sd/r/ryotaro/data/generative/overleaf/`
- Code in `/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent/`
- References in `/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent/references/`

## Implementation Discussion

When user asks "can we implement this?":

1. **Understand the concept** from paper
2. **Check current code** in SCIGEN_p_agent
3. **Assess resources**:
   - What data do we have?
   - What compute is available?
   - What's already implemented?
4. **Create realistic plan**:
   - What can be done now
   - What needs new resources
   - Estimated effort/timeline
5. **Document in `implementation_plan.md`**

## Ending the Session

When done:
```
📖 Reading session summary for {paper_name}

Notes updated:
- technical_terms.md: [+X new terms]
- key_concepts.md: [+Y new insights]
- code_correspondence.md: [+Z mappings]
- implementation_plan.md: [updated]

All notes saved to: /pscratch/sd/r/ryotaro/data/generative/overleaf_notes/{paper_name}/
```

## See Also

- [OVERLEAF.md](../OVERLEAF.md) - Full overleaf handling rules
- [overleaf-sync.md](./overleaf-sync.md) - Sync repository
- [overleaf-plan.md](./overleaf-plan.md) - Implementation planning
