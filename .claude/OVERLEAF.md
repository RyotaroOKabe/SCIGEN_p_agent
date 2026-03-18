# Overleaf Handling Rules

## Critical Rules: DO NOT EDIT Unless Explicitly Asked

**⚠️ IMPORTANT**: Unless the user explicitly asks to edit Overleaf documents, **NEVER** modify any files in the overleaf repositories. These are collaborative documents that may have changes from other contributors.

### Overleaf Repository Locations

```
/pscratch/sd/r/ryotaro/data/generative/overleaf/
├── scigenp_overview/          # Active - Main SCIGEN+ paper (git repo)
├── scigenp_overview_MA/       # Active - SCIGEN+ MA version (git repo)
├── Diffusion_DPO_archive/     # Archive - DPO paper
├── Diffusion_finetune_archive/# Archive - Finetuning paper
└── SCIGEN_archive/            # Archive - Original SCIGEN paper
```

### Reading Notes Location

All reading notes are stored separately at:
```
/pscratch/sd/r/ryotaro/data/generative/overleaf_notes/
```

This keeps notes separate from the collaborative overleaf git repositories to avoid conflicts.

---

## Git Sync Protocol

### Before Reading/Analyzing Overleaf

**ALWAYS** check for updates first:

1. **Check current status**:
   ```bash
   cd /pscratch/sd/r/ryotaro/data/generative/overleaf/<paper_name>
   git status
   ```

2. **If there are local modifications**:
   - STOP and inform the user
   - Ask: "You have local changes in the overleaf repo. What would you like to do?"
   - Options:
     - Commit local changes first
     - Stash local changes
     - Abort sync

3. **Pull latest changes**:
   ```bash
   git pull
   ```

4. **If there are conflicts**:
   - STOP immediately
   - Inform the user about the conflict
   - Show conflict details: `git status` and `git diff`
   - **DO NOT** attempt to resolve conflicts automatically
   - Wait for explicit user instructions

5. **Report new changes**:
   - If the pull brings new commits, summarize what changed:
     ```bash
     git log --oneline -5
     git diff HEAD~1 HEAD --stat
     ```
   - Log the changes to reading notes (see "Change Logging" section below)

### Sync Workflow Example

```bash
# 1. Navigate to overleaf repo
cd /pscratch/sd/r/ryotaro/data/generative/overleaf/scigenp_overview

# 2. Check status
git status
# If dirty: STOP and ask user

# 3. Pull latest
git pull origin main  # or master, depending on branch

# 4. Summarize changes
git log --oneline -5
git diff HEAD~1..HEAD --stat

# 5. Log to reading notes
# See "Change Logging" section
```

---

## Change Logging

When new changes are pulled from overleaf, create a dated log entry in the reading notes.

### Log Location

```
/pscratch/sd/r/ryotaro/data/generative/overleaf_notes/<paper_name>/changes_log.md
```

### Log Format

```markdown
## YYYY-MM-DD: [Brief Description]

### Source
- **Commits**: [commit hash] - [commit message]
- **Author**: [author name/collaborator]
- **Files changed**: [list of files]

### Summary of Changes
[Brief summary of what was modified]

### User Comments
[Space for user to paste comments/messages from collaborators]

### Impact on Implementation
- [ ] Requires code changes
- [ ] Requires new experiments
- [ ] Documentation only
- [ ] No action needed

### Related Tasks
- Link to implementation_plan.md tasks if applicable
```

### Example

```markdown
## 2026-03-16: Added new DPO formulation section

### Source
- **Commits**: abc1234 - "Add theoretical foundation for DPO loss"
- **Author**: Collaborator Name
- **Files changed**: main.tex, preamble.tex, figures/dpo_formulation.png

### Summary of Changes
Added Section 3.2 explaining the theoretical derivation of the DPO loss function
with new notation and a figure comparing it to standard RLHF.

### User Comments
[User can paste collaborator's message here]

### Impact on Implementation
- [x] Requires code changes - need to update loss function implementation
- [ ] Requires new experiments
- [x] Documentation only - update code comments

### Related Tasks
- See implementation_plan.md: Update DPO loss implementation (Task 2.3)
```

---

## Reading & Analysis Workflow

### When User Asks Questions About Overleaf Papers

1. **Sync first** (follow Git Sync Protocol above)

2. **Read the relevant sections**:
   ```bash
   # Read the main tex file
   cat /pscratch/sd/r/ryotaro/data/generative/overleaf/<paper>/main.tex

   # Or specific sections
   cat /pscratch/sd/r/ryotaro/data/generative/overleaf/<paper>/<section>.tex
   ```

3. **Check reading notes** for existing context:
   ```bash
   # Check if we have notes already
   ls /pscratch/sd/r/ryotaro/data/generative/overleaf_notes/<paper>/
   ```

4. **Answer the question**:
   - Use content from the overleaf tex files
   - Reference existing notes if available
   - If needed, search online for clarification (WebSearch tool)
   - Update reading notes with new insights

5. **Update reading notes**:
   - Add new technical terms to `technical_terms.md`
   - Add insights to `key_concepts.md`
   - Update implementation plans if needed

### Iterative Reading Sessions

When user is reading and asking multiple questions:

1. Keep track of which sections have been discussed
2. Build up notes incrementally in the reading notes
3. Cross-reference between different papers when relevant
4. Suggest related code in SCIGEN_p_agent if applicable

### Using Online Resources

When user asks challenging questions:

1. First try to answer from the paper
2. If unclear or complex:
   - Use WebSearch to find clarifying resources
   - Look for related papers or tutorials
   - Find code examples if applicable
3. Summarize findings and link back to the paper's context
4. Add clarifications to `technical_terms.md` or `key_concepts.md`

---

## Implementation Planning from Overleaf

When discussing implementation based on overleaf papers:

1. **Reference the paper section**:
   - Note which section/equation/algorithm
   - Extract key parameters and requirements

2. **Check current resources**:
   - What's already implemented in SCIGEN_p_agent?
   - What datasets do we have?
   - What computational resources are available?

3. **Create realistic plan**:
   - Break down into phases in `implementation_plan.md`
   - Mark what can be done now vs. what needs new resources
   - Link to specific code files that need modification

4. **Document correspondence**:
   - Update `code_correspondence.md` with paper-to-code mappings
   - Note any deviations from the paper

### Example Discussion Flow

```
User: "Can we implement the DPO loss from the paper?"

Agent:
1. Syncs overleaf/Diffusion_DPO_archive
2. Reads Section 3 on DPO loss formulation
3. Checks current SCIGEN_p_agent code for loss functions
4. Analyzes: Do we have the required data structure? (pairwise preferences)
5. Proposes:
   - What we can do now: Add loss function code
   - What we need: Preference dataset generation
   - Timeline estimate
6. Updates implementation_plan.md with the proposal
```

---

## Slash Commands

Use these commands for common overleaf tasks:

- `/overleaf-sync [paper_name]` - Safely sync an overleaf repo
- `/overleaf-read [paper_name]` - Open a reading session
- `/overleaf-plan [concept]` - Discuss implementation planning

---

## Summary of DO NOTs

❌ **DO NOT** edit overleaf files unless explicitly asked
❌ **DO NOT** auto-resolve git conflicts
❌ **DO NOT** pull without checking for local modifications first
❌ **DO NOT** commit to overleaf repos without explicit permission
❌ **DO NOT** skip change logging when new commits are pulled
❌ **DO NOT** store reading notes inside overleaf directories

## Summary of DOs

✅ **DO** always sync before reading
✅ **DO** inform user of local modifications before pulling
✅ **DO** stop and ask when conflicts occur
✅ **DO** log all changes to reading notes
✅ **DO** use reading notes for all your notes and plans
✅ **DO** cross-reference between papers and code
✅ **DO** consider current resources when planning implementation
