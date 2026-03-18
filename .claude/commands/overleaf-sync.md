---
description: Safely sync an overleaf git repository with conflict checking
---

# Task: Sync Overleaf Repository

Sync the specified overleaf repository following the safety protocol in `.claude/OVERLEAF.md`.

## Arguments

- **paper_name** (required): Name of the overleaf paper to sync
  - Options: `scigenp_overview`, `scigenp_overview_MA`, `Diffusion_DPO_archive`, `Diffusion_finetune_archive`, `SCIGEN_archive`

## Protocol

Follow this exact sequence:

1. **Navigate to repo**:
   ```bash
   cd /pscratch/sd/r/ryotaro/data/generative/overleaf/{paper_name}
   ```

2. **Check for local modifications**:
   ```bash
   git status
   ```
   - If dirty (modified files): STOP and report to user
   - If clean: continue

3. **Fetch and check for updates**:
   ```bash
   git fetch
   git status
   ```

4. **Pull if behind**:
   ```bash
   git pull
   ```
   - If conflicts: STOP and report to user with `git status` and `git diff`
   - If clean merge: continue

5. **Report changes**:
   ```bash
   git log --oneline -5
   git diff HEAD~1..HEAD --stat
   ```

6. **Create change log** (if there were updates):
   - Create/update: `/pscratch/sd/r/ryotaro/data/generative/overleaf_notes/{paper_name}/changes_log.md`
   - Use the format from `.claude/OVERLEAF.md`
   - Include: date, commits, files changed, summary

## Example Usage

User: `/overleaf-sync scigenp_overview`

Expected output:
```
✓ Checked overleaf/scigenp_overview
✓ No local modifications
✓ Pulled latest changes
✓ 2 new commits from collaborator:
  - abc1234: Updated introduction
  - def5678: Added new figures

Files changed:
  main.tex | 15 +++++++++------
  figures/new_fig.png | Bin 0 -> 45823 bytes

Change log created at: overleaf_notes/scigenp_overview/changes_log.md
```

## Error Handling

If any step fails:
1. STOP immediately
2. Report the specific issue to user
3. Show relevant git output
4. Ask for user instructions
5. DO NOT attempt automatic resolution

## See Also

- [OVERLEAF.md](../OVERLEAF.md) - Full overleaf handling rules
- [overleaf-read.md](./overleaf-read.md) - Start reading session
