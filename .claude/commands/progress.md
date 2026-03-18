# Command: Write Progress Report

Write a markdown progress report to `docs/progress/` whenever a critical update occurs.

## When to Write

- A SLURM job completes with noteworthy results (generation stats, screening pass rates)
- A bug is found and fixed (root cause + fix must be documented)
- A new script, SLURM job, or pipeline stage is added
- An architectural or design decision is made
- A session milestone is reached (end of a working session with meaningful changes)

## File Naming

```
docs/progress/YYYY-MM-DD_<slug>.md
```

- Date = today in PST (`date +%Y-%m-%d`)
- Slug = short snake_case topic, e.g.:
  - `session1_pipeline_setup_screening`
  - `screening_smact_he_bugfix`
  - `gen_elem_results`

## Template

```markdown
# <Title> — YYYY-MM-DD

## Summary
One paragraph: what happened, what changed, what the key outcome is.

---

## Background
(Optional) Context needed to understand the update.

---

## Results
Tables, numbers, pass rates, job IDs, file paths.

---

## Code Changes
| File | Change |
|------|--------|
| `script/mat_utils.py` | Added None guard in smact_validity |

---

## Bugs Fixed
### Bug: <short name>
- **Root cause**: ...
- **Fix**: ...
- **Commit**: `abc1234`

---

## Environment Notes
Missing packages, env fixes, PYTHONNOUSERSITE issues.

---

## Next Steps
Ordered list of what to do next.

---

## Commits
```
abc1234  <message>
```
```

## How to Write (Claude instructions)

1. Gather all facts from the current session: SLURM job IDs, exit codes, pass-rate numbers, file paths, error messages.
2. Fill the template — omit sections that are empty.
3. Save to `docs/progress/YYYY-MM-DD_<slug>.md`.
4. Stage and commit alongside any code changes: `git add docs/progress/ && git commit`.
