# Command: Write Discussion Report

Write a markdown discussion report to `docs/discussion/` after meetings, feedback exchanges, or significant collaborative decisions.

## When to Write

- After a meeting with a collaborator (Masaki, Mingda, advisor, etc.)
- When receiving detailed feedback on a document or design
- When a key design decision is made collaboratively
- After a Slack/email exchange that resolves an open question

## File Naming

```
docs/discussion/YYYY-MM-DD_<slug>.md
```

- Date = today (`date +%Y-%m-%d`)
- Slug = short snake_case topic, e.g.:
  - `masaki_feedback_material_dpo`
  - `mingda_overview_review`
  - `dpo_loss_design_decision`

## Template

```markdown
# <Title> — YYYY-MM-DD

## Participants
Who was involved (names, roles).

## Context
What document/topic was discussed and why.

## Key Points Discussed
Numbered list of the main topics, with details on:
- What was raised
- What was decided or left open
- Specific changes suggested (with file/equation references)

## Changes Made
| File | Change | Suggested by |
|------|--------|-------------|

## Open Questions
Items that need follow-up.

## Action Items
- [ ] Task (owner, deadline if any)

## Next Steps
What happens next as a result of this discussion.
```

## How to Write (Claude instructions)

1. Review the feedback source (Slack message, meeting notes, diff between document versions).
2. If two versions of a document exist (original vs. modified), diff them and describe each change.
3. For each feedback point: state the issue, what changed, why it matters, and what to do next.
4. Save to `docs/discussion/YYYY-MM-DD_<slug>.md`.
5. Stage and commit: `git add docs/discussion/ && git commit`.
